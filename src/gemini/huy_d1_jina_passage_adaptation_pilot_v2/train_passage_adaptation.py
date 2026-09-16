"""Train Pure-LoRA passage adaptation with pairwise ranking and teacher preservation."""

from __future__ import annotations

import hashlib
import json
import math
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW

from .common import (
    ADAPTER_DIR,
    RES_DIR,
    ROOT,
    build_pure_lora_jina_model,
    get_git_status,
    seed_everything,
    sha256_file,
)

TEACHER_PAIRS_FILE = RES_DIR / "TEACHER_TRAINING_PAIRS.jsonl"
TEACHER_SEAL_FILE = RES_DIR / "TEACHER_SCORE_CACHE_SEAL.json"


def get_classifier_state_hash(model) -> str:
    """Compute SHA256 of the classifier head parameters."""
    classifier_state = model.model.classifier.state_dict()
    h = hashlib.sha256()
    for k in sorted(classifier_state.keys()):
        h.update(f"{k}:".encode("utf-8"))
        h.update(classifier_state[k].detach().cpu().numpy().tobytes())
    return h.hexdigest()


def train_adaptation(
    epochs: int = 1,
    learning_rate: float = 1e-5,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.10,
    max_grad_norm: float = 1.0,
    micro_batch_size: int = 4,
    effective_batch_size: int = 16,
    seed: int = 2026,
) -> Dict[str, Any]:
    print("=== STAGE: PURE-LORA PASSAGE ADAPTATION TRAINING ===", flush=True)
    RES_DIR.mkdir(parents=True, exist_ok=True)
    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    seed_everything(seed)
    git_info = get_git_status()

    # 1. Load training pairs
    if not TEACHER_PAIRS_FILE.exists():
        raise FileNotFoundError(f"Training pairs file not found: {TEACHER_PAIRS_FILE}")

    pairs_data: List[Dict[str, Any]] = []
    with open(TEACHER_PAIRS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                pairs_data.append(json.loads(line))

    print(f"[TRAIN] Loaded {len(pairs_data)} training pairs.", flush=True)

    # Shuffle training pairs deterministically with seed 2026
    rng = random.Random(seed)
    rng.shuffle(pairs_data)

    # 2. Build Pure-LoRA Model
    print("[TRAIN] Instantiating Pure-LoRA model...", flush=True)
    model, tok = build_pure_lora_jina_model(r=8, lora_alpha=16, lora_dropout=0.05)
    model.train().to("cuda")

    # Snapshot initial classifier state
    initial_classifier_hash = get_classifier_state_hash(model)
    print(f"[TRAIN] Initial classifier head SHA256: {initial_classifier_hash}", flush=True)

    # Snapshot initial LoRA parameter states
    initial_lora_states = {
        n: p.detach().clone().cpu()
        for n, p in model.named_parameters()
        if p.requires_grad
    }

    # 3. Setup Optimizer and Cosine Scheduler
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=learning_rate, weight_decay=weight_decay)

    accum_steps = max(1, effective_batch_size // micro_batch_size)
    total_micro_batches = math.ceil(len(pairs_data) / micro_batch_size)
    total_optimizer_steps = math.ceil(total_micro_batches / accum_steps)
    warmup_steps = int(total_optimizer_steps * warmup_ratio)

    print(
        f"[TRAIN] Config: epochs={epochs}, lr={learning_rate}, warmup_steps={warmup_steps}, "
        f"total_steps={total_optimizer_steps}, micro_batch={micro_batch_size}, accum_steps={accum_steps}",
        flush=True,
    )

    def get_lr(step: int) -> float:
        if step < warmup_steps:
            return learning_rate * (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_optimizer_steps - warmup_steps)
        return learning_rate * 0.5 * (1.0 + math.cos(math.pi * progress))

    # 4. Training Loop
    t0_train = time.time()
    optimizer_step = 0
    raw_grad_norms: List[float] = []
    clipped_grad_norms: List[float] = []
    ranking_losses: List[float] = []
    teacher_losses: List[float] = []
    total_losses: List[float] = []
    nan_inf_count = 0

    optimizer.zero_grad()
    current_accum = 0

    for mb_idx in range(total_micro_batches):
        mb_pairs = pairs_data[mb_idx * micro_batch_size : (mb_idx + 1) * micro_batch_size]
        mb_loss = torch.tensor(0.0, device="cuda", requires_grad=True)

        for pair in mb_pairs:
            q_text = pair["query_text"]
            pos_secs = pair["pos_sections"] or [q_text]
            neg_secs = pair["neg_sections"] or [q_text]
            t_pos = float(pair["teacher_pos_logit"])
            t_neg = float(pair["teacher_neg_logit"])
            w = float(pair["pair_weight"])

            # Forward pass for positive sections
            pos_inputs = tok(
                [(q_text, s) for s in pos_secs],
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512,
            ).to("cuda")
            pos_logits = model(**pos_inputs, return_dict=True).logits.view(-1).float()
            s_pos = torch.max(pos_logits)

            # Forward pass for negative sections
            neg_inputs = tok(
                [(q_text, s) for s in neg_secs],
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512,
            ).to("cuda")
            neg_logits = model(**neg_inputs, return_dict=True).logits.view(-1).float()
            s_neg = torch.max(neg_logits)

            # Losses
            l_rank = F.softplus(-(s_pos - s_neg))
            l_teacher = 0.5 * ((s_pos - t_pos) ** 2 + (s_neg - t_neg) ** 2)
            pair_loss = w * (l_rank + 0.5 * l_teacher)

            ranking_losses.append(float(l_rank.detach().cpu().item()))
            teacher_losses.append(float(l_teacher.detach().cpu().item()))
            total_losses.append(float(pair_loss.detach().cpu().item()))

            mb_loss = mb_loss + pair_loss

        # Scale by accum_steps and backprop
        mb_loss_scaled = mb_loss / accum_steps
        mb_loss_scaled.backward()
        current_accum += 1

        if current_accum == accum_steps or (mb_idx + 1) == total_micro_batches:
            # Set LR for this step
            cur_lr = get_lr(optimizer_step)
            for param_group in optimizer.param_groups:
                param_group["lr"] = cur_lr

            # Compute raw gradient norm
            total_norm = 0.0
            for p in trainable_params:
                if p.grad is not None:
                    param_norm = p.grad.detach().data.norm(2).item()
                    if math.isnan(param_norm) or math.isinf(param_norm):
                        nan_inf_count += 1
                    total_norm += param_norm ** 2
            raw_norm = math.sqrt(total_norm)
            raw_grad_norms.append(raw_norm)

            # Clip gradient norm
            clipped_norm = float(torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=max_grad_norm).item())
            clipped_grad_norms.append(clipped_norm)

            # Step optimizer
            optimizer.step()
            optimizer.zero_grad()
            optimizer_step += 1
            current_accum = 0

            if optimizer_step % 50 == 0 or optimizer_step == total_optimizer_steps:
                print(
                    f"[TRAIN] Step {optimizer_step}/{total_optimizer_steps} | LR: {cur_lr:.2e} | "
                    f"Total Loss: {np.mean(total_losses[-50:]):.4f} | "
                    f"Rank Loss: {np.mean(ranking_losses[-50:]):.4f} | "
                    f"Teacher Loss: {np.mean(teacher_losses[-50:]):.4f} | "
                    f"Grad Norm: {raw_norm:.3f} -> {clipped_norm:.3f}",
                    flush=True,
                )

    train_elapsed = time.time() - t0_train
    print(f"[TRAIN] Training complete in {train_elapsed:.1f}s ({optimizer_step} optimizer steps).", flush=True)

    # 5. Post-training Verifications
    print("[TRAIN] Running post-training parameter integrity assertions...", flush=True)

    # Check classifier head bit-exact immutability
    post_classifier_hash = get_classifier_state_hash(model)
    classifier_delta_is_zero = (initial_classifier_hash == post_classifier_hash)
    print(f"[TRAIN] Post-training classifier head SHA256: {post_classifier_hash}", flush=True)
    if not classifier_delta_is_zero:
        raise RuntimeError(
            f"BLOCKED_CLASSIFIER_DRIFT: Classifier head changed during training!\n"
            f"Initial: {initial_classifier_hash}\n"
            f"Post:    {post_classifier_hash}"
        )

    # Check LoRA parameter update proof
    lora_drifts = []
    for n, p in model.named_parameters():
        if p.requires_grad:
            p_post = p.detach().cpu()
            p_pre = initial_lora_states[n]
            drift = float(torch.norm(p_post - p_pre).item())
            lora_drifts.append({
                "parameter_name": n,
                "l2_drift": drift,
                "max_abs_change": float(torch.max(torch.abs(p_post - p_pre)).item()),
            })

    total_lora_l2_drift = sum(d["l2_drift"] for d in lora_drifts)
    if total_lora_l2_drift == 0.0:
        raise RuntimeError("BLOCKED_ZERO_NEURAL_UPDATE: No LoRA parameter changed during training!")

    print(f"[TRAIN] Total LoRA L2 drift: {total_lora_l2_drift:.6f}", flush=True)

    # 6. Save Adapter
    print(f"[TRAIN] Saving adapted LoRA weights to {ADAPTER_DIR}...", flush=True)
    model.save_pretrained(ADAPTER_DIR)

    # Save artifacts
    stability_data = {
        "schema_version": "dsc2026.gemini.huy_d1_jina_passage_adaptation_pilot_v2.training_stability.v1",
        "experiment_id": "HUY_D1_JINA_PASSAGE_ADAPTATION_PILOT_V2",
        "status": "PASS",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_info["head_commit"],
        "training_time_seconds": train_elapsed,
        "total_optimizer_steps": optimizer_step,
        "effective_batch_size": effective_batch_size,
        "micro_batch_size": micro_batch_size,
        "warmup_steps": warmup_steps,
        "final_learning_rate": cur_lr,
        "gradient_statistics": {
            "mean_raw_grad_norm": float(np.mean(raw_grad_norms)),
            "max_raw_grad_norm": float(np.max(raw_grad_norms)),
            "mean_clipped_grad_norm": float(np.mean(clipped_grad_norms)),
            "max_clipped_grad_norm": float(np.max(clipped_grad_norms)),
            "nan_inf_grad_count": nan_inf_count,
        },
        "loss_statistics": {
            "mean_total_loss": float(np.mean(total_losses)),
            "mean_ranking_loss": float(np.mean(ranking_losses)),
            "mean_teacher_loss": float(np.mean(teacher_losses)),
        },
        "total_lora_l2_drift": total_lora_l2_drift,
    }
    stability_path = RES_DIR / "TRAINING_STABILITY.json"
    stability_path.write_text(json.dumps(stability_data, indent=2, ensure_ascii=False), encoding="utf-8")

    update_proof_data = {
        "schema_version": "dsc2026.gemini.huy_d1_jina_passage_adaptation_pilot_v2.neural_update_proof.v1",
        "experiment_id": "HUY_D1_JINA_PASSAGE_ADAPTATION_PILOT_V2",
        "status": "PASS",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_info["head_commit"],
        "classifier_integrity": {
            "initial_sha256": initial_classifier_hash,
            "post_training_sha256": post_classifier_hash,
            "delta_is_strictly_zero": classifier_delta_is_zero,
        },
        "lora_update_summary": {
            "total_trainable_tensors": len(lora_drifts),
            "total_l2_drift": total_lora_l2_drift,
            "at_least_one_lora_changed": total_lora_l2_drift > 0.0,
        },
        "per_parameter_drifts": lora_drifts,
    }
    update_proof_path = RES_DIR / "NEURAL_UPDATE_PROOF.json"
    update_proof_path.write_text(json.dumps(update_proof_data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[TRAIN] Saved stability and update proof artifacts.", flush=True)

    return stability_data


if __name__ == "__main__":
    train_adaptation()
