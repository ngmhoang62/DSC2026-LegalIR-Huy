"""Train LoRA continuation on non-CAL labeled population with real-time VRAM trace and audits."""

from __future__ import annotations

import gc
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForSequenceClassification

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

ROOT = Path("D:/Study/DSC2026/sota")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src/huy_fasttrack") not in sys.path:
    sys.path.insert(0, str(ROOT / "src/huy_fasttrack"))

import run_huy_5fold_fasttrack as core
from src.gemini.huy_d1_lal_case_memory_v1.audit_data_isolation import load_duplicate_graph
from tune_corpus_cap32_fusion import build_training_cap
import src.gemini.huy_d1_jina_ft_continuation_v1.common as common
from src.gemini.huy_d1_jina_ft_continuation_v1.audit_train_data import sample_query_negatives

RES_DIR = ROOT / "results/gemini/huy_d1_jina_ft_continuation_v1"
ADAPTER_DIR = RES_DIR / "jina_ft_continued_adapter"
VRAM_TRACE_PATH = RES_DIR / "VRAM_TRACE.jsonl"
UPDATE_PROOF_PATH = RES_DIR / "NEURAL_UPDATE_PROOF.json"
ROUNDTRIP_PATH = RES_DIR / "JINA_ADAPTER_ROUNDTRIP.json"


def train_jina_continuation() -> dict:
    common.seed_everything(2026)
    RES_DIR.mkdir(parents=True, exist_ok=True)

    # Read calibrated microbatch
    calib_file = RES_DIR / "VRAM_CALIBRATION.json"
    if calib_file.exists():
        with open(calib_file, "r") as f:
            calib = json.load(f)
        pair_mb = calib.get("chosen_pair_microbatch", 8)
    else:
        pair_mb = 8

    print(f"Loading data with pair_microbatch={pair_mb}...", flush=True)
    folds, pools, questions, v2_golds, _, _, _, _ = core.load_inputs()
    contexts = common.load_contexts()
    h_local = common.load_h_local()
    v2_doc_ids = set(contexts.keys())

    # Determine training query list strictly matching JINA_TRAIN_DATA_AUDIT.json
    _, _, all_ids, _, _, _ = build_training_cap(
        ROOT, 32, "results/corpus_index/holdout_extended_scores_cap32.pkl", depth=20
    )
    cal_qids = set(str(q) for q in all_ids)
    _, _, _, dup_map = load_duplicate_graph()
    cal_dups = {str(dup) for q in cal_qids for dup in dup_map.get(q, set())}
    forbidden = cal_qids | cal_dups

    eligible_qids = [
        str(q) for q in sorted(list(pools.keys()), key=int) if str(q) not in forbidden
    ]
    train_data = []
    for q in eligible_qids:
        pool = [str(d) for d in pools[q]]
        gold = {str(d) for d in v2_golds.get(q, set())}
        positives = [d for d in pool if d in gold]
        if not positives:
            continue
        negs = sample_query_negatives(q, pool, gold, h_local, v2_doc_ids, seed=2026)
        train_data.append((q, positives, negs))

    print(f"Total training queries: {len(train_data)}", flush=True)

    # Build LoRA model
    print("Building LoRA Jina model...", flush=True)
    model, tok = common.build_lora_jina_model(r=16, lora_alpha=32, lora_dropout=0.05)
    model.to("cuda")

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    trainable_names = [n for n, p in model.named_parameters() if p.requires_grad]
    trainable_count = sum(p.numel() for p in trainable_params)
    total_count = sum(p.numel() for p in model.parameters())

    # Snapshot initial adapter weights
    initial_adapter_weights = {
        n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad
    }

    # Setup audit pairs (first 64 pairs from first queries)
    print("Preparing 64 audit pairs for neural update proof...", flush=True)
    audit_pairs = []
    for q, pos, neg in train_data:
        qtext = questions[q]
        for d in pos[:1] + neg[:2]:
            p_text = common.top_passages(qtext, contexts[d], count=1)[0]
            audit_pairs.append((qtext, p_text))
            if len(audit_pairs) >= 64:
                break
        if len(audit_pairs) >= 64:
            break
    audit_pairs = audit_pairs[:64]

    # Pre-training audit logits
    model.eval()
    pre_audit_logits = []
    with torch.no_grad():
        for i in range(0, len(audit_pairs), 16):
            chunk = audit_pairs[i : i + 16]
            batch = tok(
                [c[0] for c in chunk],
                [c[1] for c in chunk],
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True,
            ).to("cuda")
            with torch.amp.autocast("cuda", dtype=torch.float16):
                logits = model(**batch).logits.view(-1).cpu().float().tolist()
                pre_audit_logits.extend(logits)

    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(trainable_params, lr=5e-5, weight_decay=0.01)
    accum_steps = 16
    total_opt_steps = len(train_data) // accum_steps

    # Reset VRAM trace
    if VRAM_TRACE_PATH.exists():
        VRAM_TRACE_PATH.unlink()

    print(
        f"Starting training: {len(train_data)} queries, accum={accum_steps}, "
        f"total_steps={total_opt_steps}...",
        flush=True,
    )

    model.train()
    optimizer.zero_grad(set_to_none=True)
    step_count = 0
    t_train_start = time.perf_counter()
    last_step_time = time.perf_counter()
    grad_norms = []
    non_zero_grad_counts = 0
    total_grad_evals = 0

    for idx, (q, positives, negs) in enumerate(train_data):
        qtext = questions[q]
        all_docs = positives + negs
        passages = [common.top_passages(qtext, contexts[d], count=1)[0] for d in all_docs]
        pairs = [(qtext, p) for p in passages]

        # Forward pass in pair microbatches
        logits_list = []
        for mb_start in range(0, len(pairs), pair_mb):
            chunk = pairs[mb_start : mb_start + pair_mb]
            batch = tok(
                [c[0] for c in chunk],
                [c[1] for c in chunk],
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True,
            ).to("cuda")
            with torch.amp.autocast("cuda", dtype=torch.float16):
                out = model(**batch)
                logits_list.append(out.logits.view(-1))

        all_logits = torch.cat(logits_list)
        pos_logits = all_logits[: len(positives)]
        neg_logits = all_logits[len(positives) :]

        # Pairwise RankNet loss: softplus(-(s_pos - s_neg))
        pair_diff = -(pos_logits.unsqueeze(1) - neg_logits.unsqueeze(0))
        pair_loss = torch.nn.functional.softplus(pair_diff).mean()

        scaled_loss = pair_loss / accum_steps
        scaled_loss.backward()

        if (idx + 1) % accum_steps == 0 or (idx + 1) == len(train_data):
            # Compute gradient norm
            total_norm = 0.0
            for p in trainable_params:
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2).item()
                    total_norm += param_norm**2
                    total_grad_evals += 1
                    if param_norm > 0:
                        non_zero_grad_counts += 1
            total_norm = total_norm**0.5
            grad_norms.append(total_norm)

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step_count += 1
            torch.cuda.synchronize()

            now = time.perf_counter()
            step_dt = now - last_step_time
            last_step_time = now

            # VRAM watchdog metrics
            curr_alloc = torch.cuda.memory_allocated() / (1024**2)
            curr_res = torch.cuda.memory_reserved() / (1024**2)
            peak_alloc = torch.cuda.max_memory_allocated() / (1024**2)
            peak_res = torch.cuda.max_memory_reserved() / (1024**2)
            free_mem, _ = torch.cuda.mem_get_info()
            free_mib = free_mem / (1024**2)

            remaining_steps = total_opt_steps - step_count
            eta_sec = remaining_steps * step_dt

            trace_entry = {
                "optimizer_step": step_count,
                "processed_query_groups": idx + 1,
                "total_query_groups": len(train_data),
                "current_allocated_mib": curr_alloc,
                "current_reserved_mib": curr_res,
                "peak_allocated_mib": peak_alloc,
                "peak_reserved_mib": peak_res,
                "free_mib": free_mib,
                "step_runtime_sec": step_dt,
                "running_eta_sec": eta_sec,
                "grad_norm": total_norm,
            }
            with open(VRAM_TRACE_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(trace_entry) + "\n")

            if step_count % 25 == 0 or step_count == total_opt_steps:
                print(
                    f"Step {step_count}/{total_opt_steps} ({idx+1}/{len(train_data)} queries): "
                    f"res={peak_res:.1f}MiB, free={free_mib:.1f}MiB, "
                    f"dt={step_dt:.2f}s, eta={eta_sec/60:.1f}m, grad_norm={total_norm:.4f}",
                    flush=True,
                )

            # Hard stop watchdog
            if peak_res > 5300.0 or free_mib < 300.0:
                print(f"HARD STOP TRIGGERED: peak_res={peak_res:.1f}MiB", flush=True)
                model.save_pretrained(ADAPTER_DIR)
                return {"status": "LOCAL_VRAM_CAP_EXCEEDED", "peak_res": peak_res}

    train_total_time = time.perf_counter() - t_train_start
    print(f"Training completed in {train_total_time/60:.2f} minutes.", flush=True)

    # Post-training audit logits
    model.eval()
    post_audit_logits = []
    with torch.no_grad():
        for i in range(0, len(audit_pairs), 16):
            chunk = audit_pairs[i : i + 16]
            batch = tok(
                [c[0] for c in chunk],
                [c[1] for c in chunk],
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True,
            ).to("cuda")
            with torch.amp.autocast("cuda", dtype=torch.float16):
                logits = model(**batch).logits.view(-1).cpu().float().tolist()
                post_audit_logits.extend(logits)

    # Compute adapter L2 delta
    adapter_l2_delta = 0.0
    for n, p in model.named_parameters():
        if p.requires_grad:
            diff = (p.detach().cpu() - initial_adapter_weights[n]).float()
            adapter_l2_delta += diff.norm(2).item() ** 2
    adapter_l2_delta = adapter_l2_delta**0.5

    # Changed audit logits
    changed_logits = sum(
        1 for pre, post in zip(pre_audit_logits, post_audit_logits) if abs(pre - post) > 1e-4
    )

    # Pairwise ordering changes in audit pairs
    pre_ranks = np.argsort(np.argsort(-np.array(pre_audit_logits)))
    post_ranks = np.argsort(np.argsort(-np.array(post_audit_logits)))
    changed_orderings = int(np.sum(pre_ranks != post_ranks))

    neural_proof = {
        "experiment_id": "HUY_D1_JINA_FT_CONTINUATION_V1",
        "trainable_parameter_names": trainable_names,
        "trainable_parameter_count": trainable_count,
        "total_parameter_count": total_count,
        "trainable_ratio": trainable_count / total_count,
        "optimizer_steps": step_count,
        "total_grad_evals": total_grad_evals,
        "non_zero_grad_fraction": non_zero_grad_counts / max(1, total_grad_evals),
        "grad_norm_min": float(min(grad_norms)) if grad_norms else 0.0,
        "grad_norm_max": float(max(grad_norms)) if grad_norms else 0.0,
        "grad_norm_mean": float(np.mean(grad_norms)) if grad_norms else 0.0,
        "adapter_l2_delta": float(adapter_l2_delta),
        "audit_pair_count": len(audit_pairs),
        "audit_changed_logits_count": changed_logits,
        "audit_changed_orderings_count": changed_orderings,
        "proof_passed": (step_count > 0) and (adapter_l2_delta > 0) and (changed_logits > 0),
        "status": "PASS" if ((step_count > 0) and (adapter_l2_delta > 0) and (changed_logits > 0)) else "FAIL",
    }
    with open(UPDATE_PROOF_PATH, "w", encoding="utf-8") as f:
        json.dump(neural_proof, f, indent=2)
    print(f"Wrote {UPDATE_PROOF_PATH}", flush=True)

    # Save LoRA adapter
    print(f"Saving LoRA adapter to {ADAPTER_DIR}...", flush=True)
    if ADAPTER_DIR.exists():
        shutil.rmtree(ADAPTER_DIR)
    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ADAPTER_DIR)

    # Section 13: Checkpoint round-trip test
    print("Executing checkpoint round-trip test on fresh base model...", flush=True)
    del model, optimizer
    gc.collect()
    torch.cuda.empty_cache()

    fresh_base, _ = common.load_jina_base_with_shipped_weights()
    reloaded_lora = PeftModel.from_pretrained(fresh_base, ADAPTER_DIR)
    common.patch_tuple_returning_lora(reloaded_lora)
    reloaded_lora.eval().to("cuda")

    reloaded_audit_logits = []
    with torch.no_grad():
        for i in range(0, len(audit_pairs), 16):
            chunk = audit_pairs[i : i + 16]
            batch = tok(
                [c[0] for c in chunk],
                [c[1] for c in chunk],
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True,
            ).to("cuda")
            with torch.amp.autocast("cuda", dtype=torch.float16):
                logits = reloaded_lora(**batch).logits.view(-1).cpu().float().tolist()
                reloaded_audit_logits.extend(logits)

    logit_diffs = [
        abs(r - p) for r, p in zip(reloaded_audit_logits, post_audit_logits)
    ]
    max_logit_err = max(logit_diffs)
    reloaded_ranks = np.argsort(np.argsort(-np.array(reloaded_audit_logits)))
    ordering_identical = bool(np.array_equal(reloaded_ranks, post_ranks))

    roundtrip_passed = (max_logit_err <= 1e-4) and ordering_identical
    roundtrip = {
        "experiment_id": "HUY_D1_JINA_FT_CONTINUATION_V1",
        "adapter_dir": str(ADAPTER_DIR).replace("\\", "/"),
        "audit_pair_count": len(audit_pairs),
        "max_abs_logit_error": float(max_logit_err),
        "error_tolerance": 1e-4,
        "ordering_identical": ordering_identical,
        "roundtrip_passed": roundtrip_passed,
        "status": "PASS" if roundtrip_passed else "BLOCKED_CHECKPOINT_ROUNDTRIP",
    }
    with open(ROUNDTRIP_PATH, "w", encoding="utf-8") as f:
        json.dump(roundtrip, f, indent=2)
    print(f"Wrote {ROUNDTRIP_PATH}: max_err={max_logit_err:.8f}, identical={ordering_identical}", flush=True)

    del reloaded_lora, fresh_base
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "status": "SUCCESS" if roundtrip_passed else "BLOCKED_CHECKPOINT_ROUNDTRIP",
        "neural_proof": neural_proof,
        "roundtrip": roundtrip,
        "training_time_seconds": train_total_time,
    }


if __name__ == "__main__":
    train_jina_continuation()
