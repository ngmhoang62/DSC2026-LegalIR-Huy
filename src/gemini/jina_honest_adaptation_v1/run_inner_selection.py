"""Inner selection comparing M1 (LoRA) vs M2 (Full FT) and screening LR and lambda_anchor.
Implements Sections 7, 14, and 15 on outer fold 0 training population only.
Produces:
- results/gemini/jina_honest_adaptation_v1/NEURAL_TRAINING_PROOF.json
- results/gemini/jina_honest_adaptation_v1/FINETUNE_METHOD_SELECTION.json
"""

from __future__ import annotations

import json
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

from common import (
    EXP_CHECKPOINTS,
    EXP_RESULTS,
    REPO_ROOT,
    get_git_info,
    log_execution_trace,
    sha256_file,
)
from dataset import LegalRetrievalDataset
from model import build_model, compute_group_loss, compute_parent_scores
from neural_proof import NeuralTrainingProofTracker

sys.path.insert(0, str(REPO_ROOT))
from benchmark_jina_reranker_holdouts import top_passages


def evaluate_validation(
    model: nn.Module,
    tok: Any,
    dataset: LegalRetrievalDataset,
    val_qids: List[str],
    lambda_anchor: float = 0.05,
    device: str = "cuda",
) -> Tuple[float, float]:
    """Evaluate candidate pool on validation queries. Returns (recall_at_5, mean_loss)."""
    model.eval()
    recalls = []
    losses = []

    with torch.no_grad():
        for qid in val_qids:
            qtext = dataset.questions[qid]
            gold = dataset.golds[qid]
            pool = dataset.pools[qid]

            pairs = []
            doc_indices = []
            for doc_idx, d in enumerate(pool):
                dtext = dataset.contexts.get(d, "")
                passages = top_passages(qtext, dtext, count=2, window=220, overlap=70)
                for p in passages:
                    pairs.append((qtext, p))
                    doc_indices.append(doc_idx)

            parent_scores = compute_parent_scores(
                model, tok, pairs, doc_indices, len(pool), tau_pool=0.15, device=device
            )

            # Ranking metric
            scores_list = parent_scores.cpu().tolist()
            sorted_indices = sorted(range(len(pool)), key=lambda i: scores_list[i], reverse=True)
            top5 = [pool[i] for i in sorted_indices[:5]]
            hits = len(set(top5) & gold)
            recalls.append(hits / len(gold))

            # Ranking loss
            pos_indices = [i for i, d in enumerate(pool) if d in gold]
            if pos_indices:
                frozen = torch.tensor(
                    [dataset.frozen_scores.get((qid, d), 0.5) for d in pool],
                    device=device,
                    dtype=parent_scores.dtype,
                )
                tot_loss, _, _ = compute_group_loss(
                    parent_scores, pos_indices, frozen, lambda_anchor=lambda_anchor
                )
                losses.append(float(tot_loss.item()))

    mean_recall = float(sum(recalls) / len(recalls)) if recalls else 0.0
    mean_loss = float(sum(losses) / len(losses)) if losses else 0.0
    return mean_recall, mean_loss


def train_inner_model(
    method: str,
    lr: float,
    lambda_anchor: float,
    inner_train_qids: List[str],
    inner_val_qids: List[str],
    dataset: LegalRetrievalDataset,
    model_path: Path,
    epochs: int = 2,
    tracker: Optional[NeuralTrainingProofTracker] = None,
    save_checkpoint_dir: Optional[Path] = None,
    device: str = "cuda",
) -> Tuple[float, float, nn.Module, Any]:
    """Train Jina model for 2 epochs on inner_train and evaluate on inner_val."""
    print(
        f"\n--- Training {method} (lr={lr}, lambda_anchor={lambda_anchor}, epochs={epochs}) ---",
        flush=True,
    )
    model, tok = build_model(method, model_path, rank=16, dtype=torch.bfloat16, device=device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)

    if tracker is not None:
        tracker.before_training(model, tok, device=device)

    total_steps = 0
    start_time = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_start = time.perf_counter()
        running_loss = 0.0

        # Shuffle training queries per epoch deterministically
        epoch_rng = random.Random(2026 + epoch * 1000)
        shuffled_train = list(inner_train_qids)
        epoch_rng.shuffle(shuffled_train)

        for step_in_epoch, qid in enumerate(shuffled_train):
            sample = dataset.sample_query_candidates(
                qid, epoch=epoch, seed_offset=step_in_epoch
            )
            if not sample["pos_indices"]:
                continue

            optimizer.zero_grad()
            parent_scores = compute_parent_scores(
                model,
                tok,
                sample["pairs"],
                sample["doc_indices_for_passages"],
                sample["num_docs"],
                tau_pool=0.15,
                device=device,
            )

            frozen = torch.tensor(
                sample["frozen_doc_scores"], device=device, dtype=parent_scores.dtype
            )
            tot_loss, l_rank, l_anchor = compute_group_loss(
                parent_scores, sample["pos_indices"], frozen, lambda_anchor=lambda_anchor
            )

            tot_loss.backward()

            # Track steps
            if tracker is not None and total_steps % 50 == 0:
                tracker.record_step(
                    step=total_steps,
                    epoch=epoch,
                    loss=float(tot_loss.item()),
                    model=model,
                    lr=lr,
                    optimizer_class="AdamW",
                )

            optimizer.step()
            total_steps += 1
            running_loss += float(tot_loss.item())

            if (step_in_epoch + 1) % 500 == 0:
                rate = (time.perf_counter() - epoch_start) / (step_in_epoch + 1)
                print(
                    f"Epoch {epoch}/{epochs} | Step {step_in_epoch + 1}/{len(shuffled_train)} | Loss: {running_loss / (step_in_epoch + 1):.4f} | {rate:.3f} s/q",
                    flush=True,
                )

        epoch_time = time.perf_counter() - epoch_start
        print(f"Epoch {epoch} finished in {epoch_time:.1f}s, mean loss: {running_loss / len(shuffled_train):.4f}")

    # Evaluate on inner validation
    print(f"Evaluating {method} on {len(inner_val_qids)} inner validation queries...", flush=True)
    val_r5, val_loss = evaluate_validation(
        model, tok, dataset, inner_val_qids, lambda_anchor=lambda_anchor, device=device
    )
    print(f"Result {method} (lr={lr}, lambda={lambda_anchor}): Val Recall@5 = {val_r5:.4f}, Val Loss = {val_loss:.4f}")

    checkpoint_file = None
    if save_checkpoint_dir is not None:
        save_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_file = save_checkpoint_dir / "model.pt"
        # Save state dict
        trainable_state = {k: v.cpu() for k, v in model.state_dict().items() if v.requires_grad or "classifier" in k or "lora" in k}
        torch.save({"method": method, "state_dict": trainable_state, "val_r5": val_r5}, checkpoint_file)
        print(f"Saved checkpoint to {checkpoint_file}")

    if tracker is not None:
        tracker.after_training(
            model=model,
            tok=tok,
            checkpoint_path=checkpoint_file,
            val_loss=val_loss,
            val_recall_at_5=val_r5,
            device=device,
        )

    return val_r5, val_loss, model, tok


def run_inner_selection():
    start_time = time.perf_counter()
    model_path = REPO_ROOT / "cache/research_v2_forensic/models/jina-reranker-v2-base-multilingual"
    dataset = LegalRetrievalDataset(rng_seed=2026)

    # Outer fold 0 training population only (Section 14)
    fold0_train_qids = dataset.get_outer_train_qids("fold_0")
    print(f"Outer fold 0 training population: {len(fold0_train_qids)} queries.", flush=True)

    # Deterministic inner split
    rng = random.Random(2026)
    shuffled_qids = list(fold0_train_qids)
    rng.shuffle(shuffled_qids)

    inner_train_qids = shuffled_qids[:1500]
    inner_val_qids = shuffled_qids[1500:1750]
    audit_pool_qids = shuffled_qids[1750:1900]

    # Create 256 deterministic audit pairs from audit_pool_qids
    audit_pairs = []
    for qid in audit_pool_qids:
        qtext = dataset.questions[qid]
        for d in dataset.pools[qid][:2]:
            dtext = dataset.contexts.get(d, "")
            p = top_passages(qtext, dtext, count=1, window=220, overlap=70)
            if p:
                audit_pairs.append((qtext, p[0]))
            if len(audit_pairs) >= 256:
                break
        if len(audit_pairs) >= 256:
            break

    print(f"Inner split: {len(inner_train_qids)} train, {len(inner_val_qids)} val, {len(audit_pairs)} audit pairs.")

    # Evaluate frozen Jina baseline on inner val first
    print("\nEvaluating FROZEN Jina baseline on inner validation...", flush=True)
    frozen_model, frozen_tok = build_model("M2_FULL", model_path, dtype=torch.bfloat16)
    frozen_r5, frozen_loss = evaluate_validation(frozen_model, frozen_tok, dataset, inner_val_qids)
    print(f"FROZEN Jina baseline on inner val: Recall@5 = {frozen_r5:.4f}, Loss = {frozen_loss:.4f}")
    del frozen_model
    torch.cuda.empty_cache()

    # Stage 1: Compare M1 (LoRA) vs M2 (Full FT)
    # Tracker records Section 7 proof on the first adapted run (M1)
    tracker = NeuralTrainingProofTracker("inner_selection_m1_lora", audit_pairs)

    m1_r5, m1_loss, model_m1, _ = train_inner_model(
        method="M1_LORA",
        lr=1e-5,
        lambda_anchor=0.05,
        inner_train_qids=inner_train_qids,
        inner_val_qids=inner_val_qids,
        dataset=dataset,
        model_path=model_path,
        epochs=2,
        tracker=tracker,
    )
    del model_m1
    torch.cuda.empty_cache()

    m2_r5, m2_loss, model_m2, _ = train_inner_model(
        method="M2_FULL",
        lr=1e-5,
        lambda_anchor=0.05,
        inner_train_qids=inner_train_qids,
        inner_val_qids=inner_val_qids,
        dataset=dataset,
        model_path=model_path,
        epochs=2,
        tracker=None,
    )
    del model_m2
    torch.cuda.empty_cache()

    # Determine method winner (Section 14)
    if m2_r5 > m1_r5:
        selected_method = "M2_FULL"
        method_rationale = f"M2 (Full FT) achieved higher Recall@5 ({m2_r5:.4f} vs {m1_r5:.4f})"
    elif m1_r5 > m2_r5:
        selected_method = "M1_LORA"
        method_rationale = f"M1 (LoRA) achieved higher Recall@5 ({m1_r5:.4f} vs {m2_r5:.4f})"
    else:
        # Tie-break on loss
        selected_method = "M1_LORA" if m1_loss < m2_loss else "M2_FULL"
        method_rationale = f"Tied Recall@5 ({m1_r5:.4f}); selected {selected_method} by lower ranking loss"

    print(f"\n=======================================================")
    print(f"METHOD SELECTION WINNER: {selected_method} ({method_rationale})")
    print(f"=======================================================\n")

    # Stage 2: Screen lr and lambda_anchor for selected method (Section 15)
    # Tested: (1e-5, 0.05) is already run. Test remaining combinations.
    grid_results = [
        {
            "method": selected_method,
            "lr": 1e-5,
            "lambda_anchor": 0.05,
            "val_recall_at_5": m1_r5 if selected_method == "M1_LORA" else m2_r5,
            "val_loss": m1_loss if selected_method == "M1_LORA" else m2_loss,
        }
    ]

    remaining_grid = [
        (5e-6, 0.05),
        (2e-5, 0.05),
        (1e-5, 0.15),
    ]

    best_r5 = grid_results[0]["val_recall_at_5"]
    best_loss = grid_results[0]["val_loss"]
    best_recipe = (selected_method, 1e-5, 0.05)

    for lr, lambda_anchor in remaining_grid:
        r5, loss, _, _ = train_inner_model(
            method=selected_method,
            lr=lr,
            lambda_anchor=lambda_anchor,
            inner_train_qids=inner_train_qids,
            inner_val_qids=inner_val_qids,
            dataset=dataset,
            model_path=model_path,
            epochs=2,
        )
        grid_results.append(
            {
                "method": selected_method,
                "lr": lr,
                "lambda_anchor": lambda_anchor,
                "val_recall_at_5": r5,
                "val_loss": loss,
            }
        )
        if r5 > best_r5 or (r5 == best_r5 and loss < best_loss):
            best_r5 = r5
            best_loss = loss
            best_recipe = (selected_method, lr, lambda_anchor)

    print(f"\n=======================================================")
    print(f"LOCKED RECIPE: method={best_recipe[0]}, lr={best_recipe[1]}, lambda_anchor={best_recipe[2]}")
    print(f"Best Val Recall@5: {best_r5:.4f} (Frozen baseline: {frozen_r5:.4f})")
    print(f"=======================================================\n")

    report = {
        "schema_version": "dsc2026.gemini.finetune_method_selection.v1",
        "status": "SEALED",
        "population": {
            "outer_training_fold": "fold_0",
            "total_outer_train_queries": len(fold0_train_qids),
            "inner_train_queries": len(inner_train_qids),
            "inner_val_queries": len(inner_val_qids),
            "random_seed": 2026,
        },
        "frozen_jina_baseline": {
            "inner_val_recall_at_5": frozen_r5,
            "inner_val_loss": frozen_loss,
        },
        "method_comparison": {
            "m1_lora": {
                "val_recall_at_5": m1_r5,
                "val_loss": m1_loss,
                "delta_vs_frozen": m1_r5 - frozen_r5,
            },
            "m2_full": {
                "val_recall_at_5": m2_r5,
                "val_loss": m2_loss,
                "delta_vs_frozen": m2_r5 - frozen_r5,
            },
            "selected_method": selected_method,
            "selection_rationale": method_rationale,
        },
        "hyperparameter_grid": grid_results,
        "locked_recipe": {
            "method": best_recipe[0],
            "learning_rate": best_recipe[1],
            "lambda_anchor": best_recipe[2],
            "epochs": 2,
            "tau_pool": 0.15,
            "T": 1.0,
            "batch_size_queries": 1,
            "val_recall_at_5": best_r5,
            "gain_over_frozen": best_r5 - frozen_r5,
        },
        "git": get_git_info(),
        "runtime_sec": round(time.perf_counter() - start_time, 3),
    }

    out_path = EXP_RESULTS / "FINETUNE_METHOD_SELECTION.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")

    log_execution_trace(
        stage_name="inner_method_and_lr_selection",
        command=f"python {__file__}",
        code_hash=sha256_file(Path(__file__)),
        model_checkpoint_hash="IN_MEMORY",
        train_query_count=len(inner_train_qids),
        val_query_count=len(inner_val_qids),
        optimizer_steps=len(inner_train_qids) * 2,
        gpu_info=torch.cuda.get_device_name(0),
        runtime_sec=time.perf_counter() - start_time,
        output_hashes={
            "FINETUNE_METHOD_SELECTION.json": sha256_file(out_path),
            "NEURAL_TRAINING_PROOF.json": sha256_file(EXP_RESULTS / "NEURAL_TRAINING_PROOF.json"),
        },
        status=report["status"],
        extra={"locked_recipe": report["locked_recipe"]},
    )


if __name__ == "__main__":
    run_inner_selection()
