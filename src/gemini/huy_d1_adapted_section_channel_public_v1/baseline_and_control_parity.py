"""Stage 3: Verify A0 (D1 48D baseline) and A1 (D1 + frozen Section CE 50D control) parity on CAL600."""

from __future__ import annotations

import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from tune_expanded_fusion_selection import ltr_features

from .common import (
    D1_VIEWS,
    EXPECTED_BLOCK_RECALLS,
    EXPECTED_D1_R5,
    EXPECTED_FROZEN_SECTION_R5,
    FROZEN_SECTION_CACHE_PATH,
    RESULTS_DIR,
    get_git_status,
    load_cal_data,
    seed_everything,
)


def evaluate_lobo_arm(
    arm_name: str,
    channels: Dict[str, Any],
    local_views: Dict[str, Any],
    extended: Dict[str, List[str]],
    type_rows: Dict[str, Any],
    cite_rows: Dict[str, Any],
    blocks: Dict[str, List[str]],
    all_ids: List[str],
    gold: Dict[str, Set[str]],
) -> Tuple[Dict[str, List[str]], int, Dict[str, Any], Dict[str, np.ndarray]]:
    preds: Dict[str, List[str]] = {}
    scores_dict: Dict[str, np.ndarray] = {}
    feature_dim = 0
    block_metrics = {}

    for held in sorted(blocks.keys()):
        train_ids = sum((blocks[n] for n in blocks if n != held), [])
        test_ids = blocks[held]
        eval_ids = train_ids + test_ids

        eval_rows, eval_groups = ltr_features(
            local_views, D1_VIEWS, extended, eval_ids, channels
        )
        for q in eval_rows:
            eval_rows[q] = np.concatenate(
                [eval_rows[q], type_rows[q], cite_rows[q]], axis=1
            )

        feature_dim = eval_rows[all_ids[0]].shape[1]

        X_train = np.vstack([eval_rows[q] for q in train_ids])
        y_train = np.concatenate(
            [[d in gold[q] for d in eval_groups[q]] for q in train_ids]
        ).astype(np.int8)

        scaler = StandardScaler().fit(X_train)
        model = LogisticRegression(
            C=0.15,
            class_weight="balanced",
            solver="liblinear",
            max_iter=3000,
            random_state=2026,
        )
        model.fit(scaler.transform(X_train), y_train)

        b_recalls = []
        for q in test_ids:
            X_test = scaler.transform(eval_rows[q])
            dec_scores = model.decision_function(X_test)
            order = sorted(
                range(len(dec_scores)), key=lambda i: dec_scores[i], reverse=True
            )
            top5 = [eval_groups[q][i] for i in order[:5]]
            preds[q] = top5
            scores_dict[q] = dec_scores
            b_recalls.append(len(set(top5) & gold[q]) / max(1, len(gold[q])))

        block_metrics[held] = float(np.mean(b_recalls))

    pooled_recalls = [
        len(set(preds[q]) & gold[q]) / max(1, len(gold[q])) for q in all_ids
    ]
    pooled_precisions = [len(set(preds[q]) & gold[q]) / 5.0 for q in all_ids]
    single_gold_recalls = [
        len(set(preds[q]) & gold[q]) / max(1, len(gold[q]))
        for q in all_ids
        if len(gold[q]) == 1
    ]
    multi_gold_recalls = [
        len(set(preds[q]) & gold[q]) / max(1, len(gold[q]))
        for q in all_ids
        if len(gold[q]) > 1
    ]

    metrics = {
        "arm": arm_name,
        "feature_dim": feature_dim,
        "recall_at_5": float(np.mean(pooled_recalls)),
        "precision_at_5": float(np.mean(pooled_precisions)),
        "single_gold_recall_at_5": float(np.mean(single_gold_recalls)),
        "multi_gold_recall_at_5": float(np.mean(multi_gold_recalls)),
        "block_recalls": block_metrics,
    }
    return preds, feature_dim, metrics, scores_dict


def run_baseline_and_control_parity() -> Dict[str, Any]:
    print("=== STAGE 3: BASELINE & CONTROL PARITY AUDIT ===", flush=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    seed_everything(2026)
    git_info = get_git_status()

    # 1. Load CAL dataset
    print("Loading CAL dataset...", flush=True)
    docs, queries, blocks, all_ids, extended, local_views, full_channels_cv, gold, type_rows, cite_rows = load_cal_data()

    # 2. Evaluate A0: D1 48D Baseline
    print("Evaluating A0 (D1 48D Baseline)...", flush=True)
    a0_preds, a0_dim, a0_metrics, _ = evaluate_lobo_arm(
        arm_name="A0_D1_BASELINE",
        channels=full_channels_cv,
        local_views=local_views,
        extended=extended,
        type_rows=type_rows,
        cite_rows=cite_rows,
        blocks=blocks,
        all_ids=all_ids,
        gold=gold,
    )

    a0_r5 = a0_metrics["recall_at_5"]
    a0_r5_diff = abs(a0_r5 - EXPECTED_D1_R5)
    a0_block_diffs = {
        b: abs(a0_metrics["block_recalls"][b] - EXPECTED_BLOCK_RECALLS[b])
        for b in ["A", "B", "C", "D"]
    }
    a0_parity_passed = (
        a0_dim == 48
        and a0_r5_diff < 1e-12
        and all(d < 1e-9 for d in a0_block_diffs.values())
    )

    print(f"A0 Feature Dim: {a0_dim} (Expected: 48)")
    print(f"A0 Recall@5:    {a0_r5:.16f} (Expected: {EXPECTED_D1_R5:.16f}, diff: {a0_r5_diff:.2e})")
    for b in sorted(a0_metrics["block_recalls"]):
        print(f"  Block {b}: {a0_metrics['block_recalls'][b]:.6f} (Expected: {EXPECTED_BLOCK_RECALLS[b]:.6f})")
    print(f"A0 Baseline Parity: {'PASS' if a0_parity_passed else 'FAIL'}", flush=True)

    if not a0_parity_passed:
        raise RuntimeError(f"BLOCKED_BASELINE_PARITY: A0 baseline parity failed! Diff: {a0_r5_diff:.2e}")

    # 3. Evaluate A1: D1 + Frozen Section CE (50D Control)
    print("Evaluating A1 (D1 + Frozen Section CE 50D Control)...", flush=True)
    if not FROZEN_SECTION_CACHE_PATH.exists():
        raise FileNotFoundError(f"Missing frozen Section CE cache: {FROZEN_SECTION_CACHE_PATH}")

    cached_obj = pickle.loads(FROZEN_SECTION_CACHE_PATH.read_bytes())
    frozen_scores = cached_obj["scores"] if isinstance(cached_obj, dict) and "scores" in cached_obj else cached_obj

    # Align frozen section scores to candidate pool
    fl_frozen = min(v for q in frozen_scores for v in frozen_scores[q].values())
    aligned_frozen = {
        q: {d: frozen_scores.get(q, {}).get(d, fl_frozen) for d in extended[q]}
        for q in all_ids
    }
    channels_a1 = {
        **full_channels_cv,
        "legal_section_ce": aligned_frozen,
    }

    a1_preds, a1_dim, a1_metrics, _ = evaluate_lobo_arm(
        arm_name="A1_D1_PLUS_FROZEN_SECTION_CE",
        channels=channels_a1,
        local_views=local_views,
        extended=extended,
        type_rows=type_rows,
        cite_rows=cite_rows,
        blocks=blocks,
        all_ids=all_ids,
        gold=gold,
    )

    a1_r5 = a1_metrics["recall_at_5"]
    a1_r5_diff = abs(a1_r5 - EXPECTED_FROZEN_SECTION_R5)
    a1_parity_passed = (a1_dim == 50 and a1_r5_diff < 1e-4)

    print(f"A1 Feature Dim: {a1_dim} (Expected: 50)")
    print(f"A1 Recall@5:    {a1_r5:.16f} (Expected: {EXPECTED_FROZEN_SECTION_R5:.16f}, diff: {a1_r5_diff:.2e})")
    for b in sorted(a1_metrics["block_recalls"]):
        print(f"  Block {b}: {a1_metrics['block_recalls'][b]:.6f}")
    print(f"A1 Control Parity:  {'PASS' if a1_parity_passed else 'FAIL'}", flush=True)

    if not a1_parity_passed:
        raise RuntimeError(f"BLOCKED_BASELINE_PARITY: A1 control reproduction failed! Diff: {a1_r5_diff:.2e}")

    parity_report = {
        "schema_version": "dsc2026.gemini.huy_d1_adapted_section_channel_public_v1.baseline_parity.v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_info["head_commit"],
        "status": "PASS",
        "a0_d1_baseline": {
            "metrics": a0_metrics,
            "expected_recall_5": EXPECTED_D1_R5,
            "difference": a0_r5_diff,
            "parity_passed": bool(a0_parity_passed),
        },
        "a1_frozen_section_control": {
            "metrics": a1_metrics,
            "expected_recall_5": EXPECTED_FROZEN_SECTION_R5,
            "difference": float(a1_r5_diff),
            "parity_passed": bool(a1_parity_passed),
        },
    }

    parity_path = RESULTS_DIR / "BASELINE_AND_CONTROL_PARITY.json"
    parity_path.write_text(json.dumps(parity_report, indent=2), encoding="utf-8")
    print(f"Wrote {parity_path}", flush=True)
    return parity_report


if __name__ == "__main__":
    run_baseline_and_control_parity()
