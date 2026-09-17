"""Stage 2: Local Parity Gate - Verify P0 (D1 48D baseline) and P1 (D1 + frozen Section CE 50D control) parity on CAL600."""

from __future__ import annotations

import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from tune_expanded_fusion_selection import ltr_features

from .common import (
    CAL_FROZEN_SECTION_CACHE_PATH,
    D1_VIEWS,
    EXPECTED_BLOCK_RECALLS,
    EXPECTED_D1_R5,
    EXPECTED_FROZEN_BLOCK_RECALLS,
    EXPECTED_FROZEN_SECTION_R5,
    RESULTS_DIR,
    get_git_status,
    load_cal_data_label_free,
    load_cal_gold_labels,
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


def run_local_parity(
    cal_data: Optional[Tuple] = None,
    gold: Optional[Dict[str, Set[str]]] = None,
) -> Dict[str, Any]:
    print("=== STAGE 2: LOCAL PARITY GATE (CAL600) ===", flush=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    seed_everything(2026)
    git_info = get_git_status()

    # 1. Load CAL dataset
    if cal_data is None or gold is None:
        print("Loading CAL dataset and gold labels...", flush=True)
        docs, queries, blocks, all_ids, extended, local_views, full_channels_cv, type_rows, cite_rows = load_cal_data_label_free()
        gold, _ = load_cal_gold_labels(all_ids)
    else:
        docs, queries, blocks, all_ids, extended, local_views, full_channels_cv, type_rows, cite_rows = cal_data

    # 2. Evaluate P0: D1 48D Baseline
    print("Evaluating P0 (D1 48D Baseline)...", flush=True)
    p0_preds, p0_dim, p0_metrics, _ = evaluate_lobo_arm(
        arm_name="P0_D1_BASELINE",
        channels=full_channels_cv,
        local_views=local_views,
        extended=extended,
        type_rows=type_rows,
        cite_rows=cite_rows,
        blocks=blocks,
        all_ids=all_ids,
        gold=gold,
    )

    p0_r5 = p0_metrics["recall_at_5"]
    p0_r5_diff = abs(p0_r5 - EXPECTED_D1_R5)
    p0_block_diffs = {
        b: abs(p0_metrics["block_recalls"][b] - EXPECTED_BLOCK_RECALLS[b])
        for b in ["A", "B", "C", "D"]
    }
    p0_parity_passed = (
        p0_dim == 48
        and p0_r5_diff < 1e-12
        and all(d < 1e-9 for d in p0_block_diffs.values())
    )

    print(f"P0 Feature Dim: {p0_dim} (Expected: 48)")
    print(f"P0 Recall@5:    {p0_r5:.16f} (Expected: {EXPECTED_D1_R5:.16f}, diff: {p0_r5_diff:.2e})")
    for b in sorted(p0_metrics["block_recalls"]):
        print(f"  Block {b}: {p0_metrics['block_recalls'][b]:.6f} (Expected: {EXPECTED_BLOCK_RECALLS[b]:.6f})")
    print(f"P0 Baseline Parity: {'PASS' if p0_parity_passed else 'FAIL'}", flush=True)

    if not p0_parity_passed:
        raise RuntimeError(f"BLOCKED_LOCAL_PARITY: P0 baseline parity failed! Diff: {p0_r5_diff:.2e}")

    # 3. Evaluate P1: D1 + Frozen Section CE (50D Control)
    print("Evaluating P1 (D1 + Frozen Section CE 50D Control)...", flush=True)
    if not CAL_FROZEN_SECTION_CACHE_PATH.exists():
        raise FileNotFoundError(f"Missing frozen Section CE cache: {CAL_FROZEN_SECTION_CACHE_PATH}")

    cached_obj = pickle.loads(CAL_FROZEN_SECTION_CACHE_PATH.read_bytes())
    frozen_scores = cached_obj["scores"] if isinstance(cached_obj, dict) and "scores" in cached_obj else cached_obj

    fl_frozen = min(v for q in frozen_scores for v in frozen_scores[q].values())
    aligned_frozen = {
        q: {d: frozen_scores.get(q, {}).get(d, fl_frozen) for d in extended[q]}
        for q in all_ids
    }
    channels_p1 = {
        **full_channels_cv,
        "legal_section_ce": aligned_frozen,
    }

    p1_preds, p1_dim, p1_metrics, _ = evaluate_lobo_arm(
        arm_name="P1_D1_PLUS_FROZEN_SECTION_CE",
        channels=channels_p1,
        local_views=local_views,
        extended=extended,
        type_rows=type_rows,
        cite_rows=cite_rows,
        blocks=blocks,
        all_ids=all_ids,
        gold=gold,
    )

    p1_r5 = p1_metrics["recall_at_5"]
    p1_r5_diff = abs(p1_r5 - EXPECTED_FROZEN_SECTION_R5)
    p1_block_diffs = {
        b: abs(p1_metrics["block_recalls"][b] - EXPECTED_FROZEN_BLOCK_RECALLS[b])
        for b in ["A", "B", "C", "D"]
    }

    # Paired comparison P1 vs P0
    wins = sum(
        1
        for q in all_ids
        if (len(set(p1_preds[q]) & gold[q]) / max(1, len(gold[q])))
        > (len(set(p0_preds[q]) & gold[q]) / max(1, len(gold[q])))
    )
    losses = sum(
        1
        for q in all_ids
        if (len(set(p1_preds[q]) & gold[q]) / max(1, len(gold[q])))
        < (len(set(p0_preds[q]) & gold[q]) / max(1, len(gold[q])))
    )
    ties = len(all_ids) - wins - losses

    p1_parity_passed = (
        p1_dim == 50
        and p1_r5_diff < 1e-12
        and all(d < 1e-9 for d in p1_block_diffs.values())
        and abs(p1_metrics["precision_at_5"] - 0.206) < 1e-5
        and abs(p1_metrics["single_gold_recall_at_5"] - 0.9745454545454545) < 1e-9
        and abs(p1_metrics["multi_gold_recall_at_5"] - 0.7733333333333333) < 1e-9
        and wins == 2
        and losses == 1
        and ties == 597
    )

    print(f"P1 Feature Dim: {p1_dim} (Expected: 50)")
    print(f"P1 Recall@5:    {p1_r5:.16f} (Expected: {EXPECTED_FROZEN_SECTION_R5:.16f}, diff: {p1_r5_diff:.2e})")
    print(f"P1 Precision@5: {p1_metrics['precision_at_5']:.6f} (Expected: 0.206000)")
    print(f"P1 Single R@5:  {p1_metrics['single_gold_recall_at_5']:.6f} (Expected: 0.974545)")
    print(f"P1 Multi R@5:   {p1_metrics['multi_gold_recall_at_5']:.6f} (Expected: 0.773333)")
    for b in sorted(p1_metrics["block_recalls"]):
        print(f"  Block {b}: {p1_metrics['block_recalls'][b]:.6f} (Expected: {EXPECTED_FROZEN_BLOCK_RECALLS[b]:.6f})")
    print(f"P1 vs P0 Paired: {wins} wins / {losses} losses / {ties} ties (Expected: 2 / 1 / 597)")
    print(f"P1 Control Parity:  {'PASS' if p1_parity_passed else 'FAIL'}", flush=True)

    parity_report = {
        "schema_version": "dsc2026.gemini.huy_d1_frozen_section_public_v1.local_parity.v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_info["head_commit"],
        "status": "PASS" if (p0_parity_passed and p1_parity_passed) else "BLOCKED_LOCAL_PARITY",
        "p0_d1_baseline": {
            "metrics": p0_metrics,
            "expected_recall_5": EXPECTED_D1_R5,
            "difference": p0_r5_diff,
            "parity_passed": p0_parity_passed,
        },
        "p1_frozen_section_control": {
            "metrics": p1_metrics,
            "expected_recall_5": EXPECTED_FROZEN_SECTION_R5,
            "difference": p1_r5_diff,
            "parity_passed": p1_parity_passed,
        },
        "paired_p1_vs_p0": {
            "wins": wins,
            "losses": losses,
            "ties": ties,
            "expected": {"wins": 2, "losses": 1, "ties": 597},
            "parity_passed": (wins == 2 and losses == 1 and ties == 597),
        },
    }

    parity_path = RESULTS_DIR / "LOCAL_PARITY.json"
    parity_path.write_text(json.dumps(parity_report, indent=2), encoding="utf-8")
    print(f"Wrote {parity_path}", flush=True)

    if not (p0_parity_passed and p1_parity_passed):
        raise RuntimeError("BLOCKED_LOCAL_PARITY: Local parity check failed!")

    return parity_report


if __name__ == "__main__":
    run_local_parity()
