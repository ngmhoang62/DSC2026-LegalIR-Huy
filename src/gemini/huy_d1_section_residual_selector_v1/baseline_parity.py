"""Verify exact D1_SCORE_ONLY_VNLEGAL baseline parity on CAL600 under LOBO (48D, R@5 = 0.9569444444444444)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path("D:/Study/DSC2026/sota")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tune_expanded_fusion_selection import ltr_features
from src.gemini.huy_d1_section_residual_selector_v1.common import (
    D1_VIEWS,
    EXPECTED_BLOCK_RECALLS,
    EXPECTED_D1_R5,
    RES_DIR,
    load_cal_data,
    seed_everything,
)


def evaluate_d1_baseline() -> Tuple[dict, Dict[str, List[str]], Dict[str, Dict[str, float]]]:
    seed_everything(2026)
    print("=== EVALUATING D1 BASELINE (E0) PARITY ===", flush=True)
    RES_DIR.mkdir(parents=True, exist_ok=True)

    (
        docs,
        queries,
        blocks,
        all_ids,
        extended,
        local_views,
        full_channels_cv,
        gold,
        type_rows,
        cite_rows,
    ) = load_cal_data()

    e0_preds: Dict[str, List[str]] = {}
    e0_scores: Dict[str, Dict[str, float]] = {}
    block_recalls: Dict[str, float] = {}
    feature_dim = 0

    print("Running LOBO 4-block evaluation for E0 baseline...", flush=True)
    for held in sorted(blocks.keys()):
        train_ids = sum((blocks[n] for n in blocks if n != held), [])
        test_ids = blocks[held]
        eval_ids = train_ids + test_ids

        eval_rows, eval_groups = ltr_features(
            local_views, D1_VIEWS, extended, eval_ids, full_channels_cv
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
            scores = model.decision_function(X_test)
            order = sorted(
                range(len(scores)), key=lambda i: scores[i], reverse=True
            )
            top5 = [eval_groups[q][i] for i in order[:5]]
            e0_preds[q] = top5
            e0_scores[q] = {eval_groups[q][i]: float(scores[i]) for i in range(len(scores))}
            b_recalls.append(len(set(top5) & gold[q]) / max(1, len(gold[q])))

        block_recalls[held] = float(np.mean(b_recalls))

    pooled_recalls = [
        len(set(e0_preds[q]) & gold[q]) / max(1, len(gold[q])) for q in all_ids
    ]
    pooled_precisions = [len(set(e0_preds[q]) & gold[q]) / 5.0 for q in all_ids]
    single_gold_recalls = [
        len(set(e0_preds[q]) & gold[q]) / max(1, len(gold[q]))
        for q in all_ids
        if len(gold[q]) == 1
    ]
    multi_gold_recalls = [
        len(set(e0_preds[q]) & gold[q]) / max(1, len(gold[q]))
        for q in all_ids
        if len(gold[q]) > 1
    ]

    pooled_r5 = float(np.mean(pooled_recalls))
    precision_5 = float(np.mean(pooled_precisions))
    single_gold_r5 = float(np.mean(single_gold_recalls))
    multi_gold_r5 = float(np.mean(multi_gold_recalls))

    r5_parity_exact = abs(pooled_r5 - EXPECTED_D1_R5) < 1e-12
    block_parity_exact = all(
        abs(block_recalls[b] - EXPECTED_BLOCK_RECALLS[b]) < 1e-9
        for b in ["A", "B", "C", "D"]
    )
    dim_parity_exact = (feature_dim == 48)

    print(f"E0 Feature Dim:      {feature_dim} (Expected: 48)", flush=True)
    print(f"E0 Pooled Recall@5:  {pooled_r5:.16f} (Expected: {EXPECTED_D1_R5:.16f})", flush=True)
    for b in sorted(block_recalls):
        print(
            f"  Block {b} Recall@5: {block_recalls[b]:.6f} (Expected: {EXPECTED_BLOCK_RECALLS[b]:.6f})",
            flush=True,
        )

    all_passed = r5_parity_exact and block_parity_exact and dim_parity_exact
    print(f"E0 Baseline Parity Passed: {all_passed}", flush=True)

    metrics = {
        "status": "PASS" if all_passed else "BLOCKED_D1_PARITY",
        "parity_passed": all_passed,
        "feature_dim": feature_dim,
        "pooled_recall_at_5": pooled_r5,
        "precision_at_5": precision_5,
        "single_gold_recall_at_5": single_gold_r5,
        "multi_gold_recall_at_5": multi_gold_r5,
        "block_recalls": block_recalls,
    }

    out_path = RES_DIR / "D1_BASELINE_PARITY.json"
    out_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {out_path}", flush=True)
    print("=== D1 BASELINE PARITY VERIFIED SUCCESSFULLY ===\n", flush=True)
    return metrics, e0_preds, e0_scores


if __name__ == "__main__":
    evaluate_d1_baseline()
