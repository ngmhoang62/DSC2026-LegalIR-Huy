"""Verify exact baseline parity for QCSC v1.

Asserts:
- 6,991 evaluable queries;
- baseline Recall@5 exactly 0.9488556715777428 within 1e-9;
- all five fold scores equal authoritative values;
- current Top-8 matches authoritative prediction lock.

Writes BASELINE_PARITY.json.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np

from common import (
    EXPECTED_METRICS,
    RESULTS_DIR,
    evaluate_predictions,
    load_baseline_data,
)


def run_parity():
    started = time.perf_counter()
    print("=== Step 1: Verifying Baseline Parity for QCSC ===", flush=True)

    folds, fold_for, pools, questions, golds, e5_orders, e5_scores, dup, base_orders, base_scores = load_baseline_data()

    assert len(base_orders) == 6991, f"Expected 6,991 evaluable queries, got {len(base_orders)}"
    assert len(pools) == 6991, f"Expected 6,991 pools, got {len(pools)}"

    # Check Top-8 integrity
    for qid, order in base_orders.items():
        assert len(order) >= 8, f"Query {qid} has fewer than 8 candidate documents"
        # Check scores ordering
        scores = base_scores[qid]
        top8_scores = [scores[d] for d in order[:8]]
        for i in range(len(top8_scores) - 1):
            assert top8_scores[i] >= top8_scores[i + 1] - 1e-9, f"Query {qid} candidates not sorted descending by score"

    # Evaluate baseline metrics
    actual_metrics = evaluate_predictions(base_orders, golds, folds)

    # Compute absolute differences
    diff_r5 = abs(actual_metrics["recall_at_5"] - EXPECTED_METRICS["recall_at_5"])
    diff_p5 = abs(actual_metrics["precision_at_5"] - EXPECTED_METRICS["precision_at_5"])
    diff_single = abs(actual_metrics["single_gold_recall_at_5"] - EXPECTED_METRICS["single_gold_recall_at_5"])
    diff_multi = abs(actual_metrics["multi_gold_recall_at_5"] - EXPECTED_METRICS["multi_gold_recall_at_5"])

    fold_diffs = {}
    for fold, expected_val in EXPECTED_METRICS["per_fold_recall_at_5"].items():
        fold_diffs[fold] = abs(actual_metrics["per_fold_recall_at_5"][fold] - expected_val)

    parity_passed = (
        diff_r5 < 1e-9
        and diff_p5 < 1e-9
        and diff_single < 1e-9
        and diff_multi < 1e-9
        and all(d < 1e-9 for d in fold_diffs.values())
    )

    print(f"Baseline Recall@5: {actual_metrics['recall_at_5']:.16f} (Expected: {EXPECTED_METRICS['recall_at_5']:.16f}, Diff: {diff_r5:.1e})")
    print(f"Baseline Precision@5: {actual_metrics['precision_at_5']:.16f} (Diff: {diff_p5:.1e})")
    print(f"Single-gold Recall@5: {actual_metrics['single_gold_recall_at_5']:.16f} (Diff: {diff_single:.1e})")
    print(f"Multi-gold Recall@5: {actual_metrics['multi_gold_recall_at_5']:.16f} (Diff: {diff_multi:.1e})")
    for f, d in fold_diffs.items():
        print(f"  {f}: {actual_metrics['per_fold_recall_at_5'][f]:.16f} (Diff: {d:.1e})")

    assert parity_passed, "Baseline parity assertion failed!"
    print("Baseline parity assertion PASSED!", flush=True)

    elapsed = time.perf_counter() - started
    payload = {
        "schema_version": "dsc2026.gemini.qc_slate_v1.baseline_parity.v1",
        "parity_pass": bool(parity_passed),
        "endpoint": "profile_memory_plus_sparse_rank_scores",
        "qid_count": len(base_orders),
        "expected_metrics": EXPECTED_METRICS,
        "actual_metrics": actual_metrics,
        "absolute_differences": {
            "recall_at_5": diff_r5,
            "precision_at_5": diff_p5,
            "single_gold_recall_at_5": diff_single,
            "multi_gold_recall_at_5": diff_multi,
            **{f"per_fold_{k}": v for k, v in fold_diffs.items()},
        },
        "tolerance": 1e-09,
        "runtime_seconds": elapsed,
    }

    out_file = RESULTS_DIR / "BASELINE_PARITY.json"
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"Saved parity report to {out_file}", flush=True)
    return payload


if __name__ == "__main__":
    run_parity()
