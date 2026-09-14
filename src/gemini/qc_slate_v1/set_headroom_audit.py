"""Compute independent dataset statistics and theoretical Top8-to-Top5 headroom.

Writes SET_HEADROOM_AUDIT.json.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict

import numpy as np

from common import (
    RESULTS_DIR,
    evaluate_predictions,
    load_baseline_data,
)


def run_headroom_audit():
    started = time.perf_counter()
    print("=== Step 2: Computing Dataset Statistics and Headroom Audit ===", flush=True)

    folds, fold_for, pools, questions, golds, e5_orders, e5_scores, dup, base_orders, base_scores = load_baseline_data()

    single_qids = [q for q, g in golds.items() if len(g) == 1]
    multi_qids = [q for q, g in golds.items() if len(g) > 1]
    gold_dist = Counter(len(g) for g in golds.values())

    # Baseline metrics
    base_m = evaluate_predictions(base_orders, golds, folds)

    # Check Top-8 headroom queries and theoretical oracle
    headroom_qids = []
    oracle_hits_list = []
    oracle_single_hits = []
    oracle_multi_hits = []
    oracle_by_fold = {f: [] for f in folds}

    for qid in sorted(base_orders, key=int):
        gold = golds[qid]
        top5 = set(base_orders[qid][:5]) & gold
        top8 = set(base_orders[qid][:8]) & gold

        if len(top8) > len(top5):
            headroom_qids.append(qid)

        # Oracle chooses any 5 documents from Top-8 to maximize hit count
        oracle_hits = min(5, len(top8))
        hit_ratio = oracle_hits / len(gold)
        oracle_hits_list.append(hit_ratio)
        if len(gold) == 1:
            oracle_single_hits.append(hit_ratio)
        else:
            oracle_multi_hits.append(hit_ratio)

        oracle_by_fold[fold_for[qid]].append(hit_ratio)

    oracle_r5 = float(np.mean(oracle_hits_list))
    oracle_single_r5 = float(np.mean(oracle_single_hits))
    oracle_multi_r5 = float(np.mean(oracle_multi_hits))
    oracle_per_fold = {f: float(np.mean(v)) for f, v in oracle_by_fold.items()}
    theoretical_gain = oracle_r5 - base_m["recall_at_5"]

    print(f"Total evaluable queries: {len(golds)}")
    print(f"Single-gold count: {len(single_qids)} ({len(single_qids)/len(golds)*100:.2f}%)")
    print(f"Multi-gold count: {len(multi_qids)} ({len(multi_qids)/len(golds)*100:.2f}%)")
    print(f"Gold label set size distribution: {dict(sorted(gold_dist.items()))}")
    print(f"Queries with Top-8 containing more gold than Top-5: {len(headroom_qids)} ({len(headroom_qids)/len(golds)*100:.2f}%)")
    print(f"Theoretical Top8-to-Top5 Oracle R@5: {oracle_r5:.16f} (Gain: +{theoretical_gain:.16f})")
    print(f"Oracle Single-gold R@5: {oracle_single_r5:.16f} (Gain: +{oracle_single_r5 - base_m['single_gold_recall_at_5']:.16f})")
    print(f"Oracle Multi-gold R@5: {oracle_multi_r5:.16f} (Gain: +{oracle_multi_r5 - base_m['multi_gold_recall_at_5']:.16f})")

    elapsed = time.perf_counter() - started
    payload = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_queries": len(golds),
        "single_gold_count": len(single_qids),
        "multi_gold_count": len(multi_qids),
        "gold_size_distribution": {str(k): v for k, v in sorted(gold_dist.items())},
        "baseline_metrics": base_m,
        "headroom_queries_count": len(headroom_qids),
        "theoretical_oracle": {
            "recall_at_5": oracle_r5,
            "single_gold_recall_at_5": oracle_single_r5,
            "multi_gold_recall_at_5": oracle_multi_r5,
            "per_fold_recall_at_5": oracle_per_fold,
            "theoretical_max_macro_gain": theoretical_gain,
        },
        "runtime_seconds": elapsed,
    }

    out_file = RESULTS_DIR / "SET_HEADROOM_AUDIT.json"
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"Saved headroom audit to {out_file}", flush=True)
    return payload


if __name__ == "__main__":
    run_headroom_audit()
