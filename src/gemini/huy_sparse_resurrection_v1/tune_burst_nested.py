"""
Strict Nested Tuning of Huy BURST Fusion Parameters (Section 14).
Grid:
- second_weight in {0.15, 0.30, 0.50}
- local_weight in {0.70, 0.85, 0.90, 0.95}
- rrf_k in {5, 10, 20, 40}

Nested protocol:
For each outer fold F:
- Outer fold F labels are strictly invisible.
- Select best parameters via inner cross-validation over the 4 training folds.
- Evaluate outer fold F once using chosen parameters.

Writes:
- results/gemini/huy_sparse_resurrection_v1/BURST_NESTED_TUNING_REPORT.json
"""

import itertools
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# Ensure local imports
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

import common

SCRIPT_PATH = Path(__file__).resolve()
RETRIEVAL_FILE = common.CACHE_DIR / "BURST_V2_RETRIEVAL_RESULTS.jsonl"

SECOND_WEIGHTS = [0.15, 0.30, 0.50]
LOCAL_WEIGHTS = [0.70, 0.85, 0.90, 0.95]
RRF_KS = [5, 10, 20, 40]


def run_nested_tuning():
    start_time = time.perf_counter()
    print("=" * 70, flush=True)
    print("STRICT NESTED TUNING OF HUY BURST PARAMETERS (48 CONFIGURATIONS)", flush=True)
    print("=" * 70, flush=True)

    git_info = common.get_git_info()
    folds, fold_for, pools, questions, golds, e5_orders, e5_scores, dup, base_orders, base_scores = common.load_baseline_data()
    all_qids = sorted(questions.keys(), key=int)

    # Load top full and local retrieved results for all queries
    print("Loading retrieval cache...", flush=True)
    full_cache = {}
    evidence_cache = {}
    with RETRIEVAL_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                qid = str(rec["qid"])
                full_cache[qid] = {d: r for d, r, s in rec["h_full_top100"]}
                evidence_cache[qid] = rec["pool_evidence"]

    grid = list(itertools.product(SECOND_WEIGHTS, LOCAL_WEIGHTS, RRF_KS))
    print(f"Total parameter combinations: {len(grid)}")

    outer_selections = {}
    nested_oof_orders = {}

    for outer, test_ids in folds.items():
        t_fold = time.perf_counter()
        blocked = set(map(str, dup.get(outer, [])))
        train_ids = sorted(set(all_qids) - set(test_ids) - blocked, key=int)
        inner_folds = [f for f in folds if f != outer]

        print(f"\n--- Outer Fold {outer} (Train: {len(train_ids)}, Test: {len(test_ids)}) ---")
        best_inner_score = -1.0
        best_params = None

        for sw, lw, k in grid:
            gw = 1.0 - lw
            inner_recalls = []

            for inner in inner_folds:
                inner_val_ids = [q for q in train_ids if fold_for[q] == inner]
                val_hits = 0
                val_golds_total = 0

                for qid in inner_val_ids:
                    g = golds[qid]
                    pool = pools[qid]
                    rf_map = full_cache.get(qid, {})
                    ev_map = evidence_cache.get(qid, {})

                    # Recompute local score with sw and fuse with gw, lw, k
                    cand_scores = []
                    for doc in pool:
                        rf = rf_map.get(doc, 100000)
                        ev = ev_map.get(doc, {"best_chunk": 0.0, "second_chunk": 0.0})
                        local_sc = ev["best_chunk"] + sw * ev["second_chunk"]
                        cand_scores.append((doc, rf, local_sc))

                    # Rank local
                    cand_scores.sort(key=lambda x: (-x[2], x[0]))
                    rl_map = {doc: rank for rank, (doc, _, _) in enumerate(cand_scores, 1)}

                    # Fuse
                    fused = []
                    for doc, rf, _ in cand_scores:
                        rl = rl_map[doc]
                        f_score = (gw / (k + rf)) + (lw / (k + rl))
                        fused.append((doc, f_score))

                    fused.sort(key=lambda x: (-x[1], x[0]))
                    top5 = set([d for d, _ in fused[:5]])
                    val_hits += len(g & top5) / len(g)

                inner_recalls.append(val_hits / len(inner_val_ids))

            mean_inner_recall = float(np.mean(inner_recalls))
            if mean_inner_recall > best_inner_score:
                best_inner_score = mean_inner_recall
                best_params = (sw, lw, k)

        print(f"Outer Fold {outer} selected: second_weight={best_params[0]}, local_weight={best_params[1]}, rrf_k={best_params[2]} (Inner R@5={best_inner_score:.6f})")
        outer_selections[outer] = {
            "second_weight": best_params[0],
            "local_weight": best_params[1],
            "rrf_k": best_params[2],
            "inner_score": best_inner_score,
            "time_sec": round(time.perf_counter() - t_fold, 2),
        }

        # Evaluate test_ids with best_params
        sw, lw, k = best_params
        gw = 1.0 - lw
        for qid in test_ids:
            pool = pools[qid]
            rf_map = full_cache.get(qid, {})
            ev_map = evidence_cache.get(qid, {})

            cand_scores = []
            for doc in pool:
                rf = rf_map.get(doc, 100000)
                ev = ev_map.get(doc, {"best_chunk": 0.0, "second_chunk": 0.0})
                local_sc = ev["best_chunk"] + sw * ev["second_chunk"]
                cand_scores.append((doc, rf, local_sc))

            cand_scores.sort(key=lambda x: (-x[2], x[0]))
            rl_map = {doc: rank for rank, (doc, _, _) in enumerate(cand_scores, 1)}

            fused = []
            for doc, rf, _ in cand_scores:
                rl = rl_map[doc]
                f_score = (gw / (k + rf)) + (lw / (k + rl))
                fused.append((doc, f_score))

            fused.sort(key=lambda x: (-x[1], x[0]))
            nested_oof_orders[qid] = [d for d, _ in fused]

    # Evaluate nested OOF metrics
    nested_metrics = common.evaluate_orders(nested_oof_orders, golds, folds)
    print(f"\nNested BURST Standalone Recall@5: {nested_metrics['recall_at_5']:.8f}")

    tuning_report = {
        "schema_version": "dsc2026.gemini.huy_sparse_resurrection_v1.nested_tuning.v1",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit_sha": git_info["git_commit_sha"],
        "parameter_grid": {
            "second_weight": SECOND_WEIGHTS,
            "local_weight": LOCAL_WEIGHTS,
            "rrf_k": RRF_KS,
        },
        "outer_fold_selections": outer_selections,
        "nested_standalone_metrics": nested_metrics,
        "wall_clock_seconds": round(time.perf_counter() - start_time, 2),
    }

    report_file = common.RESULTS_DIR / "BURST_NESTED_TUNING_REPORT.json"
    with report_file.open("w", encoding="utf-8") as f:
        json.dump(tuning_report, f, indent=2)
    print(f"Wrote {report_file}")


if __name__ == "__main__":
    run_nested_tuning()
