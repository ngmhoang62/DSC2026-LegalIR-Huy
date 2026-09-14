"""
Candidate Pool Expansion Evaluation (Section 16 - Conditional Arm).
Arms:
- P0: Existing locked pool (51 candidates)
- P1: Existing pool union novel H_BURST@10
- P2: Existing pool union novel H_BURST@20

Condition:
Evaluated if SPARSE_HEADROOM_AUDIT reveals meaningful missing gold rescue capacity outside current pool.
Writes:
- results/gemini/huy_sparse_resurrection_v1/CANDIDATE_EXPANSION_REPORT.json
"""

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import numpy as np

# Ensure local imports
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

import common

SCRIPT_PATH = Path(__file__).resolve()
HEADROOM_FILE = common.RESULTS_DIR / "SPARSE_HEADROOM_AUDIT.json"
RETRIEVAL_FILE = common.CACHE_DIR / "BURST_V2_RETRIEVAL_RESULTS.jsonl"


def run_candidate_expansion_audit():
    start_time = time.perf_counter()
    print("=" * 70, flush=True)
    print("CANDIDATE POOL EXPANSION AUDIT (P0 vs P1 vs P2)", flush=True)
    print("=" * 70, flush=True)

    git_info = common.get_git_info()
    folds, fold_for, pools, questions, golds, e5_orders, e5_scores, dup, base_orders, base_scores = common.load_baseline_data()
    all_qids = sorted(questions.keys(), key=int)

    assert HEADROOM_FILE.exists(), f"Missing headroom file: {HEADROOM_FILE}"
    with HEADROOM_FILE.open("r", encoding="utf-8") as f:
        headroom = json.load(f)

    # Check headroom threshold
    pool_comp = headroom["pool_complementarity"]
    total_missing = pool_comp["total_missing_gold_occurrences_outside_pool"]
    rescued_20 = pool_comp["rescued_at_20"]["h_burst"]
    print(f"Total missing gold occurrences outside pool: {total_missing}")
    print(f"H_BURST@20 rescued occurrences: {rescued_20}")

    # Compute candidate pool ceiling under P0, P1, P2
    p0_hits = 0
    p1_hits = 0
    p2_hits = 0
    total_gold_instances = 0

    novel_docs_p1 = []
    novel_docs_p2 = []

    retrieval_cache = {}
    with RETRIEVAL_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                retrieval_cache[str(rec["qid"])] = rec

    for qid in all_qids:
        g = golds[qid]
        p = set(pools[qid])
        total_gold_instances += len(g)

        rec = retrieval_cache[qid]
        burst_10 = set(rec["h_burst_top500_ids"][:10])
        burst_20 = set(rec["h_burst_top500_ids"][:20])

        p1_pool = p | burst_10
        p2_pool = p | burst_20

        novel_docs_p1.append(len(p1_pool - p))
        novel_docs_p2.append(len(p2_pool - p))

        p0_hits += len(g & p)
        p1_hits += len(g & p1_pool)
        p2_hits += len(g & p2_pool)

    ceiling_p0 = p0_hits / total_gold_instances
    ceiling_p1 = p1_hits / total_gold_instances
    ceiling_p2 = p2_hits / total_gold_instances

    print(f"P0 Candidate Ceiling: {ceiling_p0:.6f}")
    print(f"P1 Candidate Ceiling: {ceiling_p1:.6f} (+{ceiling_p1 - ceiling_p0:+.6f}, Mean novel docs: {np.mean(novel_docs_p1):.1f})")
    print(f"P2 Candidate Ceiling: {ceiling_p2:.6f} (+{ceiling_p2 - ceiling_p0:+.6f}, Mean novel docs: {np.mean(novel_docs_p2):.1f})")

    expansion_report = {
        "schema_version": "dsc2026.gemini.huy_sparse_resurrection_v1.candidate_expansion.v1",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit_sha": git_info["git_commit_sha"],
        "pool_ceilings": {
            "P0_locked_pool": {"ceiling": ceiling_p0, "mean_novel_candidates": 0.0},
            "P1_union_burst_top10": {"ceiling": ceiling_p1, "delta_ceiling": ceiling_p1 - ceiling_p0, "mean_novel_candidates": float(np.mean(novel_docs_p1))},
            "P2_union_burst_top20": {"ceiling": ceiling_p2, "delta_ceiling": ceiling_p2 - ceiling_p0, "mean_novel_candidates": float(np.mean(novel_docs_p2))},
        },
        "findings": {
            "headroom_justification": rescued_20 > 0,
            "novel_candidates_per_query_p1": float(np.mean(novel_docs_p1)),
            "novel_candidates_per_query_p2": float(np.mean(novel_docs_p2)),
            "ceiling_gain_p1": ceiling_p1 - ceiling_p0,
            "ceiling_gain_p2": ceiling_p2 - ceiling_p0,
        },
        "wall_clock_seconds": round(time.perf_counter() - start_time, 2),
    }

    report_file = common.RESULTS_DIR / "CANDIDATE_EXPANSION_REPORT.json"
    with report_file.open("w", encoding="utf-8") as f:
        json.dump(expansion_report, f, indent=2)
    print(f"Wrote {report_file}")


if __name__ == "__main__":
    run_candidate_expansion_audit()
