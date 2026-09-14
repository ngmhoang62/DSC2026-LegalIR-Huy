"""
Reproduce authoritative baseline metrics and verify 1e-9 exact parity.
Asserts:
- 6,991 evaluable queries
- Recall@5 = 0.9488556715777428 (within 1e-9)
- Precision@5 = 0.20314690316120732 (within 1e-9)
- Single-gold Recall@5 = 0.9641304347826087 (within 1e-9)
- Multi-gold Recall@5 = 0.7703266787658803 (within 1e-9)
- Five exact fold scores (within 1e-9)
- Exact Top-5 (6,991 / 6,991)
- Exact Top-8 (6,991 / 6,991)
- Exact candidate pool membership (6,991 / 6,991)
- Top-8 oracle = 0.9631359366804939
- Candidate pool ceiling = 0.981931531

Writes:
- results/gemini/huy_sparse_resurrection_v1/BASELINE_PARITY.json
- results/gemini/huy_sparse_resurrection_v1/EXECUTION_PROOF.json
- results/gemini/huy_sparse_resurrection_v1/EXECUTION_TRACE.jsonl
"""

import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

# Ensure local imports
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

import common

SCRIPT_PATH = Path(__file__).resolve()
FASTTRACK_PRED_LOCK = common.REPO_ROOT / "results/huy_fasttrack/learner_prediction_locks/profile_memory_plus_sparse_rank_scores.jsonl"


def compute_pool_ceiling(pools: Dict[str, List[str]], golds: Dict[str, List[str]]) -> float:
    scores = []
    for qid, g_list in golds.items():
        g = set(g_list)
        p = set(pools.get(qid, []))
        hit = len(g & p)
        scores.append(hit / len(g))
    return float(np.mean(scores))


def compute_top8_oracle(orders: Dict[str, List[str]], golds: Dict[str, List[str]]) -> float:
    scores = []
    for qid, g_list in golds.items():
        g = set(g_list)
        ord_q = orders.get(qid, [])
        hit = len(g & set(ord_q[:8]))
        scores.append(hit / len(g))
    return float(np.mean(scores))


def run_baseline_parity():
    start_time = time.perf_counter()
    print("=" * 70, flush=True)
    print("REPRODUCING AUTHORITATIVE BASELINE (profile_memory_plus_sparse_rank_scores)", flush=True)
    print("=" * 70, flush=True)

    git_info = common.get_git_info()
    print(f"Dynamic Git HEAD: {git_info['git_commit_sha']}")
    print(f"Git dirty: {git_info['is_dirty']} ({git_info['git_status_porcelain']})")

    folds, fold_for, pools, questions, golds, e5_orders, e5_scores, dup, base_orders, base_scores = common.load_baseline_data()
    base_rows = common.load_cached_feature_rows()

    print(f"Loaded {len(base_orders)} baseline cached predictions.")
    print(f"Loaded {len(base_rows)} baseline feature matrices.")

    # 1. Load authoritative fasttrack prediction lock for exact Top-5 / Top-8 comparison
    assert FASTTRACK_PRED_LOCK.exists(), f"Missing fasttrack prediction lock at {FASTTRACK_PRED_LOCK}"
    lock_orders = {}
    with FASTTRACK_PRED_LOCK.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            lock_orders[str(rec["qid"])] = [str(x) for x in (rec.get("order") or rec.get("ranked_docs") or rec.get("predictions"))]

    print(f"Loaded {len(lock_orders)} authoritative lock orders from fasttrack.")

    # 2. Evaluate cached baseline orders
    cached_metrics = common.evaluate_orders(base_orders, golds, folds)
    print("\n--- Cached Baseline Orders Evaluation ---")
    print(f"Total queries: {cached_metrics['num_queries']}")
    print(f"Recall@5:     {cached_metrics['recall_at_5']:.16f}")
    print(f"Precision@5:  {cached_metrics['precision_at_5']:.16f}")
    print(f"Single-gold:  {cached_metrics['single_gold_recall_at_5']:.16f}")
    print(f"Multi-gold:   {cached_metrics['multi_gold_recall_at_5']:.16f}")
    for f, sc in cached_metrics["per_fold_recall_at_5"].items():
        print(f"  {f}: {sc:.16f}")

    # 3. Headroom & ceilings
    top8_oracle = compute_top8_oracle(base_orders, golds)
    pool_ceiling = compute_pool_ceiling(pools, golds)
    print(f"Top-8 Oracle:          {top8_oracle:.16f}")
    print(f"Candidate Pool Ceiling: {pool_ceiling:.16f}")

    # 4. Exact order parity against fasttrack prediction lock
    match_top5 = 0
    match_top8 = 0
    match_pool = 0
    for qid in base_orders:
        o_base = base_orders[qid]
        o_lock = lock_orders[qid]
        if o_base[:5] == o_lock[:5]:
            match_top5 += 1
        if o_base[:8] == o_lock[:8]:
            match_top8 += 1
        if set(o_base) == set(pools[qid]):
            match_pool += 1

    print(f"Exact Top-5 match against fasttrack lock: {match_top5}/{len(base_orders)}")
    print(f"Exact Top-8 match against fasttrack lock: {match_top8}/{len(base_orders)}")
    print(f"Exact candidate pool match: {match_pool}/{len(base_orders)}")

    # 5. Strict assertions
    tol = 1e-9
    assert cached_metrics["num_queries"] == 6991, f"Expected 6,991 evaluable queries, got {cached_metrics['num_queries']}"
    assert abs(cached_metrics["recall_at_5"] - common.EXPECTED_METRICS["recall_at_5"]) < tol, (
        f"Recall@5 mismatch: {cached_metrics['recall_at_5']} vs {common.EXPECTED_METRICS['recall_at_5']}"
    )
    assert abs(cached_metrics["precision_at_5"] - common.EXPECTED_METRICS["precision_at_5"]) < tol, (
        f"Precision@5 mismatch: {cached_metrics['precision_at_5']} vs {common.EXPECTED_METRICS['precision_at_5']}"
    )
    assert abs(cached_metrics["single_gold_recall_at_5"] - common.EXPECTED_METRICS["single_gold_recall_at_5"]) < tol, (
        f"Single-gold mismatch: {cached_metrics['single_gold_recall_at_5']} vs {common.EXPECTED_METRICS['single_gold_recall_at_5']}"
    )
    assert abs(cached_metrics["multi_gold_recall_at_5"] - common.EXPECTED_METRICS["multi_gold_recall_at_5"]) < tol, (
        f"Multi-gold mismatch: {cached_metrics['multi_gold_recall_at_5']} vs {common.EXPECTED_METRICS['multi_gold_recall_at_5']}"
    )
    for f, expected_sc in common.EXPECTED_METRICS["per_fold_recall_at_5"].items():
        actual_sc = cached_metrics["per_fold_recall_at_5"][f]
        assert abs(actual_sc - expected_sc) < tol, (
            f"Fold {f} mismatch: {actual_sc} vs {expected_sc}"
        )
    assert match_top5 == 6991, f"Top-5 order mismatch: {match_top5}/6991"
    assert match_top8 == 6991, f"Top-8 order mismatch: {match_top8}/6991"
    assert match_pool == 6991, f"Candidate pool mismatch: {match_pool}/6991"
    assert abs(top8_oracle - common.EXPECTED_METRICS["top8_oracle"]) < tol, (
        f"Top-8 oracle mismatch: {top8_oracle} vs {common.EXPECTED_METRICS['top8_oracle']}"
    )
    assert abs(pool_ceiling - common.EXPECTED_METRICS["candidate_pool_ceiling"]) < 1e-6, (
        f"Candidate pool ceiling mismatch: {pool_ceiling} vs {common.EXPECTED_METRICS['candidate_pool_ceiling']}"
    )

    elapsed = time.perf_counter() - start_time
    print(f"\nALL BASELINE PARITY CHECKS PASSED within {tol} in {elapsed:.2f}s!")

    # 6. Write BASELINE_PARITY.json
    parity_manifest = {
        "schema_version": "dsc2026.gemini.huy_sparse_resurrection_v1.baseline_parity.v1",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit_sha": git_info["git_commit_sha"],
        "is_dirty": git_info["is_dirty"],
        "git_status_porcelain": git_info["git_status_porcelain"],
        "script_sha256": common.sha256(SCRIPT_PATH),
        "input_files": {
            "folds_file": str(common.core.FOLDS_PATH),
            "folds_sha256": common.sha256(common.core.FOLDS_PATH),
            "pools_file": str(common.core.POOL_PATH),
            "pools_sha256": common.sha256(common.core.POOL_PATH),
            "boundary_manifest_file": str(common.core.BOUNDARY_MANIFEST),
            "boundary_manifest_sha256": common.sha256(common.core.BOUNDARY_MANIFEST),
            "baseline_predictions_file": str(common.BASELINE_PRED_FILE),
            "baseline_predictions_sha256": common.sha256(common.BASELINE_PRED_FILE),
            "fasttrack_prediction_lock": str(FASTTRACK_PRED_LOCK),
            "fasttrack_prediction_lock_sha256": common.sha256(FASTTRACK_PRED_LOCK),
            "baseline_features_file": str(common.BASELINE_FEAT_FILE),
            "baseline_features_sha256": common.sha256(common.BASELINE_FEAT_FILE),
        },
        "target_endpoint": "profile_memory_plus_sparse_rank_scores",
        "evaluable_queries": cached_metrics["num_queries"],
        "parity_tolerance": tol,
        "parity_verified": True,
        "exact_top5_match_count": match_top5,
        "exact_top8_match_count": match_top8,
        "exact_pool_membership_count": match_pool,
        "metrics": {
            "authoritative_baseline": cached_metrics,
            "top8_oracle": top8_oracle,
            "candidate_pool_ceiling": pool_ceiling,
            "expected_authoritative": common.EXPECTED_METRICS,
        },
        "execution_time_seconds": round(elapsed, 3),
    }

    parity_file = common.RESULTS_DIR / "BASELINE_PARITY.json"
    with parity_file.open("w", encoding="utf-8") as f:
        json.dump(parity_manifest, f, indent=2)
    print(f"Wrote {parity_file}")

    # 7. Update execution trace & proof
    common.log_trace(
        stage="BASELINE_PARITY",
        status="SUCCESS",
        script_path=SCRIPT_PATH,
        input_paths=[common.core.FOLDS_PATH, common.core.POOL_PATH, common.core.BOUNDARY_MANIFEST, common.BASELINE_PRED_FILE, FASTTRACK_PRED_LOCK, common.BASELINE_FEAT_FILE],
        output_path=parity_file,
        records_processed=cached_metrics["num_queries"],
        wall_clock_sec=elapsed,
        extra_info={
            "recall_at_5": cached_metrics["recall_at_5"],
            "parity_verified": True,
            "exact_top5_match": match_top5,
            "exact_top8_match": match_top8,
        }
    )

    common.update_execution_proof(
        stage="BASELINE_PARITY",
        stage_data={
            "status": "COMPLETED",
            "evaluable_queries": cached_metrics["num_queries"],
            "recall_at_5": cached_metrics["recall_at_5"],
            "parity_within_1e9": True,
            "exact_top5_match": match_top5,
            "exact_top8_match": match_top8,
            "exact_pool_membership": match_pool,
            "artifact": str(parity_file.relative_to(common.REPO_ROOT)),
        }
    )


if __name__ == "__main__":
    run_baseline_parity()
