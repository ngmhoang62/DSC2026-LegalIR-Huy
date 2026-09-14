"""
Compute sparse complementarity, standalone retrieval curves, and headroom audits.
Generates:
1. results/gemini/huy_sparse_resurrection_v1/SPARSE_COMPLEMENTARITY_REPORT.json
2. results/gemini/huy_sparse_resurrection_v1/SPARSE_RETRIEVAL_CURVES.json
3. results/gemini/huy_sparse_resurrection_v1/SPARSE_HEADROOM_AUDIT.json
"""

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import numpy as np
from scipy.stats import spearmanr

# Ensure local imports
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

import common

SCRIPT_PATH = Path(__file__).resolve()
RETRIEVAL_FILE = common.CACHE_DIR / "BURST_V2_RETRIEVAL_RESULTS.jsonl"


def load_retrieval_cache() -> Dict[str, Dict[str, Any]]:
    assert RETRIEVAL_FILE.exists(), f"Missing retrieval results: {RETRIEVAL_FILE}"
    cache = {}
    with RETRIEVAL_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                cache[str(rec["qid"])] = rec
    return cache


def compute_jaccard(list_a: List[str], list_b: List[str], k: int) -> float:
    set_a = set(list_a[:k])
    set_b = set(list_b[:k])
    union = set_a | set_b
    if not union:
        return 1.0
    return len(set_a & set_b) / len(union)


def run_diagnostics():
    start_time = time.perf_counter()
    print("=" * 70, flush=True)
    print("COMPUTING SPARSE COMPLEMENTARITY, STANDALONE CURVES & HEADROOM AUDIT", flush=True)
    print("=" * 70, flush=True)

    git_info = common.get_git_info()
    folds, fold_for, pools, questions, golds, e5_orders, e5_scores, dup, base_orders, base_scores = common.load_baseline_data()
    all_qids = sorted(questions.keys(), key=int)

    retrieval_cache = load_retrieval_cache()
    assert len(retrieval_cache) == len(all_qids), f"Expected {len(all_qids)} cached queries, got {len(retrieval_cache)}"
    print(f"Loaded {len(retrieval_cache)} queries from retrieval cache.")

    # Load LegalIR BM25 & trigram rankings
    # In fasttrack baseline, base_rows has:
    # col 10: legalir_bm25 reciprocal rank, col 11: legalir_bm25 raw rank / 60
    # col 12: legalir_trigram reciprocal rank, col 13: legalir_trigram raw rank / 60
    base_rows = common.load_cached_feature_rows()

    l_bm25_orders = {}
    l_tri_orders = {}
    h_full_orders = {}
    h_local_orders = {}
    h_burst_orders = {}

    for qid in all_qids:
        pool = pools[qid]
        rec = retrieval_cache[qid]
        row = base_rows[qid]

        # LegalIR BM25 and Trigram pool ordering
        bm25_ranks = np.round(row[:, 11] * 60.0)
        tri_ranks = np.round(row[:, 13] * 60.0)

        bm25_sorted_idx = np.lexsort((np.asarray(pool), bm25_ranks))
        tri_sorted_idx = np.lexsort((np.asarray(pool), tri_ranks))

        l_bm25_orders[qid] = [pool[i] for i in bm25_sorted_idx]
        l_tri_orders[qid] = [pool[i] for i in tri_sorted_idx]

        h_full_orders[qid] = [d for d, _, _ in rec["h_full_top100"]]
        h_local_orders[qid] = [d for d, _, _ in rec["h_local_top100"]]
        h_burst_orders[qid] = [d for d, _, _ in rec["h_burst_top100"]]

    # -------------------------------------------------------------
    # SECTION 8: SPARSE COMPLEMENTARITY REPORT
    # -------------------------------------------------------------
    print("\n1. Computing Sparse Complementarity Metrics...", flush=True)
    jaccard_k_values = [5, 10, 20]
    jaccards = {k: {"full_vs_lbm25": [], "local_vs_lbm25": [], "burst_vs_lbm25": [], "burst_vs_tri": []} for k in jaccard_k_values}
    spearman_corrs = {"full_vs_lbm25": [], "local_vs_lbm25": [], "burst_vs_lbm25": [], "burst_vs_tri": []}

    unique_gold_ranks = {
        "l_bm25": {5: 0, 8: 0, 10: 0, 20: 0},
        "l_tri": {5: 0, 8: 0, 10: 0, 20: 0},
        "h_full": {5: 0, 8: 0, 10: 0, 20: 0},
        "h_local": {5: 0, 8: 0, 10: 0, 20: 0},
        "h_burst": {5: 0, 8: 0, 10: 0, 20: 0},
    }

    for qid in all_qids:
        p_bm25 = l_bm25_orders[qid]
        p_tri = l_tri_orders[qid]
        p_full = h_full_orders[qid]
        p_local = h_local_orders[qid]
        p_burst = h_burst_orders[qid]

        for k in jaccard_k_values:
            jaccards[k]["full_vs_lbm25"].append(compute_jaccard(p_full, p_bm25, k))
            jaccards[k]["local_vs_lbm25"].append(compute_jaccard(p_local, p_bm25, k))
            jaccards[k]["burst_vs_lbm25"].append(compute_jaccard(p_burst, p_bm25, k))
            jaccards[k]["burst_vs_tri"].append(compute_jaccard(p_burst, p_tri, k))

        # Rank correlation over candidate pool
        pool = pools[qid]
        rec = retrieval_cache[qid]
        ev = rec["pool_evidence"]
        
        ranks_lbm25 = [np.round(base_rows[qid][i, 11] * 60.0) for i, d in enumerate(pool)]
        ranks_ltri = [np.round(base_rows[qid][i, 13] * 60.0) for i, d in enumerate(pool)]
        ranks_hfull = [ev[d]["h_full_rank"] for d in pool]
        ranks_hlocal = [ev[d]["h_local_rank"] for d in pool]
        ranks_hburst = [ev[d]["h_burst_rank"] for d in pool]

        if len(set(ranks_hfull)) > 1 and len(set(ranks_lbm25)) > 1:
            spearman_corrs["full_vs_lbm25"].append(spearmanr(ranks_hfull, ranks_lbm25)[0])
        if len(set(ranks_hlocal)) > 1 and len(set(ranks_lbm25)) > 1:
            spearman_corrs["local_vs_lbm25"].append(spearmanr(ranks_hlocal, ranks_lbm25)[0])
        if len(set(ranks_hburst)) > 1 and len(set(ranks_lbm25)) > 1:
            spearman_corrs["burst_vs_lbm25"].append(spearmanr(ranks_hburst, ranks_lbm25)[0])
        if len(set(ranks_hburst)) > 1 and len(set(ranks_ltri)) > 1:
            spearman_corrs["burst_vs_tri"].append(spearmanr(ranks_hburst, ranks_ltri)[0])

        # Unique gold ranking counts
        q_golds = golds[qid]
        for depth in [5, 8, 10, 20]:
            hits = {
                "l_bm25": q_golds & set(p_bm25[:depth]),
                "l_tri": q_golds & set(p_tri[:depth]),
                "h_full": q_golds & set(p_full[:depth]),
                "h_local": q_golds & set(p_local[:depth]),
                "h_burst": q_golds & set(p_burst[:depth]),
            }
            all_others = {
                "l_bm25": hits["l_tri"] | hits["h_full"] | hits["h_local"] | hits["h_burst"],
                "l_tri": hits["l_bm25"] | hits["h_full"] | hits["h_local"] | hits["h_burst"],
                "h_full": hits["l_bm25"] | hits["l_tri"] | hits["h_local"] | hits["h_burst"],
                "h_local": hits["l_bm25"] | hits["l_tri"] | hits["h_full"] | hits["h_burst"],
                "h_burst": hits["l_bm25"] | hits["l_tri"] | hits["h_full"] | hits["h_local"],
            }
            for expert, h_set in hits.items():
                uniques = h_set - all_others[expert]
                if uniques:
                    unique_gold_ranks[expert][depth] += 1

    comp_report = {
        "schema_version": "dsc2026.gemini.huy_sparse_resurrection_v1.sparse_complementarity.v1",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit_sha": git_info["git_commit_sha"],
        "total_queries": len(all_qids),
        "mean_jaccard": {
            f"top_{k}": {pair: float(np.mean(vals)) for pair, vals in jaccards[k].items()}
            for k in jaccard_k_values
        },
        "mean_spearman_correlation": {
            pair: float(np.nanmean(vals)) for pair, vals in spearman_corrs.items()
        },
        "unique_gold_queries": unique_gold_ranks,
    }

    comp_file = common.RESULTS_DIR / "SPARSE_COMPLEMENTARITY_REPORT.json"
    with comp_file.open("w", encoding="utf-8") as f:
        json.dump(comp_report, f, indent=2)
    print(f"Wrote {comp_file}")

    # -------------------------------------------------------------
    # SECTION 9: STANDALONE RETRIEVAL CURVES
    # -------------------------------------------------------------
    print("\n2. Computing Standalone Retrieval Curves for 5 Sparse Experts...", flush=True)
    curves = {
        "L_BM25": common.evaluate_orders(l_bm25_orders, golds, folds),
        "L_TRI": common.evaluate_orders(l_tri_orders, golds, folds),
        "H_FULL": common.evaluate_orders(h_full_orders, golds, folds),
        "H_LOCAL": common.evaluate_orders(h_local_orders, golds, folds),
        "H_BURST": common.evaluate_orders(h_burst_orders, golds, folds),
    }

    for name, m in curves.items():
        print(f"[{name:7s}] R@1={m['recall_at_1']:.4f}, R@5={m['recall_at_5']:.4f}, R@8={m['recall_at_8']:.4f}, R@10={m['recall_at_10']:.4f}, R@20={m['recall_at_20']:.4f}, R@50={m['recall_at_50']:.4f}, R@100={m['recall_at_100']:.4f} | Single={m['single_gold_recall_at_5']:.4f}, Multi={m['multi_gold_recall_at_5']:.4f}")

    curves_manifest = {
        "schema_version": "dsc2026.gemini.huy_sparse_resurrection_v1.sparse_retrieval_curves.v1",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit_sha": git_info["git_commit_sha"],
        "total_queries": len(all_qids),
        "curves": curves,
    }
    curves_file = common.RESULTS_DIR / "SPARSE_RETRIEVAL_CURVES.json"
    with curves_file.open("w", encoding="utf-8") as f:
        json.dump(curves_manifest, f, indent=2)
    print(f"Wrote {curves_file}")

    # -------------------------------------------------------------
    # SECTION 10: ORACLE COMPLEMENTARITY & HEADROOM AUDIT
    # -------------------------------------------------------------
    print("\n3. Computing Oracle Complementarity & Headroom against 0.948855 Endpoint...", flush=True)
    top5_union_oracles = {"h_full": [], "h_local": [], "h_burst": []}
    boundary_rescue_6_20 = {"h_full": 0, "h_local": 0, "h_burst": 0}
    top8_errors_rescued = {"h_full": 0, "h_local": 0, "h_burst": 0}
    
    # Missing golds outside current pool
    pool_missing_golds = 0
    novel_rescue_10 = {"h_full": 0, "h_local": 0, "h_burst": 0}
    novel_rescue_20 = {"h_full": 0, "h_local": 0, "h_burst": 0}
    novel_rescue_50 = {"h_full": 0, "h_local": 0, "h_burst": 0}

    for qid in all_qids:
        q_golds = golds[qid]
        b_top5 = set(base_orders[qid][:5])
        b_top8 = set(base_orders[qid][:8])
        pool_set = set(pools[qid])

        missing_top5_golds = q_golds - b_top5
        missing_top8_golds = q_golds - b_top8
        missing_pool_golds = q_golds - pool_set
        if missing_pool_golds:
            pool_missing_golds += len(missing_pool_golds)

        rec = retrieval_cache[qid]
        top5_full = set(rec["h_full_top500_ids"][:5])
        top5_local = set(rec["h_local_top500_ids"][:5])
        top5_burst = set(rec["h_burst_top500_ids"][:5])

        top5_union_oracles["h_full"].append(len(q_golds & (b_top5 | top5_full)) / len(q_golds))
        top5_union_oracles["h_local"].append(len(q_golds & (b_top5 | top5_local)) / len(q_golds))
        top5_union_oracles["h_burst"].append(len(q_golds & (b_top5 | top5_burst)) / len(q_golds))

        # Boundary oracle: ranks 6-20 rescue missing Top-5 golds
        if missing_top5_golds:
            if missing_top5_golds & set(rec["h_full_top500_ids"][5:20]):
                boundary_rescue_6_20["h_full"] += 1
            if missing_top5_golds & set(rec["h_local_top500_ids"][5:20]):
                boundary_rescue_6_20["h_local"] += 1
            if missing_top5_golds & set(rec["h_burst_top500_ids"][5:20]):
                boundary_rescue_6_20["h_burst"] += 1

        # Top-8 errors rescued
        if missing_top8_golds:
            if missing_top8_golds & top5_full:
                top8_errors_rescued["h_full"] += 1
            if missing_top8_golds & top5_local:
                top8_errors_rescued["h_local"] += 1
            if missing_top8_golds & top5_burst:
                top8_errors_rescued["h_burst"] += 1

        # Pool complementarity (missing golds outside locked pool)
        if missing_pool_golds:
            for depth, target_dict in [(10, novel_rescue_10), (20, novel_rescue_20), (50, novel_rescue_50)]:
                if missing_pool_golds & set(rec["h_full_top500_ids"][:depth]):
                    target_dict["h_full"] += len(missing_pool_golds & set(rec["h_full_top500_ids"][:depth]))
                if missing_pool_golds & set(rec["h_local_top500_ids"][:depth]):
                    target_dict["h_local"] += len(missing_pool_golds & set(rec["h_local_top500_ids"][:depth]))
                if missing_pool_golds & set(rec["h_burst_top500_ids"][:depth]):
                    target_dict["h_burst"] += len(missing_pool_golds & set(rec["h_burst_top500_ids"][:depth]))

    headroom_report = {
        "schema_version": "dsc2026.gemini.huy_sparse_resurrection_v1.sparse_headroom.v1",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit_sha": git_info["git_commit_sha"],
        "baseline_recall_at_5": common.EXPECTED_METRICS["recall_at_5"],
        "top5_union_oracle": {
            "h_full": float(np.mean(top5_union_oracles["h_full"])),
            "h_local": float(np.mean(top5_union_oracles["h_local"])),
            "h_burst": float(np.mean(top5_union_oracles["h_burst"])),
        },
        "boundary_oracle_rescue_queries": boundary_rescue_6_20,
        "top8_errors_rescued_queries": top8_errors_rescued,
        "pool_complementarity": {
            "total_missing_gold_occurrences_outside_pool": pool_missing_golds,
            "rescued_at_10": novel_rescue_10,
            "rescued_at_20": novel_rescue_20,
            "rescued_at_50": novel_rescue_50,
        },
    }

    headroom_file = common.RESULTS_DIR / "SPARSE_HEADROOM_AUDIT.json"
    with headroom_file.open("w", encoding="utf-8") as f:
        json.dump(headroom_report, f, indent=2)
    print(f"Wrote {headroom_file}")

    elapsed = time.perf_counter() - start_time
    print(f"\nAll diagnostics completed in {elapsed:.2f}s!")

    # Trace & proof
    common.log_trace(
        stage="SPARSE_DIAGNOSTICS_AND_HEADROOM",
        status="SUCCESS",
        script_path=SCRIPT_PATH,
        input_paths=[RETRIEVAL_FILE, common.BASELINE_FEAT_FILE],
        output_path=headroom_file,
        records_processed=len(all_qids),
        wall_clock_sec=elapsed,
        extra_info={
            "top5_union_burst": headroom_report["top5_union_oracle"]["h_burst"],
            "novel_rescue_burst_20": novel_rescue_20["h_burst"],
        }
    )

    common.update_execution_proof(
        stage="SPARSE_DIAGNOSTICS_AND_HEADROOM",
        stage_data={
            "status": "COMPLETED",
            "top5_union_burst": headroom_report["top5_union_oracle"]["h_burst"],
            "h_burst_standalone_r5": curves["H_BURST"]["recall_at_5"],
            "novel_rescue_burst_20": novel_rescue_20["h_burst"],
            "artifacts": [
                str(comp_file.relative_to(common.REPO_ROOT)),
                str(curves_file.relative_to(common.REPO_ROOT)),
                str(headroom_file.relative_to(common.REPO_ROOT)),
            ],
        }
    )


if __name__ == "__main__":
    run_diagnostics()
