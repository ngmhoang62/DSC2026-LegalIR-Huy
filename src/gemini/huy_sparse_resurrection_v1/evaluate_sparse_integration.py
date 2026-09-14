"""
Strict 5-Fold Integration of Huy Sparse Evidence into Authoritative Baseline (S0-S4 + Ablations).
Evaluates:
- S0: Authoritative baseline 44D
- S1: + H_FULL rank & score
- S2: + H_LOCAL rank & evidence distribution
- S3: + H_BURST rank & supporting scores
- S4: Full resurrection (all 11D sparse evidence features added to 44D)
- S4_NO_LEGALIR_BM25 (ablation if S4 improves)
- S4_NO_TRIGRAM (ablation if S4 improves)

Writes:
- results/gemini/huy_sparse_resurrection_v1/SPARSE_INTEGRATION_REPORT.json
"""

import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Ensure local imports
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

import common

SCRIPT_PATH = Path(__file__).resolve()
RETRIEVAL_FILE = common.CACHE_DIR / "BURST_V2_RETRIEVAL_RESULTS.jsonl"


def extract_11d_features(pool: List[str], evidence_map: Dict[str, Any]) -> np.ndarray:
    """
    Derive compact 11D sparse evidence-distribution block for candidate pool:
    1. H_FULL reciprocal rank: 1.0 / (10.0 + min(r, 1000))
    2. H_FULL normalized/raw score
    3. H_LOCAL reciprocal rank: 1.0 / (10.0 + min(r, 1000))
    4. H_LOCAL normalized/raw score
    5. H_BURST reciprocal rank: 1.0 / (10.0 + min(r, 1000))
    6. best local chunk score
    7. second local chunk score
    8. second / (abs(best) + 1e-4) ratio
    9. best - second margin
    10. log1p(chunk_count)
    11. rank disagreement: abs(h_full_rank - h_local_rank) / 60.0
    """
    n = len(pool)
    feats = np.zeros((n, 11), dtype=np.float32)

    for i, doc in enumerate(pool):
        ev = evidence_map.get(doc, {
            "h_full_rank": 100000, "h_full_score": 0.0,
            "h_local_rank": 100000, "h_local_score": 0.0,
            "h_burst_rank": 100000, "h_burst_score": 0.0,
            "best_chunk": 0.0, "second_chunk": 0.0, "third_chunk": 0.0,
            "chunk_count": 0,
        })
        rf = float(ev["h_full_rank"])
        sf = float(ev["h_full_score"])
        rl = float(ev["h_local_rank"])
        sl = float(ev["h_local_score"])
        rb = float(ev["h_burst_rank"])
        best = float(ev["best_chunk"])
        second = float(ev["second_chunk"])
        cnt = float(ev["chunk_count"])

        feats[i, 0] = 1.0 / (10.0 + min(rf, 1000.0))
        feats[i, 1] = sf
        feats[i, 2] = 1.0 / (10.0 + min(rl, 1000.0))
        feats[i, 3] = sl
        feats[i, 4] = 1.0 / (10.0 + min(rb, 1000.0))
        feats[i, 5] = best
        feats[i, 6] = second
        feats[i, 7] = second / (abs(best) + 1e-4)
        feats[i, 8] = best - second
        feats[i, 9] = math.log1p(cnt)
        feats[i, 10] = min(abs(rf - rl), 1000.0) / 60.0

    return feats


def fit_and_evaluate_strict_5fold(
    feature_matrices: Dict[str, np.ndarray],
    pools: Dict[str, List[str]],
    golds: Dict[str, List[str]],
    folds: Dict[str, List[str]],
    dup: Dict[str, List[str]],
) -> Tuple[Dict[str, List[str]], Dict[str, Dict[str, float]], Dict[str, Any]]:
    all_qids = set()
    for f_qids in folds.values():
        all_qids.update(f_qids)

    oof_orders = {}
    oof_scores = {}

    for outer, test_ids in folds.items():
        blocked = set(map(str, dup.get(outer, [])))
        train_ids = sorted(all_qids - set(test_ids) - blocked, key=int)

        train_x = np.vstack([feature_matrices[qid] for qid in train_ids])
        train_y = np.concatenate([
            np.asarray([doc in golds[qid] for doc in pools[qid]], dtype=np.int8)
            for qid in train_ids
        ])

        scaler = StandardScaler().fit(train_x)
        model = LogisticRegression(
            C=0.15,
            class_weight="balanced",
            solver="liblinear",
            max_iter=3000,
            random_state=2026,
        ).fit(scaler.transform(train_x), train_y)

        for qid in test_ids:
            pool = pools[qid]
            x_test = scaler.transform(feature_matrices[qid])
            values = model.decision_function(x_test)

            # Fasttrack canonical tie-breaking: np.lexsort((np.asarray(pool), -values))
            index = np.lexsort((np.asarray(pool), -values))
            oof_orders[qid] = [pool[i] for i in index]
            oof_scores[qid] = {pool[i]: float(values[i]) for i in range(len(pool))}

    metrics = common.evaluate_orders(oof_orders, golds, folds)
    return oof_orders, oof_scores, metrics


def run_integration():
    start_time = time.perf_counter()
    print("=" * 70, flush=True)
    print("STRICT 5-FOLD INTEGRATION OF HUY SPARSE EVIDENCE (S0 - S4)", flush=True)
    print("=" * 70, flush=True)

    git_info = common.get_git_info()
    folds, fold_for, pools, questions, golds, e5_orders, e5_scores, dup, base_orders, base_scores = common.load_baseline_data()
    all_qids = sorted(questions.keys(), key=int)
    base_rows = common.load_cached_feature_rows()

    # Load retrieval cache
    print("Loading retrieval results from cache...", flush=True)
    sparse_11d = {}
    with RETRIEVAL_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                qid = str(rec["qid"])
                sparse_11d[qid] = extract_11d_features(pools[qid], rec["pool_evidence"])

    print(f"Extracted 11D sparse features for all {len(sparse_11d)} queries.")

    # -------------------------------------------------------------
    # S0: Authoritative Baseline (base_orders / base_scores)
    # -------------------------------------------------------------
    print("\nEvaluating S0: Authoritative Baseline (44D)...", flush=True)
    s0_metrics = common.evaluate_orders(base_orders, golds, folds)
    print(f"S0 Recall@5: {s0_metrics['recall_at_5']:.8f} (Precision@5: {s0_metrics['precision_at_5']:.8f})")

    # -------------------------------------------------------------
    # S1: + H_FULL (cols 0, 1 from 11D)
    # -------------------------------------------------------------
    print("\nEvaluating S1: + H_FULL rank & score (46D)...", flush=True)
    s1_rows = {qid: np.hstack([base_rows[qid], sparse_11d[qid][:, [0, 1]]]) for qid in all_qids}
    s1_orders, s1_scores, s1_metrics = fit_and_evaluate_strict_5fold(s1_rows, pools, golds, folds, dup)
    s1_delta = s1_metrics["recall_at_5"] - s0_metrics["recall_at_5"]
    print(f"S1 Recall@5: {s1_metrics['recall_at_5']:.8f} | Delta: {s1_delta:+.8f}")

    # -------------------------------------------------------------
    # S2: + H_LOCAL (cols 2, 3, 5, 6, 7, 8, 9 from 11D)
    # -------------------------------------------------------------
    print("\nEvaluating S2: + H_LOCAL rank & evidence block (51D)...", flush=True)
    s2_cols = [2, 3, 5, 6, 7, 8, 9]
    s2_rows = {qid: np.hstack([base_rows[qid], sparse_11d[qid][:, s2_cols]]) for qid in all_qids}
    s2_orders, s2_scores, s2_metrics = fit_and_evaluate_strict_5fold(s2_rows, pools, golds, folds, dup)
    s2_delta = s2_metrics["recall_at_5"] - s0_metrics["recall_at_5"]
    print(f"S2 Recall@5: {s2_metrics['recall_at_5']:.8f} | Delta: {s2_delta:+.8f}")

    # -------------------------------------------------------------
    # S3: + H_BURST (cols 4, 1, 3 from 11D)
    # -------------------------------------------------------------
    print("\nEvaluating S3: + H_BURST rank & core scores (47D)...", flush=True)
    s3_cols = [4, 1, 3]
    s3_rows = {qid: np.hstack([base_rows[qid], sparse_11d[qid][:, s3_cols]]) for qid in all_qids}
    s3_orders, s3_scores, s3_metrics = fit_and_evaluate_strict_5fold(s3_rows, pools, golds, folds, dup)
    s3_delta = s3_metrics["recall_at_5"] - s0_metrics["recall_at_5"]
    print(f"S3 Recall@5: {s3_metrics['recall_at_5']:.8f} | Delta: {s3_delta:+.8f}")

    # -------------------------------------------------------------
    # S4: Full Sparse Resurrection (all 11D added to 44D -> 55D)
    # -------------------------------------------------------------
    print("\nEvaluating S4: Full Sparse Resurrection (55D)...", flush=True)
    s4_rows = {qid: np.hstack([base_rows[qid], sparse_11d[qid]]) for qid in all_qids}
    s4_orders, s4_scores, s4_metrics = fit_and_evaluate_strict_5fold(s4_rows, pools, golds, folds, dup)
    s4_delta = s4_metrics["recall_at_5"] - s0_metrics["recall_at_5"]
    print(f"S4 Recall@5: {s4_metrics['recall_at_5']:.8f} | Delta: {s4_delta:+.8f}")

    # Ablations if S4 improves
    ablations = {}
    if s4_delta > 0:
        print("\nS4 improved over baseline! Running key ablations...", flush=True)
        # S4_NO_LEGALIR_BM25: Drop LegalIR BM25 (rank cols 10,11; score cols 24,25 in 44D)
        bm25_drop_indices = [10, 11, 24, 25]
        keep_indices_no_bm25 = [i for i in range(44) if i not in bm25_drop_indices]
        s4_no_bm25_rows = {qid: np.hstack([base_rows[qid][:, keep_indices_no_bm25], sparse_11d[qid]]) for qid in all_qids}
        _, _, m_no_bm25 = fit_and_evaluate_strict_5fold(s4_no_bm25_rows, pools, golds, folds, dup)
        ablations["S4_NO_LEGALIR_BM25"] = {
            "metrics": m_no_bm25,
            "delta_vs_baseline": m_no_bm25["recall_at_5"] - s0_metrics["recall_at_5"],
            "delta_vs_s4": m_no_bm25["recall_at_5"] - s4_metrics["recall_at_5"],
        }
        print(f"S4_NO_LEGALIR_BM25 Recall@5: {m_no_bm25['recall_at_5']:.8f} | Delta vs S4: {m_no_bm25['recall_at_5'] - s4_metrics['recall_at_5']:+.8f}")

        # S4_NO_TRIGRAM: Drop LegalIR trigram (rank cols 12,13; score cols 26,27 in 44D)
        tri_drop_indices = [12, 13, 26, 27]
        keep_indices_no_tri = [i for i in range(44) if i not in tri_drop_indices]
        s4_no_tri_rows = {qid: np.hstack([base_rows[qid][:, keep_indices_no_tri], sparse_11d[qid]]) for qid in all_qids}
        _, _, m_no_tri = fit_and_evaluate_strict_5fold(s4_no_tri_rows, pools, golds, folds, dup)
        ablations["S4_NO_TRIGRAM"] = {
            "metrics": m_no_tri,
            "delta_vs_baseline": m_no_tri["recall_at_5"] - s0_metrics["recall_at_5"],
            "delta_vs_s4": m_no_tri["recall_at_5"] - s4_metrics["recall_at_5"],
        }
        print(f"S4_NO_TRIGRAM Recall@5: {m_no_tri['recall_at_5']:.8f} | Delta vs S4: {m_no_tri['recall_at_5'] - s4_metrics['recall_at_5']:+.8f}")
    else:
        print("\nS4 did not beat baseline; skipping ablations.", flush=True)

    # Pick best arm
    arms = {
        "S0": (s0_metrics, 0.0, base_orders, base_scores),
        "S1": (s1_metrics, s1_delta, s1_orders, s1_scores),
        "S2": (s2_metrics, s2_delta, s2_orders, s2_scores),
        "S3": (s3_metrics, s3_delta, s3_orders, s3_scores),
        "S4": (s4_metrics, s4_delta, s4_orders, s4_scores),
    }
    best_arm_name = max(arms.keys(), key=lambda k: arms[k][1])
    best_metrics, best_delta, best_orders, best_scores = arms[best_arm_name]

    print(f"\nBest Integration Arm: {best_arm_name} with Delta: {best_delta:+.8f}")

    # Compare best predictions vs baseline
    wins = losses = ties = 0
    for qid in all_qids:
        g = golds[qid]
        r_base = len(g & set(base_orders[qid][:5])) / len(g)
        r_cand = len(g & set(best_orders[qid][:5])) / len(g)
        if r_cand > r_base:
            wins += 1
        elif r_cand < r_base:
            losses += 1
        else:
            ties += 1

    print(f"Wins: {wins}, Losses: {losses}, Ties: {ties} (Net: {wins - losses:+d})")

    # Write report
    report = {
        "schema_version": "dsc2026.gemini.huy_sparse_resurrection_v1.sparse_integration.v1",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit_sha": git_info["git_commit_sha"],
        "baseline_recall_at_5": s0_metrics["recall_at_5"],
        "arms": {
            "S0": {"features": 44, "metrics": s0_metrics, "delta": 0.0},
            "S1": {"features": 46, "metrics": s1_metrics, "delta": s1_delta},
            "S2": {"features": 51, "metrics": s2_metrics, "delta": s2_delta},
            "S3": {"features": 47, "metrics": s3_metrics, "delta": s3_delta},
            "S4": {"features": 55, "metrics": s4_metrics, "delta": s4_delta},
        },
        "ablations": ablations,
        "best_arm": {
            "name": best_arm_name,
            "delta": best_delta,
            "metrics": best_metrics,
            "wins": wins,
            "losses": losses,
            "ties": ties,
            "net_wins": wins - losses,
        },
        "wall_clock_seconds": round(time.perf_counter() - start_time, 2),
    }

    report_file = common.RESULTS_DIR / "SPARSE_INTEGRATION_REPORT.json"
    with report_file.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Wrote {report_file}")

    # Save best orders and scores for generalization audit and casebook
    best_pred_file = common.CACHE_DIR / "BEST_SPARSE_INTEGRATION_PREDICTIONS.jsonl"
    with best_pred_file.open("w", encoding="utf-8") as f:
        for qid in all_qids:
            f.write(json.dumps({
                "qid": qid,
                "order": best_orders[qid],
                "scores": best_scores[qid],
            }, ensure_ascii=False) + "\n")
    print(f"Wrote {best_pred_file}")

    # Trace & proof
    common.log_trace(
        stage="SPARSE_INTEGRATION_EVALUATION",
        status="SUCCESS",
        script_path=SCRIPT_PATH,
        input_paths=[RETRIEVAL_FILE, common.BASELINE_FEAT_FILE],
        output_path=report_file,
        records_processed=len(all_qids),
        wall_clock_sec=time.perf_counter() - start_time,
        extra_info={
            "best_arm": best_arm_name,
            "best_delta": best_delta,
            "wins": wins,
            "losses": losses,
        }
    )

    common.update_execution_proof(
        stage="SPARSE_INTEGRATION",
        stage_data={
            "status": "COMPLETED",
            "best_arm": best_arm_name,
            "best_recall_at_5": best_metrics["recall_at_5"],
            "delta_recall_at_5": best_delta,
            "wins": wins,
            "losses": losses,
            "artifact": str(report_file.relative_to(common.REPO_ROOT)),
        }
    )


if __name__ == "__main__":
    run_integration()
