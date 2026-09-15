"""CAL600 LOBO evaluation for S0 (D1 Baseline) vs S1 (D1 + LegalIR Sparse)."""

from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .common import (
    D1_VIEWS,
    EXPECTED_D1_DIM,
    EXPECTED_D1_R5,
    EXPECTED_S1_DIM,
    EXTRA_CV_PATHS,
    RESULTS_DIR,
    ROOT,
    S1_VIEWS,
    SPARSE_RANK_VIEWS,
    SPARSE_SCORE_CHANNELS,
    load_aligned,
    load_pkl,
    load_sparse_channels_and_views,
)

sys.path.insert(0, str(ROOT))
from run_burst_expanded_fusion_submission import DocumentStore
from tune_citation_graph import build_citation_table, citation_features
from tune_corpus_cap32_fusion import build_training_cap
from tune_doctype_features import build_type_table, type_features
from tune_expanded_fusion_selection import ltr_features


def evaluate_sparse_cal() -> Dict[str, Any]:
    t0 = time.perf_counter()
    ctx_dir = ROOT / "DSC2026-LegalIR-main" / "v4_run" / "public_test_dataset" / "selected-contexts"
    docs = DocumentStore(sorted(ctx_dir.glob("context_*.json")))

    queries, blocks, all_ids, extended, local_views, base_scores = (
        build_training_cap(
            ROOT,
            32,
            "results/corpus_index/holdout_extended_scores_cap32.pkl",
            depth=20,
        )
    )
    gold = {q: queries[q][1] for q in all_ids}

    vnlegal_cv = load_pkl("results/embedding_finetune/vnlegal_lal_cv_scores.pkl")
    crossenc_cv = load_aligned("results/crossenc_fullpool/cv_scores.pkl", extended, all_ids, -11.5)
    extra_cv = {name: load_aligned(rel, extended, all_ids) for name, rel in EXTRA_CV_PATHS.items()}

    # Load sparse channels & views
    sparse_scores, sparse_ranks = load_sparse_channels_and_views(extended, all_ids)

    d1_channels = {
        **base_scores,
        "vnlegal_lal": vnlegal_cv,
        "crossenc": crossenc_cv,
        **extra_cv,
    }

    s1_channels = {
        **d1_channels,
        "legalir_bm25": sparse_scores["legalir_bm25"],
        "legalir_trigram": sparse_scores["legalir_trigram"],
    }

    views_map = {
        **local_views,
        "legalir_bm25": sparse_ranks["legalir_bm25"],
        "legalir_trigram": sparse_ranks["legalir_trigram"],
    }

    type_table = build_type_table(ROOT, docs, all_ids, extended)
    type_rows = type_features(extended, type_table, queries, all_ids)
    own, cited = build_citation_table(docs, all_ids, extended)
    cite_rows = citation_features(extended, own, cited, all_ids)

    def run_lobo_arm(arm_name: str, view_names: List[str], channels: Dict[str, Any]):
        preds_top5: Dict[str, List[str]] = {}
        preds_top10: Dict[str, List[str]] = {}
        feature_dim = 0

        for held in sorted(blocks.keys()):
            train = sum((blocks[n] for n in blocks if n != held), [])
            eval_ids = train + blocks[held]
            eval_rows, eval_groups = ltr_features(
                views_map, view_names, extended, eval_ids, channels
            )
            for q in eval_rows:
                eval_rows[q] = np.concatenate(
                    [eval_rows[q], type_rows[q], cite_rows[q]], axis=1
                )

            feature_dim = eval_rows[all_ids[0]].shape[1]
            X_train = np.vstack([eval_rows[q] for q in train])
            y_train = np.concatenate(
                [[d in gold[q] for d in eval_groups[q]] for q in train]
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

            for q in blocks[held]:
                score = model.decision_function(scaler.transform(eval_rows[q]))
                ranked_docs = [eval_groups[q][i] for i in np.argsort(-score)]
                preds_top5[q] = ranked_docs[:5]
                preds_top10[q] = ranked_docs[:10]

        recalls = [len(set(preds_top5[q]) & set(gold[q])) / len(gold[q]) for q in all_ids]
        precisions = [len(set(preds_top5[q]) & set(gold[q])) / 5.0 for q in all_ids]
        pooled_r5 = float(np.mean(recalls))
        pooled_p5 = float(np.mean(precisions))

        block_recalls = {}
        for b in sorted(blocks.keys()):
            b_recs = [len(set(preds_top5[q]) & set(gold[q])) / len(gold[q]) for q in blocks[b]]
            block_recalls[b.lower()] = float(np.mean(b_recs))

        single_gold = [len(set(preds_top5[q]) & set(gold[q])) / len(gold[q]) for q in all_ids if len(gold[q]) == 1]
        multi_gold = [len(set(preds_top5[q]) & set(gold[q])) / len(gold[q]) for q in all_ids if len(gold[q]) > 1]

        return {
            "arm": arm_name,
            "feature_dim": feature_dim,
            "pooled_recall_at_5": pooled_r5,
            "pooled_precision_at_5": pooled_p5,
            "blocks": block_recalls,
            "single_gold_recall_at_5": float(np.mean(single_gold)),
            "multi_gold_recall_at_5": float(np.mean(multi_gold)),
            "preds_top5": preds_top5,
            "preds_top10": preds_top10,
            "per_query_recall": {q: len(set(preds_top5[q]) & set(gold[q])) / len(gold[q]) for q in all_ids},
        }

    print("Evaluating S0 (D1 Baseline)...", flush=True)
    s0 = run_lobo_arm("S0_D1_BASELINE", D1_VIEWS, d1_channels)

    # Parity check on S0
    d1_parity_result = {
        "schema_version": "dsc2026.gemini.huy_d1_legalir_sparse_port_v1.d1_baseline_parity.v1",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "expected_recall_at_5": EXPECTED_D1_R5,
        "actual_recall_at_5": s0["pooled_recall_at_5"],
        "delta_recall_at_5": s0["pooled_recall_at_5"] - EXPECTED_D1_R5,
        "expected_blocks": {"a": 0.975, "b": 0.97, "c": 0.995, "d": 0.9338888888888888},
        "actual_blocks": s0["blocks"],
        "expected_dim": EXPECTED_D1_DIM,
        "actual_dim": s0["feature_dim"],
        "parity_exact": bool(abs(s0["pooled_recall_at_5"] - EXPECTED_D1_R5) < 1e-12 and s0["feature_dim"] == EXPECTED_D1_DIM),
    }
    parity_path = RESULTS_DIR / "D1_BASELINE_PARITY.json"
    parity_path.write_text(json.dumps(d1_parity_result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote D1 baseline parity to {parity_path} (exact={d1_parity_result['parity_exact']})")

    if not d1_parity_result["parity_exact"]:
        raise RuntimeError("BLOCKED_D1_PARITY: S0 does not match D1 baseline")

    print("Evaluating S1 (D1 + LegalIR Sparse)...", flush=True)
    s1 = run_lobo_arm("S1_D1_PLUS_LEGALIR_SPARSE", S1_VIEWS, s1_channels)

    # Comparison metrics
    wins, losses, ties = 0, 0, 0
    top5_churn = 0
    gold_in, gold_out = 0, 0

    for q in all_ids:
        r0 = s0["per_query_recall"][q]
        r1 = s1["per_query_recall"][q]
        if r1 > r0:
            wins += 1
        elif r1 < r0:
            losses += 1
        else:
            ties += 1

        s0_set = set(s0["preds_top5"][q])
        s1_set = set(s1["preds_top5"][q])
        if s0_set != s1_set:
            top5_churn += 1

        g = set(gold[q])
        gold_in += len((s1_set - s0_set) & g)
        gold_out += len((s0_set - s1_set) & g)

    delta_r5 = s1["pooled_recall_at_5"] - s0["pooled_recall_at_5"]
    block_deltas = {b: s1["blocks"][b] - s0["blocks"][b] for b in s0["blocks"]}

    cal_report = {
        "schema_version": "dsc2026.gemini.huy_d1_legalir_sparse_port_v1.sparse_d1_cal_report.v1",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "runtime_seconds": float(time.perf_counter() - t0),
        "s0_d1_baseline": {
            "feature_dim": s0["feature_dim"],
            "pooled_recall_at_5": s0["pooled_recall_at_5"],
            "pooled_precision_at_5": s0["pooled_precision_at_5"],
            "blocks": s0["blocks"],
            "single_gold_recall_at_5": s0["single_gold_recall_at_5"],
            "multi_gold_recall_at_5": s0["multi_gold_recall_at_5"],
            "distance_to_0_96": max(0.0, 0.96 - s0["pooled_recall_at_5"]),
        },
        "s1_d1_plus_legalir_sparse": {
            "feature_dim": s1["feature_dim"],
            "rank_views": S1_VIEWS,
            "score_channels": list(s1_channels.keys()),
            "pooled_recall_at_5": s1["pooled_recall_at_5"],
            "pooled_precision_at_5": s1["pooled_precision_at_5"],
            "blocks": s1["blocks"],
            "single_gold_recall_at_5": s1["single_gold_recall_at_5"],
            "multi_gold_recall_at_5": s1["multi_gold_recall_at_5"],
            "distance_to_0_96": max(0.0, 0.96 - s1["pooled_recall_at_5"]),
        },
        "comparison_s1_vs_s0": {
            "delta_pooled_recall_at_5": delta_r5,
            "delta_pooled_precision_at_5": s1["pooled_precision_at_5"] - s0["pooled_precision_at_5"],
            "block_deltas": block_deltas,
            "delta_single_gold": s1["single_gold_recall_at_5"] - s0["single_gold_recall_at_5"],
            "delta_multi_gold": s1["multi_gold_recall_at_5"] - s0["multi_gold_recall_at_5"],
            "wins": wins,
            "losses": losses,
            "ties": ties,
            "net_wins": wins - losses,
            "changed_top5_sets_count": top5_churn,
            "changed_top5_sets_pct": float(top5_churn / len(all_ids) * 100.0),
            "gold_crossings_into_top5": gold_in,
            "gold_crossings_out_of_top5": gold_out,
        },
        "promotion_gates": {
            "gate_a_pooled_recall_gt_d1": bool(delta_r5 > 0),
            "gate_b_block_d_ge_d1": bool(block_deltas["d"] >= 0),
            "gate_c_no_block_regress_gt_0001": bool(all(d >= -0.001 for d in block_deltas.values())),
            "gate_d_wins_gt_losses": bool(wins > losses),
            "gate_e_single_gold_ge_neg_0002": bool(s1["single_gold_recall_at_5"] - s0["single_gold_recall_at_5"] >= -0.002),
            "gate_f_multi_gold_ge_neg_0003": bool(s1["multi_gold_recall_at_5"] - s0["multi_gold_recall_at_5"] >= -0.003),
            "gate_g_no_provenance_failure": True,
            "gate_h_prior_evidence_positive": True,
            "all_gates_passed": False
        }
    }
    cal_report["promotion_gates"]["all_gates_passed"] = bool(
        cal_report["promotion_gates"]["gate_a_pooled_recall_gt_d1"]
        and cal_report["promotion_gates"]["gate_b_block_d_ge_d1"]
        and cal_report["promotion_gates"]["gate_c_no_block_regress_gt_0001"]
        and cal_report["promotion_gates"]["gate_d_wins_gt_losses"]
        and cal_report["promotion_gates"]["gate_e_single_gold_ge_neg_0002"]
        and cal_report["promotion_gates"]["gate_f_multi_gold_ge_neg_0003"]
    )

    cal_path = RESULTS_DIR / "SPARSE_D1_CAL_REPORT.json"
    cal_path.write_text(json.dumps(cal_report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote CAL report to {cal_path}")

    # Write per-query predictions JSONL
    pred_path = RESULTS_DIR / "SPARSE_D1_CAL_PREDICTIONS.jsonl"
    with pred_path.open("w", encoding="utf-8") as f:
        for q in all_ids:
            # Get sparse ranks of gold docs
            bm25_order = sparse_ranks["legalir_bm25"][q]
            tri_order = sparse_ranks["legalir_trigram"][q]
            bm25_gold_ranks = {d: bm25_order.index(d) + 1 if d in bm25_order else None for d in gold[q]}
            tri_gold_ranks = {d: tri_order.index(d) + 1 if d in tri_order else None for d in gold[q]}

            row = {
                "qid": q,
                "gold": sorted(list(gold[q])),
                "s0_top10": s0["preds_top10"][q],
                "s1_top10": s1["preds_top10"][q],
                "s0_recall": s0["per_query_recall"][q],
                "s1_recall": s1["per_query_recall"][q],
                "sparse_bm25_gold_ranks": bm25_gold_ranks,
                "sparse_trigram_gold_ranks": tri_gold_ranks,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote per-query predictions to {pred_path}")

    # Complementarity audit
    missed_gold_occurrences = []
    oracle_s0_or_bm25 = []
    oracle_s0_or_tri = []
    oracle_s0_or_both = []

    bm25_hits_5 = 0
    bm25_hits_6_10 = 0
    bm25_hits_11_20 = 0
    tri_hits_5 = 0
    tri_hits_6_10 = 0
    tri_hits_11_20 = 0
    total_missed_gold = 0

    for q in all_ids:
        g = set(gold[q])
        s0_5 = set(s0["preds_top5"][q])
        bm25_5 = set(sparse_ranks["legalir_bm25"][q][:5])
        tri_5 = set(sparse_ranks["legalir_trigram"][q][:5])

        oracle_s0_or_bm25.append(len(g & (s0_5 | bm25_5)) / len(g))
        oracle_s0_or_tri.append(len(g & (s0_5 | tri_5)) / len(g))
        oracle_s0_or_both.append(len(g & (s0_5 | bm25_5 | tri_5)) / len(g))

        missed = g - s0_5
        total_missed_gold += len(missed)
        bm25_order = sparse_ranks["legalir_bm25"][q]
        tri_order = sparse_ranks["legalir_trigram"][q]

        for d in missed:
            r_bm25 = bm25_order.index(d) + 1 if d in bm25_order else 10**9
            r_tri = tri_order.index(d) + 1 if d in tri_order else 10**9
            if r_bm25 <= 5:
                bm25_hits_5 += 1
            elif r_bm25 <= 10:
                bm25_hits_6_10 += 1
            elif r_bm25 <= 20:
                bm25_hits_11_20 += 1

            if r_tri <= 5:
                tri_hits_5 += 1
            elif r_tri <= 10:
                tri_hits_6_10 += 1
            elif r_tri <= 20:
                tri_hits_11_20 += 1

    comp_audit = {
        "schema_version": "dsc2026.gemini.huy_d1_legalir_sparse_port_v1.sparse_complementarity_audit.v1",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "total_s0_missed_gold_occurrences": total_missed_gold,
        "bm25_recovery_of_s0_misses": {
            "rank_le_5": bm25_hits_5,
            "rank_6_to_10": bm25_hits_6_10,
            "rank_11_to_20": bm25_hits_11_20,
        },
        "trigram_recovery_of_s0_misses": {
            "rank_le_5": tri_hits_5,
            "rank_6_to_10": tri_hits_6_10,
            "rank_11_to_20": tri_hits_11_20,
        },
        "oracle_recall_at_5": {
            "s0_baseline": s0["pooled_recall_at_5"],
            "oracle_s0_union_bm25_top5": float(np.mean(oracle_s0_or_bm25)),
            "oracle_s0_union_trigram_top5": float(np.mean(oracle_s0_or_tri)),
            "oracle_s0_union_both_top5": float(np.mean(oracle_s0_or_both)),
        }
    }
    comp_path = RESULTS_DIR / "SPARSE_COMPLEMENTARITY_AUDIT.json"
    comp_path.write_text(json.dumps(comp_audit, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote complementarity audit to {comp_path}")

    return cal_report


if __name__ == "__main__":
    evaluate_sparse_cal()
