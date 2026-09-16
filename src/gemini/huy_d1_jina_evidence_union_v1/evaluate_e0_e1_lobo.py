"""Evaluate E0 (D1 Baseline 48D) vs E1 (D1 + Evidence Union 48D) under CAL600 LOBO."""

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

ROOT = Path("D:/Study/DSC2026/sota")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tune_expanded_fusion_selection import ltr_features
from src.gemini.huy_d1_jina_evidence_union_v1.common import (
    D1_VIEWS,
    EVIDENCE_UNION_CACHE_PKL,
    OLD_JINA_CACHE_PKL,
    RES_DIR,
    SECTION_CE_CACHE_PKL,
    load_cal_data,
    seed_everything,
)


def run_lobo_pipeline(
    arm_name: str,
    channels: Dict[str, Any],
    local_views: Dict[str, Any],
    extended: Dict[str, List[str]],
    type_rows: Dict[str, Any],
    cite_rows: Dict[str, Any],
    blocks: Dict[str, List[str]],
    all_ids: List[str],
    gold: Dict[str, Set[str]],
) -> Tuple[
    Dict[str, List[str]],
    int,
    Dict[str, Any],
    Dict[str, np.ndarray],
    Dict[str, List[str]],
    Dict[str, Dict[str, float]],
]:
    preds: Dict[str, List[str]] = {}
    scores_dict: Dict[str, np.ndarray] = {}
    full_rankings: Dict[str, List[str]] = {}
    full_scores: Dict[str, Dict[str, float]] = {}
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
            full_rankings[q] = [eval_groups[q][i] for i in order]
            full_scores[q] = {
                eval_groups[q][i]: float(dec_scores[i]) for i in range(len(dec_scores))
            }
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
        "per_query_recalls": {
            q: len(set(preds[q]) & gold[q]) / max(1, len(gold[q])) for q in all_ids
        },
    }
    return preds, feature_dim, metrics, scores_dict, full_rankings, full_scores


def run_paired_bootstrap(
    per_query_deltas: Dict[str, float],
    blocks: Dict[str, List[str]],
    all_ids: List[str],
    n_samples: int = 10000,
    seed: int = 2026,
) -> dict:
    rng = np.random.default_rng(seed)
    delta_arr = np.array([per_query_deltas[q] for q in all_ids])
    n_queries = len(all_ids)

    # 1. Ordinary query bootstrap
    ordinary_means = np.empty(n_samples, dtype=np.float64)
    for i in range(n_samples):
        indices = rng.integers(0, n_queries, size=n_queries)
        ordinary_means[i] = np.mean(delta_arr[indices])

    # 2. Block-stratified bootstrap
    stratified_means = np.empty(n_samples, dtype=np.float64)
    block_names = sorted(blocks.keys())
    block_deltas = {b: np.array([per_query_deltas[q] for q in blocks[b]]) for b in block_names}
    block_sizes = {b: len(blocks[b]) for b in block_names}

    for i in range(n_samples):
        sample_sum = 0.0
        for b in block_names:
            b_arr = block_deltas[b]
            b_n = block_sizes[b]
            b_indices = rng.integers(0, b_n, size=b_n)
            sample_sum += np.sum(b_arr[b_indices])
        stratified_means[i] = sample_sum / n_queries

    ordinary_stats = {
        "mean_delta": float(np.mean(ordinary_means)),
        "median_delta": float(np.median(ordinary_means)),
        "ci_2_5": float(np.percentile(ordinary_means, 2.5)),
        "ci_97_5": float(np.percentile(ordinary_means, 97.5)),
        "p_delta_gt_zero": float(np.mean(ordinary_means > 0)),
        "p_delta_ge_zero": float(np.mean(ordinary_means >= 0)),
    }

    stratified_stats = {
        "mean_delta": float(np.mean(stratified_means)),
        "median_delta": float(np.median(stratified_means)),
        "ci_2_5": float(np.percentile(stratified_means, 2.5)),
        "ci_97_5": float(np.percentile(stratified_means, 97.5)),
        "p_delta_gt_zero": float(np.mean(stratified_means > 0)),
        "p_delta_ge_zero": float(np.mean(stratified_means >= 0)),
    }

    return {
        "n_samples": n_samples,
        "seed": seed,
        "ordinary_bootstrap": ordinary_stats,
        "block_stratified_bootstrap": stratified_stats,
    }


def evaluate_e0_e1_lobo() -> Tuple[dict, dict, dict, dict, dict, dict, dict, dict, dict]:
    seed_everything(2026)
    print("=== EVALUATING E0 (BASELINE 48D) VS E1 (EVIDENCE UNION 48D) UNDER LOBO ===", flush=True)

    # 1. Load CAL dataset
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

    # 2. Load caches
    assert OLD_JINA_CACHE_PKL.exists(), f"Old cache missing: {OLD_JINA_CACHE_PKL}"
    assert SECTION_CE_CACHE_PKL.exists(), f"Section CE cache missing: {SECTION_CE_CACHE_PKL}"
    assert EVIDENCE_UNION_CACHE_PKL.exists(), f"Union cache missing: {EVIDENCE_UNION_CACHE_PKL}"

    raw_old = pickle.loads(OLD_JINA_CACHE_PKL.read_bytes())
    old_scores = raw_old.get("scores", raw_old) if isinstance(raw_old, dict) else raw_old

    raw_sec = pickle.loads(SECTION_CE_CACHE_PKL.read_bytes())
    section_scores = raw_sec.get("scores", raw_sec) if isinstance(raw_sec, dict) else raw_sec

    raw_union = pickle.loads(EVIDENCE_UNION_CACHE_PKL.read_bytes())
    union_scores = raw_union.get("scores", raw_union) if isinstance(raw_union, dict) else raw_union

    def align_channel(raw_map, floor=None):
        fl = floor if floor is not None else min(v for q in raw_map for v in raw_map[q].values())
        return {q: {d: raw_map.get(q, {}).get(d, fl) for d in extended[q]} for q in all_ids}

    aligned_union = align_channel(union_scores)
    aligned_old = align_channel(old_scores)
    aligned_sec = align_channel(section_scores)

    # E0: exact 48D channels with historical jina_ft
    channels_e0 = full_channels_cv

    # E1: exact 48D channels where jina_ft is REPLACED by jina_ft_evidence_union
    channels_e1 = {
        k: (aligned_union if k == "jina_ft" else v)
        for k, v in full_channels_cv.items()
    }

    # Verify strictly 48D feature structure
    print("Running E0 baseline evaluation...", flush=True)
    preds_e0, dim_e0, metrics_e0, scores_e0, full_rankings_e0, full_scores_e0 = run_lobo_pipeline(
        "E0_D1_CURRENT",
        channels_e0,
        local_views,
        extended,
        type_rows,
        cite_rows,
        blocks,
        all_ids,
        gold,
    )
    assert dim_e0 == 48, f"E0 feature dimension must be 48, got {dim_e0}"

    print("Running E1 evidence union evaluation...", flush=True)
    preds_e1, dim_e1, metrics_e1, scores_e1, full_rankings_e1, full_scores_e1 = run_lobo_pipeline(
        "E1_D1_JINA_EVIDENCE_UNION",
        channels_e1,
        local_views,
        extended,
        type_rows,
        cite_rows,
        blocks,
        all_ids,
        gold,
    )
    assert dim_e1 == 48, f"E1 feature dimension must be 48, got {dim_e1}"

    # Paired comparisons
    wins, losses, ties = 0, 0, 0
    top5_set_churn = 0
    top5_ordered_churn = 0
    gold_in_count = 0
    gold_out_count = 0
    changed_queries = []
    per_query_deltas = {}

    for q in all_ids:
        q_gold = gold[q]
        top5_e0 = preds_e0[q]
        top5_e1 = preds_e1[q]
        s_top5_e0 = set(top5_e0)
        s_top5_e1 = set(top5_e1)

        r5_e0 = len(s_top5_e0 & q_gold) / max(1, len(q_gold))
        r5_e1 = len(s_top5_e1 & q_gold) / max(1, len(q_gold))
        delta_r5 = r5_e1 - r5_e0
        per_query_deltas[q] = delta_r5

        if delta_r5 > 1e-9:
            wins += 1
        elif delta_r5 < -1e-9:
            losses += 1
        else:
            ties += 1

        if s_top5_e0 != s_top5_e1:
            top5_set_churn += 1
            changed_queries.append(q)

        if top5_e0 != top5_e1:
            top5_ordered_churn += 1

        for g in q_gold:
            in_e0 = g in s_top5_e0
            in_e1 = g in s_top5_e1
            if not in_e0 and in_e1:
                gold_in_count += 1
            elif in_e0 and not in_e1:
                gold_out_count += 1

    deltas = {
        "recall_at_5": metrics_e1["recall_at_5"] - metrics_e0["recall_at_5"],
        "precision_at_5": metrics_e1["precision_at_5"] - metrics_e0["precision_at_5"],
        "single_gold_recall_at_5": metrics_e1["single_gold_recall_at_5"] - metrics_e0["single_gold_recall_at_5"],
        "multi_gold_recall_at_5": metrics_e1["multi_gold_recall_at_5"] - metrics_e0["multi_gold_recall_at_5"],
        "block_deltas": {
            b: metrics_e1["block_recalls"][b] - metrics_e0["block_recalls"][b]
            for b in ["A", "B", "C", "D"]
        },
    }

    paired_comparison = {
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "net_wins": wins - losses,
        "top5_set_churn_count": top5_set_churn,
        "top5_set_churn_pct": float(top5_set_churn / len(all_ids) * 100),
        "top5_ordered_churn_count": top5_ordered_churn,
        "top5_ordered_churn_pct": float(top5_ordered_churn / len(all_ids) * 100),
        "gold_crossings_into_top5": gold_in_count,
        "gold_crossings_out_of_top5": gold_out_count,
        "net_gold_crossings": gold_in_count - gold_out_count,
    }

    # 3. Boundary rank 4-8 diagnostics
    boundary_diagnostics = []
    for q in changed_queries:
        q_gold = gold[q]
        cand_list = extended[q]
        r_e0 = full_rankings_e0[q]
        r_e1 = full_rankings_e1[q]

        for d in cand_list:
            rank_e0 = r_e0.index(d) + 1
            rank_e1 = r_e1.index(d) + 1
            is_boundary = (4 <= rank_e0 <= 8) or (4 <= rank_e1 <= 8) or (d in q_gold and (rank_e0 <= 10 or rank_e1 <= 10))

            if is_boundary and (rank_e0 != rank_e1 or d in q_gold):
                sc_old = float(aligned_old[q][d])
                sc_sec = float(aligned_sec[q][d])
                sc_uni = float(aligned_union[q][d])

                if sc_old > sc_sec + 1e-9:
                    union_source = "OLD"
                elif sc_sec > sc_old + 1e-9:
                    union_source = "SECTION"
                else:
                    union_source = "TIE"

                boundary_diagnostics.append({
                    "qid": q,
                    "doc_id": d,
                    "is_gold": d in q_gold,
                    "e0_rank": rank_e0,
                    "e1_rank": rank_e1,
                    "rank_shift": rank_e0 - rank_e1,
                    "e0_decision_score": full_scores_e0[q].get(d),
                    "e1_decision_score": full_scores_e1[q].get(d),
                    "old_jina_raw": sc_old,
                    "section_ce_raw": sc_sec,
                    "union_raw": sc_uni,
                    "union_source": union_source,
                })

    # 4. Oracle headroom diagnostics (headroom measurement only)
    oracle_old_top5_recalls = []
    oracle_sec_top5_recalls = []
    oracle_uni_top5_recalls = []

    for q in all_ids:
        q_gold = gold[q]
        t5_e0 = set(preds_e0[q])

        t5_old_exp = set(sorted(extended[q], key=lambda d: aligned_old[q].get(d, -1e9), reverse=True)[:5])
        t5_sec_exp = set(sorted(extended[q], key=lambda d: aligned_sec[q].get(d, -1e9), reverse=True)[:5])
        t5_uni_exp = set(sorted(extended[q], key=lambda d: aligned_union[q].get(d, -1e9), reverse=True)[:5])

        r_old_oracle = len((t5_e0 | t5_old_exp) & q_gold) / max(1, len(q_gold))
        r_sec_oracle = len((t5_e0 | t5_sec_exp) & q_gold) / max(1, len(q_gold))
        r_uni_oracle = len((t5_e0 | t5_uni_exp) & q_gold) / max(1, len(q_gold))

        oracle_old_top5_recalls.append(r_old_oracle)
        oracle_sec_top5_recalls.append(r_sec_oracle)
        oracle_uni_top5_recalls.append(r_uni_oracle)

    oracle_headroom = {
        "e0_recall_at_5": metrics_e0["recall_at_5"],
        "e0_union_old_jina_top5_recall": float(np.mean(oracle_old_top5_recalls)),
        "e0_union_section_ce_top5_recall": float(np.mean(oracle_sec_top5_recalls)),
        "e0_union_evidence_union_top5_recall": float(np.mean(oracle_uni_top5_recalls)),
    }

    # 5. Paired bootstrap
    print("Running 10,000 paired bootstrap resamples...", flush=True)
    bootstrap_results = run_paired_bootstrap(per_query_deltas, blocks, all_ids, n_samples=10000, seed=2026)

    # 6. Generalization gates & verdict
    pooled_gain = deltas["recall_at_5"]
    block_d_gain = deltas["block_deltas"]["D"]
    single_gain = deltas["single_gold_recall_at_5"]
    multi_gain = deltas["multi_gold_recall_at_5"]
    min_block_delta = min(deltas["block_deltas"].values())

    # Standalone check: load standalone report
    standalone_rep_path = RES_DIR / "JINA_UNION_STANDALONE_REPORT.json"
    standalone_gain = 0.0
    if standalone_rep_path.exists():
        s_rep = json.loads(standalone_rep_path.read_text(encoding="utf-8"))
        standalone_gain = s_rep.get("deltas", {}).get("recall_at_5", 0.0)

    standalone_passed = standalone_gain >= -1e-9
    safety_passed = (
        wins > losses
        and block_d_gain >= -1e-9
        and min_block_delta >= -0.001 - 1e-9
        and single_gain >= -0.001 - 1e-9
        and multi_gain >= -0.003 - 1e-9
        and standalone_passed
    )

    if metrics_e1["recall_at_5"] >= 0.960000 - 1e-9 and safety_passed:
        verdict = "BREAK_096_CAL_EVIDENCE_UNION"
    elif pooled_gain >= 0.001 - 1e-9 and safety_passed:
        verdict = "KEEP_JINA_EVIDENCE_UNION"
    elif pooled_gain > 0 and safety_passed:
        verdict = "INCONCLUSIVE_JINA_EVIDENCE_UNION"
    else:
        verdict = "KILL_JINA_EVIDENCE_UNION"

    print("\n========================= METRIC SUMMARY =========================")
    print(f"Metric                 | E0 (Baseline 48D) | E1 (Union 48D)    | Delta")
    print(f"-----------------------+-------------------+-------------------+----------")
    print(f"Pooled Recall@5        | {metrics_e0['recall_at_5']:.16f} | {metrics_e1['recall_at_5']:.16f} | {deltas['recall_at_5']:+.16f}")
    print(f"Precision@5            | {metrics_e0['precision_at_5']:.6f}          | {metrics_e1['precision_at_5']:.6f}          | {deltas['precision_at_5']:+.6f}")
    print(f"Block A Recall@5       | {metrics_e0['block_recalls']['A']:.6f}          | {metrics_e1['block_recalls']['A']:.6f}          | {deltas['block_deltas']['A']:+.6f}")
    print(f"Block B Recall@5       | {metrics_e0['block_recalls']['B']:.6f}          | {metrics_e1['block_recalls']['B']:.6f}          | {deltas['block_deltas']['B']:+.6f}")
    print(f"Block C Recall@5       | {metrics_e0['block_recalls']['C']:.6f}          | {metrics_e1['block_recalls']['C']:.6f}          | {deltas['block_deltas']['C']:+.6f}")
    print(f"Block D Recall@5       | {metrics_e0['block_recalls']['D']:.6f}          | {metrics_e1['block_recalls']['D']:.6f}          | {deltas['block_deltas']['D']:+.6f}")
    print(f"Single-gold Recall@5   | {metrics_e0['single_gold_recall_at_5']:.6f}          | {metrics_e1['single_gold_recall_at_5']:.6f}          | {deltas['single_gold_recall_at_5']:+.6f}")
    print(f"Multi-gold Recall@5    | {metrics_e0['multi_gold_recall_at_5']:.6f}          | {metrics_e1['multi_gold_recall_at_5']:.6f}          | {deltas['multi_gold_recall_at_5']:+.6f}")
    print(f"Wins / Losses / Ties   | -                 | -                 | {wins} / {losses} / {ties} (Net: {wins - losses:+d})")
    print(f"Gold In / Out of Top5  | -                 | -                 | {gold_in_count} / {gold_out_count} (Net: {gold_in_count - gold_out_count:+d})")
    print(f"Top-5 Set Churn        | -                 | -                 | {top5_set_churn} ({top5_set_churn/len(all_ids)*100:.1f}%)")
    print(f"Top-5 Ordered Churn    | -                 | -                 | {top5_ordered_churn} ({top5_ordered_churn/len(all_ids)*100:.1f}%)")
    print(f"Bootstrap 95% CI (Ord) | -                 | -                 | [{bootstrap_results['ordinary_bootstrap']['ci_2_5']:+.6f}, {bootstrap_results['ordinary_bootstrap']['ci_97_5']:+.6f}] (P>0: {bootstrap_results['ordinary_bootstrap']['p_delta_gt_zero']:.3f})")
    print(f"Bootstrap 95% CI (Blk) | -                 | -                 | [{bootstrap_results['block_stratified_bootstrap']['ci_2_5']:+.6f}, {bootstrap_results['block_stratified_bootstrap']['ci_97_5']:+.6f}] (P>0: {bootstrap_results['block_stratified_bootstrap']['p_delta_gt_zero']:.3f})")
    print(f"VERDICT                | -                 | -                 | {verdict}")
    print("==================================================================\n", flush=True)

    eval_summary = {
        "experiment_id": "HUY_D1_JINA_EVIDENCE_UNION_V1",
        "verdict": verdict,
        "e0_baseline": {k: v for k, v in metrics_e0.items() if k != "per_query_recalls"},
        "e1_evidence_union": {k: v for k, v in metrics_e1.items() if k != "per_query_recalls"},
        "deltas": deltas,
        "paired_comparison": paired_comparison,
        "oracle_headroom": oracle_headroom,
        "bootstrap": bootstrap_results,
        "changed_queries": changed_queries,
        "boundary_diagnostics_count": len(boundary_diagnostics),
    }

    return (
        eval_summary,
        preds_e0,
        preds_e1,
        scores_e0,
        scores_e1,
        full_rankings_e0,
        full_rankings_e1,
        full_scores_e0,
        full_scores_e1,
        boundary_diagnostics,
        bootstrap_results,
    )


if __name__ == "__main__":
    evaluate_e0_e1_lobo()
