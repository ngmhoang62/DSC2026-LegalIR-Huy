"""Evaluate S0 (D1 Baseline 48D) vs S1 (D1 + legal_section_ce 50D) on CAL600 under LOBO."""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path("D:/Study/DSC2026/sota")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tune_expanded_fusion_selection import ltr_features
from src.gemini.huy_d1_legal_section_evidence_v1.common import (
    D1_VIEWS,
    RES_DIR,
    SCORE_CACHE_PKL,
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
) -> Tuple[Dict[str, List[str]], int, Dict[str, Any], Dict[str, np.ndarray]]:
    preds: Dict[str, List[str]] = {}
    scores_dict: Dict[str, np.ndarray] = {}
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
    }
    return preds, feature_dim, metrics, scores_dict


def evaluate_both_arms() -> dict:
    seed_everything(2026)
    print("=== EVALUATING S0 (BASELINE 48D) VS S1 (SECTION EVIDENCE 50D) ===", flush=True)

    # 1. Load CAL data
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

    # 2. Load legal_section_ce cache
    if not SCORE_CACHE_PKL.exists():
        raise FileNotFoundError(f"Score cache not found: {SCORE_CACHE_PKL}")
    legal_section_ce_cv = pickle.loads(SCORE_CACHE_PKL.read_bytes())
    print(f"Loaded legal_section_ce cache for {len(legal_section_ce_cv)} queries.", flush=True)

    # Align channel scores with candidate pool
    def align_channel(raw_map, floor=None):
        fl = floor if floor is not None else min(v for q in raw_map for v in raw_map[q].values())
        return {q: {d: raw_map.get(q, {}).get(d, fl) for d in extended[q]} for q in all_ids}

    aligned_legal_section_ce = align_channel(legal_section_ce_cv)

    # Arm S0: D1 48D channels
    channels_s0 = full_channels_cv

    # Arm S1: D1 48D + legal_section_ce -> 50D channels
    channels_s1 = {
        **full_channels_cv,
        "legal_section_ce": aligned_legal_section_ce,
    }

    # Evaluate Arm S0
    print("\n--- Running S0 LOBO Evaluation (Baseline 48D) ---", flush=True)
    preds_s0, dim_s0, metrics_s0, scores_s0 = run_lobo_pipeline(
        "S0_D1_BASELINE",
        channels_s0,
        local_views,
        extended,
        type_rows,
        cite_rows,
        blocks,
        all_ids,
        gold,
    )

    # Evaluate Arm S1
    print("\n--- Running S1 LOBO Evaluation (Section Evidence 50D) ---", flush=True)
    preds_s1, dim_s1, metrics_s1, scores_s1 = run_lobo_pipeline(
        "S1_D1_LEGAL_SECTION_CE",
        channels_s1,
        local_views,
        extended,
        type_rows,
        cite_rows,
        blocks,
        all_ids,
        gold,
    )

    # Compute paired comparisons
    print("\n--- Computing Paired Comparisons (S1 vs S0) ---", flush=True)
    wins = 0
    losses = 0
    ties = 0
    changed_top5_count = 0
    gold_in_count = 0
    gold_out_count = 0
    changed_queries = []

    for q in all_ids:
        q_gold = gold[q]
        top5_s0 = set(preds_s0[q])
        top5_s1 = set(preds_s1[q])

        rec_s0 = len(top5_s0 & q_gold) / max(1, len(q_gold))
        rec_s1 = len(top5_s1 & q_gold) / max(1, len(q_gold))

        diff = rec_s1 - rec_s0
        if diff > 1e-9:
            wins += 1
        elif diff < -1e-9:
            losses += 1
        else:
            ties += 1

        is_changed_top5 = top5_s0 != top5_s1
        if is_changed_top5:
            changed_top5_count += 1
            changed_queries.append(q)

        # Crossings
        for g in q_gold:
            in_s0 = g in top5_s0
            in_s1 = g in top5_s1
            if not in_s0 and in_s1:
                gold_in_count += 1
            elif in_s0 and not in_s1:
                gold_out_count += 1

    # Deltas
    deltas = {
        "recall_at_5": metrics_s1["recall_at_5"] - metrics_s0["recall_at_5"],
        "precision_at_5": metrics_s1["precision_at_5"] - metrics_s0["precision_at_5"],
        "single_gold_recall_at_5": metrics_s1["single_gold_recall_at_5"] - metrics_s0["single_gold_recall_at_5"],
        "multi_gold_recall_at_5": metrics_s1["multi_gold_recall_at_5"] - metrics_s0["multi_gold_recall_at_5"],
        "block_deltas": {
            b: metrics_s1["block_recalls"][b] - metrics_s0["block_recalls"][b]
            for b in ["A", "B", "C", "D"]
        },
    }

    paired_comparison = {
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "net_wins": wins - losses,
        "changed_top5_queries_count": changed_top5_count,
        "changed_top5_queries_pct": float(changed_top5_count / len(all_ids) * 100),
        "gold_crossings_into_top5": gold_in_count,
        "gold_crossings_out_of_top5": gold_out_count,
        "net_gold_crossings": gold_in_count - gold_out_count,
    }

    # Standalone legal_section_ce quality
    standalone_recalls = []
    standalone_precisions = []
    for q in all_ids:
        q_gold = gold[q]
        q_scores = aligned_legal_section_ce.get(q, {})
        ranked_cand = sorted(extended[q], key=lambda d: q_scores.get(d, -1e12), reverse=True)
        top5 = set(ranked_cand[:5])
        standalone_recalls.append(len(top5 & q_gold) / max(1, len(q_gold)))
        standalone_precisions.append(len(top5 & q_gold) / 5.0)

    standalone_metrics = {
        "standalone_recall_at_5": float(np.mean(standalone_recalls)),
        "standalone_precision_at_5": float(np.mean(standalone_precisions)),
    }

    # Print summary table
    print("\n========================= METRIC SUMMARY =========================")
    print(f"Metric                 | S0 (Baseline 48D) | S1 (Section 50D)  | Delta")
    print(f"-----------------------+-------------------+-------------------+----------")
    print(f"Pooled Recall@5        | {metrics_s0['recall_at_5']:.16f} | {metrics_s1['recall_at_5']:.16f} | {deltas['recall_at_5']:+.16f}")
    print(f"Precision@5            | {metrics_s0['precision_at_5']:.6f}          | {metrics_s1['precision_at_5']:.6f}          | {deltas['precision_at_5']:+.6f}")
    print(f"Block A Recall@5       | {metrics_s0['block_recalls']['A']:.6f}          | {metrics_s1['block_recalls']['A']:.6f}          | {deltas['block_deltas']['A']:+.6f}")
    print(f"Block B Recall@5       | {metrics_s0['block_recalls']['B']:.6f}          | {metrics_s1['block_recalls']['B']:.6f}          | {deltas['block_deltas']['B']:+.6f}")
    print(f"Block C Recall@5       | {metrics_s0['block_recalls']['C']:.6f}          | {metrics_s1['block_recalls']['C']:.6f}          | {deltas['block_deltas']['C']:+.6f}")
    print(f"Block D Recall@5       | {metrics_s0['block_recalls']['D']:.6f}          | {metrics_s1['block_recalls']['D']:.6f}          | {deltas['block_deltas']['D']:+.6f}")
    print(f"Single-gold Recall@5   | {metrics_s0['single_gold_recall_at_5']:.6f}          | {metrics_s1['single_gold_recall_at_5']:.6f}          | {deltas['single_gold_recall_at_5']:+.6f}")
    print(f"Multi-gold Recall@5    | {metrics_s0['multi_gold_recall_at_5']:.6f}          | {metrics_s1['multi_gold_recall_at_5']:.6f}          | {deltas['multi_gold_recall_at_5']:+.6f}")
    print(f"Wins / Losses / Ties   | -                 | -                 | {wins} / {losses} / {ties}")
    print(f"Gold In / Out of Top5  | -                 | -                 | {gold_in_count} / {gold_out_count} (Net: {gold_in_count - gold_out_count:+d})")
    print(f"Changed Top-5 sets     | -                 | -                 | {changed_top5_count} ({changed_top5_count/len(all_ids)*100:.1f}%)")
    print(f"Standalone Section R@5 | -                 | {standalone_metrics['standalone_recall_at_5']:.6f}          | -")
    print("==================================================================")

    # Promotion decision logic per Section 12
    pooled_gain = deltas["recall_at_5"]
    block_d_gain = deltas["block_deltas"]["D"]
    single_gain = deltas["single_gold_recall_at_5"]
    multi_gain = deltas["multi_gold_recall_at_5"]

    if pooled_gain >= 0.001 - 1e-9 and wins > losses and block_d_gain >= -1e-9 and single_gain >= -0.002 and multi_gain >= -0.003:
        verdict = "KEEP_FOR_NEXT_STAGE"
    elif pooled_gain > 0 and wins > losses and block_d_gain >= -0.001:
        verdict = "INCONCLUSIVE"
    else:
        verdict = "KILL_LEGAL_SECTION_SCORE_V1"

    print(f"\nPROMOTION DECISION VERDICT: {verdict}\n", flush=True)

    evaluation_report = {
        "experiment_id": "HUY_D1_LEGAL_SECTION_EVIDENCE_V1",
        "verdict": verdict,
        "s0_baseline": metrics_s0,
        "s1_section_evidence": metrics_s1,
        "deltas": deltas,
        "paired_comparison": paired_comparison,
        "standalone_legal_section_ce": standalone_metrics,
        "changed_queries": changed_queries,
    }

    return evaluation_report, preds_s0, preds_s1, scores_s0, scores_s1


if __name__ == "__main__":
    report, _, _, _, _ = evaluate_both_arms()
    print("Evaluation completed successfully.")
