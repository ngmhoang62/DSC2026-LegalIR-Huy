"""Standalone expert evaluation: OLD_JINA_FT vs JINA_FT_EVIDENCE_UNION on CAL candidate pool."""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import scipy.stats as stats

ROOT = Path("D:/Study/DSC2026/sota")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.gemini.huy_d1_jina_evidence_union_v1.common import (
    EVIDENCE_UNION_CACHE_PKL,
    OLD_JINA_CACHE_PKL,
    RES_DIR,
    load_cal_data,
    seed_everything,
)


def evaluate_standalone_experts() -> dict:
    seed_everything(2026)
    print("=== EVALUATING STANDALONE EXPERTS: OLD_JINA_FT VS JINA_FT_EVIDENCE_UNION ===", flush=True)
    RES_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load CAL dataset
    docs, queries, blocks, all_ids, extended, local_views, full_channels_cv, gold, type_rows, cite_rows = load_cal_data()

    # 2. Load caches
    assert OLD_JINA_CACHE_PKL.exists(), f"Old cache missing: {OLD_JINA_CACHE_PKL}"
    assert EVIDENCE_UNION_CACHE_PKL.exists(), f"Union cache missing: {EVIDENCE_UNION_CACHE_PKL}"

    raw_old = pickle.loads(OLD_JINA_CACHE_PKL.read_bytes())
    old_scores = raw_old.get("scores", raw_old) if isinstance(raw_old, dict) else raw_old

    raw_union = pickle.loads(EVIDENCE_UNION_CACHE_PKL.read_bytes())
    union_scores = raw_union.get("scores", raw_union) if isinstance(raw_union, dict) else raw_union

    def calc_metrics(scores_map):
        recalls_1, recalls_5, recalls_8, recalls_10 = [], [], [], []
        single_rec_5, multi_rec_5 = [], []
        rankings = {}

        for q in all_ids:
            q_gold = gold[q]
            ranked = sorted(extended[q], key=lambda d: scores_map.get(q, {}).get(d, -1e9), reverse=True)
            rankings[q] = ranked

            r1 = len(set(ranked[:1]) & q_gold) / max(1, len(q_gold))
            r5 = len(set(ranked[:5]) & q_gold) / max(1, len(q_gold))
            r8 = len(set(ranked[:8]) & q_gold) / max(1, len(q_gold))
            r10 = len(set(ranked[:10]) & q_gold) / max(1, len(q_gold))

            recalls_1.append(r1)
            recalls_5.append(r5)
            recalls_8.append(r8)
            recalls_10.append(r10)

            if len(q_gold) == 1:
                single_rec_5.append(r5)
            else:
                multi_rec_5.append(r5)

        return {
            "recall_at_1": float(np.mean(recalls_1)),
            "recall_at_5": float(np.mean(recalls_5)),
            "recall_at_8": float(np.mean(recalls_8)),
            "recall_at_10": float(np.mean(recalls_10)),
            "single_gold_recall_at_5": float(np.mean(single_rec_5)),
            "multi_gold_recall_at_5": float(np.mean(multi_rec_5)),
            "rankings": rankings,
        }

    old_metrics = calc_metrics(old_scores)
    union_metrics = calc_metrics(union_scores)

    # Paired comparisons at Recall@5
    wins, losses, ties = 0, 0, 0
    within_query_rhos = []

    for q in all_ids:
        q_gold = gold[q]
        top5_old = set(old_metrics["rankings"][q][:5])
        top5_union = set(union_metrics["rankings"][q][:5])

        r5_old = len(top5_old & q_gold) / max(1, len(q_gold))
        r5_union = len(top5_union & q_gold) / max(1, len(q_gold))

        if r5_union > r5_old + 1e-9:
            wins += 1
        elif r5_union < r5_old - 1e-9:
            losses += 1
        else:
            ties += 1

        # Rank correlation on full candidates of query q
        cands = extended[q]
        vec_old = [old_scores[q][d] for d in cands]
        vec_union = [union_scores[q][d] for d in cands]
        if len(vec_old) > 1 and len(set(vec_old)) > 1 and len(set(vec_union)) > 1:
            rho, _ = stats.spearmanr(vec_old, vec_union)
            if np.isfinite(rho):
                within_query_rhos.append(float(rho))

    mean_rank_corr = float(np.mean(within_query_rhos)) if within_query_rhos else 1.0

    print("\n--- STANDALONE EXPERT COMPARISON ---")
    print(f"Metric                 | Old Jina-FT  | Jina Union   | Delta")
    print(f"-----------------------+--------------+--------------+---------")
    print(f"Recall@1               | {old_metrics['recall_at_1']:.6f}     | {union_metrics['recall_at_1']:.6f}     | {union_metrics['recall_at_1'] - old_metrics['recall_at_1']:+.6f}")
    print(f"Recall@5               | {old_metrics['recall_at_5']:.6f}     | {union_metrics['recall_at_5']:.6f}     | {union_metrics['recall_at_5'] - old_metrics['recall_at_5']:+.6f}")
    print(f"Recall@8               | {old_metrics['recall_at_8']:.6f}     | {union_metrics['recall_at_8']:.6f}     | {union_metrics['recall_at_8'] - old_metrics['recall_at_8']:+.6f}")
    print(f"Recall@10              | {old_metrics['recall_at_10']:.6f}     | {union_metrics['recall_at_10']:.6f}     | {union_metrics['recall_at_10'] - old_metrics['recall_at_10']:+.6f}")
    print(f"Single-gold Recall@5   | {old_metrics['single_gold_recall_at_5']:.6f}     | {union_metrics['single_gold_recall_at_5']:.6f}     | {union_metrics['single_gold_recall_at_5'] - old_metrics['single_gold_recall_at_5']:+.6f}")
    print(f"Multi-gold Recall@5    | {old_metrics['multi_gold_recall_at_5']:.6f}     | {union_metrics['multi_gold_recall_at_5']:.6f}     | {union_metrics['multi_gold_recall_at_5'] - old_metrics['multi_gold_recall_at_5']:+.6f}")
    print(f"R@5 Wins/Losses/Ties   | -            | -            | {wins} / {losses} / {ties} (Net: {wins - losses:+d})")
    print(f"Mean Rank Correlation  | -            | -            | {mean_rank_corr:.6f}")
    print("------------------------------------------------------------------\n", flush=True)

    report = {
        "schema_version": "dsc2026.gemini.huy_d1_jina_evidence_union_v1.standalone.v1",
        "experiment_id": "HUY_D1_JINA_EVIDENCE_UNION_V1",
        "old_jina_ft": {k: v for k, v in old_metrics.items() if k != "rankings"},
        "jina_ft_evidence_union": {k: v for k, v in union_metrics.items() if k != "rankings"},
        "deltas": {
            "recall_at_1": union_metrics["recall_at_1"] - old_metrics["recall_at_1"],
            "recall_at_5": union_metrics["recall_at_5"] - old_metrics["recall_at_5"],
            "recall_at_8": union_metrics["recall_at_8"] - old_metrics["recall_at_8"],
            "recall_at_10": union_metrics["recall_at_10"] - old_metrics["recall_at_10"],
            "single_gold_recall_at_5": union_metrics["single_gold_recall_at_5"] - old_metrics["single_gold_recall_at_5"],
            "multi_gold_recall_at_5": union_metrics["multi_gold_recall_at_5"] - old_metrics["multi_gold_recall_at_5"],
        },
        "paired_at_recall_at_5": {
            "wins": wins,
            "losses": losses,
            "ties": ties,
            "net_wins": wins - losses,
        },
        "mean_within_query_rank_correlation": mean_rank_corr,
    }

    out_file = RES_DIR / "JINA_UNION_STANDALONE_REPORT.json"
    out_file.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {out_file}", flush=True)
    return report


if __name__ == "__main__":
    evaluate_standalone_experts()
