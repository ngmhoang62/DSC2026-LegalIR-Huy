"""Evaluate standalone ranking quality of old vs adapted Jina channel on CAL pool."""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path
import numpy as np
import scipy.stats as stats

ROOT = Path("D:/Study/DSC2026/sota")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tune_corpus_cap32_fusion import build_training_cap

RES_DIR = ROOT / "results/gemini/huy_d1_jina_ft_continuation_v1"
OLD_JINA_PKL = ROOT / "results/from_drive/jina_ft_cv.pkl"
NEW_JINA_PKL = RES_DIR / "jina_ft_continued_cv.pkl"
OUT_REPORT = RES_DIR / "JINA_CHANNEL_STANDALONE_REPORT.json"


def eval_metrics_at_k(order: list, golds: set, k: int) -> float:
    return len(set(order[:k]) & golds) / max(1, len(golds))


def run_standalone_evaluation() -> dict:
    print("Loading CAL600 annotations and score caches...", flush=True)
    queries, blocks, all_ids, extended, local, _ = build_training_cap(
        ROOT, 32, "results/corpus_index/holdout_extended_scores_cap32.pkl", depth=20
    )

    with open(OLD_JINA_PKL, "rb") as f:
        old_scores = pickle.load(f)
    with open(NEW_JINA_PKL, "rb") as f:
        new_scores = pickle.load(f)

    # Metrics containers
    metrics_old = {1: [], 5: [], 8: [], 10: []}
    metrics_new = {1: [], 5: [], 8: [], 10: []}
    single_old, single_new = [], []
    multi_old, multi_new = [], []

    wins, losses, ties = 0, 0, 0
    spearman_corrs = []

    for q in all_ids:
        q_str = str(q)
        golds = set(queries[q_str][1])
        cands = list(extended[q_str])

        s_old = old_scores[q_str]
        s_new = new_scores[q_str]

        # Rank candidates
        order_old = sorted(cands, key=lambda d: s_old.get(d, -1e9), reverse=True)
        order_new = sorted(cands, key=lambda d: s_new.get(d, -1e9), reverse=True)

        for k in [1, 5, 8, 10]:
            r_old = eval_metrics_at_k(order_old, golds, k)
            r_new = eval_metrics_at_k(order_new, golds, k)
            metrics_old[k].append(r_old)
            metrics_new[k].append(r_new)

        r5_old = eval_metrics_at_k(order_old, golds, 5)
        r5_new = eval_metrics_at_k(order_new, golds, 5)

        if len(golds) == 1:
            single_old.append(r5_old)
            single_new.append(r5_new)
        else:
            multi_old.append(r5_old)
            multi_new.append(r5_new)

        if r5_new > r5_old + 1e-9:
            wins += 1
        elif r5_new < r5_old - 1e-9:
            losses += 1
        else:
            ties += 1

        # Spearman correlation
        v_old = [s_old.get(d, 0.0) for d in cands]
        v_new = [s_new.get(d, 0.0) for d in cands]
        rho, _ = stats.spearmanr(v_old, v_new)
        if np.isfinite(rho):
            spearman_corrs.append(float(rho))

    report = {
        "experiment_id": "HUY_D1_JINA_FT_CONTINUATION_V1",
        "total_cal_queries": len(all_ids),
        "old_jina_standalone": {
            "recall_at_1": float(np.mean(metrics_old[1])),
            "recall_at_5": float(np.mean(metrics_old[5])),
            "recall_at_8": float(np.mean(metrics_old[8])),
            "recall_at_10": float(np.mean(metrics_old[10])),
            "single_gold_recall_at_5": float(np.mean(single_old)),
            "multi_gold_recall_at_5": float(np.mean(multi_old)),
        },
        "new_jina_standalone": {
            "recall_at_1": float(np.mean(metrics_new[1])),
            "recall_at_5": float(np.mean(metrics_new[5])),
            "recall_at_8": float(np.mean(metrics_new[8])),
            "recall_at_10": float(np.mean(metrics_new[10])),
            "single_gold_recall_at_5": float(np.mean(single_new)),
            "multi_gold_recall_at_5": float(np.mean(multi_new)),
        },
        "delta_standalone": {
            "recall_at_1": float(np.mean(metrics_new[1]) - np.mean(metrics_old[1])),
            "recall_at_5": float(np.mean(metrics_new[5]) - np.mean(metrics_old[5])),
            "recall_at_8": float(np.mean(metrics_new[8]) - np.mean(metrics_old[8])),
            "recall_at_10": float(np.mean(metrics_new[10]) - np.mean(metrics_old[10])),
            "single_gold_recall_at_5": float(np.mean(single_new) - np.mean(single_old)),
            "multi_gold_recall_at_5": float(np.mean(multi_new) - np.mean(multi_old)),
        },
        "paired_comparison_at_5": {
            "wins": wins,
            "losses": losses,
            "ties": ties,
        },
        "mean_within_query_spearman": float(np.mean(spearman_corrs)) if spearman_corrs else 1.0,
        "gate_g_passed": float(np.mean(metrics_new[5])) >= float(np.mean(metrics_old[5])),
    }

    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(
        f"Standalone Jina R@5: old={report['old_jina_standalone']['recall_at_5']:.6f} -> "
        f"new={report['new_jina_standalone']['recall_at_5']:.6f} "
        f"(delta={report['delta_standalone']['recall_at_5']:+.6f}, wins={wins}, losses={losses}, ties={ties})",
        flush=True,
    )
    print(f"Wrote {OUT_REPORT}", flush=True)
    return report


if __name__ == "__main__":
    run_standalone_evaluation()
