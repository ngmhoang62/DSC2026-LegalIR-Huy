"""10,000 paired bootstrap resamples for S1 - S0 Recall@5 on CAL600."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np

from .common import RESULTS_DIR, ROOT


def run_paired_bootstrap(n_resamples: int = 10000, seed: int = 2026) -> Dict[str, Any]:
    t0 = time.perf_counter()
    pred_path = RESULTS_DIR / "SPARSE_D1_CAL_PREDICTIONS.jsonl"
    if not pred_path.exists():
        raise FileNotFoundError(f"Missing predictions file: {pred_path}")

    qids = []
    s0_recalls = []
    s1_recalls = []

    with pred_path.open("r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            qids.append(r["qid"])
            s0_recalls.append(float(r["s0_recall"]))
            s1_recalls.append(float(r["s1_recall"]))

    qids = np.array(qids)
    deltas = np.array(s1_recalls) - np.array(s0_recalls)
    n = len(deltas)

    # 1. Ordinary query bootstrap
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, n, size=(n_resamples, n))
    resampled_deltas = np.mean(deltas[indices], axis=1)

    mean_delta = float(np.mean(resampled_deltas))
    median_delta = float(np.median(resampled_deltas))
    ci_2_5 = float(np.percentile(resampled_deltas, 2.5))
    ci_97_5 = float(np.percentile(resampled_deltas, 97.5))
    p_gt_0 = float(np.mean(resampled_deltas > 0))

    # 2. Block-stratified bootstrap
    # Load blocks
    from tune_corpus_cap32_fusion import build_training_cap
    queries, blocks, all_ids, extended, local_views, base_scores = (
        build_training_cap(
            ROOT,
            32,
            "results/corpus_index/holdout_extended_scores_cap32.pkl",
            depth=20,
        )
    )

    qid_to_idx = {q: i for i, q in enumerate(qids)}
    block_indices = {b: np.array([qid_to_idx[q] for q in q_list if q in qid_to_idx]) for b, q_list in blocks.items()}

    strat_resamples = np.zeros(n_resamples, dtype=np.float64)
    rng_strat = np.random.default_rng(seed + 1)

    for b, b_idxs in block_indices.items():
        b_n = len(b_idxs)
        b_resamp_idxs = b_idxs[rng_strat.integers(0, b_n, size=(n_resamples, b_n))]
        strat_resamples += np.sum(deltas[b_resamp_idxs], axis=1)
    strat_resamples /= n

    strat_mean = float(np.mean(strat_resamples))
    strat_median = float(np.median(strat_resamples))
    strat_2_5 = float(np.percentile(strat_resamples, 2.5))
    strat_97_5 = float(np.percentile(strat_resamples, 97.5))
    strat_p_gt_0 = float(np.mean(strat_resamples > 0))

    boot_result = {
        "schema_version": "dsc2026.gemini.huy_d1_legalir_sparse_port_v1.sparse_d1_bootstrap.v1",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "runtime_seconds": float(time.perf_counter() - t0),
        "n_resamples": n_resamples,
        "seed": seed,
        "query_bootstrap": {
            "mean": mean_delta,
            "median": median_delta,
            "ci_2_5": ci_2_5,
            "ci_97_5": ci_97_5,
            "p_delta_gt_0": p_gt_0,
        },
        "block_stratified_bootstrap": {
            "mean": strat_mean,
            "median": strat_median,
            "ci_2_5": strat_2_5,
            "ci_97_5": strat_97_5,
            "p_delta_gt_0": strat_p_gt_0,
        }
    }

    out_path = RESULTS_DIR / "SPARSE_D1_BOOTSTRAP.json"
    out_path.write_text(json.dumps(boot_result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote paired bootstrap report to {out_path}")
    return boot_result


if __name__ == "__main__":
    run_paired_bootstrap()
