"""Deterministic 10,000-sample paired bootstrap on CAL600."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = ROOT / "results" / "gemini" / "huy_d1_lal_case_memory_v1"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def run_bootstrap_analysis(n_bootstraps: int = 10000, seed: int = 2026) -> Dict[str, Any]:
    preds_file = RESULTS_DIR / "LAL_MEMORY_CAL_PREDICTIONS.jsonl"
    if not preds_file.exists():
        raise FileNotFoundError(f"Missing {preds_file}. Run evaluate_cal_memory.py first.")

    import sys
    sys.path.insert(0, str(ROOT))
    from src.gemini.huy_vnlegal_rank_ablation_v1.evaluate_ablation_cal import load_cal_inputs

    queries, blocks, all_ids, extended, local_views, full_channels_cv, gold, vnlegal_cv, type_rows, cite_rows = load_cal_inputs()

    records = []
    with open(preds_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    rec_by_qid = {r["qid"]: r for r in records}

    m1_deltas = np.asarray([rec_by_qid[q]["m1_delta_r5"] for q in all_ids], dtype=np.float64)
    m2_deltas = np.asarray([rec_by_qid[q]["m2_delta_r5"] for q in all_ids], dtype=np.float64)

    rng = np.random.RandomState(seed)

    # 1. Ordinary unstratified bootstrap
    unstrat_idx = rng.randint(0, len(all_ids), size=(n_bootstraps, len(all_ids)))

    def summarize_deltas(deltas: np.ndarray, indices: np.ndarray) -> Dict[str, float]:
        sample_means = np.mean(deltas[indices], axis=1)
        p_val = float(np.mean(sample_means > 0.0))
        return {
            "mean_delta": float(np.mean(sample_means)),
            "median_delta": float(np.median(sample_means)),
            "ci_2_5": float(np.percentile(sample_means, 2.5)),
            "ci_97_5": float(np.percentile(sample_means, 97.5)),
            "prob_positive": p_val,
        }

    unstrat_m1 = summarize_deltas(m1_deltas, unstrat_idx)
    unstrat_m2 = summarize_deltas(m2_deltas, unstrat_idx)

    # 2. Block-stratified bootstrap preserving block sizes (A=100, B=100, C=100, D=300)
    block_indices = {}
    qid_to_pos = {q: i for i, q in enumerate(all_ids)}
    for b_name, b_qids in blocks.items():
        block_indices[b_name] = np.asarray([qid_to_pos[q] for q in b_qids], dtype=np.int64)

    strat_indices = np.empty((n_bootstraps, len(all_ids)), dtype=np.int64)
    col_offset = 0
    for b_name in sorted(blocks.keys()):
        b_idx = block_indices[b_name]
        b_size = len(b_idx)
        sampled_b = b_idx[rng.randint(0, b_size, size=(n_bootstraps, b_size))]
        strat_indices[:, col_offset : col_offset + b_size] = sampled_b
        col_offset += b_size

    strat_m1 = summarize_deltas(m1_deltas, strat_indices)
    strat_m2 = summarize_deltas(m2_deltas, strat_indices)

    bootstrap_report = {
        "schema_version": "dsc2026.gemini.huy_d1_lal_case_memory_v1.paired_bootstrap.v1",
        "n_samples": n_bootstraps,
        "seed": seed,
        "total_queries": len(all_ids),
        "m1_vs_m0": {
            "unstratified": unstrat_m1,
            "block_stratified": strat_m1,
        },
        "m2_vs_m0": {
            "unstratified": unstrat_m2,
            "block_stratified": strat_m2,
        },
    }

    out_path = RESULTS_DIR / "LAL_MEMORY_BOOTSTRAP.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(bootstrap_report, f, indent=2)
    print(f"Wrote {out_path}", flush=True)
    return bootstrap_report


if __name__ == "__main__":
    run_bootstrap_analysis()
