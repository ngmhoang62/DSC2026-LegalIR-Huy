"""Generalization diagnostics and slices for LAL case-memory."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = ROOT / "results" / "gemini" / "huy_d1_lal_case_memory_v1"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

LAL_QUERIES = ROOT.parent / "LegalIR" / "cache" / "exp109b_encoder_complementarity" / "embeddings" / "vnlegal_lal" / "queries.npz"


def normalize(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    return array / np.maximum(norms, 1e-12)


def run_generalization_audit() -> Dict[str, Any]:
    preds_file = RESULTS_DIR / "LAL_MEMORY_CAL_PREDICTIONS.jsonl"
    if not preds_file.exists():
        raise FileNotFoundError(f"Missing {preds_file}. Run evaluate_cal_memory.py first.")

    import sys
    sys.path.insert(0, str(ROOT))
    from src.gemini.huy_vnlegal_rank_ablation_v1.evaluate_ablation_cal import load_cal_inputs
    import run_huy_5fold_fasttrack as core

    queries, blocks, all_ids, extended, local_views, full_channels_cv, gold, vnlegal_cv, type_rows, cite_rows = load_cal_inputs()
    folds, pools, questions, v2_golds, e5_orders, e5_scores, v2_dup, _ = core.load_inputs()

    v2_population = sorted(pools.keys(), key=int)
    v2_set = set(v2_population)

    # Load LAL query embeddings
    with np.load(LAL_QUERIES, allow_pickle=False) as z:
        all_npz_qids = list(map(str, z["query_ids"].tolist()))
        all_npz_vecs = normalize(z["vectors"].astype(np.float32))

    qid_to_vec_idx = {qid: i for i, qid in enumerate(all_npz_qids)}

    # Read predictions
    records = []
    with open(preds_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    rec_by_qid = {r["qid"]: r for r in records}

    # Memory gold document frequency across all V2 queries
    v2_doc_freq = defaultdict(int)
    for q in v2_population:
        for d in v2_golds.get(q, ()):
            v2_doc_freq[d] += 1

    # For each CAL query, find its nearest neighbor in V2 (excluding itself) and maximum gold frequency
    query_diagnostics = {}
    nearest_sims = []

    for q in all_ids:
        q_vec = all_npz_vecs[qid_to_vec_idx[q]]
        # Support pool: V2 excluding q
        support_pool = [s for s in v2_population if s != q]
        support_indices = [qid_to_vec_idx[s] for s in support_pool]
        support_matrix = all_npz_vecs[support_indices]

        sims = support_matrix @ q_vec
        max_sim = float(np.max(sims))
        nearest_sims.append(max_sim)

        # Max frequency among gold docs of q
        gold_q = gold[q]
        max_freq = max([v2_doc_freq[d] for d in gold_q], default=0)

        query_diagnostics[q] = {
            "max_sim": max_sim,
            "max_gold_frequency": max_freq,
            "num_gold": len(gold_q),
            "m0_r5": rec_by_qid[q]["m0_r5"],
            "m1_r5": rec_by_qid[q]["m1_r5"],
            "m1_delta": rec_by_qid[q]["m1_delta_r5"],
            "m2_r5": rec_by_qid[q]["m2_r5"],
            "m2_delta": rec_by_qid[q]["m2_delta_r5"],
        }

    # Similarity quartiles
    q25, q50, q75 = np.percentile(nearest_sims, [25, 50, 75])

    # Slices
    def slice_metrics(qids: List[str]) -> Dict[str, Any]:
        if not qids:
            return {"count": 0, "m0_r5": 0.0, "m1_r5": 0.0, "m1_delta": 0.0, "m2_r5": 0.0, "m2_delta": 0.0}
        m0_mean = float(np.mean([query_diagnostics[q]["m0_r5"] for q in qids]))
        m1_mean = float(np.mean([query_diagnostics[q]["m1_r5"] for q in qids]))
        m2_mean = float(np.mean([query_diagnostics[q]["m2_r5"] for q in qids]))
        return {
            "count": len(qids),
            "m0_recall_at_5": m0_mean,
            "m1_recall_at_5": m1_mean,
            "m1_delta": m1_mean - m0_mean,
            "m2_recall_at_5": m2_mean,
            "m2_delta": m2_mean - m0_mean,
        }

    # 1. Single vs Multi
    single_qids = [q for q in all_ids if query_diagnostics[q]["num_gold"] == 1]
    multi_qids = [q for q in all_ids if query_diagnostics[q]["num_gold"] > 1]

    # 2. Block D
    block_d_qids = blocks["d"]

    # 3. Gold frequency in memory: 0, 1, 2-3, >=4
    freq_0 = [q for q in all_ids if query_diagnostics[q]["max_gold_frequency"] == 0]
    freq_1 = [q for q in all_ids if query_diagnostics[q]["max_gold_frequency"] == 1]
    freq_2_3 = [q for q in all_ids if 2 <= query_diagnostics[q]["max_gold_frequency"] <= 3]
    freq_gte_4 = [q for q in all_ids if query_diagnostics[q]["max_gold_frequency"] >= 4]

    # 4. Nearest similarity quartiles
    sim_q1 = [q for q in all_ids if query_diagnostics[q]["max_sim"] < q25]
    sim_q2 = [q for q in all_ids if q25 <= query_diagnostics[q]["max_sim"] < q50]
    sim_q3 = [q for q in all_ids if q50 <= query_diagnostics[q]["max_sim"] < q75]
    sim_q4 = [q for q in all_ids if query_diagnostics[q]["max_sim"] >= q75]

    # 5. Queries with no useful memory support (sim < 0.65 or freq == 0)
    weak_support = [q for q in all_ids if query_diagnostics[q]["max_sim"] < 0.65 or query_diagnostics[q]["max_gold_frequency"] == 0]

    # 6. Queries with strong nearest-neighbor support (sim >= 0.85)
    strong_support = [q for q in all_ids if query_diagnostics[q]["max_sim"] >= 0.85]

    # 7. CAL qid absent from V2 cache (163826)
    absent_qid = "163826"
    absent_metrics = query_diagnostics.get(absent_qid, {})

    audit = {
        "schema_version": "dsc2026.gemini.huy_d1_lal_case_memory_v1.generalization_audit.v1",
        "total_queries": len(all_ids),
        "similarity_quartiles": {
            "q25": float(q25),
            "median": float(q50),
            "q75": float(q75),
        },
        "slices": {
            "single_gold": slice_metrics(single_qids),
            "multi_gold": slice_metrics(multi_qids),
            "block_d": slice_metrics(block_d_qids),
            "gold_frequency_in_memory": {
                "freq_0": slice_metrics(freq_0),
                "freq_1": slice_metrics(freq_1),
                "freq_2_to_3": slice_metrics(freq_2_3),
                "freq_gte_4": slice_metrics(freq_gte_4),
            },
            "nearest_similarity_quartiles": {
                "quartile_1": slice_metrics(sim_q1),
                "quartile_2": slice_metrics(sim_q2),
                "quartile_3": slice_metrics(sim_q3),
                "quartile_4": slice_metrics(sim_q4),
            },
            "no_useful_memory_support": slice_metrics(weak_support),
            "strong_nearest_neighbor_support": slice_metrics(strong_support),
        },
        "cal_qid_absent_from_v2_cache": {
            "qid": absent_qid,
            "metrics": absent_metrics,
        },
    }

    out_path = RESULTS_DIR / "LAL_MEMORY_GENERALIZATION_AUDIT.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)
    print(f"Wrote {out_path}", flush=True)
    return audit


if __name__ == "__main__":
    run_generalization_audit()
