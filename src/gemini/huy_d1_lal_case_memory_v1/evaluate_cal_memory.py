"""Dual CAL LOBO evaluation for M0, M1, M2 with nested cross-fitting."""

from __future__ import annotations

import json
import math
import pickle
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import sys
ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = ROOT / "results" / "gemini" / "huy_d1_lal_case_memory_v1"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src" / "huy_fasttrack"))
sys.path.insert(0, str(ROOT.parent / "LegalIR" / "scripts"))

from src.gemini.huy_vnlegal_rank_ablation_v1.evaluate_ablation_cal import (
    load_cal_inputs, ltr_features, D1_VIEWS
)
import run_huy_5fold_fasttrack as core
from exp_final_memory_ltr_probe import MEMORY_NAMES, memory_features, support_index
from src.gemini.huy_d1_lal_case_memory_v1.audit_data_isolation import load_duplicate_graph

LAL_QUERIES = ROOT.parent / "LegalIR" / "cache" / "exp109b_encoder_complementarity" / "embeddings" / "vnlegal_lal" / "queries.npz"

EXPECTED_D1_R5 = 0.9569444444444444
EXPECTED_BLOCKS = {
    "a": 0.975,
    "b": 0.97,
    "c": 0.995,
    "d": 0.9338888888888888,
}


def normalize(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    return array / np.maximum(norms, 1e-12)


def run_cal_lobo_evaluation() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    print("Loading CAL inputs...", flush=True)
    queries, blocks, all_ids, extended, local_views, full_channels_cv, gold, vnlegal_cv, type_rows, cite_rows = load_cal_inputs()
    folds, pools, questions, v2_golds, e5_orders, e5_scores, v2_dup, _ = core.load_inputs()
    _, _, _, dup_map = load_duplicate_graph()

    v2_population = sorted(pools.keys(), key=int)
    v2_set = set(v2_population)

    # Load LAL query embeddings
    with np.load(LAL_QUERIES, allow_pickle=False) as z:
        all_npz_qids = list(map(str, z["query_ids"].tolist()))
        all_npz_vecs = normalize(z["vectors"].astype(np.float32))

    qid_to_vec_idx = {qid: i for i, qid in enumerate(all_npz_qids)}

    # Base D1 rank + score features (32 columns)
    base_d1_rows, _ = ltr_features(
        local_views, D1_VIEWS, extended, all_ids, full_channels_cv
    )

    block_names = sorted(blocks.keys())
    arm_names = ["M0_D1_BASELINE", "M1_D1_PLUS_LAL_MEMORY", "M2_D1_PLUS_LAL_MEMORY_NO_DOCTYPE"]
    predictions: Dict[str, Dict[str, List[str]]] = {arm: {} for arm in arm_names}
    feature_dims: Dict[str, int] = {}

    for held_name in block_names:
        held_ids = blocks[held_name]
        held_set = set(held_ids)
        held_dups = {dup for q in held_set for dup in dup_map.get(q, set())}

        train_names = [b for b in block_names if b != held_name]
        train_ids = [q for b in train_names for q in blocks[b]]

        # Precompute memory features for test queries in held block: P - held - dup(held)
        test_support = sorted(v2_set - held_set - held_dups, key=int)
        test_by_doc, test_freq = support_index(v2_golds, test_support)
        test_support_indices = [qid_to_vec_idx[q] for q in test_support]
        test_support_matrix = all_npz_vecs[test_support_indices]

        test_mem_rows = {}
        for q in held_ids:
            q_vec = all_npz_vecs[qid_to_vec_idx[q]]
            sims = test_support_matrix @ q_vec
            test_mem_rows[q] = memory_features(
                sims, extended[q], test_support, v2_golds, test_by_doc, test_freq
            )

        # Precompute training memory features nested by training block T: P - held - T - dup(held) - dup(T)
        train_mem_rows = {}
        for t_name in train_names:
            t_ids = blocks[t_name]
            t_set = set(t_ids)
            t_dups = {dup for q in t_set for dup in dup_map.get(q, set())}

            t_support = sorted(v2_set - held_set - t_set - held_dups - t_dups, key=int)
            t_by_doc, t_freq = support_index(v2_golds, t_support)
            t_support_indices = [qid_to_vec_idx[q] for q in t_support]
            t_support_matrix = all_npz_vecs[t_support_indices]

            for q in t_ids:
                q_vec = all_npz_vecs[qid_to_vec_idx[q]]
                sims = t_support_matrix @ q_vec
                train_mem_rows[q] = memory_features(
                    sims, extended[q], t_support, v2_golds, t_by_doc, t_freq
                )

        y_train = np.concatenate([
            np.asarray([doc in gold[q] for doc in extended[q]], dtype=np.int8)
            for q in train_ids
        ])

        for arm in arm_names:
            if arm == "M0_D1_BASELINE":
                x_train = np.vstack([
                    np.concatenate([base_d1_rows[q], type_rows[q], cite_rows[q]], axis=1)
                    for q in train_ids
                ])
                rows_eval = {
                    q: np.concatenate([base_d1_rows[q], type_rows[q], cite_rows[q]], axis=1)
                    for q in held_ids
                }
            elif arm == "M1_D1_PLUS_LAL_MEMORY":
                x_train = np.vstack([
                    np.concatenate([base_d1_rows[q], type_rows[q], cite_rows[q], train_mem_rows[q]], axis=1)
                    for q in train_ids
                ])
                rows_eval = {
                    q: np.concatenate([base_d1_rows[q], type_rows[q], cite_rows[q], test_mem_rows[q]], axis=1)
                    for q in held_ids
                }
            elif arm == "M2_D1_PLUS_LAL_MEMORY_NO_DOCTYPE":
                x_train = np.vstack([
                    np.concatenate([base_d1_rows[q], cite_rows[q], train_mem_rows[q]], axis=1)
                    for q in train_ids
                ])
                rows_eval = {
                    q: np.concatenate([base_d1_rows[q], cite_rows[q], test_mem_rows[q]], axis=1)
                    for q in held_ids
                }

            feature_dims[arm] = x_train.shape[1]

            scaler = StandardScaler().fit(x_train)
            model = LogisticRegression(
                C=0.15, class_weight="balanced", solver="liblinear",
                max_iter=3000, random_state=2026
            ).fit(scaler.transform(x_train), y_train)

            for q in held_ids:
                docs = extended[q]
                scores = model.decision_function(scaler.transform(rows_eval[q]))
                order = np.lexsort((np.asarray(docs), -scores))
                predictions[arm][q] = [docs[i] for i in order]

    # Verify M0 parity
    m0_recalls = {q: len(set(predictions["M0_D1_BASELINE"][q][:5]) & gold[q]) / len(gold[q]) for q in all_ids}
    m0_pooled_r5 = float(np.mean(list(m0_recalls.values())))
    m0_blocks = {b: float(np.mean([m0_recalls[q] for q in blocks[b]])) for b in block_names}

    d1_parity_passed = (
        abs(m0_pooled_r5 - EXPECTED_D1_R5) < 1e-12
        and all(abs(m0_blocks[b] - EXPECTED_BLOCKS[b]) < 1e-12 for b in block_names)
    )

    baseline_parity = {
        "schema_version": "dsc2026.gemini.huy_d1_lal_case_memory_v1.d1_baseline_parity.v1",
        "status": "PASS" if d1_parity_passed else "BLOCKED_BASELINE_PARITY",
        "m0_d1_baseline": {
            "expected_pooled_recall_at_5": EXPECTED_D1_R5,
            "computed_pooled_recall_at_5": m0_pooled_r5,
            "difference": abs(m0_pooled_r5 - EXPECTED_D1_R5),
            "expected_blocks": EXPECTED_BLOCKS,
            "computed_blocks": m0_blocks,
            "feature_dim": feature_dims["M0_D1_BASELINE"],
            "expected_feature_dim": 48,
            "parity_passed": d1_parity_passed,
        },
        "all_parity_passed": d1_parity_passed,
    }

    parity_out = RESULTS_DIR / "D1_BASELINE_PARITY.json"
    with open(parity_out, "w", encoding="utf-8") as f:
        json.dump(baseline_parity, f, indent=2)
    print(f"Wrote {parity_out} (status: {baseline_parity['status']})", flush=True)

    if not d1_parity_passed:
        raise RuntimeError(f"D1 baseline parity failed! Expected {EXPECTED_D1_R5}, got {m0_pooled_r5}")

    # Compute comparative metrics for M0, M1, M2
    cal_report: Dict[str, Any] = {
        "schema_version": "dsc2026.gemini.huy_d1_lal_case_memory_v1.cal_report.v1",
        "anchor_baseline": "D1_SCORE_ONLY_VNLEGAL",
        "total_queries": len(all_ids),
        "arms": {},
    }

    per_query_recalls: Dict[str, Dict[str, float]] = {}

    for arm in arm_names:
        arm_preds = predictions[arm]
        recalls = {q: len(set(arm_preds[q][:5]) & gold[q]) / len(gold[q]) for q in all_ids}
        precisions = {q: len(set(arm_preds[q][:5]) & gold[q]) / 5.0 for q in all_ids}
        per_query_recalls[arm] = recalls

        pooled_r5 = float(np.mean(list(recalls.values())))
        precision_5 = float(np.mean(list(precisions.values())))
        arm_blocks = {b: float(np.mean([recalls[q] for q in blocks[b]])) for b in block_names}

        single_qids = [q for q in all_ids if len(gold[q]) == 1]
        multi_qids = [q for q in all_ids if len(gold[q]) > 1]
        single_r5 = float(np.mean([recalls[q] for q in single_qids]))
        multi_r5 = float(np.mean([recalls[q] for q in multi_qids]))

        # Comparison vs M0
        delta_r5 = pooled_r5 - m0_pooled_r5
        wins = sum(1 for q in all_ids if recalls[q] > m0_recalls[q])
        losses = sum(1 for q in all_ids if recalls[q] < m0_recalls[q])
        ties = sum(1 for q in all_ids if recalls[q] == m0_recalls[q])

        changed_top5 = 0
        gold_entering = 0
        gold_leaving = 0

        for q in all_ids:
            top_m0 = set(predictions["M0_D1_BASELINE"][q][:5])
            top_arm = set(arm_preds[q][:5])
            if top_m0 != top_arm:
                changed_top5 += 1
            gold_q = gold[q]
            gold_entering += len((top_arm - top_m0) & gold_q)
            gold_leaving += len((top_m0 - top_arm) & gold_q)

        distance_to_096 = 0.960000 - pooled_r5

        cal_report["arms"][arm] = {
            "feature_dim": feature_dims[arm],
            "pooled_recall_at_5": pooled_r5,
            "precision_at_5": precision_5,
            "block_recalls": arm_blocks,
            "single_gold_recall_at_5": single_r5,
            "multi_gold_recall_at_5": multi_r5,
            "delta_recall_at_5_vs_m0": delta_r5,
            "single_gold_delta_vs_m0": single_r5 - float(np.mean([m0_recalls[q] for q in single_qids])),
            "multi_gold_delta_vs_m0": multi_r5 - float(np.mean([m0_recalls[q] for q in multi_qids])),
            "wins_vs_m0": wins,
            "losses_vs_m0": losses,
            "ties_vs_m0": ties,
            "changed_top5_sets": changed_top5,
            "gold_entering_top5": gold_entering,
            "gold_leaving_top5": gold_leaving,
            "distance_to_096": distance_to_096,
        }

    report_out = RESULTS_DIR / "LAL_MEMORY_CAL_REPORT.json"
    with open(report_out, "w", encoding="utf-8") as f:
        json.dump(cal_report, f, indent=2)
    print(f"Wrote {report_out}", flush=True)

    # Write per-query predictions and deltas
    preds_out = RESULTS_DIR / "LAL_MEMORY_CAL_PREDICTIONS.jsonl"
    with open(preds_out, "w", encoding="utf-8") as f:
        for q in all_ids:
            record = {
                "qid": q,
                "gold": sorted(gold[q]),
                "m0_top5": predictions["M0_D1_BASELINE"][q][:5],
                "m0_r5": per_query_recalls["M0_D1_BASELINE"][q],
                "m1_top5": predictions["M1_D1_PLUS_LAL_MEMORY"][q][:5],
                "m1_r5": per_query_recalls["M1_D1_PLUS_LAL_MEMORY"][q],
                "m1_delta_r5": per_query_recalls["M1_D1_PLUS_LAL_MEMORY"][q] - per_query_recalls["M0_D1_BASELINE"][q],
                "m2_top5": predictions["M2_D1_PLUS_LAL_MEMORY_NO_DOCTYPE"][q][:5],
                "m2_r5": per_query_recalls["M2_D1_PLUS_LAL_MEMORY_NO_DOCTYPE"][q],
                "m2_delta_r5": per_query_recalls["M2_D1_PLUS_LAL_MEMORY_NO_DOCTYPE"][q] - per_query_recalls["M0_D1_BASELINE"][q],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Wrote {preds_out}", flush=True)

    return baseline_parity, cal_report


if __name__ == "__main__":
    run_cal_lobo_evaluation()
