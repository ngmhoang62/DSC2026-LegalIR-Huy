"""Phase A: Cheap Fusion Geometry Retune for provision_reranker_v1 (Parallelized).

Evaluates C in {0.01, 0.03, 0.10, 0.15, 0.30, 1.0} and k in {2, 5, 10, 20, 40}.
Performs strict nested CV to select (C, k) without held-fold leakage.
Evaluates both nested-retuned OOF and full 30-config global OOF grid landscape.
Verifies exact parity against authoritative baseline at (C=0.15, k=10).
Writes FUSION_GEOMETRY_REPORT.json.
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Ensure stdout uses UTF-8
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from common import (
    CURRENT_DIR,
    EXPECTED_METRICS,
    GIT_COMMIT_SHA,
    REPO_ROOT,
    RESULTS_DIR,
    WORKSPACE_ROOT,
    core,
    evaluate_predictions,
    load_baseline_data,
)

CACHE_DIR = RESULTS_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

C_VALUES = [0.01, 0.03, 0.10, 0.15, 0.30, 1.0]
K_VALUES = [2, 5, 10, 20, 40]

sys.path.insert(0, str(WORKSPACE_ROOT / "LegalIR/scripts"))
from tune_burst_supervised_profile_bm25 import build_profiles, profile_rank
from exp_final_memory_ltr_probe import MEMORY_NAMES, memory_features, support_index


def normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)


def transform_k(row_mat: np.ndarray, k_val: float) -> np.ndarray:
    """Transform 7 reciprocal rank views using offset k.
    
    col 2*i: 1 / (k + rank)
    col 2*i+1: rank / 60.0
    """
    if k_val == 10.0:
        return row_mat
    out = row_mat.copy()
    for i in range(7):
        r = np.round(out[:, 2 * i + 1] * 60.0)
        out[:, 2 * i] = (1.0 / (k_val + r)).astype(np.float32)
    return out


def fit_and_predict_subset(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_rows_dict: Dict[str, np.ndarray],
    pools: Dict[str, List[str]],
    c_val: float,
    k_val: float,
) -> Dict[str, List[str]]:
    """Fit scaler + LogisticRegression and predict sorted doc orders."""
    tx = transform_k(train_x, k_val)
    scaler = StandardScaler().fit(tx)
    scaled_tx = scaler.transform(tx)
    
    model = LogisticRegression(
        C=c_val,
        class_weight="balanced",
        solver="liblinear",
        max_iter=3000,
        random_state=2026,
    ).fit(scaled_tx, train_y)
    
    predictions = {}
    for qid, q_mat in test_rows_dict.items():
        q_transformed = transform_k(q_mat, k_val)
        vals = model.decision_function(scaler.transform(q_transformed))
        pool = pools[qid]
        predictions[qid] = [pool[i] for i in np.lexsort((np.asarray(pool), -vals))]
    return predictions


def eval_inner_config(
    c_val: float,
    k_val: int,
    inner_folds_list: List[str],
    inner_train_xs: Dict[str, np.ndarray],
    inner_train_ys: Dict[str, np.ndarray],
    inner_test_rows_dict: Dict[str, Dict[str, np.ndarray]],
    pools: Dict[str, List[str]],
    golds: Dict[str, Set[str]],
    train_ids: List[str],
    fold_for: Dict[str, str],
) -> Tuple[Tuple[float, int], Dict[str, float]]:
    inner_r5_list = []
    inner_p5_list = []
    for inner_test_fold in inner_folds_list:
        inner_preds = fit_and_predict_subset(
            inner_train_xs[inner_test_fold],
            inner_train_ys[inner_test_fold],
            inner_test_rows_dict[inner_test_fold],
            pools,
            c_val=c_val,
            k_val=float(k_val),
        )
        inner_test_qids = [q for q in train_ids if fold_for[q] == inner_test_fold]
        r5_vals = [len(set(inner_preds[q][:5]) & golds[q]) / len(golds[q]) for q in inner_test_qids]
        p5_vals = [len(set(inner_preds[q][:5]) & golds[q]) / 5.0 for q in inner_test_qids]
        inner_r5_list.append(np.mean(r5_vals))
        inner_p5_list.append(np.mean(p5_vals))

    return (c_val, k_val), {
        "mean_recall_at_5": float(np.mean(inner_r5_list)),
        "mean_precision_at_5": float(np.mean(inner_p5_list)),
    }


def eval_outer_grid_config(
    c_val: float,
    k_val: int,
    outer_train_x_base: np.ndarray,
    outer_train_y: np.ndarray,
    outer_test_rows: Dict[str, np.ndarray],
    pools: Dict[str, List[str]],
) -> Tuple[Tuple[float, int], Dict[str, List[str]]]:
    preds = fit_and_predict_subset(
        outer_train_x_base, outer_train_y, outer_test_rows, pools, c_val=c_val, k_val=float(k_val)
    )
    return (c_val, k_val), preds


def build_fold_features(
    outer: str,
    folds: Dict[str, List[str]],
    fold_for: Dict[str, str],
    pools: Dict[str, List[str]],
    questions: Dict[str, str],
    golds: Dict[str, Set[str]],
    dup: Dict[str, List[str]],
    similarities: np.ndarray,
    row_map: Dict[str, int],
    frozen_rank: Dict[str, Dict[str, np.ndarray]],
    frozen_score: Dict[str, Dict[str, np.ndarray]],
    frozen_meta: Dict[str, Dict[str, np.ndarray]],
    config: Dict[str, Any],
) -> Dict[str, np.ndarray]:
    """Build outer-fold cross-fitted feature rows with exact baseline semantics."""
    cache_file = CACHE_DIR / f"features_{outer}.npz"
    if cache_file.exists():
        print(f"Loading cached outer fold features for {outer} from {cache_file.name}...", flush=True)
        with np.load(cache_file) as z:
            return {f.replace("row_", ""): z[f].astype(np.float32) for f in z.files}

    print(f"Building outer fold cross-fitted features for {outer}...", flush=True)
    started = time.perf_counter()
    all_qids = set(pools)
    test_ids = folds[outer]
    blocked = set(map(str, dup.get(outer, [])))
    train_ids = sorted(all_qids - set(test_ids) - blocked, key=int)
    queries = {qid: (questions[qid], golds[qid]) for qid in pools}

    profile_orders = {}
    memory_rows = {}
    test_profile_model = build_profiles(queries, train_ids)
    test_by_doc, test_frequency = support_index(golds, train_ids)
    train_index = [row_map[qid] for qid in train_ids]
    
    for qid in test_ids:
        profile_orders[qid] = profile_rank(questions[qid], test_profile_model, 2, 1.2, 0.75, 0.3)
        memory_rows[qid] = memory_features(
            similarities[row_map[qid], train_index],
            pools[qid],
            train_ids,
            golds,
            test_by_doc,
            test_frequency,
        )

    for inner in folds:
        if inner == outer:
            continue
        inner_ids = [qid for qid in train_ids if fold_for[qid] == inner]
        inner_blocked = set(map(str, dup.get(inner, [])))
        memory_ids = [qid for qid in train_ids if fold_for[qid] != inner and qid not in inner_blocked]
        profile_model = build_profiles(queries, memory_ids)
        by_doc, frequency = support_index(golds, memory_ids)
        memory_index = [row_map[qid] for qid in memory_ids]
        for qid in inner_ids:
            profile_orders[qid] = profile_rank(questions[qid], profile_model, 2, 1.2, 0.75, 0.3)
            memory_rows[qid] = memory_features(
                similarities[row_map[qid], memory_index],
                pools[qid],
                memory_ids,
                golds,
                by_doc,
                frequency,
            )

    local_ids = train_ids + list(test_ids)
    local_pools = {qid: pools[qid] for qid in local_ids}
    rank_features = {name: {qid: frozen_rank[name][qid] for qid in local_ids} for name in frozen_rank}
    rank_features["huy_profile"] = core.rank_columns(profile_orders, local_pools)
    score_features = {name: {qid: frozen_score[name][qid] for qid in local_ids} for name in frozen_score}
    metadata = {name: {qid: value[qid] for qid in local_ids} for name, value in frozen_meta.items()}
    metadata["lal_memory"] = memory_rows

    rows = core.make_rows(config, local_pools, rank_features, score_features, metadata)
    
    # Save cache
    np.savez_compressed(cache_file, **{f"row_{qid}": rows[qid] for qid in rows})
    elapsed = time.perf_counter() - started
    print(f"Outer fold {outer} features computed and cached in {elapsed:.1f}s", flush=True)
    return rows


def run_phase_a():
    total_start = time.perf_counter()
    print("=" * 70, flush=True)
    print("PHASE A: CHEAP FUSION GEOMETRY RETUNE (PARALLEL 8 CORES)", flush=True)
    print("=" * 70, flush=True)

    folds, fold_for, pools, questions, golds, e5_orders, e5_scores, dup, base_orders, base_scores = load_baseline_data()
    all_qids = set(pools)

    # 1. Load frozen channels
    print("Loading upstream channels...", flush=True)
    jina_order, jina_scores, _ = core.load_jina(pools)
    lal_order, lal_scores, _ = core.load_source_channel("lal", pools)
    legalir_jina_order, _, _ = core.load_source_channel("jina", pools)
    bm25_order, bm25_scores, _ = core.load_source_channel("bm25", pools)
    trigram_order, trigram_scores, _ = core.load_source_channel("trigram", pools)
    heads = core.document_heads(pools)
    doctype, citation = core.metadata_arrays(pools, questions, heads)

    # 2. LAL query embeddings & similarities
    lal_path = core.WORKSPACE / "LegalIR/cache/exp109b_encoder_complementarity/embeddings/vnlegal_lal/queries.npz"
    with np.load(lal_path, allow_pickle=False) as payload:
        embedding_ids = list(map(str, payload["query_ids"].tolist()))
        embedding_values = normalize(payload["vectors"])
    source_row = {qid: i for i, qid in enumerate(embedding_ids)}
    ordered_qids = sorted(pools, key=int)
    vectors = embedding_values[[source_row[qid] for qid in ordered_qids]]
    row_map = {qid: i for i, qid in enumerate(ordered_qids)}
    similarities = np.asarray(vectors @ vectors.T, dtype=np.float32)

    base_orders_map = {
        "jina_ce": jina_order,
        "adapted_e5": e5_orders["adapted_e5"],
        "lal_native": lal_order,
        "legalir_jina": legalir_jina_order,
        "legalir_bm25": bm25_order,
        "legalir_trigram": trigram_order,
    }
    base_scores_map = {
        "jina_ce": jina_scores,
        "adapted_e5": e5_scores["adapted_e5"],
        "lal_native": lal_scores,
        "legalir_bm25": bm25_scores,
        "legalir_trigram": trigram_scores,
    }
    frozen_rank = {name: core.rank_columns(value, pools) for name, value in base_orders_map.items()}
    frozen_score = {name: core.score_columns(value, pools) for name, value in base_scores_map.items()}
    frozen_meta = {"doctype": doctype, "citation": citation}
    base_rank = ["jina_ce", "adapted_e5", "lal_native", "legalir_jina", "huy_profile"]
    base_score = ["jina_ce", "adapted_e5", "lal_native"]
    config = dict(
        rank_views=base_rank + ["legalir_bm25", "legalir_trigram"],
        score_channels=base_score + ["legalir_bm25", "legalir_trigram"],
        metadata=["citation", "lal_memory"],
    )

    # 3. Build/load outer fold feature sets
    print("\n--- Precomputing/loading outer fold features ---", flush=True)
    fold_features: Dict[str, Dict[str, np.ndarray]] = {}
    for outer in folds:
        fold_features[outer] = build_fold_features(
            outer, folds, fold_for, pools, questions, golds, dup,
            similarities, row_map, frozen_rank, frozen_score, frozen_meta, config
        )

    # 4. Nested CV and Global Grid Evaluation
    print("\n--- Executing Strict Nested CV & Global Grid Evaluation ---", flush=True)
    nested_predictions: Dict[str, List[str]] = {}
    legacy_predictions: Dict[str, List[str]] = {}
    global_predictions: Dict[Tuple[float, int], Dict[str, List[str]]] = defaultdict(dict)
    nested_selections: Dict[str, Dict[str, Any]] = {}

    for outer in sorted(folds):
        f_start = time.perf_counter()
        print(f"\nProcessing Outer Fold [{outer}]...", flush=True)
        test_ids = folds[outer]
        blocked = set(map(str, dup.get(outer, [])))
        train_ids = sorted(all_qids - set(test_ids) - blocked, key=int)
        
        rows = fold_features[outer]
        outer_train_x_base = np.vstack([rows[qid] for qid in train_ids])
        outer_train_y = np.concatenate([
            np.asarray([doc in golds[qid] for doc in pools[qid]], dtype=np.int8)
            for qid in train_ids
        ])
        outer_test_rows = {qid: rows[qid] for qid in test_ids}

        # First, verify baseline parity on outer fold at (C=0.15, k=10)
        p_check = fit_and_predict_subset(
            outer_train_x_base, outer_train_y, outer_test_rows, pools, c_val=0.15, k_val=10.0
        )
        for qid in test_ids:
            legacy_predictions[qid] = p_check[qid]
        hits_f = np.mean([len(set(p_check[q][:5]) & golds[q]) / len(golds[q]) for q in test_ids])
        expected_f = EXPECTED_METRICS["per_fold_recall_at_5"][outer]
        diff_f = abs(hits_f - expected_f)
        print(f"[{outer}] Baseline Check (C=0.15, k=10) Recall@5 = {hits_f:.16f} (expected: {expected_f:.16f}, diff: {diff_f:.2e})", flush=True)
        assert diff_f < 1e-9, f"Baseline parity mismatch on outer fold {outer}: {diff_f}"

        # Inner CV over remaining 4 folds
        inner_folds_list = [f for f in sorted(folds) if f != outer]
        inner_train_xs = {}
        inner_train_ys = {}
        inner_test_rows_dict = {}

        for inner_test_fold in inner_folds_list:
            inner_test_qids = [q for q in train_ids if fold_for[q] == inner_test_fold]
            inner_blocked = set(map(str, dup.get(inner_test_fold, [])))
            inner_train_qids = [q for q in train_ids if fold_for[q] != inner_test_fold and q not in inner_blocked]

            inner_train_xs[inner_test_fold] = np.vstack([rows[q] for q in inner_train_qids])
            inner_train_ys[inner_test_fold] = np.concatenate([
                np.asarray([doc in golds[q] for doc in pools[q]], dtype=np.int8)
                for q in inner_train_qids
            ])
            inner_test_rows_dict[inner_test_fold] = {q: rows[q] for q in inner_test_qids}

        # Parallel Grid search over C and k using inner CV (8 workers)
        print(f"[{outer}] Running parallel inner CV across 30 configs on 8 workers...", flush=True)
        inner_results = joblib.Parallel(n_jobs=8, batch_size=1)(
            joblib.delayed(eval_inner_config)(
                c_val, k_val, inner_folds_list, inner_train_xs, inner_train_ys,
                inner_test_rows_dict, pools, golds, train_ids, fold_for
            )
            for k_val in K_VALUES
            for c_val in C_VALUES
        )
        grid_inner_scores = dict(inner_results)

        # Select best (C, k) based strictly on inner CV
        def selection_key(item):
            (c_val, k_val), sc = item
            return (
                round(sc["mean_recall_at_5"], 8),
                round(sc["mean_precision_at_5"], 8),
                -abs(k_val - 10),
                -abs(c_val - 0.15),
            )

        best_config, best_score = max(grid_inner_scores.items(), key=selection_key)
        best_c, best_k = best_config
        print(f"[{outer}] Selected (C={best_c}, k={best_k}) via Inner CV: Inner R@5={best_score['mean_recall_at_5']:.6f}, P@5={best_score['mean_precision_at_5']:.6f}", flush=True)

        # Refit on entire outer training set with selected (C, k)
        nested_preds = fit_and_predict_subset(
            outer_train_x_base, outer_train_y, outer_test_rows, pools, c_val=best_c, k_val=float(best_k)
        )
        for qid in test_ids:
            nested_predictions[qid] = nested_preds[qid]
        nested_held_r5 = np.mean([len(set(nested_preds[q][:5]) & golds[q]) / len(golds[q]) for q in test_ids])
        print(f"[{outer}] Outer Held Evaluation with Selected Config: Recall@5 = {nested_held_r5:.6f} (baseline: {hits_f:.6f}, delta: {nested_held_r5 - hits_f:+.6f})", flush=True)

        nested_selections[outer] = {
            "selected_C": best_c,
            "selected_k": best_k,
            "inner_mean_recall_at_5": best_score["mean_recall_at_5"],
            "inner_mean_precision_at_5": best_score["mean_precision_at_5"],
            "outer_held_recall_at_5": float(nested_held_r5),
            "outer_legacy_recall_at_5": float(hits_f),
            "delta_recall_at_5": float(nested_held_r5 - hits_f),
        }

        # Also populate global grid predictions for outer test queries in parallel
        print(f"[{outer}] Computing global grid landscape across 30 configs on 8 workers...", flush=True)
        outer_grid_results = joblib.Parallel(n_jobs=8, batch_size=1)(
            joblib.delayed(eval_outer_grid_config)(
                c_val, k_val, outer_train_x_base, outer_train_y, outer_test_rows, pools
            )
            for k_val in K_VALUES
            for c_val in C_VALUES
        )
        for (c_val, k_val), g_preds in outer_grid_results:
            for qid in test_ids:
                global_predictions[(c_val, k_val)][qid] = g_preds[qid]

        elapsed_f = time.perf_counter() - f_start
        print(f"[{outer}] Completed in {elapsed_f:.1f}s", flush=True)

    # 5. Full 5-fold OOF Metrics Compilation
    print("\n" + "=" * 70, flush=True)
    print("EVALUATION & VERIFICATION", flush=True)
    print("=" * 70, flush=True)

    legacy_metrics = evaluate_predictions(legacy_predictions, golds, folds)
    nested_metrics = evaluate_predictions(nested_predictions, golds, folds)

    print(f"Authoritative Legacy Metrics (asserting exact baseline parity):")
    print(json.dumps(legacy_metrics, indent=2))
    
    # Assert exact baseline parity across all metrics
    assert abs(legacy_metrics["recall_at_5"] - EXPECTED_METRICS["recall_at_5"]) < 1e-9
    assert abs(legacy_metrics["precision_at_5"] - EXPECTED_METRICS["precision_at_5"]) < 1e-9
    assert abs(legacy_metrics["single_gold_recall_at_5"] - EXPECTED_METRICS["single_gold_recall_at_5"]) < 1e-9
    assert abs(legacy_metrics["multi_gold_recall_at_5"] - EXPECTED_METRICS["multi_gold_recall_at_5"]) < 1e-9

    print("\nStrict Nested-Retuned 5-Fold OOF Metrics:")
    print(json.dumps(nested_metrics, indent=2))
    nested_delta = nested_metrics["recall_at_5"] - legacy_metrics["recall_at_5"]
    print(f"\nNested OOF Recall@5 Delta vs Baseline: {nested_delta:+.16f}")

    # 6. Global Landscape Analysis
    print("\nGlobal 30-Configuration OOF Grid Landscape:")
    landscape_records = []
    for c_val in C_VALUES:
        for k_val in K_VALUES:
            preds_ck = global_predictions[(c_val, k_val)]
            m_ck = evaluate_predictions(preds_ck, golds, folds)
            delta_ck = m_ck["recall_at_5"] - legacy_metrics["recall_at_5"]
            landscape_records.append({
                "C": c_val,
                "k": k_val,
                "recall_at_5": m_ck["recall_at_5"],
                "precision_at_5": m_ck["precision_at_5"],
                "single_gold_recall_at_5": m_ck["single_gold_recall_at_5"],
                "multi_gold_recall_at_5": m_ck["multi_gold_recall_at_5"],
                "per_fold_recall_at_5": m_ck["per_fold_recall_at_5"],
                "delta_vs_legacy": delta_ck,
            })
            print(f"C={c_val:4.2f}, k={k_val:2d} | R@5={m_ck['recall_at_5']:.6f} (delta: {delta_ck:+.6f}) | P@5={m_ck['precision_at_5']:.6f} | Single={m_ck['single_gold_recall_at_5']:.6f} | Multi={m_ck['multi_gold_recall_at_5']:.6f}")

    # Find globally best OOF config (for diagnostic reporting only)
    best_global = max(landscape_records, key=lambda r: (r["recall_at_5"], r["precision_at_5"]))
    print(f"\nGlobally Best OOF Config (DIAGNOSTIC ONLY): C={best_global['C']}, k={best_global['k']} -> Recall@5={best_global['recall_at_5']:.6f} (delta: {best_global['delta_vs_legacy']:+.6f})")

    # Determine FUSION_ANCHOR
    fusion_anchor_defined = bool(nested_metrics["recall_at_5"] > legacy_metrics["recall_at_5"])
    if fusion_anchor_defined:
        print(f"\n>>> FUSION_ANCHOR DEFINED! Nested retuning improves baseline from {legacy_metrics['recall_at_5']:.6f} to {nested_metrics['recall_at_5']:.6f} (delta: {nested_delta:+.6f})")
    else:
        print(f"\n>>> FUSION_ANCHOR NOT PROMOTED. Authoritative baseline remains baseline anchor (delta: {nested_delta:+.6f})")

    # Write FUSION_GEOMETRY_REPORT.json
    report_data = {
        "schema_version": "dsc2026.gemini.provision_reranker_v1.fusion_geometry.v1",
        "git_commit_sha": GIT_COMMIT_SHA,
        "authoritative_baseline_metrics": legacy_metrics,
        "legacy_setting": {"C": 0.15, "k": 10},
        "nested_selection_per_fold": nested_selections,
        "nested_retuned_metrics": nested_metrics,
        "nested_delta_vs_baseline": nested_delta,
        "fusion_anchor_defined": fusion_anchor_defined,
        "diagnostic_best_global_config": {
            "C": best_global["C"],
            "k": best_global["k"],
            "recall_at_5": best_global["recall_at_5"],
            "precision_at_5": best_global["precision_at_5"],
            "delta_vs_baseline": best_global["delta_vs_legacy"],
        },
        "global_landscape": landscape_records,
        "execution_seconds": time.perf_counter() - total_start,
    }

    report_path = RESULTS_DIR / "FUSION_GEOMETRY_REPORT.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    print(f"\nReport written to {report_path}", flush=True)


if __name__ == "__main__":
    run_phase_a()
