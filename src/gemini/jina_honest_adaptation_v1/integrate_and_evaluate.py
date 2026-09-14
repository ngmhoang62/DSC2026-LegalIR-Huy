"""Authoritative feature integration and strict 5-fold OOF evaluation for jina_honest_adaptation_v1.
Evaluates J0 (Baseline 44D), J1_REPLACE, J2_AUGMENT, J3_RESIDUAL, J4_ENSEMBLE.
Produces results/gemini/jina_honest_adaptation_v1/FINAL_INTEGRATION_REPORT.json.
"""

from __future__ import annotations

import json
import sqlite3
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from common import (
    EXP_RESULTS,
    REPO_ROOT,
    get_git_info,
    log_execution_trace,
    sha256_file,
)

sys.path.insert(0, str(REPO_ROOT / "src/huy_fasttrack"))
import run_huy_5fold_fasttrack as core

AUTHORITATIVE_BASELINE_R5 = 0.9488556715777428
AUTHORITATIVE_BASELINE_P5 = 0.20314690316120732
AUTHORITATIVE_SINGLE_R5 = 0.9641304347826087
AUTHORITATIVE_MULTI_R5 = 0.7703266787658803


def rank_columns(order_map: Dict[str, List[str]], pools: Dict[str, List[str]]) -> Dict[str, np.ndarray]:
    """Compute rank features: reciprocal rank and normalized rank."""
    out = {}
    for qid, docs in pools.items():
        order = order_map.get(qid, [])
        ranks = {doc: i + 1 for i, doc in enumerate(order)}
        values = np.asarray([ranks.get(doc, 60) for doc in docs], dtype=np.float32)
        out[qid] = np.column_stack((1.0 / (10.0 + values), values / 60.0)).astype(np.float32)
    return out


def score_columns(score_map: Dict[str, Dict[str, float]], pools: Dict[str, List[str]]) -> Dict[str, np.ndarray]:
    """Compute standardized score features: within-query z-score and margin-to-top."""
    out = {}
    for qid, docs in pools.items():
        raw = score_map.get(qid, {})
        values = np.asarray([raw.get(doc, np.nan) for doc in docs], dtype=np.float64)
        present = values[~np.isnan(values)]
        if present.size:
            mean = float(present.mean())
            std = float(present.std()) or 1.0
            top = float(present.max())
        else:
            mean, std, top = 0.0, 1.0, 0.0
        filled = np.where(np.isnan(values), mean - 2 * std, values)
        out[qid] = np.column_stack(((filled - mean) / std, (filled - top) / std)).astype(np.float32)
    return out


def compute_metrics(predictions: Dict[str, List[str]], golds: Dict[str, Set[str]], folds: Dict[str, List[str]]) -> Dict[str, Any]:
    values, precisions = [], []
    single, multi = [], []
    by_fold = {}
    for fold, qids in folds.items():
        fold_values = []
        for qid in qids:
            gold = golds[qid]
            top5 = predictions[qid][:5]
            hits = len(set(top5) & gold)
            val = hits / len(gold)
            values.append(val)
            precisions.append(hits / 5.0)
            fold_values.append(val)
            if len(gold) == 1:
                single.append(val)
            else:
                multi.append(val)
        by_fold[fold] = float(np.mean(fold_values))

    return {
        "queries": len(values),
        "recall_at_5": float(np.mean(values)),
        "precision_at_5": float(np.mean(precisions)),
        "single_gold_recall_at_5": float(np.mean(single)),
        "multi_gold_recall_at_5": float(np.mean(multi)),
        "per_fold_recall_at_5": by_fold,
    }


def compare_predictions(cand_preds: Dict[str, List[str]], ref_preds: Dict[str, List[str]], golds: Dict[str, Set[str]]) -> Dict[str, Any]:
    wins = losses = ties = 0
    all_qids = [q for q in cand_preds if q in golds and golds[q]]
    for q in all_qids:
        gold = golds[q]
        r_ref = len(set(ref_preds[q][:5]) & gold) / len(gold)
        r_cand = len(set(cand_preds[q][:5]) & gold) / len(gold)
        if r_cand > r_ref:
            wins += 1
        elif r_cand < r_ref:
            losses += 1
        else:
            ties += 1
    return {"wins": wins, "losses": losses, "ties": ties, "win_minus_loss": wins - losses}


def run_integration_pipeline(
    adapted_jina_scores: Dict[str, Dict[str, float]],
    adapted_jina_orders: Dict[str, List[str]],
    ensemble_scores: Optional[Dict[str, Dict[str, float]]] = None,
    ensemble_orders: Optional[Dict[str, List[str]]] = None,
) -> Dict[str, Any]:
    start_time = time.perf_counter()

    # Load baseline fasttrack components
    print("Loading baseline fasttrack environment...", flush=True)
    folds, pools, questions, golds, e5_orders, e5_scores, dup_exclusions, pred_hashes = core.load_inputs()
    metadata = core.load_metadata(pools)

    # Base channels
    rank_features: Dict[str, Dict[str, np.ndarray]] = {}
    score_features: Dict[str, Dict[str, np.ndarray]] = {}

    for name in core.CHANNEL_SPECS:
        order_map, raw_scores, _ = core.load_source_channel(name, pools)
        rank_features[name] = rank_columns(order_map, pools)
        score_features[name] = score_columns(raw_scores, pools)

    for name in ("frozen_e5", "adapted_e5"):
        rank_features[name] = rank_columns(e5_orders[name], pools)
        score_features[name] = score_columns(e5_scores[name], pools)

    # Frozen Jina
    jina_orders, jina_scores, _ = core.load_jina(pools)
    rank_features["frozen_jina"] = rank_columns(jina_orders, pools)
    score_features["frozen_jina"] = score_columns(jina_scores, pools)

    # Adapted Jina
    rank_features["adapted_jina"] = rank_columns(adapted_jina_orders, pools)
    score_features["adapted_jina"] = score_columns(adapted_jina_scores, pools)

    # Residual features (Section 20 J3_RESIDUAL)
    # adapted_z - frozen_z, and adapted_rank - frozen_rank
    residual_features = {}
    for qid, docs in pools.items():
        f_z = score_features["frozen_jina"][qid][:, 0]
        a_z = score_features["adapted_jina"][qid][:, 0]
        f_r = rank_features["frozen_jina"][qid][:, 1] * 60.0
        a_r = rank_features["adapted_jina"][qid][:, 1] * 60.0
        residual_features[qid] = np.column_stack((a_z - f_z, a_r - f_r)).astype(np.float32)

    # 3-seed Ensemble channel if provided
    if ensemble_scores and ensemble_orders:
        rank_features["ensemble_jina"] = rank_columns(ensemble_orders, pools)
        score_features["ensemble_jina"] = score_columns(ensemble_scores, pools)

    # Define Configurations
    # Baseline J0 config (exact 44D authoritative endpoint)
    base_config = core.make_config(
        ("frozen_e5", "adapted_e5", "bge_m3_dense", "monot5_reranker", "bge_reranker_large", "frozen_jina"),
        ("frozen_e5", "adapted_e5", "bge_m3_dense", "monot5_reranker", "bge_reranker_large", "frozen_jina"),
        ("document_types", "citation_count"),
    )

    # J1_REPLACE: Replace frozen_jina with adapted_jina
    j1_config = core.make_config(
        ("frozen_e5", "adapted_e5", "bge_m3_dense", "monot5_reranker", "bge_reranker_large", "adapted_jina"),
        ("frozen_e5", "adapted_e5", "bge_m3_dense", "monot5_reranker", "bge_reranker_large", "adapted_jina"),
        ("document_types", "citation_count"),
    )

    # J2_AUGMENT: Baseline + adapted_jina rank & score
    j2_config = core.make_config(
        ("frozen_e5", "adapted_e5", "bge_m3_dense", "monot5_reranker", "bge_reranker_large", "frozen_jina", "adapted_jina"),
        ("frozen_e5", "adapted_e5", "bge_m3_dense", "monot5_reranker", "bge_reranker_large", "frozen_jina", "adapted_jina"),
        ("document_types", "citation_count"),
    )

    configurations = [
        ("J0_BASELINE", base_config, {}),
        ("J1_REPLACE", j1_config, {}),
        ("J2_AUGMENT", j2_config, {}),
    ]

    all_qids = set(pools)
    results = {}
    all_predictions = {}

    for name, cfg, extra_dict in configurations:
        print(f"Evaluating {name}...", flush=True)
        # Build rows
        rows = core.make_rows(cfg, pools, rank_features, score_features, metadata)
        # If J3, add residual
        if name == "J3_RESIDUAL":
            for q in pools:
                rows[q] = np.column_stack((rows[q], residual_features[q])).astype(np.float32)

        predictions = {}
        for fold, test_ids in folds.items():
            held = set(test_ids)
            blocked = set(map(str, dup_exclusions.get(fold, [])))
            train_ids = sorted(all_qids - held - blocked, key=int)
            x_train = np.vstack([rows[q] for q in train_ids])
            y_train = np.concatenate([
                np.asarray([doc in golds[q] for doc in pools[q]], dtype=np.int8)
                for q in train_ids
            ])
            scaler = StandardScaler().fit(x_train)
            model = LogisticRegression(
                C=0.15, class_weight="balanced", solver="liblinear",
                max_iter=3000, random_state=2026,
            ).fit(scaler.transform(x_train), y_train)

            for qid in test_ids:
                vals = model.decision_function(scaler.transform(rows[qid]))
                predictions[qid] = [
                    pools[qid][i] for i in np.lexsort((np.asarray(pools[qid]), -vals))
                ]

        metric_res = compute_metrics(predictions, golds, folds)
        delta_r5 = metric_res["recall_at_5"] - AUTHORITATIVE_BASELINE_R5
        comp = compare_predictions(predictions, all_predictions.get("J0_BASELINE", predictions), golds)

        results[name] = {
            "metrics": metric_res,
            "delta_vs_authoritative_baseline": delta_r5,
            "single_gold_delta": metric_res["single_gold_recall_at_5"] - AUTHORITATIVE_SINGLE_R5,
            "multi_gold_delta": metric_res["multi_gold_recall_at_5"] - AUTHORITATIVE_MULTI_R5,
            "per_fold_delta": {
                f: metric_res["per_fold_recall_at_5"][f] - core.EXPECTED_OOF_BY_FOLD[f]
                for f in sorted(folds.keys())
            } if hasattr(core, "EXPECTED_OOF_BY_FOLD") else {},
            "comparison_vs_baseline": comp,
            "feature_dim": rows[list(pools.keys())[0]].shape[1],
        }
        all_predictions[name] = predictions
        print(f"  {name}: R@5 = {metric_res['recall_at_5']:.8f} (delta = {delta_r5:+.8f})")

    # Evaluate J3_RESIDUAL
    print("Evaluating J3_RESIDUAL...", flush=True)
    rows_j3 = core.make_rows(j2_config, pools, rank_features, score_features, metadata)
    for q in pools:
        rows_j3[q] = np.column_stack((rows_j3[q], residual_features[q])).astype(np.float32)

    predictions_j3 = {}
    for fold, test_ids in folds.items():
        held = set(test_ids)
        blocked = set(map(str, dup_exclusions.get(fold, [])))
        train_ids = sorted(all_qids - held - blocked, key=int)
        x_train = np.vstack([rows_j3[q] for q in train_ids])
        y_train = np.concatenate([
            np.asarray([doc in golds[q] for doc in pools[q]], dtype=np.int8)
            for q in train_ids
        ])
        scaler = StandardScaler().fit(x_train)
        model = LogisticRegression(
            C=0.15, class_weight="balanced", solver="liblinear",
            max_iter=3000, random_state=2026,
        ).fit(scaler.transform(x_train), y_train)

        for qid in test_ids:
            vals = model.decision_function(scaler.transform(rows_j3[qid]))
            predictions_j3[qid] = [
                pools[qid][i] for i in np.lexsort((np.asarray(pools[qid]), -vals))
            ]

    metric_j3 = compute_metrics(predictions_j3, golds, folds)
    delta_j3 = metric_j3["recall_at_5"] - AUTHORITATIVE_BASELINE_R5
    comp_j3 = compare_predictions(predictions_j3, all_predictions["J0_BASELINE"], golds)
    results["J3_RESIDUAL"] = {
        "metrics": metric_j3,
        "delta_vs_authoritative_baseline": delta_j3,
        "single_gold_delta": metric_j3["single_gold_recall_at_5"] - AUTHORITATIVE_SINGLE_R5,
        "multi_gold_delta": metric_j3["multi_gold_recall_at_5"] - AUTHORITATIVE_MULTI_R5,
        "comparison_vs_baseline": comp_j3,
        "feature_dim": rows_j3[list(pools.keys())[0]].shape[1],
    }
    all_predictions["J3_RESIDUAL"] = predictions_j3
    print(f"  J3_RESIDUAL: R@5 = {metric_j3['recall_at_5']:.8f} (delta = {delta_j3:+.8f})")

    # J4_ENSEMBLE if available
    if ensemble_scores and ensemble_orders:
        print("Evaluating J4_ENSEMBLE...", flush=True)
        j4_config = core.make_config(
            ("frozen_e5", "adapted_e5", "bge_m3_dense", "monot5_reranker", "bge_reranker_large", "ensemble_jina"),
            ("frozen_e5", "adapted_e5", "bge_m3_dense", "monot5_reranker", "bge_reranker_large", "ensemble_jina"),
            ("document_types", "citation_count"),
        )
        rows_j4 = core.make_rows(j4_config, pools, rank_features, score_features, metadata)
        predictions_j4 = {}
        for fold, test_ids in folds.items():
            held = set(test_ids)
            blocked = set(map(str, dup_exclusions.get(fold, [])))
            train_ids = sorted(all_qids - held - blocked, key=int)
            x_train = np.vstack([rows_j4[q] for q in train_ids])
            y_train = np.concatenate([
                np.asarray([doc in golds[q] for doc in pools[q]], dtype=np.int8)
                for q in train_ids
            ])
            scaler = StandardScaler().fit(x_train)
            model = LogisticRegression(
                C=0.15, class_weight="balanced", solver="liblinear",
                max_iter=3000, random_state=2026,
            ).fit(scaler.transform(x_train), y_train)

            for qid in test_ids:
                vals = model.decision_function(scaler.transform(rows_j4[qid]))
                predictions_j4[qid] = [
                    pools[qid][i] for i in np.lexsort((np.asarray(pools[qid]), -vals))
                ]
        metric_j4 = compute_metrics(predictions_j4, golds, folds)
        delta_j4 = metric_j4["recall_at_5"] - AUTHORITATIVE_BASELINE_R5
        comp_j4 = compare_predictions(predictions_j4, all_predictions["J0_BASELINE"], golds)
        results["J4_ENSEMBLE"] = {
            "metrics": metric_j4,
            "delta_vs_authoritative_baseline": delta_j4,
            "single_gold_delta": metric_j4["single_gold_recall_at_5"] - AUTHORITATIVE_SINGLE_R5,
            "multi_gold_delta": metric_j4["multi_gold_recall_at_5"] - AUTHORITATIVE_MULTI_R5,
            "comparison_vs_baseline": comp_j4,
            "feature_dim": rows_j4[list(pools.keys())[0]].shape[1],
        }
        all_predictions["J4_ENSEMBLE"] = predictions_j4
        print(f"  J4_ENSEMBLE: R@5 = {metric_j4['recall_at_5']:.8f} (delta = {delta_j4:+.8f})")

    # Find best arm
    best_arm = max(results.keys(), key=lambda k: results[k]["metrics"]["recall_at_5"])
    best_delta = results[best_arm]["delta_vs_authoritative_baseline"]

    report = {
        "schema_version": "dsc2026.gemini.final_integration_report.v1",
        "status": "COMPLETE",
        "authoritative_baseline": {
            "name": "profile_memory_plus_sparse_rank_scores",
            "recall_at_5": AUTHORITATIVE_BASELINE_R5,
            "precision_at_5": AUTHORITATIVE_BASELINE_P5,
            "single_gold_recall_at_5": AUTHORITATIVE_SINGLE_R5,
            "multi_gold_recall_at_5": AUTHORITATIVE_MULTI_R5,
        },
        "best_arm": best_arm,
        "best_delta": best_delta,
        "integration_arms": results,
        "git": get_git_info(),
        "runtime_sec": round(time.perf_counter() - start_time, 3),
    }

    out_path = EXP_RESULTS / "FINAL_INTEGRATION_REPORT.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {out_path} (Best arm: {best_arm}, delta: {best_delta:+.8f})")
    return report, all_predictions
