"""Query-level cardinality prediction model for QCSC v1.

Trains fold-isolated Logistic Regression to predict P_multi(q) = P(|Gold(q)| > 1).
Features:
- Support neighbor cardinality features (K=32 LAL query neighbors)
- Deployment-available uncertainty features from Top-8 LR scores and expert ranks

Writes CARDINALITY_OOF_REPORT.json.
"""

from __future__ import annotations

import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from common import (
    RESULTS_DIR,
    core,
    load_baseline_data,
    load_lal_query_vectors,
)


def extract_uncertainty_features(
    qid: str,
    top8: List[str],
    scores: Dict[str, float],
    expert_ranks: Dict[str, Dict[str, int]],
) -> List[float]:
    """Extract 6 deployment-available uncertainty features over Top-8."""
    s = [scores[d] for d in top8]
    gap_1_2 = s[0] - s[1]
    gap_4_5 = s[3] - s[4]
    gap_5_6 = s[4] - s[5]
    gap_5_8 = s[4] - s[7]
    std_top8 = float(np.std(s))

    # Expert rank disagreement over Top-8
    cand_stds = []
    for d in top8:
        ranks = [expert_ranks[exp].get(d, 999) for exp in expert_ranks]
        cand_stds.append(float(np.std(ranks)))
    exp_disagreement = float(np.mean(cand_stds))

    return [gap_1_2, gap_4_5, gap_5_6, gap_5_8, std_top8, exp_disagreement]


def compute_support_cardinality_features(
    sims: np.ndarray,
    neighbor_qids: List[str],
    golds: Dict[str, Set[str]],
) -> List[float]:
    """Extract 7 support neighbor cardinality features from K=32 neighbours."""
    K = len(neighbor_qids)
    assert K > 0

    weights = np.array([math.exp(20.0 * (float(s) - 1.0)) for s in sims], dtype=np.float64)
    sum_w = float(np.sum(weights)) + 1e-12
    norm_w = weights / sum_w

    gold_lens = [len(golds[nq]) for nq in neighbor_qids]
    is_multi = [1.0 if gl > 1 else 0.0 for gl in gold_lens]

    weighted_frac_multi = float(np.sum(norm_w * is_multi))
    weighted_expected_gold = float(np.sum(norm_w * gold_lens))

    multi_sims = [float(sims[i]) for i in range(K) if is_multi[i] > 0]
    max_sim_multi = max(multi_sims) if multi_sims else 0.0
    mean_sim_multi = float(np.mean(multi_sims)) if multi_sims else 0.0

    top1_gold_count = float(gold_lens[0])

    top3_w = weights[:3]
    top3_sum_w = float(np.sum(top3_w)) + 1e-12
    top3_weighted_gold_mean = float(np.sum((top3_w / top3_sum_w) * gold_lens[:3]))

    # Gold-count entropy among neighbours
    size_weights = defaultdict(float)
    for gl, w in zip(gold_lens, norm_w):
        size_weights[gl] += float(w)
    entropy = -sum(p * math.log(max(p, 1e-12)) for p in size_weights.values())

    return [
        weighted_frac_multi,
        weighted_expected_gold,
        max_sim_multi,
        mean_sim_multi,
        top1_gold_count,
        top3_weighted_gold_mean,
        float(entropy),
    ]


def run_cardinality_oof():
    started = time.perf_counter()
    print("=== Step 3: Training Outer-Fold Cardinality Classifier (P_multi) ===", flush=True)

    folds, fold_for, pools, questions, golds, e5_orders, e5_scores, dup, base_orders, base_scores = load_baseline_data()
    qrow, lal_vecs = load_lal_query_vectors()

    # Load expert channels for rank disagreement
    print("Loading expert rank channels for uncertainty features...", flush=True)
    jina_order, _, _ = core.load_jina(pools)
    lal_order, _, _ = core.load_source_channel("lal", pools)
    bm25_order, _, _ = core.load_source_channel("bm25", pools)
    trigram_order, _, _ = core.load_source_channel("trigram", pools)
    legalir_jina_order, _, _ = core.load_source_channel("jina", pools)

    expert_ranks = {}
    for qid in pools:
        expert_ranks[qid] = {
            "jina": {d: i + 1 for i, d in enumerate(jina_order[qid])},
            "lal": {d: i + 1 for i, d in enumerate(lal_order[qid])},
            "e5": {d: i + 1 for i, d in enumerate(e5_orders["adapted_e5"][qid])},
            "bm25": {d: i + 1 for i, d in enumerate(bm25_order[qid])},
            "trigram": {d: i + 1 for i, d in enumerate(trigram_order[qid])},
            "legalir_jina": {d: i + 1 for i, d in enumerate(legalir_jina_order[qid])},
        }

    # Pre-extract uncertainty features for all queries
    print("Extracting uncertainty features...", flush=True)
    uncertainty_features = {}
    for qid in pools:
        top8 = base_orders[qid][:8]
        uncertainty_features[qid] = extract_uncertainty_features(
            qid, top8, base_scores[qid], expert_ranks[qid]
        )

    p_multi_oof: Dict[str, float] = {}
    oof_y_true: List[int] = []
    oof_y_pred: List[float] = []
    per_fold_auc = {}
    per_fold_pr_auc = {}
    all_qids = set(pools)

    K = 32

    # Fold-isolated training and prediction
    for fold, test_ids in folds.items():
        fold_started = time.perf_counter()
        held = set(test_ids)
        blocked = set(map(str, dup.get(fold, [])))
        train_ids = sorted(all_qids - held - blocked, key=int)

        print(f"Processing {fold}: {len(train_ids)} train queries, {len(test_ids)} test queries...", flush=True)

        # Build support matrices
        train_idx_arr = np.array([qrow[q] for q in train_ids], dtype=np.int64)
        train_mat = lal_vecs[train_idx_arr]  # (N_train, 1024)

        # 1. Compute training features with self-exclusion
        sim_train = np.dot(train_mat, train_mat.T)
        np.fill_diagonal(sim_train, -np.inf)
        topk_train_idx = np.argpartition(-sim_train, K, axis=1)[:, :K]

        X_train = []
        y_train = []
        for row_i, qid in enumerate(train_ids):
            part_idx = topk_train_idx[row_i]
            sorted_part = sorted(part_idx, key=lambda j: (-sim_train[row_i, j], train_ids[j]))
            neighbor_sims = sim_train[row_i, sorted_part]
            neighbor_qids = [train_ids[j] for j in sorted_part]

            supp_feats = compute_support_cardinality_features(neighbor_sims, neighbor_qids, golds)
            unc_feats = uncertainty_features[qid]
            X_train.append(supp_feats + unc_feats)
            y_train.append(1 if len(golds[qid]) > 1 else 0)

        X_train_arr = np.array(X_train, dtype=np.float32)
        y_train_arr = np.array(y_train, dtype=np.int8)

        # Train small Logistic Regression
        scaler = StandardScaler().fit(X_train_arr)
        model = LogisticRegression(
            class_weight="balanced",
            C=0.15,
            solver="liblinear",
            random_state=2026,
            max_iter=1000,
        ).fit(scaler.transform(X_train_arr), y_train_arr)

        # 2. Compute test features using support (train_ids)
        test_idx_arr = np.array([qrow[q] for q in test_ids], dtype=np.int64)
        test_mat = lal_vecs[test_idx_arr]  # (N_test, 1024)

        sim_test = np.dot(test_mat, train_mat.T)
        topk_test_idx = np.argpartition(-sim_test, K, axis=1)[:, :K]

        X_test = []
        y_test = []
        for row_i, qid in enumerate(test_ids):
            part_idx = topk_test_idx[row_i]
            sorted_part = sorted(part_idx, key=lambda j: (-sim_test[row_i, j], train_ids[j]))
            neighbor_sims = sim_test[row_i, sorted_part]
            neighbor_qids = [train_ids[j] for j in sorted_part]

            supp_feats = compute_support_cardinality_features(neighbor_sims, neighbor_qids, golds)
            unc_feats = uncertainty_features[qid]
            X_test.append(supp_feats + unc_feats)
            y_test.append(1 if len(golds[qid]) > 1 else 0)

        X_test_arr = np.array(X_test, dtype=np.float32)
        test_probs = model.predict_proba(scaler.transform(X_test_arr))[:, 1]

        for qid, prob, y_true in zip(test_ids, test_probs, y_test):
            p_multi_oof[qid] = float(prob)
            oof_y_true.append(y_true)
            oof_y_pred.append(float(prob))

        fold_auc = float(roc_auc_score(y_test, test_probs))
        fold_pr_auc = float(average_precision_score(y_test, test_probs))
        per_fold_auc[fold] = fold_auc
        per_fold_pr_auc[fold] = fold_pr_auc
        print(f"  {fold} ROC-AUC: {fold_auc:.4f}, PR-AUC: {fold_pr_auc:.4f} in {time.perf_counter() - fold_started:.2f}s")

    pooled_auc = float(roc_auc_score(oof_y_true, oof_y_pred))
    pooled_pr_auc = float(average_precision_score(oof_y_true, oof_y_pred))
    baseline_prior = float(np.mean(oof_y_true))

    print(f"\nPooled OOF Multi-Gold ROC-AUC: {pooled_auc:.4f}")
    print(f"Pooled OOF Multi-Gold PR-AUC:  {pooled_pr_auc:.4f} (Random prior: {baseline_prior:.4f})")

    elapsed = time.perf_counter() - started
    payload = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "target": "P(|Gold(q)| > 1)",
        "model": "StandardScaler + LogisticRegression(C=0.15, class_weight='balanced', solver='liblinear', random_state=2026)",
        "feature_names": [
            "weighted_frac_multi",
            "weighted_expected_gold",
            "max_sim_multi",
            "mean_sim_multi",
            "top1_gold_count",
            "top3_weighted_gold_mean",
            "gold_count_entropy",
            "lr_gap_1_2",
            "lr_gap_4_5",
            "lr_gap_5_6",
            "lr_gap_5_8",
            "lr_top8_std",
            "expert_disagreement_top8",
        ],
        "evaluable_queries": len(oof_y_true),
        "multi_gold_prior": baseline_prior,
        "pooled_roc_auc": pooled_auc,
        "pooled_pr_auc": pooled_pr_auc,
        "per_fold_roc_auc": per_fold_auc,
        "per_fold_pr_auc": per_fold_pr_auc,
        "runtime_seconds": elapsed,
        "oof_predictions": p_multi_oof,
    }

    out_file = RESULTS_DIR / "CARDINALITY_OOF_REPORT.json"
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"Saved cardinality OOF report to {out_file}", flush=True)
    return p_multi_oof, payload


if __name__ == "__main__":
    run_cardinality_oof()
