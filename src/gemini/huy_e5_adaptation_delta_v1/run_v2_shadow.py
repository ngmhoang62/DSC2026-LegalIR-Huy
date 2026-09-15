"""Strict-V2 Shadow Sanity Check on authoritative endpoint: profile_memory_plus_sparse_rank_scores.

Compares:
- V0: Exact authoritative endpoint parity (44 features, R@5 = 0.9488556715777428)
- V1: V0 + 6 adaptation-delta features (50 features)

Evaluates on all 6,991 evaluable queries across 5 strict outer folds.
Produces results/gemini/huy_e5_adaptation_delta_v1/V2_SHADOW_REPORT.json.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = ROOT / "results/gemini/huy_e5_adaptation_delta_v1"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT / "src/huy_fasttrack"))
import run_huy_5fold_fasttrack as core
from run_huy_memory_port import (
    LAL_QUERIES,
    build_graph,
    build_profiles,
    graph_rerank,
    memory_features,
    normalize,
    profile_rank,
    support_index,
)

EXPECTED_V0_METRICS = {
    "recall_at_5": 0.9488556715777428,
    "precision_at_5": 0.20314690316120732,
    "single_gold_recall_at_5": 0.9641304347826087,
    "multi_gold_recall_at_5": 0.7703266787658803,
    "per_fold_recall_at_5": {
        "fold_0": 0.9524320457796852,
        "fold_1": 0.9451597520267048,
        "fold_2": 0.9511555873242793,
        "fold_3": 0.9420243204577969,
        "fold_4": 0.9535050071530758,
    },
}


def compute_standardized_scores(raw_scores: Dict[str, float], docs: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    values = np.asarray([raw_scores.get(d, np.nan) for d in docs], dtype=np.float64)
    present = values[~np.isnan(values)]
    if present.size:
        mean = float(present.mean())
        std = float(present.std()) or 1.0
        top = float(present.max())
    else:
        mean, std, top = 0.0, 1.0, 0.0
    filled = np.where(np.isnan(values), mean - 2 * std, values)
    z = (filled - mean) / std
    gap = (filled - top) / std
    return z, gap


def evaluate_predictions(
    predictions: Dict[str, List[str]],
    golds: Dict[str, Set[str]],
    folds: Dict[str, List[str]],
) -> Dict[str, Any]:
    values, precisions = [], []
    single, multi = [], []
    by_fold = {}

    for fold, qids in sorted(folds.items()):
        fold_vals = []
        for qid in qids:
            gold = golds[qid]
            top5 = predictions[qid][:5]
            hits = len(set(top5) & gold)
            val = hits / len(gold)
            values.append(val)
            precisions.append(hits / 5.0)
            fold_vals.append(val)
            if len(gold) == 1:
                single.append(val)
            else:
                multi.append(val)
        by_fold[fold] = float(np.mean(fold_vals))

    return {
        "queries": len(values),
        "recall_at_5": float(np.mean(values)),
        "precision_at_5": float(np.mean(precisions)),
        "single_gold_recall_at_5": float(np.mean(single)),
        "multi_gold_recall_at_5": float(np.mean(multi)),
        "per_fold_recall_at_5": by_fold,
    }


def compare_preds(
    cand_preds: Dict[str, List[str]],
    ref_preds: Dict[str, List[str]],
    golds: Dict[str, Set[str]],
    pools: Dict[str, List[str]],
) -> Dict[str, Any]:
    wins = losses = ties = 0
    churn = 0
    gold_in = gold_out = 0

    for q in pools:
        gold = golds[q]
        top5_c = set(cand_preds[q][:5])
        top5_r = set(ref_preds[q][:5])

        rec_c = len(top5_c & gold) / len(gold)
        rec_r = len(top5_r & gold) / len(gold)

        if rec_c > rec_r:
            wins += 1
        elif rec_c < rec_r:
            losses += 1
        else:
            ties += 1

        if top5_c != top5_r:
            churn += 1

        gold_in += len((top5_c - top5_r) & gold)
        gold_out += len((top5_r - top5_c) & gold)

    return {
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "win_loss_diff": wins - losses,
        "top5_churn": churn,
        "gold_crossings_into_top5": gold_in,
        "gold_crossings_out_of_top5": gold_out,
    }


def main():
    started_all = time.perf_counter()
    print("=== Step 3: Strict-V2 Shadow Sanity Check ===", flush=True)

    # 1. Load inputs
    print("Loading fasttrack environment...", flush=True)
    folds, pools, questions, golds, e5_orders, e5_scores, dup, _ = core.load_inputs()
    fold_for = {qid: fold for fold, qids in folds.items() for qid in qids}
    queries = {qid: (questions[qid], golds[qid]) for qid in pools}

    jina_order, jina_scores, _ = core.load_jina(pools)
    lal_order, lal_scores, _ = core.load_source_channel("lal", pools)
    legalir_jina_order, _, _ = core.load_source_channel("jina", pools)
    bm25_order, bm25_scores, _ = core.load_source_channel("bm25", pools)
    trigram_order, trigram_scores, _ = core.load_source_channel("trigram", pools)
    heads = core.document_heads(pools)
    doctype, citation = core.metadata_arrays(pools, questions, heads)

    # LAL query similarities
    print("Loading LAL query embeddings...", flush=True)
    with np.load(LAL_QUERIES, allow_pickle=False) as payload:
        embedding_ids = list(map(str, payload["query_ids"].tolist()))
        embedding_values = normalize(payload["vectors"])
    source_row = {qid: i for i, qid in enumerate(embedding_ids)}
    ordered_qids = sorted(pools, key=int)
    vectors = embedding_values[[source_row[qid] for qid in ordered_qids]]
    row = {qid: i for i, qid in enumerate(ordered_qids)}
    similarities = np.asarray(vectors @ vectors.T, dtype=np.float32)

    base_orders = {
        "jina_ce": jina_order,
        "adapted_e5": e5_orders["adapted_e5"],
        "lal_native": lal_order,
        "legalir_jina": legalir_jina_order,
        "legalir_bm25": bm25_order,
        "legalir_trigram": trigram_order,
    }
    base_scores = {
        "jina_ce": jina_scores,
        "adapted_e5": e5_scores["adapted_e5"],
        "lal_native": lal_scores,
        "legalir_bm25": bm25_scores,
        "legalir_trigram": trigram_scores,
    }

    frozen_rank = {name: core.rank_columns(order_map, pools) for name, order_map in base_orders.items()}
    frozen_score = {name: core.score_columns(score_map, pools) for name, score_map in base_scores.items()}
    frozen_meta = {"doctype": doctype, "citation": citation}

    # Authoritative endpoint config
    config_v0 = dict(
        rank_views=[
            "jina_ce", "adapted_e5", "lal_native", "legalir_jina", "huy_profile",
            "legalir_bm25", "legalir_trigram",
        ],
        score_channels=[
            "jina_ce", "adapted_e5", "lal_native", "legalir_bm25", "legalir_trigram",
        ],
        metadata=["citation", "lal_memory"],
    )

    # 2. Precompute 6 Adaptation Delta Features for all 6,991 queries
    print("Computing 6 Adaptation Delta Features for all 6,991 V2 queries...", flush=True)
    delta_features = {}
    for qid, docs in pools.items():
        n_cands = len(docs)
        ad_scores = e5_scores["adapted_e5"][qid]
        fr_scores = e5_scores["frozen_e5"][qid]
        ad_order = e5_orders["adapted_e5"][qid]
        fr_order = e5_orders["frozen_e5"][qid]

        ad_z, ad_gap = compute_standardized_scores(ad_scores, docs)
        fr_z, fr_gap = compute_standardized_scores(fr_scores, docs)

        ranks_ad = {d: i + 1 for i, d in enumerate(ad_order)}
        ranks_fr = {d: i + 1 for i, d in enumerate(fr_order)}

        d_rows = []
        for i, d in enumerate(docs):
            has_ad = d in ad_scores and not (math.isnan(ad_scores[d]) or np.isnan(ad_scores[d]))
            has_fr = d in fr_scores and not (math.isnan(fr_scores[d]) or np.isnan(fr_scores[d]))

            if has_ad and has_fr:
                d_z = float(ad_z[i] - fr_z[i])
                d_gap = float(ad_gap[i] - fr_gap[i])
                r_ad = ranks_ad.get(d, 60)
                r_fr = ranks_fr.get(d, 60)
                r_gain = float(r_fr - r_ad) / float(n_cands)
                rr_gain = 1.0 / (10.0 + r_ad) - 1.0 / (10.0 + r_fr)
                prom = 1.0 if (r_ad <= 5 and r_fr > 5) else 0.0
                dem = 1.0 if (r_fr <= 5 and r_ad > 5) else 0.0
            else:
                d_z = d_gap = r_gain = rr_gain = prom = dem = 0.0

            d_rows.append([d_z, d_gap, r_gain, rr_gain, prom, dem])

        delta_features[qid] = np.asarray(d_rows, dtype=np.float32)

    # 3. Fit 5-fold CV for V0 and V1
    print("Fitting 5-fold CV across outer folds...", flush=True)
    all_qids = set(pools)
    preds_v0: Dict[str, List[str]] = {}
    preds_v1: Dict[str, List[str]] = {}

    for outer, test_ids in folds.items():
        t_fold = time.perf_counter()
        blocked = set(map(str, dup.get(outer, [])))
        train_ids = sorted(all_qids - set(test_ids) - blocked, key=int)
        profile_orders = {}
        memory_rows = {}

        # Outer test features
        test_profile_model = build_profiles(queries, train_ids)
        test_by_doc, test_frequency = support_index(golds, train_ids)
        train_index = [row[qid] for qid in train_ids]
        for qid in test_ids:
            profile_orders[qid] = profile_rank(questions[qid], test_profile_model, 2, 1.2, 0.75, 0.3)
            memory_rows[qid] = memory_features(
                similarities[row[qid], train_index], pools[qid], train_ids,
                golds, test_by_doc, test_frequency,
            )

        # Outer train features cross-fit
        for inner in folds:
            if inner == outer:
                continue
            inner_ids = [qid for qid in train_ids if fold_for[qid] == inner]
            inner_blocked = set(map(str, dup.get(inner, [])))
            memory_ids = [qid for qid in train_ids if fold_for[qid] != inner and qid not in inner_blocked]
            profile_model = build_profiles(queries, memory_ids)
            by_doc, frequency = support_index(golds, memory_ids)
            memory_index = [row[qid] for qid in memory_ids]
            for qid in inner_ids:
                profile_orders[qid] = profile_rank(questions[qid], profile_model, 2, 1.2, 0.75, 0.3)
                memory_rows[qid] = memory_features(
                    similarities[row[qid], memory_index], pools[qid], memory_ids,
                    golds, by_doc, frequency,
                )

        local_ids = train_ids + list(test_ids)
        local_pools = {qid: pools[qid] for qid in local_ids}
        rank_features = {name: {qid: frozen_rank[name][qid] for qid in local_ids} for name in frozen_rank}
        rank_features["huy_profile"] = core.rank_columns(profile_orders, local_pools)
        score_features = {name: {qid: frozen_score[name][qid] for qid in local_ids} for name in frozen_score}
        metadata = {name: {qid: value[qid] for qid in local_ids} for name, value in frozen_meta.items()}
        metadata["lal_memory"] = memory_rows

        # Make base V0 rows
        rows_v0 = core.make_rows(config_v0, local_pools, rank_features, score_features, metadata)

        # Make V1 rows: V0 + 6 delta features
        rows_v1 = {qid: np.concatenate([rows_v0[qid], delta_features[qid]], axis=1) for qid in local_ids}

        # Fit V0
        x_tr0 = np.vstack([rows_v0[qid] for qid in train_ids])
        y_tr = np.concatenate([
            np.asarray([doc in golds[qid] for doc in local_pools[qid]], dtype=np.int8)
            for qid in train_ids
        ])
        scaler0 = StandardScaler().fit(x_tr0)
        lr0 = LogisticRegression(C=0.15, class_weight="balanced", solver="liblinear", max_iter=3000, random_state=2026)
        lr0.fit(scaler0.transform(x_tr0), y_tr)
        for qid in test_ids:
            vals0 = lr0.decision_function(scaler0.transform(rows_v0[qid]))
            preds_v0[qid] = [local_pools[qid][i] for i in np.lexsort((np.asarray(local_pools[qid]), -vals0))]

        # Fit V1
        x_tr1 = np.vstack([rows_v1[qid] for qid in train_ids])
        scaler1 = StandardScaler().fit(x_tr1)
        lr1 = LogisticRegression(C=0.15, class_weight="balanced", solver="liblinear", max_iter=3000, random_state=2026)
        lr1.fit(scaler1.transform(x_tr1), y_tr)
        for qid in test_ids:
            vals1 = lr1.decision_function(scaler1.transform(rows_v1[qid]))
            preds_v1[qid] = [local_pools[qid][i] for i in np.lexsort((np.asarray(local_pools[qid]), -vals1))]

        print(f"  Outer {outer} finished in {time.perf_counter() - t_fold:.2f}s", flush=True)

    # 4. Evaluate V0 and V1
    metrics_v0 = evaluate_predictions(preds_v0, golds, folds)
    metrics_v1 = evaluate_predictions(preds_v1, golds, folds)
    comp = compare_preds(preds_v1, preds_v0, golds, pools)

    diff_v0_expected = abs(metrics_v0["recall_at_5"] - EXPECTED_V0_METRICS["recall_at_5"])
    print(f"\nV0 Recall@5: {metrics_v0['recall_at_5']:.8f} (Expected: {EXPECTED_V0_METRICS['recall_at_5']:.8f}, Diff: {diff_v0_expected:.1e})")
    assert diff_v0_expected < 1e-9, f"V0 parity failed: {metrics_v0['recall_at_5']} vs {EXPECTED_V0_METRICS['recall_at_5']}"
    print("V0 Authoritative Baseline Parity PASSED!", flush=True)

    delta_r5 = metrics_v1["recall_at_5"] - metrics_v0["recall_at_5"]
    print(f"V1 Recall@5: {metrics_v1['recall_at_5']:.8f} (Delta vs V0: {delta_r5:+.8f})")
    print(f"  V0 P@5: {metrics_v0['precision_at_5']:.8f} -> V1 P@5: {metrics_v1['precision_at_5']:.8f} ({metrics_v1['precision_at_5'] - metrics_v0['precision_at_5']:+.8f})")
    print(f"  V0 Single: {metrics_v0['single_gold_recall_at_5']:.8f} -> V1 Single: {metrics_v1['single_gold_recall_at_5']:.8f} ({metrics_v1['single_gold_recall_at_5'] - metrics_v0['single_gold_recall_at_5']:+.8f})")
    print(f"  V0 Multi: {metrics_v0['multi_gold_recall_at_5']:.8f} -> V1 Multi: {metrics_v1['multi_gold_recall_at_5']:.8f} ({metrics_v1['multi_gold_recall_at_5'] - metrics_v0['multi_gold_recall_at_5']:+.8f})")

    fold_deltas = {}
    for f in sorted(folds.keys()):
        d_f = metrics_v1["per_fold_recall_at_5"][f] - metrics_v0["per_fold_recall_at_5"][f]
        fold_deltas[f] = d_f
        print(f"  {f}: V0 {metrics_v0['per_fold_recall_at_5'][f]:.6f} -> V1 {metrics_v1['per_fold_recall_at_5'][f]:.6f} ({d_f:+.6f})")

    print(f"Comparison: Wins={comp['wins']}, Losses={comp['losses']}, Ties={comp['ties']} (Win-Loss={comp['win_loss_diff']})")
    print(f"Top-5 Churn={comp['top5_churn']}, Crossings In={comp['gold_crossings_into_top5']}, Crossings Out={comp['gold_crossings_out_of_top5']}")

    elapsed = time.perf_counter() - started_all

    # 5. Save Report
    report = {
        "schema_version": "dsc2026.gemini.huy_e5_adaptation_delta_v1.v2_shadow_report.v1",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "runtime_seconds": elapsed,
        "authoritative_baseline_v0": {
            "name": "profile_memory_plus_sparse_rank_scores",
            "features": 44,
            "metrics": metrics_v0,
            "parity_passed": True,
        },
        "shadow_arm_v1": {
            "name": "profile_memory_plus_sparse_rank_scores_plus_6_adaptation_delta",
            "features": 50,
            "metrics": metrics_v1,
            "delta_recall_at_5": delta_r5,
            "delta_precision_at_5": metrics_v1["precision_at_5"] - metrics_v0["precision_at_5"],
            "delta_single_gold": metrics_v1["single_gold_recall_at_5"] - metrics_v0["single_gold_recall_at_5"],
            "delta_multi_gold": metrics_v1["multi_gold_recall_at_5"] - metrics_v0["multi_gold_recall_at_5"],
            "per_fold_delta": fold_deltas,
            "fold_regressions": {f: d for f, d in fold_deltas.items() if d < -1e-9},
            "comparison": comp,
        },
    }

    out_file = RESULTS_DIR / "V2_SHADOW_REPORT.json"
    out_file.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved: {out_file}", flush=True)


if __name__ == "__main__":
    main()
