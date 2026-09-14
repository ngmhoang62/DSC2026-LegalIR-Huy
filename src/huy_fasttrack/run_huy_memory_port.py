"""Port the proven LegalIR semantic case-memory into the Huy endpoint.

The LAL query embeddings are frozen.  All label-derived memory features are
cross-fit inside each V2 outer fold, including duplicate-linked exclusions.
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import run_huy_5fold_fasttrack as core


sys.path.insert(0, str(core.ROOT))
sys.path.insert(0, str(core.WORKSPACE / "LegalIR/scripts"))
from tune_burst_supervised_profile_bm25 import build_profiles, profile_rank
from exp_final_memory_ltr_probe import MEMORY_NAMES, memory_features, support_index
from tune_burst_graph_posterior import build_graph, graph_rerank


LAL_QUERIES = core.WORKSPACE / "LegalIR/cache/exp109b_encoder_complementarity/embeddings/vnlegal_lal/queries.npz"
RUN_EXPENSIVE_LEARNER_SCREENS = False
RUN_ACTION_UTILITY_SCREEN = False


def normalize(values):
    values = np.asarray(values, dtype=np.float32)
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)


def fit_predict(config, local_pools, train_ids, test_ids, golds,
                rank_features, score_features, metadata):
    rows = core.make_rows(config, local_pools, rank_features, score_features, metadata)
    x = np.vstack([rows[qid] for qid in train_ids])
    y = np.concatenate([
        np.asarray([doc in golds[qid] for doc in local_pools[qid]], dtype=np.int8)
        for qid in train_ids
    ])
    scaler = StandardScaler().fit(x)
    model = LogisticRegression(
        C=.15, class_weight="balanced", solver="liblinear",
        max_iter=3000, random_state=2026,
    ).fit(scaler.transform(x), y)
    pred = {}
    for qid in test_ids:
        values = model.decision_function(scaler.transform(rows[qid]))
        pred[qid] = [
            local_pools[qid][i]
            for i in np.lexsort((np.asarray(local_pools[qid]), -values))
        ]
    return pred, next(iter(rows.values())).shape[1]


def fit_predict_query_balanced(config, local_pools, train_ids, test_ids, golds,
                               rank_features, score_features, metadata):
    """Pointwise LR where every query contributes equal positive/negative mass."""
    rows = core.make_rows(config, local_pools, rank_features, score_features, metadata)
    x = np.vstack([rows[qid] for qid in train_ids])
    labels, weights = [], []
    for qid in train_ids:
        docs = local_pools[qid]
        yq = np.asarray([doc in golds[qid] for doc in docs], dtype=np.int8)
        positives = int(yq.sum())
        negatives = len(yq) - positives
        if not negatives:
            raise RuntimeError(f"all-positive query group {qid}")
        # Mean weight is exactly one in every query; C therefore remains on
        # the same approximate scale while multi-gold queries stop receiving
        # extra positive mass merely because they have more gold documents.
        if positives:
            wq = np.where(
                yq == 1, len(yq) / (2.0 * positives),
                len(yq) / (2.0 * negatives),
            )
        else:
            # Candidate-ceiling misses have no learnable positive inside this
            # fixed pool.  Keep their total query mass without inventing one.
            wq = np.ones(len(yq), dtype=np.float64)
        labels.append(yq)
        weights.append(wq)
    y = np.concatenate(labels)
    sample_weight = np.concatenate(weights)
    scaler = StandardScaler().fit(x)
    model = LogisticRegression(
        C=.15, class_weight=None, solver="liblinear",
        max_iter=3000, random_state=2026,
    ).fit(scaler.transform(x), y, sample_weight=sample_weight)
    pred = {}
    for qid in test_ids:
        values = model.decision_function(scaler.transform(rows[qid]))
        pred[qid] = [
            local_pools[qid][i]
            for i in np.lexsort((np.asarray(local_pools[qid]), -values))
        ]
    return pred, next(iter(rows.values())).shape[1]


def fit_predict_lgbm(config, local_pools, train_ids, test_ids, golds,
                     rank_features, score_features, metadata, tree_config):
    import lightgbm as lgb
    rows = core.make_rows(config, local_pools, rank_features, score_features, metadata)
    x = np.vstack([rows[qid] for qid in train_ids])
    y = np.concatenate([
        np.asarray([doc in golds[qid] for doc in local_pools[qid]], dtype=np.int8)
        for qid in train_ids
    ])
    groups = [len(local_pools[qid]) for qid in train_ids]
    model = lgb.LGBMRanker(
        objective="lambdarank", learning_rate=.05, n_estimators=300,
        min_child_samples=50, feature_fraction=1., bagging_fraction=1.,
        deterministic=True, force_col_wise=True, n_jobs=4,
        random_state=4112, verbosity=-1, **tree_config,
    )
    model.fit(x, y, group=groups, eval_at=[5])
    pred = {}
    for qid in test_ids:
        values = model.predict(rows[qid])
        pred[qid] = [
            local_pools[qid][i]
            for i in np.lexsort((np.asarray(local_pools[qid]), -values))
        ]
    return pred, next(iter(rows.values())).shape[1]


def fit_predict_xgb(config, local_pools, train_ids, test_ids, golds,
                    rank_features, score_features, metadata, tree_config):
    import xgboost as xgb
    rows = core.make_rows(config, local_pools, rank_features, score_features, metadata)
    x = np.vstack([rows[qid] for qid in train_ids])
    y = np.concatenate([
        np.asarray([doc in golds[qid] for doc in local_pools[qid]], dtype=np.int8)
        for qid in train_ids
    ])
    groups = [len(local_pools[qid]) for qid in train_ids]
    model = xgb.XGBRanker(
        objective="rank:ndcg", eval_metric="ndcg@5", tree_method="hist",
        n_jobs=8, random_state=4200, **tree_config,
    )
    model.fit(x, y, group=groups, verbose=False)
    pred = {}
    for qid in test_ids:
        values = model.predict(rows[qid])
        pred[qid] = [
            local_pools[qid][i]
            for i in np.lexsort((np.asarray(local_pools[qid]), -values))
        ]
    return pred, next(iter(rows.values())).shape[1]


def fit_lr_orders_scores(rows, pools, train_ids, test_ids, golds):
    x = np.vstack([rows[qid] for qid in train_ids])
    y = np.concatenate([
        np.asarray([doc in golds[qid] for doc in pools[qid]], dtype=np.int8)
        for qid in train_ids
    ])
    scaler = StandardScaler().fit(x)
    model = LogisticRegression(
        C=.15, class_weight="balanced", solver="liblinear",
        max_iter=3000, random_state=2026,
    ).fit(scaler.transform(x), y)
    orders, scores = {}, {}
    for qid in test_ids:
        values = model.decision_function(scaler.transform(rows[qid]))
        index = np.lexsort((np.asarray(pools[qid]), -values))
        orders[qid] = [pools[qid][i] for i in index]
        scores[qid] = {pools[qid][i]: float(values[i]) for i in range(len(values))}
    return orders, scores


def action_vector(rows, score_map, qid, incumbent, challenger,
                  incumbent_rank, challenger_rank, order):
    docs = order[qid]
    doc_pos = {doc: i for i, doc in enumerate(docs)}
    ci, ii = doc_pos[challenger], doc_pos[incumbent]
    raw = rows[qid]
    boundary_margin = score_map[qid][docs[4]] - score_map[qid][docs[5]]
    return np.concatenate([
        raw[ci], raw[ii], raw[ci] - raw[ii],
        np.asarray([
            score_map[qid][challenger] - score_map[qid][incumbent],
            boundary_margin,
            incumbent_rank / 20.0,
            challenger_rank / 20.0,
        ], dtype=np.float32),
    ]).astype(np.float32)


def main():
    started = time.perf_counter()
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

    with np.load(LAL_QUERIES, allow_pickle=False) as payload:
        embedding_ids = list(map(str, payload["query_ids"].tolist()))
        embedding_values = normalize(payload["vectors"])
    source_row = {qid: i for i, qid in enumerate(embedding_ids)}
    missing = set(pools) - set(source_row)
    if missing:
        raise RuntimeError(f"LAL query embedding misses {len(missing)} V2 qids")
    ordered_qids = sorted(pools, key=int)
    vectors = embedding_values[[source_row[qid] for qid in ordered_qids]]
    row = {qid: i for i, qid in enumerate(ordered_qids)}
    sim_started = time.perf_counter()
    similarities = np.asarray(vectors @ vectors.T, dtype=np.float32)
    similarity_seconds = time.perf_counter() - sim_started
    print(json.dumps({"query_similarity_seconds": similarity_seconds, "shape": list(similarities.shape)}), flush=True)

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
    frozen_rank = {name: core.rank_columns(value, pools) for name, value in base_orders.items()}
    frozen_score = {name: core.score_columns(value, pools) for name, value in base_scores.items()}
    frozen_meta = {"doctype": doctype, "citation": citation}
    base_rank = ["jina_ce", "adapted_e5", "lal_native", "legalir_jina", "huy_profile"]
    base_score = ["jina_ce", "adapted_e5", "lal_native"]
    experiments = {
        "profile_plus_lal_memory": dict(rank_views=base_rank, score_channels=base_score, metadata=["doctype", "citation", "lal_memory"]),
        "lal_memory_no_profile": dict(rank_views=base_rank[:-1], score_channels=base_score, metadata=["doctype", "citation", "lal_memory"]),
        "memory_winner_no_doctype": dict(rank_views=base_rank, score_channels=base_score, metadata=["citation", "lal_memory"]),
        "memory_winner_no_lal_retrieval": dict(rank_views=["jina_ce", "adapted_e5", "legalir_jina", "huy_profile"], score_channels=["jina_ce", "adapted_e5"], metadata=["doctype", "citation", "lal_memory"]),
        "memory_minimal_no_doctype_no_lal_retrieval": dict(rank_views=["jina_ce", "adapted_e5", "legalir_jina", "huy_profile"], score_channels=["jina_ce", "adapted_e5"], metadata=["citation", "lal_memory"]),
        "profile_memory_plus_graph": dict(rank_views=base_rank + ["huy_graph"], score_channels=base_score, metadata=["citation", "lal_memory"]),
        "profile_plus_graph_no_memory": dict(rank_views=base_rank + ["huy_graph"], score_channels=base_score, metadata=["citation"]),
        "memory_plus_graph_no_profile": dict(rank_views=base_rank[:-1] + ["huy_graph"], score_channels=base_score, metadata=["citation", "lal_memory"]),
        "profile_memory_plus_bm25_rank": dict(rank_views=base_rank + ["legalir_bm25"], score_channels=base_score, metadata=["citation", "lal_memory"]),
        "profile_memory_plus_trigram_rank": dict(rank_views=base_rank + ["legalir_trigram"], score_channels=base_score, metadata=["citation", "lal_memory"]),
        "profile_memory_plus_sparse_ranks": dict(rank_views=base_rank + ["legalir_bm25", "legalir_trigram"], score_channels=base_score, metadata=["citation", "lal_memory"]),
        "profile_memory_plus_sparse_rank_scores": dict(rank_views=base_rank + ["legalir_bm25", "legalir_trigram"], score_channels=base_score + ["legalir_bm25", "legalir_trigram"], metadata=["citation", "lal_memory"]),
        "profile_memory_plus_bm25_rank_score": dict(rank_views=base_rank + ["legalir_bm25"], score_channels=base_score + ["legalir_bm25"], metadata=["citation", "lal_memory"]),
        "profile_memory_plus_trigram_rank_score": dict(rank_views=base_rank + ["legalir_trigram"], score_channels=base_score + ["legalir_trigram"], metadata=["citation", "lal_memory"]),
        "sparse_winner_no_huy_jina_ce": dict(
            rank_views=["adapted_e5", "lal_native", "legalir_jina", "huy_profile", "legalir_bm25", "legalir_trigram"],
            score_channels=["adapted_e5", "lal_native", "legalir_bm25", "legalir_trigram"],
            metadata=["citation", "lal_memory"],
        ),
        "sparse_winner_no_legalir_jina": dict(
            rank_views=["jina_ce", "adapted_e5", "lal_native", "huy_profile", "legalir_bm25", "legalir_trigram"],
            score_channels=base_score + ["legalir_bm25", "legalir_trigram"],
            metadata=["citation", "lal_memory"],
        ),
        "sparse_winner_no_lal_retrieval": dict(
            rank_views=["jina_ce", "adapted_e5", "legalir_jina", "huy_profile", "legalir_bm25", "legalir_trigram"],
            score_channels=["jina_ce", "adapted_e5", "legalir_bm25", "legalir_trigram"],
            metadata=["citation", "lal_memory"],
        ),
    }
    tree_experiments = {
        "lgbm_l7_t30_profile_memory": (
            dict(rank_views=base_rank, score_channels=base_score, metadata=["citation", "lal_memory"]),
            dict(num_leaves=7, lambdarank_truncation_level=30),
        ),
        "lgbm_l15_t5_profile_memory": (
            dict(rank_views=base_rank, score_channels=base_score, metadata=["citation", "lal_memory"]),
            dict(num_leaves=15, lambdarank_truncation_level=5),
        ),
    } if RUN_EXPENSIVE_LEARNER_SCREENS else {}
    xgb_experiments = {
        "xgb_d4_profile_memory": (
            dict(rank_views=base_rank, score_channels=base_score, metadata=["citation", "lal_memory"]),
            dict(n_estimators=450, learning_rate=.04, max_depth=4, colsample_bytree=1., subsample=1.),
        ),
        "xgb_d5_profile_memory": (
            dict(rank_views=base_rank, score_channels=base_score, metadata=["citation", "lal_memory"]),
            dict(n_estimators=500, learning_rate=.035, max_depth=5, colsample_bytree=.8, subsample=.85),
        ),
    } if RUN_EXPENSIVE_LEARNER_SCREENS else {}
    predictions = {name: {} for name in experiments}
    predictions["query_balanced_sparse_winner"] = {}
    predictions.update({name: {} for name in tree_experiments})
    predictions.update({name: {} for name in xgb_experiments})
    if RUN_ACTION_UTILITY_SCREEN:
        predictions["action_utility_depth20"] = {}
    feature_counts = {}
    fold_runtime = {}
    all_qids = set(pools)

    for outer, test_ids in folds.items():
        fold_started = time.perf_counter()
        blocked = set(map(str, dup.get(outer, [])))
        train_ids = sorted(all_qids - set(test_ids) - blocked, key=int)
        profile_orders = {}
        graph_orders = {}
        memory_rows = {}

        # Outer test features: support contains only the four allowed folds.
        test_profile_model = build_profiles(queries, train_ids)
        test_by_doc, test_frequency = support_index(golds, train_ids)
        test_graph_frequency, test_graph_adjacency, _ = build_graph(queries, train_ids)
        train_index = [row[qid] for qid in train_ids]
        for qid in test_ids:
            profile_orders[qid] = profile_rank(questions[qid], test_profile_model, 2, 1.2, .75, .3)
            graph_orders[qid] = graph_rerank(
                e5_orders["adapted_e5"][qid], test_graph_frequency,
                test_graph_adjacency, 3, .5, "conditional", 3, .4, 0,
            )
            memory_rows[qid] = memory_features(
                similarities[row[qid], train_index], pools[qid], train_ids,
                golds, test_by_doc, test_frequency,
            )

        # Outer-training features are cross-fit by their original V2 fold.
        for inner in folds:
            if inner == outer:
                continue
            inner_ids = [qid for qid in train_ids if fold_for[qid] == inner]
            inner_blocked = set(map(str, dup.get(inner, [])))
            memory_ids = [
                qid for qid in train_ids
                if fold_for[qid] != inner and qid not in inner_blocked
            ]
            profile_model = build_profiles(queries, memory_ids)
            by_doc, frequency = support_index(golds, memory_ids)
            graph_frequency, graph_adjacency, _ = build_graph(queries, memory_ids)
            memory_index = [row[qid] for qid in memory_ids]
            for qid in inner_ids:
                profile_orders[qid] = profile_rank(questions[qid], profile_model, 2, 1.2, .75, .3)
                graph_orders[qid] = graph_rerank(
                    e5_orders["adapted_e5"][qid], graph_frequency,
                    graph_adjacency, 3, .5, "conditional", 3, .4, 0,
                )
                memory_rows[qid] = memory_features(
                    similarities[row[qid], memory_index], pools[qid], memory_ids,
                    golds, by_doc, frequency,
                )

        local_ids = train_ids + list(test_ids)
        local_pools = {qid: pools[qid] for qid in local_ids}
        rank_features = {name: {qid: frozen_rank[name][qid] for qid in local_ids} for name in frozen_rank}
        rank_features["huy_profile"] = core.rank_columns(profile_orders, local_pools)
        rank_features["huy_graph"] = core.rank_columns(graph_orders, local_pools)
        score_features = {name: {qid: frozen_score[name][qid] for qid in local_ids} for name in frozen_score}
        metadata = {name: {qid: value[qid] for qid in local_ids} for name, value in frozen_meta.items()}
        metadata["lal_memory"] = memory_rows

        for name, config in experiments.items():
            pred, count = fit_predict(
                config, local_pools, train_ids, test_ids, golds,
                rank_features, score_features, metadata,
            )
            predictions[name].update(pred)
            feature_counts[name] = count
        pred, count = fit_predict_query_balanced(
            experiments["profile_memory_plus_sparse_rank_scores"],
            local_pools, train_ids, test_ids, golds,
            rank_features, score_features, metadata,
        )
        predictions["query_balanced_sparse_winner"].update(pred)
        feature_counts["query_balanced_sparse_winner"] = count
        for name, (config, tree_config) in tree_experiments.items():
            pred, count = fit_predict_lgbm(
                config, local_pools, train_ids, test_ids, golds,
                rank_features, score_features, metadata, tree_config,
            )
            predictions[name].update(pred)
            feature_counts[name] = count
        for name, (config, tree_config) in xgb_experiments.items():
            pred, count = fit_predict_xgb(
                config, local_pools, train_ids, test_ids, golds,
                rank_features, score_features, metadata, tree_config,
            )
            predictions[name].update(pred)
            feature_counts[name] = count

        if RUN_ACTION_UTILITY_SCREEN:
            # Strict nested action utility: generate boundary actions with base
            # rankers that exclude both the outer fold and the action query's fold.
            sparse_config = experiments["profile_memory_plus_sparse_rank_scores"]
            sparse_rows = core.make_rows(
                sparse_config, local_pools, rank_features, score_features, metadata
            )
            action_x, action_y, action_weight = [], [], []
            for inner in folds:
                if inner == outer:
                    continue
                inner_ids = [qid for qid in train_ids if fold_for[qid] == inner]
                inner_blocked = set(map(str, dup.get(inner, [])))
                base_train_ids = [
                    qid for qid in train_ids
                    if fold_for[qid] != inner and qid not in inner_blocked
                ]
                inner_order, inner_scores = fit_lr_orders_scores(
                    sparse_rows, local_pools, base_train_ids, inner_ids, golds
                )
                for qid in inner_ids:
                    incumbent = inner_order[qid][4]
                    for challenger_rank, challenger in enumerate(inner_order[qid][5:20], 6):
                        target = (
                            float(challenger in golds[qid])
                            - float(incumbent in golds[qid])
                        ) / len(golds[qid])
                        action_x.append(action_vector(
                            sparse_rows, inner_scores, qid, incumbent, challenger,
                            5, challenger_rank, inner_order,
                        ))
                        action_y.append(target)
                        action_weight.append(1.0 if target else .15)
            import xgboost as xgb
            gate = xgb.XGBRegressor(
                objective="reg:squarederror", n_estimators=200,
                learning_rate=.05, max_depth=3, min_child_weight=20,
                subsample=.85, colsample_bytree=.8, reg_lambda=5.,
                tree_method="hist", n_jobs=8, random_state=731,
            )
            gate.fit(
                np.asarray(action_x, dtype=np.float32),
                np.asarray(action_y, dtype=np.float32),
                sample_weight=np.asarray(action_weight, dtype=np.float32),
                verbose=False,
            )
            test_order, test_scores = fit_lr_orders_scores(
                sparse_rows, local_pools, train_ids, test_ids, golds
            )
            for qid in test_ids:
                incumbent = test_order[qid][4]
                challengers = test_order[qid][5:20]
                matrix = np.vstack([
                    action_vector(
                        sparse_rows, test_scores, qid, incumbent, challenger,
                        5, challenger_rank, test_order,
                    )
                    for challenger_rank, challenger in enumerate(challengers, 6)
                ])
                utility = gate.predict(matrix)
                best_action = int(np.argmax(utility))
                result = list(test_order[qid])
                if float(utility[best_action]) > 0.0:
                    result[4] = challengers[best_action]
                    result = list(dict.fromkeys(result))
                predictions["action_utility_depth20"][qid] = result
            feature_counts["action_utility_depth20"] = next(iter(sparse_rows.values())).shape[1]
        fold_runtime[outer] = time.perf_counter() - fold_started
        print(json.dumps({"fold": outer, "seconds": fold_runtime[outer]}), flush=True)

    reference = {
        str(item["qid"]): list(map(str, item["top5"]))
        for item in core.read_jsonl(core.OUT / "HUY_PROFILE_5FOLD_PREDICTIONS.jsonl")
    }
    results = []
    all_configs = dict(experiments)
    all_configs["query_balanced_sparse_winner"] = {
        **experiments["profile_memory_plus_sparse_rank_scores"],
        "learner": "LogisticRegression query-balanced sample weights",
        "weighting": "each query contributes total positive mass 0.5 and negative mass 0.5",
    }
    all_configs.update({
        name: {**config, "learner": "LightGBM LambdaRank", "tree_config": tree_config}
        for name, (config, tree_config) in tree_experiments.items()
    })
    if RUN_ACTION_UTILITY_SCREEN:
        all_configs["action_utility_depth20"] = {
            "base": experiments["profile_memory_plus_sparse_rank_scores"],
            "learner": "nested XGBoost expected-utility regressor",
            "action": "optionally replace rank 5 with one challenger from ranks 6-20",
            "threshold": 0.0,
            "multiple_swaps": False,
        }
    all_configs.update({
        name: {**config, "learner": "XGBoost rank:ndcg", "tree_config": tree_config}
        for name, (config, tree_config) in xgb_experiments.items()
    })
    for name in all_configs:
        results.append({
            "name": name,
            "config": all_configs[name],
            "feature_count": feature_counts[name],
            "metrics": core.metrics(predictions[name], golds, folds),
            "paired_vs_profile_reference": core.compare(predictions[name], reference, golds, folds),
        })
    results.sort(key=lambda item: item["metrics"]["recall_at_5"], reverse=True)
    best = results[0]
    report = {
        "schema_version": "dsc2026.huy_fasttrack.lal_memory_graph_port.v2",
        "status": "COMPLETE_NESTED_STRICT_5FOLD_OOF",
        "component": "LegalIR 14D LAL case-memory plus Huy co-relevance graph ablation",
        "graph_config": {"seed": "adapted_e5", "seed_depth": 3, "seed_power": .5, "mode": "conditional", "min_edge": 3, "alpha": .4, "base_k": 0},
        "memory_feature_names": list(MEMORY_NAMES),
        "similarity_seconds": similarity_seconds,
        "reference_recall_at_5": core.metrics(reference, golds, folds)["recall_at_5"],
        "best": best,
        "results": results,
        "fold_runtime_seconds": fold_runtime,
        "runtime_seconds": time.perf_counter() - started,
    }
    core.write_json(core.OUT / "HUY_LAL_MEMORY_PORT_REPORT.json", report)
    lock_dir = core.OUT / "learner_prediction_locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    for name, pred in predictions.items():
        with (lock_dir / f"{name}.jsonl").open("w", encoding="utf-8", newline="\n") as f:
            for qid in sorted(pools, key=int):
                f.write(json.dumps({"qid": qid, "order": pred[qid][:20]}, ensure_ascii=False, separators=(",", ":")) + "\n")
    with (core.OUT / "BEST_LAL_MEMORY_PREDICTIONS.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for qid in sorted(pools, key=int):
            f.write(json.dumps({"qid": qid, "top5": predictions[best["name"]][qid][:5]}, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(json.dumps({
        "best": best,
        "results": [{"name": item["name"], "recall_at_5": item["metrics"]["recall_at_5"], "delta": item["paired_vs_profile_reference"]["delta_recall_at_5"]} for item in results],
        "runtime_seconds": report["runtime_seconds"],
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
