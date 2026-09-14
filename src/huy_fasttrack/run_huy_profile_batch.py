"""Focused strict-V2 profile batch around the winning Huy mechanism.

This is deliberately narrow: three historically motivated profile views,
one direct-score test, and leave-one-component-out ablations of the winner.
"""

from __future__ import annotations

import json
import math
import sys
import time
from collections import defaultdict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import run_huy_5fold_fasttrack as core


sys.path.insert(0, str(core.ROOT))
from tune_burst_supervised_profile_bm25 import build_profiles, features


PROFILE_CONFIGS = {
    "profile_1g_prior0": (1, 1.2, .75, 0.0),
    "profile_2g_prior03": (2, 1.2, .75, .3),
    "profile_3g_prior03": (3, 1.2, .75, .3),
}


def profile_score_map(text, model, config):
    max_ngram, k1, b, prior_power = config
    postings, document_frequency, document_length, label_frequency = model
    docs_count = max(len(document_length), 1)
    avg_length = sum(document_length.values()) / docs_count
    scores = defaultdict(float)
    for feat in features(text, max_ngram):
        posting = postings.get(feat)
        if not posting:
            continue
        df = document_frequency[feat]
        idf = math.log1p((docs_count - df + .5) / (df + .5))
        order = int(feat[0])
        phrase_weight = (1.0, 1.35, 1.65)[order - 1]
        for doc, tf in posting.items():
            norm = k1 * (1.0 - b + b * document_length[doc] / avg_length)
            scores[doc] += phrase_weight * idf * tf * (k1 + 1.0) / (tf + norm)
    max_freq = max(label_frequency.values(), default=1)
    for doc in scores:
        prior = (label_frequency[doc] + .5) / (max_freq + .5)
        scores[doc] *= prior ** prior_power
    return dict(scores)


def score_and_order(text, model, config):
    scores = profile_score_map(text, model, config)
    order = sorted(scores, key=lambda doc: (-scores[doc], doc))[:100]
    return order, scores


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


def main():
    started = time.perf_counter()
    folds, pools, questions, golds, e5_orders, e5_scores, dup, _ = core.load_inputs()
    fold_for = {qid: fold for fold, qids in folds.items() for qid in qids}
    queries = {qid: (questions[qid], golds[qid]) for qid in pools}
    jina_order, jina_scores, _ = core.load_jina(pools)
    lal_order, lal_scores, _ = core.load_source_channel("lal", pools)
    legalir_jina_order, _, _ = core.load_source_channel("jina", pools)
    heads = core.document_heads(pools)
    doctype, citation = core.metadata_arrays(pools, questions, heads)

    base_orders = {
        "jina_ce": jina_order,
        "adapted_e5": e5_orders["adapted_e5"],
        "lal_native": lal_order,
        "legalir_jina": legalir_jina_order,
    }
    base_scores = {
        "jina_ce": jina_scores,
        "adapted_e5": e5_scores["adapted_e5"],
        "lal_native": lal_scores,
    }
    frozen_rank = {name: core.rank_columns(value, pools) for name, value in base_orders.items()}
    frozen_score = {name: core.score_columns(value, pools) for name, value in base_scores.items()}
    metadata = {"doctype": doctype, "citation": citation}

    base_rank = ["jina_ce", "adapted_e5", "lal_native", "legalir_jina"]
    base_score = ["jina_ce", "adapted_e5", "lal_native"]
    experiments = {
        "profile_1g_rank": dict(rank_views=base_rank + ["profile_1g_prior0"], score_channels=base_score, metadata=["doctype", "citation"]),
        "profile_2g_rank": dict(rank_views=base_rank + ["profile_2g_prior03"], score_channels=base_score, metadata=["doctype", "citation"]),
        "profile_3g_rank": dict(rank_views=base_rank + ["profile_3g_prior03"], score_channels=base_score, metadata=["doctype", "citation"]),
        "profile_all3_rank": dict(rank_views=base_rank + list(PROFILE_CONFIGS), score_channels=base_score, metadata=["doctype", "citation"]),
        "profile_2g_rank_score": dict(rank_views=base_rank + ["profile_2g_prior03"], score_channels=base_score + ["profile_2g_prior03"], metadata=["doctype", "citation"]),
        "profile_1g3g_scores_plus_2g_rank": dict(rank_views=base_rank + ["profile_2g_prior03"], score_channels=base_score + ["profile_1g_prior0", "profile_3g_prior03"], metadata=["doctype", "citation"]),
        "winner_no_doctype": dict(rank_views=base_rank + ["profile_2g_prior03"], score_channels=base_score, metadata=["citation"]),
        "winner_no_citation": dict(rank_views=base_rank + ["profile_2g_prior03"], score_channels=base_score, metadata=["doctype"]),
        "winner_no_legalir_jina": dict(rank_views=["jina_ce", "adapted_e5", "lal_native", "profile_2g_prior03"], score_channels=base_score, metadata=["doctype", "citation"]),
        "winner_no_lal": dict(rank_views=["jina_ce", "adapted_e5", "legalir_jina", "profile_2g_prior03"], score_channels=["jina_ce", "adapted_e5"], metadata=["doctype", "citation"]),
        "winner_no_jina_ce": dict(rank_views=["adapted_e5", "lal_native", "legalir_jina", "profile_2g_prior03"], score_channels=["adapted_e5", "lal_native"], metadata=["doctype", "citation"]),
        "winner_no_adapted_e5": dict(rank_views=["jina_ce", "lal_native", "legalir_jina", "profile_2g_prior03"], score_channels=["jina_ce", "lal_native"], metadata=["doctype", "citation"]),
    }
    predictions = {name: {} for name in experiments}
    feature_counts = {}
    fold_runtime = {}
    all_qids = set(pools)

    for outer, test_ids in folds.items():
        fold_started = time.perf_counter()
        blocked = set(map(str, dup.get(outer, [])))
        train_ids = sorted(all_qids - set(test_ids) - blocked, key=int)
        profile_orders = {name: {} for name in PROFILE_CONFIGS}
        profile_scores = {name: {} for name in PROFILE_CONFIGS}

        test_model = build_profiles(queries, train_ids)
        for qid in test_ids:
            for name, config in PROFILE_CONFIGS.items():
                order, score = score_and_order(questions[qid], test_model, config)
                profile_orders[name][qid] = order
                profile_scores[name][qid] = score

        for inner in folds:
            if inner == outer:
                continue
            inner_ids = [qid for qid in train_ids if fold_for[qid] == inner]
            inner_blocked = set(map(str, dup.get(inner, [])))
            memory_ids = [
                qid for qid in train_ids
                if fold_for[qid] != inner and qid not in inner_blocked
            ]
            model = build_profiles(queries, memory_ids)
            for qid in inner_ids:
                for name, config in PROFILE_CONFIGS.items():
                    order, score = score_and_order(questions[qid], model, config)
                    profile_orders[name][qid] = order
                    profile_scores[name][qid] = score

        local_ids = train_ids + list(test_ids)
        local_pools = {qid: pools[qid] for qid in local_ids}
        rank_features = {name: {qid: frozen_rank[name][qid] for qid in local_ids} for name in frozen_rank}
        score_features = {name: {qid: frozen_score[name][qid] for qid in local_ids} for name in frozen_score}
        for name in PROFILE_CONFIGS:
            rank_features[name] = core.rank_columns(profile_orders[name], local_pools)
            score_features[name] = core.score_columns(profile_scores[name], local_pools)
        local_meta = {name: {qid: value[qid] for qid in local_ids} for name, value in metadata.items()}

        for name, config in experiments.items():
            pred, count = fit_predict(
                config, local_pools, train_ids, test_ids, golds,
                rank_features, score_features, local_meta,
            )
            predictions[name].update(pred)
            feature_counts[name] = count
        fold_runtime[outer] = time.perf_counter() - fold_started
        print(json.dumps({"fold": outer, "seconds": fold_runtime[outer]}), flush=True)

    reference = {
        str(row["qid"]): list(map(str, row["top5"]))
        for row in core.read_jsonl(core.OUT / "HUY_PROFILE_5FOLD_PREDICTIONS.jsonl")
    }
    results = []
    for name in experiments:
        result = {
            "name": name,
            "config": experiments[name],
            "feature_count": feature_counts[name],
            "metrics": core.metrics(predictions[name], golds, folds),
            "paired_vs_profile_2g_reference": core.compare(predictions[name], reference, golds, folds),
        }
        results.append(result)
    results.sort(key=lambda row: row["metrics"]["recall_at_5"], reverse=True)
    best = results[0]
    report = {
        "schema_version": "dsc2026.huy_fasttrack.profile_batch.v1",
        "status": "COMPLETE_NESTED_STRICT_5FOLD_OOF",
        "scope": "Three historically motivated profile views, direct score test, and winner LOCO ablation.",
        "profile_configs": {name: list(config) for name, config in PROFILE_CONFIGS.items()},
        "reference_recall_at_5": core.metrics(reference, golds, folds)["recall_at_5"],
        "best": best,
        "results": results,
        "fold_runtime_seconds": fold_runtime,
        "runtime_seconds": time.perf_counter() - started,
    }
    core.write_json(core.OUT / "HUY_PROFILE_BATCH_REPORT.json", report)
    with (core.OUT / "BEST_PROFILE_BATCH_PREDICTIONS.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for qid in sorted(pools, key=int):
            f.write(json.dumps({"qid": qid, "top5": predictions[best["name"]][qid][:5]}, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(json.dumps({"best": best, "results": [{"name": r["name"], "recall_at_5": r["metrics"]["recall_at_5"], "delta": r["paired_vs_profile_2g_reference"]["delta_recall_at_5"]} for r in results], "runtime_seconds": report["runtime_seconds"]}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
