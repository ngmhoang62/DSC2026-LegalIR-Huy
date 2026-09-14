"""Reconstruct Huy's supervised label-profile rank view under nested V2 folds."""

from __future__ import annotations

import json
import sys
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import run_huy_5fold_fasttrack as core


sys.path.insert(0, str(core.ROOT))
from tune_burst_supervised_profile_bm25 import build_profiles, profile_rank


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
    orders = {
        "jina_ce": jina_order,
        "adapted_e5": e5_orders["adapted_e5"],
        "lal_native": lal_order,
        "legalir_jina": legalir_jina_order,
    }
    score_maps = {
        "jina_ce": jina_scores,
        "adapted_e5": e5_scores["adapted_e5"],
        "lal_native": lal_scores,
    }
    frozen_rank_features = {name: core.rank_columns(values, pools) for name, values in orders.items()}
    score_features = {name: core.score_columns(values, pools) for name, values in score_maps.items()}
    metadata = {"doctype": doctype, "citation": citation}
    config = {
        "rank_views": ["jina_ce", "adapted_e5", "lal_native", "legalir_jina", "huy_profile"],
        "score_channels": ["jina_ce", "adapted_e5", "lal_native"],
        "metadata": ["doctype", "citation"],
    }
    predictions = {}
    fold_runtime = {}
    all_qids = set(pools)
    for outer, test_ids in folds.items():
        fold_started = time.perf_counter()
        outer_blocked = set(map(str, dup.get(outer, [])))
        train_ids = sorted(all_qids - set(test_ids) - outer_blocked, key=int)

        # Target-fold profile: only the four allowed folds.
        test_model = build_profiles(queries, train_ids)
        profile_orders = {
            qid: profile_rank(questions[qid], test_model, 2, 1.2, .75, .3)
            for qid in test_ids
        }

        # Training rows are cross-fit again inside the outer training set, so
        # neither their own fold nor the outer held fold contributes labels.
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
                profile_orders[qid] = profile_rank(
                    questions[qid], model, 2, 1.2, .75, .3
                )
        if set(profile_orders) != set(train_ids) | set(test_ids):
            raise RuntimeError(f"profile population mismatch for {outer}")

        rank_features = dict(frozen_rank_features)
        rank_features["huy_profile"] = core.rank_columns(profile_orders, {
            qid: pools[qid] for qid in profile_orders
        })
        local_pools = {qid: pools[qid] for qid in profile_orders}
        rows = core.make_rows(config, local_pools, rank_features, score_features, metadata)
        x = np.vstack([rows[q] for q in train_ids])
        y = np.concatenate([
            np.asarray([doc in golds[q] for doc in pools[q]], dtype=np.int8)
            for q in train_ids
        ])
        scaler = StandardScaler().fit(x)
        model = LogisticRegression(
            C=.15, class_weight="balanced", solver="liblinear",
            max_iter=3000, random_state=2026,
        ).fit(scaler.transform(x), y)
        for qid in test_ids:
            values = model.decision_function(scaler.transform(rows[qid]))
            predictions[qid] = [
                pools[qid][i] for i in np.lexsort((np.asarray(pools[qid]), -values))
            ]
        fold_runtime[outer] = time.perf_counter() - fold_started
        print(json.dumps({"fold": outer, "seconds": fold_runtime[outer]}), flush=True)

    reference = {
        str(row["qid"]): list(map(str, row["top5"]))
        for row in core.read_jsonl(core.OUT / "BEST_5FOLD_PREDICTIONS.jsonl")
    }
    report = {
        "schema_version": "dsc2026.huy_fasttrack.profile_rank_port.v1",
        "status": "COMPLETE_NESTED_STRICT_5FOLD_OOF",
        "component": "Huy supervised label-profile BM25 rank view",
        "profile_config": {"max_ngram": 2, "k1": 1.2, "b": .75, "prior_power": .3},
        "isolation": "For each outer fold, target profiles use four folds; every outer-training query profile additionally excludes its own entire inner fold and duplicate-linked qids.",
        "config": config,
        "metrics": core.metrics(predictions, golds, folds),
        "paired_vs_current_lr_best": core.compare(predictions, reference, golds, folds),
        "fold_runtime_seconds": fold_runtime,
        "runtime_seconds": time.perf_counter() - started,
    }
    report["verdict"] = "KEEP" if report["paired_vs_current_lr_best"]["delta_recall_at_5"] >= .001 else "DROP"
    core.write_json(core.OUT / "HUY_PROFILE_RANK_PORT_REPORT.json", report)
    with (core.OUT / "HUY_PROFILE_5FOLD_PREDICTIONS.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for qid in sorted(pools, key=int):
            f.write(json.dumps({"qid": qid, "top5": predictions[qid][:5]}, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
