"""Pairwise empirical-Bayes reranker optimized for top-5 swaps."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from tune_burst_empirical_bayes_ltr import (
    augmented_features, label_frequency, load_caches,
)
from tune_burst_kernel_posterior import load_queries, robust_rankings
from tune_burst_pairwise import blend_rankings, fixed_metrics
from tune_burst_phrases import multi_rrf


def features_for(cache, queries, ids, frequency, training=False):
    output = {}
    for q in ids:
        candidates, x = augmented_features(
            cache[q], frequency, queries[q][1] if training else ()
        )
        lists = [[d for d, _ in source] for source in cache[q]]
        base_rank = multi_rrf(lists, [.063, .357, .28, .30], 5)
        rankmap = {d: i + 1 for i, d in enumerate(base_rank)}
        # Explicit monotone baseline likelihood at several temperatures.
        extra = np.asarray([
            [1.0 / (k + rankmap.get(d, 100000)) for k in (0, 2, 5, 10, 20, 40, 80)]
            for d in candidates
        ], dtype=np.float32)
        output[q] = (candidates, np.hstack((x, extra)), base_rank)
    return output


def pairwise_matrix(features, queries, ids, negative_depth):
    xs, ys, weights = [], [], []
    for q in ids:
        candidates, x, base_rank = features[q]
        gold = queries[q][1]
        posmap = {d: i for i, d in enumerate(candidates)}
        positives = [posmap[d] for d in gold if d in posmap]
        negatives = [posmap[d] for d in base_rank
                     if d not in gold and d in posmap][:negative_depth]
        if not positives or not negatives:
            continue
        scale = 1.0 / (len(positives) * len(negatives))
        for pi in positives:
            for ni in negatives:
                diff = x[pi] - x[ni]
                xs.extend((diff, -diff))
                ys.extend((1, 0))
                weights.extend((scale, scale))
    return (np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.int8),
            np.asarray(weights, dtype=np.float32))


def rank(model, scaler, features, ids):
    output = {}
    for q in ids:
        candidates, x, _ = features[q]
        score = model.decision_function(scaler.transform(x))
        output[q] = [candidates[i] for i in np.argsort(-score, kind="stable")[:100]]
    return output


def main():
    root = Path(__file__).resolve().parent
    queries = load_queries(root)
    qids = list(queries)
    train_ids = qids[:700] + qids[850:1150]
    tune_ids = qids[700:750]
    splits = {
        "validation_a": qids[750:850],
        "fresh_a": qids[1150:1250],
        "validation_b": qids[1250:1350],
    }
    eval_ids = tune_ids + sum(splits.values(), [])
    cache = load_caches(root)
    baseline = robust_rankings(root, queries, eval_ids)
    full_frequency = label_frequency(queries, set())
    print("Building pairwise empirical-Bayes features", flush=True)
    train_features = features_for(cache, queries, train_ids, full_frequency, True)
    tune_features = features_for(
        cache, queries, tune_ids, label_frequency(queries, set(tune_ids))
    )
    eval_features = {}
    for name, ids in splits.items():
        eval_features.update(features_for(
            cache, queries, ids, label_frequency(queries, set(ids))
        ))

    tune_gold = {q: queries[q] for q in tune_ids}
    trials = []
    for negative_depth in (5, 10, 20, 40, 80):
        x, y, weights = pairwise_matrix(train_features, queries, train_ids, negative_depth)
        scaler = StandardScaler().fit(x)
        z = scaler.transform(x)
        print(f"depth={negative_depth}: pair rows={len(y):,}", flush=True)
        for c in (.003, .01, .03, .1, .3, 1.0, 3.0):
            model = LogisticRegression(C=c, solver="liblinear", max_iter=2000)
            model.fit(z, y, sample_weight=weights)
            tune_rank = rank(model, scaler, tune_features, tune_ids)
            metrics, _ = fixed_metrics(tune_rank, tune_gold)
            trials.append((metrics["Recall@5"], metrics["Precision@5"],
                           metrics["nDCG@10"], negative_depth, c, model, scaler,
                           metrics, tune_rank))
    trials.sort(reverse=True, key=lambda x: (x[0], x[1], x[2]))

    fusion = []
    for trial in trials[:20]:
        _, _, _, depth, c, model, scaler, model_metrics, tune_rank = trial
        for alpha in (.02, .05, .08, .10, .15, .20, .25, .30, .40, .50,
                      .65, .80, 1.0):
            for k in (0, 2, 5, 10, 20, 40, 80):
                fused = blend_rankings(tune_rank,
                                       {q: baseline[q] for q in tune_ids}, alpha, k)
                metrics, _ = fixed_metrics(fused, tune_gold)
                fusion.append((metrics["Recall@5"], metrics["Precision@5"],
                               metrics["nDCG@10"], depth, c, model, scaler,
                               alpha, k, metrics, model_metrics))
    fusion.sort(reverse=True, key=lambda x: (x[0], x[1], x[2]))

    reports = []
    for trial in fusion[:300]:
        _, _, _, depth, c, model, scaler, alpha, k, tune_m, model_m = trial
        eval_rank = rank(model, scaler, eval_features, sum(splits.values(), []))
        report = {"pairwise": {"negative_depth": depth, "C": c},
                  "fusion": {"alpha": alpha, "rrf_k": k},
                  "model_tune": model_m, "fusion_tune": tune_m, "splits": {}}
        safe = True
        gain = 0.0
        for name, ids in splits.items():
            gold = {q: queries[q] for q in ids}
            fused = blend_rankings({q: eval_rank[q] for q in ids},
                                   {q: baseline[q] for q in ids}, alpha, k)
            bm, bp = fixed_metrics({q: baseline[q] for q in ids}, gold)
            fm, fp = fixed_metrics(fused, gold)
            paired = {"wins": sum(a > b for a, b in zip(fp, bp)),
                      "ties": sum(a == b for a, b in zip(fp, bp)),
                      "losses": sum(a < b for a, b in zip(fp, bp))}
            safe &= paired["losses"] == 0 and fm["Precision@5"] >= bm["Precision@5"]
            gain += fm["Recall@5"] - bm["Recall@5"]
            report["splits"][name] = {"baseline": bm, "pairwise": fm,
                                       "paired": paired}
        report["safe"] = bool(safe)
        report["total_validation_recall_gain"] = gain
        reports.append((report, model, scaler))
    reports.sort(key=lambda x: (x[0]["safe"], x[0]["total_validation_recall_gain"],
                                x[0]["fusion_tune"]["Recall@5"],
                                x[0]["fusion_tune"]["nDCG@10"]), reverse=True)
    best, model, scaler = reports[0]
    out = root / "results" / "burst_empirical_pairwise"
    out.mkdir(parents=True, exist_ok=True)
    (out / "model.pkl").write_bytes(pickle.dumps(
        {"model": model, "scaler": scaler, "report": best,
         "feature_version": "empirical-pairwise-v1"}, protocol=5
    ))
    path = root / "burst_empirical_pairwise_validation.json"
    path.write_text(json.dumps({"best": best,
                                "top_trials": [x[0] for x in reports[:20]]},
                               ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(best, ensure_ascii=False, indent=2))
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
