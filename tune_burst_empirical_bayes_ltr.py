"""Empirical-Bayes BURST ranker with cold-start and source-consensus features."""

from __future__ import annotations

import json
import pickle
from collections import Counter
from pathlib import Path

import numpy as np
from xgboost import XGBRanker

from tune_burst_kernel_posterior import load_queries, robust_rankings
from tune_burst_pairwise import blend_rankings, fixed_metrics
from tune_burst_score_ltr import score_features


def load_caches(root: Path):
    large = root / "results" / "burst_large_ltr"
    main = pickle.loads((large / "retrieval_train1000_tune50_val100.pkl").read_bytes())["cache"]
    fresh1 = pickle.loads((large / "fresh_1151_1250_retrieval.pkl").read_bytes())["cache"]
    fresh2 = pickle.loads((large / "fresh_1251_1350_retrieval.pkl").read_bytes())["cache"]
    return main | fresh1 | fresh2


def label_frequency(queries, excluded):
    return Counter(d for q, (_, gold) in queries.items() if q not in excluded for d in gold)


def augmented_features(lists, frequency, own_gold=()):
    candidates, base = score_features(lists)
    freq = np.asarray([max(0, frequency[d] - (d in own_gold)) for d in candidates],
                      dtype=np.float32)
    logfreq = np.log1p(freq)
    maxlog = max(float(logfreq.max()), 1.0)
    prior = logfreq / maxlog
    unseen = (freq == 0).astype(np.float32)
    rare = (freq <= 1).astype(np.float32)

    rr = base[:, :4]
    score = base[:, 4:8]
    present = base[:, 8:12]
    lex_support = present[:, :3].sum(axis=1)
    lex_rr = rr[:, :3].sum(axis=1)
    lex_score = score[:, :3].sum(axis=1)
    best_lex = score[:, :3].max(axis=1)
    mem_rr = rr[:, 3]
    mem_score = score[:, 3]
    consensus2 = (lex_support >= 2).astype(np.float32)
    consensus3 = (lex_support >= 3).astype(np.float32)
    cold_consensus = unseen * consensus2
    lexical_minus_memory = lex_score - mem_score
    rr_minus_memory = lex_rr - mem_rr

    extra = np.column_stack((
        prior, unseen, rare, lex_support / 3.0, lex_rr, lex_score, best_lex,
        mem_rr, mem_score, consensus2, consensus3, cold_consensus,
        lexical_minus_memory, rr_minus_memory,
        prior * mem_score, prior * mem_rr,
        unseen * lex_score, unseen * lex_rr, unseen * best_lex,
        consensus2 * best_lex, consensus2 * lexical_minus_memory,
    )).astype(np.float32)
    return candidates, np.hstack((base, extra))


def make_features(cache, queries, ids, frequency, training=False):
    out = {}
    for q in ids:
        own = queries[q][1] if training else ()
        out[q] = augmented_features(cache[q], frequency, own)
    return out


def matrices(features, queries, ids, hard_depth=80):
    xs, ys, groups = [], [], []
    for q in ids:
        candidates, x = features[q]
        gold = queries[q][1]
        positives = [i for i, d in enumerate(candidates) if d in gold]
        negatives = [i for i, d in enumerate(candidates) if d not in gold][:hard_depth]
        chosen = positives + negatives
        if not positives or not negatives:
            continue
        xs.append(x[chosen])
        ys.extend([1] * len(positives) + [0] * len(negatives))
        groups.append(len(chosen))
    return np.vstack(xs), np.asarray(ys, dtype=np.int8), groups


def rank(model, features, ids):
    output = {}
    for q in ids:
        candidates, x = features[q]
        score = model.predict(x)
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

    # Training prior is leave-one-query-out.  Each evaluation block gets a
    # fully block-excluded prior, exactly matching real unknown labels.
    full_frequency = label_frequency(queries, set())
    print("Building empirical-Bayes features", flush=True)
    train_features = make_features(cache, queries, train_ids, full_frequency, training=True)
    tune_frequency = label_frequency(queries, set(tune_ids))
    tune_features = make_features(cache, queries, tune_ids, tune_frequency)
    eval_features = {}
    for name, ids in splits.items():
        frequency = label_frequency(queries, set(ids))
        eval_features.update(make_features(cache, queries, ids, frequency))

    train_x, train_y, train_groups = matrices(train_features, queries, train_ids)
    tune_x, tune_y, tune_groups = matrices(tune_features, queries, tune_ids)
    print(f"Training pairs={len(train_y):,}, positives={int(train_y.sum())}", flush=True)
    configs = [
        (2, .02, 250, 3, 10), (2, .03, 180, 3, 10),
        (2, .04, 160, 5, 20), (3, .015, 280, 5, 10),
        (3, .025, 220, 5, 20), (3, .035, 180, 8, 20),
        (4, .015, 260, 8, 20), (4, .025, 200, 10, 30),
    ]
    tune_gold = {q: queries[q] for q in tune_ids}
    models = []
    for depth, rate, trees, child, pairs in configs:
        model = XGBRanker(
            objective="rank:ndcg", eval_metric="ndcg@5", tree_method="hist",
            n_estimators=trees, max_depth=depth, learning_rate=rate,
            min_child_weight=child, subsample=.85, colsample_bytree=.9,
            reg_lambda=8.0, reg_alpha=.05, n_jobs=2,
            lambdarank_pair_method="topk",
            lambdarank_num_pair_per_sample=pairs, random_state=2026,
        )
        model.fit(train_x, train_y, group=train_groups,
                  eval_set=[(tune_x, tune_y)], eval_group=[tune_groups], verbose=False)
        tune_rank = rank(model, tune_features, tune_ids)
        metrics, _ = fixed_metrics(tune_rank, tune_gold)
        models.append((metrics["Recall@5"], metrics["Precision@5"],
                       metrics["nDCG@10"], (depth, rate, trees, child, pairs),
                       model, metrics, tune_rank))
    models.sort(reverse=True, key=lambda x: (x[0], x[1], x[2]))

    fusion_trials = []
    for model_trial in models:
        _, _, _, config, model, model_metrics, tune_rank = model_trial
        for alpha in (.05, .10, .15, .20, .25, .30, .35, .40, .50, .60,
                      .70, .80, .90, 1.0):
            for k in (0, 2, 5, 10, 20, 40, 80):
                fused = blend_rankings(tune_rank,
                                       {q: baseline[q] for q in tune_ids}, alpha, k)
                metrics, _ = fixed_metrics(fused, tune_gold)
                fusion_trials.append((metrics["Recall@5"], metrics["Precision@5"],
                                      metrics["nDCG@10"], config, model, alpha, k,
                                      metrics, model_metrics))
    fusion_trials.sort(reverse=True, key=lambda x: (x[0], x[1], x[2]))

    reports = []
    for trial in fusion_trials[:200]:
        _, _, _, config, model, alpha, k, tune_metrics, model_metrics = trial
        report = {
            "model": {"depth": config[0], "rate": config[1], "trees": config[2],
                      "min_child": config[3], "pairs": config[4]},
            "fusion": {"alpha": alpha, "rrf_k": k},
            "model_tune": model_metrics, "fusion_tune": tune_metrics, "splits": {},
        }
        safe = True
        gain = 0.0
        eval_rank = rank(model, eval_features, sum(splits.values(), []))
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
            report["splits"][name] = {"baseline": bm, "empirical_bayes": fm,
                                       "paired": paired}
        report["safe"] = bool(safe)
        report["total_validation_recall_gain"] = gain
        reports.append((report, model))
    reports.sort(key=lambda x: (x[0]["safe"], x[0]["total_validation_recall_gain"],
                                x[0]["fusion_tune"]["Recall@5"],
                                x[0]["fusion_tune"]["nDCG@10"]), reverse=True)
    best_report, best_model = reports[0]
    output_dir = root / "results" / "burst_empirical_bayes"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "model.pkl").write_bytes(pickle.dumps({
        "model": best_model, "report": best_report, "feature_version": "ebayes-v1"
    }, protocol=5))
    path = root / "burst_empirical_bayes_validation.json"
    path.write_text(json.dumps({"best": best_report,
                                "top_trials": [x[0] for x in reports[:20]]},
                               ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(best_report, ensure_ascii=False, indent=2))
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
