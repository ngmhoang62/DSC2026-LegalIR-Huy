"""Supervised BM25/Bayes document profiles learned from every train qrel."""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from benchmark_burst_v4_full_sqlite import tokens
from tune_burst_kernel_posterior import load_queries, robust_rankings
from tune_burst_pairwise import blend_rankings, fixed_metrics


def features(text, max_ngram=3):
    words = tokens(text)
    out = []
    for n in range(1, max_ngram + 1):
        out.extend(f"{n}:" + " ".join(words[i:i+n])
                   for i in range(len(words) - n + 1))
    return set(out)


def build_profiles(queries, memory_ids):
    postings = defaultdict(Counter)
    doc_length = Counter()
    label_frequency = Counter()
    for q in memory_ids:
        feats = features(queries[q][0], 3)
        for doc in queries[q][1]:
            label_frequency[doc] += 1
            for feat in feats:
                postings[feat][doc] += 1
                doc_length[doc] += 1
    doc_frequency = {feat: len(docs) for feat, docs in postings.items()}
    return postings, doc_frequency, doc_length, label_frequency


def profile_rank(text, model, max_ngram, k1, b, prior_power):
    postings, doc_frequency, doc_length, label_frequency = model
    docs_count = max(len(doc_length), 1)
    avg_length = sum(doc_length.values()) / docs_count
    scores = defaultdict(float)
    for feat in features(text, max_ngram):
        posting = postings.get(feat)
        if not posting:
            continue
        df = doc_frequency[feat]
        idf = math.log1p((docs_count - df + .5) / (df + .5))
        # Higher-order phrases carry more specific evidence.
        order = int(feat[0])
        phrase_weight = (1.0, 1.35, 1.65)[order - 1]
        for doc, tf in posting.items():
            norm = k1 * (1.0 - b + b * doc_length[doc] / avg_length)
            scores[doc] += phrase_weight * idf * tf * (k1 + 1.0) / (tf + norm)
    max_freq = max(label_frequency.values(), default=1)
    for doc in scores:
        prior = (label_frequency[doc] + .5) / (max_freq + .5)
        scores[doc] *= prior ** prior_power
    return sorted(scores, key=lambda d: (-scores[d], d))[:100]


def main():
    root = Path(__file__).resolve().parent
    queries = load_queries(root)
    qids = list(queries)
    splits = {
        "tune_a": qids[700:750],
        "validation_a": qids[750:850],
        "tune_b": qids[1150:1250],
        "validation_b": qids[1250:1350],
    }
    eval_ids = sum(splits.values(), [])
    baseline = robust_rankings(root, queries, eval_ids)

    models = {}
    for split, ids in splits.items():
        held = set(ids)
        print(f"Building supervised profiles for {split}", flush=True)
        models[split] = build_profiles(queries, [q for q in qids if q not in held])

    # Orthogonal coarse grid first.  The earlier 630-point dense grid repeated
    # many effectively identical rankings and could outlive command runners.
    params = [(n, k1, b, pp)
              for n in (1, 2, 3)
              for k1 in (.5, 1.2, 2.5)
              for b in (.25, .75, 1.0)
              for pp in (-.3, 0.0, .3)]
    rankings = []
    tune_ids = splits["tune_a"] + splits["tune_b"]
    tune_gold = {q: queries[q] for q in tune_ids}
    profile_trials = []
    for i, config in enumerate(params, 1):
        ranked = {}
        for split, ids in splits.items():
            for q in ids:
                ranked[q] = profile_rank(queries[q][0], models[split], *config)
        metrics, _ = fixed_metrics({q: ranked[q] for q in tune_ids}, tune_gold)
        profile_trials.append((metrics["Recall@5"], metrics["Precision@5"],
                               metrics["nDCG@10"], config, metrics, ranked))
        if i % 100 == 0:
            print(f"Profile configs {i}/{len(params)}", flush=True)
    profile_trials.sort(reverse=True, key=lambda x: (x[0], x[1], x[2]))

    trials = []
    for _, _, _, config, profile_metrics, ranked in profile_trials[:20]:
        for alpha in (.02, .05, .08, .10, .15, .20, .25, .30, .40, .50,
                      .65, .80, 1.0):
            for k in (0, 2, 5, 10, 20, 40, 80):
                fused = blend_rankings({q: ranked[q] for q in tune_ids},
                                       {q: baseline[q] for q in tune_ids}, alpha, k)
                metrics, _ = fixed_metrics(fused, tune_gold)
                trials.append((metrics["Recall@5"], metrics["Precision@5"],
                               metrics["nDCG@10"], config, alpha, k, metrics,
                               profile_metrics, ranked))
    trials.sort(reverse=True, key=lambda x: (x[0], x[1], x[2]))

    reports = []
    for trial in trials[:300]:
        _, _, _, config, alpha, k, tune_metrics, profile_metrics, ranked = trial
        report = {
            "params": {"max_ngram": config[0], "k1": config[1], "b": config[2],
                       "prior_power": config[3], "alpha": alpha, "rrf_k": k},
            "profile_only_tune": profile_metrics, "fused_tune": tune_metrics,
            "splits": {},
        }
        safe = True
        gain = 0.0
        for split in ("validation_a", "validation_b"):
            ids = splits[split]
            gold = {q: queries[q] for q in ids}
            fused = blend_rankings({q: ranked[q] for q in ids},
                                   {q: baseline[q] for q in ids}, alpha, k)
            bm, bp = fixed_metrics({q: baseline[q] for q in ids}, gold)
            fm, fp = fixed_metrics(fused, gold)
            paired = {"wins": sum(a > b for a, b in zip(fp, bp)),
                      "ties": sum(a == b for a, b in zip(fp, bp)),
                      "losses": sum(a < b for a, b in zip(fp, bp))}
            safe &= paired["losses"] == 0 and fm["Precision@5"] >= bm["Precision@5"]
            gain += fm["Recall@5"] - bm["Recall@5"]
            report["splits"][split] = {"baseline": bm, "profile_fusion": fm,
                                        "paired": paired}
        report["safe"] = bool(safe)
        report["total_validation_recall_gain"] = gain
        reports.append(report)
    reports.sort(key=lambda x: (x["safe"], x["total_validation_recall_gain"],
                                x["fused_tune"]["Recall@5"],
                                x["fused_tune"]["nDCG@10"]), reverse=True)
    output = {"best": reports[0], "top_trials": reports[:30]}
    path = root / "burst_supervised_profile_bm25_validation.json"
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(reports[0], ensure_ascii=False, indent=2))
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
