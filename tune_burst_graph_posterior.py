"""Bayesian document co-relevance graph reranker for CPU-only BURST."""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from tune_burst_kernel_posterior import load_queries, robust_rankings
from tune_burst_pairwise import fixed_metrics


def build_graph(queries, memory_ids):
    frequency = Counter()
    edges = Counter()
    for q in memory_ids:
        docs = sorted(queries[q][1])
        frequency.update(docs)
        for i, left in enumerate(docs):
            for right in docs[i + 1:]:
                edges[left, right] += 1
                edges[right, left] += 1
    adjacency = defaultdict(list)
    for (left, right), count in edges.items():
        adjacency[left].append((right, count))
    return frequency, adjacency, edges


def edge_strength(left, right, count, frequency, mode):
    fl = frequency[left]
    fr = frequency[right]
    if mode == "conditional":
        return count / (fl + 1.0)
    if mode == "cosine":
        return count / math.sqrt((fl + .5) * (fr + .5))
    if mode == "jaccard":
        return count / (fl + fr - count + 1.0)
    if mode == "lift":
        # Positive PMI-like association with conservative square-root scaling.
        return count / ((fl + .5) * (fr + .5)) ** .35
    raise ValueError(mode)


def graph_rerank(base, frequency, adjacency, seed_depth, seed_power, mode,
                 min_edge, alpha, base_k):
    graph = defaultdict(float)
    for rank, seed in enumerate(base[:seed_depth], 1):
        seed_weight = 1.0 / (rank ** seed_power)
        for target, count in adjacency.get(seed, ()):
            if count >= min_edge:
                graph[target] += seed_weight * edge_strength(
                    seed, target, count, frequency, mode
                )
    graph_max = max(graph.values(), default=0.0)
    base_pos = {doc: rank for rank, doc in enumerate(base, 1)}
    docs = set(base_pos) | set(graph)
    scores = {}
    for doc in docs:
        bscore = 1.0 / (base_k + base_pos.get(doc, 100000))
        gscore = graph.get(doc, 0.0) / graph_max if graph_max else 0.0
        scores[doc] = bscore + alpha * gscore / (base_k + 1.0)
    return sorted(docs, key=lambda d: (-scores[d], d))[:100]


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

    graphs = {}
    pair_coverage = {}
    for split, ids in splits.items():
        held = set(ids)
        memory = [q for q in qids if q not in held]
        frequency, adjacency, edges = build_graph(queries, memory)
        graphs[split] = (frequency, adjacency)
        multi = [q for q in ids if len(queries[q][1]) > 1]
        covered = 0
        for q in multi:
            docs = sorted(queries[q][1])
            if any(edges[a, b] for i, a in enumerate(docs) for b in docs[i + 1:]):
                covered += 1
        pair_coverage[split] = {"multi_queries": len(multi),
                                "pair_seen_in_memory": covered}

    configs = []
    for seed_depth in (1, 2, 3, 5, 8, 10):
        for seed_power in (.5, 1.0, 1.5, 2.0):
            for mode in ("conditional", "cosine", "jaccard", "lift"):
                for min_edge in (1, 2, 3):
                    for alpha in (.02, .05, .10, .15, .20, .30, .40, .60, .80, 1.0, 1.5, 2.0):
                        for base_k in (0, 5, 20, 40):
                            configs.append((seed_depth, seed_power, mode,
                                            min_edge, alpha, base_k))

    tune_ids = splits["tune_a"] + splits["tune_b"]
    tune_gold = {q: queries[q] for q in tune_ids}
    trials = []
    for config in configs:
        ranked = {}
        for split in ("tune_a", "tune_b"):
            frequency, adjacency = graphs[split]
            for q in splits[split]:
                ranked[q] = graph_rerank(baseline[q], frequency, adjacency, *config)
        metrics, _ = fixed_metrics(ranked, tune_gold)
        trials.append((metrics["Recall@5"], metrics["Precision@5"],
                       metrics["nDCG@10"], config, metrics))
    trials.sort(reverse=True, key=lambda x: (x[0], x[1], x[2]))

    reports = []
    for _, _, _, config, tune_metrics in trials[:300]:
        report = {
            "params": dict(zip(("seed_depth", "seed_power", "mode", "min_edge",
                                 "alpha", "base_k"), config)),
            "tune": tune_metrics, "splits": {},
        }
        safe = True
        gain = 0.0
        for split in ("validation_a", "validation_b"):
            frequency, adjacency = graphs[split]
            ids = splits[split]
            gold = {q: queries[q] for q in ids}
            ranked = {q: graph_rerank(baseline[q], frequency, adjacency, *config)
                      for q in ids}
            bm, bp = fixed_metrics({q: baseline[q] for q in ids}, gold)
            gm, gp = fixed_metrics(ranked, gold)
            paired = {"wins": sum(a > b for a, b in zip(gp, bp)),
                      "ties": sum(a == b for a, b in zip(gp, bp)),
                      "losses": sum(a < b for a, b in zip(gp, bp))}
            safe &= paired["losses"] == 0 and gm["Precision@5"] >= bm["Precision@5"]
            gain += gm["Recall@5"] - bm["Recall@5"]
            report["splits"][split] = {"baseline": bm, "graph": gm,
                                        "paired": paired}
        report["safe"] = bool(safe)
        report["total_validation_recall_gain"] = gain
        reports.append(report)
    reports.sort(key=lambda x: (x["safe"], x["total_validation_recall_gain"],
                                x["tune"]["Recall@5"], x["tune"]["nDCG@10"]),
                 reverse=True)
    output = {"pair_coverage": pair_coverage, "best": reports[0],
              "top_trials": reports[:30]}
    path = root / "burst_graph_posterior_validation.json"
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"pair_coverage": pair_coverage, "best": reports[0]},
                     ensure_ascii=False, indent=2))
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
