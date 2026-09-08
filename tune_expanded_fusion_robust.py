"""Leave-one-block-out weight selection for the expanded multi-view fusion.

The published expanded-rerank weights were picked on blocks a+b and lost to the
multistage baseline on block c.  This searches the same view stack but scores a
configuration by its worst holdout block, so the weights that ship generalize.
"""

from __future__ import annotations

import itertools
import json
import pickle
from pathlib import Path

import numpy as np

from benchmark_expanded_rerank_holdouts import load_expanded
from tune_burst_kernel_posterior import load_queries
from tune_burst_multistage_posterior import weighted_rrf
from tune_burst_pairwise import fixed_metrics


TAGS = {"a": "validation_a", "b": "fresh_1251_1350", "c": "fresh_1351_1450",
        "d": "fresh_1451_1750"}


def rank_by(candidates, scores):
    return {q: sorted(candidates[q],
                      key=lambda d: (-scores.get(q, {}).get(d, -1e9), d))
            for q in candidates}


def build_views(root, expanded_depth=20):
    queries = load_queries(root)
    qids = list(queries)
    blocks = {"a": qids[750:850], "b": qids[1250:1350], "c": qids[1350:1450],
              "d": qids[1450:1750]}
    all_ids = sum(blocks.values(), [])
    load = lambda p: pickle.loads((root / p).read_bytes())

    old_jina = load("results/jina_reranker/holdout_scores_finetuned.pkl")["scores"]
    old_dense = load("results/aiteamvn_dense/holdout_scores_512.pkl")["scores"]
    old_e5 = load("results/e5_dense/holdout_scores.pkl")["scores"]
    vi_rerank = load("results/vietnamese_reranker/holdout_scores_512_finetuned.pkl")["scores"]
    expansion = load("results/dense_expansion/union50_scores.pkl")["scores"]
    expanded_scores = load("results/expanded_rerank/scores.pkl")

    raw, _, expanded = load_expanded(root, queries, {TAGS[n]: b for n, b in blocks.items()},
                                     expansion)
    base = {q: list(old_jina[q]) for q in all_ids}
    candidates = {q: list(dict.fromkeys(base[q] + expanded[q][:expanded_depth]))
                  for q in all_ids}

    views = {
        "base": base,
        "expanded": {q: expanded[q][:expanded_depth] for q in all_ids},
        "raw": {q: raw[q][:20] for q in all_ids},
        "jina": rank_by(candidates, expanded_scores["jina"]),
        "dense": rank_by(candidates, expanded_scores["dense"]),
        "vi": rank_by(base, vi_rerank),
        "e5": rank_by(base, old_e5),
        "old_jina": rank_by(base, old_jina),
        "old_dense": rank_by(base, old_dense),
    }
    return queries, blocks, all_ids, candidates, views


def evaluate(views, names, weights, k, queries, ids):
    ranked = weighted_rrf([{q: views[n][q] for q in ids} for n in names], weights, k)
    return fixed_metrics(ranked, {q: queries[q] for q in ids})


def simplex(n, step):
    """All weight vectors on a grid of the n-simplex."""
    steps = int(round(1 / step))
    for cut in itertools.combinations(range(1, steps + n), n - 1):
        parts = np.diff((0,) + cut + (steps + n,)) - 1
        yield tuple(float(p) * step for p in parts)


def search(views, names, queries, blocks, step, ks):
    """Score every grid point by its worst block; report per-block detail."""
    trials = []
    for weights in simplex(len(names), step):
        for k in ks:
            per_block = {}
            for name, ids in blocks.items():
                m, _ = evaluate(views, names, weights, k, queries, ids)
                per_block[name] = m
            recalls = [per_block[n]["Recall@5"] for n in blocks]
            trials.append({
                "views": names, "weights": weights, "rrf_k": k,
                "min_recall": min(recalls), "mean_recall": float(np.mean(recalls)),
                "mean_precision": float(np.mean([per_block[n]["Precision@5"]
                                                 for n in blocks])),
                "mean_f2": float(np.mean([per_block[n]["F2@5"] for n in blocks])),
                "blocks": per_block,
            })
    trials.sort(reverse=True, key=lambda t: (t["min_recall"], t["mean_recall"],
                                             t["mean_precision"]))
    return trials


def lobo(views, names, queries, blocks, step, ks):
    """Honest estimate: tune on two blocks, report the held-out third."""
    out = {}
    for held in blocks:
        tune_blocks = {n: ids for n, ids in blocks.items() if n != held}
        best = search(views, names, queries, tune_blocks, step, ks)[0]
        m, _ = evaluate(views, names, best["weights"], best["rrf_k"], queries,
                        blocks[held])
        out[held] = {"weights": best["weights"], "rrf_k": best["rrf_k"],
                     "held_out": m}
    out["mean_held_out_recall"] = float(np.mean(
        [out[h]["held_out"]["Recall@5"] for h in blocks]))
    out["mean_held_out_precision"] = float(np.mean(
        [out[h]["held_out"]["Precision@5"] for h in blocks]))
    return out


def main():
    root = Path(__file__).resolve().parent
    queries, blocks, all_ids, candidates, views = build_views(root)

    report = {"candidate_ceiling": {}, "single_views": {}, "stacks": {}}
    for name, ids in blocks.items():
        report["candidate_ceiling"][name] = sum(
            len(set(candidates[q]) & queries[q][1]) / len(queries[q][1])
            for q in ids) / len(ids)
    for view in views:
        report["single_views"][view] = {
            name: fixed_metrics({q: views[view][q] for q in ids},
                                {q: queries[q] for q in ids})[0]["Recall@5"]
            for name, ids in blocks.items()}

    # Published expanded-rerank configuration, for reference.
    published = ["base", "expanded", "jina", "dense"]
    report["published"] = {
        name: evaluate(views, published, (.25, .35, .05, .35), 2, queries, ids)[0]
        for name, ids in blocks.items()}

    stacks = {
        "four_view": (["base", "expanded", "jina", "dense"], .05),
        "five_view_vi": (["base", "expanded", "jina", "dense", "vi"], .10),
        "six_view": (["base", "expanded", "jina", "dense", "vi", "e5"], .20),
    }
    ks = (0, 2, 5, 10, 20, 40)
    for label, (names, step) in stacks.items():
        trials = search(views, names, queries, blocks, step, ks)
        report["stacks"][label] = {
            "grid_points": len(trials),
            "top": trials[:5],
            "lobo": lobo(views, names, queries, blocks, step, ks),
        }
        best = trials[0]
        print(f"{label}: min_recall={best['min_recall']:.4f} "
              f"mean={best['mean_recall']:.4f} w={best['weights']} k={best['rrf_k']} "
              f"| LOBO mean held-out recall="
              f"{report['stacks'][label]['lobo']['mean_held_out_recall']:.4f}",
              flush=True)

    path = root / "burst_expanded_fusion_robust.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {path}", flush=True)


if __name__ == "__main__":
    main()
