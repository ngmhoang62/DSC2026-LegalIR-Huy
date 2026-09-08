"""Does the dense-retrieved candidate pool actually raise Recall@5?

A bigger pool lifts the ceiling but also adds distractors, so the only test that
matters is the same leave-one-block-out comparison used for every other change.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np

from tune_burst_multistage_posterior import weighted_rrf
from tune_burst_pairwise import fixed_metrics
from tune_deep_passage_fusion import lobo
from tune_expanded_fusion_robust import build_views
from tune_expanded_fusion_selection import holdout_scores


def build_training(root, depth=20, cap=16,
                   extended_scores_path="results/corpus_index/holdout_extended_scores.pkl"):
    """Holdout views, candidates and score channels for the shipped configuration.

    The submission trains its ranker on exactly this, so the feature layout on the
    public side has to be produced the same way.  cap=32 (tune_corpus_cap32_fusion.py)
    beats cap=16 on LOBO Recall@5 (0.9481 -> 0.9511, no block regressing) and is what
    the shipped submission now uses; cap16 stays the default here since several
    older one-off diagnostic scripts import this function and still expect it, and
    cap16/cap32 extended_scores files are NOT interchangeable (each is keyed to that
    cap's own extended candidate set) -- pass extended_scores_path explicitly to
    avoid silently mixing them.
    """
    queries, blocks, all_ids, candidates, views = build_views(root)
    load = lambda p: pickle.loads((root / p).read_bytes())
    e5 = load("results/e5_dense/holdout_scores.pkl")["scores"]
    dense_saved = load(f"results/corpus_index/holdout_dense_rank_cap{cap}.pkl")
    extended_scores = load(extended_scores_path)
    corpus_rank, corpus_score = dense_saved["ranking"], dense_saved["scores"]

    extended = {q: list(dict.fromkeys(list(candidates[q]) + corpus_rank[q][:depth]))
                for q in all_ids}
    local = dict(views)
    for name, table in (("jina", extended_scores["jina"]),
                        ("dense", extended_scores["dense"])):
        local[name] = {q: sorted(extended[q],
                                 key=lambda d: (-table[q].get(d, -1e9), d))
                       for q in all_ids}
    # 4-fold LOBO on N=600 (fresh block "d" added beyond a/b/c): corpus as a full
    # RRF view beats corpus-as-channel-only, 4 wins/1 loss/595 ties.
    local["corpus"] = {q: [d for d in corpus_rank[q] if d in set(extended[q])]
                       for q in all_ids}
    scores = {"jina": extended_scores["jina"], "dense": extended_scores["dense"],
              "expansion": holdout_scores(root)["expansion"], "e5": e5,
              "corpus": {q: {d: corpus_score[q].get(d, -1.0) for d in extended[q]}
                         for q in all_ids}}
    return queries, blocks, all_ids, extended, local, scores


def main():
    root = Path(__file__).resolve().parent
    queries, blocks, all_ids, candidates, views = build_views(root)
    base_scores = holdout_scores(root)
    e5 = pickle.loads(
        (root / "results/e5_dense/holdout_scores.pkl").read_bytes())["scores"]
    dense_saved = pickle.loads(
        (root / "results/corpus_index/holdout_dense_rank_cap16.pkl").read_bytes())
    extended_scores = pickle.loads(
        (root / "results/corpus_index/holdout_extended_scores.pkl").read_bytes())
    corpus_rank, corpus_score = dense_saved["ranking"], dense_saved["scores"]

    report = {"shipped": lobo(views, ["base", "expanded", "jina", "dense"],
                              candidates, queries, blocks,
                              {**base_scores, "e5": e5})}
    print(f"shipped              recall={report['shipped']['recall']:.4f} "
          f"precision={report['shipped']['precision']:.4f} "
          f"f2={report['shipped']['f2']:.4f}", flush=True)

    for depth in (10, 20):
        extended = {q: list(dict.fromkeys(list(candidates[q]) +
                                          corpus_rank[q][:depth]))
                    for q in all_ids}
        rank_by = lambda table: {q: sorted(extended[q],
                                           key=lambda d: (-table[q].get(d, -1e9), d))
                                 for q in all_ids}
        local = dict(views)
        local["jina"] = rank_by(extended_scores["jina"])
        local["dense"] = rank_by(extended_scores["dense"])
        local["corpus"] = {q: [d for d in corpus_rank[q] if d in set(extended[q])]
                           for q in all_ids}
        scores = {"jina": extended_scores["jina"], "dense": extended_scores["dense"],
                  "expansion": base_scores["expansion"], "e5": e5,
                  "corpus": {q: {d: corpus_score[q].get(d, -1.0) for d in extended[q]}
                             for q in all_ids}}
        ceiling = float(np.mean([len(set(extended[q]) & queries[q][1]) /
                                 len(queries[q][1]) for q in all_ids]))
        variants = {
            f"pool{depth}_views4": (["base", "expanded", "jina", "dense"], scores),
            f"pool{depth}_views5": (["base", "expanded", "jina", "dense", "corpus"],
                                    scores),
            f"pool{depth}_views5_nochannel":
                (["base", "expanded", "jina", "dense", "corpus"],
                 {k: v for k, v in scores.items() if k != "corpus"}),
        }
        for label, (names, score_set) in variants.items():
            result = lobo(local, names, extended, queries, blocks, score_set)
            result["ceiling"] = ceiling
            result["mean_pool"] = float(np.mean([len(extended[q]) for q in all_ids]))
            report[label] = result
            print(f"{label:20s} recall={result['recall']:.4f} "
                  f"precision={result['precision']:.4f} f2={result['f2']:.4f} "
                  f"(ceiling {ceiling:.4f}, pool {result['mean_pool']:.1f})",
                  flush=True)

    path = root / "burst_corpus_fusion_validation.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {path}", flush=True)


if __name__ == "__main__":
    main()
