"""Does deeper passage coverage improve the fused ranking?

benchmark_deep_passage_holdouts.py caches one score per passage per document, so
this compares depth-1/2/3/4 and max- versus mean-pooling without any rescoring.
Every variant is judged by leave-one-block-out, the same rule that selected the
shipped ranker.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from tune_burst_pairwise import fixed_metrics
from tune_expanded_fusion_robust import build_views
from tune_expanded_fusion_selection import holdout_scores, ltr_features


def pooled(per_passage, depth, how):
    """Collapse the per-passage scores of one document into a single score."""
    out = {}
    for q, table in per_passage.items():
        row = {}
        for doc, values in table.items():
            head = values[:depth]
            row[doc] = float(max(head) if how == "max" else np.mean(head))
        out[q] = row
    return out


def lobo(views, names, candidates, queries, blocks, scores, c=.3):
    """Pool every fold's held-out prediction before scoring.

    Averaging per-block Recall@5 and then meaning across blocks gives each
    block equal weight regardless of size; with block d holding 300 of 600
    queries against 100 each for a/b/c, that inflates the reported number by
    roughly +0.003. Pooling first gives the correct query-level average.
    """
    rows, groups = ltr_features(views, names, candidates,
                                sum(blocks.values(), []), scores)
    ranked = {}
    for held in blocks:
        train = sum((blocks[n] for n in blocks if n != held), [])
        x = np.vstack([rows[q] for q in train])
        y = np.concatenate([[d in queries[q][1] for d in groups[q]]
                            for q in train]).astype(np.int8)
        scaler = StandardScaler().fit(x)
        model = LogisticRegression(C=c, class_weight="balanced", solver="liblinear",
                                   max_iter=3000, random_state=2026)
        model.fit(scaler.transform(x), y)
        for q in blocks[held]:
            value = model.decision_function(scaler.transform(rows[q]))
            ranked[q] = [groups[q][i] for i in np.argsort(-value)]
    m, _ = fixed_metrics(ranked, {q: queries[q] for q in ranked})
    per_block = {n: fixed_metrics(ranked, {q: queries[q] for q in blocks[n]})[0]
                for n in blocks}
    return {"recall": m["Recall@5"], "precision": m["Precision@5"],
            "f2": m["F2@5"], "ndcg": m["nDCG@10"], "blocks": per_block}


def main():
    root = Path(__file__).resolve().parent
    queries, blocks, all_ids, candidates, views = build_views(root)
    deep = pickle.loads((root / "results/deep_passage/holdout_scores.pkl").read_bytes())
    expansion = holdout_scores(root)["expansion"]
    names = ["base", "expanded", "jina", "dense"]

    report = {"config": deep["config"], "variants": {}}
    baseline = lobo(views, names, candidates, queries, blocks, holdout_scores(root))
    report["variants"]["shipped_depth2"] = baseline
    print(f"shipped_depth2        recall={baseline['recall']:.4f} "
          f"precision={baseline['precision']:.4f} f2={baseline['f2']:.4f}", flush=True)

    for depth in (1, 2, 3, 4):
        for how in ("max", "mean"):
            jina = pooled(deep["jina"], depth, how)
            dense = pooled(deep["dense"], depth, how)
            local = dict(views)
            local["jina"] = {q: sorted(candidates[q], key=lambda d: (-jina[q][d], d))
                             for q in all_ids}
            local["dense"] = {q: sorted(candidates[q], key=lambda d: (-dense[q][d], d))
                              for q in all_ids}
            scores = {"jina": jina, "dense": dense, "expansion": expansion}
            result = lobo(local, names, candidates, queries, blocks, scores)
            report["variants"][f"depth{depth}_{how}"] = result
            print(f"depth{depth}_{how:4s}          recall={result['recall']:.4f} "
                  f"precision={result['precision']:.4f} f2={result['f2']:.4f}",
                  flush=True)

    # Max and mean describe different things: peak evidence and how consistently
    # the document matches.  Offer both to the ranker at the best depth.
    for depth in (2, 3, 4):
        jina_max = pooled(deep["jina"], depth, "max")
        dense_max = pooled(deep["dense"], depth, "max")
        local = dict(views)
        local["jina"] = {q: sorted(candidates[q], key=lambda d: (-jina_max[q][d], d))
                         for q in all_ids}
        local["dense"] = {q: sorted(candidates[q], key=lambda d: (-dense_max[q][d], d))
                          for q in all_ids}
        scores = {"jina": jina_max, "dense": dense_max, "expansion": expansion,
                  "jina_mean": pooled(deep["jina"], depth, "mean"),
                  "dense_mean": pooled(deep["dense"], depth, "mean")}
        result = lobo(local, names, candidates, queries, blocks, scores)
        report["variants"][f"depth{depth}_max_and_mean"] = result
        print(f"depth{depth}_max_and_mean  recall={result['recall']:.4f} "
              f"precision={result['precision']:.4f} f2={result['f2']:.4f}", flush=True)

    path = root / "burst_deep_passage_validation.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {path}", flush=True)


if __name__ == "__main__":
    main()
