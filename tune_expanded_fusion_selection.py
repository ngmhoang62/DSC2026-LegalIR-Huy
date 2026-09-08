"""Compare selection rules for the expanded-fusion weights under leave-one-block-out.

Grid argmax overfits 100-query blocks, so this measures three alternatives on the
same views: argmax, the centroid of the top region, and a per-document logistic
ranker over cross-view rank and score features.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from tune_burst_multistage_posterior import weighted_rrf
from tune_burst_pairwise import fixed_metrics
from tune_expanded_fusion_robust import build_views, evaluate, simplex


KS = (0, 2, 5, 10, 20, 40)


def grid(views, names, queries, blocks, step):
    """Cache every grid point's per-block Recall@5 once; rules reuse the table."""
    points = [(w, k) for w in simplex(len(names), step) for k in KS]
    table = np.zeros((len(points), len(blocks)))
    order = list(blocks)
    for i, (w, k) in enumerate(points):
        for j, name in enumerate(order):
            m, _ = evaluate(views, names, w, k, queries, blocks[name])
            table[i, j] = m["Recall@5"]
    return points, table, order


def argmax_rule(points, table, columns):
    scores = [(table[i, columns].min(), table[i, columns].mean(), i)
              for i in range(len(points))]
    scores.sort(reverse=True)
    return points[scores[0][2]]


def centroid_rule(points, table, columns, quantile=.02):
    """Average the top region so the shipped weights sit on a plateau, not a spike."""
    key = table[:, columns].min(axis=1) + .5 * table[:, columns].mean(axis=1)
    cutoff = np.quantile(key, 1 - quantile)
    keep = [i for i in range(len(points)) if key[i] >= cutoff]
    weights = np.mean([points[i][0] for i in keep], axis=0)
    weights = weights / weights.sum()
    ks = [points[i][1] for i in keep]
    return tuple(float(x) for x in weights), int(np.median(ks))


def ltr_features(views, names, candidates, ids, scores=None):
    """Per-candidate cross-view ranks, plus per-query standardized model scores.

    Scores are z-scored within a query so the ranker sees margins ("how far
    ahead of the field") rather than raw model scales, which differ per query.
    """
    rows, groups = {}, {}
    for q in ids:
        ranks = [{d: i + 1 for i, d in enumerate(views[n][q])} for n in names]
        docs = candidates[q]
        columns = []
        if scores:
            for view in sorted(scores):
                raw = scores[view].get(q, {})
                values = np.asarray([raw.get(d, np.nan) for d in docs], dtype=np.float64)
                present = values[~np.isnan(values)]
                if present.size:
                    mean, std = present.mean(), present.std() or 1.0
                    top = present.max()
                else:
                    mean, std, top = 0.0, 1.0, 0.0
                filled = np.where(np.isnan(values), mean - 2 * std, values)
                columns.append((filled - mean) / std)
                columns.append((filled - top) / std)
        feature = []
        for i, d in enumerate(docs):
            r = [ranks[j].get(d, 60) for j in range(len(names))]
            row = ([1.0 / (10 + x) for x in r] + [x / 60 for x in r] +
                   [float(min(r)), float(np.mean(r))])
            row.extend(float(column[i]) for column in columns)
            feature.append(row)
        rows[q] = np.asarray(feature, dtype=np.float32)
        groups[q] = docs
    return rows, groups


def ltr_rule(views, names, candidates, queries, train_ids, test_ids, c=.3, scores=None):
    rows, groups = ltr_features(views, names, candidates, train_ids + test_ids, scores)
    x = np.vstack([rows[q] for q in train_ids])
    y = np.concatenate([[d in queries[q][1] for d in groups[q]] for q in train_ids])
    scaler = StandardScaler().fit(x)
    model = LogisticRegression(C=c, class_weight="balanced", solver="liblinear",
                               max_iter=3000, random_state=2026)
    model.fit(scaler.transform(x), y.astype(np.int8))
    ranked = {}
    for q in test_ids:
        score = model.decision_function(scaler.transform(rows[q]))
        ranked[q] = [groups[q][i] for i in np.argsort(-score)]
    return ranked


def holdout_scores(root):
    """Raw model scores behind the holdout views, keyed by view name."""
    expanded = pickle.loads((root / "results/expanded_rerank/scores.pkl").read_bytes())
    expansion = pickle.loads(
        (root / "results/dense_expansion/union50_scores.pkl").read_bytes())["scores"]
    return {"jina": expanded["jina"], "dense": expanded["dense"],
            "expansion": expansion}


def main():
    root = Path(__file__).resolve().parent
    queries, blocks, all_ids, candidates, views = build_views(root)
    stacks = {
        "four_view": (["base", "expanded", "jina", "dense"], .05),
        "five_view_vi": (["base", "expanded", "jina", "dense", "vi"], .10),
    }
    report = {}
    for label, (names, step) in stacks.items():
        points, table, order = grid(views, names, queries, blocks, step)
        entry = {"grid_points": len(points), "rules": {}}
        for rule in ("argmax", "centroid"):
            held_out, configs = [], {}
            for held in order:
                columns = [j for j, n in enumerate(order) if n != held]
                config = (argmax_rule if rule == "argmax" else centroid_rule)(
                    points, table, columns)
                m, _ = evaluate(views, names, config[0], config[1], queries,
                                blocks[held])
                held_out.append(m)
                configs[held] = {"weights": list(config[0]), "rrf_k": config[1],
                                 "held_out": m}
            full = (argmax_rule if rule == "argmax" else centroid_rule)(
                points, table, list(range(len(order))))
            entry["rules"][rule] = {
                "lobo_recall": float(np.mean([m["Recall@5"] for m in held_out])),
                "lobo_precision": float(np.mean([m["Precision@5"] for m in held_out])),
                "lobo_f2": float(np.mean([m["F2@5"] for m in held_out])),
                "folds": configs,
                "all_block_fit": {"weights": list(full[0]), "rrf_k": full[1],
                                  "blocks": {n: evaluate(views, names, full[0],
                                                         full[1], queries, blocks[n])[0]
                                             for n in order}},
            }
        for tag, table_scores in (("ltr", None), ("ltr_scored", holdout_scores(root))):
            for c in (.03, .1, .3, 1.0):
                held_out = []
                for held in order:
                    train = sum((blocks[n] for n in order if n != held), [])
                    ranked = ltr_rule(views, names, candidates, queries, train,
                                      blocks[held], c, table_scores)
                    m, _ = fixed_metrics(ranked, {q: queries[q] for q in blocks[held]})
                    held_out.append(m)
                entry["rules"][f"{tag}_C{c}"] = {
                    "lobo_recall": float(np.mean([m["Recall@5"] for m in held_out])),
                    "lobo_precision": float(np.mean([m["Precision@5"]
                                                     for m in held_out])),
                    "lobo_f2": float(np.mean([m["F2@5"] for m in held_out])),
                    "blocks": {n: m for n, m in zip(order, held_out)},
                }
        report[label] = entry
        for rule, data in entry["rules"].items():
            print(f"{label:14s} {rule:9s} LOBO recall={data['lobo_recall']:.4f} "
                  f"precision={data['lobo_precision']:.4f} f2={data['lobo_f2']:.4f}",
                  flush=True)

    path = root / "burst_expanded_fusion_selection.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {path}", flush=True)


if __name__ == "__main__":
    main()
