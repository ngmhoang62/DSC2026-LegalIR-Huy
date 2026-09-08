"""Validate a dynamic-threshold cutoff on top of the shipped LTR fusion.

scoring.py (the real competition scorer, confirmed by reading legalir_v2.zip)
only requires 0 < len(pred) <= 5 -- NOT exactly 5.  Precision is
hits/len(pred), so for the ~93% of queries with a single gold document,
returning just that one document when the model is confident turns
precision 0.20 (1 hit / 5 returned) into precision 1.00, at zero recall
cost.  Every validation run so far in this project used a metrics()
that divides by a fixed 5, matching the WRONG "always return 5" rule --
this script uses the real formula instead.

Strategy: always keep rank-1.  Keep rank i (2..5) only if
predict_proba(i) >= alpha * predict_proba(1).  alpha=0 reproduces the
shipped "always 5" behaviour exactly (every candidate passes).  Swept via
pooled leave-one-block-out (same 4 blocks as every other tuning script
here); alpha is a single scalar so pooling across folds before picking
the arg-max is low-risk with N=600.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from run_burst_expanded_fusion_submission import DocumentStore
from tune_citation_graph import build_citation_table, citation_features
from tune_corpus_dense_fusion import build_training
from tune_doctype_features import build_type_table, type_features
from tune_expanded_fusion_selection import ltr_features


def real_metrics(pred, queries):
    """Exact scoring.py formula: valid iff 0 < len(pred) <= 5."""
    recalls, precisions = [], []
    for q, (_, gold) in queries.items():
        p = pred.get(q, [])
        ok = 0 < len(p) <= 5
        hits = len(gold.intersection(p)) if ok else 0
        recalls.append(hits / len(gold) if ok else 0.0)
        precisions.append(hits / len(p) if ok else 0.0)
    r, p_ = float(np.mean(recalls)), float(np.mean(precisions))
    f2 = 0.0 if 4 * p_ + r == 0 else 5 * p_ * r / (4 * p_ + r)
    return {"Recall": r, "Precision": p_, "F2": f2, "mean_len": float(
        np.mean([len(pred.get(q, [])) for q in queries]))}


def lobo_probabilities(names, local, extended, queries, blocks, scores,
                       extra_rows_list, c=.15):
    """Same LOBO loop as production's ltr_fusion, but keep calibrated
    probabilities (not just rank order) so a threshold can be applied."""
    rows, groups = ltr_features(local, names, extended,
                                sum(blocks.values(), []), scores)
    for extra_rows in extra_rows_list:
        for q in rows:
            rows[q] = np.concatenate([rows[q], extra_rows[q]], axis=1)
    ranked, proba = {}, {}
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
            p = model.predict_proba(scaler.transform(rows[q]))[:, 1]
            order = np.argsort(-p)
            ranked[q] = [groups[q][i] for i in order]
            proba[q] = [float(p[i]) for i in order]
    return ranked, proba


def apply_alpha(ranked, proba, alpha, keep_min=1):
    """keep_min: ranks 1..keep_min are always kept (protects recall for the
    queries where rank 1 is wrong but the gold doc is at rank 2); ranks beyond
    keep_min are kept only if score >= alpha * rank-1's score."""
    out = {}
    for q, docs in ranked.items():
        p = proba[q]
        top5, p5 = docs[:5], p[:5]
        kept = list(top5[:keep_min])
        for d, s in zip(top5[keep_min:], p5[keep_min:]):
            if s >= alpha * p5[0]:
                kept.append(d)
        out[q] = kept
    return out


def apply_rank_alpha(ranked, proba, alphas):
    """alphas: per-position thresholds for ranks 2..len(alphas)+1, each
    relative to rank-1's probability -- a looser alpha for rank 2 (protects
    recall for the "gold is at rank 2" case) and a stricter one for the
    ranks-3-5 tail (which rarely holds gold but always costs precision)."""
    out = {}
    for q, docs in ranked.items():
        p = proba[q]
        top5, p5 = docs[:5], p[:5]
        kept = [top5[0]]
        for d, s, a in zip(top5[1:], p5[1:], alphas):
            if s >= a * p5[0]:
                kept.append(d)
        out[q] = kept
    return out


def main():
    root = Path(__file__).resolve().parent
    queries, blocks, all_ids, extended, local, scores = build_training(root, depth=20)
    names = ["base", "expanded", "jina", "dense", "corpus"]

    docs = DocumentStore(sorted(
        (root / "DSC2026-LegalIR-main/v4_run/public_test_dataset/selected-contexts")
        .glob("context_*.json")))
    type_tab = build_type_table(root, docs, all_ids, extended)
    type_rows = type_features(extended, type_tab, queries, all_ids)
    own, cited = build_citation_table(docs, all_ids, extended)
    cite_rows = citation_features(extended, own, cited, all_ids)

    ranked, proba = lobo_probabilities(names, local, extended, queries, blocks,
                                       scores, [type_rows, cite_rows])
    held_queries = {q: queries[q] for q in ranked}

    print("=== Baseline (always 5, WRONG metrics -- /5 fixed divisor) ===",
          flush=True)
    always5 = {q: ranked[q][:5] for q in ranked}
    print(f"real_metrics on always-5: {real_metrics(always5, held_queries)}",
          flush=True)

    report = {}
    for keep_min in (1, 2):
        print(f"\n=== Dynamic threshold sweep (keep_min={keep_min}) ===", flush=True)
        for alpha in (0.0, .05, .1, .15, .2, .25, .3, .35, .4, .45, .5, .55, .6,
                     .65, .7, .75, .8, .85, .9, .95):
            pred = apply_alpha(ranked, proba, alpha, keep_min)
            m = real_metrics(pred, held_queries)
            per_block = {n: real_metrics({q: pred[q] for q in blocks[n]},
                                         {q: queries[q] for q in blocks[n]})
                        for n in blocks}
            report[f"keepmin{keep_min}_alpha_{alpha}"] = {"pooled": m,
                                                           "blocks": per_block}
            print(f"alpha={alpha:<5} R={m['Recall']:.4f} P={m['Precision']:.4f} "
                  f"F2={m['F2']:.4f} mean_len={m['mean_len']:.2f}", flush=True)

    best = max(report, key=lambda k: report[k]["pooled"]["F2"])
    print(f"\nBest by pooled F2: {best} -> {report[best]['pooled']}", flush=True)
    print("Per-block breakdown of best:", flush=True)
    for n, m in report[best]["blocks"].items():
        print(f"  block {n}: {m}", flush=True)

    print("\n=== Configs with Recall >= 0.90 (keep_min=2), sorted by F2 ===",
          flush=True)
    high_recall = {k: v for k, v in report.items()
                   if k.startswith("keepmin2_") and v["pooled"]["Recall"] >= 0.90}
    for k, v in sorted(high_recall.items(), key=lambda kv: -kv[1]["pooled"]["F2"]):
        print(f"  {k}: {v['pooled']}", flush=True)

    print("\n=== Per-rank alpha (loose for rank2, strict for tail 3-5), "
          "recall-preserving ===", flush=True)
    rank_report = {}
    for a2 in (0.0, .05, .1, .15, .2):
        for a_tail in (.5, .6, .7, .8, .85, .9, .95):
            alphas = (a2, a_tail, a_tail, a_tail)
            pred = apply_rank_alpha(ranked, proba, alphas)
            m = real_metrics(pred, held_queries)
            rank_report[f"a2_{a2}_atail_{a_tail}"] = m
    for k, m in sorted(rank_report.items(), key=lambda kv: -kv[1]["Recall"]):
        if m["Recall"] >= 0.935:
            print(f"  {k}: R={m['Recall']:.4f} P={m['Precision']:.4f} "
                  f"F2={m['F2']:.4f} mean_len={m['mean_len']:.2f}", flush=True)
    report["rank_alpha"] = rank_report

    path = root / "burst_dynamic_threshold_validation.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {path}", flush=True)


if __name__ == "__main__":
    main()
