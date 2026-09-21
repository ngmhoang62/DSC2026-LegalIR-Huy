#!/usr/bin/env python
"""
HUY PIPELINE AUDIT F — EXPANSION DEPTH DOWNSTREAM LTR V1
========================================================

CPU-only exact D1 LOBO.

Candidate-depth oracle audit found:
  current exp20+corpus20: oracle 0.9847222222, mean pool 39.22
  exp15+corpus20:       oracle 0.9847222222, mean pool 36.7
  exp10+corpus20:       oracle 0.9847222222, mean pool 34.9

This script tests whether removing expanded-tail distractors improves actual
Top-5 ranking, while preserving the same 48D D1 feature contract.

Variants are preregistered before CAL Top-5 evaluation:
  E20 = current D1 control
  E15 = mild simplification
  E10 = minimum-depth simplification with equal candidate oracle

No public labels. No new neural scoring: all candidate sets are strict subsets
of the existing cap32 pool, so existing score caches remain valid.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

EXPECTED_D1 = 0.9569444444444444
D1_VIEWS = ["base", "expanded", "jina", "dense", "corpus"]
EXTRA = {
    "aiteamvn_ft": "results/from_drive/aiteamvn_ft_cv.pkl",
    "jina_ft": "results/from_drive/jina_ft_cv.pkl",
    "title_embed": "results/burst_fresh_block/title_embed_scores.pkl",
}


def load_pickle(root: Path, rel: str):
    return pickle.loads((root / rel).read_bytes())


def align_scores(raw, candidates, ids, floor=None):
    if isinstance(raw, dict) and isinstance(raw.get("scores"), dict):
        raw = raw["scores"]
    if floor is None:
        vals = [v for q in raw.values() for v in q.values()]
        floor = min(vals) if vals else -1e9
    return {
        q: {d: float(raw.get(q, {}).get(d, floor)) for d in candidates[q]}
        for q in ids
    }


def build_world(root: Path, expanded_depth: int):
    from run_burst_expanded_fusion_submission import DocumentStore
    from tune_citation_graph import build_citation_table, citation_features
    from tune_corpus_cap32_fusion import build_training_cap
    from tune_doctype_features import build_type_table, type_features

    docs = DocumentStore(sorted(
        (
            root
            / "DSC2026-LegalIR-main/v4_run/public_test_dataset/selected-contexts"
        ).glob("context_*.json")
    ))

    queries, blocks, ids, candidates, views, base_scores = build_training_cap(
        root,
        32,
        "results/corpus_index/holdout_extended_scores_cap32.pkl",
        depth=20,
        expanded_depth=expanded_depth,
    )
    gold = {q: set(map(str, queries[q][1])) for q in ids}

    vn = align_scores(
        load_pickle(root, "results/embedding_finetune/vnlegal_lal_cv_scores.pkl"),
        candidates, ids
    )
    ce = align_scores(
        load_pickle(root, "results/crossenc_fullpool/cv_scores.pkl"),
        candidates, ids, -11.5
    )
    extra = {
        k: align_scores(load_pickle(root, rel), candidates, ids)
        for k, rel in EXTRA.items()
    }

    channels = {
        **base_scores,
        "vnlegal_lal": vn,
        "crossenc": ce,
        **extra,
    }

    type_table = build_type_table(root, docs, ids, candidates)
    type_rows = type_features(candidates, type_table, queries, ids)
    own, cited = build_citation_table(docs, ids, candidates)
    cite_rows = citation_features(candidates, own, cited, ids)

    return {
        "queries": queries,
        "blocks": blocks,
        "ids": ids,
        "candidates": candidates,
        "views": views,
        "channels": channels,
        "gold": gold,
        "type_rows": type_rows,
        "cite_rows": cite_rows,
    }


def lobo(world):
    from tune_expanded_fusion_selection import ltr_features

    q = world
    rows, groups = ltr_features(
        q["views"], D1_VIEWS, q["candidates"], q["ids"], q["channels"]
    )
    for qid in q["ids"]:
        rows[qid] = np.concatenate(
            [rows[qid], q["type_rows"][qid], q["cite_rows"][qid]],
            axis=1,
        )

    dim = rows[q["ids"][0]].shape[1]
    if dim != 48:
        raise RuntimeError(f"Expected 48D, got {dim}")

    pred = {}
    per_q = {}

    for held in sorted(q["blocks"]):
        train = sum((q["blocks"][b] for b in q["blocks"] if b != held), [])
        test = q["blocks"][held]

        X = np.vstack([rows[x] for x in train])
        y = np.concatenate([
            [d in q["gold"][x] for d in groups[x]]
            for x in train
        ]).astype(np.int8)

        scaler = StandardScaler().fit(X)
        model = LogisticRegression(
            C=.15,
            class_weight="balanced",
            solver="liblinear",
            max_iter=3000,
            random_state=2026,
        )
        model.fit(scaler.transform(X), y)

        for x in test:
            s = model.decision_function(scaler.transform(rows[x]))
            order = np.argsort(-s, kind="stable")
            pred[x] = [groups[x][i] for i in order[:5]]

    for x in q["ids"]:
        per_q[x] = len(set(pred[x]) & q["gold"][x]) / len(q["gold"][x])

    return {
        "recall": float(np.mean([per_q[x] for x in q["ids"]])),
        "precision": float(np.mean([
            len(set(pred[x]) & q["gold"][x]) / 5.0 for x in q["ids"]
        ])),
        "blocks": {
            b: float(np.mean([per_q[x] for x in q["blocks"][b]]))
            for b in sorted(q["blocks"])
        },
        "single": float(np.mean([
            per_q[x] for x in q["ids"] if len(q["gold"][x]) == 1
        ])),
        "multi": float(np.mean([
            per_q[x] for x in q["ids"] if len(q["gold"][x]) > 1
        ])),
        "pred": pred,
        "per_q": per_q,
        "mean_pool": float(np.mean([len(q["candidates"][x]) for x in q["ids"]])),
        "oracle": float(np.mean([
            len(set(q["candidates"][x]) & q["gold"][x]) / len(q["gold"][x])
            for x in q["ids"]
        ])),
    }


def compare(base, cand, ids):
    diff = np.asarray([cand["per_q"][q] - base["per_q"][q] for q in ids])
    return {
        "delta_recall": cand["recall"] - base["recall"],
        "delta_precision": cand["precision"] - base["precision"],
        "wins": int(np.sum(diff > 1e-12)),
        "losses": int(np.sum(diff < -1e-12)),
        "ties": int(np.sum(np.abs(diff) <= 1e-12)),
        "top5_exact_matches": int(sum(
            base["pred"][q] == cand["pred"][q] for q in ids
        )),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=Path, required=True)
    args = ap.parse_args()
    root = args.repo_root.resolve()
    sys.path.insert(0, str(root))

    print("[1/4] Building current E20 control...", flush=True)
    w20 = build_world(root, 20)
    r20 = lobo(w20)
    print(
        f"  E20 pool={r20['mean_pool']:.2f} oracle={r20['oracle']:.10f} "
        f"R={r20['recall']:.10f}",
        flush=True,
    )
    if abs(r20["recall"] - EXPECTED_D1) > 1e-12:
        raise RuntimeError(f"D1 parity failed: {r20['recall']}")

    print("[2/4] Testing E15 simplification...", flush=True)
    w15 = build_world(root, 15)
    r15 = lobo(w15)

    print("[3/4] Testing E10 simplification...", flush=True)
    w10 = build_world(root, 10)
    r10 = lobo(w10)

    c15 = compare(r20, r15, w20["ids"])
    c10 = compare(r20, r10, w20["ids"])

    def block_delta(r):
        return {b: r["blocks"][b] - r20["blocks"][b] for b in r20["blocks"]}

    rows = {}
    for depth, r, c in ((15, r15, c15), (10, r10, c10)):
        bd = block_delta(r)
        strict = (
            abs(r["oracle"] - r20["oracle"]) < 1e-12
            and c["delta_recall"] >= -1e-12
            and all(v >= -1e-12 for v in bd.values())
            and c["wins"] > c["losses"]
        )
        rows[str(depth)] = {
            "expanded_depth": depth,
            "mean_pool": r["mean_pool"],
            "oracle": r["oracle"],
            "recall": r["recall"],
            "precision": r["precision"],
            "single": r["single"],
            "multi": r["multi"],
            "blocks": r["blocks"],
            "comparison": c,
            "block_deltas": bd,
            "strict_promote": bool(strict),
        }

    report = {
        "schema": "manual.expansion_depth_downstream_ltr_v1",
        "baseline_e20": {
            "mean_pool": r20["mean_pool"],
            "oracle": r20["oracle"],
            "recall": r20["recall"],
            "precision": r20["precision"],
            "blocks": r20["blocks"],
        },
        "variants": rows,
        "promotion_rule": (
            "same candidate oracle as E20; pooled Recall >= E20; no block "
            "regression; wins > losses"
        ),
        "public_labels_used": False,
    }

    out = root / "results/manual/huy_expansion_depth_downstream_ltr_v1"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "REPORT.json"
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("[4/4] RESULT")
    print("=" * 112)
    print(
        f"E20  pool={r20['mean_pool']:.2f} oracle={r20['oracle']:.10f} "
        f"R={r20['recall']:.10f} P={r20['precision']:.10f}"
    )
    for depth in (15, 10):
        x = rows[str(depth)]
        c = x["comparison"]
        print(
            f"E{depth:<2d}  pool={x['mean_pool']:.2f} oracle={x['oracle']:.10f} "
            f"R={x['recall']:.10f} dR={c['delta_recall']:+.10f} "
            f"P={x['precision']:.10f} "
            f"W/L/T={c['wins']}/{c['losses']}/{c['ties']} "
            f"blocks={x['block_deltas']} "
            f"strict={x['strict_promote']}"
        )
    print("Report:", path)
    print("=" * 112)


if __name__ == "__main__":
    main()
