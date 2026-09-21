#!/usr/bin/env python
"""
HUY PIPELINE AUDIT I — CLASS WEIGHTING / POINTWISE OBJECTIVE V1
===============================================================

CPU-only exact D1 4-block LOBO on the current 48D features.

Separates two questions:

A) Class / query weighting
   CONTROL_BALANCED
   NO_CLASS_WEIGHT
   QUERY_EQUAL_BALANCED
   QUERY_EQUAL_UNWEIGHTED

B) Linear pointwise loss/objective
   LOGISTIC_CONTROL
   LINEAR_SVM
   RIDGE_CLASSIFIER

No public labels. All variants use identical candidate pools/features/folds.
No hyperparameter grid: fixed conservative parameters are preregistered here.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


EXPECTED_R = 0.9569444444444444
D1_VIEWS = ["base", "expanded", "jina", "dense", "corpus"]
EXTRA = {
    "aiteamvn_ft": "results/from_drive/aiteamvn_ft_cv.pkl",
    "jina_ft": "results/from_drive/jina_ft_cv.pkl",
    "title_embed": "results/burst_fresh_block/title_embed_scores.pkl",
}


def load_pickle(root, rel):
    return pickle.loads((root / rel).read_bytes())


def align(raw, candidates, ids, floor=None):
    if isinstance(raw, dict) and isinstance(raw.get("scores"), dict):
        raw = raw["scores"]
    if floor is None:
        vals = [v for q in raw.values() for v in q.values()]
        floor = min(vals) if vals else -1e9
    return {
        q: {d: float(raw.get(q, {}).get(d, floor)) for d in candidates[q]}
        for q in ids
    }


def prepare(root):
    from run_burst_expanded_fusion_submission import DocumentStore
    from tune_citation_graph import build_citation_table, citation_features
    from tune_corpus_cap32_fusion import build_training_cap
    from tune_doctype_features import build_type_table, type_features
    from tune_expanded_fusion_selection import ltr_features

    docs = DocumentStore(sorted(
        (
            root
            / "DSC2026-LegalIR-main/v4_run/public_test_dataset/selected-contexts"
        ).glob("context_*.json")
    ))
    queries, blocks, ids, candidates, views, base_scores = build_training_cap(
        root, 32,
        "results/corpus_index/holdout_extended_scores_cap32.pkl",
        depth=20,
    )
    gold = {q: set(map(str, queries[q][1])) for q in ids}

    channels = {
        **base_scores,
        "vnlegal_lal": align(
            load_pickle(root, "results/embedding_finetune/vnlegal_lal_cv_scores.pkl"),
            candidates, ids
        ),
        "crossenc": align(
            load_pickle(root, "results/crossenc_fullpool/cv_scores.pkl"),
            candidates, ids, -11.5
        ),
        **{
            k: align(load_pickle(root, rel), candidates, ids)
            for k, rel in EXTRA.items()
        },
    }

    rows, groups = ltr_features(
        views, D1_VIEWS, candidates, ids, channels
    )

    type_table = build_type_table(root, docs, ids, candidates)
    type_rows = type_features(candidates, type_table, queries, ids)
    own, cited = build_citation_table(docs, ids, candidates)
    cite_rows = citation_features(candidates, own, cited, ids)

    for q in ids:
        rows[q] = np.concatenate(
            [rows[q], type_rows[q], cite_rows[q]],
            axis=1,
        )

    if rows[ids[0]].shape[1] != 48:
        raise RuntimeError(f"Expected 48D, got {rows[ids[0]].shape[1]}")

    return queries, blocks, ids, groups, rows, gold


def make_model(kind):
    if kind in {
        "CONTROL_BALANCED",
        "NO_CLASS_WEIGHT",
        "QUERY_EQUAL_BALANCED",
        "QUERY_EQUAL_UNWEIGHTED",
        "LOGISTIC_CONTROL",
    }:
        cw = "balanced" if kind in {
            "CONTROL_BALANCED",
            "QUERY_EQUAL_BALANCED",
            "LOGISTIC_CONTROL",
        } else None
        return LogisticRegression(
            C=.15,
            class_weight=cw,
            solver="liblinear",
            max_iter=3000,
            random_state=2026,
        )
    if kind == "LINEAR_SVM":
        return LinearSVC(
            C=.15,
            class_weight="balanced",
            max_iter=10000,
            random_state=2026,
        )
    if kind == "RIDGE_CLASSIFIER":
        return RidgeClassifier(
            alpha=1.0,
            class_weight="balanced",
        )
    raise ValueError(kind)


def run(kind, world):
    queries, blocks, ids, groups, rows, gold = world
    pred = {}
    per_q = {}

    for held in sorted(blocks):
        train = sum((blocks[b] for b in blocks if b != held), [])

        X_list = []
        y_list = []
        w_list = []

        query_equal = kind in {
            "QUERY_EQUAL_BALANCED",
            "QUERY_EQUAL_UNWEIGHTED",
        }

        for q in train:
            X_list.append(rows[q])
            yy = np.asarray([d in gold[q] for d in groups[q]], dtype=np.int8)
            y_list.append(yy)
            if query_equal:
                w_list.append(
                    np.full(len(yy), 1.0 / len(yy), dtype=np.float64)
                )

        X = np.vstack(X_list)
        y = np.concatenate(y_list)
        sample_weight = np.concatenate(w_list) if query_equal else None

        scaler = StandardScaler().fit(X)
        Xt = scaler.transform(X)

        model = make_model(kind)
        if sample_weight is None:
            model.fit(Xt, y)
        else:
            model.fit(Xt, y, sample_weight=sample_weight)

        for q in blocks[held]:
            score = model.decision_function(scaler.transform(rows[q]))
            order = np.argsort(-score, kind="stable")
            pred[q] = [groups[q][i] for i in order[:5]]

    for q in ids:
        per_q[q] = len(set(pred[q]) & gold[q]) / len(gold[q])

    return {
        "kind": kind,
        "recall": float(np.mean([per_q[q] for q in ids])),
        "precision": float(np.mean([
            len(set(pred[q]) & gold[q]) / 5.0 for q in ids
        ])),
        "single": float(np.mean([
            per_q[q] for q in ids if len(gold[q]) == 1
        ])),
        "multi": float(np.mean([
            per_q[q] for q in ids if len(gold[q]) > 1
        ])),
        "blocks": {
            b: float(np.mean([per_q[q] for q in blocks[b]]))
            for b in sorted(blocks)
        },
        "pred": pred,
        "per_q": per_q,
    }


def compare(base, cand, ids):
    d = np.asarray([cand["per_q"][q] - base["per_q"][q] for q in ids])
    return {
        "delta_recall": cand["recall"] - base["recall"],
        "delta_precision": cand["precision"] - base["precision"],
        "delta_single": cand["single"] - base["single"],
        "delta_multi": cand["multi"] - base["multi"],
        "wins": int(np.sum(d > 1e-12)),
        "losses": int(np.sum(d < -1e-12)),
        "ties": int(np.sum(np.abs(d) <= 1e-12)),
        "block_deltas": {
            b: cand["blocks"][b] - base["blocks"][b]
            for b in base["blocks"]
        },
        "top5_exact_matches": int(sum(
            cand["pred"][q] == base["pred"][q] for q in ids
        )),
    }


def strip(r):
    return {k: v for k, v in r.items() if k not in ("pred", "per_q")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=Path, required=True)
    args = ap.parse_args()
    root = args.repo_root.resolve()
    sys.path.insert(0, str(root))

    print("[1/3] Loading exact D1 48D feature world...", flush=True)
    world = prepare(root)
    ids = world[2]

    variants = [
        "CONTROL_BALANCED",
        "NO_CLASS_WEIGHT",
        "QUERY_EQUAL_BALANCED",
        "QUERY_EQUAL_UNWEIGHTED",
        "LINEAR_SVM",
        "RIDGE_CLASSIFIER",
    ]

    print("[2/3] Running weighting/objective variants...", flush=True)
    results = {}
    for v in variants:
        r = run(v, world)
        results[v] = r
        print(
            f"  {v:<24s} R={r['recall']:.10f} P={r['precision']:.10f}",
            flush=True,
        )

    base = results["CONTROL_BALANCED"]
    if abs(base["recall"] - EXPECTED_R) > 1e-12:
        raise RuntimeError(f"D1 parity failed: {base['recall']}")

    rows = []
    for v in variants[1:]:
        c = compare(base, results[v], ids)
        strict = (
            c["delta_recall"] > 1e-12
            and all(x >= -1e-12 for x in c["block_deltas"].values())
            and c["wins"] > c["losses"]
        )
        rows.append({
            **strip(results[v]),
            "comparison": c,
            "strict_promote": bool(strict),
        })

    report = {
        "schema": "manual.class_weight_pointwise_objective_ablation_v1",
        "control": strip(base),
        "variants": rows,
        "promotion_rule": (
            "strictly higher pooled Recall, no block regression, wins>losses"
        ),
        "public_labels_used": False,
        "notes": {
            "query_equal": (
                "each query contributes equal total sample weight independent "
                "of candidate-pool size"
            ),
            "linear_svm": "same 48D features; hinge-loss linear ranking score",
            "ridge": "same 48D features; squared-loss linear classifier score",
        },
    }

    out = root / "results/manual/huy_class_weight_objective_ablation_v1"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "REPORT.json"
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("[3/3] RESULT")
    print("=" * 112)
    print(
        f"CONTROL R={base['recall']:.10f} P={base['precision']:.10f}"
    )
    for x in rows:
        c = x["comparison"]
        print(
            f"{x['kind']:<24s} dR={c['delta_recall']:+.10f} "
            f"dP={c['delta_precision']:+.10f} "
            f"W/L/T={c['wins']}/{c['losses']}/{c['ties']} "
            f"blocks={c['block_deltas']} "
            f"strict={x['strict_promote']}"
        )
    print("Report:", path)
    print("=" * 112)


if __name__ == "__main__":
    main()
