#!/usr/bin/env python
"""
HUY DEADLINE ROBUSTNESS — CANDIDATE-POOL AUGMENTED D1 TRAINING V1
=================================================================

CPU-only exact D1 LOBO.

Observed failure motivating this experiment:
  E20 oracle == E15 oracle == E10 oracle == 0.984722...
  but inference/training on E15/E10 loses substantial Top-5 Recall.

Hypothesis:
  D1 is sensitive to the exact negative-candidate geometry and per-query
  score normalization.  Train-time exposure to multiple label-preserving
  candidate subsets may reduce this brittleness while keeping production
  inference EXACTLY on the current E20 pool.

Variants:
  CONTROL_E20:
      train E20, test E20
  AUG_E20_E15:
      train rows from E20 + E15 for every training query, test E20
  AUG_E20_E15_E10:
      train rows from E20 + E15 + E10, test E20

All feature contracts remain 48D.
No public labels. No new neural scoring.
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


EXPECTED_R = 0.9569444444444444
EXPECTED_P = 0.20566666666666666
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


def build_world(root: Path, expanded_depth: int, docs):
    from tune_citation_graph import build_citation_table, citation_features
    from tune_corpus_cap32_fusion import build_training_cap
    from tune_doctype_features import build_type_table, type_features
    from tune_expanded_fusion_selection import ltr_features

    queries, blocks, ids, candidates, views, base_scores = build_training_cap(
        root,
        32,
        "results/corpus_index/holdout_extended_scores_cap32.pkl",
        depth=20,
        expanded_depth=expanded_depth,
    )
    gold = {q: set(map(str, queries[q][1])) for q in ids}

    channels = {
        **base_scores,
        "vnlegal_lal": align_scores(
            load_pickle(root, "results/embedding_finetune/vnlegal_lal_cv_scores.pkl"),
            candidates, ids,
        ),
        "crossenc": align_scores(
            load_pickle(root, "results/crossenc_fullpool/cv_scores.pkl"),
            candidates, ids, -11.5,
        ),
        **{
            k: align_scores(load_pickle(root, rel), candidates, ids)
            for k, rel in EXTRA.items()
        },
    }

    type_table = build_type_table(root, docs, ids, candidates)
    type_rows = type_features(candidates, type_table, queries, ids)
    own, cited = build_citation_table(docs, ids, candidates)
    cite_rows = citation_features(candidates, own, cited, ids)

    rows, groups = ltr_features(
        views, D1_VIEWS, candidates, ids, channels
    )
    for q in ids:
        rows[q] = np.concatenate(
            [rows[q], type_rows[q], cite_rows[q]], axis=1
        )
        if rows[q].shape[1] != 48:
            raise RuntimeError(
                f"E{expanded_depth} q={q}: expected 48D, got {rows[q].shape[1]}"
            )

    return {
        "expanded_depth": expanded_depth,
        "queries": queries,
        "blocks": blocks,
        "ids": ids,
        "gold": gold,
        "rows": rows,
        "groups": groups,
        "mean_pool": float(np.mean([len(candidates[q]) for q in ids])),
        "oracle": float(np.mean([
            len(set(candidates[q]) & gold[q]) / len(gold[q])
            for q in ids
        ])),
    }


def fit_eval(world20, train_worlds):
    ids = world20["ids"]
    blocks = world20["blocks"]
    gold = world20["gold"]

    pred = {}
    perq = {}

    for held in sorted(blocks):
        train_ids = sum(
            (blocks[b] for b in blocks if b != held),
            []
        )

        X_parts = []
        y_parts = []

        for w in train_worlds:
            X_parts.extend(w["rows"][q] for q in train_ids)
            y_parts.extend(
                np.asarray(
                    [d in gold[q] for d in w["groups"][q]],
                    dtype=np.int8,
                )
                for q in train_ids
            )

        X = np.vstack(X_parts)
        y = np.concatenate(y_parts)

        scaler = StandardScaler().fit(X)
        model = LogisticRegression(
            C=.15,
            class_weight="balanced",
            solver="liblinear",
            max_iter=3000,
            random_state=2026,
        )
        model.fit(scaler.transform(X), y)

        # Production inference contract is ALWAYS full E20.
        for q in blocks[held]:
            Xq = world20["rows"][q]
            s = model.decision_function(scaler.transform(Xq))
            order = np.argsort(-s, kind="stable")
            pred[q] = [
                world20["groups"][q][i]
                for i in order[:5]
            ]

    for q in ids:
        perq[q] = (
            len(set(pred[q]) & gold[q])
            / len(gold[q])
        )

    return {
        "recall": float(np.mean([perq[q] for q in ids])),
        "precision": float(np.mean([
            len(set(pred[q]) & gold[q]) / 5.0
            for q in ids
        ])),
        "blocks": {
            b: float(np.mean([perq[q] for q in blocks[b]]))
            for b in sorted(blocks)
        },
        "single": float(np.mean([
            perq[q] for q in ids if len(gold[q]) == 1
        ])),
        "multi": float(np.mean([
            perq[q] for q in ids if len(gold[q]) > 1
        ])),
        "pred": pred,
        "perq": perq,
        "train_depths": [w["expanded_depth"] for w in train_worlds],
    }


def compare(base, x, ids):
    diff = np.asarray([
        x["perq"][q] - base["perq"][q]
        for q in ids
    ])
    return {
        "delta_recall": x["recall"] - base["recall"],
        "delta_precision": x["precision"] - base["precision"],
        "delta_single": x["single"] - base["single"],
        "delta_multi": x["multi"] - base["multi"],
        "wins": int(np.sum(diff > 1e-12)),
        "losses": int(np.sum(diff < -1e-12)),
        "ties": int(np.sum(np.abs(diff) <= 1e-12)),
        "block_deltas": {
            b: x["blocks"][b] - base["blocks"][b]
            for b in base["blocks"]
        },
        "exact_top5_matches": int(sum(
            x["pred"][q] == base["pred"][q]
            for q in ids
        )),
    }


def slim(x):
    return {
        k: v for k, v in x.items()
        if k not in ("pred", "perq")
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=Path, required=True)
    args = ap.parse_args()

    root = args.repo_root.resolve()
    sys.path.insert(0, str(root))

    from run_burst_expanded_fusion_submission import DocumentStore

    docs = DocumentStore(sorted(
        (
            root
            / "DSC2026-LegalIR-main/v4_run/public_test_dataset/selected-contexts"
        ).glob("context_*.json")
    ))

    print("[1/4] Building E20/E15/E10 feature worlds...", flush=True)
    w20 = build_world(root, 20, docs)
    w15 = build_world(root, 15, docs)
    w10 = build_world(root, 10, docs)

    if w20["ids"] != w15["ids"] or w20["ids"] != w10["ids"]:
        raise RuntimeError("World qid mismatch")

    print(
        f"  E20 pool={w20['mean_pool']:.2f} oracle={w20['oracle']:.10f}",
        flush=True,
    )
    print(
        f"  E15 pool={w15['mean_pool']:.2f} oracle={w15['oracle']:.10f}",
        flush=True,
    )
    print(
        f"  E10 pool={w10['mean_pool']:.2f} oracle={w10['oracle']:.10f}",
        flush=True,
    )

    print("[2/4] Exact E20 control...", flush=True)
    control = fit_eval(w20, [w20])

    if (
        abs(control["recall"] - EXPECTED_R) > 1e-12
        or abs(control["precision"] - EXPECTED_P) > 1e-12
    ):
        raise RuntimeError(
            f"D1 parity failed: R={control['recall']} "
            f"P={control['precision']}"
        )

    print("[3/4] Pool-augmented training...", flush=True)
    aug15 = fit_eval(w20, [w20, w15])
    aug1510 = fit_eval(w20, [w20, w15, w10])

    ids = w20["ids"]
    c15 = compare(control, aug15, ids)
    c1510 = compare(control, aug1510, ids)

    variants = []
    for name, r, c in (
        ("AUG_E20_E15", aug15, c15),
        ("AUG_E20_E15_E10", aug1510, c1510),
    ):
        strict = (
            c["delta_recall"] > 1e-12
            and all(v >= -1e-12 for v in c["block_deltas"].values())
            and c["wins"] > c["losses"]
        )
        robust_parity = (
            abs(c["delta_recall"]) <= 1e-12
            and all(v >= -1e-12 for v in c["block_deltas"].values())
            and c["losses"] == 0
        )
        variants.append({
            "name": name,
            **slim(r),
            "comparison": c,
            "strict_promote": bool(strict),
            "robust_parity": bool(robust_parity),
        })

    report = {
        "schema": "manual.candidate_pool_augmented_d1_training_v1",
        "motivation": (
            "reduce sensitivity to label-preserving candidate-pool geometry "
            "changes observed in E20->E15/E10"
        ),
        "control": slim(control),
        "worlds": {
            "E20": {
                "mean_pool": w20["mean_pool"],
                "oracle": w20["oracle"],
            },
            "E15": {
                "mean_pool": w15["mean_pool"],
                "oracle": w15["oracle"],
            },
            "E10": {
                "mean_pool": w10["mean_pool"],
                "oracle": w10["oracle"],
            },
        },
        "variants": variants,
        "production_inference_pool": "E20 unchanged",
        "public_labels_used": False,
    }

    out = (
        root
        / "results/manual/huy_candidate_pool_augmented_d1_training_v1"
    )
    out.mkdir(parents=True, exist_ok=True)
    path = out / "REPORT.json"
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("[4/4] RESULT")
    print("=" * 118)
    print(
        f"CONTROL          R={control['recall']:.10f} "
        f"P={control['precision']:.10f}"
    )
    for x in variants:
        c = x["comparison"]
        print(
            f"{x['name']:<18s} "
            f"R={x['recall']:.10f} "
            f"dR={c['delta_recall']:+.10f} "
            f"P={x['precision']:.10f} "
            f"dP={c['delta_precision']:+.10f} "
            f"W/L/T={c['wins']}/{c['losses']}/{c['ties']} "
            f"blocks={c['block_deltas']} "
            f"strict={x['strict_promote']} "
            f"parity={x['robust_parity']}"
        )
    print("Report:", path)
    print("=" * 118)


if __name__ == "__main__":
    main()
