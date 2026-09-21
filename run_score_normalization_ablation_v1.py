#!/usr/bin/env python
"""
HUY PIPELINE AUDIT H — SCORE NORMALIZATION ABLATION V1
======================================================

CPU-only exact D1 4-block LOBO on the CURRENT candidate pool.

Current D1 score geometry uses, per score channel:
  1) within-query z-score
  2) (score - query_top_score) / std

This audit asks whether that 20D score geometry is overparameterized or fragile.

Preregistered variants:
  CURRENT_Z_PLUS_TOP  : exact D1 control, 20 score dims
  Z_ONLY              : one z-score/channel, 10 score dims
  TOPDIST_ONLY        : one top-distance/std feature/channel, 10 score dims
  ROBUST_MAD_PLUS_TOP : median/MAD robust normalization, 20 score dims
  ROBUST_IQR_PLUS_TOP : median/IQR robust normalization, 20 score dims

All keep:
  5 D1 rank views + doctype12 + citation4 + LR C=.15 balanced.

No public labels. No threshold search.
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

    type_table = build_type_table(root, docs, ids, candidates)
    type_rows = type_features(candidates, type_table, queries, ids)
    own, cited = build_citation_table(docs, ids, candidates)
    cite_rows = citation_features(candidates, own, cited, ids)

    return queries, blocks, ids, candidates, views, channels, gold, type_rows, cite_rows


def safe_mad(x):
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med))) * 1.4826
    if mad <= 1e-12:
        mad = float(np.std(x))
    return med, (mad if mad > 1e-12 else 1.0)


def safe_iqr(x):
    med = float(np.median(x))
    q25, q75 = np.percentile(x, [25, 75])
    scale = float((q75 - q25) / 1.349)
    if scale <= 1e-12:
        scale = float(np.std(x))
    return med, (scale if scale > 1e-12 else 1.0)


def build_features(views, candidates, ids, channels, type_rows, cite_rows, mode):
    rows = {}
    groups = {}

    for q in ids:
        docs = candidates[q]
        rank_maps = [
            {d: i + 1 for i, d in enumerate(views[name][q])}
            for name in D1_VIEWS
        ]

        score_columns = []
        for name in sorted(channels):
            raw = channels[name].get(q, {})
            values = np.asarray(
                [raw.get(d, np.nan) for d in docs],
                dtype=np.float64,
            )
            present = values[~np.isnan(values)]

            if present.size:
                if mode in ("CURRENT_Z_PLUS_TOP", "Z_ONLY", "TOPDIST_ONLY"):
                    center = float(present.mean())
                    scale = float(present.std()) or 1.0
                elif mode == "ROBUST_MAD_PLUS_TOP":
                    center, scale = safe_mad(present)
                elif mode == "ROBUST_IQR_PLUS_TOP":
                    center, scale = safe_iqr(present)
                else:
                    raise ValueError(mode)
                top = float(present.max())
            else:
                center, scale, top = 0.0, 1.0, 0.0

            filled = np.where(np.isnan(values), center - 2 * scale, values)
            z = (filled - center) / scale
            topdist = (filled - top) / scale

            if mode == "Z_ONLY":
                score_columns.append(z)
            elif mode == "TOPDIST_ONLY":
                score_columns.append(topdist)
            else:
                score_columns.extend([z, topdist])

        feats = []
        for i, d in enumerate(docs):
            r = [rm.get(d, 60) for rm in rank_maps]
            row = (
                [1.0 / (10 + x) for x in r]
                + [x / 60.0 for x in r]
                + [float(min(r)), float(np.mean(r))]
            )
            row.extend(float(col[i]) for col in score_columns)
            row.extend(type_rows[q][i].tolist())
            row.extend(cite_rows[q][i].tolist())
            feats.append(row)

        rows[q] = np.asarray(feats, dtype=np.float32)
        groups[q] = docs

    return rows, groups


def run(mode, world):
    queries, blocks, ids, candidates, views, channels, gold, type_rows, cite_rows = world
    rows, groups = build_features(
        views, candidates, ids, channels, type_rows, cite_rows, mode
    )
    dim = rows[ids[0]].shape[1]

    pred = {}
    per_q = {}

    for held in sorted(blocks):
        train = sum((blocks[b] for b in blocks if b != held), [])
        X = np.vstack([rows[q] for q in train])
        y = np.concatenate([
            [d in gold[q] for d in groups[q]]
            for q in train
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

        for q in blocks[held]:
            s = model.decision_function(scaler.transform(rows[q]))
            order = np.argsort(-s, kind="stable")
            pred[q] = [groups[q][i] for i in order[:5]]

    for q in ids:
        per_q[q] = len(set(pred[q]) & gold[q]) / len(gold[q])

    return {
        "mode": mode,
        "feature_dim": int(dim),
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
        "top5_exact_matches": int(sum(
            cand["pred"][q] == base["pred"][q] for q in ids
        )),
        "block_deltas": {
            b: cand["blocks"][b] - base["blocks"][b]
            for b in base["blocks"]
        },
    }


def strip(x):
    return {k: v for k, v in x.items() if k not in ("pred", "per_q")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=Path, required=True)
    args = ap.parse_args()
    root = args.repo_root.resolve()
    sys.path.insert(0, str(root))

    print("[1/3] Loading exact D1 world...", flush=True)
    world = prepare(root)
    ids = world[2]

    modes = [
        "CURRENT_Z_PLUS_TOP",
        "Z_ONLY",
        "TOPDIST_ONLY",
        "ROBUST_MAD_PLUS_TOP",
        "ROBUST_IQR_PLUS_TOP",
    ]

    print("[2/3] Running normalization variants...", flush=True)
    result = {}
    for mode in modes:
        r = run(mode, world)
        result[mode] = r
        print(
            f"  {mode:<22s} dim={r['feature_dim']:2d} "
            f"R={r['recall']:.10f} P={r['precision']:.10f}",
            flush=True,
        )

    base = result["CURRENT_Z_PLUS_TOP"]
    if abs(base["recall"] - EXPECTED_R) > 1e-12 or base["feature_dim"] != 48:
        raise RuntimeError(
            f"Current normalization parity failed: dim={base['feature_dim']} "
            f"R={base['recall']}"
        )

    rows = []
    for mode in modes[1:]:
        c = compare(base, result[mode], ids)
        strict = (
            c["delta_recall"] >= -1e-12
            and all(x >= -1e-12 for x in c["block_deltas"].values())
            and (
                c["wins"] > c["losses"]
                or (
                    result[mode]["feature_dim"] < base["feature_dim"]
                    and c["top5_exact_matches"] == len(ids)
                )
            )
        )
        rows.append({
            **strip(result[mode]),
            "comparison": c,
            "strict_promote": bool(strict),
        })

    report = {
        "schema": "manual.score_normalization_ablation_v1",
        "baseline": strip(base),
        "variants": rows,
        "promotion_rule": (
            "Recall >= current, no block regression, and wins>losses; "
            "or exact Top5 parity at lower dimension"
        ),
        "public_labels_used": False,
    }

    out = root / "results/manual/huy_score_normalization_ablation_v1"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "REPORT.json"
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("[3/3] RESULT")
    print("=" * 112)
    print(
        f"CONTROL {base['feature_dim']}D "
        f"R={base['recall']:.10f} P={base['precision']:.10f}"
    )
    for x in rows:
        c = x["comparison"]
        print(
            f"{x['mode']:<22s} dim={x['feature_dim']:2d} "
            f"dR={c['delta_recall']:+.10f} "
            f"dP={c['delta_precision']:+.10f} "
            f"W/L/T={c['wins']}/{c['losses']}/{c['ties']} "
            f"blocks={c['block_deltas']} "
            f"strict={x['strict_promote']}"
        )
    print("Report:", path)
    print("=" * 112)


if __name__ == "__main__":
    main()
