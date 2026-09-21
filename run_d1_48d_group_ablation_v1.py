#!/usr/bin/env python
"""
HUY PIPELINE MICRO-AUDIT B — D1 48D GROUP ABLATION V1
======================================================

CPU-only exact 4-block LOBO audit of the current D1 champion.

Tests whether D1 can GENERALIZE better by removing redundant feature groups.
No public labels. No threshold tuning against public.

Baseline D1 = 5 rank views + 10 score channels + doctype + citation:
  rank geometry: 12D
  score geometry: 20D
  doctype: 12D
  citation: 4D
  total: 48D

Ablations:
  - remove each one of 5 rank views (features are recomputed, not sliced)
  - remove each one of 10 score channels
  - remove doctype
  - remove citation
  - remove doctype+citation
  - remove provenance-risk bundle: crossenc + aiteamvn_ft + jina_ft
  - diagnostic regularization sweep for exact baseline C

Selection is NOT "best CAL score wins". Report exposes:
  pooled delta, every block delta, W/L/T, single/multi-gold delta,
  feature count, paired bootstrap CI.
Strict candidates require:
  pooled >= baseline,
  no block regression,
  wins > losses OR exact prediction parity with fewer features.

Run:
  python ../run_d1_48d_group_ablation_v1.py \
    --repo-root /d/Study/DSC2026/sota
"""

from __future__ import annotations

import argparse
import hashlib
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
SEED = 2026


def sha256(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_pkl(root, rel):
    p = root / rel
    obj = pickle.loads(p.read_bytes())
    if isinstance(obj, dict) and isinstance(obj.get("scores"), dict):
        return obj["scores"]
    return obj


def load_inputs(root: Path):
    sys.path.insert(0, str(root))
    from run_burst_expanded_fusion_submission import DocumentStore
    from tune_citation_graph import build_citation_table, citation_features
    from tune_corpus_cap32_fusion import build_training_cap
    from tune_doctype_features import build_type_table, type_features

    docs = DocumentStore(
        sorted(
            (
                root
                / "DSC2026-LegalIR-main/v4_run/public_test_dataset/selected-contexts"
            ).glob("context_*.json")
        )
    )

    queries, blocks, ids, extended, views, base_scores = build_training_cap(
        root,
        32,
        "results/corpus_index/holdout_extended_scores_cap32.pkl",
        depth=20,
    )
    gold = {q: set(map(str, queries[q][1])) for q in ids}

    def aligned(rel, floor=None):
        obj = load_pkl(root, rel)
        if floor is None:
            floor = min(v for q in obj for v in obj[q].values())
        return {
            q: {d: obj.get(q, {}).get(d, floor) for d in extended[q]}
            for q in ids
        }

    vnlegal = load_pkl(
        root, "results/embedding_finetune/vnlegal_lal_cv_scores.pkl"
    )
    crossenc = aligned("results/crossenc_fullpool/cv_scores.pkl", -11.5)
    extras = {k: aligned(v) for k, v in EXTRA.items()}

    scores = {
        **base_scores,
        "vnlegal_lal": vnlegal,
        "crossenc": crossenc,
        **extras,
    }

    type_table = build_type_table(root, docs, ids, extended)
    type_rows = type_features(extended, type_table, queries, ids)
    own, cited = build_citation_table(docs, ids, extended)
    cite_rows = citation_features(extended, own, cited, ids)

    return queries, blocks, ids, extended, views, scores, gold, type_rows, cite_rows


def build_rows(
    views,
    view_names,
    extended,
    ids,
    scores,
    type_rows,
    cite_rows,
    *,
    use_doctype=True,
    use_citation=True,
):
    from tune_expanded_fusion_selection import ltr_features
    rows, groups = ltr_features(
        views,
        view_names,
        extended,
        ids,
        scores,
    )
    for q in ids:
        add = [rows[q]]
        if use_doctype:
            add.append(type_rows[q])
        if use_citation:
            add.append(cite_rows[q])
        rows[q] = np.concatenate(add, axis=1)
    return rows, groups


def run_lobo(
    *,
    variant,
    queries,
    blocks,
    ids,
    extended,
    views,
    scores,
    gold,
    type_rows,
    cite_rows,
    view_names,
    use_doctype,
    use_citation,
    C,
):
    rows, groups = build_rows(
        views,
        view_names,
        extended,
        ids,
        scores,
        type_rows,
        cite_rows,
        use_doctype=use_doctype,
        use_citation=use_citation,
    )
    feature_dim = int(rows[ids[0]].shape[1])
    pred = {}
    decision = {}

    for held in sorted(blocks):
        train = sum((blocks[b] for b in blocks if b != held), [])
        test = blocks[held]

        X = np.vstack([rows[q] for q in train])
        y = np.concatenate([
            [d in gold[q] for d in groups[q]]
            for q in train
        ]).astype(np.int8)

        scaler = StandardScaler().fit(X)
        model = LogisticRegression(
            C=C,
            class_weight="balanced",
            solver="liblinear",
            max_iter=3000,
            random_state=SEED,
        )
        model.fit(scaler.transform(X), y)

        for q in test:
            s = model.decision_function(scaler.transform(rows[q]))
            # Use stable NumPy ordering consistently for all variants.
            order = np.argsort(-s, kind="stable")
            pred[q] = [groups[q][i] for i in order[:5]]
            decision[q] = [float(s[i]) for i in order[:10]]

    per_r = {}
    per_p = {}
    for q in ids:
        h = len(set(pred[q]) & gold[q])
        per_r[q] = h / len(gold[q])
        per_p[q] = h / 5.0

    return {
        "name": variant,
        "feature_dim": feature_dim,
        "C": C,
        "rank_views": list(view_names),
        "score_channels": sorted(scores),
        "use_doctype": use_doctype,
        "use_citation": use_citation,
        "recall": float(np.mean([per_r[q] for q in ids])),
        "precision": float(np.mean([per_p[q] for q in ids])),
        "single_recall": float(np.mean([
            per_r[q] for q in ids if len(gold[q]) == 1
        ])),
        "multi_recall": float(np.mean([
            per_r[q] for q in ids if len(gold[q]) > 1
        ])),
        "blocks": {
            b: float(np.mean([per_r[q] for q in blocks[b]]))
            for b in sorted(blocks)
        },
        "pred": pred,
        "per_query_recall": per_r,
    }


def compare(base, cand, ids, blocks, gold, bootstrap=5000):
    diff = np.asarray([
        cand["per_query_recall"][q] - base["per_query_recall"][q]
        for q in ids
    ], dtype=np.float64)

    wins = int(np.sum(diff > 1e-12))
    losses = int(np.sum(diff < -1e-12))
    ties = len(ids) - wins - losses

    block_delta = {
        b: cand["blocks"][b] - base["blocks"][b]
        for b in sorted(blocks)
    }

    rng = np.random.default_rng(SEED)
    n = len(diff)
    boot = np.empty(bootstrap, dtype=np.float64)
    for i in range(bootstrap):
        idx = rng.integers(0, n, size=n)
        boot[i] = float(np.mean(diff[idx]))

    same_pred = sum(
        base["pred"][q] == cand["pred"][q]
        for q in ids
    )

    return {
        "delta_recall": cand["recall"] - base["recall"],
        "delta_precision": cand["precision"] - base["precision"],
        "delta_single": cand["single_recall"] - base["single_recall"],
        "delta_multi": cand["multi_recall"] - base["multi_recall"],
        "block_deltas": block_delta,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "ordered_top5_exact_matches": same_pred,
        "ordered_top5_churn": len(ids) - same_pred,
        "bootstrap_mean_delta": float(np.mean(boot)),
        "bootstrap_ci95": [
            float(np.percentile(boot, 2.5)),
            float(np.percentile(boot, 97.5)),
        ],
        "bootstrap_p_delta_gt_0": float(np.mean(boot > 0)),
        "strict_no_block_regression": all(v >= -1e-12 for v in block_delta.values()),
    }


def strip_runtime(x):
    return {k: v for k, v in x.items() if k not in ("pred", "per_query_recall")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=Path, required=True)
    ap.add_argument("--bootstrap", type=int, default=5000)
    args = ap.parse_args()
    root = args.repo_root.resolve()

    out = root / "results/manual/huy_d1_48d_group_ablation_v1"
    out.mkdir(parents=True, exist_ok=True)

    print("[1/5] Loading exact D1 inputs...", flush=True)
    (
        queries, blocks, ids, extended, views, full_scores,
        gold, type_rows, cite_rows,
    ) = load_inputs(root)

    print("[2/5] Exact baseline parity...", flush=True)
    baseline = run_lobo(
        variant="D1_48D_BASELINE",
        queries=queries,
        blocks=blocks,
        ids=ids,
        extended=extended,
        views=views,
        scores=full_scores,
        gold=gold,
        type_rows=type_rows,
        cite_rows=cite_rows,
        view_names=D1_VIEWS,
        use_doctype=True,
        use_citation=True,
        C=.15,
    )
    print(
        f"  baseline dim={baseline['feature_dim']} "
        f"R={baseline['recall']:.12f} P={baseline['precision']:.12f}",
        flush=True,
    )
    if baseline["feature_dim"] != 48:
        raise RuntimeError(f"D1 dimension parity failed: {baseline['feature_dim']}")
    if abs(baseline["recall"] - EXPECTED_R) > 1e-12:
        raise RuntimeError(f"D1 recall parity failed: {baseline['recall']}")
    if abs(baseline["precision"] - EXPECTED_P) > 1e-12:
        raise RuntimeError(f"D1 precision parity failed: {baseline['precision']}")

    variants = []

    # Rank-view leave-one-out.
    for drop in D1_VIEWS:
        variants.append({
            "name": f"DROP_RANKVIEW_{drop}",
            "view_names": [x for x in D1_VIEWS if x != drop],
            "scores": full_scores,
            "doctype": True,
            "citation": True,
            "C": .15,
            "family": "rank_view",
            "removed": [drop],
        })

    # Score-channel leave-one-out.
    for drop in sorted(full_scores):
        variants.append({
            "name": f"DROP_SCORE_{drop}",
            "view_names": D1_VIEWS,
            "scores": {k: v for k, v in full_scores.items() if k != drop},
            "doctype": True,
            "citation": True,
            "C": .15,
            "family": "score_channel",
            "removed": [drop],
        })

    # Metadata groups.
    variants += [
        {
            "name": "DROP_DOCTYPE",
            "view_names": D1_VIEWS,
            "scores": full_scores,
            "doctype": False,
            "citation": True,
            "C": .15,
            "family": "metadata",
            "removed": ["doctype_12D"],
        },
        {
            "name": "DROP_CITATION",
            "view_names": D1_VIEWS,
            "scores": full_scores,
            "doctype": True,
            "citation": False,
            "C": .15,
            "family": "metadata",
            "removed": ["citation_4D"],
        },
        {
            "name": "DROP_DOCTYPE_CITATION",
            "view_names": D1_VIEWS,
            "scores": full_scores,
            "doctype": False,
            "citation": False,
            "C": .15,
            "family": "metadata",
            "removed": ["doctype_12D", "citation_4D"],
        },
        {
            "name": "DROP_PROVENANCE_RISK_BUNDLE",
            "view_names": D1_VIEWS,
            "scores": {
                k: v for k, v in full_scores.items()
                if k not in {"crossenc", "aiteamvn_ft", "jina_ft"}
            },
            "doctype": True,
            "citation": True,
            "C": .15,
            "family": "provenance",
            "removed": ["crossenc", "aiteamvn_ft", "jina_ft"],
        },
    ]

    # C sweep is diagnostic, not part of strict pruning selection.
    for c in (.03, .05, .10, .30, .50, 1.0):
        variants.append({
            "name": f"BASELINE_C_{c}",
            "view_names": D1_VIEWS,
            "scores": full_scores,
            "doctype": True,
            "citation": True,
            "C": c,
            "family": "regularization",
            "removed": [],
        })

    print(f"[3/5] Running {len(variants)} exact LOBO variants...", flush=True)
    results = []
    for i, spec in enumerate(variants, 1):
        r = run_lobo(
            variant=spec["name"],
            queries=queries,
            blocks=blocks,
            ids=ids,
            extended=extended,
            views=views,
            scores=spec["scores"],
            gold=gold,
            type_rows=type_rows,
            cite_rows=cite_rows,
            view_names=spec["view_names"],
            use_doctype=spec["doctype"],
            use_citation=spec["citation"],
            C=spec["C"],
        )
        cmp = compare(
            baseline, r, ids, blocks, gold, bootstrap=args.bootstrap
        )
        row = {
            **strip_runtime(r),
            "family": spec["family"],
            "removed": spec["removed"],
            "comparison": cmp,
        }
        results.append(row)
        print(
            f"  {i:02d}/{len(variants)} {spec['name']:<36s} "
            f"dim={r['feature_dim']:2d} "
            f"R={r['recall']:.10f} "
            f"dR={cmp['delta_recall']:+.10f} "
            f"W/L={cmp['wins']}/{cmp['losses']} "
            f"minBlock={min(cmp['block_deltas'].values()):+.6f}",
            flush=True,
        )

    print("[4/5] Ranking robust ablations...", flush=True)
    baseline_public = strip_runtime(baseline)

    for r in results:
        c = r["comparison"]
        is_removal = bool(r["removed"])
        exact_parity_smaller = (
            is_removal
            and r["feature_dim"] < baseline["feature_dim"]
            and c["ordered_top5_exact_matches"] == len(ids)
        )
        strict = (
            is_removal
            and c["delta_recall"] >= -1e-12
            and c["strict_no_block_regression"]
            and (
                c["wins"] > c["losses"]
                or exact_parity_smaller
            )
        )
        r["strict_promote_candidate"] = bool(strict)
        r["exact_parity_smaller"] = bool(exact_parity_smaller)

    # Robust ordering: strict gate, worst block, pooled delta, paired net wins,
    # fewer dimensions, bootstrap support.
    ranked = sorted(
        results,
        key=lambda r: (
            r["strict_promote_candidate"],
            min(r["comparison"]["block_deltas"].values()),
            r["comparison"]["delta_recall"],
            r["comparison"]["wins"] - r["comparison"]["losses"],
            -r["feature_dim"],
            r["comparison"]["bootstrap_p_delta_gt_0"],
        ),
        reverse=True,
    )

    report = {
        "schema": "manual.d1_48d_group_ablation_v1",
        "baseline": baseline_public,
        "feature_contract": {
            "rank_geometry": {
                "views": D1_VIEWS,
                "dimension": 12,
                "definition": "2 features/view + global min_rank + mean_rank",
            },
            "score_geometry": {
                "channels": sorted(full_scores),
                "dimension": 2 * len(full_scores),
                "definition": "within-query z-score + distance-to-query-top per channel",
            },
            "doctype_dimension": int(type_rows[ids[0]].shape[1]),
            "citation_dimension": int(cite_rows[ids[0]].shape[1]),
            "total_dimension": 48,
        },
        "provenance_risk_note": {
            "crossenc": (
                "historical CV cache source script is not fully committed in current repo"
            ),
            "aiteamvn_ft": (
                "checkpoint training chunk/negative-mining provenance not fully established"
            ),
            "jina_ft": (
                "checkpoint training chunk/negative-mining provenance not fully established"
            ),
            "interpretation": (
                "This is a provenance-risk flag, not proof of leakage. "
                "Ablation tests whether D1 depends materially on these channels."
            ),
        },
        "selection_policy": {
            "public_labels_used": False,
            "strict_candidate": (
                "feature-removal variant with pooled recall >= baseline, "
                "no block regression, and wins>losses OR exact prediction parity "
                "at lower dimension"
            ),
            "multiple_comparison_warning": (
                "CAL600 is discovery data here. Any chosen candidate must be "
                "confirmed by public leaderboard before private deployment."
            ),
        },
        "results_ranked": ranked,
    }

    (out / "REPORT.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("[5/5] RESULT", flush=True)
    print("=" * 112)
    print(
        f"Baseline 48D: R={baseline['recall']:.10f} "
        f"P={baseline['precision']:.10f}"
    )
    strict = [r for r in ranked if r["strict_promote_candidate"]]
    print(f"Strict removal candidates: {len(strict)}")
    for r in ranked[:10]:
        c = r["comparison"]
        print(
            f"{r['name']:<38s} dim={r['feature_dim']:2d} "
            f"dR={c['delta_recall']:+.10f} "
            f"dP={c['delta_precision']:+.10f} "
            f"W/L/T={c['wins']}/{c['losses']}/{c['ties']} "
            f"blocks={{{', '.join(f'{k}:{v:+.4f}' for k,v in c['block_deltas'].items())}}} "
            f"strict={r['strict_promote_candidate']}"
        )
    print("Report:", out / "REPORT.json")
    print("=" * 112)


if __name__ == "__main__":
    main()
