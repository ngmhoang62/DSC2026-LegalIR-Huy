#!/usr/bin/env python
"""
AUDIT PRISM SCORE+RANK CHANNEL PRUNING
======================================

CPU-only. No private labels. No neural inference.

Starting point
--------------
This audit starts from the already-promising Prism arm:

    D1 + Prism score channel + Prism rank view

At C=.15 the previously observed CAL600 LOBO result was:

    Recall@5 = 0.9636111111111111

The question here is NOT "does legacy channel X help D1?"
It is:

    "After Prism is added, is legacy channel X still useful,
     or is it now redundant/noisy?"

This is a new conditional ablation around the Prism arm.

Control
-------
PRISM_SCORE_RANK_CONTROL
  rank views:
      base, expanded, jina, dense, corpus, prism_ft
  score channels:
      original 10 D1 score channels + prism_ft
  expected dimension:
      52D

Pruning arms
------------
  DROP_AITEAM_FT
  DROP_JINA_FT
  DROP_AITEAM_JINA

  DROP_CROSSENC
  DROP_CROSSENC_JINA
  DROP_CROSSENC_AITEAM

  DROP_PROVENANCE_BUNDLE
      = crossenc + aiteamvn_ft + jina_ft

Robustness
----------
Every arm is compared against Prism score+rank CONTROL at the SAME C:

    C in {0.10, 0.15, 0.20}

Mandatory parity:
    D1 C=.15                     = 0.9569444444444444
    Prism score+rank C=.15       = 0.9636111111111111

Promotion has two levels:

STRONG_PROMOTE
  - median delta across C > 0
  - positive delta at >=2/3 C
  - C=.15 delta >= +0.0010
  - C=.15 wins > losses
  - C=.15 worst block delta >= -0.005
  - C=.15 multi delta >= -0.005

ROBUST_PARITY_PRUNE
  - all 3 C deltas >= 0
  - C=.15 delta >= 0
  - C=.15 wins >= losses
  - no C=.15 block regression
  - C=.15 multi delta >= 0
  - feature dimension lower than control

ROBUST_PARITY_PRUNE is intentionally weaker evidence than STRONG_PROMOTE.
It may justify an alternate private submission, not replacing the Prism control.

Scientific warning
------------------
Latest source inspection indicates the final Prism fine-tune notebook may have
trained on a pool containing CAL600 queries. Therefore Prism-related CAL metrics
must be treated as contamination-biased mechanism/stability evidence, NOT a
clean generalization estimate.

Output
------
results/manual/huy_prism_channel_pruning_v1/
  REPORT.json
  PROMOTED_ARM.json     (only when strong or parity gate passes)
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


EXPECTED_D1_C015 = 0.9569444444444444
EXPECTED_PRISM_SR_C015 = 0.9636111111111111
SEED = 2026
C_GRID = [0.10, 0.15, 0.20]

BASE_VIEWS = ["base", "expanded", "jina", "dense", "corpus"]
CONTROL_NAME = "PRISM_SCORE_RANK_CONTROL"

PRUNE_ARMS = {
    "DROP_AITEAM_FT": {"aiteamvn_ft"},
    "DROP_JINA_FT": {"jina_ft"},
    "DROP_AITEAM_JINA": {"aiteamvn_ft", "jina_ft"},
    "DROP_CROSSENC": {"crossenc"},
    "DROP_CROSSENC_JINA": {"crossenc", "jina_ft"},
    "DROP_CROSSENC_AITEAM": {"crossenc", "aiteamvn_ft"},
    "DROP_PROVENANCE_BUNDLE": {"crossenc", "aiteamvn_ft", "jina_ft"},
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_pickle_scores(path: Path):
    obj = pickle.loads(path.read_bytes())
    if isinstance(obj, dict) and isinstance(obj.get("scores"), dict):
        obj = obj["scores"]
    if not isinstance(obj, dict):
        raise RuntimeError(
            f"Unsupported Prism score artifact: {type(obj)}"
        )
    return {
        str(q): {str(d): float(s) for d, s in row.items()}
        for q, row in obj.items()
    }


def aligned_scores(raw, ids, candidates):
    values = [v for q in raw.values() for v in q.values()]
    if not values:
        raise RuntimeError("Prism artifact contains no scores")

    floor = float(min(values))
    out = {
        q: {
            d: raw.get(q, {}).get(d, floor)
            for d in candidates[q]
        }
        for q in ids
    }

    total = sum(len(candidates[q]) for q in ids)
    present = sum(
        sum(d in raw.get(q, {}) for d in candidates[q])
        for q in ids
    )
    perq = {
        q: (
            sum(d in raw.get(q, {}) for d in candidates[q])
            / max(1, len(candidates[q]))
        )
        for q in ids
    }

    return out, floor, {
        "present": int(present),
        "total": int(total),
        "coverage": float(present / max(1, total)),
        "min_query_coverage": float(min(perq.values())),
        "mean_query_coverage": float(np.mean(list(perq.values()))),
        "queries_with_any_missing": int(
            sum(v < 1.0 for v in perq.values())
        ),
    }


def build_contract(
    *,
    arm,
    full_channels,
    local_views,
    prism,
    candidates,
    ids,
):
    # Prism score+rank is always present in every arm in this experiment.
    channels = dict(full_channels)
    channels["prism_ft"] = prism

    views = dict(local_views)
    views["prism_ft"] = {
        q: sorted(
            candidates[q],
            key=lambda d: (-float(prism[q][d]), str(d)),
        )
        for q in ids
    }
    view_names = BASE_VIEWS + ["prism_ft"]

    if arm == CONTROL_NAME:
        dropped = set()
    elif arm in PRUNE_ARMS:
        dropped = set(PRUNE_ARMS[arm])
        for ch in dropped:
            if ch not in channels:
                raise RuntimeError(
                    f"{arm}: requested drop channel absent: {ch}"
                )
            channels.pop(ch)
    elif arm == "D1":
        # Exact 48D baseline, no Prism at all.
        channels = dict(full_channels)
        views = dict(local_views)
        view_names = list(BASE_VIEWS)
        dropped = set()
    else:
        raise ValueError(arm)

    # Rank geometry: 2/view + 2 global.
    rank_dim = 2 * len(view_names) + 2
    # Score geometry: 2/channel.
    score_dim = 2 * len(channels)
    # Metadata fixed in authoritative D1.
    expected_dim = rank_dim + score_dim + 12 + 4

    return channels, views, view_names, expected_dim, sorted(dropped)


def build_rows(
    *,
    arm,
    ids,
    candidates,
    local_views,
    full_channels,
    prism,
    type_rows,
    cite_rows,
):
    from tune_expanded_fusion_selection import ltr_features

    channels, views, view_names, expected_dim, dropped = build_contract(
        arm=arm,
        full_channels=full_channels,
        local_views=local_views,
        prism=prism,
        candidates=candidates,
        ids=ids,
    )

    rows0, groups = ltr_features(
        views,
        view_names,
        candidates,
        ids,
        channels,
    )

    rows = {
        q: np.concatenate(
            [rows0[q], type_rows[q], cite_rows[q]],
            axis=1,
        )
        for q in ids
    }

    for q in ids:
        got = int(rows[q].shape[1])
        if got != expected_dim:
            raise RuntimeError(
                f"{arm}: feature dim {got} != expected {expected_dim}"
            )

    return rows, groups, channels, view_names, expected_dim, dropped


def run_lobo(
    *,
    arm,
    C,
    blocks,
    all_ids,
    candidates,
    local_views,
    full_channels,
    prism,
    type_rows,
    cite_rows,
    gold,
):
    (
        rows,
        groups,
        channels,
        view_names,
        feature_dim,
        dropped,
    ) = build_rows(
        arm=arm,
        ids=all_ids,
        candidates=candidates,
        local_views=local_views,
        full_channels=full_channels,
        prism=prism,
        type_rows=type_rows,
        cite_rows=cite_rows,
    )

    pred = {}
    decision = {}

    for held in sorted(blocks):
        train = sum(
            (list(blocks[b]) for b in sorted(blocks) if b != held),
            [],
        )
        test = list(blocks[held])

        X = np.vstack([rows[q] for q in train])
        y = np.concatenate([
            [d in gold[q] for d in groups[q]]
            for q in train
        ]).astype(np.int8)

        scaler = StandardScaler().fit(X)
        model = LogisticRegression(
            C=float(C),
            class_weight="balanced",
            solver="liblinear",
            max_iter=3000,
            random_state=SEED,
        )
        model.fit(scaler.transform(X), y)

        for q in test:
            s = np.asarray(
                model.decision_function(scaler.transform(rows[q])),
                dtype=np.float64,
            )
            order = np.argsort(-s, kind="stable")
            pred[q] = [groups[q][i] for i in order[:5]]
            decision[q] = {
                groups[q][i]: float(s[i])
                for i in range(len(groups[q]))
            }

    return summarize(
        arm=arm,
        C=C,
        feature_dim=feature_dim,
        view_names=view_names,
        score_channels=sorted(channels),
        dropped=dropped,
        pred=pred,
        decision=decision,
        gold=gold,
        all_ids=all_ids,
        blocks=blocks,
    )


def summarize(
    *,
    arm,
    C,
    feature_dim,
    view_names,
    score_channels,
    dropped,
    pred,
    decision,
    gold,
    all_ids,
    blocks,
):
    per_r = {}
    per_p = {}

    for q in all_ids:
        hit = len(set(pred[q]) & set(gold[q]))
        per_r[q] = hit / max(1, len(gold[q]))
        per_p[q] = hit / 5.0

    return {
        "arm": arm,
        "C": float(C),
        "feature_dim": int(feature_dim),
        "rank_views": list(view_names),
        "score_channels": list(score_channels),
        "dropped_score_channels": list(dropped),
        "recall": float(np.mean([per_r[q] for q in all_ids])),
        "precision": float(np.mean([per_p[q] for q in all_ids])),
        "single_recall": float(np.mean([
            per_r[q]
            for q in all_ids
            if len(gold[q]) == 1
        ])),
        "multi_recall": float(np.mean([
            per_r[q]
            for q in all_ids
            if len(gold[q]) > 1
        ])),
        "blocks": {
            b: float(np.mean([per_r[q] for q in blocks[b]]))
            for b in sorted(blocks)
        },
        "predictions": pred,
        "decision_scores": decision,
        "per_query_recall": per_r,
    }


def compare(base, cand, all_ids):
    wins = losses = set_churn = ordered_churn = 0

    for q in all_ids:
        a = base["per_query_recall"][q]
        b = cand["per_query_recall"][q]
        wins += int(b > a + 1e-12)
        losses += int(b < a - 1e-12)

        set_churn += int(
            set(base["predictions"][q])
            != set(cand["predictions"][q])
        )
        ordered_churn += int(
            base["predictions"][q]
            != cand["predictions"][q]
        )

    return {
        "delta_recall": cand["recall"] - base["recall"],
        "delta_precision": cand["precision"] - base["precision"],
        "delta_single": cand["single_recall"] - base["single_recall"],
        "delta_multi": cand["multi_recall"] - base["multi_recall"],
        "block_delta": {
            b: cand["blocks"][b] - base["blocks"][b]
            for b in sorted(base["blocks"])
        },
        "wins": int(wins),
        "losses": int(losses),
        "net_wins": int(wins - losses),
        "set_top5_churn": int(set_churn),
        "ordered_top5_churn": int(ordered_churn),
    }


def compact(r):
    return {
        k: v
        for k, v in r.items()
        if k not in (
            "predictions",
            "decision_scores",
            "per_query_recall",
        )
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=Path, required=True)
    ap.add_argument("--prism-heldout-scores", type=Path, required=True)
    ap.add_argument("--min-coverage", type=float, default=.95)
    args = ap.parse_args()

    root = args.repo_root.expanduser().resolve()
    score_path = args.prism_heldout_scores.expanduser().resolve()
    sys.path.insert(0, str(root))

    if not score_path.is_file():
        raise FileNotFoundError(score_path)

    print("[1/6] Loading authoritative D1 CAL600 bundle...", flush=True)
    from src.gemini.huy_vnlegal_rank_ablation_v1.evaluate_ablation_cal import (
        load_cal_inputs,
    )
    (
        _queries,
        blocks,
        all_ids,
        extended,
        local_views,
        full_channels,
        gold,
        _vnlegal,
        type_rows,
        cite_rows,
    ) = load_cal_inputs()

    required = {"crossenc", "aiteamvn_ft", "jina_ft"}
    missing = sorted(required - set(full_channels))
    if missing:
        raise RuntimeError(
            f"Authoritative D1 missing expected legacy channels: {missing}"
        )

    print("[2/6] Loading Prism CAL score artifact...", flush=True)
    raw_prism = load_pickle_scores(score_path)
    prism, prism_floor, coverage = aligned_scores(
        raw_prism,
        all_ids,
        extended,
    )
    print(
        f"  coverage={coverage['coverage']:.4%} "
        f"min-query={coverage['min_query_coverage']:.4%} "
        f"missing-queries={coverage['queries_with_any_missing']}",
        flush=True,
    )
    if coverage["coverage"] < args.min_coverage:
        raise RuntimeError(
            f"Prism coverage {coverage['coverage']:.4%} "
            f"< {args.min_coverage:.4%}"
        )

    print("[3/6] Running D1 and Prism score+rank controls...", flush=True)
    results = {}
    comparisons = {}

    control_arms = ["D1", CONTROL_NAME]
    for arm in control_arms:
        for c in C_GRID:
            r = run_lobo(
                arm=arm,
                C=c,
                blocks=blocks,
                all_ids=all_ids,
                candidates=extended,
                local_views=local_views,
                full_channels=full_channels,
                prism=prism,
                type_rows=type_rows,
                cite_rows=cite_rows,
                gold=gold,
            )
            results[(arm, c)] = r
            print(
                f"  {arm:29s} C={c:.2f} "
                f"dim={r['feature_dim']:2d} "
                f"R={r['recall']:.10f} "
                f"P={r['precision']:.10f} "
                f"multi={r['multi_recall']:.6f}",
                flush=True,
            )

    if abs(
        results[("D1", 0.15)]["recall"] - EXPECTED_D1_C015
    ) > 1e-9:
        raise RuntimeError(
            "D1 parity failed: "
            f"{results[('D1', 0.15)]['recall']} "
            f"!= {EXPECTED_D1_C015}"
        )

    if abs(
        results[(CONTROL_NAME, 0.15)]["recall"]
        - EXPECTED_PRISM_SR_C015
    ) > 1e-9:
        raise RuntimeError(
            "Prism score+rank parity failed: "
            f"{results[(CONTROL_NAME, 0.15)]['recall']} "
            f"!= {EXPECTED_PRISM_SR_C015}"
        )

    print("[4/6] Running Prism conditional pruning arms...", flush=True)

    for arm in PRUNE_ARMS:
        print(f"\n--- {arm} ---", flush=True)
        deltas = []

        for c in C_GRID:
            r = run_lobo(
                arm=arm,
                C=c,
                blocks=blocks,
                all_ids=all_ids,
                candidates=extended,
                local_views=local_views,
                full_channels=full_channels,
                prism=prism,
                type_rows=type_rows,
                cite_rows=cite_rows,
                gold=gold,
            )
            results[(arm, c)] = r

            cmp = compare(
                results[(CONTROL_NAME, c)],
                r,
                all_ids,
            )
            comparisons[(arm, c)] = cmp
            deltas.append(cmp["delta_recall"])

            print(
                f"  C={c:.2f} dim={r['feature_dim']:2d} "
                f"R={r['recall']:.10f} "
                f"dR_vs_PrismSR={cmp['delta_recall']:+.10f} "
                f"W/L={cmp['wins']}/{cmp['losses']} "
                f"multiΔ={cmp['delta_multi']:+.6f} "
                f"blocks="
                + ",".join(
                    f"{b}:{cmp['block_delta'][b]:+.4f}"
                    for b in sorted(cmp["block_delta"])
                ),
                flush=True,
            )

        print(
            f"  sensitivity median={np.median(deltas):+.10f} "
            f"positive={sum(x > 1e-12 for x in deltas)}/3 "
            f"nonnegative={sum(x >= -1e-12 for x in deltas)}/3 "
            f"range=[{min(deltas):+.10f},{max(deltas):+.10f}]",
            flush=True,
        )

    print("\n[5/6] Promotion gates relative to Prism score+rank...", flush=True)

    summaries = {}
    strong = []
    parity = []

    control_dim = results[(CONTROL_NAME, 0.15)]["feature_dim"]

    for arm in PRUNE_ARMS:
        cmps = [comparisons[(arm, c)] for c in C_GRID]
        deltas = [x["delta_recall"] for x in cmps]
        cmp15 = comparisons[(arm, 0.15)]
        res15 = results[(arm, 0.15)]

        summary = {
            "arm": arm,
            "dropped_score_channels": sorted(PRUNE_ARMS[arm]),
            "feature_dim": res15["feature_dim"],
            "median_delta_across_C": float(np.median(deltas)),
            "min_delta_across_C": float(min(deltas)),
            "max_delta_across_C": float(max(deltas)),
            "positive_C_count": int(
                sum(x > 1e-12 for x in deltas)
            ),
            "nonnegative_C_count": int(
                sum(x >= -1e-12 for x in deltas)
            ),
            "deltas_by_C": {
                f"{c:.2f}": comparisons[(arm, c)]["delta_recall"]
                for c in C_GRID
            },
            "C015": {
                **compact(res15),
                "comparison_vs_sameC_PrismSR": cmp15,
            },
        }

        strong_gates = {
            "median_delta_positive": (
                summary["median_delta_across_C"] > 1e-12
            ),
            "positive_at_least_2_of_3_C": (
                summary["positive_C_count"] >= 2
            ),
            "C015_delta_ge_0_001": (
                cmp15["delta_recall"] >= 0.001 - 1e-12
            ),
            "C015_wins_gt_losses": (
                cmp15["wins"] > cmp15["losses"]
            ),
            "C015_worst_block_ge_neg_0_005": (
                min(cmp15["block_delta"].values())
                >= -0.005 - 1e-12
            ),
            "C015_multi_ge_neg_0_005": (
                cmp15["delta_multi"] >= -0.005 - 1e-12
            ),
        }

        parity_gates = {
            "all_C_nonnegative": (
                summary["nonnegative_C_count"] == len(C_GRID)
            ),
            "C015_delta_nonnegative": (
                cmp15["delta_recall"] >= -1e-12
            ),
            "C015_wins_ge_losses": (
                cmp15["wins"] >= cmp15["losses"]
            ),
            "C015_no_block_regression": (
                min(cmp15["block_delta"].values())
                >= -1e-12
            ),
            "C015_multi_nonnegative": (
                cmp15["delta_multi"] >= -1e-12
            ),
            "fewer_features_than_control": (
                res15["feature_dim"] < control_dim
            ),
        }

        summary["strong_gates"] = strong_gates
        summary["parity_gates"] = parity_gates
        summary["strong_pass"] = all(strong_gates.values())
        summary["parity_pass"] = all(parity_gates.values())
        summaries[arm] = summary

        if summary["strong_pass"]:
            strong.append(summary)
        elif summary["parity_pass"]:
            parity.append(summary)

        status = (
            "STRONG"
            if summary["strong_pass"]
            else "PARITY"
            if summary["parity_pass"]
            else "FAIL"
        )

        print(
            f"  {arm:29s} "
            f"median={summary['median_delta_across_C']:+.6f} "
            f"C015={cmp15['delta_recall']:+.6f} "
            f"W/L={cmp15['wins']}/{cmp15['losses']} "
            f"positiveC={summary['positive_C_count']}/3 "
            f"=> {status}",
            flush=True,
        )

    recommended = None
    tier = None

    if strong:
        recommended = max(
            strong,
            key=lambda s: (
                s["median_delta_across_C"],
                s["C015"]["comparison_vs_sameC_PrismSR"]["delta_recall"],
                s["C015"]["comparison_vs_sameC_PrismSR"]["net_wins"],
                -s["C015"]["comparison_vs_sameC_PrismSR"]["set_top5_churn"],
                -s["feature_dim"],
            ),
        )
        tier = "STRONG_PROMOTE"
    elif parity:
        recommended = min(
            parity,
            key=lambda s: (
                s["feature_dim"],
                -s["median_delta_across_C"],
                s["C015"]["comparison_vs_sameC_PrismSR"]["set_top5_churn"],
            ),
        )
        tier = "ROBUST_PARITY_PRUNE"

    verdict = (
        f"{tier}_PRISM_CHANNEL_PRUNING"
        if recommended is not None
        else "KILL_PRISM_CHANNEL_PRUNING"
    )

    print("[6/6] Writing report...", flush=True)

    out = root / "results/manual/huy_prism_channel_pruning_v1"
    out.mkdir(parents=True, exist_ok=True)

    report = {
        "schema": "manual.prism_channel_pruning_v1",
        "status": verdict,
        "scientific_warning": (
            "Latest Prism source inspection indicates the final fine-tune "
            "may include CAL600 queries in the training pool. Prism-related "
            "CAL metrics are contamination-biased mechanism/stability evidence, "
            "not a clean generalization estimate."
        ),
        "prism_artifact": {
            "path": str(score_path),
            "sha256": sha256(score_path),
            "floor": prism_floor,
            "coverage": coverage,
        },
        "C_grid": C_GRID,
        "mandatory_parity": {
            "D1_C015_expected": EXPECTED_D1_C015,
            "D1_C015_observed": results[("D1", 0.15)]["recall"],
            "PrismSR_C015_expected": EXPECTED_PRISM_SR_C015,
            "PrismSR_C015_observed": results[
                (CONTROL_NAME, 0.15)
            ]["recall"],
        },
        "controls": {
            "D1": {
                f"{c:.2f}": compact(results[("D1", c)])
                for c in C_GRID
            },
            CONTROL_NAME: {
                f"{c:.2f}": compact(results[(CONTROL_NAME, c)])
                for c in C_GRID
            },
        },
        "arms": summaries,
        "promotion_logic": {
            "strong": {
                "median_delta_across_C_gt": 0.0,
                "positive_C_count_ge": 2,
                "C015_delta_recall_ge": 0.001,
                "C015_wins_gt_losses": True,
                "C015_worst_block_delta_ge": -0.005,
                "C015_multi_delta_ge": -0.005,
            },
            "robust_parity": {
                "all_C_delta_nonnegative": True,
                "C015_wins_ge_losses": True,
                "C015_no_block_regression": True,
                "C015_multi_nonnegative": True,
                "feature_dim_lt_control": True,
                "interpretation": (
                    "alternate private arm only; weaker evidence than strong promote"
                ),
            },
        },
        "recommended_tier": tier,
        "recommended_arm": (
            recommended["arm"] if recommended is not None else None
        ),
    }

    report_path = out / "REPORT.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    promoted_path = out / "PROMOTED_ARM.json"
    if recommended is not None:
        promoted = {
            "schema": "manual.prism_channel_pruning_arm.v1",
            "tier": tier,
            "arm": recommended["arm"],
            "drop_score_channels": recommended["dropped_score_channels"],
            "feature_dim": recommended["feature_dim"],
            "C": 0.15,
            "source_report": str(report_path),
            "cal_contaminated_metrics": recommended,
            "private_deployment_note": (
                "Reuse the same private Prism scores as Prism score+rank. "
                "No additional neural inference is required. Retrain the "
                "selected full-CAL600 LR at C=.15 with the listed legacy "
                "score channels removed, then materialize private predictions."
            ),
        }
        promoted_path.write_text(
            json.dumps(promoted, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    elif promoted_path.exists():
        promoted_path.unlink()

    print("=" * 118)
    print("VERDICT:", verdict)
    print(
        "RECOMMENDED ARM:",
        recommended["arm"] if recommended is not None else None,
    )
    print("TIER:", tier)
    if recommended is not None:
        cmp15 = recommended["C015"]["comparison_vs_sameC_PrismSR"]
        print(
            f"C=.15 dR_vs_PrismSR={cmp15['delta_recall']:+.10f} "
            f"W/L={cmp15['wins']}/{cmp15['losses']} "
            f"churn={cmp15['set_top5_churn']} "
            f"dim={recommended['feature_dim']}"
        )
        print("Promoted arm:", promoted_path)
    print("Report:", report_path)
    print("=" * 118)


if __name__ == "__main__":
    main()
