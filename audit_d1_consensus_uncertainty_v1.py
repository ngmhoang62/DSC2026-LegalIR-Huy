#!/usr/bin/env python
"""
D1 CROSS-EXPERT CONSENSUS + BOUNDARY-UNCERTAINTY AUDIT
=======================================================

CPU-only. No private labels. No neural inference.

Why this experiment
-------------------
The 1,977 scored private queries are distributionally very close to public/CAL,
so broad query-archetype routing is not the first bet. The stronger label-free
signal is D1 boundary uncertainty: a substantial slice of queries has a small
rank5-vs-rank6 decision margin.

This audit therefore tests whether D1 is missing *agreement geometry* between
its existing experts, and whether that information is safest when applied only
to the most boundary-uncertain queries.

Existing D1 48D already has:
  - 5 rank views, with reciprocal ranks, normalized ranks, min/mean rank
  - 10 score channels, each z-score + top-distance
  - 12 doctype/type features
  - 4 citation features

It does NOT explicitly encode:
  - how many rank experts put a candidate in top-5/top-10
  - median/std of cross-expert rank
  - how many SCORE channels rank a candidate in top-5/top-10
  - median/std of score-channel ordinal rank

Arms
----
D1                          48D control

RANK_CONSENSUS              +4D = 52D
  rank_top5_support
  rank_top10_support
  rank_median / 60
  rank_std / 60

SCORE_CONSENSUS             +4D = 52D
  score_top5_support
  score_top10_support
  score_rank_median / 60
  score_rank_std / 60

FULL_CONSENSUS              +8D = 56D
  all rank + score consensus features

ROBUST_CONSENSUS            +6D = 54D
  rank_top5_support, rank_median, rank_std
  score_top5_support, score_rank_median, score_rank_std

Selective routing
-----------------
For every consensus arm, also evaluate a fixed label-free selective policy:

    SELECTIVE_Q25:
      use consensus-arm prediction only for the bottom 25% of queries by
      baseline D1 rank5-rank6 decision margin;
      keep exact D1 prediction for the other 75%.

The Q25 fraction is preregistered from the private label-free audit, where the
D1 private margin distribution had a meaningful low-margin quartile. No private
labels or Codabench per-query outcomes are used.

Q10 is reported as a DIAGNOSTIC ONLY and is never eligible for automatic
promotion.

C sensitivity
-------------
Run C in {0.10, 0.15, 0.20}; compare every arm to D1 at the SAME C.

Mandatory parity:
    D1 C=.15 Recall@5 == 0.9569444444444444

Promotion
---------
Eligible candidates:
  - GLOBAL arm
  - SELECTIVE_Q25 version of each arm

Gate:
  - median delta across C > 0
  - positive delta at >=2/3 C
  - C=.15 delta Recall >= +0.0025
  - C=.15 wins > losses
  - C=.15 worst block delta >= -0.005
  - C=.15 multi-gold delta >= -0.005

When candidates are close, SELECTIVE_Q25 is preferred because it changes fewer
queries and preserves the champion on confident cases.

Outputs
-------
results/manual/huy_d1_consensus_uncertainty_v1/
  REPORT.json
  PROMOTED_ARM.json       (only if a gate passes)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


EXPECTED_D1_C015 = 0.9569444444444444
SEED = 2026
C_GRID = [0.10, 0.15, 0.20]
BASE_VIEWS = ["base", "expanded", "jina", "dense", "corpus"]

ARMS = [
    "D1",
    "RANK_CONSENSUS",
    "SCORE_CONSENSUS",
    "FULL_CONSENSUS",
    "ROBUST_CONSENSUS",
]


def score_rank_maps(full_channels, candidates, all_ids):
    """
    Convert every D1 score channel into a per-query ordinal rank map.

    Missing values use the exact D1-style fallback mean - 2*std before ranking.
    This makes score-consensus about ordering, not raw cross-model scale.
    """
    channel_names = sorted(full_channels)
    out = {q: [] for q in all_ids}

    for q in all_ids:
        docs = list(candidates[q])
        for name in channel_names:
            raw = full_channels[name].get(q, {})
            vals = np.asarray(
                [raw.get(d, np.nan) for d in docs],
                dtype=np.float64,
            )
            present = vals[~np.isnan(vals)]
            if present.size:
                mean = float(present.mean())
                std = float(present.std())
                if std <= 1e-12:
                    std = 1.0
            else:
                mean, std = 0.0, 1.0
            filled = np.where(np.isnan(vals), mean - 2.0 * std, vals)

            order = sorted(
                range(len(docs)),
                key=lambda i: (-float(filled[i]), str(docs[i])),
            )
            rank = {docs[i]: j + 1 for j, i in enumerate(order)}
            out[q].append(rank)

    return channel_names, out


def consensus_features(
    *,
    candidates,
    local_views,
    score_rankers,
    all_ids,
):
    """
    Build four rank-consensus and four score-consensus features per candidate.
    """
    rank_rows = {}
    score_rows = {}

    for q in all_ids:
        docs = list(candidates[q])
        rank_maps = [
            {d: i + 1 for i, d in enumerate(local_views[name][q])}
            for name in BASE_VIEWS
        ]
        score_maps = score_rankers[q]

        rr = []
        sr = []

        for d in docs:
            r = np.asarray(
                [rm.get(d, 60) for rm in rank_maps],
                dtype=np.float64,
            )
            s = np.asarray(
                [rm.get(d, 60) for rm in score_maps],
                dtype=np.float64,
            )

            rr.append([
                float(np.mean(r <= 5)),
                float(np.mean(r <= 10)),
                float(np.median(r) / 60.0),
                float(np.std(r) / 60.0),
            ])
            sr.append([
                float(np.mean(s <= 5)),
                float(np.mean(s <= 10)),
                float(np.median(s) / 60.0),
                float(np.std(s) / 60.0),
            ])

        rank_rows[q] = np.asarray(rr, dtype=np.float32)
        score_rows[q] = np.asarray(sr, dtype=np.float32)

    return rank_rows, score_rows


def build_rows(
    *,
    arm,
    all_ids,
    candidates,
    local_views,
    full_channels,
    type_rows,
    cite_rows,
    rank_consensus,
    score_consensus,
):
    from tune_expanded_fusion_selection import ltr_features

    rows0, groups = ltr_features(
        local_views,
        BASE_VIEWS,
        candidates,
        all_ids,
        full_channels,
    )

    rows = {}
    for q in all_ids:
        pieces = [rows0[q], type_rows[q], cite_rows[q]]

        if arm == "D1":
            expected_dim = 48

        elif arm == "RANK_CONSENSUS":
            pieces.append(rank_consensus[q])
            expected_dim = 52

        elif arm == "SCORE_CONSENSUS":
            pieces.append(score_consensus[q])
            expected_dim = 52

        elif arm == "FULL_CONSENSUS":
            pieces += [rank_consensus[q], score_consensus[q]]
            expected_dim = 56

        elif arm == "ROBUST_CONSENSUS":
            # [top5_support, median, std] from each family; omit top10 support.
            rc = rank_consensus[q][:, [0, 2, 3]]
            sc = score_consensus[q][:, [0, 2, 3]]
            pieces += [rc, sc]
            expected_dim = 54

        else:
            raise ValueError(arm)

        rows[q] = np.concatenate(pieces, axis=1)
        got = int(rows[q].shape[1])
        if got != expected_dim:
            raise RuntimeError(
                f"{arm}: feature dim {got} != expected {expected_dim}"
            )

    return rows, groups, expected_dim


def evaluate_prediction(pred, gold, all_ids, blocks):
    per_r = {}
    per_p = {}
    for q in all_ids:
        hit = len(set(pred[q][:5]) & set(gold[q]))
        per_r[q] = hit / max(1, len(gold[q]))
        per_p[q] = hit / 5.0

    single = [per_r[q] for q in all_ids if len(gold[q]) == 1]
    multi = [per_r[q] for q in all_ids if len(gold[q]) > 1]

    return {
        "recall": float(np.mean([per_r[q] for q in all_ids])),
        "precision": float(np.mean([per_p[q] for q in all_ids])),
        "single_recall": float(np.mean(single)),
        "multi_recall": float(np.mean(multi)),
        "blocks": {
            b: float(np.mean([per_r[q] for q in blocks[b]]))
            for b in sorted(blocks)
        },
        "per_query_recall": per_r,
    }


def run_lobo(
    *,
    arm,
    C,
    blocks,
    all_ids,
    candidates,
    local_views,
    full_channels,
    type_rows,
    cite_rows,
    rank_consensus,
    score_consensus,
    gold,
):
    rows, groups, feature_dim = build_rows(
        arm=arm,
        all_ids=all_ids,
        candidates=candidates,
        local_views=local_views,
        full_channels=full_channels,
        type_rows=type_rows,
        cite_rows=cite_rows,
        rank_consensus=rank_consensus,
        score_consensus=score_consensus,
    )

    pred = {}
    decision_scores = {}
    full_order = {}

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
            order_idx = np.argsort(-s, kind="stable")
            order = [groups[q][i] for i in order_idx]
            pred[q] = order[:5]
            full_order[q] = order
            decision_scores[q] = {
                groups[q][i]: float(s[i])
                for i in range(len(groups[q]))
            }

    metrics = evaluate_prediction(
        pred, gold, all_ids, blocks
    )

    return {
        "arm": arm,
        "C": float(C),
        "feature_dim": int(feature_dim),
        "predictions": pred,
        "full_order": full_order,
        "decision_scores": decision_scores,
        **metrics,
    }


def margin5_6(run, all_ids):
    out = {}
    for q in all_ids:
        order = run["full_order"][q]
        if len(order) < 6:
            raise RuntimeError(f"Need >=6 candidates for qid={q}")
        s = run["decision_scores"][q]
        out[q] = float(s[order[4]] - s[order[5]])
    return out


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
        "delta_single": (
            cand["single_recall"] - base["single_recall"]
        ),
        "delta_multi": (
            cand["multi_recall"] - base["multi_recall"]
        ),
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


def compact_run(run):
    return {
        k: v
        for k, v in run.items()
        if k not in (
            "predictions",
            "full_order",
            "decision_scores",
            "per_query_recall",
        )
    }


def selective_prediction(
    base_run,
    arm_run,
    margins,
    all_ids,
    fraction,
):
    vals = np.asarray([margins[q] for q in all_ids], dtype=np.float64)
    threshold = float(np.quantile(vals, fraction))

    selected = {
        q for q in all_ids
        if margins[q] <= threshold
    }

    pred = {
        q: (
            arm_run["predictions"][q]
            if q in selected
            else base_run["predictions"][q]
        )
        for q in all_ids
    }
    return pred, threshold, selected


def subset_delta(base, cand, ids):
    if not ids:
        return None
    b = np.mean([base["per_query_recall"][q] for q in ids])
    c = np.mean([cand["per_query_recall"][q] for q in ids])
    return {
        "queries": len(ids),
        "baseline_recall": float(b),
        "candidate_recall": float(c),
        "delta_recall": float(c - b),
        "wins": int(sum(
            cand["per_query_recall"][q]
            > base["per_query_recall"][q] + 1e-12
            for q in ids
        )),
        "losses": int(sum(
            cand["per_query_recall"][q]
            < base["per_query_recall"][q] - 1e-12
            for q in ids
        )),
    }


def summarize_candidate(
    *,
    name,
    mode,
    source_arm,
    results_by_C,
    comparisons_by_C,
    selection_meta_by_C=None,
):
    deltas = [
        comparisons_by_C[c]["delta_recall"]
        for c in C_GRID
    ]
    cmp15 = comparisons_by_C[0.15]
    res15 = results_by_C[0.15]

    summary = {
        "name": name,
        "mode": mode,
        "source_arm": source_arm,
        "feature_dim": int(res15["feature_dim"]),
        "median_delta_across_C": float(np.median(deltas)),
        "min_delta_across_C": float(min(deltas)),
        "max_delta_across_C": float(max(deltas)),
        "positive_C_count": int(
            sum(x > 1e-12 for x in deltas)
        ),
        "deltas_by_C": {
            f"{c:.2f}": comparisons_by_C[c]["delta_recall"]
            for c in C_GRID
        },
        "C015": {
            **compact_run(res15),
            "comparison_vs_sameC_D1": cmp15,
        },
    }

    if selection_meta_by_C is not None:
        summary["selection_by_C"] = {
            f"{c:.2f}": selection_meta_by_C[c]
            for c in C_GRID
        }

    gates = {
        "median_delta_positive": (
            summary["median_delta_across_C"] > 1e-12
        ),
        "positive_at_least_2_of_3_C": (
            summary["positive_C_count"] >= 2
        ),
        "C015_delta_ge_0_0025": (
            cmp15["delta_recall"] >= 0.0025 - 1e-12
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
    summary["gates"] = gates
    summary["all_gates_pass"] = all(gates.values())
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=Path, required=True)
    args = ap.parse_args()

    root = args.repo_root.expanduser().resolve()
    sys.path.insert(0, str(root))

    print("[1/7] Loading authoritative D1 CAL600 bundle...", flush=True)
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

    if len(BASE_VIEWS) != 5:
        raise RuntimeError("Expected exactly five D1 rank views")
    if len(full_channels) != 10:
        raise RuntimeError(
            f"Expected exactly ten D1 score channels, got {len(full_channels)}: "
            f"{sorted(full_channels)}"
        )

    print("[2/7] Building cross-expert ordinal consensus geometry...", flush=True)
    score_channel_names, score_rankers = score_rank_maps(
        full_channels,
        extended,
        all_ids,
    )
    rank_consensus, score_consensus = consensus_features(
        candidates=extended,
        local_views=local_views,
        score_rankers=score_rankers,
        all_ids=all_ids,
    )
    print(
        f"  rank experts={len(BASE_VIEWS)} "
        f"score experts={len(score_channel_names)}",
        flush=True,
    )

    print("[3/7] Running exact D1 controls and OOF boundary margins...", flush=True)
    runs = {}
    margins_by_C = {}

    for c in C_GRID:
        r = run_lobo(
            arm="D1",
            C=c,
            blocks=blocks,
            all_ids=all_ids,
            candidates=extended,
            local_views=local_views,
            full_channels=full_channels,
            type_rows=type_rows,
            cite_rows=cite_rows,
            rank_consensus=rank_consensus,
            score_consensus=score_consensus,
            gold=gold,
        )
        runs[("D1", c)] = r
        margins = margin5_6(r, all_ids)
        margins_by_C[c] = margins
        vals = np.asarray(list(margins.values()))

        print(
            f"  D1 C={c:.2f} "
            f"R={r['recall']:.10f} "
            f"P={r['precision']:.10f} "
            f"multi={r['multi_recall']:.6f} | "
            f"margin5-6 p10={np.quantile(vals,.10):.4f} "
            f"p25={np.quantile(vals,.25):.4f} "
            f"p50={np.quantile(vals,.50):.4f}",
            flush=True,
        )

    if abs(
        runs[("D1", 0.15)]["recall"] - EXPECTED_D1_C015
    ) > 1e-9:
        raise RuntimeError(
            f"D1 parity failed: {runs[('D1',0.15)]['recall']} "
            f"!= {EXPECTED_D1_C015}"
        )

    print("[4/7] Running global consensus feature arms...", flush=True)
    global_cmp = {}
    uncertainty_diagnostics = {}

    for arm in ARMS[1:]:
        print(f"\n--- {arm} ---", flush=True)
        for c in C_GRID:
            r = run_lobo(
                arm=arm,
                C=c,
                blocks=blocks,
                all_ids=all_ids,
                candidates=extended,
                local_views=local_views,
                full_channels=full_channels,
                type_rows=type_rows,
                cite_rows=cite_rows,
                rank_consensus=rank_consensus,
                score_consensus=score_consensus,
                gold=gold,
            )
            runs[(arm, c)] = r
            cmp = compare(runs[("D1", c)], r, all_ids)
            global_cmp[(arm, c)] = cmp

            margins = margins_by_C[c]
            vals = np.asarray([margins[q] for q in all_ids])
            q10 = float(np.quantile(vals, .10))
            q25 = float(np.quantile(vals, .25))
            ids10 = [q for q in all_ids if margins[q] <= q10]
            ids25 = [q for q in all_ids if margins[q] <= q25]

            uncertainty_diagnostics[(arm, c)] = {
                "baseline_margin_q10": q10,
                "baseline_margin_q25": q25,
                "bottom10": subset_delta(
                    runs[("D1", c)], r, ids10
                ),
                "bottom25": subset_delta(
                    runs[("D1", c)], r, ids25
                ),
            }

            print(
                f"  C={c:.2f} dim={r['feature_dim']:2d} "
                f"R={r['recall']:.10f} "
                f"dR={cmp['delta_recall']:+.10f} "
                f"W/L={cmp['wins']}/{cmp['losses']} "
                f"multiΔ={cmp['delta_multi']:+.6f} "
                f"low25Δ={uncertainty_diagnostics[(arm,c)]['bottom25']['delta_recall']:+.6f}",
                flush=True,
            )

    print("\n[5/7] Evaluating preregistered selective-Q25 routing...", flush=True)
    selective_runs = {}
    selective_cmp = {}
    selective_meta = {}
    q10_diagnostic = {}

    for arm in ARMS[1:]:
        print(f"\n--- SELECTIVE_Q25::{arm} ---", flush=True)

        for c in C_GRID:
            base = runs[("D1", c)]
            cand = runs[(arm, c)]
            margins = margins_by_C[c]

            pred25, threshold25, selected25 = selective_prediction(
                base,
                cand,
                margins,
                all_ids,
                .25,
            )
            m25 = evaluate_prediction(
                pred25, gold, all_ids, blocks
            )
            run25 = {
                "arm": arm,
                "mode": "SELECTIVE_Q25",
                "C": c,
                "feature_dim": cand["feature_dim"],
                "predictions": pred25,
                **m25,
            }
            selective_runs[(arm, c)] = run25
            cmp25 = compare(base, run25, all_ids)
            selective_cmp[(arm, c)] = cmp25
            selective_meta[(arm, c)] = {
                "fraction": .25,
                "margin_threshold": threshold25,
                "selected_queries": len(selected25),
                "selected_fraction": len(selected25) / len(all_ids),
                "candidate_changes_within_selected": int(sum(
                    set(base["predictions"][q])
                    != set(cand["predictions"][q])
                    for q in selected25
                )),
            }

            # Q10 diagnostic only.
            pred10, threshold10, selected10 = selective_prediction(
                base,
                cand,
                margins,
                all_ids,
                .10,
            )
            m10 = evaluate_prediction(
                pred10, gold, all_ids, blocks
            )
            q10_diagnostic[(arm, c)] = {
                "margin_threshold": threshold10,
                "selected_queries": len(selected10),
                "recall": m10["recall"],
                "delta_recall_vs_D1": m10["recall"] - base["recall"],
            }

            print(
                f"  C={c:.2f} "
                f"threshold={threshold25:.6f} "
                f"selected={len(selected25):3d}/{len(all_ids)} "
                f"R={run25['recall']:.10f} "
                f"dR={cmp25['delta_recall']:+.10f} "
                f"W/L={cmp25['wins']}/{cmp25['losses']} "
                f"churn={cmp25['set_top5_churn']}",
                flush=True,
            )

    print("\n[6/7] Promotion gates...", flush=True)
    candidate_summaries = []

    # Global candidates.
    for arm in ARMS[1:]:
        results_by_C = {
            c: runs[(arm, c)]
            for c in C_GRID
        }
        cmps_by_C = {
            c: global_cmp[(arm, c)]
            for c in C_GRID
        }
        s = summarize_candidate(
            name=f"GLOBAL::{arm}",
            mode="GLOBAL",
            source_arm=arm,
            results_by_C=results_by_C,
            comparisons_by_C=cmps_by_C,
        )
        candidate_summaries.append(s)

    # Selective Q25 candidates.
    for arm in ARMS[1:]:
        results_by_C = {
            c: selective_runs[(arm, c)]
            for c in C_GRID
        }
        cmps_by_C = {
            c: selective_cmp[(arm, c)]
            for c in C_GRID
        }
        sel_meta = {
            c: selective_meta[(arm, c)]
            for c in C_GRID
        }
        s = summarize_candidate(
            name=f"SELECTIVE_Q25::{arm}",
            mode="SELECTIVE_Q25",
            source_arm=arm,
            results_by_C=results_by_C,
            comparisons_by_C=cmps_by_C,
            selection_meta_by_C=sel_meta,
        )
        candidate_summaries.append(s)

    eligible = [x for x in candidate_summaries if x["all_gates_pass"]]

    for s in candidate_summaries:
        cmp15 = s["C015"]["comparison_vs_sameC_D1"]
        print(
            f"  {s['name']:<38s} "
            f"median={s['median_delta_across_C']:+.6f} "
            f"C015={cmp15['delta_recall']:+.6f} "
            f"W/L={cmp15['wins']}/{cmp15['losses']} "
            f"positiveC={s['positive_C_count']}/3 "
            f"=> {'PASS' if s['all_gates_pass'] else 'FAIL'}",
            flush=True,
        )

    recommended = None
    if eligible:
        # Robust performance first. If nearly tied, prefer selective routing
        # because it perturbs fewer queries.
        best_med = max(x["median_delta_across_C"] for x in eligible)
        near = [
            x for x in eligible
            if x["median_delta_across_C"] >= best_med - 0.0005
        ]
        recommended = max(
            near,
            key=lambda s: (
                1 if s["mode"] == "SELECTIVE_Q25" else 0,
                s["C015"]["comparison_vs_sameC_D1"]["delta_recall"],
                s["C015"]["comparison_vs_sameC_D1"]["net_wins"],
                -s["C015"]["comparison_vs_sameC_D1"]["set_top5_churn"],
            ),
        )

    verdict = (
        "PROMOTE_D1_CONSENSUS_UNCERTAINTY_ARM"
        if recommended is not None
        else "KILL_D1_CONSENSUS_UNCERTAINTY"
    )

    print("[7/7] Writing report...", flush=True)
    out = root / "results/manual/huy_d1_consensus_uncertainty_v1"
    out.mkdir(parents=True, exist_ok=True)

    report = {
        "schema": "manual.d1_consensus_uncertainty_v1",
        "status": verdict,
        "private_labels_used": False,
        "hypothesis": (
            "D1 may be missing nonlinear cross-expert agreement geometry. "
            "A fixed bottom-quartile boundary-uncertainty router may preserve "
            "the champion on confident queries while using consensus features "
            "where rank5/rank6 is ambiguous."
        ),
        "score_channel_names": score_channel_names,
        "C_grid": C_GRID,
        "controls": {
            f"{c:.2f}": {
                **compact_run(runs[("D1", c)]),
                "margin5_6": {
                    "p10": float(np.quantile(
                        list(margins_by_C[c].values()), .10
                    )),
                    "p25": float(np.quantile(
                        list(margins_by_C[c].values()), .25
                    )),
                    "median": float(np.median(
                        list(margins_by_C[c].values())
                    )),
                },
            }
            for c in C_GRID
        },
        "global_arms": {
            arm: {
                f"{c:.2f}": {
                    **compact_run(runs[(arm, c)]),
                    "comparison_vs_D1": global_cmp[(arm, c)],
                    "uncertainty_diagnostic": (
                        uncertainty_diagnostics[(arm, c)]
                    ),
                }
                for c in C_GRID
            }
            for arm in ARMS[1:]
        },
        "selective_q25": {
            arm: {
                f"{c:.2f}": {
                    **compact_run(selective_runs[(arm, c)]),
                    "comparison_vs_D1": selective_cmp[(arm, c)],
                    "selection": selective_meta[(arm, c)],
                }
                for c in C_GRID
            }
            for arm in ARMS[1:]
        },
        "q10_diagnostic_only": {
            arm: {
                f"{c:.2f}": q10_diagnostic[(arm, c)]
                for c in C_GRID
            }
            for arm in ARMS[1:]
        },
        "candidate_summaries": candidate_summaries,
        "promotion_gate": {
            "median_delta_across_C_positive": True,
            "positive_C_count_ge": 2,
            "C015_delta_recall_ge": 0.0025,
            "C015_wins_gt_losses": True,
            "C015_worst_block_delta_ge": -0.005,
            "C015_multi_delta_ge": -0.005,
            "Q10_is_diagnostic_only": True,
        },
        "recommended": recommended,
    }

    report_path = out / "REPORT.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    promoted_path = out / "PROMOTED_ARM.json"
    if recommended is not None:
        promoted = {
            "schema": "manual.d1_consensus_uncertainty_arm.v1",
            "mode": recommended["mode"],
            "source_arm": recommended["source_arm"],
            "feature_dim": recommended["feature_dim"],
            "C": 0.15,
            "selective_quantile": (
                0.25
                if recommended["mode"] == "SELECTIVE_Q25"
                else None
            ),
            "source_report": str(report_path),
            "cal_metrics": recommended,
            "private_materialization_note": (
                "All needed inputs already exist in D1 private caches. "
                "No neural inference is required. If mode=SELECTIVE_Q25, "
                "compute the bottom 25% D1 rank5-rank6 margin on the 1,977 "
                "scored private queries (exclude the known 103 warmup overlaps), "
                "use the consensus arm only there, and preserve exact D1 "
                "predictions elsewhere. Submission still contains all 2,080 qids."
            ),
        }
        promoted_path.write_text(
            json.dumps(promoted, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    elif promoted_path.exists():
        promoted_path.unlink()

    print("=" * 120)
    print("VERDICT:", verdict)
    print(
        "RECOMMENDED:",
        recommended["name"] if recommended is not None else None,
    )
    if recommended is not None:
        cmp15 = recommended["C015"]["comparison_vs_sameC_D1"]
        print(
            f"C=.15 dR={cmp15['delta_recall']:+.10f} "
            f"W/L={cmp15['wins']}/{cmp15['losses']} "
            f"churn={cmp15['set_top5_churn']}"
        )
        print("Promoted arm:", promoted_path)
    print("Report:", report_path)
    print("=" * 120)


if __name__ == "__main__":
    main()
