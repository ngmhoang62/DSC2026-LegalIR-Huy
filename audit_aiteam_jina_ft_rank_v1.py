#!/usr/bin/env python
"""
AUDIT AITEAMVN-FT / JINA-FT RANK TRANSPLANT INTO D1
====================================================

CPU-only. No private labels. No neural inference.

Motivation
----------
Authoritative D1 already contains `aiteamvn_ft` and `jina_ft` as SCORE channels,
but NOT as rank views.

D1 rank views:
    base, expanded, jina, dense, corpus

D1 score channels include:
    ..., aiteamvn_ft, jina_ft, ...

This audit asks whether the ORDERING learned by the fine-tuned models contains
useful information that the score-only D1 representation is currently throwing
away.

Families
--------
A) Rank augmentation
   ADD_AITEAM_FT_RANK
   ADD_JINA_FT_RANK
   ADD_BOTH_FT_RANKS

B) Rank replacement
   REPLACE_DENSE_WITH_AITEAM_FT
   REPLACE_JINA_WITH_JINA_FT
   REPLACE_BOTH_FT_RANKS

C) Fine-tuning residual
   RESIDUAL_AITEAM
   RESIDUAL_JINA
   RESIDUAL_BOTH

Residual features explicitly represent "what fine-tuning changed":

    rank_shift = (old_rank - ft_rank) / 60
    reciprocal_shift =
        1/(10 + ft_rank) - 1/(10 + old_rank)

Positive values mean the fine-tuned model promotes the candidate relative to
the old D1 rank view.

Robustness
----------
Every arm is compared against D1 at SAME LogisticRegression C:

    C in {0.10, 0.15, 0.20}

C=.15 MUST reproduce authoritative D1 Recall@5 exactly:
    0.9569444444444444

Because the final FT artifacts have provenance/leakage risk, promotion is
deliberately strict:
  - positive median delta across C;
  - positive delta at >=2/3 C values;
  - C=.15 delta Recall >= +0.0025;
  - C=.15 wins > losses;
  - C=.15 worst block delta >= -0.005;
  - C=.15 multi-gold delta >= -0.005.

Outputs
-------
results/manual/huy_aiteam_jina_ft_rank_gate_v1/
  REPORT.json
  PROMOTED_ARM.json       (only if an arm passes)
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
    "ADD_AITEAM_FT_RANK",
    "ADD_JINA_FT_RANK",
    "ADD_BOTH_FT_RANKS",
    "REPLACE_DENSE_WITH_AITEAM_FT",
    "REPLACE_JINA_WITH_JINA_FT",
    "REPLACE_BOTH_FT_RANKS",
    "RESIDUAL_AITEAM",
    "RESIDUAL_JINA",
    "RESIDUAL_BOTH",
]


def build_ft_rank_views(extended, full_channels, all_ids):
    required = ["aiteamvn_ft", "jina_ft"]
    missing = [k for k in required if k not in full_channels]
    if missing:
        raise RuntimeError(f"Missing FT score channels: {missing}")

    out = {}
    for channel, view_name in [
        ("aiteamvn_ft", "aiteam_ft_rank"),
        ("jina_ft", "jina_ft_rank"),
    ]:
        table = full_channels[channel]
        out[view_name] = {
            q: sorted(
                extended[q],
                key=lambda d: (-float(table[q][d]), str(d)),
            )
            for q in all_ids
        }
    return out


def rank_map(view, q):
    return {d: i + 1 for i, d in enumerate(view[q])}


def residual_features(
    *,
    extended,
    ids,
    base_views,
    ft_views,
    use_aiteam,
    use_jina,
):
    """
    Returns q -> [n_candidates, residual_dim].

    AITeam residual compares:
        old D1 `dense` rank view vs FT `aiteam_ft_rank`

    Jina residual compares:
        old D1 `jina` rank view vs FT `jina_ft_rank`
    """
    rows = {}

    for q in ids:
        docs = extended[q]
        cols = []

        if use_aiteam:
            old = rank_map(base_views["dense"], q)
            new = rank_map(ft_views["aiteam_ft_rank"], q)

            shift = []
            recip = []
            for d in docs:
                ro = int(old.get(d, 60))
                rn = int(new.get(d, 60))
                shift.append((ro - rn) / 60.0)
                recip.append(
                    1.0 / (10.0 + rn)
                    - 1.0 / (10.0 + ro)
                )
            cols += [shift, recip]

        if use_jina:
            old = rank_map(base_views["jina"], q)
            new = rank_map(ft_views["jina_ft_rank"], q)

            shift = []
            recip = []
            for d in docs:
                ro = int(old.get(d, 60))
                rn = int(new.get(d, 60))
                shift.append((ro - rn) / 60.0)
                recip.append(
                    1.0 / (10.0 + rn)
                    - 1.0 / (10.0 + ro)
                )
            cols += [shift, recip]

        if cols:
            rows[q] = np.asarray(cols, dtype=np.float32).T
        else:
            rows[q] = np.zeros((len(docs), 0), dtype=np.float32)

    return rows


def arm_contract(arm, base_views, ft_views):
    views = dict(base_views)
    names = list(BASE_VIEWS)
    residual = (False, False)

    if arm == "D1":
        expected_dim = 48

    elif arm == "ADD_AITEAM_FT_RANK":
        views["aiteam_ft_rank"] = ft_views["aiteam_ft_rank"]
        names.append("aiteam_ft_rank")
        expected_dim = 50

    elif arm == "ADD_JINA_FT_RANK":
        views["jina_ft_rank"] = ft_views["jina_ft_rank"]
        names.append("jina_ft_rank")
        expected_dim = 50

    elif arm == "ADD_BOTH_FT_RANKS":
        views["aiteam_ft_rank"] = ft_views["aiteam_ft_rank"]
        views["jina_ft_rank"] = ft_views["jina_ft_rank"]
        names += ["aiteam_ft_rank", "jina_ft_rank"]
        expected_dim = 52

    elif arm == "REPLACE_DENSE_WITH_AITEAM_FT":
        views["aiteam_ft_rank"] = ft_views["aiteam_ft_rank"]
        names = [
            "aiteam_ft_rank" if x == "dense" else x
            for x in BASE_VIEWS
        ]
        expected_dim = 48

    elif arm == "REPLACE_JINA_WITH_JINA_FT":
        views["jina_ft_rank"] = ft_views["jina_ft_rank"]
        names = [
            "jina_ft_rank" if x == "jina" else x
            for x in BASE_VIEWS
        ]
        expected_dim = 48

    elif arm == "REPLACE_BOTH_FT_RANKS":
        views["aiteam_ft_rank"] = ft_views["aiteam_ft_rank"]
        views["jina_ft_rank"] = ft_views["jina_ft_rank"]
        names = [
            (
                "aiteam_ft_rank" if x == "dense"
                else "jina_ft_rank" if x == "jina"
                else x
            )
            for x in BASE_VIEWS
        ]
        expected_dim = 48

    elif arm == "RESIDUAL_AITEAM":
        residual = (True, False)
        expected_dim = 50

    elif arm == "RESIDUAL_JINA":
        residual = (False, True)
        expected_dim = 50

    elif arm == "RESIDUAL_BOTH":
        residual = (True, True)
        expected_dim = 52

    else:
        raise ValueError(arm)

    return views, names, residual, expected_dim


def build_rows(
    *,
    arm,
    ids,
    extended,
    base_views,
    ft_views,
    full_channels,
    type_rows,
    cite_rows,
):
    from tune_expanded_fusion_selection import ltr_features

    views, view_names, residual_spec, expected_dim = arm_contract(
        arm,
        base_views,
        ft_views,
    )

    rows, groups = ltr_features(
        views,
        view_names,
        extended,
        ids,
        full_channels,
    )

    use_aiteam, use_jina = residual_spec
    residual = None
    if use_aiteam or use_jina:
        residual = residual_features(
            extended=extended,
            ids=ids,
            base_views=base_views,
            ft_views=ft_views,
            use_aiteam=use_aiteam,
            use_jina=use_jina,
        )

    for q in ids:
        pieces = [rows[q], type_rows[q], cite_rows[q]]
        if residual is not None:
            pieces.append(residual[q])
        rows[q] = np.concatenate(pieces, axis=1)

        got = int(rows[q].shape[1])
        if got != expected_dim:
            raise RuntimeError(
                f"{arm}: feature dim {got} != expected {expected_dim}"
            )

    return rows, groups, view_names, expected_dim


def run_lobo(
    *,
    arm,
    C,
    queries,
    blocks,
    all_ids,
    extended,
    base_views,
    ft_views,
    full_channels,
    gold,
    type_rows,
    cite_rows,
):
    rows, groups, view_names, feature_dim = build_rows(
        arm=arm,
        ids=all_ids,
        extended=extended,
        base_views=base_views,
        ft_views=ft_views,
        full_channels=full_channels,
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
        "recall": float(np.mean([per_r[q] for q in all_ids])),
        "precision": float(np.mean([per_p[q] for q in all_ids])),
        "single_recall": float(np.mean([
            per_r[q] for q in all_ids if len(gold[q]) == 1
        ])),
        "multi_recall": float(np.mean([
            per_r[q] for q in all_ids if len(gold[q]) > 1
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


def rank_similarity_diagnostics(
    *,
    all_ids,
    extended,
    base_views,
    ft_views,
):
    def stats(old_name, new_name):
        same_top5 = []
        top5_jacc = []
        mean_abs_rank_shift = []
        promoted_5plus = []
        demoted_5plus = []

        for q in all_ids:
            old = rank_map(base_views[old_name], q)
            new = rank_map(ft_views[new_name], q)
            docs = extended[q]

            old_order = sorted(
                docs,
                key=lambda d: (old.get(d, 60), str(d)),
            )
            new_order = sorted(
                docs,
                key=lambda d: (new.get(d, 60), str(d)),
            )

            o5 = old_order[:5]
            n5 = new_order[:5]
            same_top5.append(set(o5) == set(n5))
            top5_jacc.append(
                len(set(o5) & set(n5))
                / len(set(o5) | set(n5))
            )

            shifts = [
                abs(old.get(d, 60) - new.get(d, 60))
                for d in docs
            ]
            mean_abs_rank_shift.append(float(np.mean(shifts)))
            promoted_5plus.append(sum(
                1 for d in docs
                if old.get(d, 60) > 5 and new.get(d, 60) <= 5
            ))
            demoted_5plus.append(sum(
                1 for d in docs
                if old.get(d, 60) <= 5 and new.get(d, 60) > 5
            ))

        return {
            "queries": len(all_ids),
            "same_top5_set_fraction": float(np.mean(same_top5)),
            "mean_top5_jaccard": float(np.mean(top5_jacc)),
            "mean_abs_rank_shift": float(np.mean(mean_abs_rank_shift)),
            "mean_promotions_into_top5_per_query": float(np.mean(promoted_5plus)),
            "mean_demotions_out_of_top5_per_query": float(np.mean(demoted_5plus)),
        }

    return {
        "dense_vs_aiteam_ft": stats("dense", "aiteam_ft_rank"),
        "jina_vs_jina_ft": stats("jina", "jina_ft_rank"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=Path, required=True)
    args = ap.parse_args()

    root = args.repo_root.expanduser().resolve()
    sys.path.insert(0, str(root))

    print("[1/6] Loading authoritative D1 CAL600 bundle...", flush=True)
    from src.gemini.huy_vnlegal_rank_ablation_v1.evaluate_ablation_cal import (
        load_cal_inputs,
    )
    (
        queries,
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

    for ch in ("aiteamvn_ft", "jina_ft"):
        if ch not in full_channels:
            raise RuntimeError(f"Missing D1 channel: {ch}")

    print("[2/6] Building FT rank views from existing D1 score caches...", flush=True)
    ft_views = build_ft_rank_views(
        extended,
        full_channels,
        all_ids,
    )

    diag = rank_similarity_diagnostics(
        all_ids=all_ids,
        extended=extended,
        base_views=local_views,
        ft_views=ft_views,
    )
    for name, d in diag.items():
        print(
            f"  {name}: sameTop5={d['same_top5_set_fraction']:.1%} "
            f"Jacc={d['mean_top5_jaccard']:.3f} "
            f"mean|Δrank|={d['mean_abs_rank_shift']:.2f}",
            flush=True,
        )

    print("[3/6] Running D1 C-sensitivity controls...", flush=True)
    results = {}
    for c in C_GRID:
        r = run_lobo(
            arm="D1",
            C=c,
            queries=queries,
            blocks=blocks,
            all_ids=all_ids,
            extended=extended,
            base_views=local_views,
            ft_views=ft_views,
            full_channels=full_channels,
            gold=gold,
            type_rows=type_rows,
            cite_rows=cite_rows,
        )
        results[("D1", c)] = r
        print(
            f"  D1 C={c:.2f} "
            f"R={r['recall']:.10f} "
            f"P={r['precision']:.10f} "
            f"multi={r['multi_recall']:.6f}",
            flush=True,
        )

    if abs(
        results[("D1", 0.15)]["recall"]
        - EXPECTED_D1_C015
    ) > 1e-9:
        raise RuntimeError(
            "D1 parity failed: "
            f"{results[('D1', 0.15)]['recall']} "
            f"!= {EXPECTED_D1_C015}"
        )

    print("[4/6] Running FT rank/residual arms...", flush=True)
    comparisons = {}

    for arm in ARMS[1:]:
        print(f"\n--- {arm} ---", flush=True)
        deltas = []

        for c in C_GRID:
            r = run_lobo(
                arm=arm,
                C=c,
                queries=queries,
                blocks=blocks,
                all_ids=all_ids,
                extended=extended,
                base_views=local_views,
                ft_views=ft_views,
                full_channels=full_channels,
                gold=gold,
                type_rows=type_rows,
                cite_rows=cite_rows,
            )
            results[(arm, c)] = r
            cmp = compare(results[("D1", c)], r, all_ids)
            comparisons[(arm, c)] = cmp
            deltas.append(cmp["delta_recall"])

            print(
                f"  C={c:.2f} dim={r['feature_dim']:2d} "
                f"R={r['recall']:.10f} "
                f"dR={cmp['delta_recall']:+.10f} "
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
            f"range=[{min(deltas):+.10f},{max(deltas):+.10f}]",
            flush=True,
        )

    print("\n[5/6] Leakage-aware promotion gates...", flush=True)
    summaries = {}
    eligible = []

    for arm in ARMS[1:]:
        deltas = [
            comparisons[(arm, c)]["delta_recall"]
            for c in C_GRID
        ]
        cmp15 = comparisons[(arm, 0.15)]
        res15 = results[(arm, 0.15)]

        summary = {
            "arm": arm,
            "family": (
                "augmentation" if arm.startswith("ADD_")
                else "replacement" if arm.startswith("REPLACE_")
                else "residual"
            ),
            "feature_dim": res15["feature_dim"],
            "rank_views_C015": res15["rank_views"],
            "median_delta_across_C": float(np.median(deltas)),
            "min_delta_across_C": float(min(deltas)),
            "max_delta_across_C": float(max(deltas)),
            "positive_C_count": int(
                sum(x > 1e-12 for x in deltas)
            ),
            "deltas_by_C": {
                f"{c:.2f}": comparisons[(arm, c)]["delta_recall"]
                for c in C_GRID
            },
            "C015": {
                **compact(res15),
                "comparison_vs_sameC_D1": cmp15,
            },
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
        summaries[arm] = summary

        if summary["all_gates_pass"]:
            eligible.append(summary)

        print(
            f"  {arm:31s} "
            f"median={summary['median_delta_across_C']:+.6f} "
            f"C015={cmp15['delta_recall']:+.6f} "
            f"W/L={cmp15['wins']}/{cmp15['losses']} "
            f"positiveC={summary['positive_C_count']}/3 "
            f"=> {'PASS' if summary['all_gates_pass'] else 'FAIL'}",
            flush=True,
        )

    recommended = None
    if eligible:
        recommended = max(
            eligible,
            key=lambda s: (
                s["median_delta_across_C"],
                s["C015"]["comparison_vs_sameC_D1"]["delta_recall"],
                s["C015"]["comparison_vs_sameC_D1"]["net_wins"],
                -s["C015"]["comparison_vs_sameC_D1"]["set_top5_churn"],
            ),
        )

    verdict = (
        "PROMOTE_AITEAM_JINA_FT_RANK_ARM"
        if recommended is not None
        else "KILL_AITEAM_JINA_FT_RANK_ARMS"
    )

    print("[6/6] Writing report...", flush=True)
    out = root / "results/manual/huy_aiteam_jina_ft_rank_gate_v1"
    out.mkdir(parents=True, exist_ok=True)

    report = {
        "schema": "manual.aiteam_jina_ft_rank_gate_v1",
        "status": verdict,
        "scientific_warning": (
            "The FT artifacts have training/provenance contamination risk. "
            "Treat this as a mechanism/stability gate, not a clean "
            "generalization estimate."
        ),
        "hypothesis": (
            "D1 currently uses AITeamVN-FT and Jina-FT only as score "
            "channels. Test whether their ordinal ranking, replacement of "
            "their corresponding old rank views, or explicit fine-tuning "
            "rank residual adds marginal utility."
        ),
        "rank_similarity_diagnostics": diag,
        "C_grid": C_GRID,
        "baseline_D1": {
            f"{c:.2f}": compact(results[("D1", c)])
            for c in C_GRID
        },
        "arms": summaries,
        "promotion_gate": {
            "median_delta_across_C_positive": True,
            "positive_C_count_ge": 2,
            "C015_delta_recall_ge": 0.0025,
            "C015_wins_gt_losses": True,
            "C015_worst_block_delta_ge": -0.005,
            "C015_multi_delta_ge": -0.005,
        },
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
            "schema": "manual.aiteam_jina_ft_rank_arm.v1",
            "arm": recommended["arm"],
            "family": recommended["family"],
            "feature_dim": recommended["feature_dim"],
            "C": 0.15,
            "source_report": str(report_path),
            "cal_contaminated_metrics": recommended,
            "private_deployment_note": (
                "No new model inference is required if private D1 caches "
                "aiteamvn_ft_scores.pkl and jina_ft_scores.pkl are present. "
                "Materialize with the exact same feature transform and train "
                "the selected full CAL600 LR at C=.15."
            ),
        }
        promoted_path.write_text(
            json.dumps(promoted, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    elif promoted_path.exists():
        promoted_path.unlink()

    print("=" * 116)
    print("VERDICT:", verdict)
    print(
        "RECOMMENDED ARM:",
        recommended["arm"] if recommended is not None else None,
    )
    if recommended is not None:
        c = recommended["C015"]["comparison_vs_sameC_D1"]
        print(
            f"C=.15 dR={c['delta_recall']:+.10f} "
            f"W/L={c['wins']}/{c['losses']} "
            f"churn={c['set_top5_churn']}"
        )
        print("Promoted arm:", promoted_path)
    print("Report:", report_path)
    print("=" * 116)


if __name__ == "__main__":
    main()
