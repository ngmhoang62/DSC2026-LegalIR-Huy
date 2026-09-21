#!/usr/bin/env python
"""
D1 NESTED LAMBDARANK / NONLINEAR SELECTOR AUDIT
===============================================

CPU-only. No private labels. No neural inference.

Hypothesis
----------
After many feature-level additions failed, the remaining selection headroom may
come from D1's *selector class/objective*:

    current D1:
      48D evidence -> StandardScaler -> pointwise LogisticRegression

This audit keeps EXACTLY the same:
  - candidate pool
  - 5 rank views
  - 10 score channels
  - 12 type/doctype features
  - 4 citation features

and changes only the selector to a shallow XGBRanker / LambdaRank model that can:
  - model nonlinear feature interactions;
  - optimize within-query ranking rather than independent binary relevance.

This is NOT the historical `burst_large_ltr` experiment. That model used an
older feature/retrieval world. Here XGBRanker consumes the authoritative D1 48D
matrix directly.

Scientific protocol
-------------------
4-block OUTER LOBO is the reported evaluation.

Inside each outer training world (3 blocks), a 3-fold INNER block-CV selects
hyperparameters. Therefore the outer held block never participates in:
  - XGB config selection;
  - RRF alpha/k selection;
  - family-specific model selection.

Families
--------
1) XGB_FULL
   Rank the entire D1 candidate pool by XGBRanker.

2) XGB_BOUNDARY_5_10
   Freeze exact D1 ranks 1..4.
   Let XGB choose slot #5 only from D1 ranks 5..10.
   This targets the known selection headroom while limiting churn.

3) LR_XGB_RRF
   Rank-fuse the exact D1 LR ranking and XGB ranking.
   alpha and RRF-k are selected ONLY inside outer-train inner CV.

XGB configs
-----------
Use the six shallow historical LambdaRank configurations as a *predeclared*
small model family, but train them on D1 48D:

  (depth=2, lr=.03, trees=180, pairs=10)
  (depth=2, lr=.05, trees=140, pairs=10)
  (depth=3, lr=.03, trees=200, pairs=10)
  (depth=3, lr=.05, trees=160, pairs=10)
  (depth=4, lr=.03, trees=180, pairs=10)
  (depth=3, lr=.03, trees=220, pairs=20)

Selection rule inside each outer fold
-------------------------------------
For each family, select the candidate maximizing:
    1. minimum inner-block Recall@5
    2. mean inner-block Recall@5
    3. mean inner-block Precision@5
    4. lower complexity tie-break

This intentionally prefers plateaus/generalization over an inner-CV spike.

Mandatory parity
----------------
Outer D1 baseline at C=.15 must reproduce:
    Recall@5 = 0.9569444444444444

Promotion gate
--------------
Nested OOF family must satisfy:
  - delta Recall >= +0.0025
  - wins > losses
  - worst outer-block Recall delta >= -0.005
  - multi-gold delta >= -0.005
  - positive/nonnegative behavior on at least 3/4 outer blocks

A pass is still followed by seed/config stability before private deployment.

Outputs
-------
results/manual/huy_d1_nested_lambdarank_v1/
  REPORT.json
  PROMOTED_FAMILY.json   (only if gate passes)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBRanker
except Exception as exc:
    raise RuntimeError(
        "xgboost is required. The repo already uses XGBRanker historically; "
        "install/activate the same environment. Original import error: "
        + repr(exc)
    )


EXPECTED_D1 = 0.9569444444444444
SEED = 2026
D1_VIEWS = ["base", "expanded", "jina", "dense", "corpus"]

XGB_CONFIGS = [
    {"depth": 2, "rate": .03, "trees": 180, "pairs": 10},
    {"depth": 2, "rate": .05, "trees": 140, "pairs": 10},
    {"depth": 3, "rate": .03, "trees": 200, "pairs": 10},
    {"depth": 3, "rate": .05, "trees": 160, "pairs": 10},
    {"depth": 4, "rate": .03, "trees": 180, "pairs": 10},
    {"depth": 3, "rate": .03, "trees": 220, "pairs": 20},
]

RRF_ALPHAS = [.25, .50, .75]  # alpha = XGB weight
RRF_KS = [0, 10, 40]

FAMILIES = [
    "XGB_FULL",
    "XGB_BOUNDARY_5_10",
    "LR_XGB_RRF",
]


def config_id(cfg):
    return (
        f"d{cfg['depth']}_lr{cfg['rate']:.2f}_"
        f"t{cfg['trees']}_p{cfg['pairs']}"
    )


def metrics(pred, gold, ids, blocks=None):
    per_r = {}
    per_p = {}
    for q in ids:
        docs = pred[q][:5]
        hit = len(set(docs) & set(gold[q]))
        per_r[q] = hit / max(1, len(gold[q]))
        per_p[q] = hit / 5.0

    out = {
        "recall": float(np.mean([per_r[q] for q in ids])),
        "precision": float(np.mean([per_p[q] for q in ids])),
        "single_recall": float(np.mean([
            per_r[q] for q in ids if len(gold[q]) == 1
        ])),
        "multi_recall": float(np.mean([
            per_r[q] for q in ids if len(gold[q]) > 1
        ])),
        "per_query_recall": per_r,
    }
    if blocks is not None:
        out["blocks"] = {
            b: float(np.mean([per_r[q] for q in blocks[b]]))
            for b in sorted(blocks)
            if all(q in per_r for q in blocks[b])
        }
    return out


def build_d1_rows(
    local_views,
    full_channels,
    extended,
    all_ids,
    type_rows,
    cite_rows,
):
    from tune_expanded_fusion_selection import ltr_features

    base_rows, groups = ltr_features(
        local_views,
        D1_VIEWS,
        extended,
        all_ids,
        full_channels,
    )
    rows = {
        q: np.concatenate(
            [base_rows[q], type_rows[q], cite_rows[q]],
            axis=1,
        ).astype(np.float32, copy=False)
        for q in all_ids
    }
    for q in all_ids:
        if rows[q].shape[1] != 48:
            raise RuntimeError(
                f"D1 feature parity failed q={q}: {rows[q].shape}"
            )
    return rows, groups


def xy_groups(train_ids, rows, groups, gold):
    X = np.vstack([rows[q] for q in train_ids])
    y = np.concatenate([
        [d in gold[q] for d in groups[q]]
        for q in train_ids
    ]).astype(np.int8)
    group_sizes = [len(groups[q]) for q in train_ids]
    return X, y, group_sizes


def fit_lr(train_ids, rows, groups, gold):
    X, y, _ = xy_groups(train_ids, rows, groups, gold)
    scaler = StandardScaler().fit(X)
    model = LogisticRegression(
        C=.15,
        class_weight="balanced",
        solver="liblinear",
        max_iter=3000,
        random_state=SEED,
    )
    model.fit(scaler.transform(X), y)
    return scaler, model


def score_lr(scaler, model, ids, rows, groups):
    pred = {}
    scores = {}
    full = {}
    for q in ids:
        s = np.asarray(
            model.decision_function(scaler.transform(rows[q])),
            dtype=np.float64,
        )
        order_idx = np.argsort(-s, kind="stable")
        order = [groups[q][i] for i in order_idx]
        full[q] = order
        pred[q] = order[:5]
        scores[q] = {
            groups[q][i]: float(s[i])
            for i in range(len(groups[q]))
        }
    return pred, full, scores


def make_xgb(cfg):
    # Parameters mirror the historical shallow LambdaRank family, but the
    # feature matrix is the CURRENT D1 48D world.
    return XGBRanker(
        objective="rank:ndcg",
        eval_metric="ndcg@5",
        tree_method="hist",
        n_estimators=int(cfg["trees"]),
        max_depth=int(cfg["depth"]),
        learning_rate=float(cfg["rate"]),
        min_child_weight=3,
        subsample=.85,
        colsample_bytree=.9,
        reg_lambda=5.0,
        n_jobs=2,
        lambdarank_pair_method="topk",
        lambdarank_num_pair_per_sample=int(cfg["pairs"]),
        random_state=SEED,
    )


def fit_xgb(cfg, train_ids, rows, groups, gold):
    X, y, group_sizes = xy_groups(train_ids, rows, groups, gold)
    model = make_xgb(cfg)
    model.fit(
        X,
        y,
        group=group_sizes,
        verbose=False,
    )
    return model


def score_xgb(model, ids, rows, groups):
    pred = {}
    full = {}
    scores = {}
    for q in ids:
        s = np.asarray(model.predict(rows[q]), dtype=np.float64)
        order_idx = np.argsort(-s, kind="stable")
        order = [groups[q][i] for i in order_idx]
        full[q] = order
        pred[q] = order[:5]
        scores[q] = {
            groups[q][i]: float(s[i])
            for i in range(len(groups[q]))
        }
    return pred, full, scores


def boundary_5_10(lr_full, xgb_scores, ids):
    out = {}
    for q in ids:
        order = lr_full[q]
        if len(order) < 5:
            raise RuntimeError(f"<5 candidates q={q}")
        if len(order) <= 5:
            out[q] = order[:5]
            continue
        pool = order[4:min(10, len(order))]
        chosen = max(
            pool,
            key=lambda d: (xgb_scores[q][d], -order.index(d)),
        )
        out[q] = order[:4] + [chosen]
    return out


def rrf_fuse(lr_full, xgb_full, ids, alpha, k):
    """
    alpha is XGB weight.
    All D1 candidates are present in both rankings.
    """
    out = {}
    for q in ids:
        r_lr = {d: i + 1 for i, d in enumerate(lr_full[q])}
        r_xg = {d: i + 1 for i, d in enumerate(xgb_full[q])}
        docs = lr_full[q]
        scored = []
        for d in docs:
            s = (
                (1.0 - alpha) / (k + r_lr[d])
                + alpha / (k + r_xg[d])
            )
            scored.append((float(s), d))
        scored.sort(key=lambda x: (-x[0], str(x[1])))
        out[q] = [d for _, d in scored[:5]]
    return out


def compare(base, cand, gold, ids):
    bm = metrics(base, gold, ids)
    cm = metrics(cand, gold, ids)

    wins = losses = set_churn = 0
    for q in ids:
        a = bm["per_query_recall"][q]
        b = cm["per_query_recall"][q]
        wins += int(b > a + 1e-12)
        losses += int(b < a - 1e-12)
        set_churn += int(set(base[q]) != set(cand[q]))

    return {
        "base_recall": bm["recall"],
        "candidate_recall": cm["recall"],
        "delta_recall": cm["recall"] - bm["recall"],
        "base_precision": bm["precision"],
        "candidate_precision": cm["precision"],
        "delta_precision": cm["precision"] - bm["precision"],
        "delta_single": cm["single_recall"] - bm["single_recall"],
        "delta_multi": cm["multi_recall"] - bm["multi_recall"],
        "wins": int(wins),
        "losses": int(losses),
        "net_wins": int(wins - losses),
        "set_top5_churn": int(set_churn),
    }


def candidate_complexity_key(spec):
    cfg = spec["config"]
    # Lower is simpler / more conservative.
    fam_order = {
        "XGB_BOUNDARY_5_10": 0,
        "LR_XGB_RRF": 1,
        "XGB_FULL": 2,
    }
    return (
        fam_order[spec["family"]],
        cfg["depth"],
        cfg["trees"],
        cfg["pairs"],
        cfg["rate"],
        spec.get("alpha", 0.0),
        spec.get("rrf_k", 0),
    )


def summarize_inner(rows):
    recalls = [x["recall"] for x in rows]
    precs = [x["precision"] for x in rows]
    return {
        "min_recall": float(min(recalls)),
        "mean_recall": float(np.mean(recalls)),
        "std_recall": float(np.std(recalls)),
        "mean_precision": float(np.mean(precs)),
        "fold_recalls": [float(x) for x in recalls],
        "fold_precisions": [float(x) for x in precs],
    }


def select_best(candidates):
    """
    Robust selection:
      min inner-block Recall first,
      then mean Recall,
      then precision,
      then simpler spec.
    """
    return max(
        candidates,
        key=lambda x: (
            x["inner"]["min_recall"],
            x["inner"]["mean_recall"],
            x["inner"]["mean_precision"],
            tuple(-v if isinstance(v, (int, float)) else 0
                  for v in candidate_complexity_key(x)),
        ),
    )


def compact_metrics(m):
    return {
        k: v for k, v in m.items()
        if k != "per_query_recall"
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=Path, required=True)
    args = ap.parse_args()

    root = args.repo_root.expanduser().resolve()
    sys.path.insert(0, str(root))

    print("[1/7] Loading authoritative D1 CAL600 world...", flush=True)
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

    block_names = sorted(blocks)
    if len(block_names) != 4:
        raise RuntimeError(f"Expected 4 CAL blocks, got {block_names}")

    print("[2/7] Building exact 48D D1 feature matrix...", flush=True)
    rows, groups = build_d1_rows(
        local_views,
        full_channels,
        extended,
        all_ids,
        type_rows,
        cite_rows,
    )
    print(
        f"  queries={len(all_ids)} "
        f"candidate_rows={sum(len(groups[q]) for q in all_ids):,} "
        f"dim=48",
        flush=True,
    )

    # Outer OOF accumulators.
    base_pred_all = {}
    family_pred_all = {f: {} for f in FAMILIES}
    fold_reports = {}

    # For a rough stability picture.
    selected_specs = defaultdict(list)

    print("[3/7] Nested 4-block outer LOBO...", flush=True)
    t_all = time.perf_counter()

    for outer_held in block_names:
        print("\n" + "=" * 112, flush=True)
        print(f"OUTER HELD = {outer_held}", flush=True)

        outer_train_blocks = [
            b for b in block_names if b != outer_held
        ]
        outer_train_ids = [
            q for b in outer_train_blocks for q in blocks[b]
        ]
        outer_test_ids = list(blocks[outer_held])

        # True outer D1 baseline.
        lr_scaler, lr_model = fit_lr(
            outer_train_ids, rows, groups, gold
        )
        (
            outer_lr_pred,
            outer_lr_full,
            outer_lr_scores,
        ) = score_lr(
            lr_scaler, lr_model, outer_test_ids, rows, groups
        )
        base_pred_all.update(outer_lr_pred)

        outer_base_m = metrics(
            outer_lr_pred, gold, outer_test_ids
        )
        print(
            f"  D1 outer baseline R={outer_base_m['recall']:.10f} "
            f"P={outer_base_m['precision']:.10f}",
            flush=True,
        )

        # inner_results[key] = list of fold metric dicts
        # Keys are serialized spec tuples.
        inner_results = defaultdict(list)
        spec_lookup = {}

        for inner_held in outer_train_blocks:
            inner_train_ids = [
                q
                for b in outer_train_blocks
                if b != inner_held
                for q in blocks[b]
            ]
            inner_test_ids = list(blocks[inner_held])

            # LR once for this inner fold.
            in_scaler, in_lr_model = fit_lr(
                inner_train_ids, rows, groups, gold
            )
            (
                in_lr_pred,
                in_lr_full,
                _in_lr_scores,
            ) = score_lr(
                in_scaler,
                in_lr_model,
                inner_test_ids,
                rows,
                groups,
            )

            for cfg in XGB_CONFIGS:
                cid = config_id(cfg)
                xgb = fit_xgb(
                    cfg, inner_train_ids, rows, groups, gold
                )
                (
                    xgb_pred,
                    xgb_full,
                    xgb_scores,
                ) = score_xgb(
                    xgb,
                    inner_test_ids,
                    rows,
                    groups,
                )

                # 1) Full.
                spec = {
                    "family": "XGB_FULL",
                    "config": dict(cfg),
                }
                key = ("XGB_FULL", cid)
                spec_lookup[key] = spec
                inner_results[key].append(
                    compact_metrics(metrics(
                        xgb_pred, gold, inner_test_ids
                    ))
                )

                # 2) Boundary.
                bpred = boundary_5_10(
                    in_lr_full,
                    xgb_scores,
                    inner_test_ids,
                )
                spec = {
                    "family": "XGB_BOUNDARY_5_10",
                    "config": dict(cfg),
                }
                key = ("XGB_BOUNDARY_5_10", cid)
                spec_lookup[key] = spec
                inner_results[key].append(
                    compact_metrics(metrics(
                        bpred, gold, inner_test_ids
                    ))
                )

                # 3) LR-XGB RRF family.
                for alpha in RRF_ALPHAS:
                    for k in RRF_KS:
                        rpred = rrf_fuse(
                            in_lr_full,
                            xgb_full,
                            inner_test_ids,
                            alpha,
                            k,
                        )
                        spec = {
                            "family": "LR_XGB_RRF",
                            "config": dict(cfg),
                            "alpha": float(alpha),
                            "rrf_k": int(k),
                        }
                        key = (
                            "LR_XGB_RRF",
                            cid,
                            float(alpha),
                            int(k),
                        )
                        spec_lookup[key] = spec
                        inner_results[key].append(
                            compact_metrics(metrics(
                                rpred, gold, inner_test_ids
                            ))
                        )

            print(
                f"  inner held={inner_held}: "
                f"evaluated {len(XGB_CONFIGS)} XGB configs",
                flush=True,
            )

        # Assemble candidates with 3 inner folds and select separately by family.
        family_candidates = {f: [] for f in FAMILIES}
        for key, fold_ms in inner_results.items():
            if len(fold_ms) != len(outer_train_blocks):
                raise RuntimeError(
                    f"Incomplete inner CV for {key}: {len(fold_ms)}"
                )
            spec = dict(spec_lookup[key])
            spec["inner"] = summarize_inner(fold_ms)
            family_candidates[spec["family"]].append(spec)

        selected = {
            family: select_best(family_candidates[family])
            for family in FAMILIES
        }

        for family in FAMILIES:
            s = selected[family]
            selected_specs[family].append(s)
            print(
                f"  SELECT {family:18s} "
                f"{config_id(s['config'])} "
                + (
                    f"alpha={s['alpha']:.2f} k={s['rrf_k']} "
                    if family == "LR_XGB_RRF" else ""
                )
                + (
                    f"| inner minR={s['inner']['min_recall']:.6f} "
                    f"meanR={s['inner']['mean_recall']:.6f} "
                    f"std={s['inner']['std_recall']:.6f}"
                ),
                flush=True,
            )

        # Fit only unique selected XGB configs on all 3 outer-train blocks.
        model_cache = {}
        output_cache = {}
        for family, spec in selected.items():
            cid = config_id(spec["config"])
            if cid not in model_cache:
                mdl = fit_xgb(
                    spec["config"],
                    outer_train_ids,
                    rows,
                    groups,
                    gold,
                )
                model_cache[cid] = mdl
                output_cache[cid] = score_xgb(
                    mdl,
                    outer_test_ids,
                    rows,
                    groups,
                )

        outer_family = {}

        for family, spec in selected.items():
            cid = config_id(spec["config"])
            xgb_pred, xgb_full, xgb_scores = output_cache[cid]

            if family == "XGB_FULL":
                pred = xgb_pred
            elif family == "XGB_BOUNDARY_5_10":
                pred = boundary_5_10(
                    outer_lr_full,
                    xgb_scores,
                    outer_test_ids,
                )
            elif family == "LR_XGB_RRF":
                pred = rrf_fuse(
                    outer_lr_full,
                    xgb_full,
                    outer_test_ids,
                    spec["alpha"],
                    spec["rrf_k"],
                )
            else:
                raise ValueError(family)

            outer_family[family] = pred
            family_pred_all[family].update(pred)

            cmp = compare(
                outer_lr_pred,
                pred,
                gold,
                outer_test_ids,
            )
            print(
                f"  OUTER {family:18s} "
                f"R={cmp['candidate_recall']:.10f} "
                f"dR={cmp['delta_recall']:+.10f} "
                f"W/L={cmp['wins']}/{cmp['losses']} "
                f"churn={cmp['set_top5_churn']}",
                flush=True,
            )

        fold_reports[outer_held] = {
            "train_blocks": outer_train_blocks,
            "test_queries": len(outer_test_ids),
            "baseline": compact_metrics(outer_base_m),
            "selected": selected,
            "family_comparisons": {
                f: compare(
                    outer_lr_pred,
                    outer_family[f],
                    gold,
                    outer_test_ids,
                )
                for f in FAMILIES
            },
        }

    print("\n[4/7] Mandatory D1 parity...", flush=True)
    base_m = metrics(
        base_pred_all,
        gold,
        all_ids,
        blocks=blocks,
    )
    print(
        f"  D1 nested outer OOF "
        f"R={base_m['recall']:.10f} "
        f"P={base_m['precision']:.10f} "
        f"single={base_m['single_recall']:.6f} "
        f"multi={base_m['multi_recall']:.6f}",
        flush=True,
    )
    if abs(base_m["recall"] - EXPECTED_D1) > 1e-9:
        raise RuntimeError(
            f"D1 parity failed: {base_m['recall']} != {EXPECTED_D1}"
        )

    print("[5/7] Aggregate nested nonlinear families...", flush=True)
    family_summary = {}

    for family in FAMILIES:
        pred = family_pred_all[family]
        fm = metrics(pred, gold, all_ids, blocks=blocks)
        cmp = compare(
            base_pred_all,
            pred,
            gold,
            all_ids,
        )
        block_delta = {
            b: fm["blocks"][b] - base_m["blocks"][b]
            for b in block_names
        }
        nonnegative_blocks = sum(
            v >= -1e-12 for v in block_delta.values()
        )

        gates = {
            "delta_recall_ge_0_0025": (
                cmp["delta_recall"] >= 0.0025 - 1e-12
            ),
            "wins_gt_losses": (
                cmp["wins"] > cmp["losses"]
            ),
            "worst_block_delta_ge_neg_0_005": (
                min(block_delta.values())
                >= -0.005 - 1e-12
            ),
            "multi_delta_ge_neg_0_005": (
                cmp["delta_multi"] >= -0.005 - 1e-12
            ),
            "nonnegative_blocks_ge_3": (
                nonnegative_blocks >= 3
            ),
        }

        family_summary[family] = {
            "metrics": compact_metrics(fm),
            "comparison_vs_D1": cmp,
            "block_delta": block_delta,
            "nonnegative_blocks": int(nonnegative_blocks),
            "gates": gates,
            "pass": all(gates.values()),
        }

        print(
            f"  {family:18s} "
            f"R={fm['recall']:.10f} "
            f"dR={cmp['delta_recall']:+.10f} "
            f"W/L={cmp['wins']}/{cmp['losses']} "
            f"singleΔ={cmp['delta_single']:+.6f} "
            f"multiΔ={cmp['delta_multi']:+.6f} "
            f"blocks="
            + ",".join(
                f"{b}:{block_delta[b]:+.4f}"
                for b in block_names
            )
            + f" => {'PASS' if all(gates.values()) else 'FAIL'}",
            flush=True,
        )

    print("[6/7] Hyperparameter-selection stability...", flush=True)
    stability = {}

    for family in FAMILIES:
        specs = selected_specs[family]
        cfg_counts = Counter(config_id(s["config"]) for s in specs)

        row = {
            "selected_config_counts": dict(cfg_counts),
            "unique_configs": len(cfg_counts),
        }
        if family == "LR_XGB_RRF":
            row["alpha_counts"] = dict(Counter(
                str(s["alpha"]) for s in specs
            ))
            row["rrf_k_counts"] = dict(Counter(
                str(s["rrf_k"]) for s in specs
            ))
        stability[family] = row
        print(
            f"  {family:18s} configs={dict(cfg_counts)}"
            + (
                f" alpha={row.get('alpha_counts')} k={row.get('rrf_k_counts')}"
                if family == "LR_XGB_RRF" else ""
            ),
            flush=True,
        )

    passing = [
        f for f in FAMILIES
        if family_summary[f]["pass"]
    ]
    recommended = None
    if passing:
        # Favor higher nested OOF recall; near ties prefer lower churn,
        # then more conservative family.
        conservative = {
            "XGB_BOUNDARY_5_10": 2,
            "LR_XGB_RRF": 1,
            "XGB_FULL": 0,
        }
        best_r = max(
            family_summary[f]["metrics"]["recall"]
            for f in passing
        )
        near = [
            f for f in passing
            if family_summary[f]["metrics"]["recall"]
            >= best_r - 0.0005
        ]
        recommended = max(
            near,
            key=lambda f: (
                conservative[f],
                -family_summary[f]["comparison_vs_D1"]["set_top5_churn"],
            ),
        )

    verdict = (
        "PROMOTE_NESTED_LAMBDARANK_FAMILY"
        if recommended is not None
        else "KILL_D1_NESTED_LAMBDARANK"
    )

    print("[7/7] Writing report...", flush=True)
    out = root / "results/manual/huy_d1_nested_lambdarank_v1"
    out.mkdir(parents=True, exist_ok=True)

    report = {
        "schema": "manual.d1_nested_lambdarank_v1",
        "status": verdict,
        "protocol": {
            "outer": "4-block LOBO",
            "inner": "3-block CV inside each outer-training world",
            "features": "authoritative D1 48D only",
            "candidate_pool": "authoritative D1 candidate pool",
            "private_labels": False,
            "gpu": False,
            "selector_baseline": "StandardScaler + LogisticRegression C=.15",
            "xgb_objective": "rank:ndcg / ndcg@5",
            "inner_selection": (
                "maximize min inner-block Recall@5, then mean Recall@5, "
                "then mean Precision@5, then lower complexity"
            ),
        },
        "xgb_configs": XGB_CONFIGS,
        "rrf_grid": {
            "alphas_xgb_weight": RRF_ALPHAS,
            "k": RRF_KS,
        },
        "baseline_D1": compact_metrics(base_m),
        "folds": fold_reports,
        "families": family_summary,
        "selection_stability": stability,
        "promotion_gate": {
            "delta_recall_ge": 0.0025,
            "wins_gt_losses": True,
            "worst_outer_block_delta_ge": -0.005,
            "multi_delta_ge": -0.005,
            "nonnegative_outer_blocks_ge": 3,
            "note": (
                "A pass should still undergo seed/config stability before "
                "private deployment."
            ),
        },
        "recommended_family": recommended,
        "elapsed_seconds": time.perf_counter() - t_all,
    }

    report_path = out / "REPORT.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    promoted_path = out / "PROMOTED_FAMILY.json"
    if recommended is not None:
        promoted = {
            "schema": "manual.d1_nested_lambdarank_family.v1",
            "family": recommended,
            "nested_oof": family_summary[recommended],
            "selection_stability": stability[recommended],
            "source_report": str(report_path),
            "next_step": (
                "Run seed/config stability, then derive one final full-CAL600 "
                "spec and train it on all CAL600 for private inference. "
                "No neural inference is needed."
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
    print("RECOMMENDED FAMILY:", recommended)
    print(f"Elapsed: {report['elapsed_seconds']/60:.1f} min")
    print("Report:", report_path)
    if recommended is not None:
        print("Promoted:", promoted_path)
    print("=" * 120)


if __name__ == "__main__":
    main()
