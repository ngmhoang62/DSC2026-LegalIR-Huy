#!/usr/bin/env python
"""
AUDIT HARRIER-FT RANK SPECIALIST AGAINST D1 ON CAL600
=====================================================

CPU-only. No neural inference.

Purpose
-------
Use the saved full-corpus Harrier fine-tuned retrieval artifact
`train_finetuned_best.json` only as a rank signal, then test a deliberately
low-capacity one-swap boundary policy on the authoritative D1 LOBO ranking.

IMPORTANT:
The final Harrier checkpoint was trained on a pool that includes the CAL600
queries. Therefore these results are NOT a clean generalization estimate.
The audit is only a mechanism/stability gate before spending hours on private
full-corpus inference.

Policy family
-------------
For each query:
  1. keep D1 ranks 1..4 fixed;
  2. current defender = D1 rank 5;
  3. inspect challengers in D1 ranks 6..TOPN;
  4. choose the challenger with best Harrier parent rank;
  5. swap challenger into slot 5 only if:
       HarrierRank(challenger) <= HMAX
       HarrierRank(defender) - HarrierRank(challenger) >= GAP
  6. at most one swap/query.

The grid is intentionally tiny and monotone. Selection favors a broad positive
plateau and then the most conservative member of that plateau.

Outputs
-------
results/manual/huy_harrier_d1_gate_v1/
  CAL_HARRIER_GATE_REPORT.json
  CAL_HARRIER_POLICY.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

EXPECTED_D1 = 0.9569444444444444
SEED = 2026
D1_VIEWS = ["base", "expanded", "jina", "dense", "corpus"]

TOPN_GRID = [8, 10, 12, 15, 20]
HMAX_GRID = [3, 5, 8, 10, 15]
GAP_GRID = [2, 4, 6, 10, 15]

WS = re.compile(r"\s+")


def norm_text(s: str) -> str:
    return WS.sub(
        " ",
        unicodedata.normalize("NFKC", str(s or "")).lower().strip(),
    )


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def autodiscover(root: Path, explicit: Path | None, basename: str) -> Path:
    if explicit is not None:
        p = explicit.expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(p)
        return p

    preferred = [
        root / "results/from_drive/vietlegal" / basename,
        root / "results/from_drive/vietlegal/vietlegal_finetuned" / basename,
        root / "results/from_drive" / basename,
    ]
    for p in preferred:
        if p.is_file():
            return p

    hits = [
        p for p in root.rglob(basename)
        if ".git" not in p.parts
    ]
    if len(hits) == 1:
        return hits[0].resolve()
    if not hits:
        raise FileNotFoundError(
            f"Could not find {basename}. Pass --harrier-train-results explicitly."
        )
    raise RuntimeError(
        f"Found multiple {basename} files; pass the intended one explicitly:\n"
        + "\n".join(f"  {p}" for p in hits)
    )


def dedup_parent_rank(results: list[dict], limit: int = 200) -> dict[str, int]:
    rank: dict[str, int] = {}
    parent_pos = 0
    for row in results[:limit]:
        doc = str(row.get("ctx_id", row.get("doc_id", "")))
        if not doc or doc in rank:
            continue
        parent_pos += 1
        rank[doc] = parent_pos
    return rank


def load_harrier(path: Path, all_ids: list[str], queries) -> tuple[dict, dict]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    ranks = {}
    mismatches = []
    missing = []
    distinct_parent_counts = []

    for q in all_ids:
        if q not in raw:
            missing.append(q)
            continue
        row = raw[q]
        expected_q = queries[q][0] if isinstance(queries[q], (tuple, list)) else queries[q]
        got_q = row.get("question", "")
        if norm_text(expected_q) != norm_text(got_q):
            mismatches.append(q)
        ranks[q] = dedup_parent_rank(row.get("results", []), 200)
        distinct_parent_counts.append(len(ranks[q]))

    if missing:
        raise RuntimeError(
            f"Harrier artifact missing {len(missing)} CAL qids; first={missing[:10]}"
        )
    if mismatches:
        raise RuntimeError(
            f"Harrier question fingerprint mismatch for {len(mismatches)} qids; "
            f"first={mismatches[:10]}"
        )

    stats = {
        "queries": len(all_ids),
        "distinct_parents_min": int(min(distinct_parent_counts)),
        "distinct_parents_mean": float(np.mean(distinct_parent_counts)),
        "distinct_parents_p50": float(np.median(distinct_parent_counts)),
        "distinct_parents_max": int(max(distinct_parent_counts)),
    }
    return ranks, stats


def run_d1_lobo(
    *,
    blocks,
    all_ids,
    extended,
    local_views,
    full_channels,
    type_rows,
    cite_rows,
    gold,
):
    from tune_expanded_fusion_selection import ltr_features

    predictions = {}
    full_orders = {}
    decision_scores = {}

    for held in sorted(blocks):
        train = sum(
            (list(blocks[b]) for b in sorted(blocks) if b != held),
            [],
        )
        test = list(blocks[held])
        eval_ids = train + test

        rows0, groups = ltr_features(
            local_views,
            D1_VIEWS,
            extended,
            eval_ids,
            full_channels,
        )
        rows = {
            q: np.concatenate(
                [rows0[q], type_rows[q], cite_rows[q]],
                axis=1,
            )
            for q in eval_ids
        }

        got_dim = rows[eval_ids[0]].shape[1]
        if got_dim != 48:
            raise RuntimeError(f"D1 feature dim {got_dim} != 48")

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
            random_state=SEED,
        )
        model.fit(scaler.transform(X), y)

        for q in test:
            s = model.decision_function(scaler.transform(rows[q]))
            order_idx = np.argsort(-s)
            order = [groups[q][i] for i in order_idx]
            full_orders[q] = order
            predictions[q] = order[:5]
            decision_scores[q] = {
                groups[q][i]: float(s[i])
                for i in range(len(groups[q]))
            }

    return predictions, full_orders, decision_scores


def metrics(pred, gold, ids, blocks):
    per_r = {}
    per_p = {}
    for q in ids:
        hit = len(set(pred[q]) & set(gold[q]))
        per_r[q] = hit / max(1, len(gold[q]))
        per_p[q] = hit / 5.0

    block_r = {
        b: float(np.mean([per_r[q] for q in blocks[b]]))
        for b in sorted(blocks)
    }
    single = [per_r[q] for q in ids if len(gold[q]) == 1]
    multi = [per_r[q] for q in ids if len(gold[q]) > 1]

    return {
        "recall": float(np.mean([per_r[q] for q in ids])),
        "precision": float(np.mean([per_p[q] for q in ids])),
        "single_recall": float(np.mean(single)) if single else None,
        "multi_recall": float(np.mean(multi)) if multi else None,
        "blocks": block_r,
        "per_query_recall": per_r,
    }


def apply_policy_one(
    d1_order: list[str],
    h_rank: dict[str, int],
    *,
    topn: int,
    hmax: int,
    gap: int,
    missing_rank: int = 1000,
) -> tuple[list[str], dict | None]:
    if len(d1_order) < 6:
        return d1_order[:5], None

    top5 = list(d1_order[:5])
    defender = top5[4]
    defender_hr = int(h_rank.get(defender, missing_rank))

    challengers = []
    upper = min(len(d1_order), topn)
    for d1_idx in range(5, upper):
        doc = d1_order[d1_idx]
        hr = int(h_rank.get(doc, missing_rank))
        if hr <= hmax and (defender_hr - hr) >= gap:
            challengers.append((hr, d1_idx + 1, doc))

    if not challengers:
        return top5, None

    hr, d1_rank, challenger = min(challengers)
    out = top5[:4] + [challenger]
    return out, {
        "defender": defender,
        "challenger": challenger,
        "defender_harrier_rank": defender_hr,
        "challenger_harrier_rank": hr,
        "challenger_d1_rank": d1_rank,
    }


def evaluate_policy(
    *,
    config,
    d1_orders,
    harrier_ranks,
    gold,
    ids,
    blocks,
    base_metrics,
    base_pred,
):
    pred = {}
    swaps = {}
    for q in ids:
        pred[q], swap = apply_policy_one(
            d1_orders[q],
            harrier_ranks[q],
            **config,
        )
        if swap is not None:
            swaps[q] = swap

    m = metrics(pred, gold, ids, blocks)
    wins = losses = set_churn = 0
    for q in ids:
        br = base_metrics["per_query_recall"][q]
        nr = m["per_query_recall"][q]
        wins += int(nr > br + 1e-12)
        losses += int(nr < br - 1e-12)
        set_churn += int(set(pred[q]) != set(base_pred[q]))

    block_delta = {
        b: m["blocks"][b] - base_metrics["blocks"][b]
        for b in sorted(blocks)
    }
    multi_delta = (
        None if m["multi_recall"] is None
        else m["multi_recall"] - base_metrics["multi_recall"]
    )

    return {
        "config": dict(config),
        "recall": m["recall"],
        "precision": m["precision"],
        "delta_recall": m["recall"] - base_metrics["recall"],
        "single_recall": m["single_recall"],
        "multi_recall": m["multi_recall"],
        "multi_delta": multi_delta,
        "blocks": m["blocks"],
        "block_delta": block_delta,
        "wins": int(wins),
        "losses": int(losses),
        "net_wins": int(wins - losses),
        "swaps": int(len(swaps)),
        "set_churn": int(set_churn),
        "swap_details": swaps,
    }


def cfg_key(c):
    return (c["topn"], c["hmax"], c["gap"])


def attach_neighborhood_stability(results):
    by_key = {cfg_key(r["config"]): r for r in results}
    topn_pos = {v: i for i, v in enumerate(TOPN_GRID)}
    hmax_pos = {v: i for i, v in enumerate(HMAX_GRID)}
    gap_pos = {v: i for i, v in enumerate(GAP_GRID)}

    for r in results:
        c = r["config"]
        p = (
            topn_pos[c["topn"]],
            hmax_pos[c["hmax"]],
            gap_pos[c["gap"]],
        )
        neighborhood = []
        for other in results:
            oc = other["config"]
            op = (
                topn_pos[oc["topn"]],
                hmax_pos[oc["hmax"]],
                gap_pos[oc["gap"]],
            )
            manhattan = sum(abs(a - b) for a, b in zip(p, op))
            if manhattan <= 1:
                neighborhood.append(other)

        positive = [
            x for x in neighborhood
            if x["delta_recall"] > 1e-12 and x["wins"] > x["losses"]
        ]
        deltas = [x["delta_recall"] for x in neighborhood]
        r["stability"] = {
            "neighbors_including_self": len(neighborhood),
            "positive_neighbors": len(positive),
            "positive_fraction": (
                len(positive) / len(neighborhood) if neighborhood else 0.0
            ),
            "median_delta_recall": float(np.median(deltas)),
            "min_delta_recall": float(min(deltas)),
            "max_delta_recall": float(max(deltas)),
        }


def compact_result(r):
    out = {k: v for k, v in r.items() if k != "swap_details"}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=Path, required=True)
    ap.add_argument("--harrier-train-results", type=Path, default=None)
    args = ap.parse_args()

    root = args.repo_root.expanduser().resolve()
    sys.path.insert(0, str(root))

    harrier_path = autodiscover(
        root,
        args.harrier_train_results,
        "train_finetuned_best.json",
    )

    print("[1/5] Loading authoritative CAL600 D1 bundle...", flush=True)
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

    print("[2/5] Reconstructing exact D1 LOBO ranking...", flush=True)
    d1_pred, d1_orders, d1_scores = run_d1_lobo(
        blocks=blocks,
        all_ids=all_ids,
        extended=extended,
        local_views=local_views,
        full_channels=full_channels,
        type_rows=type_rows,
        cite_rows=cite_rows,
        gold=gold,
    )
    base = metrics(d1_pred, gold, all_ids, blocks)
    print(
        f"  D1 R={base['recall']:.10f} P={base['precision']:.10f} "
        f"multi={base['multi_recall']:.6f}",
        flush=True,
    )
    if abs(base["recall"] - EXPECTED_D1) > 1e-9:
        raise RuntimeError(
            f"D1 parity failed: {base['recall']} != {EXPECTED_D1}"
        )

    print("[3/5] Loading Harrier full-corpus train retrieval...", flush=True)
    harrier_ranks, harrier_stats = load_harrier(
        harrier_path, all_ids, queries
    )
    print(
        "  parent-rank coverage: "
        f"min={harrier_stats['distinct_parents_min']} "
        f"mean={harrier_stats['distinct_parents_mean']:.1f} "
        f"p50={harrier_stats['distinct_parents_p50']:.0f} "
        f"max={harrier_stats['distinct_parents_max']}",
        flush=True,
    )

    print("[4/5] Sweeping conservative one-swap policy grid...", flush=True)
    results = []
    for topn in TOPN_GRID:
        for hmax in HMAX_GRID:
            for gap in GAP_GRID:
                config = {
                    "topn": topn,
                    "hmax": hmax,
                    "gap": gap,
                    "missing_rank": 1000,
                }
                results.append(
                    evaluate_policy(
                        config=config,
                        d1_orders=d1_orders,
                        harrier_ranks=harrier_ranks,
                        gold=gold,
                        ids=all_ids,
                        blocks=blocks,
                        base_metrics=base,
                        base_pred=d1_pred,
                    )
                )

    attach_neighborhood_stability(results)

    # Leakage-aware gate: require a positive plateau, not a single magic point.
    eligible = []
    for r in results:
        s = r["stability"]
        min_block_delta = min(r["block_delta"].values())
        if (
            r["delta_recall"] > 1e-12
            and r["wins"] > r["losses"]
            and min_block_delta >= -0.005 - 1e-12
            and (r["multi_delta"] is None or r["multi_delta"] >= -0.01 - 1e-12)
            and s["positive_fraction"] >= 0.70
            and s["median_delta_recall"] > 1e-12
        ):
            eligible.append(r)

    # First find the best robust plateau. Then choose the most conservative
    # config within 0.001 pooled recall of that plateau's best median.
    recommended = None
    if eligible:
        best_plateau = max(
            r["stability"]["median_delta_recall"] for r in eligible
        )
        plateau = [
            r for r in eligible
            if r["stability"]["median_delta_recall"] >= best_plateau - 0.001
        ]
        recommended = min(
            plateau,
            key=lambda r: (
                r["swaps"],
                r["losses"],
                -r["net_wins"],
                -r["delta_recall"],
                r["config"]["topn"],
                r["config"]["hmax"],
                -r["config"]["gap"],
            ),
        )

    ranked = sorted(
        results,
        key=lambda r: (
            -r["stability"]["median_delta_recall"],
            -r["delta_recall"],
            -r["net_wins"],
            r["losses"],
            r["swaps"],
        ),
    )

    print("[5/5] Gate summary...", flush=True)
    print(
        f"  tested={len(results)} eligible={len(eligible)} "
        f"leakage_warning=YES",
        flush=True,
    )
    for i, r in enumerate(ranked[:12], 1):
        c = r["config"]
        print(
            f"  #{i:02d} topn={c['topn']:2d} hmax={c['hmax']:2d} "
            f"gap={c['gap']:2d} | dR={r['delta_recall']:+.6f} "
            f"W/L={r['wins']}/{r['losses']} swaps={r['swaps']:3d} "
            f"plateau_med={r['stability']['median_delta_recall']:+.6f} "
            f"support={r['stability']['positive_fraction']:.0%}",
            flush=True,
        )

    outdir = root / "results/manual/huy_harrier_d1_gate_v1"
    outdir.mkdir(parents=True, exist_ok=True)

    verdict = (
        "PROMOTE_HARRIER_PRIVATE_RANK_SPECIALIST"
        if recommended is not None
        else "KILL_HARRIER_PRIVATE_RANK_SPECIALIST"
    )

    report = {
        "schema": "manual.harrier_d1_gate_v1",
        "status": verdict,
        "scientific_warning": (
            "Harrier final fine-tune train pool includes CAL600 queries. "
            "This audit is an in-sample mechanism/stability gate, not a "
            "generalization estimate."
        ),
        "harrier_artifact": {
            "path": str(harrier_path),
            "sha256": sha256(harrier_path),
            "stats": harrier_stats,
        },
        "baseline_d1": {
            "expected_recall": EXPECTED_D1,
            "recall": base["recall"],
            "precision": base["precision"],
            "single_recall": base["single_recall"],
            "multi_recall": base["multi_recall"],
            "blocks": base["blocks"],
        },
        "grid": {
            "topn": TOPN_GRID,
            "hmax": HMAX_GRID,
            "gap": GAP_GRID,
            "policy": (
                "Keep D1 top1..4; inspect D1 rank6..TOPN; at most one "
                "rank5 swap if Harrier challenger rank<=HMAX and "
                "defender_rank-challenger_rank>=GAP."
            ),
        },
        "gate": {
            "eligible_configs": len(eligible),
            "requirements": {
                "delta_recall_positive": True,
                "wins_gt_losses": True,
                "min_block_delta_ge": -0.005,
                "multi_delta_ge": -0.01,
                "neighbor_positive_fraction_ge": 0.70,
                "neighbor_median_delta_positive": True,
            },
        },
        "recommended": compact_result(recommended) if recommended else None,
        "top_configs": [compact_result(r) for r in ranked[:30]],
    }

    report_path = outdir / "CAL_HARRIER_GATE_REPORT.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    policy_path = outdir / "CAL_HARRIER_POLICY.json"
    if recommended is not None:
        policy = {
            "schema": "manual.harrier_boundary_policy.v1",
            "source_report": str(report_path),
            "policy_type": "d1_rank5_single_swap",
            **recommended["config"],
            "cal_metrics": {
                k: recommended[k]
                for k in (
                    "recall", "precision", "delta_recall",
                    "wins", "losses", "net_wins", "swaps",
                    "single_recall", "multi_recall",
                    "multi_delta", "blocks", "block_delta",
                    "stability",
                )
            },
            "warning": (
                "Selected on leakage-contaminated final Harrier checkpoint; "
                "use as a private experimental arm, not as a clean CV claim."
            ),
        }
        policy_path.write_text(
            json.dumps(policy, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    elif policy_path.exists():
        policy_path.unlink()

    print("=" * 108)
    print("VERDICT:", verdict)
    if recommended:
        print("POLICY:", recommended["config"])
        print(
            f"CAL dR={recommended['delta_recall']:+.10f} "
            f"W/L={recommended['wins']}/{recommended['losses']} "
            f"swaps={recommended['swaps']}"
        )
        print("Policy:", policy_path)
    print("Report:", report_path)
    print("=" * 108)


if __name__ == "__main__":
    main()
