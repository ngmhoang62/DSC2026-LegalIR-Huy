#!/usr/bin/env python
"""
AUDIT HARRIER-FT AS A D1 LTR FEATURE
====================================

CPU-only. No private labels. No GPU inference.

This is deliberately different from the failed direct rank-5 swap policy:
Harrier never makes a direct selection decision. Its full-corpus retrieval
artifact contributes one score channel and/or one rank view to the existing
D1 LogisticRegression.

WARNING
-------
The final Harrier checkpoint was trained on a pool that includes CAL600.
Therefore this is NOT a clean estimate of generalization. We use it only as
a mechanism + stability gate before spending local GPU hours.

Harrier artifact
----------------
Expected format: train_finetuned_best.json
{
  qid: {
    "question": ...,
    "results": [
      {"ctx_id": ..., "chunk": ..., "score": ...},
      ...
    ]
  }
}

For each parent document:
  parent_score = max chunk cosine score among saved top-200 chunks
  parent_rank  = first deduplicated parent position in saved top-200 chunks

Arms
----
d1                         : authoritative 48D baseline
harrier_score              : + Harrier score channel                  50D
harrier_rank               : + Harrier global parent rank view        50D
harrier_score_rank         : + both                                   52D
harrier_replace_vnlegal    : replace vnlegal_lal score with Harrier   48D
harrier_replace_aiteam_ft  : replace aiteamvn_ft score with Harrier   48D
harrier_replace_jina_ft    : replace jina_ft score with Harrier       48D
harrier_replace_crossenc   : replace crossenc score with Harrier      48D

Robustness
----------
Run LOBO at C in {0.10, 0.15, 0.20}. Each Harrier arm is compared with D1 at
the SAME C. C=.15 must exactly reproduce the authoritative D1 recall.

Promotion requires:
  * positive median delta across C;
  * positive delta at >=2/3 C values;
  * at C=.15: delta >= +0.0025, wins > losses;
  * at C=.15: worst block delta >= -0.005;
  * at C=.15: multi-gold delta >= -0.005;
  * Harrier candidate-score coverage mean >= 20%.

The threshold is intentionally stronger than "one lucky query", but this report
also prints all arms so the human can inspect borderline cases.

Output
------
results/manual/huy_harrier_d1_feature_gate_v1/
  REPORT.json
  PROMOTED_ARM.json       (only if gate passes)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

EXPECTED_D1_C015 = 0.9569444444444444
SEED = 2026
BASE_VIEWS = ["base", "expanded", "jina", "dense", "corpus"]
C_GRID = [0.10, 0.15, 0.20]

ARMS = [
    "d1",
    "harrier_score",
    "harrier_rank",
    "harrier_score_rank",
    "harrier_replace_vnlegal",
    "harrier_replace_aiteam_ft",
    "harrier_replace_jina_ft",
    "harrier_replace_crossenc",
]

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


def autodiscover(root: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        p = explicit.expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(p)
        return p

    preferred = [
        root / "results/from_drive/vietlegal/train_finetuned_best.json",
        root / "results/from_drive/vietlegal/vietlegal_finetuned/train_finetuned_best.json",
        root / "results/from_drive/train_finetuned_best.json",
    ]
    for p in preferred:
        if p.is_file():
            return p.resolve()

    hits = [
        p for p in root.rglob("train_finetuned_best.json")
        if ".git" not in p.parts
    ]
    if len(hits) == 1:
        return hits[0].resolve()
    if not hits:
        raise FileNotFoundError(
            "train_finetuned_best.json not found; pass "
            "--harrier-train-results explicitly."
        )
    raise RuntimeError(
        "Multiple train_finetuned_best.json files found; pass one explicitly:\n"
        + "\n".join(f"  {p}" for p in hits)
    )


def parse_harrier(path: Path, all_ids, queries, candidates):
    raw = json.loads(path.read_text(encoding="utf-8"))

    scores = {}
    rank_view = {}
    missing_q = []
    mismatch_q = []
    query_coverages = []
    parent_counts = []
    candidate_hits = 0
    candidate_total = 0

    for q in all_ids:
        if q not in raw:
            missing_q.append(q)
            continue

        row = raw[q]
        expected_question = queries[q][0]
        if norm_text(row.get("question", "")) != norm_text(expected_question):
            mismatch_q.append(q)

        best = {}
        order = []
        seen = set()

        for item in row.get("results", []):
            d = str(item.get("ctx_id", item.get("doc_id", "")))
            if not d:
                continue
            s = float(item["score"])

            if d not in best or s > best[d]:
                best[d] = s

            if d not in seen:
                seen.add(d)
                order.append(d)

        scores[q] = best
        rank_view[q] = order
        parent_counts.append(len(order))

        cand = list(map(str, candidates[q]))
        hit = sum(d in best for d in cand)
        candidate_hits += hit
        candidate_total += len(cand)
        query_coverages.append(hit / max(1, len(cand)))

    if missing_q:
        raise RuntimeError(
            f"Harrier artifact missing {len(missing_q)} CAL qids; "
            f"first={missing_q[:10]}"
        )
    if mismatch_q:
        raise RuntimeError(
            f"Question-text mismatch on {len(mismatch_q)} qids; "
            f"first={mismatch_q[:10]}"
        )

    stats = {
        "queries": len(all_ids),
        "dedup_parents_min": int(min(parent_counts)),
        "dedup_parents_mean": float(np.mean(parent_counts)),
        "dedup_parents_p50": float(np.median(parent_counts)),
        "dedup_parents_max": int(max(parent_counts)),
        "candidate_present": int(candidate_hits),
        "candidate_total": int(candidate_total),
        "candidate_coverage": float(candidate_hits / max(1, candidate_total)),
        "query_coverage_min": float(min(query_coverages)),
        "query_coverage_mean": float(np.mean(query_coverages)),
        "query_coverage_p50": float(np.median(query_coverages)),
        "query_coverage_p10": float(np.quantile(query_coverages, .10)),
        "queries_zero_candidate_coverage": int(sum(x == 0 for x in query_coverages)),
    }
    return scores, rank_view, stats


def build_arm(
    arm,
    *,
    full_channels,
    local_views,
    harrier_scores,
    harrier_rank_view,
):
    channels = dict(full_channels)
    views = dict(local_views)
    view_names = list(BASE_VIEWS)

    if arm == "d1":
        expected_dim = 48

    elif arm == "harrier_score":
        channels["harrier_ft"] = harrier_scores
        expected_dim = 50

    elif arm == "harrier_rank":
        views["harrier_ft"] = harrier_rank_view
        view_names.append("harrier_ft")
        expected_dim = 50

    elif arm == "harrier_score_rank":
        channels["harrier_ft"] = harrier_scores
        views["harrier_ft"] = harrier_rank_view
        view_names.append("harrier_ft")
        expected_dim = 52

    elif arm == "harrier_replace_vnlegal":
        channels.pop("vnlegal_lal", None)
        channels["harrier_ft"] = harrier_scores
        expected_dim = 48

    elif arm == "harrier_replace_aiteam_ft":
        channels.pop("aiteamvn_ft", None)
        channels["harrier_ft"] = harrier_scores
        expected_dim = 48

    elif arm == "harrier_replace_jina_ft":
        channels.pop("jina_ft", None)
        channels["harrier_ft"] = harrier_scores
        expected_dim = 48

    elif arm == "harrier_replace_crossenc":
        channels.pop("crossenc", None)
        channels["harrier_ft"] = harrier_scores
        expected_dim = 48

    else:
        raise ValueError(arm)

    return channels, views, view_names, expected_dim


def run_lobo(
    *,
    arm,
    c_value,
    blocks,
    all_ids,
    candidates,
    local_views,
    full_channels,
    harrier_scores,
    harrier_rank_view,
    type_rows,
    cite_rows,
    gold,
):
    from tune_expanded_fusion_selection import ltr_features

    channels, views, view_names, expected_dim = build_arm(
        arm,
        full_channels=full_channels,
        local_views=local_views,
        harrier_scores=harrier_scores,
        harrier_rank_view=harrier_rank_view,
    )

    pred = {}
    score_maps = {}

    for held in sorted(blocks):
        train = sum(
            (list(blocks[b]) for b in sorted(blocks) if b != held),
            [],
        )
        test = list(blocks[held])
        eval_ids = train + test

        rows0, groups = ltr_features(
            views,
            view_names,
            candidates,
            eval_ids,
            channels,
        )
        rows = {
            q: np.concatenate(
                [rows0[q], type_rows[q], cite_rows[q]],
                axis=1,
            )
            for q in eval_ids
        }

        got = rows[eval_ids[0]].shape[1]
        if got != expected_dim:
            raise RuntimeError(
                f"{arm} C={c_value}: feature dim {got} != {expected_dim}"
            )

        X = np.vstack([rows[q] for q in train])
        y = np.concatenate([
            [d in gold[q] for d in groups[q]]
            for q in train
        ]).astype(np.int8)

        scaler = StandardScaler().fit(X)
        model = LogisticRegression(
            C=float(c_value),
            class_weight="balanced",
            solver="liblinear",
            max_iter=3000,
            random_state=SEED,
        )
        model.fit(scaler.transform(X), y)

        for q in test:
            s = model.decision_function(scaler.transform(rows[q]))
            order = np.argsort(-s)
            pred[q] = [groups[q][i] for i in order[:5]]
            score_maps[q] = {
                groups[q][i]: float(s[i])
                for i in range(len(groups[q]))
            }

    return summarize(
        arm=arm,
        c_value=c_value,
        pred=pred,
        score_maps=score_maps,
        gold=gold,
        all_ids=all_ids,
        blocks=blocks,
        expected_dim=expected_dim,
    )


def summarize(
    *,
    arm,
    c_value,
    pred,
    score_maps,
    gold,
    all_ids,
    blocks,
    expected_dim,
):
    per_r = {}
    per_p = {}

    for q in all_ids:
        hit = len(set(pred[q]) & set(gold[q]))
        per_r[q] = hit / max(1, len(gold[q]))
        per_p[q] = hit / 5.0

    block_r = {
        b: float(np.mean([per_r[q] for q in blocks[b]]))
        for b in sorted(blocks)
    }

    single = [per_r[q] for q in all_ids if len(gold[q]) == 1]
    multi = [per_r[q] for q in all_ids if len(gold[q]) > 1]

    return {
        "arm": arm,
        "C": float(c_value),
        "feature_dim": int(expected_dim),
        "recall": float(np.mean([per_r[q] for q in all_ids])),
        "precision": float(np.mean([per_p[q] for q in all_ids])),
        "single_recall": float(np.mean(single)),
        "multi_recall": float(np.mean(multi)),
        "blocks": block_r,
        "predictions": pred,
        "decision_scores": score_maps,
        "per_query_recall": per_r,
    }


def compare(arm_res, base_res, all_ids):
    wins = losses = churn_set = churn_order = 0
    for q in all_ids:
        a = base_res["per_query_recall"][q]
        b = arm_res["per_query_recall"][q]
        wins += int(b > a + 1e-12)
        losses += int(b < a - 1e-12)
        churn_set += int(
            set(arm_res["predictions"][q])
            != set(base_res["predictions"][q])
        )
        churn_order += int(
            arm_res["predictions"][q]
            != base_res["predictions"][q]
        )

    return {
        "delta_recall": arm_res["recall"] - base_res["recall"],
        "delta_precision": arm_res["precision"] - base_res["precision"],
        "delta_single": arm_res["single_recall"] - base_res["single_recall"],
        "delta_multi": arm_res["multi_recall"] - base_res["multi_recall"],
        "block_delta": {
            b: arm_res["blocks"][b] - base_res["blocks"][b]
            for b in sorted(base_res["blocks"])
        },
        "wins": int(wins),
        "losses": int(losses),
        "net_wins": int(wins - losses),
        "set_top5_churn": int(churn_set),
        "ordered_top5_churn": int(churn_order),
    }


def compact_result(r):
    return {
        k: v
        for k, v in r.items()
        if k not in ("predictions", "decision_scores", "per_query_recall")
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=Path, required=True)
    ap.add_argument("--harrier-train-results", type=Path, default=None)
    args = ap.parse_args()

    root = args.repo_root.expanduser().resolve()
    sys.path.insert(0, str(root))
    harrier_path = autodiscover(root, args.harrier_train_results)

    print("[1/5] Loading authoritative D1 CAL600 bundle...", flush=True)
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

    print("[2/5] Parsing Harrier chunk retrieval -> parent score/rank...", flush=True)
    harrier_scores, harrier_rank_view, coverage = parse_harrier(
        harrier_path,
        all_ids,
        queries,
        extended,
    )
    print(
        f"  candidate coverage={coverage['candidate_coverage']:.2%} | "
        f"mean/query={coverage['query_coverage_mean']:.2%} | "
        f"p10={coverage['query_coverage_p10']:.2%} | "
        f"zero-q={coverage['queries_zero_candidate_coverage']}",
        flush=True,
    )
    print(
        f"  dedup Harrier parents/query: "
        f"min={coverage['dedup_parents_min']} "
        f"mean={coverage['dedup_parents_mean']:.1f} "
        f"p50={coverage['dedup_parents_p50']:.0f} "
        f"max={coverage['dedup_parents_max']}",
        flush=True,
    )

    print("[3/5] Running D1 baselines for C sensitivity...", flush=True)
    results = {}
    for c in C_GRID:
        r = run_lobo(
            arm="d1",
            c_value=c,
            blocks=blocks,
            all_ids=all_ids,
            candidates=extended,
            local_views=local_views,
            full_channels=full_channels,
            harrier_scores=harrier_scores,
            harrier_rank_view=harrier_rank_view,
            type_rows=type_rows,
            cite_rows=cite_rows,
            gold=gold,
        )
        results[("d1", c)] = r
        print(
            f"  D1 C={c:.2f} R={r['recall']:.10f} "
            f"P={r['precision']:.10f} multi={r['multi_recall']:.6f}",
            flush=True,
        )

    d1_015 = results[("d1", 0.15)]
    if abs(d1_015["recall"] - EXPECTED_D1_C015) > 1e-9:
        raise RuntimeError(
            f"Authoritative D1 parity failed: {d1_015['recall']} "
            f"!= {EXPECTED_D1_C015}"
        )

    print("[4/5] Running Harrier feature arms...", flush=True)
    comparisons = {}
    for arm in ARMS[1:]:
        deltas = []
        print(f"\n--- {arm} ---", flush=True)
        for c in C_GRID:
            r = run_lobo(
                arm=arm,
                c_value=c,
                blocks=blocks,
                all_ids=all_ids,
                candidates=extended,
                local_views=local_views,
                full_channels=full_channels,
                harrier_scores=harrier_scores,
                harrier_rank_view=harrier_rank_view,
                type_rows=type_rows,
                cite_rows=cite_rows,
                gold=gold,
            )
            results[(arm, c)] = r
            cmp = compare(r, results[("d1", c)], all_ids)
            comparisons[(arm, c)] = cmp
            deltas.append(cmp["delta_recall"])
            print(
                f"  C={c:.2f} R={r['recall']:.10f} "
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
            f"  sensitivity median dR={np.median(deltas):+.10f} "
            f"positive={sum(x > 1e-12 for x in deltas)}/3 "
            f"range=[{min(deltas):+.10f},{max(deltas):+.10f}]",
            flush=True,
        )

    print("\n[5/5] Leakage-aware promotion gate...", flush=True)
    arm_summaries = {}
    eligible = []

    for arm in ARMS[1:]:
        cmps = [comparisons[(arm, c)] for c in C_GRID]
        deltas = [x["delta_recall"] for x in cmps]
        c015 = comparisons[(arm, 0.15)]
        r015 = results[(arm, 0.15)]

        summary = {
            "arm": arm,
            "feature_dim": r015["feature_dim"],
            "median_delta_across_C": float(np.median(deltas)),
            "min_delta_across_C": float(min(deltas)),
            "max_delta_across_C": float(max(deltas)),
            "positive_C_count": int(sum(x > 1e-12 for x in deltas)),
            "deltas_by_C": {
                f"{c:.2f}": comparisons[(arm, c)]["delta_recall"]
                for c in C_GRID
            },
            "C015": {
                **compact_result(r015),
                "comparison_vs_sameC_d1": c015,
            },
        }

        passes = {
            "coverage_mean_ge_0_20": (
                coverage["query_coverage_mean"] >= 0.20
            ),
            "median_delta_positive": (
                summary["median_delta_across_C"] > 1e-12
            ),
            "positive_at_least_2_of_3_C": (
                summary["positive_C_count"] >= 2
            ),
            "C015_delta_ge_0_0025": (
                c015["delta_recall"] >= 0.0025 - 1e-12
            ),
            "C015_wins_gt_losses": (
                c015["wins"] > c015["losses"]
            ),
            "C015_worst_block_ge_neg_0_005": (
                min(c015["block_delta"].values()) >= -0.005 - 1e-12
            ),
            "C015_multi_ge_neg_0_005": (
                c015["delta_multi"] >= -0.005 - 1e-12
            ),
        }
        summary["gates"] = passes
        summary["all_gates_pass"] = all(passes.values())
        arm_summaries[arm] = summary

        if summary["all_gates_pass"]:
            eligible.append(summary)

        print(
            f"  {arm:27s} "
            f"median={summary['median_delta_across_C']:+.6f} "
            f"C015={c015['delta_recall']:+.6f} "
            f"W/L={c015['wins']}/{c015['losses']} "
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
                s["C015"]["comparison_vs_sameC_d1"]["delta_recall"],
                s["C015"]["comparison_vs_sameC_d1"]["net_wins"],
                -s["C015"]["comparison_vs_sameC_d1"]["set_top5_churn"],
            ),
        )

    verdict = (
        "PROMOTE_HARRIER_AS_D1_FEATURE"
        if recommended is not None
        else "KILL_HARRIER_D1_FEATURE"
    )

    outdir = root / "results/manual/huy_harrier_d1_feature_gate_v1"
    outdir.mkdir(parents=True, exist_ok=True)

    report = {
        "schema": "manual.harrier_d1_feature_gate_v1",
        "status": verdict,
        "scientific_warning": (
            "Final Harrier fine-tune includes CAL600 queries in its training pool. "
            "All Harrier-arm metrics are contamination-biased and are used only "
            "as a mechanism/stability gate."
        ),
        "harrier_artifact": {
            "path": str(harrier_path),
            "sha256": sha256(harrier_path),
            "coverage": coverage,
            "parent_score": "max saved chunk cosine score per ctx_id",
            "parent_rank": "first deduplicated ctx_id position in saved top200 chunks",
        },
        "C_grid": C_GRID,
        "baseline_d1": {
            f"{c:.2f}": compact_result(results[("d1", c)])
            for c in C_GRID
        },
        "arms": arm_summaries,
        "recommended_arm": (
            recommended["arm"] if recommended is not None else None
        ),
        "promotion_gate": {
            "coverage_mean_ge": 0.20,
            "median_delta_across_C_positive": True,
            "positive_C_count_ge": 2,
            "C015_delta_recall_ge": 0.0025,
            "C015_wins_gt_losses": True,
            "C015_worst_block_delta_ge": -0.005,
            "C015_multi_delta_ge": -0.005,
        },
    }

    report_path = outdir / "REPORT.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    promoted_path = outdir / "PROMOTED_ARM.json"
    if recommended is not None:
        promoted = {
            "schema": "manual.harrier_d1_feature_arm.v1",
            "arm": recommended["arm"],
            "feature_dim": recommended["feature_dim"],
            "C": 0.15,
            "source_report": str(report_path),
            "cal_contaminated_metrics": recommended,
            "private_scoring_requirement": (
                "Score Harrier on the exact D1 private candidate pool using "
                "source-faithful chunk embeddings. Do not use selected-context "
                "proxy passages if exact chunk-context is available."
            ),
        }
        promoted_path.write_text(
            json.dumps(promoted, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    elif promoted_path.exists():
        promoted_path.unlink()

    print("=" * 112)
    print("VERDICT:", verdict)
    print(
        "RECOMMENDED ARM:",
        recommended["arm"] if recommended is not None else None,
    )
    if recommended is not None:
        cmp = recommended["C015"]["comparison_vs_sameC_d1"]
        print(
            f"C=.15 dR={cmp['delta_recall']:+.10f} "
            f"W/L={cmp['wins']}/{cmp['losses']} "
            f"churn={cmp['set_top5_churn']}"
        )
        print("Promoted arm:", promoted_path)
    print("Report:", report_path)
    print("=" * 112)


if __name__ == "__main__":
    main()
