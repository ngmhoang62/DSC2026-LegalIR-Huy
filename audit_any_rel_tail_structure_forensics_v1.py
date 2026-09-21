#!/usr/bin/env python
"""
ANY_REL SIXTH-RESCUE STRUCTURAL FORENSICS V1
=============================================

CPU-only. No neural inference.

Purpose
-------
The anchor-support audit found:
  ANY_REL             : 6/15 outside-pool rescues, 14811 pairs
  narrower core arms  : 5/15 rescues

This script identifies the extra ANY_REL rescue and asks whether it belongs to
any simple, mechanistically motivated structural subset that was NOT part of
the original Stage-2 arms.

IMPORTANT
---------
This is FORENSIC / hypothesis-generation only.

All candidate populations below are deterministic and label-free, but several
fixed subsets are evaluated together. Do not directly deploy whichever subset
looks best on CAL. Use the result to design ONE next preregistered policy or to
decide the sixth rescue is too noisy to pursue.

Inputs
------
results/manual/huy_graph_anchor_support_certificate_v1/
  SEALED_CANDIDATES.jsonl

Outputs
-------
results/manual/huy_any_rel_tail_structure_forensics_v1/
  REPORT.json
"""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np

EXPECTED_ORACLE = 0.9847222222222222


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def candidate_oracle(pool, gold, ids):
    return float(np.mean([
        len(set(pool[q]) & set(gold[q])) / max(1, len(gold[q]))
        for q in ids
    ]))


def best_anchor_rank(r):
    return min(
        int(r.get("best_d1_anchor_rank", 99)),
        int(r.get("best_prism_anchor_rank", 99)),
    )


def family(r, name):
    return name in set(r.get("families", []))


def build_fixed_subsets(rows):
    """
    Fixed structural subsets for hypothesis generation.

    No labels are used here. The filters are semantic/structural:
      - top-ranked anchor proximity,
      - header evidence,
      - relation family,
      - direction,
      - independent support.
    """
    return OrderedDict([
        (
            "TOP1_ANCHOR",
            [
                r["doc_id"] for r in rows
                if best_anchor_rank(r) <= 1
            ],
        ),
        (
            "TOP2_ANCHOR",
            [
                r["doc_id"] for r in rows
                if best_anchor_rank(r) <= 2
            ],
        ),
        (
            "HEADER_REL",
            [
                r["doc_id"] for r in rows
                if int(r.get("header_support", 0)) > 0
            ],
        ),
        (
            "TOP2_HEADER",
            [
                r["doc_id"] for r in rows
                if best_anchor_rank(r) <= 2
                and int(r.get("header_support", 0)) > 0
            ],
        ),
        (
            "GUIDANCE_REL",
            [
                r["doc_id"] for r in rows
                if family(r, "IMPLEMENTATION_GUIDANCE")
            ],
        ),
        (
            "AMENDMENT_REL",
            [
                r["doc_id"] for r in rows
                if family(r, "AMENDMENT_REPLACEMENT_REPEAL")
            ],
        ),
        (
            "TOP2_GUIDANCE",
            [
                r["doc_id"] for r in rows
                if best_anchor_rank(r) <= 2
                and family(r, "IMPLEMENTATION_GUIDANCE")
            ],
        ),
        (
            "TOP2_AMENDMENT",
            [
                r["doc_id"] for r in rows
                if best_anchor_rank(r) <= 2
                and family(r, "AMENDMENT_REPLACEMENT_REPEAL")
            ],
        ),
        (
            "INCOMING_ONLY_OR_MIXED",
            [
                r["doc_id"] for r in rows
                if "INCOMING" in set(r.get("directions", []))
            ],
        ),
        (
            "OUTGOING_ONLY_OR_MIXED",
            [
                r["doc_id"] for r in rows
                if "OUTGOING" in set(r.get("directions", []))
            ],
        ),
        (
            "TOP2_RELATION_SUPPORT_2PLUS",
            [
                r["doc_id"] for r in rows
                if best_anchor_rank(r) <= 2
                and int(r.get("relation_support", 0)) >= 2
            ],
        ),
        (
            "TOP2_HEADER_OR_MULTI",
            [
                r["doc_id"] for r in rows
                if best_anchor_rank(r) <= 2
                and (
                    int(r.get("header_support", 0)) > 0
                    or int(r.get("anchor_support", 0)) >= 2
                )
            ],
        ),
    ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=Path, required=True)
    args = ap.parse_args()

    root = args.repo_root.expanduser().resolve()
    sealed_path = (
        root
        / "results/manual/huy_graph_anchor_support_certificate_v1/"
          "SEALED_CANDIDATES.jsonl"
    )
    if not sealed_path.is_file():
        raise FileNotFoundError(sealed_path)

    print("[1/5] Loading previously sealed ANY_REL artifact...", flush=True)
    rows = {}
    ids = []
    subsets = {}

    for row in read_jsonl(sealed_path):
        q = str(row["qid"])
        ids.append(q)
        rows[q] = row
        details = list(row.get("candidate_details", []))
        any_rel = [str(d) for d in row.get("arms", {}).get("ANY_REL", [])]
        reconstructed = [str(r["doc_id"]) for r in details]
        if any_rel != reconstructed:
            raise RuntimeError(f"ANY_REL order/detail drift q={q}")
        subsets[q] = build_fixed_subsets(details)

    if len(ids) != 600 or len(rows) != 600:
        raise RuntimeError(f"Expected 600 rows, got {len(ids)}/{len(rows)}")

    names = list(next(iter(subsets.values())).keys())
    print(
        "  fixed forensic subsets:",
        ", ".join(names),
        flush=True,
    )

    print("[2/5] Loading authoritative D1 candidate pool for outside-gold definition...", flush=True)
    # Candidate pool is needed only to define the known acquisition ceiling.
    from src.gemini.huy_vnlegal_rank_ablation_v1.evaluate_ablation_cal import (
        load_cal_inputs,
    )
    (
        _queries,
        _blocks,
        all_ids,
        extended,
        _local_views,
        _channels,
        gold,
        _vnlegal,
        _type_rows,
        _cite_rows,
    ) = load_cal_inputs()

    if set(all_ids) != set(ids):
        raise RuntimeError("CAL population mismatch")

    base_oracle = candidate_oracle(extended, gold, ids)
    if abs(base_oracle - EXPECTED_ORACLE) > 1e-9:
        raise RuntimeError(
            f"Candidate oracle parity failed: {base_oracle}"
        )

    outside = {
        (q, g)
        for q in ids
        for g in gold[q]
        if g not in set(extended[q])
    }
    print(
        f"  base oracle={base_oracle:.10f} outside={len(outside)}",
        flush=True,
    )

    print("[3/5] Identifying ANY_REL rescues and exact structural provenance...", flush=True)
    any_rescues = []
    for q, g in sorted(outside):
        any_order = [str(d) for d in rows[q]["arms"]["ANY_REL"]]
        if g not in set(any_order):
            continue

        detail = next(
            r for r in rows[q]["candidate_details"]
            if str(r["doc_id"]) == g
        )
        membership = {
            name: g in set(subsets[q][name])
            for name in names
        }
        original_membership = {
            arm: g in set(rows[q]["arms"].get(arm, []))
            for arm in (
                "SUPPORT_2PLUS",
                "CROSS_MODEL_SUPPORT",
                "TOP2_CROSS_MODEL",
                "DUAL_DIRECTION",
                "CROSS_OR_MULTI",
            )
        }

        any_rescues.append({
            "qid": q,
            "gold_doc": g,
            "any_rel_rank": any_order.index(g) + 1,
            "detail": detail,
            "original_arm_membership": original_membership,
            "fixed_subset_membership": membership,
        })

    print(f"  ANY_REL rescues={len(any_rescues)}/{len(outside)}", flush=True)
    for x in any_rescues:
        old = [
            k for k, v in x["original_arm_membership"].items() if v
        ]
        new = [
            k for k, v in x["fixed_subset_membership"].items() if v
        ]
        d = x["detail"]
        print(
            f"    q={x['qid']} gold={x['gold_doc']} "
            f"rank={x['any_rel_rank']} "
            f"anchor_support={d.get('anchor_support')} "
            f"bestD1={d.get('best_d1_anchor_rank')} "
            f"bestPrism={d.get('best_prism_anchor_rank')} "
            f"dirs={d.get('directions')} fam={d.get('families')} "
            f"header={d.get('header_support')} "
            f"old={old} fixed={new}",
            flush=True,
        )

    print("[4/5] Evaluating fixed structural subsets (FORENSIC ONLY)...", flush=True)
    evals = OrderedDict()
    for name in names:
        cand = {q: subsets[q][name] for q in ids}
        sizes = np.asarray([len(cand[q]) for q in ids])
        rescued = sorted(
            (q, g) for q, g in outside if g in set(cand[q])
        )
        augmented = {
            q: list(extended[q]) + list(cand[q])
            for q in ids
        }
        oracle = candidate_oracle(augmented, gold, ids)

        gold_rows = sum(
            1
            for q in ids
            for d in cand[q]
            if d in gold[q]
        )
        pairs = int(sizes.sum())

        evals[name] = {
            "pairs": pairs,
            "queries_with_candidates": int(np.sum(sizes > 0)),
            "mean_candidates": float(sizes.mean()),
            "median_candidates": float(np.median(sizes)),
            "p95_candidates": float(np.percentile(sizes, 95)),
            "max_candidates": int(sizes.max()),
            "rescued_outside_occurrences": len(rescued),
            "rescued_unique_queries": len(set(q for q, _ in rescued)),
            "rescued_pairs": [
                {"qid": q, "gold_doc": g} for q, g in rescued
            ],
            "acquisition_oracle": oracle,
            "acquisition_oracle_gain": oracle - base_oracle,
            "gold_rows": int(gold_rows),
            "gold_density": float(gold_rows / max(1, pairs)),
        }

        print(
            f"  {name:28s} pairs={pairs:5d} "
            f"rescue={len(rescued):2d}/{len(outside)} "
            f"oracleΔ={oracle-base_oracle:+.6f} "
            f"density={gold_rows/max(1,pairs):.5f}",
            flush=True,
        )

    # Explicitly identify the rescue(s) unique to ANY_REL relative to the
    # previously useful TOP2_CROSS_MODEL core.
    core_pairs = {
        (q, g)
        for q, g in outside
        if g in set(rows[q]["arms"].get("TOP2_CROSS_MODEL", []))
    }
    any_pairs = {
        (x["qid"], x["gold_doc"]) for x in any_rescues
    }
    tail_only = sorted(any_pairs - core_pairs)

    print("[5/5] Writing forensic report...", flush=True)
    out = root / "results/manual/huy_any_rel_tail_structure_forensics_v1"
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / "REPORT.json"

    report = {
        "schema": "manual.any_rel_tail_structure_forensics_v1",
        "status": "FORENSIC_ONLY_DO_NOT_DEPLOY",
        "warning": (
            "Multiple fixed structural subsets are compared on CAL. "
            "Use only for mechanism discovery / next preregistration."
        ),
        "baseline": {
            "candidate_oracle": base_oracle,
            "outside_pool_gold_occurrences": len(outside),
        },
        "any_rel_rescues": any_rescues,
        "any_rel_tail_only_vs_top2_cross_model": [
            {"qid": q, "gold_doc": g}
            for q, g in tail_only
        ],
        "fixed_subsets": evals,
        "next_step_rule": (
            "If the tail-only rescue belongs to a compact structural subset "
            "that retains >=6 ANY_REL rescues with a large pair reduction, "
            "design ONE new preregistered challenger-certificate arm. "
            "Otherwise treat ANY_REL sixth rescue as low-density tail noise."
        ),
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("=" * 118)
    print("STATUS: FORENSIC_ONLY_DO_NOT_DEPLOY")
    print("TAIL-ONLY vs TOP2_CROSS_MODEL:", tail_only)
    print("REPORT:", report_path)
    print("=" * 118)


if __name__ == "__main__":
    main()
