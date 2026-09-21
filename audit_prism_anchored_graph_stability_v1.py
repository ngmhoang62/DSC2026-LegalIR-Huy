#!/usr/bin/env python
"""
PRISM-ANCHORED DOCUMENT GRAPH + CROSS-SEED STABILITY AUDIT V1
==============================================================

CPU-only. No neural inference. CAL labels are used only AFTER candidate
populations are frozen to disk.

Mechanism
---------
Previous document-anchored graph acquisition used exact OOF D1 top documents as
legal-graph seeds. Prism score+rank is a stronger OOF ranking signal, so this
audit asks whether changing only the GRAPH SEEDS improves acquisition.

This is NOT the closed "consensus geometry" branch:
  - consensus geometry reranked documents already inside the D1 candidate pool;
  - this experiment uses two rankings only to choose graph anchor nodes and
    acquire NEW documents outside the D1 candidate pool.

Frozen candidate families
-------------------------
D1_CORE10
    top5 REL_BIDIR_T3 union top5 REL2_BIDIR_T3 from D1 top3 seeds.

PRISM_CORE10
    same graph procedure, but seeds = Prism score+rank top3.

SEED_INTERSECTION
    candidate appears in both D1_CORE10 and PRISM_CORE10.

SEED_UNION
    deduplicated D1_CORE10 then PRISM_CORE10.

CROSS_STABLE
    candidate appears in both seed worlds AND is itself supported by both
    one-hop and two-hop graph rankings in at least one seed world.

DUAL_DIRECTION
    candidate is supported by both INCOMING and OUTGOING legal-reference edges
    in at least one seed world.

STABLE_OR_DUAL_DIRECTION
    SEED_INTERSECTION union DUAL_DIRECTION, preserving SEED_UNION order.

Mandatory parity
----------------
D1 Recall@5           = 0.9569444444444444
Prism score+rank R@5  = 0.9636111111111111
D1 candidate oracle   = 0.9847222222222222

Outputs
-------
results/manual/huy_prism_anchored_graph_stability_v1/
  SEALED_CANDIDATES.jsonl
  SEALED_SUMMARY.json
  REPORT.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np

EXPECTED_D1 = 0.9569444444444444
EXPECTED_PRISM = 0.9636111111111111
EXPECTED_ORACLE = 0.9847222222222222

ARMS = [
    "D1_CORE10",
    "PRISM_CORE10",
    "SEED_INTERSECTION",
    "SEED_UNION",
    "CROSS_STABLE",
    "DUAL_DIRECTION",
    "STABLE_OR_DUAL_DIRECTION",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def candidate_oracle(pool, gold, ids):
    return float(np.mean([
        len(set(pool[q]) & set(gold[q])) / max(1, len(gold[q]))
        for q in ids
    ]))


def metrics(pred, gold, ids, blocks):
    per_r = {}
    per_p = {}
    for q in ids:
        hit = len(set(pred[q]) & set(gold[q]))
        per_r[q] = hit / max(1, len(gold[q]))
        per_p[q] = hit / 5.0
    return {
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
        "per_query_recall": per_r,
    }


def one_slot_oracle(base, candidates, gold, ids, blocks):
    out = {}
    gains = []
    for q in ids:
        before = list(base[q])
        if len(before) != 5:
            raise RuntimeError(f"Expected Top5 for {q}")
        incumbent = before[4]
        choices = [incumbent] + list(candidates[q])
        chosen = next((d for d in choices if d in gold[q]), incumbent)
        after = before[:4] + [chosen]
        out[q] = after

        hb = len(set(before) & set(gold[q]))
        ha = len(set(after) & set(gold[q]))
        if ha > hb:
            gains.append({
                "qid": q,
                "defender": incumbent,
                "challenger": chosen,
                "gold_count": len(gold[q]),
                "before_hits": hb,
                "after_hits": ha,
            })
    return metrics(out, gold, ids, blocks), gains


def unrestricted_union_oracle(base, candidates, gold, ids, blocks):
    out = {}
    for q in ids:
        pool = list(dict.fromkeys(list(base[q]) + list(candidates[q])))
        gold_first = [d for d in pool if d in gold[q]]
        rest = [d for d in pool if d not in gold[q]]
        out[q] = (gold_first + rest)[:5]
    return metrics(out, gold, ids, blocks)


def build_core(
    anchors,
    existing_pool,
    outgoing,
    incoming,
    graph_candidates_for_query,
    merge_candidate_detail,
):
    one = graph_candidates_for_query(
        anchors=anchors,
        existing_pool=existing_pool,
        outgoing=outgoing,
        incoming=incoming,
        relation_only=True,
        bidir=True,
        max_hops=1,
    )
    two = graph_candidates_for_query(
        anchors=anchors,
        existing_pool=existing_pool,
        outgoing=outgoing,
        incoming=incoming,
        relation_only=True,
        bidir=True,
        max_hops=2,
    )
    return merge_candidate_detail(one, two)


def row_map(rows):
    return {str(r["doc_id"]): r for r in rows}


def directions_of(row):
    return set(row.get("directions", []) if row else [])


def merge_seed_worlds(d1_rows, prism_rows):
    d1m = row_map(d1_rows)
    pm = row_map(prism_rows)

    order = []
    for r in d1_rows:
        d = str(r["doc_id"])
        if d not in order:
            order.append(d)
    for r in prism_rows:
        d = str(r["doc_id"])
        if d not in order:
            order.append(d)

    merged = []
    for d in order:
        a = d1m.get(d)
        b = pm.get(d)
        dirs = directions_of(a) | directions_of(b)

        in_d1 = a is not None
        in_prism = b is not None
        stable_graph_rank = bool(
            (a and a.get("in_both_top5", False))
            or (b and b.get("in_both_top5", False))
        )
        dual_direction = "INCOMING" in dirs and "OUTGOING" in dirs

        merged.append({
            "doc_id": d,
            "in_d1_core10": in_d1,
            "in_prism_core10": in_prism,
            "cross_seed": bool(in_d1 and in_prism),
            "stable_graph_rank": stable_graph_rank,
            "cross_stable": bool(in_d1 and in_prism and stable_graph_rank),
            "dual_direction": dual_direction,
            "directions": sorted(dirs),
            "d1": a,
            "prism": b,
        })
    return merged


def arm_docs(merged):
    return {
        "D1_CORE10": [r["doc_id"] for r in merged if r["in_d1_core10"]],
        "PRISM_CORE10": [r["doc_id"] for r in merged if r["in_prism_core10"]],
        "SEED_INTERSECTION": [r["doc_id"] for r in merged if r["cross_seed"]],
        "SEED_UNION": [r["doc_id"] for r in merged],
        "CROSS_STABLE": [r["doc_id"] for r in merged if r["cross_stable"]],
        "DUAL_DIRECTION": [r["doc_id"] for r in merged if r["dual_direction"]],
        "STABLE_OR_DUAL_DIRECTION": [
            r["doc_id"] for r in merged
            if r["cross_seed"] or r["dual_direction"]
        ],
    }


def compact_metrics(m):
    return {k: v for k, v in m.items() if k != "per_query_recall"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=Path, required=True)
    ap.add_argument("--prism-heldout-scores", type=Path, required=True)
    args = ap.parse_args()

    root = args.repo_root.expanduser().resolve()
    prism_path = args.prism_heldout_scores.expanduser().resolve()
    sys.path.insert(0, str(root))

    if not prism_path.is_file():
        raise FileNotFoundError(prism_path)

    print("[1/8] Loading authoritative CAL world...", flush=True)
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

    print("[2/8] Reconstructing exact OOF D1...", flush=True)
    from audit_d1_document_graph_acquisition_v1 import (
        load_corpus,
        build_reference_graph,
        build_d1_rows,
        exact_oof_d1,
        graph_candidates_for_query,
    )
    from audit_graph_core10_prism_headroom_v1 import merge_candidate_detail

    rows, groups = build_d1_rows(
        local_views, full_channels, extended, all_ids, type_rows, cite_rows
    )
    d1_top5, d1_full, _d1_scores, d1_recall = exact_oof_d1(
        blocks, all_ids, rows, groups, gold
    )
    print(f"  D1 R={d1_recall:.10f}", flush=True)
    if abs(d1_recall - EXPECTED_D1) > 1e-9:
        raise RuntimeError(f"D1 parity failed: {d1_recall} != {EXPECTED_D1}")

    base_oracle = candidate_oracle(extended, gold, all_ids)
    print(f"  candidate oracle={base_oracle:.10f}", flush=True)
    if abs(base_oracle - EXPECTED_ORACLE) > 1e-9:
        raise RuntimeError(
            f"Candidate oracle parity failed: {base_oracle} != {EXPECTED_ORACLE}"
        )

    print("[3/8] Reconstructing Prism score+rank OOF...", flush=True)
    from audit_prism_d1_lobo_v1 import (
        load_pickle_scores,
        aligned_scores,
        lobo_arm,
    )
    raw_prism = load_pickle_scores(prism_path)
    prism, _floor, coverage = aligned_scores(raw_prism, all_ids, extended)
    if coverage["coverage"] < .999999:
        raise RuntimeError(f"Prism coverage failed: {coverage}")

    prism_res = lobo_arm(
        arm="prism_score_rank",
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
    prism_top5 = {
        q: [str(d) for d in prism_res["predictions"][q]]
        for q in all_ids
    }
    print(
        f"  Prism R={prism_res['recall']:.10f} "
        f"P={prism_res['precision']:.10f}",
        flush=True,
    )
    if abs(prism_res["recall"] - EXPECTED_PRISM) > 1e-9:
        raise RuntimeError(
            f"Prism parity failed: {prism_res['recall']} != {EXPECTED_PRISM}"
        )

    print("[4/8] Building legal graph once...", flush=True)
    context_dir = (
        root
        / "DSC2026-LegalIR-main/v4_run/public_test_dataset/selected-contexts"
    )
    corpus = load_corpus(context_dir)
    if len(corpus) != 8532:
        raise RuntimeError(f"Expected 8532 corpus docs, got {len(corpus)}")
    outgoing, incoming, graph_audit = build_reference_graph(corpus)

    print("[5/8] Freezing D1-seeded and Prism-seeded graph populations...", flush=True)
    frozen = {}
    arm_candidates = {a: {} for a in ARMS}

    for i, q in enumerate(all_ids, 1):
        pool = set(extended[q])
        d1_anchors = [str(d) for d in d1_full[q][:3]]
        prism_anchors = [str(d) for d in prism_top5[q][:3]]

        d1_core = build_core(
            d1_anchors, pool, outgoing, incoming,
            graph_candidates_for_query, merge_candidate_detail,
        )
        prism_core = build_core(
            prism_anchors, pool, outgoing, incoming,
            graph_candidates_for_query, merge_candidate_detail,
        )
        merged = merge_seed_worlds(d1_core, prism_core)
        arms = arm_docs(merged)

        for arm in ARMS:
            arm_candidates[arm][q] = arms[arm]

        frozen[q] = {
            "qid": q,
            "question": queries[q][0],
            "d1_top5": [str(d) for d in d1_top5[q]],
            "prism_top5": prism_top5[q],
            "d1_anchor_top3": d1_anchors,
            "prism_anchor_top3": prism_anchors,
            "anchor_top3_overlap": len(set(d1_anchors) & set(prism_anchors)),
            "candidates": merged,
            "arms": arms,
        }

        if i % 100 == 0:
            print(f"  frozen {i}/{len(all_ids)}", flush=True)

    out = root / "results/manual/huy_prism_anchored_graph_stability_v1"
    out.mkdir(parents=True, exist_ok=True)

    sealed_path = out / "SEALED_CANDIDATES.jsonl"
    with sealed_path.open("w", encoding="utf-8") as f:
        for q in all_ids:
            f.write(json.dumps(frozen[q], ensure_ascii=False) + "\n")

    arm_size_summary = {}
    for arm in ARMS:
        sizes = np.asarray([len(arm_candidates[arm][q]) for q in all_ids])
        arm_size_summary[arm] = {
            "pairs": int(sizes.sum()),
            "queries_with_candidates": int(np.sum(sizes > 0)),
            "mean": float(sizes.mean()),
            "median": float(np.median(sizes)),
            "p95": float(np.percentile(sizes, 95)),
            "max": int(sizes.max()),
        }

    anchor_overlap = np.asarray([
        frozen[q]["anchor_top3_overlap"] for q in all_ids
    ])
    sealed_summary = {
        "schema": "manual.prism_anchored_graph_stability_seal.v1",
        "gold_used_for_candidate_construction": False,
        "prism_scores_path": str(prism_path),
        "prism_scores_sha256": sha256(prism_path),
        "graph_audit": graph_audit,
        "anchor_top3_overlap": {
            "mean": float(anchor_overlap.mean()),
            "median": float(np.median(anchor_overlap)),
            "zero": int(np.sum(anchor_overlap == 0)),
            "one": int(np.sum(anchor_overlap == 1)),
            "two": int(np.sum(anchor_overlap == 2)),
            "three": int(np.sum(anchor_overlap == 3)),
        },
        "arms": arm_size_summary,
        "sealed_candidates": str(sealed_path),
        "sealed_candidates_sha256": sha256(sealed_path),
    }
    seal_summary_path = out / "SEALED_SUMMARY.json"
    seal_summary_path.write_text(
        json.dumps(sealed_summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(
        "  sealed:",
        ", ".join(
            f"{a}={arm_size_summary[a]['pairs']}"
            for a in ARMS
        ),
        flush=True,
    )

    print("[6/8] Revealing CAL gold for acquisition diagnostics...", flush=True)
    outside = {
        (q, g)
        for q in all_ids
        for g in gold[q]
        if g not in set(extended[q])
    }

    evals = OrderedDict()
    for arm in ARMS:
        cand = arm_candidates[arm]
        augmented = {
            q: list(extended[q]) + list(cand[q])
            for q in all_ids
        }
        acq_oracle = candidate_oracle(augmented, gold, all_ids)
        rescued = sorted(
            (q, g)
            for q, g in outside
            if g in set(cand[q])
        )
        unique_q = sorted(set(q for q, _ in rescued))

        flat_pairs = sum(len(cand[q]) for q in all_ids)
        gold_rows = sum(
            1
            for q in all_ids
            for d in cand[q]
            if d in gold[q]
        )

        prism_slot_m, prism_gains = one_slot_oracle(
            prism_top5, cand, gold, all_ids, blocks
        )
        d1_slot_m, d1_gains = one_slot_oracle(
            d1_top5, cand, gold, all_ids, blocks
        )
        prism_union_m = unrestricted_union_oracle(
            prism_top5, cand, gold, all_ids, blocks
        )

        evals[arm] = {
            "candidate_stats": arm_size_summary[arm],
            "acquisition_oracle": acq_oracle,
            "acquisition_oracle_gain": acq_oracle - base_oracle,
            "rescued_outside_occurrences": len(rescued),
            "rescued_unique_queries": len(unique_q),
            "rescued_pairs": [
                {"qid": q, "gold_doc": g}
                for q, g in rescued
            ],
            "gold_rows": int(gold_rows),
            "gold_density": float(gold_rows / max(1, flat_pairs)),
            "d1_one_slot": {
                "metrics": compact_metrics(d1_slot_m),
                "delta_recall": d1_slot_m["recall"] - d1_recall,
                "gain_queries": d1_gains,
            },
            "prism_one_slot": {
                "metrics": compact_metrics(prism_slot_m),
                "delta_recall": prism_slot_m["recall"] - prism_res["recall"],
                "gain_queries": prism_gains,
            },
            "prism_unrestricted_union": {
                "metrics": compact_metrics(prism_union_m),
                "delta_recall": (
                    prism_union_m["recall"] - prism_res["recall"]
                ),
            },
        }

        print(
            f"  {arm:28s} pairs={flat_pairs:4d} "
            f"rescue={len(rescued):2d}/{len(outside)} "
            f"acqΔ={acq_oracle-base_oracle:+.6f} "
            f"PrismSlotΔ={prism_slot_m['recall']-prism_res['recall']:+.6f} "
            f"goldDensity={gold_rows/max(1,flat_pairs):.5f}",
            flush=True,
        )

    print("[7/8] Cross-world rescue forensics...", flush=True)
    d1_resc = {
        (x["qid"], x["gold_doc"])
        for x in evals["D1_CORE10"]["rescued_pairs"]
    }
    p_resc = {
        (x["qid"], x["gold_doc"])
        for x in evals["PRISM_CORE10"]["rescued_pairs"]
    }
    only_d1 = sorted(d1_resc - p_resc)
    only_prism = sorted(p_resc - d1_resc)
    both = sorted(d1_resc & p_resc)

    forensics = {
        "rescued_by_both": [
            {"qid": q, "gold_doc": g} for q, g in both
        ],
        "rescued_only_d1_seed": [
            {"qid": q, "gold_doc": g} for q, g in only_d1
        ],
        "rescued_only_prism_seed": [
            {"qid": q, "gold_doc": g} for q, g in only_prism
        ],
    }

    promoted = []
    for arm in ARMS:
        e = evals[arm]
        if (
            e["rescued_unique_queries"] >= 3
            and e["prism_one_slot"]["delta_recall"] >= 0.0025 - 1e-12
        ):
            promoted.append(arm)

    print("[8/8] Writing report...", flush=True)
    report = {
        "schema": "manual.prism_anchored_graph_stability_v1",
        "private_labels_used": False,
        "candidate_populations_sealed_before_gold_diagnostics": True,
        "baselines": {
            "d1_recall": d1_recall,
            "prism_score_rank_recall": prism_res["recall"],
            "candidate_oracle": base_oracle,
            "outside_pool_gold_occurrences": len(outside),
        },
        "sealed_summary": sealed_summary,
        "arms": evals,
        "cross_world_forensics": forensics,
        "decision": {
            "promoted_arms": promoted,
            "status": (
                "PROMISING_GRAPH_STABILITY_ARM"
                if promoted else
                "NO_STRUCTURAL_ARM_CLEARS_GATE"
            ),
            "gate": (
                "rescued_unique_queries>=3 AND "
                "Prism one-slot delta Recall>=+0.0025"
            ),
            "next_if_promoted": (
                "Do NOT insert candidates directly. Export only promoted-arm "
                "challengers for a small semantic challenger-vs-rank5 "
                "certificate (Prism/BGE), preferably without fitting a large "
                "selector."
            ),
        },
        "artifacts": {
            "sealed_candidates": str(sealed_path),
            "sealed_candidates_sha256": sha256(sealed_path),
            "sealed_summary": str(seal_summary_path),
            "sealed_summary_sha256": sha256(seal_summary_path),
        },
    }

    report_path = out / "REPORT.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("=" * 120)
    print("STATUS:", report["decision"]["status"])
    print("PROMOTED:", promoted)
    print("D1-only rescues:", only_d1)
    print("Prism-only rescues:", only_prism)
    print("Both:", both)
    print("SEALED:", sealed_path)
    print("REPORT:", report_path)
    print("=" * 120)


if __name__ == "__main__":
    main()
