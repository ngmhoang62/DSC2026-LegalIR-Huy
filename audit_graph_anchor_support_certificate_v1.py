#!/usr/bin/env python
"""
LEGAL GRAPH ANCHOR-SUPPORT CERTIFICATE AUDIT V1
================================================

CPU-only. No neural inference.

Goal
----
The current GRAPH_CORE10 ranking aggregates edge counts but does not preserve
the identity of every anchor document that independently points to a challenger.
This audit builds a cleaner one-hop relation-specific graph certificate from the
union of D1 and Prism top-4 documents.

A candidate is interesting when MULTIPLE independently high-ranked legal
documents point to it. That is a different mechanism from:
  - REL_L0 (rank5 weakness);
  - closed score/rank consensus reranking;
  - query-anchored explicit citation expansion.

Candidate source
----------------
For each query:
  anchors = D1 top4 UNION Prism top4
  enumerate one-hop relation-specific INCOMING + OUTGOING neighbors
  keep only docs outside the original D1 candidate pool

Frozen arms
-----------
ANY_REL
    every novel relation-specific one-hop candidate.

SUPPORT_2PLUS
    candidate supported by >=2 distinct anchor document IDs.

CROSS_MODEL_SUPPORT
    candidate supported by >=1 D1 anchor and >=1 Prism anchor.

TOP2_CROSS_MODEL
    CROSS_MODEL_SUPPORT and at least one supporting anchor is top2 in D1 or
    top2 in Prism.

DUAL_DIRECTION
    candidate receives both incoming and outgoing support.

CROSS_OR_MULTI
    CROSS_MODEL_SUPPORT or SUPPORT_2PLUS.

These are all label-free structural certificates; no threshold is tuned on CAL.

Mandatory parity
----------------
D1 Recall@5           = 0.9569444444444444
Prism score+rank R@5  = 0.9636111111111111
D1 candidate oracle   = 0.9847222222222222

Outputs
-------
results/manual/huy_graph_anchor_support_certificate_v1/
  SEALED_CANDIDATES.jsonl
  REPORT.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict, OrderedDict
from pathlib import Path

import numpy as np

EXPECTED_D1 = 0.9569444444444444
EXPECTED_PRISM = 0.9636111111111111
EXPECTED_ORACLE = 0.9847222222222222

ARMS = [
    "ANY_REL",
    "SUPPORT_2PLUS",
    "CROSS_MODEL_SUPPORT",
    "TOP2_CROSS_MODEL",
    "DUAL_DIRECTION",
    "CROSS_OR_MULTI",
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
    pred = {}
    gains = []
    for q in ids:
        before = list(base[q])
        incumbent = before[4]
        choices = [incumbent] + list(candidates[q])
        chosen = next((d for d in choices if d in gold[q]), incumbent)
        after = before[:4] + [chosen]
        pred[q] = after

        hb = len(set(before) & set(gold[q]))
        ha = len(set(after) & set(gold[q]))
        if ha > hb:
            gains.append({
                "qid": q,
                "defender": incumbent,
                "challenger": chosen,
                "gold_count": len(gold[q]),
            })
    return metrics(pred, gold, ids, blocks), gains


def enumerate_anchor_neighbors(
    *,
    d1_top5,
    prism_top5,
    existing_pool,
    outgoing,
    incoming,
):
    anchors = {}
    for rank, d in enumerate(d1_top5[:4], 1):
        anchors.setdefault(str(d), {
            "d1_rank": 99,
            "prism_rank": 99,
        })
        anchors[str(d)]["d1_rank"] = rank
    for rank, d in enumerate(prism_top5[:4], 1):
        anchors.setdefault(str(d), {
            "d1_rank": 99,
            "prism_rank": 99,
        })
        anchors[str(d)]["prism_rank"] = rank

    recs = {}

    def ensure(doc):
        return recs.setdefault(str(doc), {
            "doc_id": str(doc),
            "supporting_anchors": {},
            "directions": set(),
            "families": set(),
            "refs": set(),
            "edge_support": 0,
            "relation_support": 0,
            "header_support": 0,
        })

    for anchor_id, meta in anchors.items():
        edge_rows = []
        for e in outgoing.get(anchor_id, []):
            if e.get("is_relation"):
                edge_rows.append((str(e["dst"]), e, "OUTGOING"))
        for e in incoming.get(anchor_id, []):
            if e.get("is_relation"):
                edge_rows.append((str(e["src"]), e, "INCOMING"))

        for dst, edge, direction in edge_rows:
            if dst in existing_pool or dst in anchors:
                continue
            r = ensure(dst)
            r["supporting_anchors"][anchor_id] = {
                "d1_rank": int(meta["d1_rank"]),
                "prism_rank": int(meta["prism_rank"]),
            }
            r["directions"].add(direction)
            r["families"].update(edge.get("families", []))
            r["refs"].update(edge.get("refs", []))
            r["edge_support"] += int(edge.get("mentions", 0))
            r["relation_support"] += int(edge.get("relation_mentions", 0))
            r["header_support"] += int(edge.get("header_mentions", 0))

    out = []
    for d, r in recs.items():
        support = list(r["supporting_anchors"].values())
        d1_supported = any(x["d1_rank"] < 99 for x in support)
        prism_supported = any(x["prism_rank"] < 99 for x in support)
        best_d1 = min((x["d1_rank"] for x in support), default=99)
        best_prism = min((x["prism_rank"] for x in support), default=99)
        dirs = sorted(r["directions"])

        row = {
            "doc_id": d,
            "anchor_support": len(r["supporting_anchors"]),
            "supporting_anchors": r["supporting_anchors"],
            "d1_supported": bool(d1_supported),
            "prism_supported": bool(prism_supported),
            "cross_model_support": bool(d1_supported and prism_supported),
            "best_d1_anchor_rank": int(best_d1),
            "best_prism_anchor_rank": int(best_prism),
            "top2_cross_model": bool(
                d1_supported and prism_supported
                and (best_d1 <= 2 or best_prism <= 2)
            ),
            "dual_direction": bool(
                "INCOMING" in dirs and "OUTGOING" in dirs
            ),
            "directions": dirs,
            "families": sorted(r["families"]),
            "refs": sorted(r["refs"]),
            "edge_support": int(r["edge_support"]),
            "relation_support": int(r["relation_support"]),
            "header_support": int(r["header_support"]),
        }
        out.append(row)

    # Fully label-free order: strongest independent support first, then
    # high-ranked anchors, then relation/header evidence.
    out.sort(key=lambda r: (
        -int(r["cross_model_support"]),
        -int(r["anchor_support"]),
        min(r["best_d1_anchor_rank"], r["best_prism_anchor_rank"]),
        -int(r["dual_direction"]),
        -int(r["relation_support"]),
        -int(r["header_support"]),
        -int(r["edge_support"]),
        str(r["doc_id"]),
    ))
    return out


def arm_docs(rows):
    return {
        "ANY_REL": [r["doc_id"] for r in rows],
        "SUPPORT_2PLUS": [
            r["doc_id"] for r in rows if r["anchor_support"] >= 2
        ],
        "CROSS_MODEL_SUPPORT": [
            r["doc_id"] for r in rows if r["cross_model_support"]
        ],
        "TOP2_CROSS_MODEL": [
            r["doc_id"] for r in rows if r["top2_cross_model"]
        ],
        "DUAL_DIRECTION": [
            r["doc_id"] for r in rows if r["dual_direction"]
        ],
        "CROSS_OR_MULTI": [
            r["doc_id"] for r in rows
            if r["cross_model_support"] or r["anchor_support"] >= 2
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

    print("[1/7] Loading CAL + exact D1...", flush=True)
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

    from audit_d1_document_graph_acquisition_v1 import (
        load_corpus,
        build_reference_graph,
        build_d1_rows,
        exact_oof_d1,
    )

    rows, groups = build_d1_rows(
        local_views, full_channels, extended, all_ids, type_rows, cite_rows
    )
    d1_top5, _d1_full, _d1_scores, d1_recall = exact_oof_d1(
        blocks, all_ids, rows, groups, gold
    )
    if abs(d1_recall - EXPECTED_D1) > 1e-9:
        raise RuntimeError(f"D1 parity failed: {d1_recall}")

    base_oracle = candidate_oracle(extended, gold, all_ids)
    if abs(base_oracle - EXPECTED_ORACLE) > 1e-9:
        raise RuntimeError(f"Candidate oracle parity failed: {base_oracle}")

    print("[2/7] Reconstructing Prism score+rank OOF...", flush=True)
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
    if abs(prism_res["recall"] - EXPECTED_PRISM) > 1e-9:
        raise RuntimeError(f"Prism parity failed: {prism_res['recall']}")

    prism_top5 = {
        q: [str(d) for d in prism_res["predictions"][q]]
        for q in all_ids
    }
    d1_top5 = {
        q: [str(d) for d in d1_top5[q]]
        for q in all_ids
    }

    print("[3/7] Building legal graph...", flush=True)
    context_dir = (
        root
        / "DSC2026-LegalIR-main/v4_run/public_test_dataset/selected-contexts"
    )
    corpus = load_corpus(context_dir)
    if len(corpus) != 8532:
        raise RuntimeError(f"Expected 8532 docs, got {len(corpus)}")
    outgoing, incoming, graph_audit = build_reference_graph(corpus)

    print("[4/7] Sealing anchor-support candidates BEFORE gold diagnostics...", flush=True)
    frozen = {}
    arm_candidates = {a: {} for a in ARMS}

    for i, q in enumerate(all_ids, 1):
        rows_q = enumerate_anchor_neighbors(
            d1_top5=d1_top5[q],
            prism_top5=prism_top5[q],
            existing_pool=set(extended[q]),
            outgoing=outgoing,
            incoming=incoming,
        )
        arms = arm_docs(rows_q)
        for arm in ARMS:
            arm_candidates[arm][q] = arms[arm]

        frozen[q] = {
            "qid": q,
            "question": queries[q][0],
            "d1_top5": d1_top5[q],
            "prism_top5": prism_top5[q],
            "candidate_details": rows_q,
            "arms": arms,
        }

        if i % 100 == 0:
            print(f"  sealed {i}/{len(all_ids)}", flush=True)

    out = root / "results/manual/huy_graph_anchor_support_certificate_v1"
    out.mkdir(parents=True, exist_ok=True)

    sealed_path = out / "SEALED_CANDIDATES.jsonl"
    with sealed_path.open("w", encoding="utf-8") as f:
        for q in all_ids:
            f.write(json.dumps(frozen[q], ensure_ascii=False) + "\n")

    print("[5/7] Revealing CAL gold and measuring acquisition...", flush=True)
    outside = {
        (q, g)
        for q in all_ids
        for g in gold[q]
        if g not in set(extended[q])
    }

    evals = OrderedDict()
    for arm in ARMS:
        cand = arm_candidates[arm]
        sizes = np.asarray([len(cand[q]) for q in all_ids])
        augmented = {
            q: list(extended[q]) + list(cand[q])
            for q in all_ids
        }
        oracle = candidate_oracle(augmented, gold, all_ids)
        rescued = sorted(
            (q, g) for q, g in outside if g in set(cand[q])
        )
        flat_pairs = int(sizes.sum())
        gold_rows = sum(
            1 for q in all_ids for d in cand[q] if d in gold[q]
        )

        prism_slot, gains = one_slot_oracle(
            prism_top5, cand, gold, all_ids, blocks
        )

        evals[arm] = {
            "pairs": flat_pairs,
            "queries_with_candidates": int(np.sum(sizes > 0)),
            "mean_candidates": float(sizes.mean()),
            "median_candidates": float(np.median(sizes)),
            "p95_candidates": float(np.percentile(sizes, 95)),
            "max_candidates": int(sizes.max()),
            "acquisition_oracle": oracle,
            "acquisition_oracle_gain": oracle - base_oracle,
            "rescued_outside_occurrences": len(rescued),
            "rescued_unique_queries": len(set(q for q, _ in rescued)),
            "rescued_pairs": [
                {"qid": q, "gold_doc": g} for q, g in rescued
            ],
            "gold_rows": int(gold_rows),
            "gold_density": float(gold_rows / max(1, flat_pairs)),
            "prism_one_slot": {
                "metrics": compact_metrics(prism_slot),
                "delta_recall": prism_slot["recall"] - prism_res["recall"],
                "gain_queries": gains,
            },
        }

        print(
            f"  {arm:22s} pairs={flat_pairs:5d} "
            f"rescue={len(rescued):2d}/{len(outside)} "
            f"acqΔ={oracle-base_oracle:+.6f} "
            f"PrismSlotΔ={prism_slot['recall']-prism_res['recall']:+.6f} "
            f"density={gold_rows/max(1,flat_pairs):.5f}",
            flush=True,
        )

    print("[6/7] Structural rescue forensics...", flush=True)
    rescue_membership = defaultdict(list)
    for arm in ARMS:
        for x in evals[arm]["rescued_pairs"]:
            rescue_membership[(x["qid"], x["gold_doc"])].append(arm)

    rescue_rows = [
        {
            "qid": q,
            "gold_doc": g,
            "arms": arms,
            "candidate_detail": next(
                (
                    r for r in frozen[q]["candidate_details"]
                    if r["doc_id"] == g
                ),
                None,
            ),
        }
        for (q, g), arms in sorted(rescue_membership.items())
    ]

    promoted = []
    for arm in ARMS:
        e = evals[arm]
        if (
            e["rescued_unique_queries"] >= 3
            and e["prism_one_slot"]["delta_recall"] >= 0.0025 - 1e-12
        ):
            promoted.append(arm)

    print("[7/7] Writing report...", flush=True)
    report = {
        "schema": "manual.graph_anchor_support_certificate_v1",
        "private_labels_used": False,
        "candidate_population_sealed_before_gold_diagnostics": True,
        "baselines": {
            "d1_recall": d1_recall,
            "prism_score_rank_recall": prism_res["recall"],
            "candidate_oracle": base_oracle,
            "outside_pool_gold_occurrences": len(outside),
        },
        "graph_audit": graph_audit,
        "arms": evals,
        "rescue_forensics": rescue_rows,
        "decision": {
            "status": (
                "PROMISING_ANCHOR_CERTIFICATE"
                if promoted else
                "NO_ANCHOR_CERTIFICATE_CLEARS_GATE"
            ),
            "promoted_arms": promoted,
            "gate": (
                "rescued_unique_queries>=3 AND "
                "Prism one-slot delta Recall>=+0.0025"
            ),
            "next_if_promoted": (
                "Use promoted structural arm as a small challenger workload. "
                "Then score challenger vs Prism rank5 with a semantic model; "
                "do not fit a high-capacity selector on CAL."
            ),
        },
        "artifacts": {
            "sealed_candidates": str(sealed_path),
            "sealed_candidates_sha256": sha256(sealed_path),
            "prism_scores": str(prism_path),
            "prism_scores_sha256": sha256(prism_path),
        },
    }

    report_path = out / "REPORT.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("=" * 116)
    print("STATUS:", report["decision"]["status"])
    print("PROMOTED:", promoted)
    print("SEALED:", sealed_path)
    print("REPORT:", report_path)
    print("=" * 116)


if __name__ == "__main__":
    main()
