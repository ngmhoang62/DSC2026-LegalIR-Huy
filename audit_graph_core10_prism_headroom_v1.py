#!/usr/bin/env python
"""
GRAPH_CORE10 -> D1 / PRISM TOP-5 HEADROOM + PRISM WORKLOAD EXPORT
=================================================================

CPU-only. No private labels. No neural inference.

Why this experiment
-------------------
The document-anchored legal graph branch produced a real acquisition gain, but
artifact forensics show an important nuance:

  REL2_BIDIR_T3@20 rescued 6 outside-pool golds, yet every rescue at depth<=20
  had best_hop=1. The actual two-hop gold appeared only at depth 50.

Therefore we avoid treating "two-hop" as inherently superior. Instead we build
a small, conservative candidate proposal set:

  GRAPH_CORE10(q) =
      top5 REL_BIDIR_T3 candidates
      UNION
      top5 REL2_BIDIR_T3 candidates

Deduplicated, so <=10 novel docs/query.

This keeps:
  - the clean 1-hop relation ranking;
  - candidates whose ranking is strongly amplified by 2-hop path support;
while avoiding the noisy depth-20/50 tail.

This audit answers:
  1) acquisition oracle of GRAPH_CORE10;
  2) one-slot top5 oracle if only rank #5 may be replaced;
  3) same oracle on the stronger PRISM_SCORE_RANK baseline;
  4) how many graph candidates/pairs require new Prism scoring;
  5) whether the rescued candidates are genuinely new vs the old
     query-anchored legal-reference branch.

If headroom is useful, write:
  PRISM_GRAPH_CORE10_WORKLOAD.jsonl

No Prism inference occurs here.

Required local scripts/artifacts
--------------------------------
  audit_d1_document_graph_acquisition_v1.py
  audit_prism_d1_lobo_v1.py
  results/from_drive/prism/best_channel_scores.pkl  (passed explicitly)

Mandatory parity
----------------
  D1 OOF Recall@5             = 0.9569444444444444
  Prism score+rank Recall@5   = 0.9636111111111111
  D1 candidate oracle         = 0.9847222222222222

Outputs
-------
results/manual/huy_graph_core10_prism_headroom_v1/
  REPORT.json
  GRAPH_CORE10_CANDIDATES.jsonl
  PRISM_GRAPH_CORE10_WORKLOAD.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

EXPECTED_D1 = 0.9569444444444444
EXPECTED_PRISM = 0.9636111111111111
EXPECTED_ORACLE = 0.9847222222222222


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


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


def candidate_oracle(pool, gold, ids):
    return float(np.mean([
        len(set(pool[q]) & set(gold[q])) / max(1, len(gold[q]))
        for q in ids
    ]))


def merge_candidate_detail(onehop_recs, twohop_recs):
    one_map = {r["doc_id"]: r for r in onehop_recs}
    two_map = {r["doc_id"]: r for r in twohop_recs}

    order = []
    for r in onehop_recs[:5]:
        if r["doc_id"] not in order:
            order.append(r["doc_id"])
    for r in twohop_recs[:5]:
        if r["doc_id"] not in order:
            order.append(r["doc_id"])

    rows = []
    for d in order:
        a = one_map.get(d)
        b = two_map.get(d)

        def val(rec, key, default=0):
            return rec.get(key, default) if rec else default

        directions = sorted(set(
            (a.get("directions", []) if a else [])
            + (b.get("directions", []) if b else [])
        ))
        families = sorted(set(
            (a.get("families", []) if a else [])
            + (b.get("families", []) if b else [])
        ))
        refs = sorted(set(
            (a.get("refs", []) if a else [])
            + (b.get("refs", []) if b else [])
        ))

        rows.append({
            "doc_id": d,
            "rank_rel_bidir_t3": (
                next((i+1 for i, r in enumerate(onehop_recs) if r["doc_id"] == d), 60)
            ),
            "rank_rel2_bidir_t3": (
                next((i+1 for i, r in enumerate(twohop_recs) if r["doc_id"] == d), 60)
            ),
            "in_both_top5": bool(
                d in {r["doc_id"] for r in onehop_recs[:5]}
                and d in {r["doc_id"] for r in twohop_recs[:5]}
            ),
            "best_hop": min(
                val(a, "best_hop", 99),
                val(b, "best_hop", 99),
            ),
            "best_anchor_rank": min(
                val(a, "best_anchor_rank", 99),
                val(b, "best_anchor_rank", 99),
            ),
            "anchor_support_max": max(
                val(a, "anchor_support", 0),
                val(b, "anchor_support", 0),
            ),
            "edge_support_max": max(
                val(a, "edge_support", 0),
                val(b, "edge_support", 0),
            ),
            "relation_support_max": max(
                val(a, "relation_support", 0),
                val(b, "relation_support", 0),
            ),
            "header_support_max": max(
                val(a, "header_support", 0),
                val(b, "header_support", 0),
            ),
            "directions": directions,
            "families": families,
            "refs": refs,
        })

    return rows


def one_slot_oracle(base_pred, graph_candidates, gold, ids, blocks):
    pred = {}
    gain_queries = []
    neutral_gold_swaps = []

    for q in ids:
        base = list(base_pred[q])
        if len(base) != 5:
            raise RuntimeError(f"Expected baseline Top5 for {q}")

        fixed = base[:4]
        incumbent = base[4]
        choices = [incumbent] + list(graph_candidates[q])

        # Oracle objective: maximize whether slot #5 is gold.
        # Deterministic tie-break keeps incumbent first.
        chosen = next((d for d in choices if d in gold[q]), incumbent)

        pred[q] = fixed + [chosen]

        before_hit = len(set(base) & set(gold[q]))
        after_hit = len(set(pred[q]) & set(gold[q]))
        if after_hit > before_hit:
            gain_queries.append({
                "qid": q,
                "incumbent_rank5": incumbent,
                "chosen_graph_doc": chosen,
                "before_hits": before_hit,
                "after_hits": after_hit,
                "gold_count": len(gold[q]),
            })
        elif (
            chosen != incumbent
            and incumbent in gold[q]
            and chosen in gold[q]
        ):
            neutral_gold_swaps.append(q)

    m = metrics(pred, gold, ids, blocks)
    return pred, m, gain_queries, neutral_gold_swaps


def unrestricted_union_oracle(base_pred, graph_candidates, gold, ids, blocks):
    pred = {}
    for q in ids:
        pool = list(dict.fromkeys(
            list(base_pred[q]) + list(graph_candidates[q])
        ))
        # Gold first purely for oracle ceiling, then original order.
        pool.sort(key=lambda d: (0 if d in gold[q] else 1))
        pred[q] = pool[:5]
    return pred, metrics(pred, gold, ids, blocks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=Path, required=True)
    ap.add_argument("--prism-heldout-scores", type=Path, required=True)
    ap.add_argument(
        "--old-query-anchored-cases",
        type=Path,
        default=None,
        help=(
            "Optional old OUTSIDE_POOL_GOLD_RECOVERY_CASES.json; "
            "used only for overlap diagnostics."
        ),
    )
    args = ap.parse_args()

    root = args.repo_root.expanduser().resolve()
    prism_path = args.prism_heldout_scores.expanduser().resolve()
    sys.path.insert(0, str(root))

    if not prism_path.is_file():
        raise FileNotFoundError(prism_path)

    print("[1/7] Loading authoritative D1 CAL600 world...", flush=True)
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

    print("[2/7] Reconstructing exact D1 anchors...", flush=True)
    try:
        from audit_d1_document_graph_acquisition_v1 import (
            load_corpus,
            build_reference_graph,
            build_d1_rows,
            exact_oof_d1,
            graph_candidates_for_query,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Put audit_d1_document_graph_acquisition_v1.py in repo root "
            "before running this script."
        ) from exc

    rows, groups = build_d1_rows(
        local_views,
        full_channels,
        extended,
        all_ids,
        type_rows,
        cite_rows,
    )
    d1_top5, d1_full, _d1_scores, d1_recall = exact_oof_d1(
        blocks, all_ids, rows, groups, gold
    )
    print(f"  D1 Recall@5={d1_recall:.10f}", flush=True)
    if abs(d1_recall - EXPECTED_D1) > 1e-9:
        raise RuntimeError(
            f"D1 parity failed: {d1_recall} != {EXPECTED_D1}"
        )

    base_oracle = candidate_oracle(extended, gold, all_ids)
    print(f"  D1 candidate oracle={base_oracle:.10f}", flush=True)
    if abs(base_oracle - EXPECTED_ORACLE) > 1e-9:
        raise RuntimeError(
            f"Candidate oracle parity failed: {base_oracle} != {EXPECTED_ORACLE}"
        )

    print("[3/7] Reconstructing Prism score+rank OOF baseline...", flush=True)
    from audit_prism_d1_lobo_v1 import (
        load_pickle_scores,
        aligned_scores,
        lobo_arm,
    )
    raw_prism = load_pickle_scores(prism_path)
    prism, prism_floor, coverage = aligned_scores(
        raw_prism, all_ids, extended
    )
    if coverage["coverage"] < .999999:
        raise RuntimeError(
            f"Expected full Prism coverage, got {coverage}"
        )

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
    print(
        f"  Prism score+rank R={prism_res['recall']:.10f} "
        f"P={prism_res['precision']:.10f} "
        f"multi={prism_res['multi_recall']:.6f}",
        flush=True,
    )
    if abs(prism_res["recall"] - EXPECTED_PRISM) > 1e-9:
        raise RuntimeError(
            f"Prism parity failed: {prism_res['recall']} != {EXPECTED_PRISM}"
        )

    print("[4/7] Building GRAPH_CORE10 proposal set...", flush=True)
    context_dir = (
        root
        / "DSC2026-LegalIR-main/v4_run/public_test_dataset/selected-contexts"
    )
    corpus = load_corpus(context_dir)
    if len(corpus) != 8532:
        raise RuntimeError(f"Expected 8532 docs, got {len(corpus)}")

    outgoing, incoming, graph_audit = build_reference_graph(corpus)

    graph_core = {}
    graph_rows = {}
    for i, q in enumerate(all_ids, 1):
        pool = set(extended[q])
        anchors = d1_full[q][:3]

        one = graph_candidates_for_query(
            anchors=anchors,
            existing_pool=pool,
            outgoing=outgoing,
            incoming=incoming,
            relation_only=True,
            bidir=True,
            max_hops=1,
        )
        two = graph_candidates_for_query(
            anchors=anchors,
            existing_pool=pool,
            outgoing=outgoing,
            incoming=incoming,
            relation_only=True,
            bidir=True,
            max_hops=2,
        )

        merged = merge_candidate_detail(one, two)
        graph_rows[q] = merged
        graph_core[q] = [r["doc_id"] for r in merged]

        if i % 100 == 0:
            print(f"  graph core {i}/{len(all_ids)}", flush=True)

    sizes = np.asarray([len(graph_core[q]) for q in all_ids])

    augmented = {
        q: list(extended[q]) + graph_core[q]
        for q in all_ids
    }
    core_oracle = candidate_oracle(augmented, gold, all_ids)
    outside_base = {
        (q, g)
        for q in all_ids
        for g in gold[q]
        if g not in set(extended[q])
    }
    rescued_pairs = sorted(
        (q, g)
        for q, g in outside_base
        if g in set(graph_core[q])
    )

    print(
        f"  GRAPH_CORE10 pairs={int(sizes.sum())} "
        f"mean={sizes.mean():.2f} p50={np.median(sizes):.0f} "
        f"p95={np.percentile(sizes,95):.0f} max={sizes.max()}",
        flush=True,
    )
    print(
        f"  acquisition rescue={len(rescued_pairs)}/{len(outside_base)} "
        f"oracle={core_oracle:.10f} "
        f"oracleΔ={core_oracle-base_oracle:+.10f}",
        flush=True,
    )

    print("[5/7] Computing rank-5 admission oracle headroom...", flush=True)
    (
        d1_slot_pred,
        d1_slot_m,
        d1_gain_queries,
        d1_neutral,
    ) = one_slot_oracle(
        d1_top5, graph_core, gold, all_ids, blocks
    )
    (
        prism_slot_pred,
        prism_slot_m,
        prism_gain_queries,
        prism_neutral,
    ) = one_slot_oracle(
        prism_res["predictions"],
        graph_core,
        gold,
        all_ids,
        blocks,
    )

    _d1_union_pred, d1_union_m = unrestricted_union_oracle(
        d1_top5, graph_core, gold, all_ids, blocks
    )
    _prism_union_pred, prism_union_m = unrestricted_union_oracle(
        prism_res["predictions"],
        graph_core, gold, all_ids, blocks
    )

    print(
        f"  D1 one-slot oracle R={d1_slot_m['recall']:.10f} "
        f"dR={d1_slot_m['recall']-d1_recall:+.10f} "
        f"gainQ={len(d1_gain_queries)}",
        flush=True,
    )
    print(
        f"  Prism one-slot oracle R={prism_slot_m['recall']:.10f} "
        f"dR={prism_slot_m['recall']-prism_res['recall']:+.10f} "
        f"gainQ={len(prism_gain_queries)}",
        flush=True,
    )
    print(
        f"  D1 unrestricted union oracle R={d1_union_m['recall']:.10f} "
        f"dR={d1_union_m['recall']-d1_recall:+.10f}",
        flush=True,
    )
    print(
        f"  Prism unrestricted union oracle R={prism_union_m['recall']:.10f} "
        f"dR={prism_union_m['recall']-prism_res['recall']:+.10f}",
        flush=True,
    )

    print("[6/7] Gold-density / old-branch overlap diagnostics...", flush=True)
    flat_rows = []
    for q in all_ids:
        for r in graph_rows[q]:
            flat_rows.append({
                "qid": q,
                **r,
                "is_gold": r["doc_id"] in gold[q],
                "is_outside_gold": (
                    (q, r["doc_id"]) in outside_base
                ),
            })

    gold_rows = [x for x in flat_rows if x["is_gold"]]
    outside_gold_rows = [x for x in flat_rows if x["is_outside_gold"]]

    both_rows = [x for x in flat_rows if x["in_both_top5"]]
    both_gold = [x for x in both_rows if x["is_gold"]]

    old_pairs = set()
    old_path = args.old_query_anchored_cases
    if old_path is not None:
        old_path = old_path.expanduser().resolve()
        if old_path.is_file():
            old = json.loads(old_path.read_text(encoding="utf-8"))
            for x in old.get("recovered_cases", []):
                old_pairs.add((
                    str(x["qid"]),
                    str(x["recovered_gold_doc_id"]),
                ))

    rescued_set = set(rescued_pairs)
    overlap_old = sorted(rescued_set & old_pairs)
    new_vs_old = sorted(rescued_set - old_pairs)

    print(
        f"  graph rows={len(flat_rows)} goldRows={len(gold_rows)} "
        f"outsideGoldRows={len(outside_gold_rows)}",
        flush=True,
    )
    print(
        f"  in-both-top5 rows={len(both_rows)} "
        f"gold={len(both_gold)}",
        flush=True,
    )
    if old_pairs:
        print(
            f"  overlap old query-anchored={len(overlap_old)} "
            f"new-vs-old={len(new_vs_old)}",
            flush=True,
        )

    print("[7/7] Writing forensic report + Prism graph workload...", flush=True)
    out = root / "results/manual/huy_graph_core10_prism_headroom_v1"
    out.mkdir(parents=True, exist_ok=True)

    candidates_path = out / "GRAPH_CORE10_CANDIDATES.jsonl"
    with candidates_path.open("w", encoding="utf-8") as f:
        for q in all_ids:
            f.write(json.dumps({
                "qid": q,
                "question": queries[q][0],
                "d1_top5": d1_top5[q],
                "prism_top5": prism_res["predictions"][q],
                "graph_candidates": graph_rows[q],
            }, ensure_ascii=False) + "\n")

    workload_path = out / "PRISM_GRAPH_CORE10_WORKLOAD.jsonl"
    with workload_path.open("w", encoding="utf-8") as f:
        for q in all_ids:
            f.write(json.dumps({
                "qid": q,
                "question": queries[q][0],
                "candidate_doc_ids": graph_core[q],
            }, ensure_ascii=False) + "\n")

    report = {
        "schema": "manual.graph_core10_prism_headroom_v1",
        "private_labels_used": False,
        "definitions": {
            "GRAPH_CORE10": (
                "deduplicated union of top5 REL_BIDIR_T3 and "
                "top5 REL2_BIDIR_T3 novel candidates"
            ),
            "one_slot_oracle": (
                "freeze baseline top1..4; choose rank5 from baseline rank5 "
                "or GRAPH_CORE10 to maximize gold hit"
            ),
        },
        "source_artifacts": {
            "prism_scores": str(prism_path),
            "prism_scores_sha256": sha256(prism_path),
        },
        "graph_audit": graph_audit,
        "baseline": {
            "d1_recall": d1_recall,
            "prism_score_rank_recall": prism_res["recall"],
            "candidate_oracle": base_oracle,
        },
        "graph_core10": {
            "queries": len(all_ids),
            "pairs": int(sizes.sum()),
            "queries_with_candidates": int(np.sum(sizes > 0)),
            "size_stats": {
                "mean": float(sizes.mean()),
                "median": float(np.median(sizes)),
                "p95": float(np.percentile(sizes, 95)),
                "max": int(sizes.max()),
            },
            "acquisition_oracle": core_oracle,
            "acquisition_oracle_gain": core_oracle - base_oracle,
            "outside_pool_total": len(outside_base),
            "rescued_outside_pairs": [
                {"qid": q, "gold_doc": g}
                for q, g in rescued_pairs
            ],
        },
        "d1_integration_oracles": {
            "one_slot": {
                "metrics": {
                    k: v for k, v in d1_slot_m.items()
                    if k != "per_query_recall"
                },
                "delta_recall": d1_slot_m["recall"] - d1_recall,
                "gain_queries": d1_gain_queries,
                "neutral_gold_swap_qids": d1_neutral,
            },
            "unrestricted_union_top5": {
                "metrics": {
                    k: v for k, v in d1_union_m.items()
                    if k != "per_query_recall"
                },
                "delta_recall": d1_union_m["recall"] - d1_recall,
            },
        },
        "prism_integration_oracles": {
            "one_slot": {
                "metrics": {
                    k: v for k, v in prism_slot_m.items()
                    if k != "per_query_recall"
                },
                "delta_recall": (
                    prism_slot_m["recall"] - prism_res["recall"]
                ),
                "gain_queries": prism_gain_queries,
                "neutral_gold_swap_qids": prism_neutral,
            },
            "unrestricted_union_top5": {
                "metrics": {
                    k: v for k, v in prism_union_m.items()
                    if k != "per_query_recall"
                },
                "delta_recall": (
                    prism_union_m["recall"] - prism_res["recall"]
                ),
            },
        },
        "candidate_diagnostics": {
            "rows": len(flat_rows),
            "gold_rows": len(gold_rows),
            "outside_gold_rows": len(outside_gold_rows),
            "gold_density": (
                len(gold_rows) / max(1, len(flat_rows))
            ),
            "in_both_top5_rows": len(both_rows),
            "in_both_top5_gold_rows": len(both_gold),
            "in_both_top5_gold_density": (
                len(both_gold) / max(1, len(both_rows))
            ),
        },
        "old_query_anchored_overlap": {
            "available": bool(old_pairs),
            "old_recovered_pairs": [
                {"qid": q, "gold_doc": g}
                for q, g in sorted(old_pairs)
            ],
            "overlap_pairs": [
                {"qid": q, "gold_doc": g}
                for q, g in overlap_old
            ],
            "new_pairs_vs_old": [
                {"qid": q, "gold_doc": g}
                for q, g in new_vs_old
            ],
        },
        "workload": {
            "path": str(workload_path),
            "sha256": sha256(workload_path),
            "required_output": (
                "pickle dict {qid: {doc_id: float_prism_score}} "
                "for every candidate_doc_id in workload"
            ),
        },
    }

    report_path = out / "REPORT.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("=" * 116)
    print("GRAPH_CORE10 PAIRS:", int(sizes.sum()))
    print(
        "ACQUISITION:",
        f"{len(rescued_pairs)}/{len(outside_base)}",
        f"oracleΔ={core_oracle-base_oracle:+.10f}",
    )
    print(
        "D1 ONE-SLOT HEADROOM:",
        f"{d1_slot_m['recall']-d1_recall:+.10f}",
        f"gainQ={len(d1_gain_queries)}",
    )
    print(
        "PRISM ONE-SLOT HEADROOM:",
        f"{prism_slot_m['recall']-prism_res['recall']:+.10f}",
        f"gainQ={len(prism_gain_queries)}",
    )
    print("WORKLOAD:", workload_path)
    print("CANDIDATES:", candidates_path)
    print("REPORT:", report_path)
    print("=" * 116)


if __name__ == "__main__":
    main()
