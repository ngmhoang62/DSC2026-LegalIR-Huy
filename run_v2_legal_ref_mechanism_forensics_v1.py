#!/usr/bin/env python
"""
STRICT-V2 QUERY-ANCHORED LEGAL-REF MECHANISM FORENSICS V1
=========================================================

Purpose
-------
Use ONLY the already-sealed Strict-V2 legal-reference expansion artifact to
decompose its four known outside-pool recoveries by mechanism:

- DIRECT_REFERENCE_MATCH
- RELATION_NEIGHBOR
  - relation_family
  - relation_direction
  - header/body

This script does NOT read CAL600 gold and does NOT modify any policy.
It verifies the V2 additions seal first, then reveals canonical V2 gold for
post-seal forensic analysis.

It also computes candidate-recall oracle deltas for:
- direct-only additions
- relation-only additions
- full frozen generator

These mechanism-level V2 results can be used to decide whether a relation-only
policy is scientifically supportable before any CAL/public test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


EXPECTED_V2_Q = 6991
EXPECTED_TRIGGERED = 24
EXPECTED_ADDITIONS = 99
EXPECTED_FULL_RECOVERIES = 4


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def macro_candidate_recall(qids, pools, additions, gold):
    vals = []
    for q in qids:
        cand = set(pools[q]) | set(additions.get(q, []))
        g = gold[q]
        vals.append(len(cand & g) / max(1, len(g)))
    return float(np.mean(vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=Path, required=True)
    args = ap.parse_args()

    root = args.repo_root.resolve()

    exp = root / "results/gemini/huy_d1_query_anchored_legal_ref_expansion_v1"
    additions_path = exp / "V2_PER_QUERY_EXPANSION.jsonl"
    seal_path = exp / "V2_PER_QUERY_EXPANSION_SEAL.json"
    expected_report_path = exp / "V2_SHADOW_EXPANSION_RESULTS.json"

    queries_path = (
        root
        / "cache/research_v2_e5_confirmation/bundle-v1/"
        "V2_TRANSFER_QUERIES.jsonl"
    )
    pool_path = (
        root
        / "results/research_v2_forensic/"
        "V2_CANDIDATE_POOL.jsonl"
    )

    for p in (
        additions_path,
        seal_path,
        expected_report_path,
        queries_path,
        pool_path,
    ):
        if not p.is_file():
            raise FileNotFoundError(p)

    print("[1/5] Verifying sealed Strict-V2 additions artifact...")
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    actual_sha = sha256(additions_path)
    expected_sha = seal.get("generated_additions_artifact_sha256")
    if actual_sha != expected_sha:
        raise RuntimeError(
            f"V2 additions seal mismatch: actual={actual_sha} expected={expected_sha}"
        )

    rows = {}
    for r in read_jsonl(additions_path):
        q = str(r["qid"])
        rows[q] = r

    if len(rows) != EXPECTED_V2_Q:
        raise RuntimeError(f"V2 additions population={len(rows)} != {EXPECTED_V2_Q}")

    triggered = [q for q, r in rows.items() if r.get("newly_added_doc_ids")]
    total_additions = sum(len(r.get("newly_added_doc_ids", [])) for r in rows.values())

    print(
        f"  seal PASS | q={len(rows)} | triggered={len(triggered)} "
        f"| additions={total_additions}",
        flush=True,
    )

    if len(triggered) != EXPECTED_TRIGGERED or total_additions != EXPECTED_ADDITIONS:
        raise RuntimeError(
            "Historical V2 generator counts drifted: "
            f"triggered={len(triggered)} additions={total_additions}"
        )

    print("[2/5] Loading canonical V2 pool + post-seal gold labels...")
    pools = {
        str(r["qid"]): [str(d) for d in r["doc_ids"]]
        for r in read_jsonl(pool_path)
    }

    questions = {}
    gold = {}
    qids = []
    for r in read_jsonl(queries_path):
        q = str(r["qid"])
        qids.append(q)
        questions[q] = str(r.get("question", ""))
        gold[q] = {str(d) for d in r.get("gold", [])}

    if len(qids) != EXPECTED_V2_Q:
        raise RuntimeError(f"V2 canonical query population={len(qids)}")
    if set(qids) != set(rows) or set(qids) != set(pools) or set(qids) != set(gold):
        raise RuntimeError("V2 population mismatch among queries/pool/additions/gold")

    print("[3/5] Recomputing all outside-pool recoveries and classifying mechanism...")
    recovered = []
    addition_counts = Counter()
    gold_addition_counts = Counter()
    triggered_by_type = defaultdict(set)

    direct_adds = {}
    relation_adds = {}
    full_adds = {}

    for q in qids:
        rec = rows[q]
        additions = [str(d) for d in rec.get("newly_added_doc_ids", [])]
        details = list(rec.get("addition_details", []))

        if len(additions) != len(details):
            raise RuntimeError(f"detail/addition length mismatch q={q}")

        direct = []
        relation = []
        detail_by_doc = {}

        for d, info in zip(additions, details):
            typ = str(info.get("addition_type", "UNKNOWN"))
            detail_by_doc[d] = info
            addition_counts[typ] += 1
            triggered_by_type[typ].add(q)

            if typ == "DIRECT_REFERENCE_MATCH":
                direct.append(d)
            elif typ == "RELATION_NEIGHBOR":
                relation.append(d)

            if d in gold[q]:
                gold_addition_counts[typ] += 1

        direct_adds[q] = direct
        relation_adds[q] = relation
        full_adds[q] = additions

        orig = set(pools[q])
        newly = (set(additions) & gold[q]) - (orig & gold[q])

        for d in sorted(newly):
            info = detail_by_doc.get(d, {})
            recovered.append({
                "qid": q,
                "question": questions[q],
                "doc_id": d,
                "addition_type": info.get("addition_type"),
                "relation_family": info.get("relation_family"),
                "relation_direction": info.get("relation_direction"),
                "is_header": info.get("is_header"),
                "anchor_ref": info.get("anchor_ref"),
                "anchor_doc": info.get("anchor_doc"),
                "evidence_snippet": info.get("evidence_snippet"),
                "generator_index": additions.index(d),
                "gold_count": len(gold[q]),
            })

    if len(recovered) != EXPECTED_FULL_RECOVERIES:
        raise RuntimeError(
            f"Expected {EXPECTED_FULL_RECOVERIES} historical V2 recoveries, got {len(recovered)}"
        )

    by_type = Counter(str(x["addition_type"]) for x in recovered)
    by_family = Counter(
        str(x["relation_family"])
        for x in recovered
        if x["addition_type"] == "RELATION_NEIGHBOR"
    )
    by_direction = Counter(
        str(x["relation_direction"])
        for x in recovered
        if x["addition_type"] == "RELATION_NEIGHBOR"
    )
    by_header = Counter(
        "HEADER" if x["is_header"] else "BODY"
        for x in recovered
        if x["addition_type"] == "RELATION_NEIGHBOR"
    )

    print("[4/5] Computing mechanism-level V2 candidate recall...")
    base_r = macro_candidate_recall(qids, pools, {}, gold)
    direct_r = macro_candidate_recall(qids, pools, direct_adds, gold)
    relation_r = macro_candidate_recall(qids, pools, relation_adds, gold)
    full_r = macro_candidate_recall(qids, pools, full_adds, gold)

    historical = json.loads(expected_report_path.read_text(encoding="utf-8"))
    expected_base = float(historical["candidate_recall"]["original"])
    expected_full = float(historical["candidate_recall"]["expanded"])

    if abs(base_r - expected_base) > 1e-12 or abs(full_r - expected_full) > 1e-12:
        raise RuntimeError(
            "V2 historical metric parity failed: "
            f"base {base_r} vs {expected_base}; full {full_r} vs {expected_full}"
        )

    # Conservative mechanism promotion rule fixed BEFORE looking at CAL:
    # relation mechanism must recover >=2 outside-pool gold occurrences across
    # >=2 distinct V2 queries and have positive candidate-recall delta.
    relation_recovered = [
        x for x in recovered if x["addition_type"] == "RELATION_NEIGHBOR"
    ]
    relation_queries = len({x["qid"] for x in relation_recovered})
    relation_promote = (
        len(relation_recovered) >= 2
        and relation_queries >= 2
        and relation_r > base_r
    )

    direct_recovered = [
        x for x in recovered if x["addition_type"] == "DIRECT_REFERENCE_MATCH"
    ]

    report = {
        "schema": "manual.v2_legal_ref_mechanism_forensics_v1",
        "status": "POST_SEAL_V2_FORENSICS",
        "provenance": {
            "additions_path": str(additions_path),
            "additions_sha256": actual_sha,
            "seal_path": str(seal_path),
            "seal_sha256": sha256(seal_path),
            "queries_path": str(queries_path),
            "queries_sha256": sha256(queries_path),
            "pool_path": str(pool_path),
            "pool_sha256": sha256(pool_path),
        },
        "population": {
            "queries": len(qids),
            "triggered_queries": len(triggered),
            "total_additions": total_additions,
        },
        "addition_breakdown": {
            "counts": dict(addition_counts),
            "gold_additions": dict(gold_addition_counts),
            "triggered_queries_by_type": {
                k: len(v) for k, v in triggered_by_type.items()
            },
        },
        "recovered_outside_gold": {
            "total": len(recovered),
            "by_addition_type": dict(by_type),
            "relation_by_family": dict(by_family),
            "relation_by_direction": dict(by_direction),
            "relation_by_header_body": dict(by_header),
            "cases": recovered,
        },
        "candidate_recall": {
            "baseline": base_r,
            "direct_only": direct_r,
            "direct_delta": direct_r - base_r,
            "relation_only": relation_r,
            "relation_delta": relation_r - base_r,
            "full_generator": full_r,
            "full_delta": full_r - base_r,
        },
        "mechanism_decision": {
            "rule": (
                "PROMOTE relation mechanism iff >=2 relation recoveries across "
                ">=2 V2 queries and relation-only candidate recall > baseline"
            ),
            "relation_recoveries": len(relation_recovered),
            "relation_recovery_queries": relation_queries,
            "direct_recoveries": len(direct_recovered),
            "verdict": (
                "PROMOTE_RELATION_MECHANISM_TO_ZERO_SHOT_CAL"
                if relation_promote
                else "DO_NOT_PROMOTE_RELATION_MECHANISM"
            ),
        },
    }

    out = (
        root
        / "results/manual/"
        "huy_v2_legal_ref_mechanism_forensics_v1"
    )
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / "REPORT.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("[5/5] RESULT")
    print("=" * 108)
    print(
        f"V2 additions: direct={addition_counts.get('DIRECT_REFERENCE_MATCH',0)} "
        f"relation={addition_counts.get('RELATION_NEIGHBOR',0)}"
    )
    print(
        f"V2 recoveries: direct={len(direct_recovered)} "
        f"relation={len(relation_recovered)}"
    )
    print(
        f"R(candidate): base={base_r:.10f} "
        f"direct={direct_r:.10f} ({direct_r-base_r:+.10f}) "
        f"relation={relation_r:.10f} ({relation_r-base_r:+.10f}) "
        f"full={full_r:.10f} ({full_r-base_r:+.10f})"
    )
    print("Relation families:", dict(by_family))
    print("Relation directions:", dict(by_direction))
    print("Relation header/body:", dict(by_header))
    print("VERDICT:", report["mechanism_decision"]["verdict"])
    print("Recovered cases:")
    for x in recovered:
        print(
            f"  q={x['qid']} doc={x['doc_id']} "
            f"type={x['addition_type']} family={x['relation_family']} "
            f"dir={x['relation_direction']} header={x['is_header']} "
            f"idx={x['generator_index']}"
        )
    print("Report:", report_path)
    print("=" * 108)


if __name__ == "__main__":
    main()
