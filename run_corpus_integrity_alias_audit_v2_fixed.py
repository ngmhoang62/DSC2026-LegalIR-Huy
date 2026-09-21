#!/usr/bin/env python
"""
HUY PIPELINE AUDIT G — CORPUS INTEGRITY / ALIAS COLLISIONS V1
=============================================================

CPU-only, no model inference, no public labels.

Audits the 8,532 selected-context documents for:
- missing / empty passage and fallback-title dependence;
- duplicate IDs / duplicate links;
- byte-exact and normalized-text duplicate groups;
- legal-document-number collisions (same "Số: ...");
- whether exact D1 Top5 wastes slots on duplicate/alias-equivalent documents;
- whether CAL gold labels themselves contain alias collisions;
- optional comparison to sibling LegalIR exclusions.json if available.

This does NOT automatically collapse aliases. It only identifies whether corpus
integrity can plausibly improve recall/precision without semantic reranking.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlparse

import numpy as np


WS_RE = re.compile(r"\s+", re.UNICODE)
NONWORD_RE = re.compile(r"[^\w\d]+", re.UNICODE)


def norm_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "").lower()
    s = WS_RE.sub(" ", s).strip()
    return s


def aggressive_norm(s: str) -> str:
    s = norm_text(s)
    s = NONWORD_RE.sub(" ", s)
    return WS_RE.sub(" ", s).strip()


def canonical_link(link: str) -> str:
    if not link:
        return ""
    try:
        u = urlparse(link)
        path = u.path.rstrip("/").lower()
        return path
    except Exception:
        return link.strip().lower()


def title_from_link(link: str) -> str:
    if not link:
        return ""
    slug = urlparse(link).path.rsplit("/", 1)[-1]
    slug = re.sub(r"\.aspx$", "", slug, flags=re.I)
    slug = re.sub(r"-\d+$", "", slug)
    return slug.replace("-", " ").strip()


def load_d1(root: Path):
    """Load exact D1 Top5 from the authoritative artifact, but load CAL gold
    from build_training_cap(). The predictions JSONL intentionally does not
    guarantee a `gold` field, so never infer its schema here."""
    from tune_corpus_cap32_fusion import build_training_cap

    p = (
        root
        / "results/gemini/huy_d1_legal_section_evidence_v1/"
        "S0_S1_CAL_PREDICTIONS.jsonl"
    )

    queries, blocks, ids, _, _, _ = build_training_cap(
        root,
        32,
        "results/corpus_index/holdout_extended_scores_cap32.pkl",
        depth=20,
    )
    gold = {
        str(q): set(map(str, queries[q][1]))
        for q in ids
    }

    out = {}
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            q = str(r["qid"])
            if q not in gold:
                continue
            out[q] = {
                "top5": [str(x) for x in r["s0_top5"]],
                "gold": gold[q],
            }

    if set(out) != set(gold):
        missing = sorted(set(gold) - set(out))
        extra = sorted(set(out) - set(gold))
        raise RuntimeError(
            f"D1/CAL population mismatch: missing={missing[:5]} extra={extra[:5]}"
        )

    return out, p


def group_map(groups):
    m = {}
    for gid, docs in enumerate(groups):
        for d in docs:
            m[d] = gid
    return m


def pairs_same_group(items, gmap):
    seen = defaultdict(list)
    for d in items:
        if d in gmap:
            seen[gmap[d]].append(d)
    return [v for v in seen.values() if len(v) >= 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=Path, required=True)
    args = ap.parse_args()

    root = args.repo_root.resolve()
    sys.path.insert(0, str(root))
    from tune_citation_graph import own_number

    ctx = (
        root
        / "DSC2026-LegalIR-main/v4_run/public_test_dataset/selected-contexts"
    )
    paths = sorted(ctx.glob("context_*.json"))
    if len(paths) != 8532:
        raise RuntimeError(f"Expected 8532 contexts, found {len(paths)}")

    print("[1/5] Loading corpus and basic integrity...", flush=True)

    docs = {}
    duplicate_ids = defaultdict(list)
    empty = []
    no_link = []
    fallback_only = []

    by_link = defaultdict(list)
    by_exact = defaultdict(list)
    by_norm = defaultdict(list)
    by_aggr = defaultdict(list)
    by_number = defaultdict(list)

    for i, p in enumerate(paths, 1):
        r = json.loads(p.read_text(encoding="utf-8"))
        did = str(r.get("id", p.stem.replace("context_", "")))
        duplicate_ids[did].append(str(p))
        passage = r.get("passage") or ""
        link = r.get("link") or ""
        effective = passage or title_from_link(link)

        docs[did] = {
            "passage": passage,
            "link": link,
            "effective": effective,
        }

        if not passage.strip():
            empty.append(did)
            if effective.strip():
                fallback_only.append(did)
        if not link.strip():
            no_link.append(did)

        cl = canonical_link(link)
        if cl:
            by_link[cl].append(did)

        if passage.strip():
            by_exact[hashlib.sha256(passage.encode("utf-8")).hexdigest()].append(did)
            by_norm[hashlib.sha256(norm_text(passage).encode("utf-8")).hexdigest()].append(did)
            by_aggr[hashlib.sha256(aggressive_norm(passage).encode("utf-8")).hexdigest()].append(did)

        num = own_number(effective)
        if num:
            by_number[num].append(did)

        if i % 1000 == 0:
            print(f"  docs {i}/{len(paths)}", flush=True)

    duplicate_id_groups = [v for v in duplicate_ids.values() if len(v) > 1]
    duplicate_link_groups = [v for v in by_link.values() if len(v) > 1]
    exact_groups = [v for v in by_exact.values() if len(v) > 1]
    norm_groups = [v for v in by_norm.values() if len(v) > 1]
    aggr_groups = [v for v in by_aggr.values() if len(v) > 1]
    number_groups = [v for v in by_number.values() if len(v) > 1]

    print("[2/5] Loading exact D1 CAL Top5/gold...", flush=True)
    d1, d1_path = load_d1(root)

    exact_map = group_map(exact_groups)
    norm_map = group_map(norm_groups)
    aggr_map = group_map(aggr_groups)
    number_map = group_map(number_groups)

    print("[3/5] Measuring alias collisions in D1 Top5 and CAL gold...", flush=True)

    collision_rows = []
    counts = defaultdict(int)

    for q, r in d1.items():
        top5 = r["top5"]
        gold = sorted(r["gold"])

        top_exact = pairs_same_group(top5, exact_map)
        top_norm = pairs_same_group(top5, norm_map)
        top_aggr = pairs_same_group(top5, aggr_map)
        top_num = pairs_same_group(top5, number_map)

        gold_exact = pairs_same_group(gold, exact_map)
        gold_norm = pairs_same_group(gold, norm_map)
        gold_aggr = pairs_same_group(gold, aggr_map)
        gold_num = pairs_same_group(gold, number_map)

        if top_exact:
            counts["top5_exact_dup_queries"] += 1
        if top_norm:
            counts["top5_norm_dup_queries"] += 1
        if top_aggr:
            counts["top5_aggressive_dup_queries"] += 1
        if top_num:
            counts["top5_same_number_queries"] += 1

        if gold_exact:
            counts["gold_exact_dup_queries"] += 1
        if gold_norm:
            counts["gold_norm_dup_queries"] += 1
        if gold_aggr:
            counts["gold_aggressive_dup_queries"] += 1
        if gold_num:
            counts["gold_same_number_queries"] += 1

        if top_exact or top_norm or top_aggr or top_num or gold_exact or gold_norm or gold_aggr or gold_num:
            collision_rows.append({
                "qid": q,
                "top5": top5,
                "gold": gold,
                "top5_exact_groups": top_exact,
                "top5_norm_groups": top_norm,
                "top5_aggressive_groups": top_aggr,
                "top5_same_number_groups": top_num,
                "gold_exact_groups": gold_exact,
                "gold_norm_groups": gold_norm,
                "gold_aggressive_groups": gold_aggr,
                "gold_same_number_groups": gold_num,
            })

    print("[4/5] Optional sibling exclusions/alias audit...", flush=True)

    exclusions_path = (
        root.parent
        / "LegalIR/cache/final_preprocessed_v2/exclusions.json"
    )
    exclusions = []
    exclusion_summary = {}
    if exclusions_path.is_file():
        exclusions = json.loads(exclusions_path.read_text(encoding="utf-8"))
        reasons = defaultdict(int)
        retained_map = {}
        for row in exclusions:
            for reason in row.get("reasons", []):
                reasons[reason] += 1
            if row.get("duplicate_retained_id"):
                retained_map[str(row["doc_id"])] = str(row["duplicate_retained_id"])

        current_ids = set(docs)
        excluded_present = sorted(current_ids & set(retained_map))
        retained_targets_present = sum(
            retained in current_ids for retained in retained_map.values()
        )
        exclusion_summary = {
            "path": str(exclusions_path),
            "rows": len(exclusions),
            "reason_counts": dict(reasons),
            "duplicate_alias_rows": len(retained_map),
            "excluded_alias_ids_still_present_in_selected_contexts": excluded_present,
            "excluded_alias_ids_still_present_count": len(excluded_present),
            "retained_alias_targets_present_count": retained_targets_present,
        }
    else:
        exclusion_summary = {
            "path": str(exclusions_path),
            "available": False,
        }

    # Potential slot-waste lower bounds.
    # Use increasingly permissive definitions but do not recommend collapsing
    # legal-number collisions automatically.
    potential_exact_slot_waste = sum(
        sum(len(g)-1 for g in pairs_same_group(r["top5"], exact_map))
        for r in d1.values()
    )
    potential_norm_slot_waste = sum(
        sum(len(g)-1 for g in pairs_same_group(r["top5"], norm_map))
        for r in d1.values()
    )

    report = {
        "schema": "manual.corpus_integrity_alias_audit_v1",
        "corpus": {
            "documents": len(docs),
            "duplicate_id_groups": duplicate_id_groups,
            "empty_passage_count": len(empty),
            "empty_passage_doc_ids": empty,
            "fallback_title_only_count": len(fallback_only),
            "fallback_title_only_doc_ids": fallback_only,
            "missing_link_count": len(no_link),
            "duplicate_link_groups_count": len(duplicate_link_groups),
            "duplicate_link_groups": duplicate_link_groups,
            "exact_passage_duplicate_groups_count": len(exact_groups),
            "exact_passage_duplicate_groups": exact_groups,
            "normalized_passage_duplicate_groups_count": len(norm_groups),
            "normalized_passage_duplicate_groups": norm_groups,
            "aggressive_normalized_duplicate_groups_count": len(aggr_groups),
            "aggressive_normalized_duplicate_groups": aggr_groups,
            "same_legal_number_groups_count": len(number_groups),
            "same_legal_number_groups": number_groups,
        },
        "d1_alias_collision": {
            **dict(counts),
            "potential_exact_duplicate_slots_wasted": potential_exact_slot_waste,
            "potential_normalized_duplicate_slots_wasted": potential_norm_slot_waste,
            "cases": collision_rows,
        },
        "sibling_exclusions": exclusion_summary,
        "interpretation": {
            "exact_or_normalized_duplicate_collapse_safe_candidate": bool(
                counts["top5_norm_dup_queries"] > 0
                and counts["gold_norm_dup_queries"] == 0
            ),
            "same_legal_number_is_not_safe_alias_rule": True,
            "public_labels_used": False,
            "cal_gold_used_only_for_collision_diagnostics": True,
        },
        "provenance": {
            "d1_predictions": str(d1_path),
        },
    }

    out = root / "results/manual/huy_corpus_integrity_alias_audit_v1"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "REPORT.json"
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("[5/5] RESULT")
    print("=" * 112)
    print(
        f"Corpus: empty={len(empty)}, fallback-title-only={len(fallback_only)}, "
        f"dup-links={len(duplicate_link_groups)}, exact-dup-groups={len(exact_groups)}, "
        f"norm-dup-groups={len(norm_groups)}, same-number-groups={len(number_groups)}"
    )
    print(
        "D1 Top5 duplicate-collision queries: "
        f"exact={counts['top5_exact_dup_queries']} "
        f"normalized={counts['top5_norm_dup_queries']} "
        f"aggressive={counts['top5_aggressive_dup_queries']} "
        f"same-number={counts['top5_same_number_queries']}"
    )
    print(
        "CAL gold duplicate-collision queries: "
        f"exact={counts['gold_exact_dup_queries']} "
        f"normalized={counts['gold_norm_dup_queries']} "
        f"aggressive={counts['gold_aggressive_dup_queries']} "
        f"same-number={counts['gold_same_number_queries']}"
    )
    print(
        f"Potential exact/norm Top5 slot waste: "
        f"{potential_exact_slot_waste}/{potential_norm_slot_waste}"
    )
    if exclusions_path.is_file():
        print(
            "Sibling exclusions: "
            f"rows={exclusion_summary['rows']} "
            f"duplicate_alias_rows={exclusion_summary['duplicate_alias_rows']} "
            f"excluded_alias_ids_still_present="
            f"{exclusion_summary['excluded_alias_ids_still_present_count']}"
        )
    else:
        print("Sibling exclusions: unavailable")
    print(
        "SAFE_COLLAPSE_SIGNAL:",
        report["interpretation"]["exact_or_normalized_duplicate_collapse_safe_candidate"],
    )
    print("Report:", path)
    print("=" * 112)


if __name__ == "__main__":
    main()
