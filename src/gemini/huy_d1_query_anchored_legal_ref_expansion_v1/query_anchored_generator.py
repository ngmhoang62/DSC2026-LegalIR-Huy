"""Query-Anchored Legal Reference Candidate Generator."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

ROOT = Path("D:/Study/DSC2026/sota")
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.gemini.huy_d1_query_anchored_legal_ref_expansion_v1.common import RES_DIR
from src.gemini.huy_d1_query_anchored_legal_ref_expansion_v1.legal_ref_indexer import (
    REF_REGEX,
    canonicalize_ref,
)

MAX_ADDITIONS_PER_QUERY = 8


def extract_query_references(query_text: str) -> List[str]:
    """Extract canonical legal references from raw query text."""
    raw_matches = REF_REGEX.findall(query_text or "")
    refs = []
    for m in raw_matches:
        # Exclude pure date matches like 02/9/2020 or 1/1/2021
        if re.match(r"^\d{1,2}/\d{1,2}/\d{4}$", m):
            continue
        c_ref = canonicalize_ref(m)
        if len(c_ref) >= 3 and any(c.isdigit() for c in c_ref):
            if c_ref not in refs:
                refs.append(c_ref)
    return refs


def generate_query_additions(
    query_text: str,
    existing_pool: Set[str],
    ref_to_docs: Dict[str, List[str]],
    out_edges: Dict[str, List[Tuple[str, str, bool, str]]],
    in_edges: Dict[str, List[Tuple[str, str, bool, str]]],
    cap: int = MAX_ADDITIONS_PER_QUERY,
) -> Tuple[List[str], List[dict], List[str]]:
    """Generate up to cap deterministic query-anchored legal reference candidate additions."""
    query_refs = extract_query_references(query_text)
    if not query_refs:
        return [], [], []

    # 1. Exact matches for the referenced documents
    exact_docs: List[str] = []
    exact_details: Dict[str, dict] = {}
    for r in query_refs:
        for d in ref_to_docs.get(r, []):
            if d not in exact_docs:
                exact_docs.append(d)
                exact_details[d] = {
                    "doc_id": d,
                    "addition_type": "DIRECT_REFERENCE_MATCH",
                    "anchor_ref": r,
                    "relation_family": "EXACT_OWN_REFERENCE",
                    "evidence_snippet": f"Matched own legal reference: {r}",
                }

    # 2. One-hop explicit relation neighbors of exact_docs
    header_neighbors: List[str] = []
    body_neighbors: List[str] = []
    neighbor_details: Dict[str, dict] = {}

    for d in exact_docs:
        # Out-edges
        for td, rel_fam, is_h, snip in out_edges.get(d, []):
            if td not in exact_docs:
                if is_h and td not in header_neighbors:
                    header_neighbors.append(td)
                elif not is_h and td not in body_neighbors:
                    body_neighbors.append(td)
                if td not in neighbor_details:
                    neighbor_details[td] = {
                        "doc_id": td,
                        "addition_type": "RELATION_NEIGHBOR",
                        "anchor_doc": d,
                        "relation_direction": "OUTGOING",
                        "relation_family": rel_fam,
                        "is_header": is_h,
                        "evidence_snippet": snip,
                    }

        # In-edges
        for sd, rel_fam, is_h, snip in in_edges.get(d, []):
            if sd not in exact_docs:
                if is_h and sd not in header_neighbors:
                    header_neighbors.append(sd)
                elif not is_h and sd not in body_neighbors:
                    body_neighbors.append(sd)
                if sd not in neighbor_details:
                    neighbor_details[sd] = {
                        "doc_id": sd,
                        "addition_type": "RELATION_NEIGHBOR",
                        "anchor_doc": d,
                        "relation_direction": "INCOMING",
                        "relation_family": rel_fam,
                        "is_header": is_h,
                        "evidence_snippet": snip,
                    }

    # Deterministic tie-breaking by canonical document ID
    header_neighbors.sort(key=lambda x: int(x) if x.isdigit() else x)
    body_neighbors.sort(key=lambda x: int(x) if x.isdigit() else x)

    # Ordered candidate stream: exact -> header neighbors -> body neighbors
    candidates_stream = exact_docs + header_neighbors + body_neighbors

    # Filter out existing pool and deduplicate
    final_additions = []
    final_details = []
    for d in candidates_stream:
        if d not in final_additions and d not in existing_pool:
            final_additions.append(d)
            dt = exact_details.get(d) or neighbor_details.get(d) or {"doc_id": d}
            final_details.append(dt)
        if len(final_additions) >= cap:
            break

    return final_additions, final_details, query_refs


def generate_all_cal_additions(
    queries: Dict[str, Any],
    all_ids: List[str],
    extended: Dict[str, List[str]],
    ref_to_docs: Dict[str, List[str]],
    out_edges: Dict[str, List[Tuple[str, str, bool, str]]],
    in_edges: Dict[str, List[Tuple[str, str, bool, str]]],
    cap: int = MAX_ADDITIONS_PER_QUERY,
) -> Tuple[Dict[str, List[str]], List[dict]]:
    """Generate query-anchored additions for all CAL queries and save JSONL."""
    print("=== GENERATING CAL CANDIDATE ADDITIONS ===", flush=True)
    RES_DIR.mkdir(parents=True, exist_ok=True)

    additions: Dict[str, List[str]] = {}
    jsonl_records: List[dict] = []

    triggered_count = 0
    total_additions_count = 0

    for q in all_ids:
        q_text = queries[q][0]
        exist_pool = set(extended[q])

        adds, details, q_refs = generate_query_additions(
            query_text=q_text,
            existing_pool=exist_pool,
            ref_to_docs=ref_to_docs,
            out_edges=out_edges,
            in_edges=in_edges,
            cap=cap,
        )

        additions[q] = adds
        if adds:
            triggered_count += 1
            total_additions_count += len(adds)

        rec = {
            "qid": q,
            "query_text": q_text,
            "extracted_references": q_refs,
            "original_pool_size": len(exist_pool),
            "newly_added_doc_ids": adds,
            "addition_count": len(adds),
            "addition_details": details,
        }
        jsonl_records.append(rec)

    jsonl_path = RES_DIR / "CAL_PER_QUERY_EXPANSION.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in jsonl_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Triggered Queries Count:    {triggered_count} / {len(all_ids)}", flush=True)
    print(f"Total New Additions Across: {total_additions_count}", flush=True)
    print(f"Saved {jsonl_path}", flush=True)

    return additions, jsonl_records
