"""Query-Anchored Legal Reference Candidate Generator."""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

ROOT = Path("D:/Study/DSC2026/sota")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.gemini.huy_d1_query_anchored_legal_ref_expansion_v1.common import (
    RES_DIR,
    get_git_status,
    sha256_file,
)
from src.gemini.huy_d1_query_anchored_legal_ref_expansion_v1.legal_ref_indexer import (
    REF_REGEX,
    canonicalize_ref,
)
from src.gemini.huy_d1_query_anchored_legal_ref_expansion_v1.relation_graph import (
    AMEND_TRIGGERS,
    GUIDE_TRIGGERS,
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
    query_texts: Dict[str, Any],
    all_ids: List[str],
    extended: Dict[str, List[str]],
    ref_to_docs: Dict[str, List[str]],
    out_edges: Dict[str, List[Tuple[str, str, bool, str]]],
    in_edges: Dict[str, List[Tuple[str, str, bool, str]]],
    cap: int = MAX_ADDITIONS_PER_QUERY,
) -> Tuple[Dict[str, List[str]], List[dict]]:
    """Generate query-anchored additions for all CAL queries strictly without gold labels."""
    print("=== GENERATING CAL CANDIDATE ADDITIONS ===", flush=True)
    RES_DIR.mkdir(parents=True, exist_ok=True)

    additions: Dict[str, List[str]] = {}
    jsonl_records: List[dict] = []

    triggered_count = 0
    total_additions_count = 0

    for q in all_ids:
        val = query_texts[q]
        q_text = val[0] if isinstance(val, (list, tuple)) else val
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

    print(f"CAL Triggered Queries Count: {triggered_count} / {len(all_ids)}", flush=True)
    print(f"CAL Total New Additions:     {total_additions_count}", flush=True)
    print(f"Saved {jsonl_path}", flush=True)

    return additions, jsonl_records


def generate_all_v2_additions(
    query_texts: Dict[str, str],
    v2_qids: List[str],
    candidate_pools: Dict[str, List[str]],
    ref_to_docs: Dict[str, List[str]],
    out_edges: Dict[str, List[Tuple[str, str, bool, str]]],
    in_edges: Dict[str, List[Tuple[str, str, bool, str]]],
    cap: int = MAX_ADDITIONS_PER_QUERY,
) -> Tuple[Dict[str, List[str]], List[dict]]:
    """Generate query-anchored additions for all V2 queries strictly without gold labels."""
    print("=== GENERATING V2 CANDIDATE ADDITIONS ===", flush=True)
    RES_DIR.mkdir(parents=True, exist_ok=True)

    additions: Dict[str, List[str]] = {}
    jsonl_records: List[dict] = []

    triggered_count = 0
    total_additions_count = 0

    for q in v2_qids:
        q_text = query_texts[q]
        exist_pool = set(candidate_pools[q])

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

    jsonl_path = RES_DIR / "V2_PER_QUERY_EXPANSION.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in jsonl_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"V2 Triggered Queries Count: {triggered_count} / {len(v2_qids)}", flush=True)
    print(f"V2 Total New Additions:     {total_additions_count}", flush=True)
    print(f"Saved {jsonl_path}", flush=True)

    return additions, jsonl_records


def seal_additions_artifact(
    dataset_name: str,
    jsonl_path: Path,
    jsonl_records: List[dict],
    query_fp: str,
    cand_fp: str,
    corpus_fp: str,
    ref_index_sha256: str,
    rel_graph_sha256: str,
    cap: int = MAX_ADDITIONS_PER_QUERY,
) -> dict:
    """Create authoritative provenance seal for materialized additions artifact."""
    git_info = get_git_status()
    art_sha256 = sha256_file(jsonl_path)

    triggered_count = sum(1 for r in jsonl_records if r.get("addition_count", 0) > 0)
    total_additions = sum(r.get("addition_count", 0) for r in jsonl_records)

    seal_doc = {
        "schema_version": "dsc2026.gemini.huy_d1_query_anchored_legal_ref_expansion_v1.additions_seal.v1",
        "experiment_id": "HUY_D1_QUERY_ANCHORED_LEGAL_REF_EXPANSION_V1",
        "dataset": dataset_name.upper(),
        "git_commit_sha": git_info.get("head_commit"),
        "query_fingerprint": query_fp,
        "baseline_candidate_pool_fingerprint": cand_fp,
        "corpus_fingerprint": corpus_fp,
        "reference_index_audit_sha256": ref_index_sha256,
        "relation_graph_audit_sha256": rel_graph_sha256,
        "generation_policy_config": {
            "reference_regex": str(REF_REGEX.pattern),
            "relation_triggers": {
                "amend_triggers_count": len(AMEND_TRIGGERS),
                "guide_triggers_count": len(GUIDE_TRIGGERS),
            },
            "relation_context_chars": 80,
            "header_threshold_chars": 1200,
            "max_additions_cap": cap,
            "traversal_hops": 1,
            "candidate_ordering": [
                "DIRECT_REFERENCE_MATCH",
                "HEADER_RELATION_NEIGHBOR",
                "BODY_RELATION_NEIGHBOR",
            ],
            "tie_breaking": "CANONICAL_DOC_ID_NUMERIC_ASC",
        },
        "cap": cap,
        "generated_additions_artifact_path": str(jsonl_path.relative_to(ROOT)).replace("\\", "/"),
        "generated_additions_artifact_sha256": art_sha256,
        "total_queries_evaluated": len(jsonl_records),
        "triggered_queries_count": triggered_count,
        "total_new_additions": total_additions,
        "sealed_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    seal_path = RES_DIR / f"{dataset_name.upper()}_PER_QUERY_EXPANSION_SEAL.json"
    seal_path.write_text(json.dumps(seal_doc, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {seal_path} (Artifact SHA256: {art_sha256[:16]}...)", flush=True)
    return seal_doc
