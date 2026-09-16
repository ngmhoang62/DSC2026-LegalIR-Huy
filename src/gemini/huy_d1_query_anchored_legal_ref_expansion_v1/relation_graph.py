"""Explicit Legal Relation Graph Parser from Raw Corpus."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

ROOT = Path("D:/Study/DSC2026/sota")
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.gemini.huy_d1_query_anchored_legal_ref_expansion_v1.common import (
    RES_DIR,
    load_cal_corpus,
)
from src.gemini.huy_d1_query_anchored_legal_ref_expansion_v1.legal_ref_indexer import (
    REF_REGEX,
    build_legal_reference_index,
    canonicalize_ref,
)

AMEND_TRIGGERS = [
    "sửa đổi", "bổ sung", "thay thế", "bãi bỏ",
    "sua doi", "bo sung", "thay the", "bai bo",
]
GUIDE_TRIGGERS = [
    "hướng dẫn thi hành", "quy định chi tiết và hướng dẫn thi hành",
    "quy định chi tiết", "hướng dẫn", "thi hành",
    "huong dan thi hanh", "quy dinh chi tiet", "huong dan", "thi hanh",
]


def build_explicit_relation_graph(
    corpus: Dict[str, Dict[str, Any]],
    ref_to_docs: Dict[str, List[str]],
    doc_to_own_ref: Dict[str, str],
    audit_filename: str = "EXPLICIT_RELATION_GRAPH_AUDIT.json",
) -> Tuple[
    Dict[str, List[Tuple[str, str, bool, str]]],
    Dict[str, List[Tuple[str, str, bool, str]]],
    dict,
]:
    print(f"=== BUILDING EXPLICIT LEGAL RELATION GRAPH ({audit_filename}) ===", flush=True)
    RES_DIR.mkdir(parents=True, exist_ok=True)

    out_edges: Dict[str, List[Tuple[str, str, bool, str]]] = defaultdict(list)
    in_edges: Dict[str, List[Tuple[str, str, bool, str]]] = defaultdict(list)
    edge_records = []

    amend_count = 0
    guide_count = 0
    header_edge_count = 0

    for doc_id, item in corpus.items():
        passage = item.get("passage", "")
        own_ref = doc_to_own_ref.get(doc_id)

        # Look for explicit relations in passage
        for m in REF_REGEX.finditer(passage):
            raw_match = m.group(0)
            target_ref = canonicalize_ref(raw_match)

            # Skip self-reference
            if own_ref and target_ref == own_ref:
                continue

            target_doc_ids = ref_to_docs.get(target_ref, [])
            if not target_doc_ids:
                continue

            start, end = m.span()
            # Context window around the reference
            ctx_before = passage[max(0, start - 80) : start].lower()
            ctx_after = passage[end : min(len(passage), end + 80)].lower()
            full_ctx = ctx_before + " " + ctx_after
            snippet = passage[max(0, start - 50) : min(len(passage), end + 50)].strip().replace("\n", " ")

            rel_family = None
            if any(t in ctx_before for t in AMEND_TRIGGERS) or any(t in ctx_after for t in AMEND_TRIGGERS):
                rel_family = "AMENDMENT_REPLACEMENT_REPEAL"
                amend_count += 1
            elif any(t in ctx_before for t in GUIDE_TRIGGERS) or any(t in ctx_after for t in GUIDE_TRIGGERS):
                rel_family = "IMPLEMENTATION_GUIDANCE"
                guide_count += 1

            if rel_family:
                is_header = (start < 1200)
                if is_header:
                    header_edge_count += 1

                for td in target_doc_ids:
                    if td != doc_id:
                        edge = (td, rel_family, is_header, snippet)
                        rev_edge = (doc_id, rel_family, is_header, snippet)
                        if edge not in out_edges[doc_id]:
                            out_edges[doc_id].append(edge)
                        if rev_edge not in in_edges[td]:
                            in_edges[td].append(rev_edge)

                        if len(edge_records) < 100:
                            edge_records.append({
                                "source_doc": doc_id,
                                "target_ref": target_ref,
                                "target_doc": td,
                                "relation_family": rel_family,
                                "is_header": is_header,
                                "evidence_snippet": snippet[:120],
                            })

    total_out_edges = sum(len(edges) for edges in out_edges.values())

    print(f"Nodes with Outgoing Relations: {len(out_edges)}", flush=True)
    print(f"Nodes with Incoming Relations: {len(in_edges)}", flush=True)
    print(f"Total Directed Relation Edges:  {total_out_edges}", flush=True)
    print(f"  Amendment/Repeal Mentions:   {amend_count}", flush=True)
    print(f"  Guidance/Impl Mentions:      {guide_count}", flush=True)
    print(f"  Header Relations Count:      {header_edge_count}", flush=True)

    audit_doc = {
        "schema_version": "dsc2026.gemini.huy_d1_query_anchored_legal_ref_expansion_v1.relation_audit.v1",
        "experiment_id": "HUY_D1_QUERY_ANCHORED_LEGAL_REF_EXPANSION_V1",
        "status": "PASS",
        "nodes_with_out_edges": len(out_edges),
        "nodes_with_in_edges": len(in_edges),
        "total_directed_edges": total_out_edges,
        "relation_family_breakdown": {
            "amendment_replacement_repeal_mentions": amend_count,
            "implementation_guidance_mentions": guide_count,
            "header_evidence_count": header_edge_count,
        },
        "sample_edges": edge_records[:20],
    }

    if audit_filename:
        out_path = RES_DIR / audit_filename
        out_path.write_text(json.dumps(audit_doc, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Saved {out_path}", flush=True)
    print("=== EXPLICIT RELATION GRAPH AUDIT PASSED ===\n", flush=True)

    return dict(out_edges), dict(in_edges), audit_doc


if __name__ == "__main__":
    c = load_cal_corpus()
    ref_to_docs, doc_to_own_ref, _ = build_legal_reference_index(c)
    build_explicit_relation_graph(c, ref_to_docs, doc_to_own_ref)
