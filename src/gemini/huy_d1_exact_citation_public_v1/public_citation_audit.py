"""Stage 3: Public Citation Indexing and Label-Free Audit for HUY_D1_EXACT_CITATION_PUBLIC_V1."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from run_burst_expanded_fusion_submission import title_from_link

from src.gemini.huy_d1_exact_citation_public_v1.common import (
    FROZEN_PRIOR_SOURCE_SHA,
    PUBLIC_CONTEXTS_DIR,
    RESULTS_DIR,
    ROOT,
    apply_tier1_repair,
    extract_doc_own_reference,
    extract_query_references,
    resolve_tier1_anchor,
    sha256_file,
)


def build_public_citation_index(contexts_dir: Path) -> Tuple[Dict[str, List[str]], Dict[str, str], Dict[str, Any]]:
    """Build public corpus own-reference index using frozen extract_doc_own_reference()."""
    print("=== BUILDING PUBLIC CITATION INDEX ===", flush=True)
    ref_to_docs = defaultdict(list)
    doc_to_own_ref = {}
    total_docs = 0
    doc_ids_sorted = []

    for p in sorted(contexts_dir.glob("context_*.json")):
        total_docs += 1
        doc_id = p.stem[len("context_") :]
        doc_ids_sorted.append(doc_id)
        row = json.loads(p.read_text(encoding="utf-8"))
        passage = row.get("passage", "")
        link = row.get("link", "")
        own_ref = extract_doc_own_reference(passage, link)
        if own_ref:
            doc_to_own_ref[doc_id] = own_ref
            ref_to_docs[own_ref].append(doc_id)

    for r in ref_to_docs:
        ref_to_docs[r].sort(key=lambda x: int(x) if x.isdigit() else x)

    collision_groups = {r: docs for r, docs in ref_to_docs.items() if len(docs) > 1}
    corpus_fp = sha256_file(contexts_dir) if contexts_dir.is_file() else None

    print(f"Public Corpus Documents:       {total_docs}", flush=True)
    print(f"Documents with Own Reference: {len(doc_to_own_ref)} ({len(doc_to_own_ref)/max(1, total_docs):.1%})", flush=True)
    print(f"Unique References:            {len(ref_to_docs)}", flush=True)
    print(f"Collision Groups (>1 doc):    {len(collision_groups)}", flush=True)

    index_manifest = {
        "schema_version": "dsc2026.gemini.huy_d1_exact_citation_public_v1.citation_index_manifest.v1",
        "experiment_id": "HUY_D1_EXACT_CITATION_PUBLIC_V1",
        "frozen_prior_source_sha": FROZEN_PRIOR_SOURCE_SHA,
        "parser_source_reference": "src/gemini/huy_d1_selective_repair_v1/tier1_citation.py",
        "public_corpus_contexts_dir": str(contexts_dir.relative_to(ROOT)).replace("\\", "/"),
        "total_documents": total_docs,
        "indexed_documents_count": len(doc_to_own_ref),
        "unindexed_documents_count": total_docs - len(doc_to_own_ref),
        "unique_references_count": len(ref_to_docs),
        "collision_groups_count": len(collision_groups),
        "sample_collision_groups": {
            r: collision_groups[r] for r in sorted(collision_groups.keys())[:10]
        },
    }

    out_path = RESULTS_DIR / "PUBLIC_CITATION_INDEX_MANIFEST.json"
    out_path.write_text(json.dumps(index_manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {out_path}", flush=True)

    return dict(ref_to_docs), doc_to_own_ref, index_manifest


def run_public_citation_audit(
    public_ids: List[str],
    public_meta: Dict[str, str],
    d1_preds: Dict[str, List[str]],
    public_candidates: Dict[str, List[str]],
    ref_to_docs: Dict[str, List[str]],
    doc_to_own_ref: Dict[str, str],
    docs_store: Any,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Perform label-free citation audit across all 1000 public queries."""
    print("=== RUNNING LABEL-FREE PUBLIC CITATION AUDIT ===", flush=True)

    total_queries = len(public_ids)
    queries_with_refs = 0
    queries_no_refs = 0
    unique_exact_matches = 0
    already_in_top5_count = 0
    unique_anchors_outside_top5 = 0
    ambiguous_doc_matches = 0
    multiple_eligible_abstentions = 0
    unmatched_references = 0

    unique_match_cases = []
    actions_dict = {}

    for q in public_ids:
        qtext = public_meta[q]
        anchor_doc, status, refs = resolve_tier1_anchor(qtext, ref_to_docs)

        if refs:
            queries_with_refs += 1
        else:
            queries_no_refs += 1

        if status == "AMBIGUOUS_DOC_MATCH":
            ambiguous_doc_matches += 1
        elif status == "MULTIPLE_ELIGIBLE_DOCS":
            multiple_eligible_abstentions += 1
        elif status == "UNMATCHED_REFERENCES":
            unmatched_references += 1
        elif status == "UNIQUE_EXACT_MATCH" and anchor_doc is not None:
            unique_exact_matches += 1
            d1_top5 = d1_preds[q]
            already_in_top5 = anchor_doc in d1_top5

            # Get titles
            def get_doc_title(did: str) -> str:
                try:
                    ctx_p = PUBLIC_CONTEXTS_DIR / f"context_{did}.json"
                    if ctx_p.exists():
                        row = json.loads(ctx_p.read_text(encoding="utf-8"))
                        return title_from_link(row.get("link")) or row.get("passage", "")[:80]
                except Exception:
                    pass
                return f"Doc {did}"

            anchor_title = get_doc_title(anchor_doc)
            anchor_own_ref = doc_to_own_ref.get(anchor_doc, "UNKNOWN")

            # Determine anchor original D1 rank in candidate pool
            q_cands = public_candidates[q]
            if anchor_doc in d1_top5:
                anchor_d1_rank = d1_top5.index(anchor_doc) + 1
            elif anchor_doc in q_cands:
                anchor_d1_rank = f"candidate_pool_index_{q_cands.index(anchor_doc)}"
            else:
                anchor_d1_rank = "outside_candidate_pool"

            defender_doc = d1_top5[4]
            defender_title = get_doc_title(defender_doc)

            new_top5, action_type, inj, evict = apply_tier1_repair(d1_top5, anchor_doc)

            if action_type == "KEEP_ALREADY_IN_TOP5":
                already_in_top5_count += 1
                action_str = "KEEP"
            elif action_type == "EXACT_CITATION_INJECTION":
                unique_anchors_outside_top5 += 1
                action_str = "EXACT_CITATION_INJECTION"
                actions_dict[q] = {
                    "qid": q,
                    "query_text": qtext,
                    "extracted_references": refs,
                    "anchor_doc_id": anchor_doc,
                    "anchor_title": anchor_title,
                    "anchor_own_reference": anchor_own_ref,
                    "anchor_original_d1_rank": anchor_d1_rank,
                    "defender_doc_id": defender_doc,
                    "defender_title": defender_title,
                    "d1_top5_before": d1_top5,
                    "d1_top5_after": new_top5,
                }
            else:
                action_str = "ABSTAIN"

            case_record = {
                "qid": q,
                "query_text": qtext,
                "extracted_references": refs,
                "resolution_status": status,
                "anchor_doc_id": anchor_doc,
                "anchor_title": anchor_title,
                "anchor_own_reference": anchor_own_ref,
                "anchor_original_d1_rank": anchor_d1_rank,
                "whether_anchor_already_in_top5": already_in_top5,
                "rank5_defender_doc_id": defender_doc,
                "rank5_defender_title": defender_title,
                "action": action_str,
            }
            unique_match_cases.append(case_record)

    audit_summary = {
        "schema_version": "dsc2026.gemini.huy_d1_exact_citation_public_v1.public_audit.v1",
        "experiment_id": "HUY_D1_EXACT_CITATION_PUBLIC_V1",
        "total_queries": total_queries,
        "queries_with_parsed_legal_references": queries_with_refs,
        "queries_with_no_references": queries_no_refs,
        "unique_exact_matches": unique_exact_matches,
        "already_in_d1_top5_anchors": already_in_top5_count,
        "unique_anchors_outside_top5": unique_anchors_outside_top5,
        "ambiguous_document_matches": ambiguous_doc_matches,
        "multiple_eligible_document_abstentions": multiple_eligible_abstentions,
        "unmatched_references": unmatched_references,
        "intervention_count": len(actions_dict),
        "unique_match_cases": unique_match_cases,
    }

    audit_path = RESULTS_DIR / "PUBLIC_EXACT_CITATION_AUDIT.json"
    audit_path.write_text(json.dumps(audit_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {audit_path}", flush=True)

    actions_path = RESULTS_DIR / "PUBLIC_CITATION_ACTIONS.json"
    actions_doc = {
        "schema_version": "dsc2026.gemini.huy_d1_exact_citation_public_v1.public_actions.v1",
        "experiment_id": "HUY_D1_EXACT_CITATION_PUBLIC_V1",
        "total_interventions": len(actions_dict),
        "actions": actions_dict,
    }
    actions_path.write_text(json.dumps(actions_doc, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {actions_path}", flush=True)

    return audit_summary, actions_doc, actions_dict
