"""Tier 1: Exact Legal Citation Protected Injection and Generalization Audit."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from src.gemini.huy_d1_selective_repair_v1.common import (
    CAL_CONTEXTS_DIR,
    CANONICAL_V2_CONTEXTS_JSONL,
    CANONICAL_V2_QUERIES_JSONL,
    ROOT,
    canonicalize_ref,
    extract_doc_own_reference,
    extract_query_references,
)


def index_corpus_own_references(contexts_dir: Path) -> Tuple[Dict[str, List[str]], Dict[str, str]]:
    """Index own official document references for CAL corpus."""
    ref_to_docs = defaultdict(list)
    doc_to_own_ref = {}

    for p in sorted(contexts_dir.glob("context_*.json")):
        doc_id = p.stem[len("context_") :]
        row = json.loads(p.read_text(encoding="utf-8"))
        passage = row.get("passage", "")
        link = row.get("link", "")
        own_ref = extract_doc_own_reference(passage, link)
        if own_ref:
            doc_to_own_ref[doc_id] = own_ref
            ref_to_docs[own_ref].append(doc_id)

    # Deterministic sorting
    for r in ref_to_docs:
        ref_to_docs[r].sort(key=lambda x: int(x) if x.isdigit() else x)

    return dict(ref_to_docs), doc_to_own_ref


def index_v2_corpus_own_references(contexts_jsonl: Path) -> Tuple[Dict[str, List[str]], Dict[str, str]]:
    """Index own official document references for Strict-V2 corpus."""
    ref_to_docs = defaultdict(list)
    doc_to_own_ref = {}

    if not contexts_jsonl.exists():
        return {}, {}

    with open(contexts_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            doc_id = str(row["doc_id"])
            passage = row.get("passage", "")
            link = row.get("link", "")
            own_ref = extract_doc_own_reference(passage, link)
            if own_ref:
                doc_to_own_ref[doc_id] = own_ref
                ref_to_docs[own_ref].append(doc_id)

    for r in ref_to_docs:
        ref_to_docs[r].sort(key=lambda x: int(x) if x.isdigit() else x)

    return dict(ref_to_docs), doc_to_own_ref


def resolve_tier1_anchor(
    query_text: str,
    ref_to_docs: Dict[str, List[str]],
) -> Tuple[Optional[str], str, List[str]]:
    """Resolve query text to at most ONE exact anchor document.
    
    Returns:
        (anchor_doc, resolution_status, extracted_references)
        resolution_status: 'UNIQUE_EXACT_MATCH' | 'NO_REFERENCES' | 'UNMATCHED_REFERENCES' |
                           'AMBIGUOUS_DOC_MATCH' | 'MULTIPLE_ELIGIBLE_DOCS'
    """
    refs = extract_query_references(query_text)
    if not refs:
        return None, "NO_REFERENCES", []

    matched_docs_per_ref = {}
    has_ambiguous_doc_match = False

    for r in refs:
        matching_docs = ref_to_docs.get(r, [])
        if len(matching_docs) == 1:
            matched_docs_per_ref[r] = matching_docs[0]
        elif len(matching_docs) > 1:
            has_ambiguous_doc_match = True

    if has_ambiguous_doc_match:
        return None, "AMBIGUOUS_DOC_MATCH", refs

    distinct_matched_docs = set(matched_docs_per_ref.values())
    if len(distinct_matched_docs) == 0:
        return None, "UNMATCHED_REFERENCES", refs
    elif len(distinct_matched_docs) > 1:
        return None, "MULTIPLE_ELIGIBLE_DOCS", refs
    else:
        anchor_doc = next(iter(distinct_matched_docs))
        return anchor_doc, "UNIQUE_EXACT_MATCH", refs


def apply_tier1_repair(
    d1_ranking: List[str],
    anchor_doc: Optional[str],
) -> Tuple[List[str], str, Optional[str], Optional[str]]:
    """Apply protected injection of unique exact citation anchor into D1 Top-5.
    
    Returns:
        (new_top5, action_type, injected_doc, evicted_doc)
        action_type: 'KEEP_ALREADY_IN_TOP5' | 'EXACT_CITATION_INJECTION' | 'ABSTAIN'
    """
    top5 = list(d1_ranking[:5])
    if anchor_doc is None:
        return top5, "ABSTAIN", None, None

    if anchor_doc in top5:
        return top5, "KEEP_ALREADY_IN_TOP5", None, None

    # Protected injection: replace rank 5 (index 4), preserve ranks 1-4
    evicted_doc = top5[4]
    new_top5 = top5[:4] + [anchor_doc]
    return new_top5, "EXACT_CITATION_INJECTION", anchor_doc, evicted_doc


def run_cal_citation_audit(
    all_ids: List[str],
    queries_label_free: Dict[str, Any],
    d1_predictions: Dict[str, List[str]],
    gold: Dict[str, Set[str]],
    ref_to_docs: Dict[str, List[str]],
) -> Dict[str, Any]:
    """Audit Tier 1 citations across CAL600."""
    total_queries = len(all_ids)
    queries_with_refs = 0
    unique_exact_matches = 0
    already_in_top5 = 0
    interventions = 0
    ambiguous_abstentions = 0
    unmatched_references = 0
    matched_anchor_is_gold = 0
    matched_anchor_is_not_gold = 0

    beneficial_interventions = 0
    harmful_interventions = 0
    neutral_interventions = 0

    details = []

    for q in all_ids:
        qval = queries_label_free[q]
        qtext = qval[0] if isinstance(qval, (list, tuple)) else str(qval)
        anchor_doc, status, refs = resolve_tier1_anchor(qtext, ref_to_docs)

        if refs:
            queries_with_refs += 1

        if status == "AMBIGUOUS_DOC_MATCH" or status == "MULTIPLE_ELIGIBLE_DOCS":
            ambiguous_abstentions += 1
        elif status == "UNMATCHED_REFERENCES":
            unmatched_references += 1
        elif status == "UNIQUE_EXACT_MATCH":
            unique_exact_matches += 1
            is_gold = anchor_doc in gold[q]
            if is_gold:
                matched_anchor_is_gold += 1
            else:
                matched_anchor_is_not_gold += 1

            new_top5, act, inj, evict = apply_tier1_repair(d1_predictions[q], anchor_doc)
            if act == "KEEP_ALREADY_IN_TOP5":
                already_in_top5 += 1
            elif act == "EXACT_CITATION_INJECTION":
                interventions += 1
                orig_hits = len(set(d1_predictions[q][:5]) & gold[q])
                new_hits = len(set(new_top5) & gold[q])
                if new_hits > orig_hits:
                    beneficial_interventions += 1
                    effect = "BENEFICIAL"
                elif new_hits < orig_hits:
                    harmful_interventions += 1
                    effect = "HARMFUL"
                else:
                    neutral_interventions += 1
                    effect = "NEUTRAL"

                details.append({
                    "qid": q,
                    "refs": refs,
                    "anchor_doc": anchor_doc,
                    "anchor_is_gold": is_gold,
                    "evicted_doc": evict,
                    "evicted_is_gold": evict in gold[q],
                    "effect": effect,
                    "orig_hits": orig_hits,
                    "new_hits": new_hits,
                })

    anchor_precision = (
        float(matched_anchor_is_gold / max(1, unique_exact_matches))
        if unique_exact_matches > 0
        else 0.0
    )

    audit_doc = {
        "schema_version": "dsc2026.gemini.huy_d1_selective_repair_v1.exact_citation_audit.v1",
        "experiment_id": "HUY_D1_SELECTIVE_REPAIR_V1",
        "total_queries": total_queries,
        "queries_containing_parsed_references": queries_with_refs,
        "unique_exact_corpus_matches": unique_exact_matches,
        "already_in_top5_count": already_in_top5,
        "intervention_count": interventions,
        "ambiguous_abstentions": ambiguous_abstentions,
        "unmatched_references": unmatched_references,
        "matched_anchor_is_gold_count": matched_anchor_is_gold,
        "matched_anchor_is_not_gold_count": matched_anchor_is_not_gold,
        "anchor_gold_precision": anchor_precision,
        "top5_interventions": {
            "total": interventions,
            "beneficial": beneficial_interventions,
            "harmful": harmful_interventions,
            "neutral": neutral_interventions,
        },
        "intervention_cases": details,
        "gates": {
            "harmful_interventions_eq_0": harmful_interventions == 0,
            "anchor_precision_ge_095": anchor_precision >= 0.95,
            "overall_tier1_cal_pass": (harmful_interventions == 0) and (anchor_precision >= 0.95),
        },
    }
    return audit_doc


def run_v2_citation_shadow(
    v2_ref_to_docs: Dict[str, List[str]],
) -> Dict[str, Any]:
    """Audit Tier 1 citations on Strict-V2 as a shadow generalization evaluation."""
    if not CANONICAL_V2_QUERIES_JSONL.exists():
        return {
            "schema_version": "dsc2026.gemini.huy_d1_selective_repair_v1.v2_citation_shadow.v1",
            "status": "NOT_AVAILABLE",
            "notes": "V2 queries file not found",
        }

    total_queries = 0
    queries_with_refs = 0
    evaluable_matches = []

    with open(CANONICAL_V2_QUERIES_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            total_queries += 1
            row = json.loads(line)
            qid = str(row["qid"])
            qtext = row.get("question", "")
            gold = set(str(g) for g in row.get("gold", []))

            anchor_doc, status, refs = resolve_tier1_anchor(qtext, v2_ref_to_docs)
            if refs:
                queries_with_refs += 1

            if status == "UNIQUE_EXACT_MATCH" and anchor_doc is not None:
                is_gold = anchor_doc in gold
                evaluable_matches.append({
                    "qid": qid,
                    "refs": refs,
                    "anchor_doc": anchor_doc,
                    "is_gold": is_gold,
                })

    evaluable_count = len(evaluable_matches)
    gold_count = sum(1 for m in evaluable_matches if m["is_gold"])
    precision = float(gold_count / max(1, evaluable_count)) if evaluable_count > 0 else 0.0
    gate_evaluated = evaluable_count >= 20
    gate_pass = precision >= 0.90 if gate_evaluated else True

    shadow_doc = {
        "schema_version": "dsc2026.gemini.huy_d1_selective_repair_v1.v2_citation_shadow.v1",
        "experiment_id": "HUY_D1_SELECTIVE_REPAIR_V1",
        "status": "AVAILABLE",
        "sample_size": evaluable_count,
        "reference_query_coverage": {
            "total_queries": total_queries,
            "queries_with_parsed_references": queries_with_refs,
            "coverage_pct": float(queries_with_refs / max(1, total_queries) * 100),
        },
        "unique_exact_matches": evaluable_count,
        "matched_anchor_is_gold_count": gold_count,
        "matched_anchor_gold_precision": precision,
        "hard_gate_eligible": gate_evaluated,
        "gate_pass": gate_pass,
        "notes": "Gate evaluated (>= 20 matches required for hard gate)" if gate_evaluated else "Descriptive only (< 20 matches)",
        "sample_cases": evaluable_matches[:10],
    }
    return shadow_doc
