"""Candidate-Coverage and Generalization Evaluation for HUY_D1_QUERY_ANCHORED_LEGAL_REF_EXPANSION_V1."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import numpy as np

ROOT = Path("D:/Study/DSC2026/sota")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.gemini.huy_d1_query_anchored_legal_ref_expansion_v1.common import (
    RES_DIR,
    load_cal_data,
    load_v2_data,
)
from src.gemini.huy_d1_query_anchored_legal_ref_expansion_v1.legal_ref_indexer import (
    build_legal_reference_index,
)
from src.gemini.huy_d1_query_anchored_legal_ref_expansion_v1.query_anchored_generator import (
    generate_query_additions,
)
from src.gemini.huy_d1_query_anchored_legal_ref_expansion_v1.relation_graph import (
    build_explicit_relation_graph,
)


def evaluate_cal_expansion(
    queries: Dict[str, Any],
    blocks: Dict[str, List[str]],
    all_ids: List[str],
    extended: Dict[str, List[str]],
    gold: Dict[str, Set[str]],
    cal_additions: Dict[str, List[str]],
    jsonl_records: List[dict],
) -> Tuple[dict, dict]:
    print("=== EVALUATING CAL CANDIDATE EXPANSION COVERAGE ===", flush=True)
    RES_DIR.mkdir(parents=True, exist_ok=True)

    qid_to_record = {r["qid"]: r for r in jsonl_records}

    # Macro recall per query
    orig_recalls = []
    exp_recalls = []

    recovered_outside_records = []
    direct_recoveries_count = 0
    relation_recoveries_count = 0

    for q in all_ids:
        orig_pool = set(extended[q])
        exp_pool = orig_pool | set(cal_additions[q])
        g = gold[q]

        r_orig = len(orig_pool & g) / max(1, len(g))
        r_exp = len(exp_pool & g) / max(1, len(g))

        orig_recalls.append(r_orig)
        exp_recalls.append(r_exp)

        # Check recovered outside golds
        newly_recovered = (exp_pool & g) - (orig_pool & g)
        if newly_recovered:
            rec = qid_to_record.get(q, {})
            detail_by_doc = {d["doc_id"]: d for d in rec.get("addition_details", [])}

            for rec_doc in newly_recovered:
                d_info = detail_by_doc.get(rec_doc, {})
                rec_src = d_info.get("addition_type", "UNKNOWN")
                if rec_src == "DIRECT_REFERENCE_MATCH":
                    direct_recoveries_count += 1
                elif rec_src == "RELATION_NEIGHBOR":
                    relation_recoveries_count += 1

                recovered_outside_records.append({
                    "qid": q,
                    "query_text": queries[q][0],
                    "recovered_gold_doc_id": rec_doc,
                    "recovery_source": rec_src,
                    "anchor_ref": d_info.get("anchor_ref"),
                    "anchor_doc": d_info.get("anchor_doc"),
                    "relation_family": d_info.get("relation_family"),
                    "relation_direction": d_info.get("relation_direction"),
                    "evidence_snippet": d_info.get("evidence_snippet"),
                })

    orig_pooled = float(np.mean(orig_recalls))
    exp_pooled = float(np.mean(exp_recalls))
    delta_pooled = exp_pooled - orig_pooled

    # Block breakdowns
    block_results = {}
    for b in sorted(blocks.keys()):
        b_ids = blocks[b]
        b_orig = float(np.mean([orig_recalls[all_ids.index(q)] for q in b_ids]))
        b_exp = float(np.mean([exp_recalls[all_ids.index(q)] for q in b_ids]))
        block_results[b] = {
            "original_candidate_recall": b_orig,
            "expanded_candidate_recall": b_exp,
            "delta": b_exp - b_orig,
        }

    # Single vs Multi gold
    single_ids = [q for q in all_ids if len(gold[q]) == 1]
    multi_ids = [q for q in all_ids if len(gold[q]) > 1]

    single_orig = float(np.mean([orig_recalls[all_ids.index(q)] for q in single_ids]))
    single_exp = float(np.mean([exp_recalls[all_ids.index(q)] for q in single_ids]))
    multi_orig = float(np.mean([orig_recalls[all_ids.index(q)] for q in multi_ids]))
    multi_exp = float(np.mean([exp_recalls[all_ids.index(q)] for q in multi_ids]))

    unique_recovered_queries = len(set(r["qid"] for r in recovered_outside_records))

    cal_results_doc = {
        "schema_version": "dsc2026.gemini.huy_d1_query_anchored_legal_ref_expansion_v1.cal_results.v1",
        "experiment_id": "HUY_D1_QUERY_ANCHORED_LEGAL_REF_EXPANSION_V1",
        "query_macro_metrics": {
            "original_candidate_recall": orig_pooled,
            "expanded_candidate_recall": exp_pooled,
            "candidate_recall_gain": delta_pooled,
        },
        "block_candidate_recalls": block_results,
        "single_gold_candidate_recall": {
            "original": single_orig,
            "expanded": single_exp,
            "delta": single_exp - single_orig,
        },
        "multi_gold_candidate_recall": {
            "original": multi_orig,
            "expanded": multi_exp,
            "delta": multi_exp - multi_orig,
        },
        "outside_gold_recovery": {
            "recovered_outside_gold_occurrences": len(recovered_outside_records),
            "recovered_queries_count": unique_recovered_queries,
            "direct_reference_recoveries": direct_recoveries_count,
            "relation_neighbor_recoveries": relation_recoveries_count,
            "recovered_cases": recovered_outside_records,
        },
    }

    out_path = RES_DIR / "CAL_CANDIDATE_EXPANSION_RESULTS.json"
    out_path.write_text(json.dumps(cal_results_doc, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {out_path}", flush=True)
    # Also write alias for backwards compatibility
    (RES_DIR / "CAL_EXPANSION_RESULTS.json").write_text(json.dumps(cal_results_doc, indent=2, ensure_ascii=False), encoding="utf-8")

    # Save OUTSIDE_POOL_GOLD_RECOVERY_CASES.json
    cases_doc = {
        "schema_version": "dsc2026.gemini.huy_d1_query_anchored_legal_ref_expansion_v1.recovery_cases.v1",
        "experiment_id": "HUY_D1_QUERY_ANCHORED_LEGAL_REF_EXPANSION_V1",
        "total_recovered_cases": len(recovered_outside_records),
        "recovered_queries_count": unique_recovered_queries,
        "direct_reference_recoveries": direct_recoveries_count,
        "relation_neighbor_recoveries": relation_recoveries_count,
        "recovered_cases": recovered_outside_records,
    }
    cases_path = RES_DIR / "OUTSIDE_POOL_GOLD_RECOVERY_CASES.json"
    cases_path.write_text(json.dumps(cases_doc, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {cases_path}", flush=True)

    # Expansion noise audit
    all_additions_counts = [len(cal_additions[q]) for q in all_ids]
    triggered_qids = [q for q in all_ids if len(cal_additions[q]) > 0]
    triggered_counts = [len(cal_additions[q]) for q in triggered_qids]

    queries_with_refs = [q for q in all_ids if qid_to_record[q].get("extracted_references")]

    total_adds = sum(all_additions_counts)
    gold_additions_count = sum(
        sum(1 for d in cal_additions[q] if d in gold[q]) for q in all_ids
    )
    triggered_with_gold_count = sum(
        1 for q in triggered_qids if any(d in gold[q] for d in cal_additions[q])
    )

    all_details = [
        d
        for r in jsonl_records
        for d in r.get("addition_details", [])
    ]
    direct_adds_count = sum(1 for d in all_details if d.get("addition_type") == "DIRECT_REFERENCE_MATCH")
    rel_adds_count = sum(1 for d in all_details if d.get("addition_type") == "RELATION_NEIGHBOR")

    noise_doc = {
        "schema_version": "dsc2026.gemini.huy_d1_query_anchored_legal_ref_expansion_v1.noise_audit.v1",
        "experiment_id": "HUY_D1_QUERY_ANCHORED_LEGAL_REF_EXPANSION_V1",
        "total_queries": len(all_ids),
        "queries_containing_legal_references": len(queries_with_refs),
        "triggered_queries_count": len(triggered_qids),
        "total_new_additions": total_adds,
        "additions_per_triggered_query": {
            "mean": float(np.mean(triggered_counts)) if triggered_counts else 0.0,
            "median": float(np.median(triggered_counts)) if triggered_counts else 0.0,
            "p95": float(np.percentile(triggered_counts, 95)) if triggered_counts else 0.0,
            "max": int(np.max(triggered_counts)) if triggered_counts else 0,
        },
        "breakdown_by_addition_type": {
            "direct_reference_additions": direct_adds_count,
            "relation_neighbor_additions": rel_adds_count,
        },
        "precision_and_gold_density": {
            "gold_additions_count": gold_additions_count,
            "fraction_additions_that_are_gold": float(gold_additions_count / max(1, total_adds)),
            "triggered_queries_with_at_least_one_gold": triggered_with_gold_count,
            "fraction_triggered_queries_with_gold": float(triggered_with_gold_count / max(1, len(triggered_qids))),
        },
    }

    noise_path = RES_DIR / "EXPANSION_NOISE_AUDIT.json"
    noise_path.write_text(json.dumps(noise_doc, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {noise_path}", flush=True)

    print(f"CAL Candidate Recall: {orig_pooled:.6f} -> {exp_pooled:.6f} (Delta: {delta_pooled:+.6f})", flush=True)
    print(f"Recovered Outside Golds: {len(recovered_outside_records)} across {unique_recovered_queries} queries", flush=True)
    print(f"  Direct: {direct_recoveries_count}, Relation: {relation_recoveries_count}", flush=True)
    print(f"Total New Additions: {total_adds} across {len(triggered_qids)} triggered queries (Precision: {noise_doc['precision_and_gold_density']['fraction_additions_that_are_gold']:.1%})\n", flush=True)

    return cal_results_doc, noise_doc


def evaluate_v2_shadow_generalization() -> dict:
    print("=== EVALUATING STRICT-V2 SHADOW GENERALIZATION ===", flush=True)
    RES_DIR.mkdir(parents=True, exist_ok=True)

    v2_corpus, v2_queries, v2_gold, v2_pools = load_v2_data()

    if not v2_corpus or not v2_queries or not v2_pools:
        print("Strict-V2 data missing or unprovenanced! Setting V2_SHADOW_UNAVAILABLE.", flush=True)
        v2_doc = {
            "schema_version": "dsc2026.gemini.huy_d1_query_anchored_legal_ref_expansion_v1.v2_shadow.v1",
            "experiment_id": "HUY_D1_QUERY_ANCHORED_LEGAL_REF_EXPANSION_V1",
            "status": "V2_SHADOW_UNAVAILABLE",
            "reason": "Corpus, queries, or candidate pool files missing.",
        }
        out_path = RES_DIR / "V2_SHADOW_EXPANSION_RESULTS.json"
        out_path.write_text(json.dumps(v2_doc, indent=2, ensure_ascii=False), encoding="utf-8")
        return v2_doc

    print(f"Loaded V2 Data: {len(v2_corpus)} docs, {len(v2_queries)} queries, {len(v2_pools)} candidate pools", flush=True)

    # 1. Build legal reference index on V2 corpus
    ref_to_docs_v2, doc_to_own_ref_v2, _ = build_legal_reference_index(v2_corpus)

    # 2. Build relation graph on V2 corpus
    out_edges_v2, in_edges_v2, _ = build_explicit_relation_graph(v2_corpus, ref_to_docs_v2, doc_to_own_ref_v2)

    # 3. Run generator on all V2 queries
    v2_qids = sorted(v2_queries.keys(), key=lambda x: int(x) if x.isdigit() else x)
    v2_orig_recalls = []
    v2_exp_recalls = []
    v2_recovered_outside = 0
    v2_benefited_queries = 0
    v2_triggered_count = 0
    v2_total_additions = 0

    for q in v2_qids:
        q_text = v2_queries[q]
        orig_pool = set(v2_pools[q])
        g = v2_gold[q]

        adds, details, _ = generate_query_additions(
            query_text=q_text,
            existing_pool=orig_pool,
            ref_to_docs=ref_to_docs_v2,
            out_edges=out_edges_v2,
            in_edges=in_edges_v2,
            cap=8,
        )

        exp_pool = orig_pool | set(adds)
        if adds:
            v2_triggered_count += 1
            v2_total_additions += len(adds)

        r_orig = len(orig_pool & g) / max(1, len(g))
        r_exp = len(exp_pool & g) / max(1, len(g))

        v2_orig_recalls.append(r_orig)
        v2_exp_recalls.append(r_exp)

        newly_rec = (exp_pool & g) - (orig_pool & g)
        if newly_rec:
            v2_benefited_queries += 1
            v2_recovered_outside += len(newly_rec)

    v2_orig_mean = float(np.mean(v2_orig_recalls))
    v2_exp_mean = float(np.mean(v2_exp_recalls))
    v2_delta = v2_exp_mean - v2_orig_mean

    v2_doc = {
        "schema_version": "dsc2026.gemini.huy_d1_query_anchored_legal_ref_expansion_v1.v2_shadow.v1",
        "experiment_id": "HUY_D1_QUERY_ANCHORED_LEGAL_REF_EXPANSION_V1",
        "status": "AVAILABLE",
        "total_queries": len(v2_qids),
        "triggered_queries_count": v2_triggered_count,
        "total_additions_count": v2_total_additions,
        "candidate_recall": {
            "original": v2_orig_mean,
            "expanded": v2_exp_mean,
            "delta": v2_delta,
        },
        "recovered_outside_gold_occurrences": v2_recovered_outside,
        "benefited_queries_count": v2_benefited_queries,
    }

    out_path = RES_DIR / "V2_SHADOW_EXPANSION_RESULTS.json"
    out_path.write_text(json.dumps(v2_doc, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {out_path}", flush=True)

    print(f"V2 Shadow Candidate Recall: {v2_orig_mean:.6f} -> {v2_exp_mean:.6f} (Delta: {v2_delta:+.6f})", flush=True)
    print(f"V2 Recovered Outside Golds: {v2_recovered_outside} across {v2_benefited_queries} queries\n", flush=True)

    return v2_doc
