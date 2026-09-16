"""End-to-end orchestration runner for HUY_D1_QUERY_ANCHORED_LEGAL_REF_EXPANSION_V1."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path("D:/Study/DSC2026/sota")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.gemini.huy_d1_query_anchored_legal_ref_expansion_v1.audit_pool_fingerprint import (
    run_candidate_pool_audit,
)
from src.gemini.huy_d1_query_anchored_legal_ref_expansion_v1.build_artifacts import (
    build_authoritative_artifacts,
)
from src.gemini.huy_d1_query_anchored_legal_ref_expansion_v1.common import (
    get_git_status,
    load_cal_corpus,
    load_cal_data,
    seed_everything,
)
from src.gemini.huy_d1_query_anchored_legal_ref_expansion_v1.evaluate_expansion import (
    evaluate_cal_expansion,
    evaluate_v2_shadow_generalization,
)
from src.gemini.huy_d1_query_anchored_legal_ref_expansion_v1.legal_ref_indexer import (
    build_legal_reference_index,
)
from src.gemini.huy_d1_query_anchored_legal_ref_expansion_v1.query_anchored_generator import (
    generate_all_cal_additions,
)
from src.gemini.huy_d1_query_anchored_legal_ref_expansion_v1.relation_graph import (
    build_explicit_relation_graph,
)
from src.gemini.huy_d1_query_anchored_legal_ref_expansion_v1.smoke_test import (
    run_synthetic_smoke_test,
)


def main():
    seed_everything(2026)
    print("===================================================================", flush=True)
    print("  EXPERIMENT: HUY_D1_QUERY_ANCHORED_LEGAL_REF_EXPANSION_V1", flush=True)
    print("===================================================================", flush=True)

    # Step 0: Hard Git Gate Audit
    print(">>> STEP 0: Git Provenance Hard Gate Audit...", flush=True)
    git_info = get_git_status()
    print(f"  HEAD:        {git_info.get('head_commit')}", flush=True)
    print(f"  origin/main: {git_info.get('origin_main_commit')}", flush=True)
    print(f"  Parity:      {git_info.get('parity')}", flush=True)
    print(f"  Clean:       {git_info.get('status_clean')}", flush=True)

    if not git_info.get("parity") or not git_info.get("status_clean"):
        print("\nFATAL: Git hard gate failed! Uncommitted or unpushed changes detected.", flush=True)
        print("Auditable evidence requires HEAD == origin/main and clean working tree.", flush=True)
        print("STATUS: UNPUSHED_NOT_AUDITABLE\n", flush=True)
        sys.exit("UNPUSHED_NOT_AUDITABLE")

    print("Step 0 PASSED: Git parity and clean tree verified.\n", flush=True)

    # Step 1: Synthetic Smoke Test
    print(">>> STEP 1: Running Synthetic Smoke Test...", flush=True)
    smoke_res = run_synthetic_smoke_test()
    if smoke_res.get("status") != "PASS":
        print("FATAL: Synthetic smoke test failed!", flush=True)
        sys.exit(1)
    print("Step 1 PASSED: Synthetic smoke test clean.\n", flush=True)

    # Step 2: D1 Baseline Candidate Pool Fingerprint Audit
    print(">>> STEP 2: Running D1 Baseline Candidate Pool Fingerprint Audit...", flush=True)
    pool_audit_res = run_candidate_pool_audit()
    if pool_audit_res.get("status") != "PASS":
        print("FATAL: Candidate pool fingerprint audit failed!", flush=True)
        sys.exit("BLOCKED_POOL_FINGERPRINT")
    print("Step 2 PASSED: Candidate pool fingerprint verified.\n", flush=True)

    # Step 3: Raw Corpus Legal Reference Indexing
    print(">>> STEP 3: Indexing Canonical Legal References from Raw Corpus...", flush=True)
    corpus = load_cal_corpus()
    ref_to_docs, doc_to_own_ref, index_audit = build_legal_reference_index(corpus)
    if index_audit.get("status") != "PASS":
        print("FATAL: Legal reference index audit failed!", flush=True)
        sys.exit("BLOCKED_INDEX_AUDIT")
    print("Step 3 PASSED: Legal reference index built and verified.\n", flush=True)

    # Step 4: Explicit Legal Relation Graph Construction
    print(">>> STEP 4: Building Explicit Legal Relation Graph...", flush=True)
    out_edges, in_edges, rel_audit = build_explicit_relation_graph(corpus, ref_to_docs, doc_to_own_ref)
    if rel_audit.get("status") != "PASS":
        print("FATAL: Legal relation graph audit failed!", flush=True)
        sys.exit("BLOCKED_RELATION_AUDIT")
    print("Step 4 PASSED: Explicit relation graph built and verified.\n", flush=True)

    # Step 5: CAL Candidate Expansion Generation
    print(">>> STEP 5: Generating CAL Candidate Additions (cap=8)...", flush=True)
    queries, blocks, all_ids, extended, gold = load_cal_data()
    cal_additions, jsonl_records = generate_all_cal_additions(
        queries=queries,
        all_ids=all_ids,
        extended=extended,
        ref_to_docs=ref_to_docs,
        out_edges=out_edges,
        in_edges=in_edges,
        cap=8,
    )
    print("Step 5 PASSED: CAL candidate additions materialized.\n", flush=True)

    # Step 6: Strict-V2 Shadow Generalization
    print(">>> STEP 6: Running Strict-V2 Shadow Generalization...", flush=True)
    v2_shadow_doc = evaluate_v2_shadow_generalization()
    print("Step 6 PASSED: Strict-V2 shadow generalization evaluated.\n", flush=True)

    # Step 7: CAL Candidate Recall & Noise Audit
    print(">>> STEP 7: Evaluating CAL Candidate Recall & Expansion Noise...", flush=True)
    cal_results_doc, noise_doc = evaluate_cal_expansion(
        queries=queries,
        blocks=blocks,
        all_ids=all_ids,
        extended=extended,
        gold=gold,
        cal_additions=cal_additions,
        jsonl_records=jsonl_records,
    )
    print("Step 7 PASSED: CAL candidate recall evaluation complete.\n", flush=True)

    # Step 8: Build Authoritative Artifacts
    print(">>> STEP 8: Building Authoritative Artifacts, Audit, and Decision Report...", flush=True)
    art_res = build_authoritative_artifacts(
        cal_results=cal_results_doc,
        noise_audit=noise_doc,
        v2_shadow=v2_shadow_doc,
        all_ids=all_ids,
        extended=extended,
        queries=queries,
    )
    print("Step 8 PASSED: Authoritative artifacts generated.\n", flush=True)

    print("===================================================================", flush=True)
    print(f"  FINAL VERDICT: {art_res['verdict']}", flush=True)
    print("===================================================================", flush=True)


if __name__ == "__main__":
    main()
