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
from src.gemini.huy_d1_query_anchored_legal_ref_expansion_v1.audit_strict_v2_provenance import (
    run_strict_v2_provenance_audit,
)
from src.gemini.huy_d1_query_anchored_legal_ref_expansion_v1.build_artifacts import (
    build_authoritative_artifacts,
)
from src.gemini.huy_d1_query_anchored_legal_ref_expansion_v1.common import (
    RES_DIR,
    compute_candidate_fingerprint,
    compute_corpus_fingerprint,
    compute_query_fingerprint,
    get_git_status,
    load_cal_corpus,
    load_cal_generation_inputs,
    load_cal_gold_labels,
    load_v2_generation_inputs,
    load_v2_gold_labels,
    seed_everything,
    sha256_file,
)
from src.gemini.huy_d1_query_anchored_legal_ref_expansion_v1.evaluate_expansion import (
    evaluate_cal_expansion,
    evaluate_v2_expansion,
)
from src.gemini.huy_d1_query_anchored_legal_ref_expansion_v1.legal_ref_indexer import (
    build_legal_reference_index,
)
from src.gemini.huy_d1_query_anchored_legal_ref_expansion_v1.query_anchored_generator import (
    generate_all_cal_additions,
    generate_all_v2_additions,
    seal_additions_artifact,
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

    # Step 2: D1 Baseline Candidate Pool Fingerprint Audit (Generation-only loader)
    print(">>> STEP 2: Running D1 Baseline Candidate Pool Fingerprint Audit...", flush=True)
    pool_audit_res = run_candidate_pool_audit()
    if pool_audit_res.get("status") != "PASS":
        print("FATAL: Candidate pool fingerprint audit failed!", flush=True)
        sys.exit("BLOCKED_POOL_FINGERPRINT")
    print("Step 2 PASSED: Candidate pool fingerprint verified.\n", flush=True)

    # Step 3: Raw Corpus Legal Reference Indexing
    print(">>> STEP 3: Indexing Canonical Legal References from Raw CAL Corpus...", flush=True)
    corpus = load_cal_corpus()
    ref_to_docs, doc_to_own_ref, index_audit = build_legal_reference_index(
        corpus, audit_filename="LEGAL_REFERENCE_INDEX_AUDIT.json"
    )
    if index_audit.get("status") != "PASS":
        print("FATAL: Legal reference index audit failed!", flush=True)
        sys.exit("BLOCKED_INDEX_AUDIT")
    print("Step 3 PASSED: Legal reference index built and verified.\n", flush=True)

    # Step 4: Explicit Legal Relation Graph Construction
    print(">>> STEP 4: Building Explicit Legal Relation Graph...", flush=True)
    out_edges, in_edges, rel_audit = build_explicit_relation_graph(
        corpus, ref_to_docs, doc_to_own_ref, audit_filename="EXPLICIT_RELATION_GRAPH_AUDIT.json"
    )
    if rel_audit.get("status") != "PASS":
        print("FATAL: Legal relation graph audit failed!", flush=True)
        sys.exit("BLOCKED_RELATION_AUDIT")
    print("Step 4 PASSED: Explicit relation graph built and verified.\n", flush=True)

    # Step 5: CAL Candidate Expansion Generation & Seal (LABEL-FREE)
    print(">>> STEP 5: Generating and Sealing CAL Candidate Additions (cap=8, LABEL-FREE)...", flush=True)
    query_texts, blocks, all_ids, extended = load_cal_generation_inputs()
    cal_additions, jsonl_records = generate_all_cal_additions(
        query_texts=query_texts,
        all_ids=all_ids,
        extended=extended,
        ref_to_docs=ref_to_docs,
        out_edges=out_edges,
        in_edges=in_edges,
        cap=8,
    )
    cal_jsonl_path = RES_DIR / "CAL_PER_QUERY_EXPANSION.jsonl"
    cal_seal_doc = seal_additions_artifact(
        dataset_name="CAL600",
        jsonl_path=cal_jsonl_path,
        jsonl_records=jsonl_records,
        query_fp=compute_query_fingerprint(all_ids, query_texts),
        cand_fp=compute_candidate_fingerprint(all_ids, extended),
        corpus_fp=compute_corpus_fingerprint(corpus),
        ref_index_sha256=sha256_file(RES_DIR / "LEGAL_REFERENCE_INDEX_AUDIT.json"),
        rel_graph_sha256=sha256_file(RES_DIR / "EXPLICIT_RELATION_GRAPH_AUDIT.json"),
        cap=8,
    )
    print("Step 5 PASSED: CAL candidate additions materialized and sealed.\n", flush=True)

    # Step 6: CAL Candidate Recall & Noise Evaluation (ONLY NOW LOAD GOLD LABELS)
    print(">>> STEP 6: Evaluating CAL Candidate Recall (Post-Seal Gold Evaluation)...", flush=True)
    gold = load_cal_gold_labels(all_ids)
    cal_results_doc, noise_doc = evaluate_cal_expansion(
        query_texts=query_texts,
        blocks=blocks,
        all_ids=all_ids,
        extended=extended,
        gold=gold,
        cal_additions=cal_additions,
        jsonl_records=jsonl_records,
        seal_doc=cal_seal_doc,
    )
    print("Step 6 PASSED: CAL candidate recall evaluation complete.\n", flush=True)

    # Step 7: Strict-V2 Provenance Hard Audit
    print(">>> STEP 7: Running Strict-V2 Provenance Hard Audit...", flush=True)
    v2_prov_doc = run_strict_v2_provenance_audit()
    if v2_prov_doc.get("status") != "PASS":
        print("WARNING: Strict-V2 provenance audit failed! Marking V2_SHADOW_UNAVAILABLE.", flush=True)
        v2_shadow_doc = {
            "schema_version": "dsc2026.gemini.huy_d1_query_anchored_legal_ref_expansion_v1.v2_shadow.v1",
            "experiment_id": "HUY_D1_QUERY_ANCHORED_LEGAL_REF_EXPANSION_V1",
            "status": "V2_SHADOW_UNAVAILABLE",
            "reason": "Strict-V2 provenance audit failed.",
        }
        (RES_DIR / "V2_SHADOW_EXPANSION_RESULTS.json").write_text(
            json.dumps(v2_shadow_doc, indent=2), encoding="utf-8"
        )
        v2_seal_doc = None
    else:
        print("Step 7 PASSED: Strict-V2 provenance hard audit verified.\n", flush=True)

        # Step 8: Strict-V2 Shadow Generalization (LABEL-FREE GENERATION -> SEAL -> GOLD EVALUATION)
        print(">>> STEP 8: Running Strict-V2 Shadow Generalization...", flush=True)
        v2_corpus, v2_queries, v2_qids, v2_pools = load_v2_generation_inputs()
        ref_to_docs_v2, doc_to_own_ref_v2, _ = build_legal_reference_index(
            v2_corpus, audit_filename="V2_LEGAL_REFERENCE_INDEX_AUDIT.json"
        )
        out_edges_v2, in_edges_v2, _ = build_explicit_relation_graph(
            v2_corpus, ref_to_docs_v2, doc_to_own_ref_v2, audit_filename="V2_EXPLICIT_RELATION_GRAPH_AUDIT.json"
        )
        v2_additions, v2_jsonl_records = generate_all_v2_additions(
            query_texts=v2_queries,
            v2_qids=v2_qids,
            candidate_pools=v2_pools,
            ref_to_docs=ref_to_docs_v2,
            out_edges=out_edges_v2,
            in_edges=in_edges_v2,
            cap=8,
        )
        v2_jsonl_path = RES_DIR / "V2_PER_QUERY_EXPANSION.jsonl"
        v2_seal_doc = seal_additions_artifact(
            dataset_name="V2",
            jsonl_path=v2_jsonl_path,
            jsonl_records=v2_jsonl_records,
            query_fp=compute_query_fingerprint(v2_qids, v2_queries),
            cand_fp=compute_candidate_fingerprint(v2_qids, v2_pools),
            corpus_fp=compute_corpus_fingerprint(v2_corpus),
            ref_index_sha256=sha256_file(RES_DIR / "V2_LEGAL_REFERENCE_INDEX_AUDIT.json"),
            rel_graph_sha256=sha256_file(RES_DIR / "V2_EXPLICIT_RELATION_GRAPH_AUDIT.json"),
            cap=8,
        )
        v2_gold = load_v2_gold_labels(v2_qids)
        v2_shadow_doc = evaluate_v2_expansion(
            v2_qids=v2_qids,
            candidate_pools=v2_pools,
            v2_additions=v2_additions,
            v2_gold=v2_gold,
            seal_doc=v2_seal_doc,
        )
        print("Step 8 PASSED: Strict-V2 shadow generalization evaluated.\n", flush=True)

    # Step 9: Build Authoritative Artifacts
    print(">>> STEP 9: Building Authoritative Artifacts, Audit, and Decision Report...", flush=True)
    art_res = build_authoritative_artifacts(
        cal_results=cal_results_doc,
        noise_audit=noise_doc,
        v2_shadow=v2_shadow_doc,
        all_ids=all_ids,
        extended=extended,
        queries=query_texts,
        cal_seal_doc=cal_seal_doc,
        v2_prov_doc=v2_prov_doc,
        v2_seal_doc=v2_seal_doc,
    )
    print("Step 9 PASSED: Authoritative artifacts generated.\n", flush=True)

    print("===================================================================", flush=True)
    print(f"  FINAL VERDICT: {art_res['verdict']}", flush=True)
    print("===================================================================", flush=True)


if __name__ == "__main__":
    main()
