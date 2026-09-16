"""Master pipeline orchestrator for HUY_D1_LEGAL_SECTION_RETRIEVAL_EXPANSION_V1.

Orchestrates:
Step 0: Git hard gate check (HEAD == origin/main, clean tree)
Step 1: Parser parity audit
Step 2: Lexical provenance audit
Step 3: Baseline parity audit (CAL600 & Strict-V2)
Step 4: Synthetic smoke test
Step 5: Candidate addition generation & sealing (CAL600 & Strict-V2)
Step 6: Recall evaluation & forensic analysis
Step 7: Generator complementarity audit
Step 8: Build artifacts & DECISION.md
Step 9: Report consistency audit
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from .audit_baseline_parity import run_baseline_parity_audits
from .audit_complementarity import run_complementarity_audit
from .audit_lexical_provenance import run_lexical_provenance_audit
from .audit_parser_parity import run_parser_parity_audit
from .build_artifacts import (
    build_decision_report,
    build_final_run_provenance,
    build_source_provenance,
    run_report_consistency_audit,
)
from .candidate_generator import run_all_candidate_generation
from .common import RES_DIR, get_git_status
from .evaluate_expansion import run_all_evaluations
from .smoke_test import run_smoke_test


def run_pipeline() -> None:
    t_start = time.perf_counter()
    print("=" * 80, flush=True)
    print("STARTING EXPERIMENT PIPELINE: HUY_D1_LEGAL_SECTION_RETRIEVAL_EXPANSION_V1", flush=True)
    print("=" * 80, flush=True)

    # STEP 0: Git hard gate
    print("\n[STEP 0] Checking Git Hard Gate...", flush=True)
    git_info = get_git_status()
    print(f"  HEAD Commit:   {git_info['head_commit']}")
    print(f"  Origin Commit: {git_info['origin_main_commit']}")
    print(f"  Origin Parity: {git_info['parity']}")
    print(f"  Status Clean:  {git_info['status_clean']}")

    if not git_info["parity"] or not git_info["status_clean"]:
        print(
            "\n[GIT HARD GATE FAILED] Working tree is dirty or HEAD != origin/main!\n"
            f"Porcelain output:\n{git_info['porcelain_output']}\n"
            "Aborting execution to maintain strict auditability: UNPUSHED_NOT_AUDITABLE.",
            flush=True,
        )
        sys.exit(1)
    print("[STEP 0] Git Hard Gate PASSED.", flush=True)

    # STEP 1: Parser parity audit
    print("\n[STEP 1] Running Parser Parity Audit...", flush=True)
    parser_res = run_parser_parity_audit()
    if parser_res.get("status") != "PASS":
        print(f"[FATAL] Parser parity audit failed: {parser_res}", flush=True)
        sys.exit(1)

    # STEP 2: Lexical provenance audit
    print("\n[STEP 2] Running Lexical Provenance Audit...", flush=True)
    lex_res = run_lexical_provenance_audit()
    if lex_res.get("status") != "PASS":
        print(f"[FATAL] Lexical provenance audit failed: {lex_res}", flush=True)
        sys.exit(1)

    # STEP 3: Baseline parity audit
    print("\n[STEP 3] Running Baseline Parity Audits (CAL & V2)...", flush=True)
    base_ok = run_baseline_parity_audits()
    if not base_ok:
        print("[FATAL] Baseline parity audit failed!", flush=True)
        sys.exit(1)

    # STEP 4: Synthetic smoke test
    print("\n[STEP 4] Running Synthetic Smoke Test...", flush=True)
    smoke_ok = run_smoke_test()
    if not smoke_ok:
        print("[FATAL] Smoke test failed!", flush=True)
        sys.exit(1)

    # STEP 5: Candidate addition generation & sealing (ZERO REUSE FRESH RUN)
    print("\n[STEP 5] Generating and Sealing Candidate Additions with FRESH REBUILD (CAL & V2)...", flush=True)
    run_all_candidate_generation(force_rebuild=True)

    # STEP 6: Recall evaluation & forensics
    print("\n[STEP 6] Evaluating Expansion Recalls and Forensics...", flush=True)
    eval_res = run_all_evaluations()

    # STEP 7: Generator complementarity audit
    print("\n[STEP 7] Running Generator Complementarity Audit...", flush=True)
    comp_res = run_complementarity_audit()

    # STEP 8: Build artifacts, final run provenance & DECISION.md
    print("\n[STEP 8] Building Authoritative Artifacts and DECISION.md...", flush=True)
    build_source_provenance()
    build_final_run_provenance()
    build_decision_report()

    # STEP 9: Report consistency audit
    print("\n[STEP 9] Running Report Consistency Audit...", flush=True)
    audit_res = run_report_consistency_audit()
    if audit_res.get("status") != "PASS":
        print(f"[FATAL] Report consistency audit failed: {audit_res}", flush=True)
        sys.exit(1)

    total_time = time.perf_counter() - t_start
    print("\n" + "=" * 80, flush=True)
    print(f"PIPELINE COMPLETED SUCCESSFULLY IN {total_time:.2f}s", flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    run_pipeline()
