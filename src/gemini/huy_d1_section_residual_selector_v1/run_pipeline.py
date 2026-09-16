"""End-to-end orchestration runner for HUY_D1_SECTION_RESIDUAL_SELECTOR_V1."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path("D:/Study/DSC2026/sota")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.gemini.huy_d1_section_residual_selector_v1.audit_cache_provenance import (
    run_cache_provenance_audit,
)
from src.gemini.huy_d1_section_residual_selector_v1.audit_one_swap_oracle import (
    compute_one_swap_oracle,
)
from src.gemini.huy_d1_section_residual_selector_v1.baseline_parity import (
    evaluate_d1_baseline,
)
from src.gemini.huy_d1_section_residual_selector_v1.build_artifacts import (
    build_authoritative_artifacts,
)
from src.gemini.huy_d1_section_residual_selector_v1.common import (
    SECTION_CE_CACHE_PKL,
    get_git_status,
    load_cal_data,
    load_pkl,
    seed_everything,
)
from src.gemini.huy_d1_section_residual_selector_v1.evaluate_nested_selector import (
    evaluate_nested_residual_selector,
)
from src.gemini.huy_d1_section_residual_selector_v1.smoke_test import (
    run_synthetic_smoke_test,
)


def main():
    seed_everything(2026)
    print("===================================================================", flush=True)
    print("  EXPERIMENT: HUY_D1_SECTION_RESIDUAL_SELECTOR_V1", flush=True)
    print("===================================================================\n", flush=True)

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

    # Step 2: Cache Provenance Audit
    print(">>> STEP 2: Running Cache Provenance Audit...", flush=True)
    cache_res = run_cache_provenance_audit()
    if cache_res.get("status") != "PASS":
        print("FATAL: Cache provenance audit failed!", flush=True)
        sys.exit("BLOCKED_CACHE_PROVENANCE")
    print("Step 2 PASSED: Cache provenance verified.\n", flush=True)

    # Step 3: D1 Baseline Parity Audit (R0)
    print(">>> STEP 3: Verifying D1 Baseline Parity (R0)...", flush=True)
    base_res, e0_preds, e0_scores = evaluate_d1_baseline()
    if base_res.get("status") != "PASS":
        print("FATAL: D1 baseline parity failed!", flush=True)
        sys.exit("BLOCKED_D1_PARITY")
    print("Step 3 PASSED: Exact D1 48D baseline parity verified.\n", flush=True)

    # Step 4: Diagnostic One-Swap Oracle Audit
    print(">>> STEP 4: Running Diagnostic One-Swap Oracle Audit...", flush=True)
    cal_data = load_cal_data()
    docs, queries, blocks, all_ids, extended, local_views, full_channels_cv, gold, type_rows, cite_rows = cal_data
    sec_scores = load_pkl(SECTION_CE_CACHE_PKL)
    oracle_res = compute_one_swap_oracle(
        e0_preds=e0_preds,
        section_scores=sec_scores,
        gold=gold,
        extended=extended,
        all_ids=all_ids,
        blocks=blocks,
    )
    if oracle_res.get("status") != "PASS":
        print("FATAL: One-swap oracle audit failed!", flush=True)
        sys.exit("BLOCKED_ORACLE_AUDIT")
    print("Step 4 PASSED: One-swap oracle audit verified.\n", flush=True)

    # Step 5: Nested Cross-Fitting & Selector Evaluation (R0 vs R1)
    print(">>> STEP 5: Running Nested Cross-Fitting & Selector Evaluation...", flush=True)
    (
        eval_summary,
        r0_preds,
        r1_preds,
        r0_scores,
        swaps_diagnostic,
        stacking_doc,
        selector_train_doc,
        coefficient_stability_doc,
        bootstrap_doc,
    ) = evaluate_nested_residual_selector()
    print("Step 5 PASSED: Nested evaluation complete.\n", flush=True)

    # Step 6: Build Authoritative Artifacts
    print(">>> STEP 6: Generating Authoritative Artifacts...", flush=True)
    cal_report = build_authoritative_artifacts(
        eval_summary=eval_summary,
        r0_preds=r0_preds,
        r1_preds=r1_preds,
        r0_scores=r0_scores,
        swaps_diagnostic=swaps_diagnostic,
        stacking_doc=stacking_doc,
        selector_train_doc=selector_train_doc,
        coefficient_stability_doc=coefficient_stability_doc,
        bootstrap_doc=bootstrap_doc,
        cal_data=cal_data,
    )
    print("Step 6 PASSED: Artifacts generated.\n", flush=True)

    print("===================================================================", flush=True)
    print(f"  FINAL VERDICT: {eval_summary['verdict']}", flush=True)
    print("===================================================================", flush=True)


if __name__ == "__main__":
    main()
