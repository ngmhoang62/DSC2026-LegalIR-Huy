"""End-to-end orchestration runner for HUY_D1_JINA_EVIDENCE_UNION_V1."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path("D:/Study/DSC2026/sota")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.gemini.huy_d1_jina_evidence_union_v1.audit_old_jina_parity import (
    run_old_jina_parity_audit,
)
from src.gemini.huy_d1_jina_evidence_union_v1.audit_score_semantics import (
    run_score_semantics_audit,
)
from src.gemini.huy_d1_jina_evidence_union_v1.baseline_parity import (
    evaluate_d1_baseline,
)
from src.gemini.huy_d1_jina_evidence_union_v1.build_artifacts import (
    build_authoritative_artifacts,
)
from src.gemini.huy_d1_jina_evidence_union_v1.build_evidence_union import (
    build_evidence_union_channel,
)
from src.gemini.huy_d1_jina_evidence_union_v1.common import (
    get_git_status,
    load_cal_data,
    seed_everything,
)
from src.gemini.huy_d1_jina_evidence_union_v1.evaluate_e0_e1_lobo import (
    evaluate_e0_e1_lobo,
)
from src.gemini.huy_d1_jina_evidence_union_v1.evaluate_standalone import (
    evaluate_standalone_experts,
)
from src.gemini.huy_d1_jina_evidence_union_v1.smoke_test import (
    run_synthetic_smoke_test,
)


def main():
    seed_everything(2026)
    print("===================================================================", flush=True)
    print("  EXPERIMENT: HUY_D1_JINA_EVIDENCE_UNION_V1", flush=True)
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

    # Step 2: Score Semantics Audit
    print(">>> STEP 2: Running Score Semantics Audit...", flush=True)
    semantics_res = run_score_semantics_audit()
    if semantics_res.get("status") != "PASS":
        print("FATAL: Score semantics audit failed!", flush=True)
        sys.exit("BLOCKED_SCORE_SEMANTICS")
    print("Step 2 PASSED: Score semantics verified.\n", flush=True)

    # Step 3: Frozen Old Jina Parity Audit
    print(">>> STEP 3: Running Frozen Jina Parity Audit...", flush=True)
    parity_res = run_old_jina_parity_audit(min_pairs=64)
    if parity_res.get("status") != "PASS":
        print("FATAL: Old Jina parity audit failed!", flush=True)
        sys.exit("BLOCKED_OLD_JINA_PARITY")
    print("Step 3 PASSED: Frozen Jina-FT parity verified.\n", flush=True)

    # Step 4: D1 Baseline Parity Audit
    print(">>> STEP 4: Verifying D1 Baseline Parity (E0)...", flush=True)
    base_res, _, _ = evaluate_d1_baseline()
    if base_res.get("status") != "PASS":
        print("FATAL: D1 baseline parity failed!", flush=True)
        sys.exit("BLOCKED_D1_PARITY")
    print("Step 4 PASSED: Exact D1 48D baseline parity verified.\n", flush=True)

    # Step 5: Build Evidence Union Channel
    print(">>> STEP 5: Building Evidence Union Channel...", flush=True)
    union_res = build_evidence_union_channel()
    print("Step 5 PASSED: Evidence union channel built.\n", flush=True)

    # Step 6: Standalone Expert Evaluation
    print(">>> STEP 6: Evaluating Standalone Experts...", flush=True)
    standalone_res = evaluate_standalone_experts()
    print("Step 6 PASSED: Standalone evaluation complete.\n", flush=True)

    # Step 7: LOBO Evaluation (E0 vs E1)
    print(">>> STEP 7: Evaluating E0 vs E1 LOBO Fusion...", flush=True)
    (
        eval_summary,
        preds_e0,
        preds_e1,
        scores_e0,
        scores_e1,
        full_rankings_e0,
        full_rankings_e1,
        full_scores_e0,
        full_scores_e1,
        boundary_diagnostics,
        bootstrap_results,
    ) = evaluate_e0_e1_lobo()
    print("Step 7 PASSED: LOBO evaluation complete.\n", flush=True)

    # Step 8: Build Authoritative Artifacts
    print(">>> STEP 8: Generating Authoritative Artifacts...", flush=True)
    cal_data = load_cal_data()
    authoritative_report = build_authoritative_artifacts(
        eval_summary=eval_summary,
        preds_e0=preds_e0,
        preds_e1=preds_e1,
        scores_e0=scores_e0,
        scores_e1=scores_e1,
        full_rankings_e0=full_rankings_e0,
        full_rankings_e1=full_rankings_e1,
        full_scores_e0=full_scores_e0,
        full_scores_e1=full_scores_e1,
        boundary_diagnostics=boundary_diagnostics,
        bootstrap_results=bootstrap_results,
        cal_data=cal_data,
    )
    print("Step 8 PASSED: Artifacts generated.\n", flush=True)

    print("===================================================================", flush=True)
    print(f"  FINAL VERDICT: {eval_summary['verdict']}", flush=True)
    print("===================================================================", flush=True)


if __name__ == "__main__":
    main()
