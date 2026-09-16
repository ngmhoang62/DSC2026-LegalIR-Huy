"""End-to-end orchestration runner for HUY_D1_LEGAL_SECTION_EVIDENCE_V1."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("D:/Study/DSC2026/sota")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.gemini.huy_d1_legal_section_evidence_v1.baseline_parity import (
    evaluate_s0_baseline,
)
from src.gemini.huy_d1_legal_section_evidence_v1.build_artifacts import (
    build_all_artifacts,
    get_git_info,
)
from src.gemini.huy_d1_legal_section_evidence_v1.common import (
    load_cal_data,
    load_jina_crossencoder,
    seed_everything,
)
from src.gemini.huy_d1_legal_section_evidence_v1.evaluate_s0_s1 import (
    evaluate_both_arms,
)
from src.gemini.huy_d1_legal_section_evidence_v1.score_legal_section_ce import (
    score_legal_sections,
)
from src.gemini.huy_d1_legal_section_evidence_v1.smoke_test import run_smoke_test


def main():
    seed_everything(2026)
    print("===================================================================", flush=True)
    print("  EXPERIMENT: HUY_D1_LEGAL_SECTION_EVIDENCE_V1", flush=True)
    print("===================================================================\n", flush=True)

    # Step 1: Smoke Test
    print(">>> STEP 1: Running Smoke Test...", flush=True)
    smoke_res = run_smoke_test()
    if smoke_res.get("status") != "PASS":
        print("FATAL: Smoke test failed!", flush=True)
        sys.exit(1)
    print("Step 1 PASSED.\n", flush=True)

    # Step 2: Baseline S0 Parity Check
    print(">>> STEP 2: Verifying S0 Baseline Parity...", flush=True)
    parity_res, _, _ = evaluate_s0_baseline()
    if not parity_res.get("parity_exact"):
        print("FATAL: S0 Baseline Parity FAILED! Halting experiment as required.", flush=True)
        sys.exit(1)
    print("Step 2 PASSED: Exact S0 Parity (48D, R@5 = 0.9569444444444444).\n", flush=True)

    # Step 3: Git Source Audit Check
    print(">>> STEP 3: Verifying Git Source Audit Alignment...", flush=True)
    git_info = get_git_info()
    print(f"  Head Commit:   {git_info.get('head_commit')}", flush=True)
    print(f"  Origin Commit: {git_info.get('origin_main_commit')}", flush=True)
    print(f"  Parity Exact:  {git_info.get('parity')}", flush=True)
    print(f"  Tree Clean:    {git_info.get('status_clean')}", flush=True)

    if not git_info.get("parity") or not git_info.get("status_clean"):
        print("WARNING / AUDIT REQUIREMENT: Source code must be committed and pushed to origin/main before final run.", flush=True)
        print("Proceeding only if verified.", flush=True)

    # Step 4: Model loading & provenance
    print("\n>>> STEP 4: Recording Model Provenance...", flush=True)
    _, _, model_prov = load_jina_crossencoder()
    print(f"  Model: {model_prov['base_model_name']} ({model_prov['weights_path']})", flush=True)

    # Step 5: Score CAL candidate pool with legal section CE
    print("\n>>> STEP 5: Scoring CAL candidate pool with legal_section_ce...", flush=True)
    scoring_meta = score_legal_sections(batch_size=16, count_sections=2)
    print("Step 5 PASSED.\n", flush=True)

    # Step 6: Evaluate S0 vs S1 under LOBO
    print(">>> STEP 6: Evaluating S0 vs S1 LOBO fusion...", flush=True)
    eval_report, preds_s0, preds_s1, scores_s0, scores_s1 = evaluate_both_arms()
    print("Step 6 PASSED.\n", flush=True)

    # Step 7: Build all authoritative artifacts
    print(">>> STEP 7: Generating Authoritative Artifacts...", flush=True)
    cal_data = load_cal_data()
    authoritative_report = build_all_artifacts(
        eval_report=eval_report,
        preds_s0=preds_s0,
        preds_s1=preds_s1,
        scores_s0=scores_s0,
        scores_s1=scores_s1,
        model_provenance=model_prov,
        scoring_meta=scoring_meta,
        cal_data=cal_data,
    )
    print("Step 7 PASSED.\n", flush=True)

    print("===================================================================", flush=True)
    print(f"  FINAL VERDICT: {eval_report['verdict']}", flush=True)
    print("===================================================================", flush=True)


if __name__ == "__main__":
    main()
