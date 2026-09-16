"""End-to-end orchestration runner for HUY_D1_LEGAL_SECTION_EVIDENCE_V1 (Clean Reproduction)."""

from __future__ import annotations

import datetime
import json
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
)
from src.gemini.huy_d1_legal_section_evidence_v1.common import (
    SCORE_CACHE_PKL,
    get_git_status,
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
    print("  EXPERIMENT: HUY_D1_LEGAL_SECTION_EVIDENCE_V1 (CLEAN REPRODUCTION)", flush=True)
    print("===================================================================\n", flush=True)

    # Step 0: Git Hard Gate Audit
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

    # Step 1: Smoke Test
    print(">>> STEP 1: Running Synthetic Smoke Test...", flush=True)
    smoke_res = run_smoke_test()
    if smoke_res.get("status") != "PASS":
        print("FATAL: Smoke test failed!", flush=True)
        sys.exit(1)
    print("Step 1 PASSED: Synthetic smoke test clean.\n", flush=True)

    # Step 2: S0 Baseline Parity Check
    print(">>> STEP 2: Verifying S0 Baseline Parity...", flush=True)
    parity_res, _, _ = evaluate_s0_baseline()
    if not parity_res.get("parity_exact"):
        print("FATAL: S0 Baseline Parity FAILED!", flush=True)
        sys.exit(1)
    print("Step 2 PASSED: Exact S0 Parity (48D, R@5 = 0.9569444444444444).\n", flush=True)

    # Step 3: Model loading & provenance
    print(">>> STEP 3: Verifying Model Provenance...", flush=True)
    _, _, model_prov = load_jina_crossencoder()
    print(f"  Model: {model_prov['base_model_name']}", flush=True)
    print(f"  Weights: {model_prov['weights_path']} (SHA256: {model_prov['weights_sha256']})", flush=True)
    print("Step 3 PASSED.\n", flush=True)

    # Step 4: Fresh scoring CAL candidate pool
    print(">>> STEP 4: Fresh-Scoring CAL candidate pool (fresh_run=True)...", flush=True)
    if SCORE_CACHE_PKL.exists():
        print(f"Removing old cache {SCORE_CACHE_PKL} to enforce clean rerun...", flush=True)
        SCORE_CACHE_PKL.unlink()

    scoring_meta = score_legal_sections(batch_size=16, count_sections=2, fresh_run=True)
    assert scoring_meta["newly_scored_queries"] == 600, (
        f"Expected 600 newly scored queries, got {scoring_meta['newly_scored_queries']}"
    )
    assert scoring_meta["reused_queries"] == 0, (
        f"Expected 0 reused queries, got {scoring_meta['reused_queries']}"
    )
    print(
        f"Step 4 PASSED: 600 queries freshly scored (Cache SHA256: {scoring_meta['cache_sha256']}).\n",
        flush=True,
    )

    # Step 5: Evaluate S0 vs S1 under LOBO
    print(">>> STEP 5: Evaluating S0 vs S1 LOBO fusion...", flush=True)
    (
        eval_report,
        preds_s0,
        preds_s1,
        scores_s0,
        scores_s1,
        full_rankings_s0,
        full_rankings_s1,
        full_scores_s0,
        full_scores_s1,
    ) = evaluate_both_arms()
    eval_completed_time = datetime.datetime.now(datetime.timezone.utc).isoformat()
    print("Step 5 PASSED: LOBO evaluation complete.\n", flush=True)

    # Step 6: Build all authoritative artifacts
    print(">>> STEP 6: Generating Authoritative Artifacts...", flush=True)
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
        full_rankings_s0=full_rankings_s0,
        full_rankings_s1=full_rankings_s1,
        full_scores_s0=full_scores_s0,
        full_scores_s1=full_scores_s1,
        eval_completed_time=eval_completed_time,
    )
    print("Step 6 PASSED.\n", flush=True)

    print("===================================================================", flush=True)
    print(f"  FINAL VERDICT: {eval_report['verdict']}", flush=True)
    print("===================================================================", flush=True)


if __name__ == "__main__":
    main()
