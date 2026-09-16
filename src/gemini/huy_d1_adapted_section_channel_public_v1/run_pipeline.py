"""Authoritative End-to-End Pipeline for HUY_D1_ADAPTED_SECTION_CHANNEL_PUBLIC_V1."""

from __future__ import annotations

import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from .audit_provenance_and_adapter import run_provenance_audit
from .audit_score_semantics import run_score_semantics_audit
from .baseline_and_control_parity import run_baseline_and_control_parity
from .common import RESULTS_DIR, get_git_status, seed_everything
from .evaluate_local_three_arms import evaluate_local_three_arms
from .materialize_public_candidate import run_public_stage
from .score_cal_adapted_section_ce import score_cal_adapted_section_ce


def main():
    t_start = time.perf_counter()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    seed_everything(2026)

    print("================================================================================", flush=True)
    print("STARTING PIPELINE: HUY_D1_ADAPTED_SECTION_CHANNEL_PUBLIC_V1", flush=True)
    print(f"Timestamp UTC: {datetime.now(timezone.utc).isoformat()}", flush=True)
    print("================================================================================", flush=True)

    # Stage 0: Git Hard Gate
    print("\n--- STAGE 0: GIT HARD GATE ---", flush=True)
    git_info = get_git_status()
    print(f"Head commit:        {git_info.get('head_commit')}", flush=True)
    print(f"Origin/main commit: {git_info.get('origin_main_commit')}", flush=True)
    print(f"Git parity:         {git_info.get('parity')}", flush=True)
    print(f"Working tree clean: {git_info.get('status_clean')}", flush=True)

    if not git_info.get("parity"):
        raise RuntimeError(f"BLOCKED_GIT_PARITY: HEAD does not match origin/main! ({git_info})")
    if not git_info.get("status_clean"):
        raise RuntimeError(f"BLOCKED_GIT_DIRTY: Working tree is not clean! ({git_info.get('porcelain_output')})")
    print("Stage 0 PASSED.", flush=True)

    # Stage 1: Audit Provenance and Seal Adapter
    run_provenance_audit()
    print("Stage 1 PASSED.", flush=True)

    # Stage 2: Audit Score Semantics
    run_score_semantics_audit(sample_size_pairs=64)
    print("Stage 2 PASSED.", flush=True)

    # Stage 3: Baseline & Control Parity
    run_baseline_and_control_parity()
    print("Stage 3 PASSED.", flush=True)

    # Stage 4: Score CAL Candidates with Adapted LoRA CE (Label-free)
    score_cal_adapted_section_ce(batch_size=64)
    print("Stage 4 PASSED.", flush=True)

    # Stage 5: Three-Arm LOBO Evaluation & Local Decision Gates
    report_data, verdict = evaluate_local_three_arms()
    print(f"Stage 5 PASSED with local verdict: {verdict}", flush=True)

    # Stage 6: Public Stage (Conditional on local verdict)
    public_manifest = run_public_stage()
    print(f"Stage 6 completed with status: {public_manifest.get('status', 'EXECUTED')}", flush=True)

    t_total = time.perf_counter() - t_start
    print("\n================================================================================", flush=True)
    print(f"PIPELINE COMPLETED SUCCESSFULLY in {t_total:.1f}s ({t_total/60.0:.2f} min)", flush=True)
    print(f"Final Local Verdict: {verdict}", flush=True)
    print("================================================================================", flush=True)


if __name__ == "__main__":
    main()
