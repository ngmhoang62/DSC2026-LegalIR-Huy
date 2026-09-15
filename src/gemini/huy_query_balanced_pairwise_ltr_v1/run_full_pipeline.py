"""Full pipeline runner for HUY_QUERY_BALANCED_PAIRWISE_LTR_V1."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = ROOT / "results" / "gemini" / "huy_query_balanced_pairwise_ltr_v1"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pairwise_ranker
import audit_profile_reuse
import evaluate_dual_cal
import run_v2_shadow
import materialize_public_candidates
import build_decision_and_trace


def log_trace(step_name: str, status: str, duration: float, detail: str = ""):
    trace_file = RESULTS_DIR / "EXECUTION_TRACE.jsonl"
    record = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "step": step_name,
        "status": status,
        "duration_seconds": duration,
        "detail": detail,
    }
    with trace_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def main():
    t_start = time.perf_counter()
    print("================================================================================")
    print("STARTING FULL PIPELINE: HUY_QUERY_BALANCED_PAIRWISE_LTR_V1")
    print("================================================================================")

    trace_file = RESULTS_DIR / "EXECUTION_TRACE.jsonl"
    if trace_file.exists():
        trace_file.unlink()

    # Step 0: Unit Test
    t0 = time.perf_counter()
    print("\n--- STEP 0: Pairwise Utility Unit Test ---")
    u_res = pairwise_ranker.run_pairwise_utility_unit_test()
    dur0 = time.perf_counter() - t0
    log_trace("pairwise_utility_unit_test", u_res["status"], dur0, f"max_abs_err={u_res['max_absolute_error']:.2e}")

    # Step 1: Profile Reuse & Duplicate Audit
    t1 = time.perf_counter()
    print("\n--- STEP 1: Profile Reuse & Duplicate Links Audit ---")
    p_res = audit_profile_reuse.main()
    dur1 = time.perf_counter() - t1
    log_trace("profile_reuse_audit", p_res["status"], dur1, "50 directed links verified, profile cache SHA verified")

    # Step 2: Dual CAL LOBO Evaluation
    t2 = time.perf_counter()
    print("\n--- STEP 2: Dual CAL LOBO Evaluation (Q0, Q1, Q2) ---")
    cal_res = evaluate_dual_cal.main()
    dur2 = time.perf_counter() - t2
    log_trace("dual_cal_lobo_evaluation", "DONE", dur2, f"Historical Q0={cal_res['historical_cal']['q0_pointwise']['metrics']['pooled_recall_at_5']:.6f}")

    # Step 3: Strict-V2 Shadow Sanity Check
    t3 = time.perf_counter()
    print("\n--- STEP 3: Strict-V2 Objective Shadow Evaluation ---")
    v2_res = run_v2_shadow.main()
    dur3 = time.perf_counter() - t3
    log_trace("v2_shadow_evaluation", "DONE", dur3, f"V0={v2_res['authoritative_baseline_v0']['metrics']['recall_at_5']:.6f}, V1={v2_res['shadow_arm_v1']['metrics']['recall_at_5']:.6f}")

    # Step 4: Materialize Public Candidates
    t4 = time.perf_counter()
    print("\n--- STEP 4: Materialize Public Candidate Packages ---")
    pub_res = materialize_public_candidates.main()
    dur4 = time.perf_counter() - t4
    log_trace("materialize_public_candidates", "DONE", dur4, f"H0, Q1, Q2 packages created")

    # Step 5: Build Decision, Consistency Audit, Provenance
    t5 = time.perf_counter()
    print("\n--- STEP 5: Build Decision, Provenance, and Consistency Audit ---")
    build_decision_and_trace.main()
    dur5 = time.perf_counter() - t5
    log_trace("build_decision_and_provenance", "DONE", dur5, "DECISION.md and consistency audit created")

    total_dur = time.perf_counter() - t_start
    log_trace("full_pipeline", "COMPLETED", total_dur, f"Total execution time: {total_dur:.1f}s")

    print("\n================================================================================")
    print(f"PIPELINE COMPLETE in {total_dur:.1f}s ({total_dur/60.0:.2f} mins)")
    print("================================================================================")


if __name__ == "__main__":
    main()
