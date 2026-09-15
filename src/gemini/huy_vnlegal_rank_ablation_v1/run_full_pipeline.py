"""Full pipeline runner for HUY_VNLEGAL_RANK_ABLATION_V1."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = ROOT / "results" / "gemini" / "huy_vnlegal_rank_ablation_v1"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(Path(__file__).resolve().parent))

import audit_contracts
import evaluate_ablation_cal
import run_paired_bootstrap
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
    print("STARTING FULL PIPELINE: HUY_VNLEGAL_RANK_ABLATION_V1")
    print("================================================================================")

    trace_file = RESULTS_DIR / "EXECUTION_TRACE.jsonl"
    if trace_file.exists():
        trace_file.unlink()

    # Step 1: Contract Difference Audit
    t1 = time.perf_counter()
    print("\n--- STEP 1: Contract Difference Audit ---")
    c_res = audit_contracts.audit_contracts()
    dur1 = time.perf_counter() - t1
    log_trace("contract_diff_audit", c_res["status"], dur1, "50D -> 48D (removed vnlegal_lal rank view only)")

    # Step 2: CAL LOBO Evaluation
    t2 = time.perf_counter()
    print("\n--- STEP 2: Dual CAL LOBO Evaluation (D0 vs D1) ---")
    parity_rep, bound_rep, cal_rep = evaluate_ablation_cal.main()
    dur2 = time.perf_counter() - t2
    log_trace(
        "cal_lobo_evaluation",
        "DONE",
        dur2,
        f"D0={cal_rep['d0_current_production']['metrics']['pooled_recall_at_5']:.6f}, D1={cal_rep['d1_score_only_vnlegal']['metrics']['pooled_recall_at_5']:.6f} (+{cal_rep['comparison_d1_vs_d0']['delta_pooled_recall_at_5']:+.6f})"
    )

    # Step 3: Paired Bootstrap Stability Analysis
    t3 = time.perf_counter()
    print("\n--- STEP 3: Paired Bootstrap Stability Analysis ---")
    boot_res = run_paired_bootstrap.run_bootstrap()
    dur3 = time.perf_counter() - t3
    log_trace(
        "paired_bootstrap",
        boot_res["status"],
        dur3,
        f"95% CI [{boot_res['unstratified_bootstrap']['ci_95_low']:+.6f}, {boot_res['unstratified_bootstrap']['ci_95_high']:+.6f}], P(delta>0)={boot_res['unstratified_bootstrap']['probability_delta_gt_0']*100:.2f}%"
    )

    # Step 4: Materialize Public Packages
    t4 = time.perf_counter()
    print("\n--- STEP 4: Materialize Public Candidate Packages ---")
    pub_res = materialize_public_candidates.main()
    dur4 = time.perf_counter() - t4
    log_trace("materialize_public_packages", "DONE", dur4, "Control D0 and Candidate D1 materialized and validated")

    # Step 5: Build Decision & Consistency Audit
    t5 = time.perf_counter()
    print("\n--- STEP 5: Build Decision, Provenance, and Consistency Audit ---")
    build_decision_and_trace.main()
    dur5 = time.perf_counter() - t5
    log_trace("build_decision_and_provenance", "DONE", dur5, "DECISION.md and REPORT_CONSISTENCY_AUDIT.json created")

    total_dur = time.perf_counter() - t_start
    log_trace("full_pipeline", "COMPLETED", total_dur, f"Total execution time: {total_dur:.1f}s")

    print("\n================================================================================")
    print(f"PIPELINE COMPLETE in {total_dur:.1f}s ({total_dur/60.0:.2f} mins)")
    print("================================================================================")


if __name__ == "__main__":
    main()
