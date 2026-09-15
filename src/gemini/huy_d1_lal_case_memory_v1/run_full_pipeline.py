"""Orchestrator for HUY_D1_LAL_CASE_MEMORY_V1 with execution tracing."""

from __future__ import annotations

import datetime as dt
import json
import time
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = ROOT / "results" / "gemini" / "huy_d1_lal_case_memory_v1"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

import sys
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src" / "gemini" / "huy_d1_lal_case_memory_v1"))

from audit_memory_source import audit_memory_source, audit_strict_memory_prior
from verify_query_embeddings import run_embedding_parity
from audit_data_isolation import audit_cal_data_isolation
from evaluate_cal_memory import run_cal_lobo_evaluation
from analyze_generalization import run_generalization_audit
from run_paired_bootstrap import run_bootstrap_analysis
from materialize_public_packages import run_public_materialization
from build_decision_and_audit import build_decision_and_reports


def log_trace(step_name: str, status: str, duration_sec: float, extra: Dict[str, Any] | None = None) -> None:
    record = {
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        "step": step_name,
        "status": status,
        "duration_seconds": duration_sec,
    }
    if extra:
        record.update(extra)
    trace_path = RESULTS_DIR / "EXECUTION_TRACE.jsonl"
    with open(trace_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    total_start = time.perf_counter()
    print("============================================================")
    print("STARTING EXPERIMENT: HUY_D1_LAL_CASE_MEMORY_V1")
    print("============================================================")

    # Clear previous trace if exists
    trace_path = RESULTS_DIR / "EXECUTION_TRACE.jsonl"
    if trace_path.exists():
        trace_path.unlink()

    # Step 1: Memory source audit
    t0 = time.perf_counter()
    audit_memory_source()
    audit_strict_memory_prior()
    dur = time.perf_counter() - t0
    log_trace("memory_source_and_prior_audit", "PASS", dur)
    print(f"[1/8] Memory source audit completed in {dur:.2f}s")

    # Step 2: Query embedding parity
    t0 = time.perf_counter()
    parity = run_embedding_parity()
    dur = time.perf_counter() - t0
    log_trace("query_embedding_parity", parity["status"], dur, {"mean_cosine": parity["mean_cosine"]})
    print(f"[2/8] Query embedding parity completed in {dur:.2f}s (status: {parity['status']})")
    if parity["status"] != "PASS":
        print("HALTING due to embedding parity failure.")
        build_decision_and_reports()
        return

    # Step 3: Data isolation audit
    t0 = time.perf_counter()
    isolation = audit_cal_data_isolation()
    dur = time.perf_counter() - t0
    log_trace("data_isolation_audit", isolation["status"], dur, {"directed_links": isolation["unique_bidirectional_directed_links_count"]})
    print(f"[3/8] Data isolation audit completed in {dur:.2f}s (status: {isolation['status']})")
    if isolation["status"] != "PASS":
        print("HALTING due to data leakage.")
        build_decision_and_reports()
        return

    # Step 4: CAL LOBO evaluation
    t0 = time.perf_counter()
    base_parity, cal_rep = run_cal_lobo_evaluation()
    dur = time.perf_counter() - t0
    log_trace("cal_lobo_evaluation", base_parity["status"], dur, {
        "m0_r5": cal_rep["arms"]["M0_D1_BASELINE"]["pooled_recall_at_5"],
        "m1_r5": cal_rep["arms"]["M1_D1_PLUS_LAL_MEMORY"]["pooled_recall_at_5"],
        "m2_r5": cal_rep["arms"]["M2_D1_PLUS_LAL_MEMORY_NO_DOCTYPE"]["pooled_recall_at_5"],
    })
    print(f"[4/8] CAL LOBO evaluation completed in {dur:.2f}s (baseline parity: {base_parity['status']})")
    if base_parity["status"] != "PASS":
        print("HALTING due to baseline parity failure.")
        build_decision_and_reports()
        return

    # Step 5: Generalization audit
    t0 = time.perf_counter()
    run_generalization_audit()
    dur = time.perf_counter() - t0
    log_trace("generalization_audit", "PASS", dur)
    print(f"[5/8] Generalization audit completed in {dur:.2f}s")

    # Step 6: Paired bootstrap
    t0 = time.perf_counter()
    boot = run_bootstrap_analysis()
    dur = time.perf_counter() - t0
    log_trace("paired_bootstrap", "PASS", dur, {
        "m1_p_pos": boot["m1_vs_m0"]["block_stratified"]["prob_positive"],
        "m2_p_pos": boot["m2_vs_m0"]["block_stratified"]["prob_positive"],
    })
    print(f"[6/8] Paired bootstrap completed in {dur:.2f}s")

    # Step 7: Public materialization
    t0 = time.perf_counter()
    pub = run_public_materialization()
    dur = time.perf_counter() - t0
    log_trace("public_materialization", pub["public_parity"]["status"], dur, {
        "exact_matches": pub["public_parity"]["exact_matches"],
    })
    print(f"[7/8] Public materialization completed in {dur:.2f}s (public parity: {pub['public_parity']['status']})")

    # Step 8: Build decision, provenance, and consistency audit
    t0 = time.perf_counter()
    dec = build_decision_and_reports()
    dur = time.perf_counter() - t0
    total_dur = time.perf_counter() - total_start
    log_trace("build_decision_and_reports", "PASS", dur, {
        "verdict": dec["verdict"],
        "total_experiment_seconds": total_dur,
    })
    print(f"[8/8] Decision and reports generated in {dur:.2f}s (Verdict: {dec['verdict']})")
    print(f"Total experiment runtime: {total_dur:.2f}s")


if __name__ == "__main__":
    main()
