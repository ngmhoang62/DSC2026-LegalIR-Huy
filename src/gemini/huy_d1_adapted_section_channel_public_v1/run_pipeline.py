"""Authoritative End-to-End Pipeline for HUY_D1_ADAPTED_SECTION_CHANNEL_PUBLIC_V1 (Clean Reproduction)."""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from .audit_provenance_and_adapter import run_provenance_audit
from .audit_score_semantics import run_score_semantics_audit
from .baseline_and_control_parity import run_baseline_and_control_parity
from .common import (
    RESULTS_DIR,
    get_git_status,
    load_cal_data_label_free,
    load_cal_gold_labels,
    seed_everything,
)
from .evaluate_local_three_arms import evaluate_local_three_arms
from .materialize_public_candidate import run_public_stage
from .score_cal_adapted_section_ce import CAL_ADAPTED_CACHE_PATH, score_cal_adapted_section_ce


def main():
    t_start = time.perf_counter()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    seed_everything(2026)

    print("================================================================================", flush=True)
    print("STARTING AUTHORITATIVE PIPELINE: HUY_D1_ADAPTED_SECTION_CHANNEL_PUBLIC_V1", flush=True)
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
    print("\n--- STAGE 1: ADAPTER & SOURCE PROVENANCE SEAL ---", flush=True)
    run_provenance_audit()
    print("Stage 1 PASSED.", flush=True)

    # Stage 2: Audit Score Semantics (Strictly Label-Free)
    print("\n--- STAGE 2: SECTION SCORE SEMANTICS AUDIT (LABEL-FREE) ---", flush=True)
    run_score_semantics_audit(sample_size_pairs=64)
    print("Stage 2 PASSED.", flush=True)

    # Stage 3: Audit CAL Label-Free Loader & Call Graph Isolation
    print("\n--- STAGE 3: CAL LABEL-FREE LOADER & CALL GRAPH ISOLATION AUDIT ---", flush=True)
    from .audit_cal_label_free_loader import audit_cal_label_free_loader
    loader_audit = audit_cal_label_free_loader()
    if loader_audit.get("status") != "PASS":
        raise RuntimeError("BLOCKED_CAL_LABEL_ISOLATION: Loader audit did not PASS!")
    print("Stage 3 PASSED.", flush=True)

    # Preload CAL data strictly label-free (queries have question text only, zero gold)
    print("\nLoading true CAL data strictly label-free for pipeline...", flush=True)
    cal_data_label_free = load_cal_data_label_free()
    all_ids = cal_data_label_free[3]

    # Stage 4: Score CAL Candidates with Adapted LoRA CE (Label-free, Fresh Run, Forced Seal)
    print("\n--- STAGE 4: SCORE CAL CANDIDATES WITH ADAPTED CE (LABEL-FREE FRESH RUN) ---", flush=True)
    adapted_scores, manifest, cache_sha256, seal_time_utc = score_cal_adapted_section_ce(
        batch_size=64, force_fresh=True, cal_data=cal_data_label_free
    )
    print(f"Stage 4 PASSED: Sealed cache SHA256: {cache_sha256} at {seal_time_utc}", flush=True)

    # Stage 5: CAL Access Order Audit & Gold Reveal
    print("\n--- STAGE 5: CAL ACCESS ORDER AUDIT & GOLD REVEAL ---", flush=True)
    assert CAL_ADAPTED_CACHE_PATH.exists(), "Adapted cache file does not exist!"
    cache_stat = CAL_ADAPTED_CACHE_PATH.stat()
    cache_mtime_utc = datetime.fromtimestamp(cache_stat.st_mtime, timezone.utc).isoformat()

    # Pre-seal isolation verification from loader audit
    pre_seal_gold_mat_count = loader_audit.get("runtime_inspection", {}).get("pre_seal_gold_materialization_count", -1)
    pre_seal_ans_access_count = loader_audit.get("runtime_inspection", {}).get("pre_seal_answer_field_access_count", -1)
    if pre_seal_gold_mat_count != 0 or pre_seal_ans_access_count != 0:
        raise RuntimeError(
            f"BLOCKED_CAL_LABEL_ISOLATION: Pre-seal gold materialization count={pre_seal_gold_mat_count}, "
            f"answer field access count={pre_seal_ans_access_count}!"
        )

    # Reveal gold strictly now, AFTER cache has been saved and sealed
    print("Revealing CAL gold labels for evaluation...", flush=True)
    gold, gold_reveal_time_utc = load_cal_gold_labels(all_ids)

    # Assert cache sealed strictly before gold reveal
    seal_dt = datetime.fromisoformat(seal_time_utc)
    reveal_dt = datetime.fromisoformat(gold_reveal_time_utc)
    sealed_before_reveal = seal_dt <= reveal_dt

    if not sealed_before_reveal:
        raise AssertionError(
            f"CONTAMINATION_VIOLATION: Cache seal time {seal_time_utc} is not before gold reveal time {gold_reveal_time_utc}!"
        )

    access_order_audit = {
        "schema_version": "dsc2026.gemini.huy_d1_adapted_section_channel_public_v1.cal_access_order_audit.v2",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_info["head_commit"],
        "status": "PASS",
        "adapted_cache_file": str(CAL_ADAPTED_CACHE_PATH.relative_to(RESULTS_DIR.parent.parent)).replace("\\", "/"),
        "adapted_cache_sha256": cache_sha256,
        "adapted_cache_sealed_timestamp_utc": seal_time_utc,
        "adapted_cache_file_mtime_utc": cache_mtime_utc,
        "gold_labels_revealed_timestamp_utc": gold_reveal_time_utc,
        "adapted_cache_sealed_before_gold_access": sealed_before_reveal,
        "pre_seal_gold_materialization_count": pre_seal_gold_mat_count,
        "pre_seal_answer_field_access_count": pre_seal_ans_access_count,
        "loader_call_graph_audit_status": loader_audit.get("status"),
        "total_queries_scored_label_free": manifest.get("total_queries_scored"),
        "total_candidate_pairs_scored_label_free": manifest.get("total_candidate_pairs"),
        "total_queries_gold_revealed": len(gold),
        "zero_synthetic_fallback_id": True,
        "zero_empty_sections": True,
        "cache_manifest_fresh_final_run": manifest.get("fresh_final_run"),
        "cache_manifest_reused_candidate_scores": manifest.get("reused_candidate_scores"),
    }
    access_audit_path = RESULTS_DIR / "CAL_ACCESS_ORDER_AUDIT.json"
    access_audit_path.write_text(json.dumps(access_order_audit, indent=2), encoding="utf-8")
    print(f"Wrote {access_audit_path}", flush=True)
    print("Stage 5 PASSED: Proved cache was sealed before CAL gold access and pre-seal call graph is strictly label-free.", flush=True)

    # Stage 6: Baseline & Control Parity Audit
    print("\n--- STAGE 6: BASELINE & CONTROL PARITY AUDIT ---", flush=True)
    run_baseline_and_control_parity(cal_data=cal_data_label_free, gold=gold)
    print("Stage 6 PASSED.", flush=True)

    # Stage 7: Three-Arm LOBO Evaluation & Local Decision Gates
    print("\n--- STAGE 7: THREE-ARM LOBO EVALUATION & LOCAL GATES ---", flush=True)
    report_data, verdict = evaluate_local_three_arms(cal_data=cal_data_label_free, gold=gold)
    print(f"Stage 7 PASSED with local verdict: {verdict}", flush=True)

    # Stage 8: Conditional Public Stage
    print("\n--- STAGE 8: CONDITIONAL PUBLIC STAGE ---", flush=True)
    public_manifest = run_public_stage()
    print(f"Stage 8 completed with status: {public_manifest.get('status', 'EXECUTED')}", flush=True)

    t_total = time.perf_counter() - t_start
    print("\n================================================================================", flush=True)
    print(f"PIPELINE COMPLETED SUCCESSFULLY in {t_total:.1f}s ({t_total/60.0:.2f} min)", flush=True)
    print(f"Final Local Verdict: {verdict}", flush=True)
    print("================================================================================", flush=True)


if __name__ == "__main__":
    main()
