"""Master pipeline orchestrator for HUY_D1_F3_SECTION_BOUNDARY_V1."""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path("D:/Study/DSC2026/sota")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.gemini.huy_d1_f3_section_boundary_v1.common import (
    CAL600_5FOLD_PATH,
    CAL_FROZEN_SECTION_CACHE_PATH,
    CAL_GOLD_PATH,
    CAL_QUESTIONS_LABEL_FREE_PATH,
    EXPECTED_5FOLD_SPLIT_SHA256,
    EXPECTED_SECTION_CE_SHA256,
    F3_FEASIBILITY_PATH,
    F3_MODEL_PATH,
    F3_SCORES_PATH,
    F3_SOURCE_PATH,
    RESULTS_DIR,
    SEED,
    SRC_DIR,
    compute_d1_lobo,
    evaluate_parity,
    get_git_status,
    get_source_files_sha256,
    load_cal_data_label_free,
    load_cal_gold_labels,
    seed_everything,
    sha256_file,
)
from src.gemini.huy_d1_f3_section_boundary_v1.confirmation_gate import (
    evaluate_confirmation_gate,
    evaluate_full_cal,
    load_confirmation_split,
)
from src.gemini.huy_d1_f3_section_boundary_v1.f3_section_rule import (
    compute_label_free_proposals,
    evaluate_f3_coverage,
    load_frozen_expert_caches,
)


def run_pipeline() -> Dict[str, Any]:
    print("==================================================================", flush=True)
    print("STARTING PIPELINE: HUY_D1_F3_SECTION_BOUNDARY_V1", flush=True)
    print("==================================================================", flush=True)

    start_time_utc = datetime.now(timezone.utc).isoformat()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    seed_everything(SEED)

    # ---------------------------------------------------------
    # Stage 1: Source Provenance Hard Gate
    # ---------------------------------------------------------
    print("\n--- STAGE 1: SOURCE PROVENANCE HARD GATE ---", flush=True)
    git_info = get_git_status()
    source_files = get_source_files_sha256()

    print(f"Git HEAD Commit:        {git_info.get('head_commit')}", flush=True)
    print(f"Git Origin/Main Commit: {git_info.get('origin_main_commit')}", flush=True)
    print(f"Remote Parity:          {git_info.get('parity')}", flush=True)
    print(f"Working Tree Clean:     {git_info.get('status_clean')}", flush=True)

    provenance_doc = {
        "schema_version": "dsc2026.gemini.huy_d1_f3_section_boundary_v1.source_provenance.v1",
        "experiment_id": "HUY_D1_F3_SECTION_BOUNDARY_V1",
        "timestamp_utc": start_time_utc,
        "git": git_info,
        "source_files": source_files,
    }
    prov_path = RESULTS_DIR / "SOURCE_PROVENANCE.json"
    prov_path.write_text(json.dumps(provenance_doc, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {prov_path}", flush=True)

    if not git_info.get("parity") or not git_info.get("status_clean"):
        print("FATAL: BLOCKED_SOURCE_PROVENANCE! HEAD must match origin/main and working tree must be clean.", flush=True)
        sys.exit(1)

    # Verify frozen input hashes
    actual_sec_sha = sha256_file(CAL_FROZEN_SECTION_CACHE_PATH)
    actual_split_sha = sha256_file(CAL600_5FOLD_PATH)
    if actual_sec_sha != EXPECTED_SECTION_CE_SHA256:
        raise RuntimeError(f"Section CE cache mismatch: {actual_sec_sha} != {EXPECTED_SECTION_CE_SHA256}")
    if actual_split_sha != EXPECTED_5FOLD_SPLIT_SHA256:
        raise RuntimeError(f"5-fold split mismatch: {actual_split_sha} != {EXPECTED_5FOLD_SPLIT_SHA256}")

    # ---------------------------------------------------------
    # Stage 2: Load CAL Data & Compute Exact D1 Parity
    # ---------------------------------------------------------
    print("\n--- STAGE 2: CAL DATA LOADER AND EXACT D1 PARITY ---", flush=True)
    (
        docs,
        queries_label_free,
        blocks,
        all_ids,
        extended,
        local_views,
        full_channels_cv,
        type_rows,
        cite_rows,
    ) = load_cal_data_label_free()

    gold, reveal_time_utc = load_cal_gold_labels(all_ids)

    d1_rankings, d1_scores = compute_d1_lobo(
        blocks=blocks,
        all_ids=all_ids,
        extended=extended,
        local_views=local_views,
        full_channels_cv=full_channels_cv,
        type_rows=type_rows,
        cite_rows=cite_rows,
        gold=gold,
    )

    parity_doc, parity_pass = evaluate_parity(
        d1_rankings=d1_rankings,
        gold=gold,
        all_ids=all_ids,
        blocks=blocks,
    )

    if not parity_pass:
        print("FATAL: BLOCKED_D1_PARITY! Exact D1 parity failed.", flush=True)
        sys.exit(1)

    # ---------------------------------------------------------
    # Stage 3: Frozen Expert Caches & F3 Coverage Gate
    # ---------------------------------------------------------
    print("\n--- STAGE 3: FROZEN EXPERT CACHES AND F3 COVERAGE ---", flush=True)
    f3_scores, sec_scores = load_frozen_expert_caches()

    coverage_doc = evaluate_f3_coverage(
        all_ids=all_ids,
        d1_rankings=d1_rankings,
        extended=extended,
        f3_scores=f3_scores,
    )

    if not coverage_doc["coverage"]["coverage_gate_passed"]:
        print("FATAL: F3 coverage gate failed on rank 5 or rank 6!", flush=True)
        sys.exit(1)

    # ---------------------------------------------------------
    # Stage 4: Label-Free Proposal Generation (Pre-Gold Action Seal)
    # ---------------------------------------------------------
    print("\n--- STAGE 4: LABEL-FREE PROPOSAL GENERATION ---", flush=True)
    proposals_doc, proposals_records, action_qids = compute_label_free_proposals(
        all_ids=all_ids,
        d1_rankings=d1_rankings,
        extended=extended,
        f3_scores=f3_scores,
        sec_scores=sec_scores,
    )
    print(f"F3 Crossovers across CAL600:      {proposals_doc['summary']['f3_crossover_count']}", flush=True)
    print(f"Section Confirmed across CAL600:  {proposals_doc['summary']['section_confirmed_actions_count']}", flush=True)

    # ---------------------------------------------------------
    # Stage 5: F3_CONFIRMATION_240 Gate Evaluation
    # ---------------------------------------------------------
    print("\n--- STAGE 5: F3_CONFIRMATION_240 EVALUATION ---", flush=True)
    folds, conf_ids, split_raw = load_confirmation_split()

    conf_report, conf_passed, conf_verdict = evaluate_confirmation_gate(
        folds=folds,
        conf_ids=conf_ids,
        d1_preds=d1_rankings,
        action_qids=action_qids,
        proposals_records=proposals_records,
        gold=gold,
    )

    print(f"Confirmation Actions:    {conf_report['actions_count']}", flush=True)
    print(f"Action Breakdown:        {conf_report['action_breakdown']}", flush=True)
    print(f"Recall Delta on 240:     {conf_report['metrics']['recall_at_5']['delta']:+.6f}", flush=True)
    print(f"Precision Delta on 240:  {conf_report['metrics']['precision_at_5']['delta']:+.6f}", flush=True)
    print(f"Confirmation Verdict:    {conf_verdict}", flush=True)

    final_verdict = conf_verdict
    full_cal_doc = None

    if conf_passed:
        print("\n--- STAGE 6: FULL CAL EVALUATION (CONFIRMATION PASSED) ---", flush=True)
        full_cal_doc, final_verdict = evaluate_full_cal(
            all_ids=all_ids,
            blocks=blocks,
            d1_preds=d1_rankings,
            action_qids=action_qids,
            proposals_records=proposals_records,
            gold=gold,
        )
    else:
        print("\nNOTE: Confirmation gate failed. Per pre-registered protocol, full CAL utility is NOT evaluated or written to disk.", flush=True)

    # ---------------------------------------------------------
    # Stage 7: Write EXPERIMENT_MANIFEST.json
    # ---------------------------------------------------------
    print("\n--- STAGE 7: EXPERIMENT MANIFEST ---", flush=True)
    manifest = {
        "schema_version": "dsc2026.gemini.huy_d1_f3_section_boundary_v1.manifest.v1",
        "experiment_id": "HUY_D1_F3_SECTION_BOUNDARY_V1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source_head_commit": git_info.get("head_commit"),
        "source_origin_commit": git_info.get("origin_main_commit"),
        "git_clean": git_info.get("status_clean"),
        "critical_input_hashes": {
            "F3_MODEL_PATH": sha256_file(F3_MODEL_PATH),
            "F3_SCORES_PATH": sha256_file(F3_SCORES_PATH),
            "F3_SOURCE_PATH": sha256_file(F3_SOURCE_PATH),
            "F3_FEASIBILITY_PATH": sha256_file(F3_FEASIBILITY_PATH),
            "CAL_FROZEN_SECTION_CACHE_PATH": sha256_file(CAL_FROZEN_SECTION_CACHE_PATH),
            "CAL600_5FOLD_PATH": sha256_file(CAL600_5FOLD_PATH),
            "CAL_QUESTIONS_LABEL_FREE_PATH": sha256_file(CAL_QUESTIONS_LABEL_FREE_PATH),
            "CAL_GOLD_PATH": sha256_file(CAL_GOLD_PATH),
        },
        "leakage_assertions": {
            "primary_rule_source_frozen_before_confirmation_labels": True,
            "no_qid_hardcoding": True,
            "no_cal_derived_numeric_threshold": True,
            "f4_used": False,
            "public_leaderboard_used_to_configure_rule": False,
            "confirmation_rule_modified_after_outcomes": False,
        },
        "d1_parity_status": parity_doc["status"],
        "f3_coverage_status": coverage_doc["status"],
        "confirmation_verdict": conf_verdict,
        "final_verdict": final_verdict,
        "submission_zip_generated": False,
    }

    manifest_path = RESULTS_DIR / "EXPERIMENT_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {manifest_path}", flush=True)

    print("\n==================================================================", flush=True)
    print(f"PIPELINE COMPLETED WITH VERDICT: {final_verdict}", flush=True)
    print("==================================================================", flush=True)

    return {
        "git": git_info,
        "parity": parity_doc,
        "coverage": coverage_doc,
        "proposals": proposals_doc,
        "confirmation": conf_report,
        "full_cal": full_cal_doc,
        "manifest": manifest,
        "final_verdict": final_verdict,
    }


if __name__ == "__main__":
    run_pipeline()
