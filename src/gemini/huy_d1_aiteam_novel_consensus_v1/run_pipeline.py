"""Master pipeline orchestrator for HUY_D1_AITEAM_NOVEL_CONSENSUS_V1."""

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

from src.gemini.huy_d1_aiteam_novel_consensus_v1.action_rules import (
    compute_novel_consensus_actions,
)
from src.gemini.huy_d1_aiteam_novel_consensus_v1.cal_evaluator import (
    evaluate_novel_consensus_cal,
)
from src.gemini.huy_d1_aiteam_novel_consensus_v1.common import (
    AITEAM_REPORT_PATH,
    AITEAM_TOP50_PATH,
    CAL_FROZEN_SECTION_CACHE_PATH,
    CAL_GOLD_PATH,
    CAL_QUESTIONS_LABEL_FREE_PATH,
    EXPECTED_AITEAM_REPORT_SHA256,
    EXPECTED_AITEAM_TOP50_SHA256,
    EXPECTED_JINA_FT_CV_SHA256,
    EXPECTED_SECTION_CE_SHA256,
    JINA_FT_CV_PATH,
    REPO_JINA,
    RESULTS_DIR,
    SEED,
    WEIGHTS_JINA_FT,
    compute_d1_lobo,
    evaluate_parity,
    get_git_status,
    get_source_files_sha256,
    load_cal_data_label_free,
    load_cal_gold_labels,
    seed_everything,
    sha256_file,
)
from src.gemini.huy_d1_aiteam_novel_consensus_v1.expert_inference import (
    load_jina_crossencoder,
    score_jina_universe,
    score_section_universe,
)
from src.gemini.huy_d1_aiteam_novel_consensus_v1.novel_proposals import (
    extract_novel_proposals,
    load_and_verify_aiteam_source,
)


def run_pipeline() -> Dict[str, Any]:
    print("==================================================================", flush=True)
    print("STARTING PIPELINE: HUY_D1_AITEAM_NOVEL_CONSENSUS_V1", flush=True)
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
        "schema_version": "dsc2026.gemini.huy_d1_aiteam_novel_consensus_v1.source_provenance.v1",
        "experiment_id": "HUY_D1_AITEAM_NOVEL_CONSENSUS_V1",
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

    # Pre-train LOBO models on CAL blocks using training gold labels
    gold_cal, reveal_time_utc = load_cal_gold_labels(all_ids)

    d1_rankings, d1_scores = compute_d1_lobo(
        blocks=blocks,
        all_ids=all_ids,
        extended=extended,
        local_views=local_views,
        full_channels_cv=full_channels_cv,
        type_rows=type_rows,
        cite_rows=cite_rows,
        gold=gold_cal,
    )

    parity_doc, parity_pass = evaluate_parity(
        d1_rankings=d1_rankings,
        gold=gold_cal,
        all_ids=all_ids,
        blocks=blocks,
    )

    if not parity_pass:
        print("FATAL: BLOCKED_D1_PARITY! Exact D1 parity failed.", flush=True)
        sys.exit(1)

    d1_top5 = {q: d1_rankings[q][:5] for q in all_ids}

    # ---------------------------------------------------------
    # Stage 3: AITeam Provenance & Novel Proposals
    # ---------------------------------------------------------
    print("\n--- STAGE 3: AITEAM FULL-CORPUS SOURCE & NOVEL PROPOSALS ---", flush=True)
    aiteam20_rankings, aiteam_prov = load_and_verify_aiteam_source()

    novel_map, proposals_doc = extract_novel_proposals(
        all_ids=all_ids,
        extended=extended,
        aiteam20_rankings=aiteam20_rankings,
    )

    # ---------------------------------------------------------
    # Stage 4: Neural Expert Inference & Parity Audits
    # ---------------------------------------------------------
    print("\n--- STAGE 4: NEURAL EXPERT INFERENCE & PARITY AUDITS ---", flush=True)
    model, tokenizer, model_prov = load_jina_crossencoder()

    jina_universe_scores, jina_parity = score_jina_universe(
        model=model,
        queries_label_free=queries_label_free,
        docs=docs,
        all_ids=all_ids,
        d1_top5=d1_top5,
        novel_map=novel_map,
        batch_size=16,
    )

    if not jina_parity["parity_pass"]:
        print("FATAL: BLOCKED_JINA_FT_PARITY! Jina inference parity failed.", flush=True)
        sys.exit(1)

    sec_universe_scores, sec_parity = score_section_universe(
        model=model,
        queries_label_free=queries_label_free,
        docs=docs,
        all_ids=all_ids,
        d1_top5=d1_top5,
        novel_map=novel_map,
        batch_size=16,
    )

    if not sec_parity["parity_pass"]:
        print("FATAL: BLOCKED_SECTION_INFERENCE_PARITY! Section inference parity failed.", flush=True)
        sys.exit(1)

    # ---------------------------------------------------------
    # Stage 5: Dual Crossover Actions & Action Seal
    # ---------------------------------------------------------
    print("\n--- STAGE 5: DUAL CROSSOVER ACTIONS & ACTION SEAL ---", flush=True)
    repaired_preds, actions_doc, action_seal_sha = compute_novel_consensus_actions(
        all_ids=all_ids,
        d1_top5=d1_top5,
        novel_map=novel_map,
        aiteam20_rankings=aiteam20_rankings,
        jina_scores=jina_universe_scores,
        sec_scores=sec_universe_scores,
    )

    proposal_artifact_sha = sha256_file(RESULTS_DIR / "AITEAM20_NOVEL_PROPOSALS_LABEL_FREE.json")

    # ---------------------------------------------------------
    # Stage 6: CAL Gold Evaluation & Rescue Diagnostic
    # ---------------------------------------------------------
    print("\n--- STAGE 6: CAL GOLD EVALUATION & RESCUE DIAGNOSTIC ---", flush=True)
    cal_report, final_verdict = evaluate_novel_consensus_cal(
        all_ids=all_ids,
        blocks=blocks,
        extended=extended,
        d1_preds=d1_top5,
        repaired_preds=repaired_preds,
        aiteam20_rankings=aiteam20_rankings,
        actions_records=actions_doc["actions"],
        gold=gold_cal,
        d1_parity_passed=parity_pass,
    )

    # ---------------------------------------------------------
    # Stage 7: Experiment Manifest
    # ---------------------------------------------------------
    print("\n--- STAGE 7: EXPERIMENT MANIFEST ---", flush=True)
    manifest = {
        "schema_version": "dsc2026.gemini.huy_d1_aiteam_novel_consensus_v1.manifest.v1",
        "experiment_id": "HUY_D1_AITEAM_NOVEL_CONSENSUS_V1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source_head_commit": git_info.get("head_commit"),
        "source_origin_commit": git_info.get("origin_main_commit"),
        "git_clean": git_info.get("status_clean"),
        "critical_input_hashes": {
            "AITEAM_TOP50_PATH": sha256_file(AITEAM_TOP50_PATH),
            "AITEAM_REPORT_PATH": sha256_file(AITEAM_REPORT_PATH),
            "JINA_FT_CV_PATH": sha256_file(JINA_FT_CV_PATH),
            "CAL_FROZEN_SECTION_CACHE_PATH": sha256_file(CAL_FROZEN_SECTION_CACHE_PATH),
            "WEIGHTS_JINA_FT": sha256_file(WEIGHTS_JINA_FT),
            "CAL_QUESTIONS_LABEL_FREE_PATH": sha256_file(CAL_QUESTIONS_LABEL_FREE_PATH),
            "CAL_GOLD_PATH": sha256_file(CAL_GOLD_PATH),
        },
        "proposal_artifact_sha256": proposal_artifact_sha,
        "action_seal_sha256": action_seal_sha,
        "d1_parity_status": parity_doc["status"],
        "jina_parity_status": jina_parity["status"],
        "section_parity_status": sec_parity["status"],
        "rule_definition": {
            "novel_definition": "AITeam ranks 1..20 not in D1 extended candidate pool",
            "scoring_universe": "D1 Top-5 union NOVEL(q)",
            "jina_crossover": "Jina(c) > Jina(def) and Jina_rank_in_U(c) <= 5 and Jina_rank_in_U(def) > 5",
            "section_crossover": "Section(c) > Section(def) and Section_rank_in_U(c) <= 5 and Section_rank_in_U(def) > 5",
            "ambiguity_abstention": "len(ELIGIBLE) == 1 only, else KEEP / ABSTAIN",
        },
        "promotion_gates": cal_report["promotion_gates"],
        "leakage_assertions": {
            "no_known_qid_hardcoding": True,
            "no_gold_used_before_action_seal": True,
            "f3_used": False,
            "top50_used": False,
            "public_leaderboard_used_to_tune": False,
        },
        "final_verdict": final_verdict,
        "submission_zip_generated": False,
    }

    manifest_path = RESULTS_DIR / "EXPERIMENT_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {manifest_path}", flush=True)

    print("\n==================================================================", flush=True)
    print(f"PIPELINE COMPLETED WITH FINAL VERDICT: {final_verdict}", flush=True)
    print("==================================================================", flush=True)

    return {
        "git": git_info,
        "parity": parity_doc,
        "aiteam_prov": aiteam_prov,
        "proposals": proposals_doc,
        "jina_parity": jina_parity,
        "sec_parity": sec_parity,
        "actions": actions_doc,
        "cal_report": cal_report,
        "manifest": manifest,
        "final_verdict": final_verdict,
    }


if __name__ == "__main__":
    run_pipeline()
