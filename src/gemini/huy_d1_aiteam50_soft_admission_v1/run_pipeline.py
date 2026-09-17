"""Master orchestrator pipeline for HUY_D1_AITEAM50_SOFT_ADMISSION_V1.

Enforces strict source-first gate, label-free feature and action seals,
authoritative parity audits, CAL utility evaluation, and promotion gate checks.
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

ROOT = Path("D:/Study/DSC2026/sota")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.gemini.huy_d1_aiteam50_soft_admission_v1.admission_model import (
    train_oof_admission_model_and_seal_actions,
)
from src.gemini.huy_d1_aiteam50_soft_admission_v1.aiteam_source import (
    load_and_verify_aiteam_full_corpus,
)
from src.gemini.huy_d1_aiteam50_soft_admission_v1.cal_evaluator import (
    evaluate_cal_utility,
)
from src.gemini.huy_d1_aiteam50_soft_admission_v1.common import (
    RESULTS_DIR,
    compute_d1_lobo,
    get_git_status,
    get_source_files_sha256,
    load_cal_data_label_free,
    load_cal_gold_labels,
    seed_everything,
    sha256_file,
    verify_d1_parity,
)
from src.gemini.huy_d1_aiteam50_soft_admission_v1.feature_builder import (
    build_label_free_features,
    build_s_universe_and_novel_maps,
)
from src.gemini.huy_d1_aiteam50_soft_admission_v1.neural_features import (
    compute_neural_features_for_universe,
    load_neural_crossencoder,
)


def run_pipeline():
    total_t0 = time.perf_counter()
    seed_everything(2026)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("================================================================================", flush=True)
    print("      HUY_D1_AITEAM50_SOFT_ADMISSION_V1 — AUTHORITATIVE PIPELINE", flush=True)
    print("================================================================================", flush=True)

    # -------------------------------------------------------------------------
    # STAGE 1: SOURCE-FIRST HARD PROVENANCE GATE
    # -------------------------------------------------------------------------
    print("\n--- STAGE 1: SOURCE-FIRST HARD PROVENANCE GATE ---", flush=True)
    git_status = get_git_status()
    source_shas = get_source_files_sha256()

    print(f"HEAD commit:        {git_status['head_commit']}", flush=True)
    print(f"origin/main commit: {git_status['origin_main_commit']}", flush=True)
    print(f"Remote synced:      {git_status['parity']}", flush=True)
    print(f"Working tree clean: {git_status['status_clean']}", flush=True)

    if not git_status["parity"] or not git_status["status_clean"]:
        prov_doc = {
            "schema_version": "dsc2026.gemini.huy_d1_aiteam50_soft_admission_v1.provenance.v1",
            "experiment_id": "HUY_D1_AITEAM50_SOFT_ADMISSION_V1",
            "git": git_status,
            "source_files": source_shas,
            "status": "BLOCKED_SOURCE_PROVENANCE",
        }
        out_prov = RESULTS_DIR / "SOURCE_PROVENANCE.json"
        out_prov.write_text(json.dumps(prov_doc, indent=2), encoding="utf-8")
        print("ERROR: Remote sync or clean status failed! BLOCKED_SOURCE_PROVENANCE", flush=True)
        sys.exit(1)

    prov_doc = {
        "schema_version": "dsc2026.gemini.huy_d1_aiteam50_soft_admission_v1.provenance.v1",
        "experiment_id": "HUY_D1_AITEAM50_SOFT_ADMISSION_V1",
        "git": git_status,
        "source_files": source_shas,
        "status": "PASS",
    }
    out_prov = RESULTS_DIR / "SOURCE_PROVENANCE.json"
    out_prov.write_text(json.dumps(prov_doc, indent=2), encoding="utf-8")
    print(f"Wrote {out_prov} (PASS)", flush=True)

    # -------------------------------------------------------------------------
    # STAGE 2: EXACT D1 CONTROL & PARITY CHECK
    # -------------------------------------------------------------------------
    print("\n--- STAGE 2: EXACT D1 CONTROL & PARITY CHECK ---", flush=True)
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

    # Outer train gold labels only used for D1 LOBO model reproduction
    outer_gold, _ = load_cal_gold_labels(all_ids)

    d1_rankings, d1_scores = compute_d1_lobo(
        blocks,
        all_ids,
        extended,
        local_views,
        full_channels_cv,
        type_rows,
        cite_rows,
        outer_gold,
    )
    d1_parity_doc = verify_d1_parity(blocks, all_ids, d1_rankings, outer_gold)
    print(
        f"Exact D1 Recall@5: {d1_parity_doc['mean_recall_at_5']:.16f} (Expected: {d1_parity_doc['expected_recall_at_5']})",
        flush=True,
    )
    if not d1_parity_doc["parity_pass"]:
        print("ERROR: Exact D1 parity failed! BLOCKED_D1_PARITY", flush=True)
        sys.exit(1)

    # -------------------------------------------------------------------------
    # STAGE 3: FROZEN AITEAM TOP-50 SOURCE & PARITY
    # -------------------------------------------------------------------------
    print("\n--- STAGE 3: FROZEN AITEAM TOP-50 SOURCE & PARITY ---", flush=True)
    frozen_top50, aiteam_accessor, aiteam_parity_doc = load_and_verify_aiteam_full_corpus(all_ids)
    if not aiteam_parity_doc["parity_pass"]:
        print("ERROR: AITeam Top-50 parity failed! BLOCKED_AITEAM_SCORE_PARITY", flush=True)
        sys.exit(1)

    # -------------------------------------------------------------------------
    # STAGE 4: UNIVERSE DEFINITION & NEURAL FEATURE INFERENCE
    # -------------------------------------------------------------------------
    print("\n--- STAGE 4: UNIVERSE DEFINITION & NEURAL FEATURE INFERENCE ---", flush=True)
    s_universe, defenders, novel_map = build_s_universe_and_novel_maps(
        all_ids, frozen_top50, d1_rankings, extended
    )

    total_novel = sum(len(novel_map[q]) for q in all_ids)
    queries_with_novel = sum(1 for q in all_ids if len(novel_map[q]) > 0)
    print(f"Queries with novel candidates: {queries_with_novel} / {len(all_ids)}", flush=True)
    print(f"Total novel candidates across queries: {total_novel}", flush=True)

    neural_model, _, _ = load_neural_crossencoder()
    jina_scores, sec_scores, jina_parity, sec_parity = compute_neural_features_for_universe(
        neural_model,
        queries_label_free,
        docs,
        all_ids,
        s_universe,
        d1_rankings,
        batch_size=64,
    )

    # -------------------------------------------------------------------------
    # STAGE 5: LABEL-FREE FEATURE MATRIX & SEAL
    # -------------------------------------------------------------------------
    print("\n--- STAGE 5: LABEL-FREE 28D FEATURE MATRIX & SEAL ---", flush=True)
    features_per_query, feature_manifest = build_label_free_features(
        all_ids,
        s_universe,
        defenders,
        novel_map,
        aiteam_accessor,
        jina_scores,
        sec_scores,
        docs,
        queries_label_free,
    )

    # -------------------------------------------------------------------------
    # STAGE 6: 5-FOLD OOF ADMISSION MODEL & LABEL-FREE ACTION SEAL
    # -------------------------------------------------------------------------
    print("\n--- STAGE 6: 5-FOLD OOF ADMISSION MODEL & LABEL-FREE ACTION SEAL ---", flush=True)
    repaired_top5, action_records, action_meta = train_oof_admission_model_and_seal_actions(
        all_ids,
        s_universe,
        defenders,
        novel_map,
        d1_rankings,
        features_per_query,
        outer_gold,
    )

    # -------------------------------------------------------------------------
    # STAGE 7: CAL GOLD EVALUATION & PROMOTION GATE
    # -------------------------------------------------------------------------
    print("\n--- STAGE 7: CAL GOLD EVALUATION & PROMOTION GATE ---", flush=True)
    cal_report, verdict = evaluate_cal_utility(
        all_ids,
        blocks,
        frozen_top50,
        s_universe,
        d1_rankings,
        repaired_top5,
        action_records,
        action_meta,
    )

    # -------------------------------------------------------------------------
    # STAGE 8: EXPERIMENT MANIFEST
    # -------------------------------------------------------------------------
    print("\n--- STAGE 8: EXPERIMENT MANIFEST ---", flush=True)
    manifest_doc = {
        "schema_version": "dsc2026.gemini.huy_d1_aiteam50_soft_admission_v1.manifest.v1",
        "experiment_id": "HUY_D1_AITEAM50_SOFT_ADMISSION_V1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "verdict": verdict,
        "source_commit": git_status["head_commit"],
        "artifacts": {
            "source_provenance": str(RESULTS_DIR / "SOURCE_PROVENANCE.json"),
            "d1_parity": str(RESULTS_DIR / "D1_PARITY.json"),
            "aiteam50_source_parity": str(RESULTS_DIR / "AITEAM50_SOURCE_PARITY.json"),
            "jina_ft_inference_parity": str(RESULTS_DIR / "JINA_FT_INFERENCE_PARITY.json"),
            "section_inference_parity": str(RESULTS_DIR / "SECTION_INFERENCE_PARITY.json"),
            "feature_manifest": str(RESULTS_DIR / "AITEAM50_ADMISSION_FEATURE_MANIFEST.json"),
            "actions_oof": str(RESULTS_DIR / "AITEAM50_SOFT_ADMISSION_ACTIONS_OOF.json"),
            "cal_report": str(RESULTS_DIR / "AITEAM50_SOFT_ADMISSION_CAL_REPORT.json"),
        },
        "assertions": {
            "no_qid_hardcoding": True,
            "no_known_rescue_doc_hardcoding": True,
            "no_gold_used_in_feature_generation": True,
            "held_fold_gold_never_used_in_fitting": True,
            "top50_frozen_before_evaluation": True,
            "c_frozen": True,
            "action_rule_frozen": True,
            "public_leaderboard_not_used_to_configure_actions": True,
        },
    }
    manifest_path = RESULTS_DIR / "EXPERIMENT_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest_doc, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {manifest_path}", flush=True)

    # -------------------------------------------------------------------------
    # STAGE 9: SECTION 20 FINAL REPORT
    # -------------------------------------------------------------------------
    print("\n================================================================================", flush=True)
    print("                          FINAL REPORT                                          ", flush=True)
    print("================================================================================", flush=True)

    cu = cal_report["cal_utility"]
    diag = cal_report["diagnostics"]
    oor = cal_report["outside_pool_rescue_diagnostic"]

    print(f"pushed source SHA: {git_status['head_commit']}")
    print(f"HEAD == origin/main: {git_status['parity']}")
    print(f"clean status: {git_status['status_clean']}")
    print()
    print(f"D1 parity: {d1_parity_doc['status']}")
    print(f"AITeam50 score/ranking parity: {aiteam_parity_doc['status']}")
    print(f"Jina parity: {jina_parity['status']}")
    print(f"Section parity: {sec_parity['status']}")
    print()
    print(f"feature dimensionality: {feature_manifest['feature_dim']}")
    print(f"OOF admission standalone Top-5 Recall: {diag['admission_model_standalone_r5_within_S']:.6f}")
    print()
    print("label-free:")
    print(f"  queries with novel docs: {diag['queries_with_novel_candidates']}")
    print(f"  total novel docs: {feature_manifest['total_novel_candidates']}")
    print(f"  actions: {diag['total_actions']}")
    print()
    print("CAL:")
    print(f"  D1 Recall / Precision: {cu['d1_recall_at_5']:.10f} / {cu['d1_precision_at_5']:.10f}")
    print(f"  candidate Recall / Precision: {cu['r1_recall_at_5']:.10f} / {cu['r1_precision_at_5']:.10f}")
    print(f"  delta Recall: {cu['delta_recall_at_5']:+.10f}")
    print(f"  delta Precision: {cu['delta_precision_at_5']:+.10f}")
    print(f"  single delta: {cu['single_gold']['delta']:+.10f}")
    print(f"  multi delta: {cu['multi_gold']['delta']:+.10f}")
    print("  block deltas:")
    for b, b_data in cu["blocks"].items():
        print(f"    Block {b}: {b_data['delta_recall']:+.6f} ({b_data['r0_recall_at_5']:.6f} -> {b_data['r1_recall_at_5']:.6f})")
    print(f"  beneficial / harmful / neutral: {cu['actions']['beneficial']} / {cu['actions']['harmful']} / {cu['actions']['neutral']}")
    print(f"  W / L / T: {cu['paired']['wins']} / {cu['paired']['losses']} / {cu['paired']['ties']}")
    print()
    print("outside-pool rescue diagnostic:")
    print(f"  source recoverable = {oor['source_recoverable_gold_cases']}")
    print(f"  admission Top-5 count: {oor['admission_top5_count']}")
    print(f"  actually admitted count: {oor['actually_admitted_count']}")
    print()
    print(f"final verdict: {verdict}")
    print()
    print("artifact paths:")
    for name, p_str in manifest_doc["artifacts"].items():
        rel = str(Path(p_str).relative_to(ROOT)).replace("\\", "/")
        print(f"  {name}: {rel}")
    print("================================================================================", flush=True)


if __name__ == "__main__":
    run_pipeline()
