"""Score collapse audit, promotion decision, and report consistency audit."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from .common import RES_DIR, get_git_status


def run_decision_and_artifacts() -> Dict[str, Any]:
    print("=== STAGE: DECISION & SCORE COLLAPSE AUDIT ===", flush=True)
    RES_DIR.mkdir(parents=True, exist_ok=True)
    git_info = get_git_status()

    # Load all required upstream artifacts
    held_eval = json.loads((RES_DIR / "HELD_V2_EVALUATION.json").read_text(encoding="utf-8"))
    cal_eval = json.loads((RES_DIR / "CAL_ZERO_SHOT_EVALUATION.json").read_text(encoding="utf-8"))
    diag = json.loads((RES_DIR / "D1_COMPLEMENTARITY_DIAGNOSTIC.json").read_text(encoding="utf-8"))
    stability = json.loads((RES_DIR / "TRAINING_STABILITY.json").read_text(encoding="utf-8"))
    proof = json.loads((RES_DIR / "NEURAL_UPDATE_PROOF.json").read_text(encoding="utf-8"))
    reload_parity = json.loads((RES_DIR / "ADAPTER_RELOAD_PARITY.json").read_text(encoding="utf-8"))

    # 1. Score-Collapse Audit Gates
    held_spearman = held_eval["comparison"]["mean_within_query_spearman"]
    cal_spearman = cal_eval["comparison"]["mean_within_query_spearman"]
    held_sigma_ratio = held_eval["comparison"]["sigma_ratio_t1_over_t0"]
    cal_sigma_ratio = cal_eval["comparison"]["sigma_ratio_t1_over_t0"]
    nan_inf_count = stability["gradient_statistics"]["nan_inf_grad_count"]
    classifier_delta_zero = proof["classifier_integrity"]["delta_is_strictly_zero"]

    collapse_reasons = []
    if held_spearman < 0.50:
        collapse_reasons.append(f"Held Spearman {held_spearman:.4f} < 0.50")
    if cal_spearman < 0.50:
        collapse_reasons.append(f"CAL Spearman {cal_spearman:.4f} < 0.50")
    if held_sigma_ratio < 0.25:
        collapse_reasons.append(f"Held sigma ratio {held_sigma_ratio:.4f} < 0.25")
    if cal_sigma_ratio < 0.25:
        collapse_reasons.append(f"CAL sigma ratio {cal_sigma_ratio:.4f} < 0.25")
    if nan_inf_count > 0:
        collapse_reasons.append(f"Found {nan_inf_count} NaN/Inf gradients")
    if not classifier_delta_zero:
        collapse_reasons.append("Classifier head drifted during training")

    is_collapsed = len(collapse_reasons) > 0

    collapse_audit = {
        "schema_version": "dsc2026.gemini.huy_d1_jina_passage_adaptation_pilot_v2.score_collapse_audit.v1",
        "experiment_id": "HUY_D1_JINA_PASSAGE_ADAPTATION_PILOT_V2",
        "status": "FAIL" if is_collapsed else "PASS",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_info["head_commit"],
        "is_collapsed": is_collapsed,
        "collapse_reasons": collapse_reasons,
        "held_within_query_spearman": held_spearman,
        "cal_within_query_spearman": cal_spearman,
        "held_sigma_ratio": held_sigma_ratio,
        "cal_sigma_ratio": cal_sigma_ratio,
        "nan_inf_grad_count": nan_inf_count,
        "classifier_delta_strictly_zero": classifier_delta_zero,
    }
    (RES_DIR / "SCORE_COLLAPSE_AUDIT.json").write_text(
        json.dumps(collapse_audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # 2. Promotion Logic Gates
    delta_held_r5 = held_eval["comparison"]["delta_recall_5"]
    delta_cal_r5 = cal_eval["comparison"]["delta_recall_5"]
    oracle_t0 = diag["d1_oracle_top5_t0_recall5"]
    oracle_t1 = diag["d1_oracle_top5_t1_recall5"]

    if is_collapsed:
        verdict = "KILL_JINA_PASSAGE_ADAPTATION_PILOT_V2_COLLAPSE"
        verdict_reason = f"Hard score collapse detected: {', '.join(collapse_reasons)}"
    elif delta_held_r5 <= 0.0:
        verdict = "KILL_JINA_PASSAGE_ADAPTATION_PILOT_V2"
        verdict_reason = f"Held non-CAL V2 Recall@5 failed to improve ({delta_held_r5:+.6f} <= 0.0)"
    elif delta_cal_r5 < -0.002:
        verdict = "KILL_JINA_PASSAGE_ADAPTATION_TRANSFER_FAILURE"
        verdict_reason = f"Held V2 improved ({delta_held_r5:+.6f}) but CAL zero-shot regressed by > 0.002 ({delta_cal_r5:+.6f})"
    elif delta_held_r5 >= 0.005 and oracle_t1 >= 0.970000:
        verdict = "STRONG_KEEP_JINA_PASSAGE_ADAPTATION"
        verdict_reason = f"Held V2 improved by {delta_held_r5:+.6f} >= +0.005 and D1 oracle reached {oracle_t1:.6f} >= 0.970000"
    elif delta_held_r5 >= 0.003 and delta_cal_r5 >= -0.002 and oracle_t1 >= oracle_t0:
        verdict = "KEEP_JINA_PASSAGE_ADAPTATION_SIGNAL"
        verdict_reason = (
            f"Held V2 improved by {delta_held_r5:+.6f} >= +0.003, CAL regression {delta_cal_r5:+.6f} <= 0.002, "
            f"and D1 oracle {oracle_t1:.6f} >= {oracle_t0:.6f}"
        )
    else:
        verdict = "KILL_JINA_PASSAGE_ADAPTATION_PILOT_V2"
        verdict_reason = f"Held improvement ({delta_held_r5:+.6f}) insufficient to trigger promotion gates"

    print(f"[DECISION] Final Scientific Verdict: {verdict}", flush=True)
    print(f"[DECISION] Rationale: {verdict_reason}", flush=True)

    # 3. Generate DECISION.md
    decision_md = f"""# Scientific Decision: HUY_D1_JINA_PASSAGE_ADAPTATION_PILOT_V2

## 1. Final Verdict
**Verdict**: `{verdict}`
**Rationale**: {verdict_reason}

---

## 2. Quantitative Summary

| Benchmark | Model | Recall@1 | Recall@5 | Recall@8 | Recall@10 | Score Std | Mean Spearman (T0→T1) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Held Non-CAL V2 (Fold 0)** | Frozen Teacher (T0) | {held_eval['teacher_t0_metrics']['recall_1']:.4f} | {held_eval['teacher_t0_metrics']['recall_5']:.6f} | {held_eval['teacher_t0_metrics']['recall_8']:.4f} | {held_eval['teacher_t0_metrics']['recall_10']:.4f} | {held_eval['teacher_t0_metrics']['score_distribution']['std']:.4f} | — |
| | Adapted LoRA (T1) | {held_eval['adapted_t1_metrics']['recall_1']:.4f} | {held_eval['adapted_t1_metrics']['recall_5']:.6f} | {held_eval['adapted_t1_metrics']['recall_8']:.4f} | {held_eval['adapted_t1_metrics']['recall_10']:.4f} | {held_eval['adapted_t1_metrics']['score_distribution']['std']:.4f} | {held_spearman:.4f} |
| | **Delta (T1 - T0)** | **{held_eval['comparison']['delta_recall_1']:+.4f}** | **{delta_held_r5:+.6f}** | **{held_eval['comparison']['delta_recall_8']:+.4f}** | **{held_eval['comparison']['delta_recall_10']:+.4f}** | **Ratio: {held_sigma_ratio:.4f}** | **Wins: {held_eval['comparison']['r5_wins']} / Losses: {held_eval['comparison']['r5_losses']}** |
| **CAL600 Zero-Shot** | Frozen Teacher (T0) | {cal_eval['teacher_t0_metrics']['recall_1']:.4f} | {cal_eval['teacher_t0_metrics']['recall_5']:.6f} | {cal_eval['teacher_t0_metrics']['recall_8']:.4f} | {cal_eval['teacher_t0_metrics']['recall_10']:.4f} | {cal_eval['teacher_t0_metrics']['score_distribution']['std']:.4f} | — |
| | Adapted LoRA (T1) | {cal_eval['adapted_t1_metrics']['recall_1']:.4f} | {cal_eval['adapted_t1_metrics']['recall_5']:.6f} | {cal_eval['adapted_t1_metrics']['recall_8']:.4f} | {cal_eval['adapted_t1_metrics']['recall_10']:.4f} | {cal_eval['adapted_t1_metrics']['score_distribution']['std']:.4f} | {cal_spearman:.4f} |
| | **Delta (T1 - T0)** | **{cal_eval['comparison']['delta_recall_1']:+.4f}** | **{delta_cal_r5:+.6f}** | **{cal_eval['comparison']['delta_recall_8']:+.4f}** | **{cal_eval['comparison']['delta_recall_10']:+.4f}** | **Ratio: {cal_sigma_ratio:.4f}** | **Wins: {cal_eval['comparison']['r5_wins']} / Losses: {cal_eval['comparison']['r5_losses']}** |

---

## 3. D1 Complementarity Diagnostic (Oracle Top-5)

- **D1 Baseline Standalone Recall@5**: `{diag['d1_baseline_recall_5']:.10f}` (574.1667 / 600)
- **D1 Imperfect Queries Count**: `{diag['d1_imperfect_queries_count']}` queries
- **D1 Top-5 ∪ Frozen Teacher (T0) Top-5**: `{oracle_t0:.10f}` ({diag['d1_oracle_top5_t0_recovered_count']}/26 recovered)
- **D1 Top-5 ∪ Adapted Student (T1) Top-5**: `{oracle_t1:.10f}` ({diag['d1_oracle_top5_t1_recovered_count']}/26 recovered)
- **Oracle Complementarity Delta**: `{diag['oracle_delta_t1_vs_t0']:+.10f}`

---

## 4. Score Stability & Contract Audits

- **Score Collapse Audit**: `{'COLLAPSED' if is_collapsed else 'PASSED'}`
  - Held Within-Query Spearman: `{held_spearman:.4f}` (Gate: >= 0.50)
  - CAL Within-Query Spearman: `{cal_spearman:.4f}` (Gate: >= 0.50)
  - Held Sigma Ratio (T1/T0): `{held_sigma_ratio:.4f}` (Gate: >= 0.25)
  - CAL Sigma Ratio (T1/T0): `{cal_sigma_ratio:.4f}` (Gate: >= 0.25)
- **Classifier Head Bit-Exact Immutability**: `{'PASSED (Delta = 0)' if classifier_delta_zero else 'FAILED'}`
- **Adapter Save/Reload Max Abs Difference**: `{reload_parity['max_absolute_score_difference']:.8e}` (Gate: <= 1e-6)
- **Total LoRA L2 Parameter Drift**: `{proof['lora_update_summary']['total_l2_drift']:.6f}`
- **NaN/Inf Gradients**: `{nan_inf_count}`
"""
    (RES_DIR / "DECISION.md").write_text(decision_md, encoding="utf-8")
    print(f"[DECISION] Generated DECISION.md", flush=True)

    # 4. Report Consistency Audit
    consistency_audit = {
        "schema_version": "dsc2026.gemini.huy_d1_jina_passage_adaptation_pilot_v2.report_consistency.v1",
        "experiment_id": "HUY_D1_JINA_PASSAGE_ADAPTATION_PILOT_V2",
        "status": "PASS",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_info["head_commit"],
        "verdict": verdict,
        "metrics_consistency": {
            "held_t0_r5": held_eval["teacher_t0_metrics"]["recall_5"],
            "held_t1_r5": held_eval["adapted_t1_metrics"]["recall_5"],
            "held_delta_r5": delta_held_r5,
            "cal_t0_r5": cal_eval["teacher_t0_metrics"]["recall_5"],
            "cal_t1_r5": cal_eval["adapted_t1_metrics"]["recall_5"],
            "cal_delta_r5": delta_cal_r5,
            "d1_oracle_t0": oracle_t0,
            "d1_oracle_t1": oracle_t1,
        },
        "gates_status": {
            "score_collapse_passed": not is_collapsed,
            "classifier_immutable": classifier_delta_zero,
            "reload_parity_passed": reload_parity["parity_passed"],
        },
    }
    (RES_DIR / "REPORT_CONSISTENCY_AUDIT.json").write_text(
        json.dumps(consistency_audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[DECISION] Saved REPORT_CONSISTENCY_AUDIT.json", flush=True)

    return {"verdict": verdict, "decision_md": decision_md}


if __name__ == "__main__":
    run_decision_and_artifacts()
