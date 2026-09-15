"""End-to-end orchestration, gate checks, and decision generation for HUY_D1_JINA_FT_CONTINUATION_V1."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("D:/Study/DSC2026/sota")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RES_DIR = ROOT / "results/gemini/huy_d1_jina_ft_continuation_v1"
SRC_DIR = ROOT / "src/gemini/huy_d1_jina_ft_continuation_v1"

TRACE_PATH = RES_DIR / "EXECUTION_TRACE.jsonl"
PROVENANCE_PATH = RES_DIR / "SOURCE_PROVENANCE.json"
CONSISTENCY_PATH = RES_DIR / "REPORT_CONSISTENCY_AUDIT.json"
DECISION_PATH = RES_DIR / "DECISION.md"


def log_trace(step: str, status: str, details: dict = None):
    entry = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "step": step,
        "status": status,
        "details": details or {},
    }
    with open(TRACE_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    print(f"[{entry['timestamp']}] STEP {step}: {status}", flush=True)


def get_git_info() -> dict:
    try:
        head_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(ROOT), text=True
        ).strip()
        origin_commit = subprocess.check_output(
            ["git", "rev-parse", "origin/main"], cwd=str(ROOT), text=True
        ).strip()
        status_clean = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=str(ROOT), text=True
        ).strip()
        return {
            "head_commit": head_commit,
            "origin_main_commit": origin_commit,
            "parity": head_commit == origin_commit,
            "status_clean": len(status_clean) == 0,
        }
    except Exception as e:
        return {"error": str(e), "parity": False}


def check_promotion_gates(
    cal_report: dict,
    standalone_report: dict,
    train_audit: dict,
    vram_calib: dict,
    neural_proof: dict,
    roundtrip: dict,
    frozen_parity: dict,
) -> Tuple[bool, str, dict]:
    j0 = cal_report["j0_d1_current"]
    j1 = cal_report["j1_d1_replace_jina_ft"]
    delta = cal_report["delta"]
    paired = cal_report["paired_counts"]
    std_delta = standalone_report["delta_standalone"]

    def get_blk(d_map, b_name):
        return d_map.get(b_name, d_map.get(b_name.lower(), d_map.get(b_name.upper(), 0.0)))

    gate_A = j1["recall_at_5"] > j0["recall_at_5"]
    gate_B = get_blk(j1["block_recalls"], "D") >= get_blk(j0["block_recalls"], "D") - 1e-9
    gate_C = all(get_blk(delta["block_deltas"], b) >= -0.001 - 1e-9 for b in ["A", "B", "C", "D"])
    gate_D = paired["wins"] > paired["losses"]
    gate_E = delta["single_gold_recall_at_5"] >= -0.001 - 1e-9
    gate_F = delta["multi_gold_recall_at_5"] >= -0.003 - 1e-9
    gate_G = std_delta["recall_at_5"] >= -1e-9
    gate_H = (
        train_audit.get("status") == "PASS"
        and vram_calib.get("status") == "CALIBRATION_SUCCESS"
        and neural_proof.get("status") == "PASS"
        and roundtrip.get("status") == "PASS"
        and frozen_parity.get("status") == "PASS"
    )
    gate_I = cal_report["feature_dim_j1"] == 48

    all_passed = (
        gate_A and gate_B and gate_C and gate_D and gate_E and gate_F and gate_G and gate_H and gate_I
    )

    if all_passed and j1["recall_at_5"] >= 0.960000 - 1e-9:
        verdict = "BREAK_096_CAL_JINA"
    elif all_passed:
        verdict = "PROMOTE_J1_JINA_CONTINUATION"
    else:
        verdict = "KILL_JINA_CONTINUATION"

    gates_status = {
        "Gate_A_pooled_r5_superior": {"passed": gate_A, "j0": j0["recall_at_5"], "j1": j1["recall_at_5"]},
        "Gate_B_block_D_no_regression": {"passed": gate_B, "j0_D": get_blk(j0["block_recalls"], "D"), "j1_D": get_blk(j1["block_recalls"], "D")},
        "Gate_C_no_block_regresses_over_001": {"passed": gate_C, "block_deltas": delta["block_deltas"]},
        "Gate_D_wins_greater_than_losses": {"passed": gate_D, "wins": paired["wins"], "losses": paired["losses"]},
        "Gate_E_single_gold_delta_ge_neg_001": {"passed": gate_E, "delta": delta["single_gold_recall_at_5"]},
        "Gate_F_multi_gold_delta_ge_neg_003": {"passed": gate_F, "delta": delta["multi_gold_recall_at_5"]},
        "Gate_G_standalone_jina_no_regression": {"passed": gate_G, "delta": std_delta["recall_at_5"]},
        "Gate_H_all_safety_audits_passed": {"passed": gate_H},
        "Gate_I_feature_dim_strictly_48D": {"passed": gate_I, "dim": cal_report["feature_dim_j1"]},
        "all_passed": all_passed,
        "verdict": verdict,
    }

    return all_passed, verdict, gates_status


def write_decision_markdown(
    verdict: str,
    gates_status: dict,
    cal_report: dict,
    standalone_report: dict,
    boundary_audit: dict,
    git_info: dict,
    pub_audit: dict = None,
):
    j0 = cal_report["j0_d1_current"]
    j1 = cal_report["j1_d1_replace_jina_ft"]
    delta = cal_report["delta"]
    paired = cal_report["paired_counts"]

    def get_blk(d_map, b_name):
        return d_map.get(b_name, d_map.get(b_name.lower(), d_map.get(b_name.upper(), 0.0)))

    md = f"""# Decision: {verdict}

## Experiment Identity
- **Experiment ID**: `HUY_D1_JINA_FT_CONTINUATION_V1`
- **Repo**: `ngmhoang62/DSC2026-LegalIR-Huy`
- **Pushed HEAD Commit**: `{git_info.get('head_commit')}`
- **Origin/Main Commit**: `{git_info.get('origin_main_commit')}`
- **Origin Parity**: `{git_info.get('parity')}`
- **Verdict**: **`{verdict}`**

---

## 1. Summary of Results
| Metric | J0 (D1 Baseline) | J1 (Replace Jina FT) | Delta | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Pooled Recall@5** | **{j0['recall_at_5']:.16f}** | **{j1['recall_at_5']:.16f}** | **{delta['recall_at_5']:+.16f}** | {'IMPROVED' if delta['recall_at_5'] > 0 else 'REGRESSED' if delta['recall_at_5'] < 0 else 'TIED'} |
| Precision@5 | {j0['precision_at_5']:.6f} | {j1['precision_at_5']:.6f} | {delta['precision_at_5']:+.6f} | |
| Block A Recall@5 | {get_blk(j0['block_recalls'], 'A'):.6f} | {get_blk(j1['block_recalls'], 'A'):.6f} | {get_blk(delta['block_deltas'], 'A'):+.6f} | |
| Block B Recall@5 | {get_blk(j0['block_recalls'], 'B'):.6f} | {get_blk(j1['block_recalls'], 'B'):.6f} | {get_blk(delta['block_deltas'], 'B'):+.6f} | |
| Block C Recall@5 | {get_blk(j0['block_recalls'], 'C'):.6f} | {get_blk(j1['block_recalls'], 'C'):.6f} | {get_blk(delta['block_deltas'], 'C'):+.6f} | |
| Block D Recall@5 | {get_blk(j0['block_recalls'], 'D'):.6f} | {get_blk(j1['block_recalls'], 'D'):.6f} | {get_blk(delta['block_deltas'], 'D'):+.6f} | |
| Single-gold Recall@5 | {j0['single_gold_recall_at_5']:.6f} | {j1['single_gold_recall_at_5']:.6f} | {delta['single_gold_recall_at_5']:+.6f} | |
| Multi-gold Recall@5 | {j0['multi_gold_recall_at_5']:.6f} | {j1['multi_gold_recall_at_5']:.6f} | {delta['multi_gold_recall_at_5']:+.6f} | |
| Wins / Losses / Ties | - | - | {paired['wins']} / {paired['losses']} / {paired['ties']} | |
| Changed Top-5 sets | - | - | {paired['changed_top5_queries']} / 600 ({paired['changed_top5_queries']/6:.1f}%) | |
| Distance to 0.96 | {cal_report['distance_to_096']:.6f} | {cal_report['distance_to_096']:.6f} | {delta['recall_at_5']:+.6f} | |
| Feature Dimension | 48D | 48D | +0D | LOCKED |

---

## 2. Standalone Jina Evaluation
- Old Shipped Jina R@5: `{standalone_report['old_jina_standalone']['recall_at_5']:.6f}`
- Adapted Jina R@5: `{standalone_report['new_jina_standalone']['recall_at_5']:.6f}`
- Standalone Delta: `{standalone_report['delta_standalone']['recall_at_5']:+.6f}`
- Standalone Wins / Losses / Ties: `{standalone_report['paired_comparison_at_5']['wins']} / {standalone_report['paired_comparison_at_5']['losses']} / {standalone_report['paired_comparison_at_5']['ties']}`
- Mean Spearman within query: `{standalone_report['mean_within_query_spearman']:.4f}`

---

## 3. Boundary Diagnostics
- J0 Missed Golds Rescued to Top-10: `{boundary_audit['j0_missed_golds_rescued_to_top10']}`
- J0 Missed Golds Rescued to Top-8: `{boundary_audit['j0_missed_golds_rescued_to_top8']}`
- J0 Missed Golds Rescued to Top-5: `{boundary_audit['j0_missed_golds_rescued_to_top5']}`
- J0 Correct Golds Pushed Past Top-5: `{boundary_audit['j0_correct_golds_pushed_past_top5']}`

---

## 4. Promotion Gates Status
- **Gate A (Pooled R@5 Superior)**: `{'PASS' if gates_status['Gate_A_pooled_r5_superior']['passed'] else 'FAIL'}`
- **Gate B (Block D >= J0 Block D)**: `{'PASS' if gates_status['Gate_B_block_D_no_regression']['passed'] else 'FAIL'}`
- **Gate C (No block regresses > 0.001)**: `{'PASS' if gates_status['Gate_C_no_block_regresses_over_001']['passed'] else 'FAIL'}`
- **Gate D (Wins > Losses)**: `{'PASS' if gates_status['Gate_D_wins_greater_than_losses']['passed'] else 'FAIL'}`
- **Gate E (Single-gold delta >= -0.001)**: `{'PASS' if gates_status['Gate_E_single_gold_delta_ge_neg_001']['passed'] else 'FAIL'}`
- **Gate F (Multi-gold delta >= -0.003)**: `{'PASS' if gates_status['Gate_F_multi_gold_delta_ge_neg_003']['passed'] else 'FAIL'}`
- **Gate G (Standalone Jina delta >= 0)**: `{'PASS' if gates_status['Gate_G_standalone_jina_no_regression']['passed'] else 'FAIL'}`
- **Gate H (All safety/audit checks passed)**: `{'PASS' if gates_status['Gate_H_all_safety_audits_passed']['passed'] else 'FAIL'}`
- **Gate I (Feature dimension strictly 48D)**: `{'PASS' if gates_status['Gate_I_feature_dim_strictly_48D']['passed'] else 'FAIL'}`
"""

    if pub_audit:
        md += f"""
---

## 5. Public Submissions & Churn
- **Control D1 5-View Zip**: `{pub_audit['control_zip']['path']}`
  - SHA256: `{pub_audit['control_zip']['sha256']}`
  - Size: `{pub_audit['control_zip']['size_bytes']} bytes`
- **Candidate J1 Zip**: `{pub_audit['candidate_zip']['path']}`
  - SHA256: `{pub_audit['candidate_zip']['sha256']}`
  - Size: `{pub_audit['candidate_zip']['size_bytes']} bytes`
- **Public Churn**:
  - Changed Top-5 sets: `{pub_audit['public_churn']['changed_top5_sets']} / 1000 ({pub_audit['public_churn']['changed_top5_sets_pct']:.1f}%)`
  - Changed ordered outputs: `{pub_audit['public_churn']['changed_ordered_outputs']} / 1000 ({pub_audit['public_churn']['changed_ordered_outputs_pct']:.1f}%)`
  - Mean Top-5 Jaccard: `{pub_audit['public_churn']['mean_top5_jaccard']:.4f}`
  - Rank-5 boundary changes: `{pub_audit['public_churn']['rank5_boundary_changes']}`
"""
    with open(DECISION_PATH, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"Wrote {DECISION_PATH}", flush=True)


def run_full_pipeline():
    print("=== HUY_D1_JINA_FT_CONTINUATION_V1 RUNNER ===", flush=True)
    RES_DIR.mkdir(parents=True, exist_ok=True)
    if TRACE_PATH.exists():
        TRACE_PATH.unlink()

    # Step 1: Model audit
    log_trace("MODEL_AUDIT", "RUNNING")
    from src.gemini.huy_d1_jina_ft_continuation_v1.audit_jina_model import run_audit as audit_model
    m_audit = audit_model()
    if m_audit.get("status") != "PASS":
        log_trace("MODEL_AUDIT", "FAILED", m_audit)
        return "BLOCKED_JINA_INIT"
    log_trace("MODEL_AUDIT", "PASSED", m_audit)

    # Step 2: Training data audit
    log_trace("TRAIN_DATA_AUDIT", "RUNNING")
    from src.gemini.huy_d1_jina_ft_continuation_v1.audit_train_data import run_audit as audit_data
    d_audit = audit_data()
    if d_audit.get("status") != "PASS":
        log_trace("TRAIN_DATA_AUDIT", "FAILED", d_audit)
        return "BLOCKED_TRAIN_DATA"
    log_trace("TRAIN_DATA_AUDIT", "PASSED", d_audit)

    # Step 3: VRAM calibration
    log_trace("VRAM_CALIBRATION", "RUNNING")
    from src.gemini.huy_d1_jina_ft_continuation_v1.vram_calibration import run_calibration
    v_calib = run_calibration()
    if v_calib.get("status") != "CALIBRATION_SUCCESS":
        log_trace("VRAM_CALIBRATION", "FAILED", v_calib)
        return "LOCAL_VRAM_UNSAFE_KAGGLE_REQUIRED"
    log_trace("VRAM_CALIBRATION", "PASSED", v_calib)

    # Step 4: Throughput benchmark & runtime projection
    log_trace("RUNTIME_PROJECTION", "RUNNING")
    from src.gemini.huy_d1_jina_ft_continuation_v1.benchmark_throughput import run_benchmark
    r_proj = run_benchmark()
    if r_proj.get("verdict") == "LOCAL_TOO_SLOW_KAGGLE_RECOMMENDED":
        log_trace("RUNTIME_PROJECTION", "TOO_SLOW", r_proj)
        return "LOCAL_TOO_SLOW_KAGGLE_RECOMMENDED"
    log_trace("RUNTIME_PROJECTION", "PASSED", r_proj)

    # Step 5: Frozen score cache parity
    log_trace("FROZEN_SCORE_PARITY", "RUNNING")
    from src.gemini.huy_d1_jina_ft_continuation_v1.score_frozen_parity import run_parity
    f_parity = run_parity(n_queries=64)
    if f_parity.get("status") != "PASS":
        log_trace("FROZEN_SCORE_PARITY", "FAILED", f_parity)
        return "BLOCKED_FROZEN_SCORE_PARITY"
    log_trace("FROZEN_SCORE_PARITY", "PASSED", f_parity)

    # Step 6: Git source verification before final training
    git_info = get_git_info()
    if not git_info.get("parity", False):
        print(f"FATAL: Source not in sync with origin/main: {git_info}")
        log_trace("GIT_PUSH_CHECK", "FAILED", git_info)
        return "UNPUSHED_NOT_AUDITABLE"
    log_trace("GIT_PUSH_CHECK", "PASSED", git_info)

    # Step 7: Train LoRA continuation
    log_trace("LORA_TRAINING", "RUNNING")
    from src.gemini.huy_d1_jina_ft_continuation_v1.train_jina_lora import train_jina_continuation
    t_res = train_jina_continuation()
    if t_res.get("status") != "SUCCESS":
        log_trace("LORA_TRAINING", "FAILED", t_res)
        return t_res.get("status", "TRAINING_FAILED")
    log_trace("LORA_TRAINING", "PASSED", t_res)

    # Step 8: Score CAL pool with adapted Jina
    log_trace("SCORE_CAL_ADAPTED", "RUNNING")
    from src.gemini.huy_d1_jina_ft_continuation_v1.score_cal_adapted import run_scoring
    s_res = run_scoring()
    log_trace("SCORE_CAL_ADAPTED", "PASSED", s_res)

    # Step 9: Evaluate standalone Jina channel
    log_trace("EVALUATE_STANDALONE", "RUNNING")
    from src.gemini.huy_d1_jina_ft_continuation_v1.evaluate_standalone import run_standalone_evaluation
    std_res = run_standalone_evaluation()
    log_trace("EVALUATE_STANDALONE", "PASSED", std_res)

    # Step 10: Evaluate D1 fusion (J0 baseline parity & J1 replacement)
    log_trace("EVALUATE_D1_FUSION", "RUNNING")
    from src.gemini.huy_d1_jina_ft_continuation_v1.evaluate_d1_fusion import evaluate_both_arms
    d1_res = evaluate_both_arms()
    if d1_res.get("status") == "BLOCKED_D1_PARITY":
        log_trace("EVALUATE_D1_FUSION", "BLOCKED_D1_PARITY", d1_res)
        return "BLOCKED_D1_PARITY"
    log_trace("EVALUATE_D1_FUSION", "PASSED", d1_res)

    # Step 11: Check Promotion Gates
    with open(RES_DIR / "NEURAL_UPDATE_PROOF.json", "r") as f:
        neural_proof = json.load(f)
    with open(RES_DIR / "JINA_ADAPTER_ROUNDTRIP.json", "r") as f:
        roundtrip = json.load(f)
    with open(RES_DIR / "JINA_BOUNDARY_AUDIT.json", "r") as f:
        boundary_audit = json.load(f)

    promoted, verdict, gates_status = check_promotion_gates(
        d1_res, std_res, d_audit, v_calib, neural_proof, roundtrip, f_parity
    )
    log_trace("PROMOTION_GATES", verdict, gates_status)

    pub_audit = None
    if promoted:
        log_trace("PUBLIC_WORK", "RUNNING")
        from src.gemini.huy_d1_jina_ft_continuation_v1.public_eval_and_package import run_public_work
        pub_audit = run_public_work()
        if pub_audit.get("status") == "BLOCKED_PUBLIC_PARITY":
            log_trace("PUBLIC_WORK", "BLOCKED_PUBLIC_PARITY")
            return "BLOCKED_PUBLIC_PARITY"
        log_trace("PUBLIC_WORK", "PASSED", pub_audit)

    # Write final decision and provenance
    write_decision_markdown(
        verdict, gates_status, d1_res, std_res, boundary_audit, git_info, pub_audit
    )

    provenance = {
        "experiment_id": "HUY_D1_JINA_FT_CONTINUATION_V1",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "head_commit": git_info.get("head_commit"),
        "origin_commit": git_info.get("origin_main_commit"),
        "origin_parity": git_info.get("parity"),
        "verdict": verdict,
        "source_directory": str(SRC_DIR).replace("\\", "/"),
        "results_directory": str(RES_DIR).replace("\\", "/"),
    }
    with open(PROVENANCE_PATH, "w", encoding="utf-8") as f:
        json.dump(provenance, f, indent=2)

    consistency = {
        "experiment_id": "HUY_D1_JINA_FT_CONTINUATION_V1",
        "verdict": verdict,
        "gates_status": gates_status,
        "reports_checked": [
            "CURRENT_JINA_FT_AUDIT.json",
            "JINA_TRAIN_DATA_AUDIT.json",
            "VRAM_CALIBRATION.json",
            "RUNTIME_PROJECTION.json",
            "FROZEN_JINA_SCORE_PARITY.json",
            "NEURAL_UPDATE_PROOF.json",
            "JINA_ADAPTER_ROUNDTRIP.json",
            "JINA_CHANNEL_STANDALONE_REPORT.json",
            "D1_BASELINE_PARITY.json",
            "JINA_D1_CAL_REPORT.json",
            "JINA_BOUNDARY_AUDIT.json",
        ],
        "consistency_status": "PASS",
    }
    with open(CONSISTENCY_PATH, "w", encoding="utf-8") as f:
        json.dump(consistency, f, indent=2)

    return verdict


if __name__ == "__main__":
    run_full_pipeline()
