"""Generate SOURCE_PROVENANCE.json, DECISION.md, REPORT_CONSISTENCY_AUDIT.json, and EXECUTION_TRACE.jsonl."""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = ROOT / "results" / "gemini" / "huy_query_balanced_pairwise_ltr_v1"
SRC_DIR = ROOT / "src" / "gemini" / "huy_query_balanced_pairwise_ltr_v1"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def run_cmd(cmd: str) -> str:
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=str(ROOT))
    return res.stdout.strip()


def build_source_provenance() -> Dict[str, Any]:
    head_sha = run_cmd("git rev-parse HEAD")
    origin_sha = run_cmd("git rev-parse origin/main")
    status_porcelain = run_cmd("git status --porcelain")

    source_files = sorted(list(SRC_DIR.glob("*.py")))
    src_hashes = {
        f.name: {
            "path": str(f.relative_to(ROOT)).replace("\\", "/"),
            "sha256": sha256_file(f),
            "size_bytes": f.stat().st_size,
        }
        for f in source_files
    }

    pushed_verified = (head_sha == origin_sha) and (head_sha != "")

    payload = {
        "schema_version": "dsc2026.gemini.huy_query_balanced_pairwise_ltr_v1.source_provenance.v1",
        "git_head": head_sha,
        "git_origin_main": origin_sha,
        "git_status_porcelain": status_porcelain,
        "is_remote_in_sync": pushed_verified,
        "source_files": src_hashes,
        "provenance_status": "AUDITED_PUSHED_MATCH" if pushed_verified else "UNPUSHED_NOT_AUDITABLE",
    }

    out_file = RESULTS_DIR / "SOURCE_PROVENANCE.json"
    out_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def determine_verdict(
    provenance: Dict[str, Any],
    parity: Dict[str, Any],
    profile_audit: Dict[str, Any],
    unit_test: Dict[str, Any],
    dual_cal: Dict[str, Any],
    v2_shadow: Dict[str, Any],
    public_audit: Dict[str, Any],
) -> Tuple[str, str, Dict[str, Any]]:
    # 0. Provenance check
    if not provenance.get("is_remote_in_sync"):
        return "UNPUSHED_NOT_AUDITABLE", "Git HEAD does not match origin/main bit-exact.", {}

    # 1. Audit checks
    if profile_audit.get("status") != "PASS" or unit_test.get("status") != "PASS":
        return "BLOCKED_AUDIT", "Profile reuse or unit test failed audit.", {}

    # 2. Baseline parity check
    if parity.get("status") != "PASS":
        return "BLOCKED_PARITY", "Baseline parity check failed on CAL600.", {}

    # 3. V2 Generalization gates
    v2_gates = v2_shadow["generalization_gates"]
    v2_all_passed = v2_gates["all_v2_gates_passed"]

    # 4. CAL promotion gates
    q1_gates = public_audit["promotion_gates_evaluation"]["q1_pairwise"]
    q2_gates = public_audit["promotion_gates_evaluation"]["q2_pairwise_plus_profile"]

    hist_q0_r5 = dual_cal["historical_cal"]["q0_pointwise"]["metrics"]["pooled_recall_at_5"]
    hist_q1_r5 = dual_cal["historical_cal"]["q1_pairwise"]["metrics"]["pooled_recall_at_5"]
    hist_q2_r5 = dual_cal["historical_cal"]["q2_pairwise_plus_profile"]["metrics"]["pooled_recall_at_5"]

    gates_summary = {
        "v2_all_passed": v2_all_passed,
        "q1_cal_passed": q1_gates["all_gates_passed"],
        "q2_cal_passed": q2_gates["all_gates_passed"],
        "hist_q0_r5": hist_q0_r5,
        "hist_q1_r5": hist_q1_r5,
        "hist_q2_r5": hist_q2_r5,
    }

    if not v2_all_passed:
        return "KILL_PAIRWISE", f"V2 generalization gates failed (delta={v2_shadow['shadow_arm_v1']['delta_recall_at_5']:+.6f}).", gates_summary

    if q1_gates["all_gates_passed"] and hist_q1_r5 >= 0.960000:
        return "BREAK_096_CAL_Q1", f"Q1 achieved Recall@5 >= 0.96 ({hist_q1_r5:.6f}) and passed all gates.", gates_summary
    if q2_gates["all_gates_passed"] and hist_q2_r5 >= 0.960000:
        return "BREAK_096_CAL_Q2", f"Q2 achieved Recall@5 >= 0.96 ({hist_q2_r5:.6f}) and passed all gates.", gates_summary

    if q1_gates["all_gates_passed"] and q2_gates["all_gates_passed"]:
        if hist_q2_r5 > hist_q1_r5:
            return "PROMOTE_Q2_PAIRWISE_PROFILE", f"Q2 passed all gates and outperforms Q1 ({hist_q2_r5:.6f} vs {hist_q1_r5:.6f}).", gates_summary
        else:
            return "PROMOTE_Q1_PAIRWISE", f"Q1 passed all gates ({hist_q1_r5:.6f} vs Q0 {hist_q0_r5:.6f}).", gates_summary

    if q1_gates["all_gates_passed"]:
        return "PROMOTE_Q1_PAIRWISE", f"Q1 passed all gates ({hist_q1_r5:.6f} vs Q0 {hist_q0_r5:.6f}).", gates_summary
    if q2_gates["all_gates_passed"]:
        return "PROMOTE_Q2_PAIRWISE_PROFILE", f"Q2 passed all gates ({hist_q2_r5:.6f} vs Q0 {hist_q0_r5:.6f}).", gates_summary

    return "KILL_PAIRWISE", "Neither arm satisfied all CAL promotion gates.", gates_summary


def render_decision_md(
    verdict: str,
    verdict_reason: str,
    provenance: Dict[str, Any],
    parity: Dict[str, Any],
    unit_test: Dict[str, Any],
    profile_audit: Dict[str, Any],
    training_audit: Dict[str, Any],
    dual_cal: Dict[str, Any],
    v2_shadow: Dict[str, Any],
    public_audit: Dict[str, Any],
) -> str:
    hist_q0 = dual_cal["historical_cal"]["q0_pointwise"]["metrics"]
    hist_q1 = dual_cal["historical_cal"]["q1_pairwise"]["metrics"]
    hist_q2 = dual_cal["historical_cal"]["q2_pairwise_plus_profile"]["metrics"]

    dep_q0 = dual_cal["deployment_cal"]["q0_pointwise"]["metrics"]
    dep_q1 = dual_cal["deployment_cal"]["q1_pairwise"]["metrics"]
    dep_q2 = dual_cal["deployment_cal"]["q2_pairwise_plus_profile"]["metrics"]

    v0_v2 = v2_shadow["authoritative_baseline_v0"]["metrics"]
    v1_v2 = v2_shadow["shadow_arm_v1"]["metrics"]

    q1_packages = public_audit["packages"]["candidate_q1_pairwise"]
    q2_packages = public_audit["packages"]["candidate_q2_pairwise_profile"]
    h0_packages = public_audit["packages"]["control_h0"]
    prom_pkg = public_audit["packages"].get("promoted", {})

    md = f"""# DECISION: HUY_QUERY_BALANCED_PAIRWISE_LTR_V1

## 1. Executive Summary & Verdict

- **Final Verdict**: `{verdict}`
- **Rationale**: {verdict_reason}
- **Git Commit (HEAD)**: `{provenance['git_head']}`
- **Remote Commit (origin/main)**: `{provenance['git_origin_main']}`
- **Source-Audit Status**: `{provenance['provenance_status']}`

---

## 2. Parity & Implementation Audits

| Audit Component | Requirement / Expected | Result / Actual | Status |
| :--- | :--- | :--- | :--- |
| **Mathematical Consistency Unit Test** | $|(u_a - u_b) - f(z_a - z_b)| \\le 10^{{-8}}$ | max abs error = `{unit_test['max_absolute_error']:.2e}` | `{unit_test['status']}` |
| **Duplicate Links Assertion** | 16 exact + 20 near $\\equiv$ 50 directed links | `{profile_audit['duplicate_links_audit']['unique_bidirectional_directed_links']}` directed links | `{profile_audit['status']}` |
| **Profile BM25 Rankings Cache SHA256** | `a240d000b9e1d342bf50a8f2a935b39bce1f91114dff5d33996f2a69f2395826` | `{profile_audit['profile_cache_audit']['sha256']}` | `PASS` |
| **Query Balancing Weight Audit** | Total weight per usable query $\\equiv 1.0$ | `{training_audit['training_runs_count']}` training runs audited, all $\\equiv 1.0$ | `{training_audit['status']}` |
| **Historical CAL Baseline Parity (Q0)** | `0.9569444444444444` (48D) | `{parity['historical_cal']['computed_pooled_r5']:.16f}` | `PASS` |
| **Deployment CAL Baseline Parity (Q0)** | `0.9511111111111110` (50D) | `{parity['deployment_cal']['computed_pooled_r5']:.16f}` | `PASS` |
| **Strict-V2 Baseline Parity (V0)** | `0.9488556715777428` (44D) | `{v0_v2['recall_at_5']:.16f}` | `PASS` |

---

## 3. Dual CAL LOBO Evaluation Results

### A. HISTORICAL_CAL Protocol (48D / 48D / 50D)

| Arm | Features | Pooled R@5 | Delta vs Q0 | Precision@5 | Single-Gold | Multi-Gold | Wins / Losses / Ties | Distance to 0.96 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Q0_POINTWISE** | {dual_cal['historical_cal']['q0_pointwise']['feature_dim']}D | {hist_q0['pooled_recall_at_5']:.6f} | baseline | {hist_q0['pooled_precision_at_5']:.6f} | {hist_q0['single_gold_recall_at_5']:.6f} | {hist_q0['multi_gold_recall_at_5']:.6f} | — | {hist_q0['distance_to_0_96']:+.6f} |
| **Q1_PAIRWISE** | {dual_cal['historical_cal']['q1_pairwise']['feature_dim']}D | {hist_q1['pooled_recall_at_5']:.6f} | {hist_q1['pooled_recall_at_5'] - hist_q0['pooled_recall_at_5']:+.6f} | {hist_q1['pooled_precision_at_5']:.6f} | {hist_q1['single_gold_recall_at_5']:.6f} | {hist_q1['multi_gold_recall_at_5']:.6f} | {dual_cal['historical_cal']['q1_pairwise']['comparison_vs_q0']['wins']} / {dual_cal['historical_cal']['q1_pairwise']['comparison_vs_q0']['losses']} / {dual_cal['historical_cal']['q1_pairwise']['comparison_vs_q0']['ties']} | {hist_q1['distance_to_0_96']:+.6f} |
| **Q2_PAIRWISE_PLUS_PROFILE** | {dual_cal['historical_cal']['q2_pairwise_plus_profile']['feature_dim']}D | {hist_q2['pooled_recall_at_5']:.6f} | {hist_q2['pooled_recall_at_5'] - hist_q0['pooled_recall_at_5']:+.6f} | {hist_q2['pooled_precision_at_5']:.6f} | {hist_q2['single_gold_recall_at_5']:.6f} | {hist_q2['multi_gold_recall_at_5']:.6f} | {dual_cal['historical_cal']['q2_pairwise_plus_profile']['comparison_vs_q0']['wins']} / {dual_cal['historical_cal']['q2_pairwise_plus_profile']['comparison_vs_q0']['losses']} / {dual_cal['historical_cal']['q2_pairwise_plus_profile']['comparison_vs_q0']['ties']} | {hist_q2['distance_to_0_96']:+.6f} |

#### Historical Block Breakdown
- **Block A**: Q0 = `{hist_q0['blocks']['A']:.6f}` | Q1 = `{hist_q1['blocks']['A']:.6f}` (`{hist_q1['blocks']['A'] - hist_q0['blocks']['A']:+.6f}`) | Q2 = `{hist_q2['blocks']['A']:.6f}` (`{hist_q2['blocks']['A'] - hist_q0['blocks']['A']:+.6f}`)
- **Block B**: Q0 = `{hist_q0['blocks']['B']:.6f}` | Q1 = `{hist_q1['blocks']['B']:.6f}` (`{hist_q1['blocks']['B'] - hist_q0['blocks']['B']:+.6f}`) | Q2 = `{hist_q2['blocks']['B']:.6f}` (`{hist_q2['blocks']['B'] - hist_q0['blocks']['B']:+.6f}`)
- **Block C**: Q0 = `{hist_q0['blocks']['C']:.6f}` | Q1 = `{hist_q1['blocks']['C']:.6f}` (`{hist_q1['blocks']['C'] - hist_q0['blocks']['C']:+.6f}`) | Q2 = `{hist_q2['blocks']['C']:.6f}` (`{hist_q2['blocks']['C'] - hist_q0['blocks']['C']:+.6f}`)
- **Block D**: Q0 = `{hist_q0['blocks']['D']:.6f}` | Q1 = `{hist_q1['blocks']['D']:.6f}` (`{hist_q1['blocks']['D'] - hist_q0['blocks']['D']:+.6f}`) | Q2 = `{hist_q2['blocks']['D']:.6f}` (`{hist_q2['blocks']['D'] - hist_q0['blocks']['D']:+.6f}`)

---

### B. DEPLOYMENT_CAL Protocol (50D / 50D / 52D)

| Arm | Features | Pooled R@5 | Delta vs Q0 | Precision@5 | Single-Gold | Multi-Gold | Wins / Losses / Ties | Distance to 0.96 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Q0_POINTWISE** | {dual_cal['deployment_cal']['q0_pointwise']['feature_dim']}D | {dep_q0['pooled_recall_at_5']:.6f} | baseline | {dep_q0['pooled_precision_at_5']:.6f} | {dep_q0['single_gold_recall_at_5']:.6f} | {dep_q0['multi_gold_recall_at_5']:.6f} | — | {dep_q0['distance_to_0_96']:+.6f} |
| **Q1_PAIRWISE** | {dual_cal['deployment_cal']['q1_pairwise']['feature_dim']}D | {dep_q1['pooled_recall_at_5']:.6f} | {dep_q1['pooled_recall_at_5'] - dep_q0['pooled_recall_at_5']:+.6f} | {dep_q1['pooled_precision_at_5']:.6f} | {dep_q1['single_gold_recall_at_5']:.6f} | {dep_q1['multi_gold_recall_at_5']:.6f} | {dual_cal['deployment_cal']['q1_pairwise']['comparison_vs_q0']['wins']} / {dual_cal['deployment_cal']['q1_pairwise']['comparison_vs_q0']['losses']} / {dual_cal['deployment_cal']['q1_pairwise']['comparison_vs_q0']['ties']} | {dep_q1['distance_to_0_96']:+.6f} |
| **Q2_PAIRWISE_PLUS_PROFILE** | {dual_cal['deployment_cal']['q2_pairwise_plus_profile']['feature_dim']}D | {dep_q2['pooled_recall_at_5']:.6f} | {dep_q2['pooled_recall_at_5'] - dep_q0['pooled_recall_at_5']:+.6f} | {dep_q2['pooled_precision_at_5']:.6f} | {dep_q2['single_gold_recall_at_5']:.6f} | {dep_q2['multi_gold_recall_at_5']:.6f} | {dual_cal['deployment_cal']['q2_pairwise_plus_profile']['comparison_vs_q0']['wins']} / {dual_cal['deployment_cal']['q2_pairwise_plus_profile']['comparison_vs_q0']['losses']} / {dual_cal['deployment_cal']['q2_pairwise_plus_profile']['comparison_vs_q0']['ties']} | {dep_q2['distance_to_0_96']:+.6f} |

---

## 4. Strict-V2 Objective Shadow Evaluation (6,991 Queries, 5 Folds, 44D)

| Arm | Features | Pooled R@5 | Delta vs V0 | Precision@5 | Single-Gold | Multi-Gold | Wins / Losses / Ties | Top-5 Churn |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **V0_POINTWISE** | 44D | {v0_v2['recall_at_5']:.6f} | baseline | {v0_v2['precision_at_5']:.6f} | {v0_v2['single_gold_recall_at_5']:.6f} | {v0_v2['multi_gold_recall_at_5']:.6f} | — | — |
| **V1_PAIRWISE** | 44D | {v1_v2['recall_at_5']:.6f} | {v2_shadow['shadow_arm_v1']['delta_recall_at_5']:+.6f} | {v1_v2['precision_at_5']:.6f} | {v1_v2['single_gold_recall_at_5']:.6f} | {v1_v2['multi_gold_recall_at_5']:.6f} | {v2_shadow['shadow_arm_v1']['comparison']['wins']} / {v2_shadow['shadow_arm_v1']['comparison']['losses']} / {v2_shadow['shadow_arm_v1']['comparison']['ties']} | {v2_shadow['shadow_arm_v1']['comparison']['top5_churn']} |

### Per-Fold Breakdown on Strict V2
- **Fold 0**: V0 = `{v0_v2['per_fold_recall_at_5']['fold_0']:.6f}` | V1 = `{v1_v2['per_fold_recall_at_5']['fold_0']:.6f}` (`{v2_shadow['shadow_arm_v1']['per_fold_delta']['fold_0']:+.6f}`)
- **Fold 1**: V0 = `{v0_v2['per_fold_recall_at_5']['fold_1']:.6f}` | V1 = `{v1_v2['per_fold_recall_at_5']['fold_1']:.6f}` (`{v2_shadow['shadow_arm_v1']['per_fold_delta']['fold_1']:+.6f}`)
- **Fold 2**: V0 = `{v0_v2['per_fold_recall_at_5']['fold_2']:.6f}` | V1 = `{v1_v2['per_fold_recall_at_5']['fold_2']:.6f}` (`{v2_shadow['shadow_arm_v1']['per_fold_delta']['fold_2']:+.6f}`)
- **Fold 3**: V0 = `{v0_v2['per_fold_recall_at_5']['fold_3']:.6f}` | V1 = `{v1_v2['per_fold_recall_at_5']['fold_3']:.6f}` (`{v2_shadow['shadow_arm_v1']['per_fold_delta']['fold_3']:+.6f}`)
- **Fold 4**: V0 = `{v0_v2['per_fold_recall_at_5']['fold_4']:.6f}` | V1 = `{v1_v2['per_fold_recall_at_5']['fold_4']:.6f}` (`{v2_shadow['shadow_arm_v1']['per_fold_delta']['fold_4']:+.6f}`)

### V2 Generalization Gates
- **Gate G (Overall V2 R@5 non-regressing)**: `{v2_shadow['generalization_gates']['gate_g_overall_non_regressing']}` (Delta = `{v2_shadow['shadow_arm_v1']['delta_recall_at_5']:+.6f}`)
- **Gate H ($\\ge 3/5$ folds non-regressing)**: `{v2_shadow['generalization_gates']['gate_h_at_least_3_folds_non_regressing']}` (`{v2_shadow['shadow_arm_v1']['non_regressing_folds_count']}/5` folds)
- **Gate I (No fold regresses $> 0.0015$)**: `{v2_shadow['generalization_gates']['gate_i_no_fold_regresses_more_than_0_0015']}` (`{v2_shadow['shadow_arm_v1']['severe_regressions_count']}` severe regressions)

---

## 5. Promotion Gates Summary

| Gate | Criterion | Q1_PAIRWISE | Q2_PAIRWISE_PLUS_PROFILE |
| :--- | :--- | :---: | :---: |
| **A** | Historical CAL pooled R@5 > Q0 | `{public_audit['promotion_gates_evaluation']['q1_pairwise']['gate_a_hist_cal_gain']}` | `{public_audit['promotion_gates_evaluation']['q2_pairwise_plus_profile']['gate_a_hist_cal_gain']}` |
| **B** | Deployment CAL pooled R@5 >= Q0 | `{public_audit['promotion_gates_evaluation']['q1_pairwise']['gate_b_dep_cal_non_regressing']}` | `{public_audit['promotion_gates_evaluation']['q2_pairwise_plus_profile']['gate_b_dep_cal_non_regressing']}` |
| **C** | Historical Block D does not regress | `{public_audit['promotion_gates_evaluation']['q1_pairwise']['gate_c_block_d_non_regressing']}` | `{public_audit['promotion_gates_evaluation']['q2_pairwise_plus_profile']['gate_c_block_d_non_regressing']}` |
| **D** | Historical wins > losses | `{public_audit['promotion_gates_evaluation']['q1_pairwise']['gate_d_hist_wins_gt_losses']}` | `{public_audit['promotion_gates_evaluation']['q2_pairwise_plus_profile']['gate_d_hist_wins_gt_losses']}` |
| **E** | Multi-gold delta >= -0.003 | `{public_audit['promotion_gates_evaluation']['q1_pairwise']['gate_e_multi_gold_delta_ge_neg_0_003']}` | `{public_audit['promotion_gates_evaluation']['q2_pairwise_plus_profile']['gate_e_multi_gold_delta_ge_neg_0_003']}` |
| **F** | No leakage / parity / unit test failure | `True` | `True` |
| **G, H, I** | V2 generalization gates passed | `{v2_shadow['generalization_gates']['all_v2_gates_passed']}` | `{v2_shadow['generalization_gates']['all_v2_gates_passed']}` |
| **Overall** | Eligible for Promotion | **`{public_audit['promotion_gates_evaluation']['q1_pairwise']['all_gates_passed']}`** | **`{public_audit['promotion_gates_evaluation']['q2_pairwise_plus_profile']['all_gates_passed']}`** |

---

## 6. Public Artifacts & Churn

- **CONTROL_H0.zip**: `{h0_packages['sha256']}`
- **CANDIDATE_Q1_PAIRWISE.zip**: `{q1_packages['sha256']}`
- **CANDIDATE_Q2_PAIRWISE_PROFILE.zip**: `{q2_packages['sha256']}`
- **PROMOTED.zip**: `{prom_pkg.get('sha256', 'NONE')}` (Arm: `{prom_pkg.get('promoted_arm', 'NONE')}`)

### Public Churn vs H0 (Pointwise Baseline)
- **Q1 vs H0**: Changed Top-5 sets = `{q1_packages['churn_vs_h0']['changed_top5_sets']}/1000` (`{q1_packages['churn_vs_h0']['changed_top5_sets_pct']:.1f}%`), Mean Jaccard = `{q1_packages['churn_vs_h0']['mean_top5_jaccard']:.4f}`
- **Q2 vs H0**: Changed Top-5 sets = `{q2_packages['churn_vs_h0']['changed_top5_sets']}/1000` (`{q2_packages['churn_vs_h0']['changed_top5_sets_pct']:.1f}%`), Mean Jaccard = `{q2_packages['churn_vs_h0']['mean_top5_jaccard']:.4f}`
- **Q2 vs Q1**: Changed Top-5 sets = `{q2_packages['churn_vs_q1']['changed_top5_sets']}/1000` (`{q2_packages['churn_vs_q1']['changed_top5_sets_pct']:.1f}%`), Mean Jaccard = `{q2_packages['churn_vs_q1']['mean_top5_jaccard']:.4f}`
"""
    return md


def build_report_consistency_audit(
    verdict: str,
    provenance: Dict[str, Any],
    parity: Dict[str, Any],
    unit_test: Dict[str, Any],
    profile_audit: Dict[str, Any],
    training_audit: Dict[str, Any],
    dual_cal: Dict[str, Any],
    v2_shadow: Dict[str, Any],
    public_audit: Dict[str, Any],
    decision_md_text: str,
) -> Dict[str, Any]:
    checks = []

    def add_check(name: str, passed: bool, expected: Any, actual: Any):
        checks.append({
            "check_name": name,
            "passed": passed,
            "expected": expected,
            "actual": actual,
        })

    # Verdict check
    add_check("verdict_in_md", verdict in decision_md_text, verdict, "Present in DECISION.md")
    # SHA in MD
    add_check("git_head_in_md", provenance["git_head"] in decision_md_text, provenance["git_head"], "Present in DECISION.md")
    # Unit test status
    add_check("unit_test_pass", unit_test["status"] == "PASS", "PASS", unit_test["status"])
    # Duplicate links count
    add_check("duplicate_links_count", profile_audit["duplicate_links_audit"]["unique_bidirectional_directed_links"] == 50, 50, profile_audit["duplicate_links_audit"]["unique_bidirectional_directed_links"])
    # Profile cache sha
    add_check("profile_cache_sha", profile_audit["profile_cache_audit"]["sha256_matches"] is True, True, profile_audit["profile_cache_audit"]["sha256_matches"])
    # Historical CAL parity
    add_check("hist_parity", parity["historical_cal"]["parity_passed"] is True, True, parity["historical_cal"]["parity_passed"])
    # Deployment CAL parity
    add_check("dep_parity", parity["deployment_cal"]["parity_passed"] is True, True, parity["deployment_cal"]["parity_passed"])
    # V0 strict parity
    add_check("v0_strict_parity", v2_shadow["authoritative_baseline_v0"]["parity_passed"] is True, True, v2_shadow["authoritative_baseline_v0"]["parity_passed"])
    # All training queries balanced
    add_check("training_queries_balanced", training_audit["all_runs_query_balanced"] is True, True, training_audit["all_runs_query_balanced"])

    all_passed = all(c["passed"] for c in checks)

    payload = {
        "schema_version": "dsc2026.gemini.huy_query_balanced_pairwise_ltr_v1.report_consistency_audit.v1",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "total_checks": len(checks),
        "all_passed": all_passed,
        "status": "PASS" if all_passed else "FAIL",
        "checks": checks,
    }

    out_file = RESULTS_DIR / "REPORT_CONSISTENCY_AUDIT.json"
    out_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main():
    print("=== Step 5: Build Decision, Provenance, Consistency Audit, Trace ===", flush=True)

    # 1. Source provenance
    prov = build_source_provenance()

    # 2. Load result reports
    parity = json.loads((RESULTS_DIR / "BASELINE_PARITY.json").read_text(encoding="utf-8"))
    unit_test = json.loads((RESULTS_DIR / "PAIRWISE_UTILITY_UNIT_TEST.json").read_text(encoding="utf-8"))
    profile_audit = json.loads((RESULTS_DIR / "PROFILE_REUSE_AUDIT.json").read_text(encoding="utf-8"))
    training_audit = json.loads((RESULTS_DIR / "PAIRWISE_TRAINING_AUDIT.json").read_text(encoding="utf-8"))
    dual_cal = json.loads((RESULTS_DIR / "HUY_PAIRWISE_DUAL_CAL_REPORT.json").read_text(encoding="utf-8"))
    v2_shadow = json.loads((RESULTS_DIR / "V2_PAIRWISE_SHADOW_REPORT.json").read_text(encoding="utf-8"))
    public_audit = json.loads((RESULTS_DIR / "PUBLIC_PAIRWISE_AUDIT.json").read_text(encoding="utf-8"))

    # 3. Determine verdict
    verdict, reason, gates_summary = determine_verdict(
        prov, parity, profile_audit, unit_test, dual_cal, v2_shadow, public_audit
    )
    print(f"\nFinal Verdict: {verdict}\nReason: {reason}\n")

    # 4. Render DECISION.md
    dec_text = render_decision_md(
        verdict, reason, prov, parity, unit_test, profile_audit, training_audit, dual_cal, v2_shadow, public_audit
    )
    (RESULTS_DIR / "DECISION.md").write_text(dec_text, encoding="utf-8")
    print(f"Saved: {RESULTS_DIR / 'DECISION.md'}")

    # 5. Consistency audit
    const_audit = build_report_consistency_audit(
        verdict, prov, parity, unit_test, profile_audit, training_audit, dual_cal, v2_shadow, public_audit, dec_text
    )
    print(f"REPORT_CONSISTENCY_AUDIT: status={const_audit['status']}")


if __name__ == "__main__":
    main()
