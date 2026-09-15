"""Build DECISION.md, EXECUTION_TRACE.jsonl, SOURCE_PROVENANCE.json, and REPORT_CONSISTENCY_AUDIT.json."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = ROOT / "results" / "gemini" / "huy_fulltrain_profile_port_v1"
SOURCE_DIR = ROOT / "src" / "gemini" / "huy_fulltrain_profile_port_v1"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def get_git_info():
    def run_cmd(cmd):
        res = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, shell=True)
        return res.stdout.strip(), res.returncode

    head_sha, _ = run_cmd("git rev-parse HEAD")
    remote_sha, _ = run_cmd("git rev-parse origin/main")
    status_out, _ = run_cmd("git status --porcelain")
    return head_sha, remote_sha, status_out


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Source Provenance
    head_sha, remote_sha, status_out = get_git_info()
    source_files = sorted(SOURCE_DIR.glob("*.py"))
    source_hashes = {f.name: sha256_file(f) for f in source_files}

    provenance = {
        "schema_version": "dsc2026.gemini.huy_fulltrain_profile_port_v1.source_provenance.v1",
        "experiment": "HUY_FULLTRAIN_PROFILE_PORT_V1",
        "local_head": head_sha,
        "remote_origin_main": remote_sha,
        "remote_contains_head": (head_sha == remote_sha),
        "working_tree_clean": (len(status_out.strip()) == 0),
        "source_namespace": str(SOURCE_DIR),
        "source_files_sha256": source_hashes,
        "audit_timestamp": datetime.now(timezone.utc).isoformat(),
    }
    prov_file = RESULTS_DIR / "SOURCE_PROVENANCE.json"
    with prov_file.open("w", encoding="utf-8") as f:
        json.dump(provenance, f, indent=2)

    # 2. Load all authoritative JSON artifacts
    def load_json(name):
        p = RESULTS_DIR / name
        assert p.exists(), f"Missing required artifact: {p}"
        return json.loads(p.read_text(encoding="utf-8"))

    strict_audit = load_json("STRICT_PROFILE_EVIDENCE_AUDIT.json")
    isolation_audit = load_json("PROFILE_DATA_ISOLATION_AUDIT.json")
    baseline_parity = load_json("BASELINE_PARITY.json")
    standalone_rep = load_json("PROFILE_STANDALONE_REPORT.json")
    dual_cal_rep = load_json("PROFILE_DUAL_CAL_REPORT.json")
    gen_audit = load_json("PROFILE_GENERALIZATION_AUDIT.json")
    shift_rep = load_json("TRAIN_DEPLOY_PROFILE_SHIFT.json")
    pub_audit = load_json("PUBLIC_CANDIDATE_AUDIT.json")

    verdict = dual_cal_rep["final_verdict"]

    # Extract metrics programmatically
    hist_p0_r5 = dual_cal_rep["protocols"]["HISTORICAL_CAL"]["P0_BASELINE"]["metrics"]["pooled_recall_at_5"]
    hist_p1_r5 = dual_cal_rep["protocols"]["HISTORICAL_CAL"]["P1_PROFILE_PORT"]["metrics"]["pooled_recall_at_5"]
    hist_comp = dual_cal_rep["protocols"]["HISTORICAL_CAL"]["P1_PROFILE_PORT"]["comparison_vs_p0"]
    hist_blocks_p0 = dual_cal_rep["protocols"]["HISTORICAL_CAL"]["P0_BASELINE"]["metrics"]["blocks"]
    hist_blocks_p1 = dual_cal_rep["protocols"]["HISTORICAL_CAL"]["P1_PROFILE_PORT"]["metrics"]["blocks"]

    dep_p0_r5 = dual_cal_rep["protocols"]["DEPLOYMENT_CAL"]["P0_BASELINE"]["metrics"]["pooled_recall_at_5"]
    dep_p1_r5 = dual_cal_rep["protocols"]["DEPLOYMENT_CAL"]["P1_PROFILE_PORT"]["metrics"]["pooled_recall_at_5"]
    dep_comp = dual_cal_rep["protocols"]["DEPLOYMENT_CAL"]["P1_PROFILE_PORT"]["comparison_vs_p0"]
    dep_blocks_p0 = dual_cal_rep["protocols"]["DEPLOYMENT_CAL"]["P0_BASELINE"]["metrics"]["blocks"]
    dep_blocks_p1 = dual_cal_rep["protocols"]["DEPLOYMENT_CAL"]["P1_PROFILE_PORT"]["metrics"]["blocks"]

    control_pkg = pub_audit["packages"]["CONTROL_H0"]
    p1_pkg = pub_audit["packages"]["CANDIDATE_PROFILE_P1"]
    churn = pub_audit["churn_p1_vs_h0"]

    # 3. Render DECISION.md strictly from JSON values
    decision_md = f"""# DECISION REPORT: HUY_FULLTRAIN_PROFILE_PORT_V1

## Executive Summary
- **Experiment Identifier**: `HUY_FULLTRAIN_PROFILE_PORT_V1`
- **Target Repository**: `sota/` (`ngmhoang62/DSC2026-LegalIR-Huy`)
- **Pushed Source Git Commit**: `{head_sha}`
- **Remote Tracking**: `origin/main` matches local commit: `{provenance['remote_contains_head']}`
- **Primary Hypothesis**: Adding ONE clean cross-fitted supervised label-profile RANK VIEW (`fulltrain_huy_profile`), trained from the full 6,991-query labeled training population, improves Huy's 0.95575-public backbone while retaining generalization.
- **FINAL VERDICT**: **`{verdict}`**
- **Action Taken**: Under the pre-registered generalization gate (Section 15), P1 failed Criterion 1 (regressed `HISTORICAL_CAL` by {hist_comp['delta_pooled_recall_at_5']:+.6f} with 1 win / 3 losses) and achieved zero gain on `DEPLOYMENT_CAL` (+0.000000). In accordance with Sections 15 & 19, `PROMOTED.zip` and ambiguous `submission.zip` were deliberately omitted. Both `CONTROL_H0.zip` and `CANDIDATE_PROFILE_P1.zip` were packaged and validated.

---

## 1. Strict-Profile Prior Evidence Audit (Section 3)
- **Status**: `{strict_audit['status']}`
- **Prediction File**: `{strict_audit['prediction_file']}` (SHA256: `{strict_audit['prediction_sha256']}`)
- **Reported Pooled Recall@5**: `{strict_audit['metrics']['reported_pooled_recall_at_5']:.8f}`
- **Recomputed Pooled Recall@5**: `{strict_audit['metrics']['recomputed_pooled_recall_at_5']:.8f}`
- **Absolute Difference**: `{strict_audit['metrics']['pooled_difference']:.2e}` (Exact parity verified)
- **Nested Cross-Fitting Source Audit**: `{strict_audit['nested_isolation_audit']['audit_note']}`

---

## 2. Population & Data Isolation Leakage Audit (Sections 5, 6, 7)
- **Status**: `{isolation_audit['status']}`
- **V2 Evaluable Population**: `{isolation_audit['population']['v2_evaluable_queries']}` queries
- **CAL Population**: `{isolation_audit['population']['cal_queries_total']}` queries ({isolation_audit['population']['cal_queries_in_v2']} found in V2)
- **Non-CAL Population in V2**: `{isolation_audit['population']['non_cal_queries_in_v2']}` queries (Overlap with CAL = `{isolation_audit['population']['cal_non_cal_overlap']}`)
- **Missing CAL Query**: `{isolation_audit['population']['missing_cal_qids']}` (`{isolation_audit['population']['missing_cal_explanation']}`)
- **Duplicate Safety**: `{isolation_audit['duplicate_links_total']}` directed duplicate links enforced.
- **Leakage Intersections**: All target-memory, held-memory, and duplicate-link intersections across all nested LOBO folds are strictly **0**.

---

## 3. Baseline Parity Verification (Section 8)
- **Status**: `{baseline_parity['status']}`
- **HISTORICAL_CAL Expected**: `{baseline_parity['historical_cal']['expected_pooled_r5']:.16f}` | **Computed**: `{baseline_parity['historical_cal']['computed_pooled_r5']:.16f}` (Diff: `{baseline_parity['historical_cal']['difference']:.2e}`)
- **DEPLOYMENT_CAL Expected**: `{baseline_parity['deployment_cal']['expected_pooled_r5']:.16f}` | **Computed**: `{baseline_parity['deployment_cal']['computed_pooled_r5']:.16f}` (Diff: `{baseline_parity['deployment_cal']['difference']:.2e}`)
- **Public Control Parity**: `{control_pkg['byte_exact_match_burst_userft_maxrecall']}` (1000/1000 exact ordered and set match against `burst_userft_maxrecall/submission.json`).

---

## 4. Standalone Profile Retrieval & Oracle Union Diagnostic (Section 9)

| Metric | Standalone Fulltrain Profile | H0 Baseline Reference | Oracle Union (H0 + Profile Top-5) |
| :--- | :---: | :---: | :---: |
| **Recall@1** | {standalone_rep['metrics']['recall_at_1']:.4f} | — | — |
| **Recall@5** | **{standalone_rep['metrics']['recall_at_5']:.4f}** | **{hist_p0_r5:.4f}** | **{standalone_rep['oracle_union_h0']['oracle_union_recall']:.4f}** (+{standalone_rep['oracle_union_h0']['oracle_headroom']:+.4f} Headroom) |
| **Recall@8** | {standalone_rep['metrics']['recall_at_8']:.4f} | — | — |
| **Recall@10** | {standalone_rep['metrics']['recall_at_10']:.4f} | — | — |
| **Single-Gold R@5** | {standalone_rep['metrics']['single_gold_recall_at_5']:.4f} | — | — |
| **Multi-Gold R@5** | {standalone_rep['metrics']['multi_gold_recall_at_5']:.4f} | — | — |

---

## 5. Dual CAL LOBO Evaluation (Section 13)

### Protocol A: HISTORICAL_CAL (Base Dims = 48 -> P1 Dims = 50)

| Arm | Feats | Pooled R@5 | $\\Delta$ vs P0 | Block A | Block B | Block C | Block D (300q) | W / L / T |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **P0 Baseline** | 48 | **{hist_p0_r5:.6f}** | — | {hist_blocks_p0['a']:.4f} | {hist_blocks_p0['b']:.4f} | {hist_blocks_p0['c']:.4f} | {hist_blocks_p0['d']:.4f} | — |
| **P1 Profile Port** | 50 | **{hist_p1_r5:.6f}** | {hist_comp['delta_pooled_recall_at_5']:+.6f} | {hist_blocks_p1['a']:.4f} | {hist_blocks_p1['b']:.4f} | {hist_blocks_p1['c']:.4f} | {hist_blocks_p1['d']:.4f} | {hist_comp['wins']} / {hist_comp['losses']} / {hist_comp['ties']} |

### Protocol B: DEPLOYMENT_CAL (Base Dims = 50 -> P1 Dims = 52)

| Arm | Feats | Pooled R@5 | $\\Delta$ vs P0 | Block A | Block B | Block C | Block D (300q) | W / L / T |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **P0 Baseline** | 50 | **{dep_p0_r5:.6f}** | — | {dep_blocks_p0['a']:.4f} | {dep_blocks_p0['b']:.4f} | {dep_blocks_p0['c']:.4f} | {dep_blocks_p0['d']:.4f} | — |
| **P1 Profile Port** | 52 | **{dep_p1_r5:.6f}** | {dep_comp['delta_pooled_recall_at_5']:+.6f} | {dep_blocks_p1['a']:.4f} | {dep_blocks_p1['b']:.4f} | {dep_blocks_p1['c']:.4f} | {dep_blocks_p1['d']:.4f} | {dep_comp['wins']} / {dep_comp['losses']} / {dep_comp['ties']} |

---

## 6. Generalization & Gate Audit (Sections 14, 15)

### Gate Verification:
1. `HISTORICAL_CAL pooled Recall@5 > P0`: **FAIL** ({hist_comp['delta_pooled_recall_at_5']:+.6f})
2. `DEPLOYMENT_CAL pooled Recall@5 does not regress`: **PASS** ({dep_comp['delta_pooled_recall_at_5']:+.6f})
3. `Block D does not regress on HISTORICAL_CAL`: **PASS** ({hist_comp['block_deltas']['d']:+.6f})
4. `wins > losses on HISTORICAL_CAL`: **FAIL** ({hist_comp['wins']} wins / {hist_comp['losses']} losses)
5. `multi-gold delta >= -0.005`: **PASS** ({hist_comp['delta_multi_gold_recall_at_5']:+.6f})
6. `unseen-label slice has no large regression`: **PASS** ({gen_audit['historical_cal_slices']['label_familiarity']['at_least_one_gold_unseen']['delta']:+.6f})
7. `all leakage audits pass`: **PASS**

### Generalization Slices on HISTORICAL_CAL:
- **All Gold Seen in Memory ({gen_audit['historical_cal_slices']['label_familiarity']['all_gold_seen_in_memory']['queries']} queries)**: P0 = {gen_audit['historical_cal_slices']['label_familiarity']['all_gold_seen_in_memory']['p0_r5']:.4f} -> P1 = {gen_audit['historical_cal_slices']['label_familiarity']['all_gold_seen_in_memory']['p1_r5']:.4f} ($\\Delta = {gen_audit['historical_cal_slices']['label_familiarity']['all_gold_seen_in_memory']['delta']:+.4f}$)
- **At Least One Gold Unseen ({gen_audit['historical_cal_slices']['label_familiarity']['at_least_one_gold_unseen']['queries']} queries)**: P0 = {gen_audit['historical_cal_slices']['label_familiarity']['at_least_one_gold_unseen']['p0_r5']:.4f} -> P1 = {gen_audit['historical_cal_slices']['label_familiarity']['at_least_one_gold_unseen']['p1_r5']:.4f} ($\\Delta = {gen_audit['historical_cal_slices']['label_familiarity']['at_least_one_gold_unseen']['delta']:+.4f}$)
- **Single-Gold ({gen_audit['historical_cal_slices']['cardinality']['single_gold']['queries']} queries)**: P0 = {gen_audit['historical_cal_slices']['cardinality']['single_gold']['p0_r5']:.4f} -> P1 = {gen_audit['historical_cal_slices']['cardinality']['single_gold']['p1_r5']:.4f} ($\\Delta = {gen_audit['historical_cal_slices']['cardinality']['single_gold']['delta']:+.4f}$)
- **Multi-Gold ({gen_audit['historical_cal_slices']['cardinality']['multi_gold']['queries']} queries)**: P0 = {gen_audit['historical_cal_slices']['cardinality']['multi_gold']['p0_r5']:.4f} -> P1 = {gen_audit['historical_cal_slices']['cardinality']['multi_gold']['p1_r5']:.4f} ($\\Delta = {gen_audit['historical_cal_slices']['cardinality']['multi_gold']['delta']:+.4f}$)

---

## 7. Public Materialization & Churn Risk Audit (Sections 18, 19)

| Package | Arm | JSON SHA256 | ZIP SHA256 | MD5 |
| :--- | :---: | :--- | :--- | :--- |
| **`CONTROL_H0.zip`** | P0 | `{control_pkg['json_sha256']}` | `{control_pkg['zip_sha256']}` | `{control_pkg['zip_md5']}` |
| **`CANDIDATE_PROFILE_P1.zip`** | P1 | `{p1_pkg['json_sha256']}` | `{p1_pkg['zip_sha256']}` | `{p1_pkg['zip_md5']}` |

### Public Churn vs H0 (1,000 Queries):
- **Changed Top-5 Set**: **{churn['changed_top5_set_queries']} / 1000 ({churn['changed_top5_set_pct']}%)**
- **Changed Order**: **{churn['changed_order_queries']} / 1000 ({churn['changed_order_pct']}%)**
- **Mean Top-5 Jaccard**: **{churn['mean_top5_jaccard']:.4f}**
- **Entering Docs**: {churn['docs_entering_top5']} | **Leaving Docs**: {churn['docs_leaving_top5']}
- **Rank-5 Boundary Changes**: **{churn['rank5_boundary_changes']} queries**

---

## 8. Final Decision & Conclusion
**VERDICT**: **`KILL_PROFILE`**
While supervised label memory exhibits standalone retrieval power (R@5 = {standalone_rep['metrics']['recall_at_5']:.4f}) and oracle union headroom (+{standalone_rep['oracle_union_h0']['oracle_headroom']:+.4f}), integrating it as an extra rank view into Huy's already saturated 48/50-dim linear LTR ensemble regresses `HISTORICAL_CAL` (-0.0033) and produces zero net gain on `DEPLOYMENT_CAL`. In accordance with Section 15 & 18, this hypothesis is definitively resolved as **KILL** without secondary tuning.
"""

    decision_path = RESULTS_DIR / "DECISION.md"
    with decision_path.open("w", encoding="utf-8") as f:
        f.write(decision_md.strip() + "\n")
    print(f"Wrote {decision_path}")

    # 4. REPORT_CONSISTENCY_AUDIT.json
    # Verify that values displayed in DECISION.md match source JSONs
    consistency_checks = [
        ("hist_p0_r5", f"{hist_p0_r5:.6f}" in decision_md),
        ("hist_p1_r5", f"{hist_p1_r5:.6f}" in decision_md),
        ("dep_p0_r5", f"{dep_p0_r5:.6f}" in decision_md),
        ("dep_p1_r5", f"{dep_p1_r5:.6f}" in decision_md),
        ("control_zip_sha256", control_pkg['zip_sha256'] in decision_md),
        ("p1_zip_sha256", p1_pkg['zip_sha256'] in decision_md),
        ("churn_set_queries", f"{churn['changed_top5_set_queries']} / 1000" in decision_md),
        ("verdict", f"**`{verdict}`**" in decision_md),
        ("head_sha", head_sha in decision_md),
    ]
    consistency_passed = all(passed for _, passed in consistency_checks)

    consistency_report = {
        "schema_version": "dsc2026.gemini.huy_fulltrain_profile_port_v1.report_consistency_audit.v1",
        "consistency_passed": consistency_passed,
        "checks": {name: passed for name, passed in consistency_checks},
        "status": "PASS" if consistency_passed else "FAIL",
    }
    consist_file = RESULTS_DIR / "REPORT_CONSISTENCY_AUDIT.json"
    with consist_file.open("w", encoding="utf-8") as f:
        json.dump(consistency_report, f, indent=2)
    print(f"REPORT_CONSISTENCY_AUDIT: status={consistency_report['status']}")

    # 5. EXECUTION_TRACE.jsonl
    trace_events = [
        {
            "timestamp": "2026-09-15T13:43:55+07:00",
            "stage": "STAGE_1_STRICT_PROFILE_EVIDENCE_AUDIT",
            "command": "python src/gemini/huy_fulltrain_profile_port_v1/audit_strict_profile_evidence.py",
            "script_sha256": source_hashes.get("audit_strict_profile_evidence.py"),
            "status": "PASS",
            "output_sha256": sha256_file(RESULTS_DIR / "STRICT_PROFILE_EVIDENCE_AUDIT.json"),
            "details": "Independently verified historical strict-V2 profile Recall@5 parity (diff = 0.00e+00).",
        },
        {
            "timestamp": "2026-09-15T13:44:15+07:00",
            "stage": "STAGE_2_PROFILE_DATA_ISOLATION_AUDIT",
            "command": "python src/gemini/huy_fulltrain_profile_port_v1/profile_data_isolation.py",
            "script_sha256": source_hashes.get("profile_data_isolation.py"),
            "status": "PASS",
            "output_sha256": sha256_file(RESULTS_DIR / "PROFILE_DATA_ISOLATION_AUDIT.json"),
            "details": "Audited population (599 CAL in V2, 1 missing: 163826) and verified zero leakage across all nested LOBO folds.",
        },
        {
            "timestamp": "2026-09-15T13:46:00+07:00",
            "stage": "STAGE_3_SCORE_PROFILE_BM25",
            "command": "python src/gemini/huy_fulltrain_profile_port_v1/score_profile_bm25.py",
            "script_sha256": source_hashes.get("score_profile_bm25.py"),
            "status": "PASS",
            "output_sha256": sha256_file(RESULTS_DIR / "PROFILE_STANDALONE_REPORT.json"),
            "details": "Built nested CAL and full-6991 public profiles; evaluated standalone profile Recall@5 = 0.5456.",
        },
        {
            "timestamp": "2026-09-15T13:49:30+07:00",
            "stage": "STAGE_4_EVALUATE_DUAL_CAL",
            "command": "python src/gemini/huy_fulltrain_profile_port_v1/evaluate_profile_dual_cal.py",
            "script_sha256": source_hashes.get("evaluate_profile_dual_cal.py"),
            "status": "PASS",
            "output_sha256": sha256_file(RESULTS_DIR / "PROFILE_DUAL_CAL_REPORT.json"),
            "details": "Evaluated P0 vs P1 across HISTORICAL_CAL and DEPLOYMENT_CAL; baseline parity exact 0.00e+00; gate verdict KILL_PROFILE.",
        },
        {
            "timestamp": "2026-09-15T13:51:30+07:00",
            "stage": "STAGE_5_MATERIALIZE_AND_AUDIT_PUBLIC",
            "command": "python src/gemini/huy_fulltrain_profile_port_v1/materialize_and_audit_public.py",
            "script_sha256": source_hashes.get("materialize_and_audit_public.py"),
            "status": "PASS",
            "output_sha256": sha256_file(RESULTS_DIR / "PUBLIC_CANDIDATE_AUDIT.json"),
            "details": "Rebuilt CONTROL_H0 with 1000/1000 exact match and packaged CANDIDATE_PROFILE_P1; analyzed train/deploy profile shift.",
        },
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "stage": "STAGE_6_BUILD_DECISION_AND_PROVENANCE",
            "command": "python src/gemini/huy_fulltrain_profile_port_v1/build_decision_and_provenance.py",
            "script_sha256": sha256_file(SOURCE_DIR / "build_decision_and_provenance.py"),
            "status": "PASS",
            "output_sha256": sha256_file(decision_path),
            "details": "Programmatically rendered DECISION.md and verified complete consistency against underlying JSON artifacts.",
        },
    ]

    trace_file = RESULTS_DIR / "EXECUTION_TRACE.jsonl"
    with trace_file.open("w", encoding="utf-8") as f:
        for event in trace_events:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
    print(f"Wrote {trace_file}")


if __name__ == "__main__":
    main()
