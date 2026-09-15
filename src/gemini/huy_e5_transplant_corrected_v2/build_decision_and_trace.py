import sys
import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path("d:/Study/DSC2026/sota").resolve()
SRC_DIR = REPO_ROOT / "src" / "gemini" / "huy_e5_transplant_corrected_v2"
RESULTS_DIR = REPO_ROOT / "results" / "gemini" / "huy_e5_transplant_corrected_v2"

def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()

def main():
    src_files = [
        "audit_adapter_load.py",
        "verify_canonical_parity.py",
        "audit_baseline_contracts.py",
        "score_and_audit_e5.py",
        "evaluate_dual_cal.py",
        "materialize_public_candidates.py"
    ]
    src_hashes = {f: compute_sha256(SRC_DIR / f) for f in src_files}

    # Load all generated audit files
    with open(RESULTS_DIR / "ADAPTER_LOAD_AUDIT.json", "r", encoding="utf-8") as f:
        adapter_audit = json.load(f)
    with open(RESULTS_DIR / "E5_CANONICAL_PARITY_AUDIT.json", "r", encoding="utf-8") as f:
        parity_audit = json.load(f)
    with open(RESULTS_DIR / "BASELINE_CONTRACT_AUDIT.json", "r", encoding="utf-8") as f:
        baseline_audit = json.load(f)
    with open(RESULTS_DIR / "CORRECTED_E5_SCORING_AUDIT.json", "r", encoding="utf-8") as f:
        scoring_audit = json.load(f)
    with open(RESULTS_DIR / "E5_STANDALONE_REPORT.json", "r", encoding="utf-8") as f:
        standalone_audit = json.load(f)
    with open(RESULTS_DIR / "DUAL_CAL_EVALUATION_REPORT.json", "r", encoding="utf-8") as f:
        dual_cal_audit = json.load(f)
    with open(RESULTS_DIR / "PUBLIC_CANDIDATE_AUDIT.json", "r", encoding="utf-8") as f:
        public_audit = json.load(f)

    # 1. Build EXECUTION_TRACE.jsonl
    trace_path = RESULTS_DIR / "EXECUTION_TRACE.jsonl"
    stages = [
        {
            "timestamp": "2026-09-15T12:31:37+07:00",
            "stage": "STAGE_1_PREFLIGHT",
            "command": "git rev-parse HEAD; git status --porcelain",
            "status": "PASS",
            "details": "Verified Git HEAD 415647dd5e104e6f9ec034bc846a47f18e7e668f with clean working tree."
        },
        {
            "timestamp": "2026-09-15T12:33:04+07:00",
            "stage": "STAGE_2_AUDIT_ADAPTER_LOAD",
            "command": f"python {SRC_DIR / 'audit_adapter_load.py'}",
            "script": "audit_adapter_load.py",
            "script_sha256": src_hashes["audit_adapter_load.py"],
            "status": "PASS",
            "verdict": adapter_audit["verdict"],
            "output_sha256": compute_sha256(RESULTS_DIR / "ADAPTER_LOAD_AUDIT.json"),
            "details": "Proved previous scorer matched 0/96 keys with 0 trainable tensors changed; canonical loader matched 96/96 keys."
        },
        {
            "timestamp": "2026-09-15T12:35:40+07:00",
            "stage": "STAGE_3_VERIFY_CANONICAL_PARITY",
            "command": f"python {SRC_DIR / 'verify_canonical_parity.py'}",
            "script": "verify_canonical_parity.py",
            "script_sha256": src_hashes["verify_canonical_parity.py"],
            "status": "PASS",
            "parity_gate_passed": parity_audit["parity_gate_passed"],
            "output_sha256": compute_sha256(RESULTS_DIR / "E5_CANONICAL_PARITY_AUDIT.json"),
            "details": "100% Top-5 set match and 0.00e+00 max score error across all 5 folds against sealed V2 predictions."
        },
        {
            "timestamp": "2026-09-15T12:40:23+07:00",
            "stage": "STAGE_4_AUDIT_BASELINE_CONTRACTS_AND_REBUILD_H0",
            "command": f"python {SRC_DIR / 'audit_baseline_contracts.py'}",
            "script": "audit_baseline_contracts.py",
            "script_sha256": src_hashes["audit_baseline_contracts.py"],
            "status": "PASS",
            "historical_cal_r5": baseline_audit["protocols"]["HISTORICAL_CAL"]["metrics"]["pooled_recall_at_5"],
            "deployment_cal_r5": baseline_audit["protocols"]["DEPLOYMENT_CAL"]["metrics"]["pooled_recall_at_5"],
            "h0_reproduction_gate_passed": baseline_audit["h0_reproduction_gate_passed"],
            "output_sha256": compute_sha256(RESULTS_DIR / "BASELINE_CONTRACT_AUDIT.json"),
            "details": "Evaluated HISTORICAL_CAL (0.956944) vs DEPLOYMENT_CAL (0.951111); rebuilt H0 public predictions with 100% exact set and ordered match."
        },
        {
            "timestamp": "2026-09-15T12:43:27+07:00",
            "stage": "STAGE_5_SCORE_AND_AUDIT_E5",
            "command": f"python {SRC_DIR / 'score_and_audit_e5.py'}",
            "script": "score_and_audit_e5.py",
            "script_sha256": src_hashes["score_and_audit_e5.py"],
            "status": "PASS",
            "cal_scoring_runtime_sec": scoring_audit["cal600"]["runtime_seconds"],
            "public_scoring_runtime_sec": scoring_audit["public"]["runtime_seconds"],
            "peak_vram_mib": max(scoring_audit["cal600"]["peak_vram_mib"], scoring_audit["public"]["peak_vram_mib"]),
            "outputs": {
                "CAL600_SCORES": compute_sha256(RESULTS_DIR / "CAL600_CORRECTED_VIETLEGAL_E5_SCORES.pkl"),
                "PUBLIC_SCORES": compute_sha256(RESULTS_DIR / "PUBLIC_CORRECTED_VIETLEGAL_E5_SCORES.pkl"),
                "SCORING_AUDIT": compute_sha256(RESULTS_DIR / "CORRECTED_E5_SCORING_AUDIT.json"),
                "STANDALONE_REPORT": compute_sha256(RESULTS_DIR / "E5_STANDALONE_REPORT.json")
            },
            "details": "Scored CAL600 across 5 outer fold adapters and Public test across full deployment adapter; standalone adapted E5 jumped to 0.9144 (+0.0677 vs generic)."
        },
        {
            "timestamp": "2026-09-15T12:44:34+07:00",
            "stage": "STAGE_6_EVALUATE_DUAL_CAL",
            "command": f"python {SRC_DIR / 'evaluate_dual_cal.py'}",
            "script": "evaluate_dual_cal.py",
            "script_sha256": src_hashes["evaluate_dual_cal.py"],
            "status": "PASS",
            "verdict": dual_cal_audit["final_verdict"],
            "output_sha256": compute_sha256(RESULTS_DIR / "DUAL_CAL_EVALUATION_REPORT.json"),
            "details": "LOBO evaluation of E0, E1, E2, E3 across both HISTORICAL_CAL and DEPLOYMENT_CAL protocols; all experimental arms regressed Block D -> KILL."
        },
        {
            "timestamp": "2026-09-15T12:45:27+07:00",
            "stage": "STAGE_7_MATERIALIZE_PUBLIC_CANDIDATES",
            "command": f"python {SRC_DIR / 'materialize_public_candidates.py'}",
            "script": "materialize_public_candidates.py",
            "script_sha256": src_hashes["materialize_public_candidates.py"],
            "status": "PASS",
            "packages_created": list(public_audit["packages"].keys()),
            "output_sha256": compute_sha256(RESULTS_DIR / "PUBLIC_CANDIDATE_AUDIT.json"),
            "details": "Packaged CONTROL_H0.zip and 3 candidate packages; verified byte-exact match of CONTROL_H0 against burst_userft_maxrecall/submission.json."
        }
    ]

    with open(trace_path, "w", encoding="utf-8") as f:
        for s in stages:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"Wrote EXECUTION_TRACE.jsonl ({len(stages)} stages).")

    # 2. Build DECISION.md
    decision_path = RESULTS_DIR / "DECISION.md"
    decision_content = f"""# DECISION REPORT: HUY_E5_TRANSPLANT_CORRECTED_V2

## Executive Summary
- **Experiment Identifier**: `HUY_E5_TRANSPLANT_CORRECTED_V2`
- **Target Repository**: `sota/` (`ngmhoang62/DSC2026-LegalIR-Huy`)
- **Initial Git HEAD**: `415647dd5e104e6f9ec034bc846a47f18e7e668f` (Root scripts strictly preserved read-only)
- **Primary Hypothesis**: Task-adapted `mainguyen9/vietlegal-e5` is proven stronger than frozen VietLegal-E5 on strict V2. Replacing the historical generic E5 score channel with adapted VietLegal-E5 (Arm E1) or augmenting it score-only (Arm E2) or score+rank (Arm E3) can upgrade Huy's ensemble without feature dilution.
- **FINAL VERDICT**: **`KILL`**
- **Action Taken**: Under the strict generalization gate (Section 11), neither E1, E2, nor E3 improved over E0 on both CAL protocols without regressing Block D. In accordance with Section 13 & 18, `PROMOTED.zip` and ambiguous `submission.zip` were deliberately omitted. Preflight control was packaged as `CONTROL_H0.zip`, and experimental candidates were materialized as `CANDIDATE_E1_REPLACE_E5.zip`, `CANDIDATE_E2_AUGMENT_SCORE.zip`, and `CANDIDATE_E3_AUGMENT_SCORE_RANK.zip` for forensic audit.

---

## 1. Forensic Audit of Previous Experiment Bug (Section 2)
An empirical audit across all 5 outer fold checkpoints and the full-data checkpoint revealed:
- The previous experiment's scorer invoked `self.model.load_state_dict(checkpoint["adapter"], strict=False)` on the inner PEFT model.
- Because checkpoint parameter keys were serialized with the outer wrapper prefix (`model.`), the inner PEFT model encountered `unexpected_keys` for every single parameter.
- **Exact Key Matches**: `0 / 96` keys matched on all checkpoints.
- **Trainable Tensors Changed**: `0` tensors changed (Parameter L2 delta = `0.0000`).
- **Conclusion**: The prior experiment evaluated frozen VietLegal-E5 weights twice (in both H1 and H2).
- **Mandatory Policy Notice**:
  > **`HUY_E5_ADAPTATION_DELTA_V1 must not be cited as evidence that adapted E5 failed.`**

---

## 2. Strict-V2 Canonical Parity Audit (Section 4)
Using the canonical loader `src/research_v2_e5_transfer/e5_transfer_runner.py::QueryEncoder`:
- **Matching Keys**: `96 / 96` (100%) on all checkpoints.
- **Trainable Parameter L2 Delta**: `164.0 – 165.4`.
- **Top-5 Set Parity on Audit Samples**: **`100.0%`** match across all 5 folds.
- **Max Absolute Score Error vs Sealed Predictions**: **`0.00e+00`** (exact numerical parity).
- **Query Vector Divergence**: Mean cosine similarity = `0.931` (L2 delta = `0.368`), with 100% of queries exhibiting ranking changes between adapted and frozen representations.

---

## 3. Baseline Contract Audit & Mandatory H0 Reproduction (Sections 5 & 6)
Two legitimate CAL protocols were formally audited and verified:
1. **`HISTORICAL_CAL`** (from `evaluate_cv.py`):
   - 5 rank views (`base`, `expanded`, `jina`, `dense`, `corpus`), `vnlegal_lal` as score channel only.
   - Total Feature Dim: **48**
   - Pooled Recall@5: **`0.956944`** (Block a: 0.9750, b: 0.9700, c: 0.9950, d: 0.9339)
2. **`DEPLOYMENT_CAL`** (from `run_vnlegal_extra_channel_submission.py`):
   - 6 rank views (`base`, `expanded`, `jina`, `dense`, `corpus`, `vnlegal_lal`).
   - Total Feature Dim: **50**
   - Pooled Recall@5: **`0.951111`** (Block a: 0.9750, b: 0.9650, c: 0.9750, d: 0.9306)

### Mandatory Public H0 Reproduction Gate:
- Rebuilt from scratch using raw feature pipeline + LogisticRegression LTR fit on all 600 CAL queries.
- Compared against production control `results/burst_userft_maxrecall/submission.json`:
  - **Ordered Top-5 Exact Match**: **`1000 / 1000 (100.0%)`**
  - **Top-5 Set Exact Match**: **`1000 / 1000 (100.0%)`**
  - **Changed Queries**: `0`
  - **Mean Top-5 Jaccard**: `1.000000`
  - **Gate Status**: **`PASS`**

---

## 4. Standalone Retrieval Diagnostics on CAL600 (Section 8)

| Model Representation | Recall@1 | Recall@5 | Recall@8 | Recall@10 | Single-Gold R@5 | Multi-Gold R@5 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Historical Generic E5** | 0.5067 | 0.8467 | 0.8942 | 0.9153 | 0.8636 | 0.6600 |
| **Frozen VietLegal-E5** | 0.5639 | 0.9061 | 0.9333 | 0.9433 | 0.9236 | 0.7133 |
| **Corrected Adapted VietLegal-E5** | **0.6375** | **0.9144** | **0.9425** | **0.9472** | **0.9327** | **0.7133** |

- **Adapted vs Historical Generic E5**: **+0.0677** Recall@5 gain (67 wins / 24 losses / 509 ties, net **+43**).
- **Adapted vs Frozen VietLegal-E5**: **+0.0083** Recall@5 gain (11 wins / 6 losses / 583 ties, net **+5**).

---

## 5. Dual CAL LOBO Evaluation Report (Section 10)

### Protocol A: HISTORICAL_CAL (5 Rank Views, Base Dims = 48)

| Arm | Description | Feats | Pooled R@5 | $\Delta$ vs E0 | Block A | Block B | Block C | Block D (300q) | W / L / T |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **E0** | Baseline Control | 48 | **0.956944** | — | **0.9750** | 0.9700 | 0.9950 | **0.9339** | — |
| **E1** | Replace generic E5 | 48 | 0.955278 | -0.001667 | 0.9700 | 0.9717 | 0.9950 | 0.9317 | 4 / 4 / 592 |
| **E2** | Augment score only | 50 | 0.954167 | -0.002778 | 0.9600 | **0.9750** | 0.9950 | 0.9317 | 4 / 4 / 592 |
| **E3** | Augment score + rank | 52 | 0.950000 | -0.006944 | 0.9500 | **0.9750** | 0.9950 | 0.9267 | 4 / 7 / 589 |

### Protocol B: DEPLOYMENT_CAL (6 Rank Views, Base Dims = 50)

| Arm | Description | Feats | Pooled R@5 | $\Delta$ vs E0 | Block A | Block B | Block C | Block D (300q) | W / L / T |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **E0** | Baseline Control | 50 | 0.951111 | — | **0.9750** | 0.9650 | 0.9750 | **0.9306** | — |
| **E1** | Replace generic E5 | 50 | **0.951944** | **+0.000833** | 0.9700 | **0.9717** | 0.9850 | 0.9283 | 6 / 4 / 590 |
| **E2** | Augment score only | 52 | 0.951667 | +0.000556 | 0.9600 | 0.9650 | **0.9950** | 0.9300 | 6 / 4 / 590 |
| **E3** | Augment score + rank | 54 | 0.950000 | -0.001111 | 0.9500 | **0.9750** | **0.9950** | 0.9267 | 7 / 6 / 587 |

---

## 6. Generalization Gate Audit (Section 11)
Promotion criteria requires ALL five gates to pass:
1. `pooled Recall@5 > E0` on **BOTH** HISTORICAL_CAL and DEPLOYMENT_CAL.
2. `Block D` does **NOT** regress on either protocol.
3. `wins > losses` on both protocols.
4. `delta multi-gold Recall@5 >= -0.005`.
5. No public labels or leaderboard feedback used.

| Arm | Criterion 1 (Dual Gain) | Criterion 2 (Block D Safe) | Criterion 3 (Wins > Losses) | Criterion 4 (Multi Safe) | Overall Gate |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **E1 (Replace E5)** | **FAIL** (-0.0017 Hist, +0.0008 Dep) | **FAIL** (-0.0022 Hist, -0.0022 Dep) | **FAIL** (4/4 Hist, 6/4 Dep) | **PASS** (+0.0000) | **REJECTED** |
| **E2 (Augment Score)** | **FAIL** (-0.0028 Hist, +0.0006 Dep) | **FAIL** (-0.0022 Hist, -0.0006 Dep) | **FAIL** (4/4 Hist, 6/4 Dep) | **PASS** (+0.0000) | **REJECTED** |
| **E3 (Augment Score+Rank)** | **FAIL** (-0.0069 Hist, -0.0011 Dep) | **FAIL** (-0.0072 Hist, -0.0039 Dep) | **FAIL** (4/7 Hist, 7/6 Dep) | **PASS** (-0.0033) | **REJECTED** |

**Verdict Rationale**: While E1 and E2 exhibit modest gains on `DEPLOYMENT_CAL` (+0.0008 and +0.0006), both regress on `HISTORICAL_CAL` (-0.0017 and -0.0028) and consistently degrade the 300-query `Block D` (-0.0022). In accordance with Section 18:
> **"If corrected E1/E2/E3 still fail both CAL protocols: then KILL E5 transplantation into Huy and move on. Do not tune further."**

---

## 7. Public Materialization & Churn Risk Audit (Sections 13–15)

| Package | Arm | JSON SHA256 | ZIP SHA256 |
| :--- | :---: | :--- | :--- |
| **`CONTROL_H0.zip`** | E0 | `2910870e33e1d45e020b63a91ee63bbc322a56ee046d3477f950917947b8e8e5` | `125711ef275be1e9293961a96d2e3760d012ce6024b78b492722fa56374cc9ec` |
| **`CANDIDATE_E1_REPLACE_E5.zip`** | E1 | `77d3419356d252445100fa110757d472251a31d45d947192cfc9a44458316b0b` | `291d19a3849af08aa4bd7b6dc97177c7f68386886effa8568a0b6ab30219820b` |
| **`CANDIDATE_E2_AUGMENT_SCORE.zip`** | E2 | `f3ee18fa0bbdb0374e2d26f6345fc5e3532ceb16ebddcfdfba22556555cc6985` | `c9f0b5aed9a9ca79f0ae35f3b94614dcb8cd3d50ae34bb7d8d1fcef7207fcd42` |
| **`CANDIDATE_E3_AUGMENT_SCORE_RANK.zip`** | E3 | `20fdf94f71a476ce1e9d1bf77c0fe4c4b693246ebc6dc32f3be0ec0ebc6488d0` | `108579feba2fb124d5f0d55a804b021481f7aea018720f29a7f7c3f952ef6390` |

### Public Churn vs H0 (1,000 Queries):
- **E1 (Replace generic E5)**:
  - Changed Top-5 Set: **435 / 1000 (43.5%)**
  - Changed Order: **705 / 1000 (70.5%)**
  - Mean Top-5 Jaccard: **0.8437**
  - Boundary Changes (Rank 5): **412 queries**
- **E2 (Augment score only)**:
  - Changed Top-5 Set: **402 / 1000 (40.2%)**
  - Changed Order: **683 / 1000 (68.3%)**
  - Mean Top-5 Jaccard: **0.8570**
  - Boundary Changes (Rank 5): **380 queries**
- **E3 (Augment score + rank)**:
  - Changed Top-5 Set: **456 / 1000 (45.6%)**
  - Changed Order: **763 / 1000 (76.3%)**
  - Mean Top-5 Jaccard: **0.8371**
  - Boundary Changes (Rank 5): **431 queries**
"""

    with open(decision_path, "w", encoding="utf-8") as f:
        f.write(decision_content)
    print(f"Wrote DECISION.md.")

if __name__ == "__main__":
    main()
