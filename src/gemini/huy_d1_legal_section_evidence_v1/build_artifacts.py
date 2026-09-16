"""Generate authoritative report JSON, predictions JSONL, section diagnostics JSON, and DECISION.md."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path("D:/Study/DSC2026/sota")
RES_DIR = ROOT / "results/gemini/huy_d1_legal_section_evidence_v1"
SRC_DIR = ROOT / "src/gemini/huy_d1_legal_section_evidence_v1"

from src.gemini.huy_d1_legal_section_evidence_v1.common import sha256_file
from src.gemini.huy_d1_legal_section_evidence_v1.legal_section_parser import (
    parse_document_into_sections,
    preselect_legal_sections,
)


def get_git_info() -> Dict[str, Any]:
    try:
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(ROOT), text=True
        ).strip()
        origin = subprocess.check_output(
            ["git", "rev-parse", "origin/main"], cwd=str(ROOT), text=True
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=str(ROOT), text=True
        ).strip()
        return {
            "head_commit": head,
            "origin_main_commit": origin,
            "parity": head == origin,
            "status_clean": len(status) == 0,
        }
    except Exception as e:
        return {"error": str(e), "parity": False}


def build_all_artifacts(
    eval_report: dict,
    preds_s0: dict,
    preds_s1: dict,
    scores_s0: dict,
    scores_s1: dict,
    model_provenance: dict,
    scoring_meta: dict,
    cal_data: tuple,
) -> dict:
    RES_DIR.mkdir(parents=True, exist_ok=True)
    git_info = get_git_info()
    print("=== BUILDING AUTHORITATIVE ARTIFACTS ===", flush=True)

    (
        docs,
        queries,
        blocks,
        all_ids,
        extended,
        local_views,
        full_channels_cv,
        gold,
        type_rows,
        cite_rows,
    ) = cal_data

    # 1. Collect source file hashes
    source_files_meta = {}
    for p in sorted(SRC_DIR.glob("*.py")):
        source_files_meta[p.name] = {
            "sha256": sha256_file(p),
            "size_bytes": p.stat().st_size,
        }

    # 2. Write S0_S1_CAL_PREDICTIONS.jsonl
    pred_path = RES_DIR / "S0_S1_CAL_PREDICTIONS.jsonl"
    with open(pred_path, "w", encoding="utf-8") as f:
        for q in all_ids:
            q_gold = sorted(list(gold[q]))
            t5_s0 = preds_s0[q]
            t5_s1 = preds_s1[q]
            rec_s0 = len(set(t5_s0) & set(q_gold)) / max(1, len(q_gold))
            rec_s1 = len(set(t5_s1) & set(q_gold)) / max(1, len(q_gold))
            entry = {
                "qid": q,
                "question": queries[q][0],
                "gold_docs": q_gold,
                "s0_top5": t5_s0,
                "s1_top5": t5_s1,
                "s0_recall_at_5": rec_s0,
                "s1_recall_at_5": rec_s1,
                "is_changed": t5_s0 != t5_s1,
                "delta_recall": rec_s1 - rec_s0,
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"Wrote {pred_path}", flush=True)

    # 3. Build SECTION_EVIDENCE_DIAGNOSTICS.json
    diag_path = RES_DIR / "SECTION_EVIDENCE_DIAGNOSTICS.json"
    changed_qids = eval_report.get("changed_queries", [])
    changed_diagnostics = []

    for q in changed_qids:
        q_text = queries[q][0]
        q_gold = set(gold[q])
        cand_list = extended[q]
        s0_ranks = {d: r + 1 for r, d in enumerate(preds_s0[q])}
        s1_ranks = {d: r + 1 for r, d in enumerate(preds_s1[q])}

        # Candidate pool ranking
        doc_details = []
        for d in cand_list:
            d_text = docs[d]
            sections = parse_document_into_sections(d, d_text)
            selected_secs = preselect_legal_sections(q_text, sections, count=2)

            in_s0_top5 = d in preds_s0[q]
            in_s1_top5 = d in preds_s1[q]

            if in_s0_top5 != in_s1_top5 or d in q_gold:
                doc_details.append({
                    "doc_id": d,
                    "is_gold": d in q_gold,
                    "s0_top5_rank": s0_ranks.get(d),
                    "s1_top5_rank": s1_ranks.get(d),
                    "selected_sections": [
                        {
                            "heading": s.heading,
                            "section_type": s.section_type,
                            "excerpt": s.text[:250] + "..." if len(s.text) > 250 else s.text,
                        }
                        for s in selected_secs
                    ],
                })

        changed_diagnostics.append({
            "qid": q,
            "question": q_text,
            "gold_doc_ids": sorted(list(q_gold)),
            "s0_top5": preds_s0[q],
            "s1_top5": preds_s1[q],
            "s0_recall_at_5": len(set(preds_s0[q]) & q_gold) / max(1, len(q_gold)),
            "s1_recall_at_5": len(set(preds_s1[q]) & q_gold) / max(1, len(q_gold)),
            "affected_documents": doc_details,
        })

    # Post-hoc audit against prior 33 CAL error queries
    prior_forensics_path = ROOT / "results/gemini/d1_error_forensics/D1_ERROR_FORENSICS.json"
    posthoc_analysis = {}
    if prior_forensics_path.exists():
        prior_data = json.loads(prior_forensics_path.read_text(encoding="utf-8"))
        prior_error_qids = [eq["qid"] for eq in prior_data.get("error_queries", [])]
        rescued_queries = []
        damaged_queries = []
        for eq_id in prior_error_qids:
            r0 = len(set(preds_s0[eq_id]) & gold[eq_id]) / max(1, len(gold[eq_id]))
            r1 = len(set(preds_s1[eq_id]) & gold[eq_id]) / max(1, len(gold[eq_id]))
            if r1 > r0:
                rescued_queries.append({"qid": eq_id, "s0_r5": r0, "s1_r5": r1, "gain": r1 - r0})
            elif r1 < r0:
                damaged_queries.append({"qid": eq_id, "s0_r5": r0, "s1_r5": r1, "loss": r0 - r1})

        # Check currently-correct queries damaged
        currently_correct_damaged = []
        for q in all_ids:
            if q not in prior_error_qids:
                r0 = len(set(preds_s0[q]) & gold[q]) / max(1, len(gold[q]))
                r1 = len(set(preds_s1[q]) & gold[q]) / max(1, len(gold[q]))
                if r1 < r0:
                    currently_correct_damaged.append({"qid": q, "s0_r5": r0, "s1_r5": r1})

        posthoc_analysis = {
            "total_prior_error_queries": len(prior_error_qids),
            "rescued_prior_error_queries_count": len(rescued_queries),
            "rescued_queries": rescued_queries,
            "damaged_prior_error_queries_count": len(damaged_queries),
            "damaged_queries": damaged_queries,
            "currently_correct_queries_damaged_count": len(currently_correct_damaged),
            "currently_correct_damaged_queries": currently_correct_damaged,
        }

    diag_payload = {
        "schema_version": "dsc2026.gemini.huy_d1_legal_section_evidence_v1.diagnostics.v1",
        "experiment_id": "HUY_D1_LEGAL_SECTION_EVIDENCE_V1",
        "total_changed_queries": len(changed_diagnostics),
        "posthoc_cal_error_analysis": posthoc_analysis,
        "changed_queries_details": changed_diagnostics,
    }
    with open(diag_path, "w", encoding="utf-8") as f:
        json.dump(diag_payload, f, indent=2, ensure_ascii=False)
    print(f"Wrote {diag_path}", flush=True)

    # 4. Authoritative report JSON
    report_path = RES_DIR / "HUY_D1_LEGAL_SECTION_EVIDENCE_V1_REPORT.json"
    authoritative_report = {
        "schema_version": "dsc2026.gemini.huy_d1_legal_section_evidence_v1.report.v1",
        "experiment_id": "HUY_D1_LEGAL_SECTION_EVIDENCE_V1",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "verdict": eval_report["verdict"],
        "git": git_info,
        "model_provenance": model_provenance,
        "scoring_metadata": scoring_meta,
        "source_files": source_files_meta,
        "s0_baseline": eval_report["s0_baseline"],
        "s1_section_evidence": eval_report["s1_section_evidence"],
        "deltas": eval_report["deltas"],
        "paired_comparison": eval_report["paired_comparison"],
        "standalone_legal_section_ce": eval_report["standalone_legal_section_ce"],
        "posthoc_summary": {
            "rescued_count": posthoc_analysis.get("rescued_prior_error_queries_count", 0),
            "damaged_prior_count": posthoc_analysis.get("damaged_prior_error_queries_count", 0),
            "currently_correct_damaged_count": posthoc_analysis.get("currently_correct_queries_damaged_count", 0),
        },
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(authoritative_report, f, indent=2, ensure_ascii=False)
    print(f"Wrote {report_path}", flush=True)

    # 5. Generate DECISION.md
    dec_path = RES_DIR / "DECISION.md"
    verdict = eval_report["verdict"]
    s0 = eval_report["s0_baseline"]
    s1 = eval_report["s1_section_evidence"]
    del_ = eval_report["deltas"]
    pc = eval_report["paired_comparison"]
    std = eval_report["standalone_legal_section_ce"]

    md = f"""# DECISION REPORT: HUY_D1_LEGAL_SECTION_EVIDENCE_V1

## 1. Executive Summary & Verdict
- **Verdict**: **`{verdict}`**
- **Experiment ID**: `HUY_D1_LEGAL_SECTION_EVIDENCE_V1`
- **Pushed Source Commit**: `{git_info.get('head_commit')}`
- **Origin/Main Commit**: `{git_info.get('origin_main_commit')}`
- **Origin Parity**: `{git_info.get('parity')}` (Working tree clean: `{git_info.get('status_clean')}`)
- **Core Hypothesis**: Scoring structured legal sections (Chương, Mục, Điều, Phụ lục) with frozen Jina cross-encoder provides answer-bearing evidence that improves D1 Top-5 fusion without modifying candidate pool or rank views.

---

## 2. LOBO CAL600 Metrics: S0 vs S1

| Metric | S0 (D1 Baseline 48D) | S1 (D1 + Section CE 50D) | Delta (S1 - S0) | Status |
| :--- | :---: | :---: | :---: | :---: |
| **Pooled Recall@5** | **{s0['recall_at_5']:.16f}** | **{s1['recall_at_5']:.16f}** | **{del_['recall_at_5']:+.16f}** | {'IMPROVED' if del_['recall_at_5'] > 0 else 'REGRESSED' if del_['recall_at_5'] < 0 else 'TIED'} |
| Precision@5 | {s0['precision_at_5']:.6f} | {s1['precision_at_5']:.6f} | {del_['precision_at_5']:+.6f} | |
| Block A Recall@5 | {s0['block_recalls']['A']:.6f} | {s1['block_recalls']['A']:.6f} | {del_['block_deltas']['A']:+.6f} | |
| Block B Recall@5 | {s0['block_recalls']['B']:.6f} | {s1['block_recalls']['B']:.6f} | {del_['block_deltas']['B']:+.6f} | |
| Block C Recall@5 | {s0['block_recalls']['C']:.6f} | {s1['block_recalls']['C']:.6f} | {del_['block_deltas']['C']:+.6f} | |
| **Block D Recall@5** | **{s0['block_recalls']['D']:.6f}** | **{s1['block_recalls']['D']:.6f}** | **{del_['block_deltas']['D']:+.6f}** | {'IMPROVED' if del_['block_deltas']['D'] > 0 else 'REGRESSED' if del_['block_deltas']['D'] < 0 else 'TIED'} |
| Single-gold Recall@5 | {s0['single_gold_recall_at_5']:.6f} | {s1['single_gold_recall_at_5']:.6f} | {del_['single_gold_recall_at_5']:+.6f} | |
| Multi-gold Recall@5 | {s0['multi_gold_recall_at_5']:.6f} | {s1['multi_gold_recall_at_5']:.6f} | {del_['multi_gold_recall_at_5']:+.6f} | |
| Feature Dimension | 48D | 50D | +2D | Exactly 50D |

---

## 3. Paired Query Comparisons & Top-5 Dynamics
- **Query Wins / Losses / Ties**: **{pc['wins']} Wins / {pc['losses']} Losses / {pc['ties']} Ties** (Net: {pc['net_wins']:+d})
- **Gold Crossings into Top-5**: `{pc['gold_crossings_into_top5']}`
- **Gold Crossings out of Top-5**: `{pc['gold_crossings_out_of_top5']}` (Net: {pc['net_gold_crossings']:+d})
- **Changed Top-5 Sets**: `{pc['changed_top5_queries_count']} / 600 ({pc['changed_top5_queries_pct']:.1f}%)`
- **Standalone `legal_section_ce` Recall@5**: `{std['standalone_recall_at_5']:.6f}`

---

## 4. Post-Hoc Error Diagnostics (Residual 33 CAL Queries)
- Prior Imperfect-Recall Queries Analyzed: `{posthoc_analysis.get('total_prior_error_queries', 0)}`
- Prior Imperfect Queries Rescued by S1: `{posthoc_analysis.get('rescued_prior_error_queries_count', 0)}`
- Prior Imperfect Queries Damaged by S1: `{posthoc_analysis.get('damaged_prior_error_queries_count', 0)}`
- Currently-Correct Queries Damaged by S1: `{posthoc_analysis.get('currently_correct_queries_damaged_count', 0)}`

---

## 5. Promotion Gate & Recommendation
- **Promotion Gate Evaluation**:
  - `KEEP_FOR_NEXT_STAGE`: {'PASS' if verdict == 'KEEP_FOR_NEXT_STAGE' else 'FAIL'}
  - `INCONCLUSIVE`: {'TRUE' if verdict == 'INCONCLUSIVE' else 'FALSE'}
  - `KILL_LEGAL_SECTION_SCORE_V1`: {'TRIGGERED' if verdict == 'KILL_LEGAL_SECTION_SCORE_V1' else 'FALSE'}
- **Recommendation**:
  - If KILL: Retain current D1 (`D1_SCORE_ONLY_VNLEGAL`, 48D, R@5 = `0.9569444444444444`) as undisputed champion. Do not materialize public candidate packages.
  - If KEEP: Preserve `legal_section_ce_cv.pkl` and prepare next-stage integration.
"""
    with open(dec_path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"Wrote {dec_path}", flush=True)

    return authoritative_report
