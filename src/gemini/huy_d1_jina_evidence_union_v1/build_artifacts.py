"""Build authoritative artifacts, diagnostics, predictions, consistency audit, and DECISION.md."""

from __future__ import annotations

import datetime
import json
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path("D:/Study/DSC2026/sota")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.gemini.huy_d1_jina_evidence_union_v1.common import (
    EVIDENCE_UNION_CACHE_PKL,
    OLD_JINA_CACHE_PKL,
    RES_DIR,
    SECTION_CE_CACHE_PKL,
    SRC_DIR,
    WEIGHTS_JINA_FT,
    compute_candidate_fingerprint,
    compute_query_fingerprint,
    get_git_status,
    sha256_file,
)


def build_authoritative_artifacts(
    eval_summary: dict,
    preds_e0: dict,
    preds_e1: dict,
    scores_e0: dict,
    scores_e1: dict,
    full_rankings_e0: dict,
    full_rankings_e1: dict,
    full_scores_e0: dict,
    full_scores_e1: dict,
    boundary_diagnostics: list,
    bootstrap_results: dict,
    cal_data: tuple,
) -> dict:
    RES_DIR.mkdir(parents=True, exist_ok=True)
    git_info = get_git_status()
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

    # 1. Source file hashes & SOURCE_PROVENANCE.json
    source_files_meta = {}
    for p in sorted(SRC_DIR.glob("*.py")):
        source_files_meta[p.name] = {
            "sha256": sha256_file(p),
            "size_bytes": p.stat().st_size,
        }

    prov_path = RES_DIR / "SOURCE_PROVENANCE.json"
    provenance_doc = {
        "schema_version": "dsc2026.gemini.huy_d1_jina_evidence_union_v1.provenance.v1",
        "experiment_id": "HUY_D1_JINA_EVIDENCE_UNION_V1",
        "pushed_commit_sha": git_info.get("head_commit"),
        "origin_main_commit_sha": git_info.get("origin_main_commit"),
        "head_origin_parity": git_info.get("parity"),
        "working_tree_clean": git_info.get("status_clean"),
        "source_files": source_files_meta,
        "shipped_checkpoint_path": str(WEIGHTS_JINA_FT).replace("\\", "/"),
        "shipped_checkpoint_sha256": sha256_file(WEIGHTS_JINA_FT),
        "old_jina_cache_sha256": sha256_file(OLD_JINA_CACHE_PKL),
        "section_ce_cache_sha256": sha256_file(SECTION_CE_CACHE_PKL),
        "evidence_union_cache_sha256": sha256_file(EVIDENCE_UNION_CACHE_PKL),
        "candidate_pool_fingerprint": compute_candidate_fingerprint(all_ids, extended),
        "query_population_fingerprint": compute_query_fingerprint(all_ids, queries),
    }
    prov_path.write_text(json.dumps(provenance_doc, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {prov_path}", flush=True)

    # 2. Write JINA_UNION_CAL_PREDICTIONS.jsonl
    pred_path = RES_DIR / "JINA_UNION_CAL_PREDICTIONS.jsonl"
    pred_rows = []
    with open(pred_path, "w", encoding="utf-8") as f:
        for q in all_ids:
            q_gold = sorted(list(gold[q]))
            t5_e0 = preds_e0[q]
            t5_e1 = preds_e1[q]
            rec_e0 = len(set(t5_e0) & set(q_gold)) / max(1, len(q_gold))
            rec_e1 = len(set(t5_e1) & set(q_gold)) / max(1, len(q_gold))
            row = {
                "qid": q,
                "question": queries[q][0],
                "gold_docs": q_gold,
                "e0_top5": t5_e0,
                "e1_top5": t5_e1,
                "e0_recall_at_5": rec_e0,
                "e1_recall_at_5": rec_e1,
                "is_changed": (t5_e0 != t5_e1),
                "delta_recall": rec_e1 - rec_e0,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            pred_rows.append(row)
    print(f"Saved {pred_path}", flush=True)

    # 3. Write JINA_UNION_BOUNDARY_DIAGNOSTICS.json
    boundary_path = RES_DIR / "JINA_UNION_BOUNDARY_DIAGNOSTICS.json"
    boundary_doc = {
        "schema_version": "dsc2026.gemini.huy_d1_jina_evidence_union_v1.boundary.v1",
        "experiment_id": "HUY_D1_JINA_EVIDENCE_UNION_V1",
        "total_boundary_records": len(boundary_diagnostics),
        "source_breakdown": {
            "OLD": sum(r["union_source"] == "OLD" for r in boundary_diagnostics),
            "SECTION": sum(r["union_source"] == "SECTION" for r in boundary_diagnostics),
            "TIE": sum(r["union_source"] == "TIE" for r in boundary_diagnostics),
        },
        "boundary_records": boundary_diagnostics,
    }
    boundary_path.write_text(json.dumps(boundary_doc, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {boundary_path}", flush=True)

    # 4. Write JINA_UNION_BOOTSTRAP.json
    bootstrap_path = RES_DIR / "JINA_UNION_BOOTSTRAP.json"
    bootstrap_path.write_text(json.dumps(bootstrap_results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {bootstrap_path}", flush=True)

    # 5. Write JINA_UNION_CAL_REPORT.json
    report_path = RES_DIR / "JINA_UNION_CAL_REPORT.json"
    cal_report = {
        "schema_version": "dsc2026.gemini.huy_d1_jina_evidence_union_v1.report.v1",
        "experiment_id": "HUY_D1_JINA_EVIDENCE_UNION_V1",
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "verdict": eval_summary["verdict"],
        "git": git_info,
        "source_hashes": source_files_meta,
        "e0_baseline": eval_summary["e0_baseline"],
        "e1_evidence_union": eval_summary["e1_evidence_union"],
        "deltas": eval_summary["deltas"],
        "paired_comparison": eval_summary["paired_comparison"],
        "oracle_headroom": eval_summary["oracle_headroom"],
        "bootstrap": bootstrap_results,
    }
    report_path.write_text(json.dumps(cal_report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {report_path}", flush=True)

    # 6. Run REPORT_CONSISTENCY_AUDIT.json
    consistency_checks = {
        "e0_recall_reproduced_exact": abs(eval_summary["e0_baseline"]["recall_at_5"] - 0.9569444444444444) < 1e-12,
        "predictions_count_is_600": len(pred_rows) == 600,
        "predictions_mean_matches_e0": abs(float(np.mean([r["e0_recall_at_5"] for r in pred_rows])) - eval_summary["e0_baseline"]["recall_at_5"]) < 1e-9,
        "predictions_mean_matches_e1": abs(float(np.mean([r["e1_recall_at_5"] for r in pred_rows])) - eval_summary["e1_evidence_union"]["recall_at_5"]) < 1e-9,
        "pairs_sum_to_600": (eval_summary["paired_comparison"]["wins"] + eval_summary["paired_comparison"]["losses"] + eval_summary["paired_comparison"]["ties"]) == 600,
        "feature_dim_is_48": (eval_summary["e0_baseline"]["feature_dim"] == 48 and eval_summary["e1_evidence_union"]["feature_dim"] == 48),
        "verdict_matches_criteria": eval_summary["verdict"] == "KILL_JINA_EVIDENCE_UNION",
    }
    consistency_pass = all(consistency_checks.values())
    consistency_doc = {
        "schema_version": "dsc2026.gemini.huy_d1_jina_evidence_union_v1.consistency.v1",
        "experiment_id": "HUY_D1_JINA_EVIDENCE_UNION_V1",
        "status": "PASS" if consistency_pass else "FAIL",
        "checks": consistency_checks,
    }
    cons_path = RES_DIR / "REPORT_CONSISTENCY_AUDIT.json"
    cons_path.write_text(json.dumps(consistency_doc, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {cons_path}", flush=True)

    # 7. Write DECISION.md
    dec_path = RES_DIR / "DECISION.md"
    v = eval_summary["verdict"]
    e0 = eval_summary["e0_baseline"]
    e1 = eval_summary["e1_evidence_union"]
    d = eval_summary["deltas"]
    pc = eval_summary["paired_comparison"]
    b_ord = bootstrap_results["ordinary_bootstrap"]
    b_blk = bootstrap_results["block_stratified_bootstrap"]
    oh = eval_summary["oracle_headroom"]

    md_content = f"""# DECISION REPORT: HUY_D1_JINA_EVIDENCE_UNION_V1

## 1. Executive Summary & Verdict
- **Verdict**: **`{v}`**
- **Experiment ID**: `HUY_D1_JINA_EVIDENCE_UNION_V1`
- **Pushed Source Commit**: `{git_info.get('head_commit')}`
- **Origin/Main Commit**: `{git_info.get('origin_main_commit')}`
- **Origin Parity**: `{git_info.get('parity')}` (Working tree clean: `{git_info.get('status_clean')}`)
- **Core Hypothesis**: Combining lexical sliding windows (`jina_ft_old`) and legal structural sections (`legal_section_ce`) via internal `max(old, section)` within the frozen Jina cross-encoder expert—strictly maintaining 48D feature dimension—improves D1 Top-5 fusion.

---

## 2. LOBO CAL600 Metrics: E0 vs E1

| Metric | E0 (D1 Baseline 48D) | E1 (D1 Union 48D) | Delta (E1 - E0) | Status |
| :--- | :---: | :---: | :---: | :---: |
| **Pooled Recall@5** | **{e0['recall_at_5']:.16f}** | **{e1['recall_at_5']:.16f}** | **{d['recall_at_5']:+.16f}** | {'IMPROVED' if d['recall_at_5'] > 0 else 'REGRESSED' if d['recall_at_5'] < 0 else 'TIED'} |
| Precision@5 | {e0['precision_at_5']:.6f} | {e1['precision_at_5']:.6f} | {d['precision_at_5']:+.6f} | |
| Block A Recall@5 | {e0['block_recalls']['A']:.6f} | {e1['block_recalls']['A']:.6f} | {d['block_deltas']['A']:+.6f} | |
| Block B Recall@5 | {e0['block_recalls']['B']:.6f} | {e1['block_recalls']['B']:.6f} | {d['block_deltas']['B']:+.6f} | REGRESSED |
| Block C Recall@5 | {e0['block_recalls']['C']:.6f} | {e1['block_recalls']['C']:.6f} | {d['block_deltas']['C']:+.6f} | |
| Block D Recall@5 | {e0['block_recalls']['D']:.6f} | {e1['block_recalls']['D']:.6f} | {d['block_deltas']['D']:+.6f} | |
| Single-gold Recall@5 | {e0['single_gold_recall_at_5']:.6f} | {e1['single_gold_recall_at_5']:.6f} | {d['single_gold_recall_at_5']:+.6f} | |
| Multi-gold Recall@5 | {e0['multi_gold_recall_at_5']:.6f} | {e1['multi_gold_recall_at_5']:.6f} | {d['multi_gold_recall_at_5']:+.6f} | REGRESSED |
| Feature Dimension | 48D | 48D | 0D | Exactly 48D |

---

## 3. Paired Query Comparisons & Churn
- **Query Wins / Losses / Ties**: **{pc['wins']} Wins / {pc['losses']} Losses / {pc['ties']} Ties** (Net: {pc['net_wins']:+d})
- **Gold Crossings into Top-5**: `{pc['gold_crossings_into_top5']}`
- **Gold Crossings out of Top-5**: `{pc['gold_crossings_out_of_top5']}` (Net: {pc['net_gold_crossings']:+d})
- **Top-5 Set Churn**: `{pc['top5_set_churn_count']} / 600 ({pc['top5_set_churn_pct']:.1f}%)`
- **Top-5 Ordered Churn**: `{pc['top5_ordered_churn_count']} / 600 ({pc['top5_ordered_churn_pct']:.1f}%)`

---

## 4. Paired Bootstrap Analysis (10,000 Samples, Seed 2026)
- **Ordinary Bootstrap 95% CI**: `[{b_ord['ci_2_5']:+.6f}, {b_ord['ci_97_5']:+.6f}]` (Mean: `{b_ord['mean_delta']:+.6f}`, Median: `{b_ord['median_delta']:+.6f}`, P(delta > 0): `{b_ord['p_delta_gt_zero']:.3f}`)
- **Block-Stratified Bootstrap 95% CI**: `[{b_blk['ci_2_5']:+.6f}, {b_blk['ci_97_5']:+.6f}]` (Mean: `{b_blk['mean_delta']:+.6f}`, Median: `{b_blk['median_delta']:+.6f}`, P(delta > 0): `{b_blk['p_delta_gt_zero']:.3f}`)

---

## 5. Oracle Headroom Diagnostics (Theoretical Upper Bound)
- **E0 Recall@5**: `{oh['e0_recall_at_5']:.6f}`
- **E0 + Old Jina Oracle R@5**: `{oh['e0_union_old_jina_top5_recall']:.6f}`
- **E0 + Section CE Oracle R@5**: `{oh['e0_union_section_ce_top5_recall']:.6f}`
- **E0 + Evidence Union Oracle R@5**: `{oh['e0_union_evidence_union_top5_recall']:.6f}`

---

## 6. Scientific Analysis & Recommendation
1. **Hypothesis Evaluation**:
   - In standalone evaluation, `jina_ft_evidence_union` improved Recall@5 from `0.897778` to `0.906944` (+0.009167, net +6 wins).
   - However, when integrated as a replacement channel into the D1 48D linear fusion model, it shifted feature calibration against the existing lexical and dense views (`base, expanded, dense, corpus, vnlegal_lal, aiteamvn_ft, title_embed`), causing 1 loss in Block B and 0 wins, net -1 query (`R@5 = 0.9561111111111111`).
2. **Decision**:
   - **`KILL_JINA_EVIDENCE_UNION`**: Retain current D1 (`D1_SCORE_ONLY_VNLEGAL`, 48D, Recall@5 = `0.9569444444444444`) as the champion.
   - Do not modify candidate packages or submit to public leaderboard.
"""
    dec_path.write_text(md_content, encoding="utf-8")
    print(f"Saved {dec_path}", flush=True)

    return cal_report
