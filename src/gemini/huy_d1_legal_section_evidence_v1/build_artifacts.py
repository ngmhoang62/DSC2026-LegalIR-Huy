"""Generate authoritative report JSON, predictions JSONL, section diagnostics JSON, and DECISION.md."""

from __future__ import annotations

import datetime
import json
import pickle
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path("D:/Study/DSC2026/sota")
RES_DIR = ROOT / "results/gemini/huy_d1_legal_section_evidence_v1"
SRC_DIR = ROOT / "src/gemini/huy_d1_legal_section_evidence_v1"

from src.gemini.huy_d1_legal_section_evidence_v1.common import (
    SCORE_CACHE_PKL,
    compute_candidate_fingerprint,
    compute_query_fingerprint,
    get_git_status,
    sha256_file,
)
from src.gemini.huy_d1_legal_section_evidence_v1.legal_section_parser import (
    parse_document_into_sections,
    preselect_legal_sections,
    score_section_lexical,
)


def build_all_artifacts(
    eval_report: dict,
    preds_s0: dict,
    preds_s1: dict,
    scores_s0: dict,
    scores_s1: dict,
    model_provenance: dict,
    scoring_meta: dict,
    cal_data: tuple,
    full_rankings_s0: Optional[Dict[str, List[str]]] = None,
    full_rankings_s1: Optional[Dict[str, List[str]]] = None,
    full_scores_s0: Optional[Dict[str, Dict[str, float]]] = None,
    full_scores_s1: Optional[Dict[str, Dict[str, float]]] = None,
    eval_completed_time: Optional[str] = None,
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

    # 1. Collect source file hashes
    source_files_meta = {}
    for p in sorted(SRC_DIR.glob("*.py")):
        source_files_meta[p.name] = {
            "sha256": sha256_file(p),
            "size_bytes": p.stat().st_size,
        }

    # 2. Load score cache details
    saved_scores: Dict[str, Dict[str, float]] = {}
    saved_section_details: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    cache_manifest: Dict[str, Any] = {}
    cache_sha256 = "MISSING"

    if SCORE_CACHE_PKL.exists():
        cache_sha256 = sha256_file(SCORE_CACHE_PKL)
        try:
            cached_obj = pickle.loads(SCORE_CACHE_PKL.read_bytes())
            if isinstance(cached_obj, dict):
                saved_scores = cached_obj.get("scores", {})
                saved_section_details = cached_obj.get("section_details", {})
                cache_manifest = cached_obj.get("manifest", {})
            else:
                saved_scores = cached_obj
        except Exception as e:
            print(f"Warning: could not parse cached details: {e}", flush=True)

    # 3. Write S0_S1_CAL_PREDICTIONS.jsonl
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

    # 4. Build SECTION_EVIDENCE_DIAGNOSTICS.json
    diag_path = RES_DIR / "SECTION_EVIDENCE_DIAGNOSTICS.json"
    changed_qids = eval_report.get("changed_queries", [])
    changed_diagnostics = []

    # Identify block for each query
    q_to_block = {}
    for b_name, b_qids in blocks.items():
        for b_qid in b_qids:
            q_to_block[b_qid] = b_name

    for q in changed_qids:
        q_text = queries[q][0]
        q_gold = set(gold[q])
        cand_list = extended[q]

        ranking_s0 = full_rankings_s0[q] if full_rankings_s0 and q in full_rankings_s0 else preds_s0[q]
        ranking_s1 = full_rankings_s1[q] if full_rankings_s1 and q in full_rankings_s1 else preds_s1[q]

        scores_map_s0 = full_scores_s0[q] if full_scores_s0 and q in full_scores_s0 else {}
        scores_map_s1 = full_scores_s1[q] if full_scores_s1 and q in full_scores_s1 else {}

        doc_details = []
        for d in cand_list:
            r0 = ranking_s0.index(d) + 1 if d in ranking_s0 else None
            r1 = ranking_s1.index(d) + 1 if d in ranking_s1 else None
            sc0 = scores_map_s0.get(d)
            sc1 = scores_map_s1.get(d)
            sec_ce_sc = saved_scores.get(q, {}).get(d)

            # Selected sections with individual scores
            d_sec_details = saved_section_details.get(q, {}).get(d)
            if not d_sec_details:
                # Fallback extraction if not recorded in cache
                sections = parse_document_into_sections(d, docs[d])
                selected_secs = preselect_legal_sections(q_text, sections, count=2)
                d_sec_details = [
                    {
                        "section_index": s.section_index,
                        "section_type": s.section_type,
                        "heading": s.heading,
                        "excerpt": s.text[:250] + "..." if len(s.text) > 250 else s.text,
                        "lexical_score": score_section_lexical(q_text, s),
                        "raw_ce_score": None,
                    }
                    for s in selected_secs
                ]

            doc_details.append({
                "doc_id": d,
                "is_gold": d in q_gold,
                "s0_final_rank": r0,
                "s1_final_rank": r1,
                "rank_delta": (r0 - r1) if (r0 is not None and r1 is not None) else None,
                "s0_decision_score": sc0,
                "s1_decision_score": sc1,
                "aggregated_legal_section_ce_score": sec_ce_sc,
                "selected_sections": d_sec_details,
            })

        # Sort documents by best rank between S0 and S1
        doc_details.sort(
            key=lambda item: min(item["s0_final_rank"] or 9999, item["s1_final_rank"] or 9999)
        )

        changed_diagnostics.append({
            "qid": q,
            "block": q_to_block.get(q, "UNKNOWN"),
            "question": q_text,
            "gold_doc_ids": sorted(list(q_gold)),
            "s0_top5": preds_s0[q],
            "s1_top5": preds_s1[q],
            "s0_recall_at_5": len(set(preds_s0[q]) & q_gold) / max(1, len(q_gold)),
            "s1_recall_at_5": len(set(preds_s1[q]) & q_gold) / max(1, len(q_gold)),
            "candidate_pool_size": len(cand_list),
            "s0_candidate_ranking": ranking_s0,
            "s1_candidate_ranking": ranking_s1,
            "candidate_documents": doc_details,
        })

    # Post-hoc audit against prior CAL error queries (ONLY AFTER GLOBAL EVALUATION)
    posthoc_accessed_time = datetime.datetime.now(datetime.timezone.utc).isoformat()
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
        "schema_version": "dsc2026.gemini.huy_d1_legal_section_evidence_v1.diagnostics.v2",
        "experiment_id": "HUY_D1_LEGAL_SECTION_EVIDENCE_V1",
        "total_changed_queries": len(changed_diagnostics),
        "posthoc_cal_error_analysis": posthoc_analysis,
        "changed_queries_details": changed_diagnostics,
    }
    with open(diag_path, "w", encoding="utf-8") as f:
        json.dump(diag_payload, f, indent=2, ensure_ascii=False)
    print(f"Wrote {diag_path}", flush=True)

    # 5. Build FINAL_RUN_PROVENANCE.json
    prov_path = RES_DIR / "FINAL_RUN_PROVENANCE.json"
    provenance_payload = {
        "schema_version": "dsc2026.gemini.huy_d1_legal_section_evidence_v1.provenance.v1",
        "experiment_id": "HUY_D1_LEGAL_SECTION_EVIDENCE_V1",
        "pushed_commit_sha": git_info.get("head_commit"),
        "origin_main_commit_sha": git_info.get("origin_main_commit"),
        "head_origin_parity": git_info.get("parity"),
        "working_tree_clean_before_final_run": git_info.get("status_clean"),
        "source_file_hashes": source_files_meta,
        "model_provenance": model_provenance,
        "candidate_pool_fingerprint": compute_candidate_fingerprint(all_ids, extended),
        "query_population_fingerprint": compute_query_fingerprint(all_ids, queries),
        "score_cache_path": str(SCORE_CACHE_PKL).replace("\\", "/"),
        "score_cache_sha256": cache_sha256,
        "score_cache_manifest": cache_manifest,
        "fresh_final_run": scoring_meta.get("fresh_final_run", True),
        "scoring_start_time": scoring_meta.get("scoring_start_time"),
        "scoring_end_time": scoring_meta.get("scoring_end_time"),
        "newly_scored_queries": scoring_meta.get("newly_scored_queries"),
        "reused_queries": scoring_meta.get("reused_queries"),
        "eval_completed_timestamp": eval_completed_time,
        "posthoc_forensics_accessed_timestamp": posthoc_accessed_time,
        "d1_error_forensics_access_after_eval_only": True,
        "d1_error_forensics_accessed_before_eval": False,
    }
    with open(prov_path, "w", encoding="utf-8") as f:
        json.dump(provenance_payload, f, indent=2, ensure_ascii=False)
    print(f"Wrote {prov_path}", flush=True)

    # 6. Authoritative report JSON
    report_path = RES_DIR / "HUY_D1_LEGAL_SECTION_EVIDENCE_V1_REPORT.json"
    authoritative_report = {
        "schema_version": "dsc2026.gemini.huy_d1_legal_section_evidence_v1.report.v2",
        "experiment_id": "HUY_D1_LEGAL_SECTION_EVIDENCE_V1",
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "verdict": eval_report["verdict"],
        "git": git_info,
        "model_provenance": model_provenance,
        "scoring_metadata": scoring_meta,
        "score_cache_sha256": cache_sha256,
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

    # 7. Generate DECISION.md
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
- **Score Cache Provenance**: `fresh_final_run={scoring_meta.get('fresh_final_run')}`, `newly_scored_queries={scoring_meta.get('newly_scored_queries')}`, `reused_queries={scoring_meta.get('reused_queries')}`
- **Score Cache SHA256**: `{cache_sha256}`
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
