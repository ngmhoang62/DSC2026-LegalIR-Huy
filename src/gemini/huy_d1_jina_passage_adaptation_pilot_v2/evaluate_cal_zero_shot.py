"""Stage 2: CAL600 zero-shot transfer evaluation and D1 complementarity diagnostic using fractional Recall."""

from __future__ import annotations

import json
import pickle
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import numpy as np
import scipy.stats as stats
import torch
from peft import PeftModel

from .common import (
    ADAPTER_DIR,
    D1_PREDICTIONS_JSONL,
    EXPECTED_D1_R5,
    EXPECTED_ORACLE_T0_APPROX,
    EXPECTED_T0_R5_APPROX,
    FROZEN_SECTION_CE_CV_PKL,
    RES_DIR,
    ROOT,
    get_git_status,
    load_cal_candidate_pools,
    load_cal_contexts,
    load_cal_gold_labels,
    load_cal_questions_label_free,
    load_jina_base_with_shipped_weights,
    patch_tuple_returning_lora,
    seed_everything,
)
from .legal_section_parser import parse_document_into_sections, preselect_legal_sections

CAL_EVAL_FILE = RES_DIR / "CAL_ZERO_SHOT_EVALUATION.json"
D1_DIAGNOSTIC_FILE = RES_DIR / "D1_COMPLEMENTARITY_DIAGNOSTIC.json"
HELD_EVAL_FILE = RES_DIR / "HELD_V2_EVALUATION.json"


def evaluate_cal_zero_shot(batch_size: int = 64) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    print("=== STAGE: CAL600 ZERO-SHOT TRANSFER EVALUATION ===", flush=True)
    RES_DIR.mkdir(parents=True, exist_ok=True)
    seed_everything(2026)
    git_info = get_git_status()

    # STRICT GATING: Held evaluation must exist before opening CAL
    if not HELD_EVAL_FILE.exists():
        raise RuntimeError(
            "STAGE_GATE_BLOCKED: HELD_V2_EVALUATION.json must be materialized on disk before evaluating CAL!"
        )

    # 1. Load CAL queries and candidates genuinely label-free
    all_ids, queries = load_cal_questions_label_free()
    extended = load_cal_candidate_pools()
    contexts = load_cal_contexts()
    print(f"[CAL_ZERO_SHOT] Loaded {len(all_ids)} CAL queries label-free.", flush=True)

    # 2. Score with Frozen Shipped Jina (T0) from authoritative cache
    if FROZEN_SECTION_CE_CV_PKL.exists():
        print(f"[CAL_ZERO_SHOT] Loading authoritative frozen section CE cache: {FROZEN_SECTION_CE_CV_PKL}", flush=True)
        cached_data = pickle.loads(FROZEN_SECTION_CE_CV_PKL.read_bytes())
        t0_doc_scores: Dict[str, Dict[str, float]] = cached_data["scores"]
    else:
        raise FileNotFoundError(f"Authoritative section CE cache not found: {FROZEN_SECTION_CE_CV_PKL}")

    # 3. Score with Adapted Pure-LoRA Jina (T1)
    print("[CAL_ZERO_SHOT] Loading adapted pure-LoRA model (T1) for CAL inference...", flush=True)
    base_model, tok = load_jina_base_with_shipped_weights()
    t1_model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
    patch_tuple_returning_lora(t1_model)
    t1_model.eval().to("cuda")

    # Document section cache for CAL contexts
    cal_doc_sections_cache = {}
    for d_id in sorted(contexts.keys()):
        raw_t = contexts[d_id]
        cal_doc_sections_cache[d_id] = parse_document_into_sections(
            d_id, raw_t, max_chunk_words=220, overlap_words=60
        )

    pair_list: List[Tuple[str, str]] = []
    pair_meta: List[Tuple[str, str, int]] = []

    for qid in all_ids:
        q_text = queries[qid]
        cand_docs = extended[qid]
        for doc_id in cand_docs:
            secs = cal_doc_sections_cache.get(doc_id, [])
            chosen_secs = preselect_legal_sections(q_text, secs, count=2)
            for s_idx, sec in enumerate(chosen_secs):
                pair_list.append((q_text, sec.text))
                pair_meta.append((qid, doc_id, s_idx))

    print(f"[CAL_ZERO_SHOT] Scoring {len(pair_list)} CAL section pairs with T1...", flush=True)
    t1_logits: List[float] = []
    for i in range(0, len(pair_list), batch_size):
        b = pair_list[i : i + batch_size]
        inputs = tok(b, padding=True, truncation=True, return_tensors="pt", max_length=512).to("cuda")
        with torch.no_grad():
            s = t1_model(**inputs, return_dict=True).logits.view(-1).float()
        t1_logits.extend(s.cpu().numpy().tolist())

    del base_model, t1_model
    torch.cuda.empty_cache()

    # Aggregate T1 doc scores: MAX over sections
    t1_doc_scores: Dict[str, Dict[str, float]] = {q: {} for q in all_ids}
    for (qid, doc_id, s_idx), logit_val in zip(pair_meta, t1_logits):
        t1_doc_scores[qid][doc_id] = max(t1_doc_scores[qid].get(doc_id, -1e9), float(logit_val))

    # 4. Now open CAL gold labels strictly for evaluation
    print("[CAL_ZERO_SHOT] Materializing CAL gold labels for evaluation...", flush=True)
    cal_golds = load_cal_gold_labels(all_ids)

    # Compute Standalone Metrics for T0 and T1 using fractional recall
    from .evaluate_held_v2 import compute_fractional_metrics
    t0_metrics = compute_fractional_metrics(all_ids, cal_golds, t0_doc_scores, extended)
    t1_metrics = compute_fractional_metrics(all_ids, cal_golds, t1_doc_scores, extended)

    # Compare query-level fractional R@5
    wins = 0
    losses = 0
    ties = 0
    spearmans = []

    for q in all_ids:
        cands = extended[q]
        gold_set = cal_golds[q]

        t0_top5 = set(sorted(cands, key=lambda d: t0_doc_scores[q].get(d, -1e9), reverse=True)[:5])
        t1_top5 = set(sorted(cands, key=lambda d: t1_doc_scores[q].get(d, -1e9), reverse=True)[:5])

        r0 = len(t0_top5 & gold_set) / len(gold_set)
        r1 = len(t1_top5 & gold_set) / len(gold_set)

        if r1 > r0 + 1e-9:
            wins += 1
        elif r0 > r1 + 1e-9:
            losses += 1
        else:
            ties += 1

        t0_v = [t0_doc_scores[q].get(d, 0.0) for d in cands]
        t1_v = [t1_doc_scores[q].get(d, 0.0) for d in cands]
        rho, _ = stats.spearmanr(t0_v, t1_v)
        if np.isfinite(rho):
            spearmans.append(float(rho))

    mean_spearman = float(np.mean(spearmans)) if spearmans else 1.0
    sigma_t0 = t0_metrics["score_distribution"]["std"]
    sigma_t1 = t1_metrics["score_distribution"]["std"]
    sigma_ratio = sigma_t1 / max(1e-9, sigma_t0)
    delta_r5 = t1_metrics["recall_5"] - t0_metrics["recall_5"]

    # Verify T0 baseline parity against authoritative previous evidence
    t0_r5 = t0_metrics["recall_5"]
    baseline_parity_diff = abs(t0_r5 - EXPECTED_T0_R5_APPROX)
    baseline_matches = (baseline_parity_diff < 1e-6)

    print(
        f"[CAL_ZERO_SHOT] Fractional Standalone: T0 R@5 = {t0_r5:.6f} | "
        f"T1 R@5 = {t1_metrics['recall_5']:.6f} | Delta = {delta_r5:+.6f} "
        f"(Wins: {wins}, Losses: {losses}, Ties: {ties})",
        flush=True,
    )
    print(
        f"[CAL_ZERO_SHOT] Stability: Mean Spearman T0->T1 = {mean_spearman:.4f} | "
        f"Sigma Ratio (T1/T0) = {sigma_ratio:.4f}",
        flush=True,
    )

    cal_eval_data = {
        "schema_version": "dsc2026.gemini.huy_d1_jina_passage_adaptation_pilot_v2.cal_zero_shot_evaluation.v2",
        "experiment_id": "HUY_D1_JINA_PASSAGE_ADAPTATION_PILOT_V2",
        "status": "PASS",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_info["head_commit"],
        "total_cal_queries": len(all_ids),
        "metric_semantics": "exact_query_macro_fractional_recall",
        "reconstructed_baseline_parity": {
            "t0_recall_5": t0_r5,
            "expected_baseline_approx": EXPECTED_T0_R5_APPROX,
            "baseline_parity_difference": baseline_parity_diff,
            "baseline_reconstructed_exactly": baseline_matches,
        },
        "teacher_t0_metrics": t0_metrics,
        "adapted_t1_metrics": t1_metrics,
        "comparison": {
            "delta_recall_1": t1_metrics["recall_1"] - t0_metrics["recall_1"],
            "delta_recall_5": delta_r5,
            "delta_recall_8": t1_metrics["recall_8"] - t0_metrics["recall_8"],
            "delta_recall_10": t1_metrics["recall_10"] - t0_metrics["recall_10"],
            "delta_single_gold_recall_5": t1_metrics["single_gold_recall_5"] - t0_metrics["single_gold_recall_5"],
            "delta_multi_gold_recall_5": t1_metrics["multi_gold_recall_5"] - t0_metrics["multi_gold_recall_5"],
            "r5_wins": wins,
            "r5_losses": losses,
            "r5_ties": ties,
            "mean_within_query_spearman": mean_spearman,
            "sigma_ratio_t1_over_t0": sigma_ratio,
        },
    }

    CAL_EVAL_FILE.write_text(json.dumps(cal_eval_data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[CAL_ZERO_SHOT] Saved -> {CAL_EVAL_FILE}", flush=True)

    # 5. D1 Complementarity Diagnostic (Oracle) with fractional Recall
    print("=== STAGE: D1 COMPLEMENTARITY DIAGNOSTIC ===", flush=True)
    d1_top5_by_qid: Dict[str, List[str]] = {}
    with open(D1_PREDICTIONS_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                d1_top5_by_qid[str(row["qid"])] = [str(d) for d in row.get("s0_top5", [])]

    # Evaluate D1 baseline standalone with fractional recall
    d1_recalls = []
    d1_imperfect_qids = []
    for q in all_ids:
        gold_set = cal_golds[q]
        top5 = set(d1_top5_by_qid[q])
        rec = len(top5 & gold_set) / len(gold_set)
        d1_recalls.append(rec)
        if rec < 1.0:
            d1_imperfect_qids.append(q)

    d1_recall_5 = float(np.mean(d1_recalls))
    print(f"[D1_DIAGNOSTIC] D1 Standalone Fractional Recall@5: {d1_recall_5:.16f}", flush=True)
    print(f"[D1_DIAGNOSTIC] D1 Imperfect queries count (< 1.0):   {len(d1_imperfect_qids)}", flush=True)

    # Hard gate: D1 baseline must equal 0.9569444444444444 within 1e-12
    d1_diff = abs(d1_recall_5 - EXPECTED_D1_R5)
    if d1_diff > 1e-12:
        raise RuntimeError(
            f"BLOCKED_D1_DIAGNOSTIC_PARITY: D1 Recall@5 mismatch!\n"
            f"Expected: {EXPECTED_D1_R5:.16f}\nActual:   {d1_recall_5:.16f}\nDiff: {d1_diff:.2e}"
        )

    # Oracle Recall@5 of D1 Top-5 UNION T0 Top-5
    d1_union_t0_recalls = []
    t0_recovered_imperfect = []
    for q in all_ids:
        gold_set = cal_golds[q]
        d1_top5 = set(d1_top5_by_qid[q])
        t0_top5 = set(sorted(extended[q], key=lambda d: t0_doc_scores[q].get(d, -1e9), reverse=True)[:5])
        union_set = d1_top5 | t0_top5
        rec = len(union_set & gold_set) / len(gold_set)
        d1_union_t0_recalls.append(rec)
        if q in d1_imperfect_qids:
            d1_r = len(d1_top5 & gold_set) / len(gold_set)
            if rec > d1_r:
                t0_recovered_imperfect.append(q)

    oracle_t0_r5 = float(np.mean(d1_union_t0_recalls))
    print(f"[D1_DIAGNOSTIC] Oracle (D1 U T0): {oracle_t0_r5:.16f} (Recovered {len(t0_recovered_imperfect)} imperfect queries)", flush=True)

    # Hard gate: Frozen T0 oracle must match ~0.9683333333333334 within 1e-6
    oracle_diff = abs(oracle_t0_r5 - EXPECTED_ORACLE_T0_APPROX)
    if oracle_diff > 1e-6:
        raise RuntimeError(
            f"BLOCKED_FROZEN_COMPLEMENTARITY_PARITY: Frozen oracle mismatch!\n"
            f"Expected: {EXPECTED_ORACLE_T0_APPROX:.16f}\nActual:   {oracle_t0_r5:.16f}\nDiff: {oracle_diff:.2e}"
        )

    # Oracle Recall@5 of D1 Top-5 UNION T1 Top-5
    d1_union_t1_recalls = []
    t1_recovered_imperfect = []
    for q in all_ids:
        gold_set = cal_golds[q]
        d1_top5 = set(d1_top5_by_qid[q])
        t1_top5 = set(sorted(extended[q], key=lambda d: t1_doc_scores[q].get(d, -1e9), reverse=True)[:5])
        union_set = d1_top5 | t1_top5
        rec = len(union_set & gold_set) / len(gold_set)
        d1_union_t1_recalls.append(rec)
        if q in d1_imperfect_qids:
            d1_r = len(d1_top5 & gold_set) / len(gold_set)
            if rec > d1_r:
                t1_recovered_imperfect.append(q)

    oracle_t1_r5 = float(np.mean(d1_union_t1_recalls))
    print(f"[D1_DIAGNOSTIC] Oracle (D1 U T1): {oracle_t1_r5:.16f} (Recovered {len(t1_recovered_imperfect)} imperfect queries)", flush=True)

    diagnostic_data = {
        "schema_version": "dsc2026.gemini.huy_d1_jina_passage_adaptation_pilot_v2.d1_complementarity.v2",
        "experiment_id": "HUY_D1_JINA_PASSAGE_ADAPTATION_PILOT_V2",
        "status": "PASS",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_info["head_commit"],
        "metric_semantics": "exact_query_macro_fractional_recall",
        "d1_baseline_recall_5": d1_recall_5,
        "d1_baseline_expected_recall_5": EXPECTED_D1_R5,
        "d1_baseline_parity_difference": d1_diff,
        "d1_imperfect_queries_count": len(d1_imperfect_qids),
        "d1_imperfect_qids": d1_imperfect_qids,
        "d1_oracle_top5_t0_recall5": oracle_t0_r5,
        "d1_oracle_top5_t0_expected_approx": EXPECTED_ORACLE_T0_APPROX,
        "d1_oracle_top5_t0_parity_difference": oracle_diff,
        "d1_oracle_top5_t0_recovered_count": len(t0_recovered_imperfect),
        "d1_oracle_top5_t0_recovered_qids": t0_recovered_imperfect,
        "d1_oracle_top5_t1_recall5": oracle_t1_r5,
        "d1_oracle_top5_t1_recovered_count": len(t1_recovered_imperfect),
        "d1_oracle_top5_t1_recovered_qids": t1_recovered_imperfect,
        "oracle_delta_t1_vs_t0": oracle_t1_r5 - oracle_t0_r5,
    }

    D1_DIAGNOSTIC_FILE.write_text(json.dumps(diagnostic_data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[D1_DIAGNOSTIC] Saved -> {D1_DIAGNOSTIC_FILE}", flush=True)

    return cal_eval_data, diagnostic_data


if __name__ == "__main__":
    evaluate_cal_zero_shot()
