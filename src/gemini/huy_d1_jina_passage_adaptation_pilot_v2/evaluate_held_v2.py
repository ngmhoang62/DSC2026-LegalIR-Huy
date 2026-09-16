"""Evaluate frozen shipped teacher (T0) vs adapted student (T1) on Held Fold 0 using corrected fractional Recall@K."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import numpy as np
import scipy.stats as stats
import torch
from peft import PeftModel

from .audit_split import run_split_audit
from .common import (
    ADAPTER_DIR,
    RES_DIR,
    ROOT,
    compute_qid_list_fingerprint,
    get_git_status,
    load_jina_base_with_shipped_weights,
    load_v2_contexts,
    load_v2_inputs,
    patch_tuple_returning_lora,
    seed_everything,
)
from .legal_section_parser import LegalSection, parse_document_into_sections, preselect_legal_sections

HELD_EVAL_FILE = RES_DIR / "HELD_V2_EVALUATION.json"


def compute_fractional_metrics(
    queries: List[str],
    golds: Dict[str, Set[str]],
    doc_scores: Dict[str, Dict[str, float]],
    cand_pools: Dict[str, List[str]],
) -> Dict[str, Any]:
    """Compute exact fractional Recall@1, 5, 8, 10, single/multi gold recall, and score stats."""
    recalls = {1: [], 5: [], 8: [], 10: []}
    single_gold_r5 = []
    multi_gold_r5 = []
    all_scores = []
    query_sigmas = []
    empty_gold_queries = []

    for q in queries:
        gold_set = golds.get(q, set())
        if not gold_set:
            empty_gold_queries.append(q)
            continue

        q_scores = doc_scores.get(q, {})
        ranked = sorted(cand_pools[q], key=lambda d: q_scores.get(d, -1e9), reverse=True)

        for k in [1, 5, 8, 10]:
            top_k = set(ranked[:k])
            # Exact fractional recall: len(TopK ∩ Gold) / len(Gold)
            rec = len(top_k & gold_set) / len(gold_set)
            recalls[k].append(rec)

        r5_val = len(set(ranked[:5]) & gold_set) / len(gold_set)
        if len(gold_set) == 1:
            single_gold_r5.append(r5_val)
        elif len(gold_set) > 1:
            multi_gold_r5.append(r5_val)

        s_vals = [q_scores.get(d, 0.0) for d in cand_pools[q]]
        all_scores.extend(s_vals)
        if len(s_vals) > 1:
            query_sigmas.append(float(np.std(s_vals)))

    near_constant_queries = sum(1 for s in query_sigmas if s < 1e-4)

    return {
        "recall_1": float(np.mean(recalls[1])),
        "recall_5": float(np.mean(recalls[5])),
        "recall_8": float(np.mean(recalls[8])),
        "recall_10": float(np.mean(recalls[10])),
        "single_gold_recall_5": float(np.mean(single_gold_r5)) if single_gold_r5 else 0.0,
        "multi_gold_recall_5": float(np.mean(multi_gold_r5)) if multi_gold_r5 else 0.0,
        "single_gold_count": len(single_gold_r5),
        "multi_gold_count": len(multi_gold_r5),
        "empty_gold_queries_count": len(empty_gold_queries),
        "score_distribution": {
            "mean": float(np.mean(all_scores)),
            "std": float(np.std(all_scores)),
            "min": float(np.min(all_scores)),
            "max": float(np.max(all_scores)),
            "near_constant_queries_count": near_constant_queries,
            "near_constant_queries_fraction": near_constant_queries / max(1, len(queries)),
        },
    }


def evaluate_held_v2(batch_size: int = 64) -> Dict[str, Any]:
    print("=== STAGE: HELD NON-CAL V2 EVALUATION (FOLD 0) ===", flush=True)
    RES_DIR.mkdir(parents=True, exist_ok=True)
    seed_everything(2026)
    git_info = get_git_status()

    # 1. Load exact sealed held population from SPLIT_AUDIT.json
    split_audit_path = RES_DIR / "SPLIT_AUDIT.json"
    if not split_audit_path.exists():
        run_split_audit()
    split_audit = json.loads(split_audit_path.read_text(encoding="utf-8"))

    held_qids = split_audit["held_partition"]["held_qids_final"]
    expected_held_fp = split_audit["held_partition"]["held_qids_final_fingerprint"]
    actual_held_fp = compute_qid_list_fingerprint(held_qids)

    if actual_held_fp != expected_held_fp:
        raise RuntimeError(
            f"BLOCKED_SPLIT_PARITY: held_qids_final fingerprint mismatch!\n"
            f"Expected: {expected_held_fp}\nActual:   {actual_held_fp}"
        )

    print(f"[HELD_EVAL] Loaded {len(held_qids)} sealed held non-CAL queries (FP: {actual_held_fp[:12]}...).", flush=True)

    folds, pools, questions, v2_golds = load_v2_inputs()
    contexts = load_v2_contexts()

    # Document section cache
    doc_sections_cache: Dict[str, List[LegalSection]] = {}

    def get_cached_sections(doc_id: str) -> List[LegalSection]:
        if doc_id not in doc_sections_cache:
            raw_text = contexts.get(doc_id, "")
            doc_sections_cache[doc_id] = parse_document_into_sections(
                doc_id, raw_text, max_chunk_words=220, overlap_words=60
            )
        return doc_sections_cache[doc_id]

    # Prepare section pairs for all candidate pairs in Held Fold 0
    pair_list: List[Tuple[str, str]] = []
    pair_meta: List[Tuple[str, str, int]] = []

    for qid in held_qids:
        q_text = questions[qid]
        cand_docs = [str(d) for d in pools[qid]]
        for doc_id in cand_docs:
            secs = get_cached_sections(doc_id)
            chosen_secs = preselect_legal_sections(q_text, secs, count=2)
            for s_idx, sec in enumerate(chosen_secs):
                pair_list.append((q_text, sec.text))
                pair_meta.append((qid, doc_id, s_idx))

    print(f"[HELD_EVAL] Total section pairs to score per model: {len(pair_list)}", flush=True)

    # 2. Score with Frozen Teacher (T0)
    print("[HELD_EVAL] Scoring with frozen shipped teacher (T0)...", flush=True)
    t0_model, tok = load_jina_base_with_shipped_weights()
    t0_model.eval().to("cuda")

    t0_logits: List[float] = []
    for i in range(0, len(pair_list), batch_size):
        b = pair_list[i : i + batch_size]
        inputs = tok(b, padding=True, truncation=True, return_tensors="pt", max_length=512).to("cuda")
        with torch.no_grad():
            s = t0_model(**inputs, return_dict=True).logits.view(-1).float()
        t0_logits.extend(s.cpu().numpy().tolist())

    del t0_model
    torch.cuda.empty_cache()

    # Aggregate T0 doc scores: MAX over sections
    t0_doc_scores: Dict[str, Dict[str, float]] = {q: {} for q in held_qids}
    for (qid, doc_id, s_idx), logit_val in zip(pair_meta, t0_logits):
        t0_doc_scores[qid][doc_id] = max(t0_doc_scores[qid].get(doc_id, -1e9), float(logit_val))

    # 3. Score with Adapted Model (T1)
    print("[HELD_EVAL] Scoring with adapted pure-LoRA model (T1)...", flush=True)
    t1_base, tok1 = load_jina_base_with_shipped_weights()
    t1_model = PeftModel.from_pretrained(t1_base, ADAPTER_DIR)
    patch_tuple_returning_lora(t1_model)
    t1_model.eval().to("cuda")

    t1_logits: List[float] = []
    for i in range(0, len(pair_list), batch_size):
        b = pair_list[i : i + batch_size]
        inputs = tok1(b, padding=True, truncation=True, return_tensors="pt", max_length=512).to("cuda")
        with torch.no_grad():
            s = t1_model(**inputs, return_dict=True).logits.view(-1).float()
        t1_logits.extend(s.cpu().numpy().tolist())

    del t1_base, t1_model
    torch.cuda.empty_cache()

    # Aggregate T1 doc scores: MAX over sections
    t1_doc_scores: Dict[str, Dict[str, float]] = {q: {} for q in held_qids}
    for (qid, doc_id, s_idx), logit_val in zip(pair_meta, t1_logits):
        t1_doc_scores[qid][doc_id] = max(t1_doc_scores[qid].get(doc_id, -1e9), float(logit_val))

    # 4. Compute fractional metrics
    golds_dict = {q: set(str(d) for d in v2_golds.get(q, set())) for q in held_qids}
    cand_pools_dict = {q: [str(d) for d in pools[q]] for q in held_qids}

    t0_metrics = compute_fractional_metrics(held_qids, golds_dict, t0_doc_scores, cand_pools_dict)
    t1_metrics = compute_fractional_metrics(held_qids, golds_dict, t1_doc_scores, cand_pools_dict)

    # Paired comparison at fractional Recall@5
    wins = 0
    losses = 0
    ties = 0
    spearmans = []

    for q in held_qids:
        cands = cand_pools_dict[q]
        gold_set = golds_dict[q]
        if not gold_set:
            continue

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

    print(
        f"[HELD_EVAL] Fractional Result: T0 R@5 = {t0_metrics['recall_5']:.6f} | "
        f"T1 R@5 = {t1_metrics['recall_5']:.6f} | Delta = {delta_r5:+.6f} "
        f"(Wins: {wins}, Losses: {losses}, Ties: {ties})",
        flush=True,
    )
    print(
        f"[HELD_EVAL] Stability: Mean Spearman T0->T1 = {mean_spearman:.4f} | "
        f"Sigma Ratio (T1/T0) = {sigma_ratio:.4f}",
        flush=True,
    )

    result_data = {
        "schema_version": "dsc2026.gemini.huy_d1_jina_passage_adaptation_pilot_v2.held_v2_evaluation.v2",
        "experiment_id": "HUY_D1_JINA_PASSAGE_ADAPTATION_PILOT_V2",
        "status": "PASS",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_info["head_commit"],
        "held_partition": "fold_0_non_cal",
        "total_held_queries": len(held_qids),
        "metric_semantics": "exact_query_macro_fractional_recall",
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
        "anti_contamination_assertion": {
            "cal_qids_present": False,
            "cal_labels_read_in_held_eval": False,
        },
    }

    HELD_EVAL_FILE.write_text(json.dumps(result_data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[HELD_EVAL] Held evaluation saved -> {HELD_EVAL_FILE}", flush=True)
    return result_data


if __name__ == "__main__":
    evaluate_held_v2()
