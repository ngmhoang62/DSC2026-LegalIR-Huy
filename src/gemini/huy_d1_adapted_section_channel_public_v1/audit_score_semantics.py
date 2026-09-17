"""Stage 2: Deterministic audit verifying frozen Section CE cache corresponds to sigmoid(raw_logit)."""

from __future__ import annotations

import json
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import scipy.stats
import torch

from .common import (
    FROZEN_SECTION_CACHE_PATH,
    RESULTS_DIR,
    get_git_status,
    load_cal_data_label_free,
    load_shipped_frozen_jina,
    seed_everything,
)
from .legal_section_parser import parse_document_into_sections, preselect_legal_sections


def run_score_semantics_audit(sample_size_pairs: int = 64) -> Dict[str, Any]:
    print("=== STAGE 2: SECTION SCORE SEMANTICS AUDIT (LABEL-FREE) ===", flush=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    seed_everything(2026)
    git_info = get_git_status()

    # 1. Load CAL data (strictly label-free) and frozen Section CE cache
    print("Loading CAL data (strictly label-free) and frozen Section CE cache...", flush=True)
    docs, queries, blocks, all_ids, extended, local_views, full_channels_cv, type_rows, cite_rows = load_cal_data_label_free()

    if not FROZEN_SECTION_CACHE_PATH.exists():
        raise FileNotFoundError(f"Missing frozen Section CE cache: {FROZEN_SECTION_CACHE_PATH}")

    cached_obj = pickle.loads(FROZEN_SECTION_CACHE_PATH.read_bytes())
    frozen_scores = cached_obj["scores"] if isinstance(cached_obj, dict) and "scores" in cached_obj else cached_obj

    # 2. Select deterministic sample of at least 64 CAL query-doc pairs
    selected_pairs: List[Tuple[str, str]] = []
    for qid in sorted(all_ids):
        cand_docs = extended[qid]
        for did in cand_docs:
            if did in frozen_scores.get(qid, {}):
                selected_pairs.append((qid, did))
                if len(selected_pairs) >= sample_size_pairs:
                    break
        if len(selected_pairs) >= sample_size_pairs:
            break

    print(f"Selected {len(selected_pairs)} deterministic CAL query-document pairs for score semantics audit.", flush=True)

    # 3. Load shipped frozen Jina model (no adapter)
    print("Loading shipped frozen Jina cross-encoder on GPU...", flush=True)
    model, tok = load_shipped_frozen_jina(device="cuda")

    # 4. Compute forward logits, apply sigmoid, and compare with frozen cache
    doc_sections_cache: Dict[str, List[Any]] = {}
    computed_probabilities: List[float] = []
    cached_probabilities: List[float] = []
    raw_logits_list: List[float] = []
    audit_records: List[Dict[str, Any]] = []

    for qid, did in selected_pairs:
        q_text = queries[qid][0]
        if did not in doc_sections_cache:
            doc_sections_cache[did] = parse_document_into_sections(
                did, docs[did], max_chunk_words=220, overlap_words=60
            )
        secs = doc_sections_cache[did]
        chosen_secs = preselect_legal_sections(q_text, secs, count=2)

        # Forward pass on the 2 sections
        pair_texts = [(q_text, sec.text) for sec in chosen_secs]
        inputs = tok(
            pair_texts, padding=True, truncation=True, max_length=512, return_tensors="pt"
        ).to("cuda")

        with torch.no_grad():
            logits = model(**inputs).logits.view(-1).float()
            probs = torch.sigmoid(logits)

        sec_logits = logits.cpu().numpy().tolist()
        sec_probs = probs.cpu().numpy().tolist()

        # MAX aggregation over top-2 sections
        computed_doc_prob = float(max(sec_probs))
        # Find the logit of the section with the max probability
        max_idx = int(np.argmax(sec_probs))
        computed_doc_logit = float(sec_logits[max_idx])

        cached_doc_prob = float(frozen_scores[qid][did])

        computed_probabilities.append(computed_doc_prob)
        cached_probabilities.append(cached_doc_prob)
        raw_logits_list.append(computed_doc_logit)

        audit_records.append({
            "qid": qid,
            "did": did,
            "raw_logit": computed_doc_logit,
            "computed_sigmoid": computed_doc_prob,
            "frozen_cached_score": cached_doc_prob,
            "abs_error": abs(computed_doc_prob - cached_doc_prob),
        })

    del model, tok
    torch.cuda.empty_cache()

    # 5. Compute audit statistics
    comp_arr = np.array(computed_probabilities, dtype=np.float64)
    cached_arr = np.array(cached_probabilities, dtype=np.float64)
    abs_errors = np.abs(comp_arr - cached_arr)

    max_abs_error = float(np.max(abs_errors))
    mean_abs_error = float(np.mean(abs_errors))
    spearman_corr, _ = scipy.stats.spearmanr(comp_arr, cached_arr)
    pearson_corr, _ = scipy.stats.pearsonr(comp_arr, cached_arr)

    print(f"Max Absolute Error:  {max_abs_error:.8e} (Threshold: <= 2e-3)")
    print(f"Mean Absolute Error: {mean_abs_error:.8e}")
    print(f"Spearman Correlation: {spearman_corr:.8f} (Threshold: >= 0.999)")
    print(f"Pearson Correlation:  {pearson_corr:.8f}")

    error_threshold = 2e-3
    spearman_threshold = 0.999

    audit_passed = (max_abs_error <= error_threshold) and (spearman_corr >= spearman_threshold)

    result = {
        "schema_version": "dsc2026.gemini.huy_d1_adapted_section_channel_public_v1.score_semantics_audit.v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_info["head_commit"],
        "status": "PASS" if audit_passed else "BLOCKED_SECTION_SCORE_SEMANTICS",
        "sample_size_pairs": len(selected_pairs),
        "error_threshold": error_threshold,
        "spearman_threshold": spearman_threshold,
        "max_absolute_error": max_abs_error,
        "mean_absolute_error": mean_abs_error,
        "spearman_correlation": float(spearman_corr),
        "pearson_correlation": float(pearson_corr),
        "audit_passed": bool(audit_passed),
        "score_semantics_rule": "adapted_probability = sigmoid(adapted_raw_logit) (compatible within predefined tolerance: MAE <= 2e-3, Spearman >= 0.999)",
        "sample_records": audit_records[:10],
    }

    audit_path = RESULTS_DIR / "SECTION_SCORE_SEMANTICS_AUDIT.json"
    audit_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Wrote {audit_path}", flush=True)

    if not audit_passed:
        raise RuntimeError(
            f"BLOCKED_SECTION_SCORE_SEMANTICS: Frozen cache is not compatible within predefined tolerance with sigmoid(raw_logit)! "
            f"Max error={max_abs_error:.6e}, Spearman={spearman_corr:.6f}"
        )

    print("Score Semantics Audit PASSED successfully (compatible within predefined tolerance).", flush=True)
    return result


if __name__ == "__main__":
    run_score_semantics_audit()
