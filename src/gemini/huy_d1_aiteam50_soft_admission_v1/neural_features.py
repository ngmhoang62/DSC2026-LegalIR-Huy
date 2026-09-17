"""Neural feature inference and parity verification for Jina-FT and Legal Section CE.

Module for HUY_D1_AITEAM50_SOFT_ADMISSION_V1.
"""

from __future__ import annotations

import json
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from benchmark_jina_reranker_holdouts import top_passages
from src.gemini.huy_d1_aiteam50_soft_admission_v1.common import (
    CAL_FROZEN_SECTION_CACHE_PATH,
    EXPECTED_JINA_FT_CV_SHA256,
    EXPECTED_SECTION_CE_SHA256,
    JINA_FT_CV_PATH,
    REPO_JINA,
    RESULTS_DIR,
    ROOT,
    WEIGHTS_JINA_FT,
    SafeDocumentStore,
    sha256_file,
)
from src.gemini.huy_d1_legal_section_evidence_v1.legal_section_parser import (
    parse_document_into_sections,
    preselect_legal_sections,
)


def patch_transformers_v5() -> None:
    import transformers.models.xlm_roberta.modeling_xlm_roberta as module

    if hasattr(module, "create_position_ids_from_input_ids"):
        return

    def helper(input_ids, padding_idx, past_key_values_length=0):
        mask = input_ids.ne(padding_idx).int()
        positions = (torch.cumsum(mask, dim=1) + past_key_values_length) * mask
        return positions.long() + padding_idx

    module.create_position_ids_from_input_ids = helper


def load_neural_crossencoder() -> Tuple[Any, Any, Dict[str, Any]]:
    """Load frozen Jina cross-encoder on GPU (shared by Jina-FT and Section CE)."""
    patch_transformers_v5()
    tok = AutoTokenizer.from_pretrained(
        REPO_JINA, trust_remote_code=True, fix_mistral_regex=True
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        REPO_JINA, trust_remote_code=True, dtype=torch.float16
    )
    state = load_file(WEIGHTS_JINA_FT)
    missing, unexpected = model.load_state_dict(
        {k: v.to(torch.float16) for k, v in state.items()}, strict=True
    )
    model._tokenizer = tok
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval().to(device)

    prov = {
        "repo_jina": str(REPO_JINA.relative_to(ROOT)).replace("\\", "/"),
        "weights_path": str(WEIGHTS_JINA_FT.relative_to(ROOT)).replace("\\", "/"),
        "weights_sha256": sha256_file(WEIGHTS_JINA_FT),
        "device": device,
        "dtype": "torch.float16",
        "missing_keys": len(missing),
        "unexpected_keys": len(unexpected),
    }
    return model, tok, prov


def compute_neural_features_for_universe(
    model: Any,
    queries_label_free: Dict[str, Tuple[str, Any]],
    docs: SafeDocumentStore,
    all_ids: List[str],
    s_universe: Dict[str, List[str]],
    d1_top5: Dict[str, List[str]],
    batch_size: int = 64,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]], Dict[str, Any], Dict[str, Any]]:
    """Computes Jina-FT and Section CE scores for all (q, d) in S(q), reusing cached values
    where available and performing fresh inference and parity audits on cached references.
    """
    print("=== COMPUTING NEURAL SCORES (JINA-FT + SECTION CE) OVER S(q) ===", flush=True)

    # 1. Load and hash check caches
    jina_sha = sha256_file(JINA_FT_CV_PATH)
    if jina_sha != EXPECTED_JINA_FT_CV_SHA256:
        raise RuntimeError(f"Jina cache SHA256 mismatch! {jina_sha} != {EXPECTED_JINA_FT_CV_SHA256}")

    sec_sha = sha256_file(CAL_FROZEN_SECTION_CACHE_PATH)
    if sec_sha != EXPECTED_SECTION_CE_SHA256:
        raise RuntimeError(f"Section CE cache SHA256 mismatch! {sec_sha} != {EXPECTED_SECTION_CE_SHA256}")

    cached_jina = pickle.loads(JINA_FT_CV_PATH.read_bytes())
    raw_sec = pickle.loads(CAL_FROZEN_SECTION_CACHE_PATH.read_bytes())
    cached_sec = raw_sec.get("scores", raw_sec)

    jina_scores: Dict[str, Dict[str, float]] = {q: {} for q in all_ids}
    sec_scores: Dict[str, Dict[str, float]] = {q: {} for q in all_ids}

    fresh_jina_pairs = []
    fresh_jina_owners = []

    fresh_sec_pairs = []
    fresh_sec_owners = []

    doc_sections_cache: Dict[str, List[Any]] = {}

    for q in all_ids:
        q_text = queries_label_free[q][0]
        u_docs = s_universe[q]

        for d in u_docs:
            # Jina-FT
            if q in cached_jina and d in cached_jina[q]:
                jina_scores[q][d] = float(cached_jina[q][d])
            else:
                passages = top_passages(q_text, docs[d], count=2)
                for p in passages:
                    fresh_jina_pairs.append((q_text, p))
                    fresh_jina_owners.append((q, d))

            # Section CE
            if q in cached_sec and d in cached_sec[q]:
                sec_scores[q][d] = float(cached_sec[q][d])
            else:
                if d not in doc_sections_cache:
                    doc_sections_cache[d] = parse_document_into_sections(d, docs[d])
                d_secs = doc_sections_cache[d]
                selected = preselect_legal_sections(q_text, d_secs, count=2)
                for sec in selected:
                    fresh_sec_pairs.append((q_text, sec.text))
                    fresh_sec_owners.append((q, d))

    print(f"Jina-FT: {sum(len(jina_scores[q]) for q in all_ids)} cached, {len(fresh_jina_pairs)} fresh pairs", flush=True)
    print(f"Section CE: {sum(len(sec_scores[q]) for q in all_ids)} cached, {len(fresh_sec_pairs)} fresh pairs", flush=True)

    # 2. Fresh inference for Jina-FT
    if fresh_jina_pairs:
        t0 = time.perf_counter()
        raw_jina_out = model.compute_score(fresh_jina_pairs, batch_size=batch_size, max_length=512)
        if isinstance(raw_jina_out, float):
            raw_jina_out = [raw_jina_out]
        for (q, d), s in zip(fresh_jina_owners, raw_jina_out):
            jina_scores[q][d] = max(jina_scores[q].get(d, -1e9), float(s))
        print(f"Completed fresh Jina-FT inference in {time.perf_counter() - t0:.2f}s", flush=True)

    # 3. Fresh inference for Section CE
    if fresh_sec_pairs:
        t0 = time.perf_counter()
        raw_sec_out = model.compute_score(fresh_sec_pairs, batch_size=batch_size, max_length=512)
        if isinstance(raw_sec_out, float):
            raw_sec_out = [raw_sec_out]
        for (q, d), s in zip(fresh_sec_owners, raw_sec_out):
            sec_scores[q][d] = max(sec_scores[q].get(d, -1e9), float(s))
        print(f"Completed fresh Section CE inference in {time.perf_counter() - t0:.2f}s", flush=True)

    # 4. Parity audit for Jina-FT on sample of cached D1 Top-5 candidates
    print("Running Jina-FT parity audit on sample of cached candidates...", flush=True)
    jina_sample_qids = [all_ids[i] for i in [0, 49, 99, 100, 149, 199, 200, 249, 299, 300, 399, 499, 549, 599][:15]]
    jina_sample_pairs = []
    jina_sample_owners = []

    for q in jina_sample_qids:
        q_text = queries_label_free[q][0]
        for d in d1_top5[q][:2]:
            passages = top_passages(q_text, docs[d], count=2)
            for p in passages:
                jina_sample_pairs.append((q_text, p))
                jina_sample_owners.append((q, d))

    jina_sample_raw = model.compute_score(jina_sample_pairs, batch_size=batch_size, max_length=512)
    if isinstance(jina_sample_raw, float):
        jina_sample_raw = [jina_sample_raw]

    jina_sample_rebuilt: Dict[str, Dict[str, float]] = {}
    for (q, d), s in zip(jina_sample_owners, jina_sample_raw):
        jina_sample_rebuilt.setdefault(q, {})[d] = max(jina_sample_rebuilt.setdefault(q, {}).get(d, -1e9), float(s))

    jina_diffs = []
    jina_comparisons = []
    for q in jina_sample_rebuilt:
        for d in jina_sample_rebuilt[q]:
            ref = float(cached_jina[q][d])
            rebuilt = float(jina_sample_rebuilt[q][d])
            d_val = abs(ref - rebuilt)
            jina_diffs.append(d_val)
            jina_comparisons.append({
                "qid": q,
                "docid": d,
                "reference": ref,
                "rebuilt": rebuilt,
                "abs_diff": d_val,
            })

    jina_mean_diff = float(np.mean(jina_diffs)) if jina_diffs else 0.0
    jina_max_diff = float(np.max(jina_diffs)) if jina_diffs else 0.0
    jina_parity_pass = jina_max_diff <= 0.01

    jina_parity_doc = {
        "schema_version": "dsc2026.gemini.huy_d1_aiteam50_soft_admission_v1.jina_parity.v1",
        "experiment_id": "HUY_D1_AITEAM50_SOFT_ADMISSION_V1",
        "cached_jina_path": str(JINA_FT_CV_PATH.relative_to(ROOT)).replace("\\", "/"),
        "cached_jina_sha256": jina_sha,
        "sample_pairs_count": len(jina_diffs),
        "mean_abs_diff": jina_mean_diff,
        "max_abs_diff": jina_max_diff,
        "tolerance": 0.01,
        "parity_pass": jina_parity_pass,
        "status": "PASS" if jina_parity_pass else "FAIL",
        "comparisons": jina_comparisons,
    }

    out_jina = RESULTS_DIR / "JINA_FT_INFERENCE_PARITY.json"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_jina.write_text(json.dumps(jina_parity_doc, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {out_jina} (Parity: {jina_parity_doc['status']}, max_diff: {jina_max_diff:.6f})", flush=True)

    if not jina_parity_pass:
        raise RuntimeError(f"Jina parity failed! max_diff={jina_max_diff} > 0.01")

    # 5. Parity audit for Section CE on sample of cached D1 Top-5 candidates
    print("Running Section CE parity audit on sample of cached candidates...", flush=True)
    sec_sample_pairs = []
    sec_sample_owners = []

    for q in jina_sample_qids:
        q_text = queries_label_free[q][0]
        for d in d1_top5[q][:2]:
            if d not in doc_sections_cache:
                doc_sections_cache[d] = parse_document_into_sections(d, docs[d])
            d_secs = doc_sections_cache[d]
            selected = preselect_legal_sections(q_text, d_secs, count=2)
            for sec in selected:
                sec_sample_pairs.append((q_text, sec.text))
                sec_sample_owners.append((q, d))

    sec_sample_raw = model.compute_score(sec_sample_pairs, batch_size=batch_size, max_length=512)
    if isinstance(sec_sample_raw, float):
        sec_sample_raw = [sec_sample_raw]

    sec_sample_rebuilt: Dict[str, Dict[str, float]] = {}
    for (q, d), s in zip(sec_sample_owners, sec_sample_raw):
        sec_sample_rebuilt.setdefault(q, {})[d] = max(sec_sample_rebuilt.setdefault(q, {}).get(d, -1e9), float(s))

    sec_diffs = []
    sec_comparisons = []
    for q in sec_sample_rebuilt:
        for d in sec_sample_rebuilt[q]:
            ref = float(cached_sec[q][d])
            rebuilt = float(sec_sample_rebuilt[q][d])
            d_val = abs(ref - rebuilt)
            sec_diffs.append(d_val)
            sec_comparisons.append({
                "qid": q,
                "docid": d,
                "reference": ref,
                "rebuilt": rebuilt,
                "abs_diff": d_val,
            })

    sec_mean_diff = float(np.mean(sec_diffs)) if sec_diffs else 0.0
    sec_max_diff = float(np.max(sec_diffs)) if sec_diffs else 0.0
    sec_parity_pass = sec_max_diff <= 0.01

    sec_parity_doc = {
        "schema_version": "dsc2026.gemini.huy_d1_aiteam50_soft_admission_v1.section_parity.v1",
        "experiment_id": "HUY_D1_AITEAM50_SOFT_ADMISSION_V1",
        "cached_section_ce_path": str(CAL_FROZEN_SECTION_CACHE_PATH.relative_to(ROOT)).replace("\\", "/"),
        "cached_section_ce_sha256": sec_sha,
        "sample_pairs_count": len(sec_diffs),
        "mean_abs_diff": sec_mean_diff,
        "max_abs_diff": sec_max_diff,
        "tolerance": 0.01,
        "parity_pass": sec_parity_pass,
        "status": "PASS" if sec_parity_pass else "FAIL",
        "comparisons": sec_comparisons,
    }

    out_sec = RESULTS_DIR / "SECTION_INFERENCE_PARITY.json"
    out_sec.write_text(json.dumps(sec_parity_doc, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {out_sec} (Parity: {sec_parity_doc['status']}, max_diff: {sec_max_diff:.6f})", flush=True)

    if not sec_parity_pass:
        raise RuntimeError(f"Section CE parity failed! max_diff={sec_max_diff} > 0.01")

    return jina_scores, sec_scores, jina_parity_doc, sec_parity_doc
