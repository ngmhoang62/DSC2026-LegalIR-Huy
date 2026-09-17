"""Inference and parity verification for frozen Jina-FT and Legal Section CE experts."""

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
from src.gemini.huy_d1_aiteam_novel_consensus_v1.common import (
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


def load_jina_crossencoder() -> Tuple[Any, Any, Dict[str, Any]]:
    """Load frozen Jina cross-encoder with fine-tuned weights on GPU."""
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


def score_jina_universe(
    model: Any,
    queries_label_free: Dict[str, Tuple[str, Any]],
    docs: SafeDocumentStore,
    all_ids: List[str],
    d1_top5: Dict[str, List[str]],
    novel_map: Dict[str, List[str]],
    batch_size: int = 16,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Any]]:
    """Freshly scores all documents in U(q) = D1 Top-5 + NOVEL(q) via Jina-FT."""
    print("--- INFERENCE: SCORING UNIVERSE U(q) VIA FROZEN JINA-FT ---", flush=True)
    t0 = time.perf_counter()

    universe_scores: Dict[str, Dict[str, float]] = {q: {} for q in all_ids}

    # Collect pairs to score
    all_pairs = []
    pair_owners = []

    for q in all_ids:
        q_text = queries_label_free[q][0]
        u_docs = list(dict.fromkeys(d1_top5[q] + novel_map[q]))

        for d in u_docs:
            passages = top_passages(q_text, docs[d], count=2)
            for p in passages:
                all_pairs.append((q_text, p))
                pair_owners.append((q, d))

    print(f"Total Jina forward pairs across CAL600: {len(all_pairs)}", flush=True)

    raw_scores = []
    if all_pairs:
        raw_scores = model.compute_score(all_pairs, batch_size=batch_size, max_length=512)
        if isinstance(raw_scores, float):
            raw_scores = [raw_scores]

    for (q, d), s in zip(pair_owners, raw_scores):
        universe_scores[q][d] = max(universe_scores[q].get(d, -1e9), float(s))

    t_elapsed = time.perf_counter() - t0
    print(f"Completed Jina inference in {t_elapsed:.2f}s", flush=True)

    # Verify parity against existing cached Jina-FT values for D1 Top-5
    cached_jina = pickle.loads(JINA_FT_CV_PATH.read_bytes())
    diffs = []
    comparison_pairs = []

    for q in all_ids:
        for d in d1_top5[q]:
            if q in cached_jina and d in cached_jina[q]:
                ref = float(cached_jina[q][d])
                rebuilt = float(universe_scores[q][d])
                d_val = abs(ref - rebuilt)
                diffs.append(d_val)
                comparison_pairs.append({
                    "qid": q,
                    "docid": d,
                    "reference": ref,
                    "rebuilt": rebuilt,
                    "abs_diff": d_val,
                })

    mean_diff = float(np.mean(diffs)) if diffs else 0.0
    max_diff = float(np.max(diffs)) if diffs else 0.0
    parity_pass = max_diff <= 0.01  # fp16 deterministic tolerance

    parity_doc = {
        "schema_version": "dsc2026.gemini.huy_d1_aiteam_novel_consensus_v1.jina_parity.v1",
        "experiment_id": "HUY_D1_AITEAM_NOVEL_CONSENSUS_V1",
        "cached_jina_path": str(JINA_FT_CV_PATH.relative_to(ROOT)).replace("\\", "/"),
        "cached_jina_sha256": sha256_file(JINA_FT_CV_PATH),
        "comparison_pairs_count": len(diffs),
        "mean_abs_diff": mean_diff,
        "max_abs_diff": max_diff,
        "tolerance": 0.01,
        "parity_pass": parity_pass,
        "status": "PASS" if parity_pass else "FAIL",
        "sample_comparisons": comparison_pairs[:20],
    }

    out_jina = RESULTS_DIR / "JINA_FT_INFERENCE_PARITY.json"
    out_jina.write_text(json.dumps(parity_doc, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {out_jina} (Parity: {parity_doc['status']}, max_diff: {max_diff:.6f})", flush=True)

    return universe_scores, parity_doc


def score_section_universe(
    model: Any,
    queries_label_free: Dict[str, Tuple[str, Any]],
    docs: SafeDocumentStore,
    all_ids: List[str],
    d1_top5: Dict[str, List[str]],
    novel_map: Dict[str, List[str]],
    batch_size: int = 16,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Any]]:
    """Scores novel documents via Section CE and combines with frozen cached D1 scores."""
    print("--- INFERENCE: SCORING NOVEL VIA LEGAL SECTION CE & PARITY AUDIT ---", flush=True)
    t0 = time.perf_counter()

    sec_sha = sha256_file(CAL_FROZEN_SECTION_CACHE_PATH)
    if sec_sha != EXPECTED_SECTION_CE_SHA256:
        raise RuntimeError(f"Section CE cache mismatch! {sec_sha} != {EXPECTED_SECTION_CE_SHA256}")

    cached_raw = pickle.loads(CAL_FROZEN_SECTION_CACHE_PATH.read_bytes())
    cached_sec = cached_raw.get("scores", cached_raw)

    universe_scores: Dict[str, Dict[str, float]] = {q: {} for q in all_ids}

    # For existing D1 Top-5 documents: use frozen cached scores
    for q in all_ids:
        for d in d1_top5[q]:
            if d in cached_sec.get(q, {}):
                universe_scores[q][d] = float(cached_sec[q][d])
            else:
                raise RuntimeError(f"Missing cached Section CE score for D1 Top-5 doc {d} in query {q}")

    # For novel documents: fresh Section CE inference
    doc_sections_cache: Dict[str, List[Any]] = {}
    pair_records = []

    for q in all_ids:
        q_text = queries_label_free[q][0]
        for d in novel_map[q]:
            if d not in doc_sections_cache:
                doc_sections_cache[d] = parse_document_into_sections(d, docs[d])
            d_secs = doc_sections_cache[d]
            selected = preselect_legal_sections(q_text, d_secs, count=2)
            for sec in selected:
                pair_records.append((q, d, q_text, sec.text))

    print(f"Total novel Section CE pairs across CAL600: {len(pair_records)}", flush=True)

    if pair_records:
        pairs = [(r[2], r[3]) for r in pair_records]
        raw_scores = model.compute_score(pairs, batch_size=batch_size, max_length=512)
        if isinstance(raw_scores, float):
            raw_scores = [raw_scores]

        for (q, d, _, _), s_val in zip(pair_records, raw_scores):
            universe_scores[q][d] = max(universe_scores[q].get(d, -1e9), float(s_val))

    t_elapsed = time.perf_counter() - t0
    print(f"Completed Section CE novel scoring in {t_elapsed:.2f}s", flush=True)

    # Section CE Parity Audit on deterministic sample of cached D1 candidates (30 pairs)
    print("Running Section CE parity audit on sample of 30 cached candidates...", flush=True)
    sample_qids = [all_ids[i] for i in [0, 49, 99, 100, 149, 199, 200, 249, 299, 300, 399, 499, 549, 599][:15]]
    sample_pairs = []
    sample_owners = []

    for q in sample_qids:
        q_text = queries_label_free[q][0]
        for d in d1_top5[q][:2]:  # 2 candidates per sample query = 30 pairs
            if d not in doc_sections_cache:
                doc_sections_cache[d] = parse_document_into_sections(d, docs[d])
            d_secs = doc_sections_cache[d]
            selected = preselect_legal_sections(q_text, d_secs, count=2)
            for sec in selected:
                sample_pairs.append((q_text, sec.text))
                sample_owners.append((q, d))

    raw_sample = model.compute_score(sample_pairs, batch_size=batch_size, max_length=512)
    if isinstance(raw_sample, float):
        raw_sample = [raw_sample]

    rebuilt_sample: Dict[str, Dict[str, float]] = {}
    for (q, d), s_val in zip(sample_owners, raw_sample):
        rebuilt_sample.setdefault(q, {})[d] = max(rebuilt_sample.setdefault(q, {}).get(d, -1e9), float(s_val))

    sec_diffs = []
    sec_comparisons = []
    for q in rebuilt_sample:
        for d in rebuilt_sample[q]:
            ref = float(cached_sec[q][d])
            rebuilt = float(rebuilt_sample[q][d])
            d_val = abs(ref - rebuilt)
            sec_diffs.append(d_val)
            sec_comparisons.append({
                "qid": q,
                "docid": d,
                "reference": ref,
                "rebuilt": rebuilt,
                "abs_diff": d_val,
            })

    mean_sec_diff = float(np.mean(sec_diffs)) if sec_diffs else 0.0
    max_sec_diff = float(np.max(sec_diffs)) if sec_diffs else 0.0
    sec_parity_pass = max_sec_diff <= 0.01

    sec_parity_doc = {
        "schema_version": "dsc2026.gemini.huy_d1_aiteam_novel_consensus_v1.section_parity.v1",
        "experiment_id": "HUY_D1_AITEAM_NOVEL_CONSENSUS_V1",
        "cached_section_ce_path": str(CAL_FROZEN_SECTION_CACHE_PATH.relative_to(ROOT)).replace("\\", "/"),
        "cached_section_ce_sha256": sec_sha,
        "sample_pairs_count": len(sec_diffs),
        "mean_abs_diff": mean_sec_diff,
        "max_abs_diff": max_sec_diff,
        "tolerance": 0.01,
        "parity_pass": sec_parity_pass,
        "status": "PASS" if sec_parity_pass else "FAIL",
        "comparisons": sec_comparisons,
    }

    out_sec = RESULTS_DIR / "SECTION_INFERENCE_PARITY.json"
    out_sec.write_text(json.dumps(sec_parity_doc, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {out_sec} (Parity: {sec_parity_doc['status']}, max_diff: {max_sec_diff:.6f})", flush=True)

    return universe_scores, sec_parity_doc
