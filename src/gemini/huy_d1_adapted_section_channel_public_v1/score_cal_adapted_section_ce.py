"""Stage 3: Compute adapted Section CE probabilities on CAL600 candidates (strictly label-free)."""

from __future__ import annotations

import hashlib
import json
import pickle
import time
from datetime import timezone, datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from .common import (
    ADAPTER_DIR,
    RESULTS_DIR,
    SOURCE_DIR,
    compute_fingerprint,
    get_git_status,
    get_source_files_sha256,
    load_adapted_jina_model,
    load_cal_data_label_free,
    seed_everything,
    sha256_file,
)
from .legal_section_parser import LegalSection, parse_document_into_sections, preselect_legal_sections

CAL_ADAPTED_CACHE_PATH = RESULTS_DIR / "adapted_section_ce_cal.pkl"
CAL_ADAPTED_MANIFEST_PATH = RESULTS_DIR / "adapted_section_ce_cal_manifest.json"


def score_cal_adapted_section_ce(
    batch_size: int = 64,
    force_fresh: bool = True,
    cal_data: Optional[Tuple] = None,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Any], str, str]:
    print("=== STAGE 3: SCORE CAL600 WITH ADAPTED JINA PASSAGE CE (LABEL-FREE) ===", flush=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    seed_everything(2026)
    git_info = get_git_status()

    # 1. Load CAL dataset WITHOUT labels
    if cal_data is None:
        print("Loading CAL data (strictly label-free)...", flush=True)
        (
            docs,
            queries,
            blocks,
            all_ids,
            extended,
            local_views,
            full_channels_cv,
            type_rows,
            cite_rows,
        ) = load_cal_data_label_free()
    else:
        (
            docs,
            queries,
            blocks,
            all_ids,
            extended,
            local_views,
            full_channels_cv,
            type_rows,
            cite_rows,
        ) = cal_data

    # Fingerprints & provenance hashes
    all_cand_docs = sorted({d for q in all_ids for d in extended[q]})
    doc_store_fingerprint = compute_fingerprint([f"{d}:{len(docs[d])}" for d in all_cand_docs])
    q_fingerprint = compute_fingerprint([f"{q}:{queries[q][0]}" for q in all_ids])
    cand_fingerprint = compute_fingerprint([f"{q}:{','.join(sorted(extended[q]))}" for q in all_ids])
    adapter_model_sha = sha256_file(ADAPTER_DIR / "adapter_model.safetensors")
    adapter_config_sha = sha256_file(ADAPTER_DIR / "adapter_config.json")
    parser_sha = sha256_file(SOURCE_DIR / "legal_section_parser.py")
    source_files_sha = get_source_files_sha256()

    contract = {
        "git_commit": git_info["head_commit"],
        "adapter_model_sha256": adapter_model_sha,
        "adapter_config_sha256": adapter_config_sha,
        "parser_sha256": parser_sha,
        "query_fingerprint": q_fingerprint,
        "candidate_pool_fingerprint": cand_fingerprint,
        "doc_store_fingerprint": doc_store_fingerprint,
        "score_semantics": "adapted_probability = sigmoid(adapted_raw_logit)",
        "aggregation": "MAX",
        "count_sections": 2,
        "max_chunk_words": 220,
        "overlap_words": 60,
    }

    # Check existing cache (only allowed if force_fresh is False and all provenance checks match)
    if not force_fresh and CAL_ADAPTED_CACHE_PATH.exists() and CAL_ADAPTED_MANIFEST_PATH.exists():
        try:
            cached_data = pickle.loads(CAL_ADAPTED_CACHE_PATH.read_bytes())
            cached_manifest = json.loads(CAL_ADAPTED_MANIFEST_PATH.read_text(encoding="utf-8"))
            if (
                cached_manifest.get("git_commit") == git_info["head_commit"]
                and cached_manifest.get("adapter_model_sha256") == adapter_model_sha
                and cached_manifest.get("parser_sha256") == parser_sha
                and cached_manifest.get("query_fingerprint") == q_fingerprint
                and cached_manifest.get("candidate_pool_fingerprint") == cand_fingerprint
                and cached_manifest.get("doc_store_fingerprint") == doc_store_fingerprint
                and cached_manifest.get("score_semantics") == contract["score_semantics"]
                and cached_manifest.get("total_queries_scored") == len(all_ids)
                and cached_manifest.get("total_candidate_pairs") == 23532
            ):
                cache_sha = sha256_file(CAL_ADAPTED_CACHE_PATH)
                seal_time = cached_manifest.get("seal_timestamp_utc", datetime.now(timezone.utc).isoformat())
                print(f"Loaded existing valid adapted score cache from {CAL_ADAPTED_CACHE_PATH}", flush=True)
                return cached_data["scores"], cached_manifest, cache_sha, seal_time
        except Exception as e:
            print(f"Warning: could not load existing cache: {e}. Re-scoring fresh...", flush=True)

    # 2. Build pairs for scoring (strictly zero FALLBACK_ID, zero pseudo-sections)
    print("Parsing candidate documents into structured legal sections...", flush=True)
    doc_sections_cache: Dict[str, List[Any]] = {}
    pair_records: List[Tuple[str, str, int, str, str]] = []  # (qid, did, sec_idx, q_text, sec_text)

    total_candidate_pairs = 0
    empty_section_violations: List[Tuple[str, str]] = []

    for qid in sorted(all_ids):
        q_text = queries[qid][0]
        cands = extended[qid]
        total_candidate_pairs += len(cands)
        for did in cands:
            if did not in doc_sections_cache:
                d_text = docs[did]
                secs = parse_document_into_sections(
                    did, d_text, max_chunk_words=220, overlap_words=60
                )
                if not secs:
                    empty_section_violations.append((qid, did))
                doc_sections_cache[did] = secs
            secs = doc_sections_cache[did]
            if not secs:
                empty_section_violations.append((qid, did))
                continue
            chosen_secs = preselect_legal_sections(q_text, secs, count=2)
            for s_idx, sec in enumerate(chosen_secs):
                pair_records.append((qid, did, s_idx, q_text, sec.text))

    if empty_section_violations:
        raise RuntimeError(
            f"BLOCKED_EMPTY_DOCUMENT_SECTION_COVERAGE: {len(empty_section_violations)} candidate pairs yielded 0 sections! "
            f"Violations (first 10): {empty_section_violations[:10]}"
        )

    print(
        f"Total candidate pairs: {total_candidate_pairs} across {len(all_ids)} queries (Empty: 0). "
        f"Generated {len(pair_records)} section pairs to score on GPU.",
        flush=True,
    )

    # 3. Load adapted Jina model
    print("Loading adapted Jina LoRA model on GPU...", flush=True)
    adapted_model, base_model, tok = load_adapted_jina_model(device="cuda")
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()

    # 4. Batch forward scoring
    raw_logits_all: List[float] = []
    probs_all: List[float] = []

    for i in range(0, len(pair_records), batch_size):
        batch = pair_records[i : i + batch_size]
        text_pairs = [(r[3], r[4]) for r in batch]
        inputs = tok(
            text_pairs, padding=True, truncation=True, max_length=512, return_tensors="pt"
        ).to("cuda")

        with torch.no_grad():
            logits = adapted_model(**inputs).logits.view(-1).float()
            probs = torch.sigmoid(logits)

        raw_logits_all.extend(logits.cpu().numpy().tolist())
        probs_all.extend(probs.cpu().numpy().tolist())

        if (i // batch_size + 1) % 100 == 0 or i + batch_size >= len(pair_records):
            elapsed = time.perf_counter() - t0
            cur_vram = torch.cuda.max_memory_allocated() / (1024 * 1024)
            pct = min(100.0, (i + len(batch)) / len(pair_records) * 100.0)
            print(
                f"[{i + len(batch)}/{len(pair_records)}] ({pct:.1f}%) "
                f"Elapsed: {elapsed:.1f}s, Peak VRAM: {cur_vram:.1f} MB",
                flush=True,
            )

    elapsed_total = time.perf_counter() - t0
    peak_vram = torch.cuda.max_memory_allocated() / (1024 * 1024)
    del adapted_model, base_model, tok
    torch.cuda.empty_cache()

    # 5. Aggregate into document-level scores (MAX over top-2 sections)
    adapted_scores: Dict[str, Dict[str, float]] = {q: {} for q in all_ids}
    adapted_raw_logits: Dict[str, Dict[str, float]] = {q: {} for q in all_ids}

    for (qid, did, s_idx, _, _), logit_val, prob_val in zip(pair_records, raw_logits_all, probs_all):
        cur_prob = adapted_scores[qid].get(did, -1e9)
        if prob_val > cur_prob:
            adapted_scores[qid][did] = float(prob_val)
            adapted_raw_logits[qid][did] = float(logit_val)

    # Validate 100% coverage
    for qid in all_ids:
        for did in extended[qid]:
            if did not in adapted_scores[qid]:
                raise RuntimeError(f"Candidate document {did} missing score for query {qid}!")

    # 6. Save cache and seal
    cache_payload = {
        "scores": adapted_scores,
        "raw_logits": adapted_raw_logits,
        "contract": contract,
        "total_queries": len(all_ids),
        "total_candidate_pairs": total_candidate_pairs,
        "total_section_pairs_scored": len(pair_records),
    }
    CAL_ADAPTED_CACHE_PATH.write_bytes(pickle.dumps(cache_payload))
    cache_sha256 = sha256_file(CAL_ADAPTED_CACHE_PATH)
    seal_timestamp_utc = datetime.now(timezone.utc).isoformat()
    print(f"Saved and sealed {CAL_ADAPTED_CACHE_PATH} (SHA256: {cache_sha256}) at {seal_timestamp_utc}", flush=True)

    manifest_payload = {
        **contract,
        "cache_path": str(CAL_ADAPTED_CACHE_PATH.relative_to(RESULTS_DIR.parent.parent)).replace("\\", "/"),
        "cache_sha256": cache_sha256,
        "cache_size_bytes": CAL_ADAPTED_CACHE_PATH.stat().st_size,
        "seal_timestamp_utc": seal_timestamp_utc,
        "fresh_final_run": True,
        "reused_candidate_scores": False,
        "total_queries_scored": len(all_ids),
        "total_candidate_pairs": total_candidate_pairs,
        "total_section_pairs_scored": len(pair_records),
        "empty_section_count": 0,
        "synthetic_fallback_count": 0,
        "elapsed_seconds": elapsed_total,
        "peak_vram_mb": peak_vram,
        "source_files_sha256": source_files_sha,
        "score_distribution": {
            "mean_prob": float(np.mean([v for q in adapted_scores for v in adapted_scores[q].values()])),
            "std_prob": float(np.std([v for q in adapted_scores for v in adapted_scores[q].values()])),
            "min_prob": float(np.min([v for q in adapted_scores for v in adapted_scores[q].values()])),
            "max_prob": float(np.max([v for q in adapted_scores for v in adapted_scores[q].values()])),
            "mean_raw_logit": float(np.mean([v for q in adapted_raw_logits for v in adapted_raw_logits[q].values()])),
            "std_raw_logit": float(np.std([v for q in adapted_raw_logits for v in adapted_raw_logits[q].values()])),
        },
    }
    CAL_ADAPTED_MANIFEST_PATH.write_text(json.dumps(manifest_payload, indent=2), encoding="utf-8")
    print(f"Saved {CAL_ADAPTED_MANIFEST_PATH}", flush=True)

    return adapted_scores, manifest_payload, cache_sha256, seal_timestamp_utc


if __name__ == "__main__":
    score_cal_adapted_section_ce(force_fresh=True)
