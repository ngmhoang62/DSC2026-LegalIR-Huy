"""Stage 4: Compute adapted Section CE probabilities on CAL600 candidates (strictly label-free)."""

from __future__ import annotations

import hashlib
import json
import pickle
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from .common import (
    ADAPTER_DIR,
    RESULTS_DIR,
    compute_fingerprint,
    get_git_status,
    load_adapted_jina_model,
    load_cal_data,
    seed_everything,
    sha256_file,
)
from .legal_section_parser import LegalSection, parse_document_into_sections, preselect_legal_sections

CAL_ADAPTED_CACHE_PATH = RESULTS_DIR / "adapted_section_ce_cal.pkl"
CAL_ADAPTED_MANIFEST_PATH = RESULTS_DIR / "adapted_section_ce_cal_manifest.json"


def score_cal_adapted_section_ce(
    batch_size: int = 64,
    force_rescore: bool = False,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Any]]:
    print("=== STAGE 4: SCORE CAL600 WITH ADAPTED JINA PASSAGE CE ===", flush=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    seed_everything(2026)
    git_info = get_git_status()

    # 1. Load CAL dataset WITHOUT labels
    print("Loading CAL data (label-free)...", flush=True)
    docs, queries, blocks, all_ids, extended, local_views, full_channels_cv, _, type_rows, cite_rows = load_cal_data()

    # Fingerprints
    q_fingerprint = compute_fingerprint([f"{q}:{queries[q][0]}" for q in all_ids])
    cand_fingerprint = compute_fingerprint([f"{q}:{','.join(sorted(extended[q]))}" for q in all_ids])
    adapter_model_sha = sha256_file(ADAPTER_DIR / "adapter_model.safetensors")
    parser_sha = sha256_file(Path(__file__).parent / "legal_section_parser.py")

    contract = {
        "git_commit": git_info["head_commit"],
        "adapter_model_sha256": adapter_model_sha,
        "parser_sha256": parser_sha,
        "query_fingerprint": q_fingerprint,
        "candidate_pool_fingerprint": cand_fingerprint,
        "score_semantics": "adapted_probability = sigmoid(adapted_raw_logit)",
        "aggregation": "MAX",
        "count_sections": 2,
        "max_chunk_words": 220,
        "overlap_words": 60,
    }

    # Check existing cache
    if not force_rescore and CAL_ADAPTED_CACHE_PATH.exists() and CAL_ADAPTED_MANIFEST_PATH.exists():
        try:
            cached_data = pickle.loads(CAL_ADAPTED_CACHE_PATH.read_bytes())
            cached_manifest = json.loads(CAL_ADAPTED_MANIFEST_PATH.read_text(encoding="utf-8"))
            if (
                cached_manifest.get("adapter_model_sha256") == adapter_model_sha
                and cached_manifest.get("query_fingerprint") == q_fingerprint
                and cached_manifest.get("candidate_pool_fingerprint") == cand_fingerprint
                and cached_manifest.get("score_semantics") == contract["score_semantics"]
                and cached_manifest.get("total_queries_scored") == len(all_ids)
            ):
                print(f"Loaded existing valid adapted score cache from {CAL_ADAPTED_CACHE_PATH}", flush=True)
                return cached_data["scores"], cached_manifest
        except Exception as e:
            print(f"Warning: could not load existing cache: {e}. Re-scoring...", flush=True)

    # 2. Build pairs for scoring
    print("Parsing candidate documents into structured legal sections...", flush=True)
    doc_sections_cache: Dict[str, List[Any]] = {}
    pair_records: List[Tuple[str, str, int, str, str]] = []  # (qid, did, sec_idx, q_text, sec_text)

    total_candidate_pairs = 0
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
                    secs = [
                        LegalSection(
                            doc_id=did,
                            section_index=0,
                            section_type="FALLBACK_ID",
                            heading=did,
                            text=did,
                            word_count=1,
                        )
                    ]
                doc_sections_cache[did] = secs
            secs = doc_sections_cache[did]
            chosen_secs = preselect_legal_sections(q_text, secs, count=2)
            for s_idx, sec in enumerate(chosen_secs):
                pair_records.append((qid, did, s_idx, q_text, sec.text))

    print(
        f"Total candidate pairs: {total_candidate_pairs} across {len(all_ids)} queries. "
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

    # 6. Save cache and manifest
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
    print(f"Saved {CAL_ADAPTED_CACHE_PATH} (SHA256: {cache_sha256})", flush=True)

    manifest_payload = {
        **contract,
        "cache_path": str(CAL_ADAPTED_CACHE_PATH.relative_to(RESULTS_DIR.parent.parent)).replace("\\", "/"),
        "cache_sha256": cache_sha256,
        "cache_size_bytes": CAL_ADAPTED_CACHE_PATH.stat().st_size,
        "total_queries_scored": len(all_ids),
        "total_candidate_pairs": total_candidate_pairs,
        "total_section_pairs_scored": len(pair_records),
        "elapsed_seconds": elapsed_total,
        "peak_vram_mb": peak_vram,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
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

    return adapted_scores, manifest_payload


if __name__ == "__main__":
    score_cal_adapted_section_ce()
