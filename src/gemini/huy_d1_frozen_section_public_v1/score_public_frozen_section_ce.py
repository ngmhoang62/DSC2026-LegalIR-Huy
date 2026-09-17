"""Stage 4: Score all 39,233 public candidate pairs using frozen Jina Cross-Encoder on structured legal sections."""

from __future__ import annotations

import json
import pickle
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch

from .common import (
    PUB_FROZEN_SECTION_CACHE_PATH,
    PUB_FROZEN_SECTION_MANIFEST_PATH,
    REPO_JINA,
    RESULTS_DIR,
    ROOT,
    SRC_DIR,
    WEIGHTS_JINA_FT,
    compute_candidate_fingerprint,
    compute_query_fingerprint,
    get_git_status,
    get_source_files_sha256,
    load_jina_crossencoder,
    seed_everything,
    sha256_file,
)
from .legal_section_parser import LegalSection, parse_document_into_sections, preselect_legal_sections


def score_public_frozen_section_ce(
    public_bundle: Dict[str, Any],
    batch_size: int = 64,
    force_fresh: bool = True,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Any], str, str]:
    print("\n=== STAGE 4: SCORE PUBLIC CANDIDATES WITH FROZEN JINA CE ===", flush=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    seed_everything(2026)
    git_info = get_git_status()

    docs_store = public_bundle["docs_store"]
    public_ids = public_bundle["public_ids"]
    public_candidates = public_bundle["public_candidates"]
    public_meta = public_bundle["public_meta"]
    pub_query_fp = public_bundle["pub_query_fp"]
    pub_cand_fp = public_bundle["pub_cand_fp"]

    total_public_queries = len(public_ids)
    total_candidate_pairs = sum(len(public_candidates[q]) for q in public_ids)
    print(f"Total public queries: {total_public_queries}, Total candidate pairs: {total_candidate_pairs}", flush=True)

    parser_path = SRC_DIR / "legal_section_parser.py"
    parser_sha256 = sha256_file(parser_path)
    weights_sha256 = sha256_file(WEIGHTS_JINA_FT)

    # Check if existing cache can be reused (strictly guarded)
    if not force_fresh and PUB_FROZEN_SECTION_CACHE_PATH.exists() and PUB_FROZEN_SECTION_MANIFEST_PATH.exists():
        try:
            cached_manifest = json.loads(PUB_FROZEN_SECTION_MANIFEST_PATH.read_text(encoding="utf-8"))
            if (
                cached_manifest.get("weights_sha256") == weights_sha256
                and cached_manifest.get("parser_sha256") == parser_sha256
                and cached_manifest.get("public_query_fingerprint") == pub_query_fp
                and cached_manifest.get("public_candidate_pool_fingerprint") == pub_cand_fp
                and cached_manifest.get("total_candidate_pairs") == total_candidate_pairs
                and cached_manifest.get("git_commit") == git_info["head_commit"]
            ):
                print(f"Valid provenance-matched cache found at {PUB_FROZEN_SECTION_CACHE_PATH}. Reusing...", flush=True)
                cached_data = pickle.loads(PUB_FROZEN_SECTION_CACHE_PATH.read_bytes())
                scores = cached_data["scores"] if isinstance(cached_data, dict) and "scores" in cached_data else cached_data
                cache_sha = sha256_file(PUB_FROZEN_SECTION_CACHE_PATH)
                seal_time = cached_manifest.get("seal_timestamp_utc", "")
                return scores, cached_manifest, cache_sha, seal_time
        except Exception as e:
            print(f"Failed loading cache: {e}. Scoring fresh...", flush=True)

    print("Executing fresh scoring on GPU...", flush=True)
    t0_start = time.perf_counter()
    start_time_iso = datetime.now(timezone.utc).isoformat()

    # In-memory document sections cache (parse each document once)
    print("Parsing candidate documents into structured legal sections...", flush=True)
    doc_sections_cache: Dict[str, List[LegalSection]] = {}
    pair_records: List[Tuple[str, str, str, str]] = []  # (qid, doc_id, q_text, sec_text)
    empty_doc_count = 0

    for q in public_ids:
        q_text = public_meta[q]
        for d in public_candidates[q]:
            if d not in doc_sections_cache:
                doc_raw = docs_store[d]
                if not doc_raw or not doc_raw.strip():
                    empty_doc_count += 1
                    raise RuntimeError(
                        f"BLOCKED_EMPTY_DOCUMENT_SECTION_COVERAGE: Candidate doc {d} has empty text!"
                    )
                secs = parse_document_into_sections(d, doc_raw, max_chunk_words=220, overlap_words=60)
                if not secs:
                    empty_doc_count += 1
                    raise RuntimeError(
                        f"BLOCKED_EMPTY_DOCUMENT_SECTION_COVERAGE: Candidate doc {d} produced 0 sections!"
                    )
                doc_sections_cache[d] = secs

            d_secs = doc_sections_cache[d]
            selected = preselect_legal_sections(q_text, d_secs, count=2)
            if not selected:
                raise RuntimeError(
                    f"BLOCKED_EMPTY_DOCUMENT_SECTION_COVERAGE: Preselection returned 0 sections for q={q}, d={d}!"
                )
            for sec in selected:
                pair_records.append((q, d, q_text, sec.text))

    total_section_pairs = len(pair_records)
    print(f"Generated {total_section_pairs} section pairs to score across {total_candidate_pairs} candidate pairs.", flush=True)

    # Load frozen Jina cross-encoder on GPU
    print("Loading frozen Jina Cross-Encoder on GPU...", flush=True)
    model, tok, prov = load_jina_crossencoder()
    torch.cuda.reset_peak_memory_stats()

    # Batch scoring
    scored_values: List[float] = []
    log_interval = max(batch_size * 50, 3200)

    for i in range(0, total_section_pairs, batch_size):
        chunk = pair_records[i : i + batch_size]
        sentence_pairs = [(r[2], r[3]) for r in chunk]
        batch_scores = model.compute_score(
            sentence_pairs, batch_size=batch_size, max_length=512
        )
        if isinstance(batch_scores, float):
            batch_scores = [batch_scores]
        scored_values.extend([float(s) for s in batch_scores])

        done = min(i + batch_size, total_section_pairs)
        if done % log_interval == 0 or done == total_section_pairs:
            elapsed = time.perf_counter() - t0_start
            vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
            speed = done / max(elapsed, 1e-4)
            pct = (done / total_section_pairs) * 100
            print(
                f"[{done}/{total_section_pairs}] ({pct:.1f}%) Elapsed: {elapsed:.1f}s, Speed: {speed:.1f} pairs/s, Peak VRAM: {vram_mb:.1f} MB",
                flush=True,
            )

    elapsed_total = time.perf_counter() - t0_start
    end_time_iso = datetime.now(timezone.utc).isoformat()
    peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    # Free model memory
    del model
    del tok
    torch.cuda.empty_cache()

    # Map scores back to (q, d) taking MAX over preselected sections
    scores: Dict[str, Dict[str, float]] = {q: {} for q in public_ids}
    for (q, d, _, _), s_val in zip(pair_records, scored_values):
        prev = scores[q].get(d, -1e9)
        if s_val > prev:
            scores[q][d] = s_val

    # Verify coverage across all candidate pairs
    missing_count = 0
    all_scores_flat = []
    for q in public_ids:
        for d in public_candidates[q]:
            if d not in scores[q]:
                missing_count += 1
            else:
                all_scores_flat.append(scores[q][d])

    if missing_count > 0:
        raise RuntimeError(f"BLOCKED_PUBLIC_SCORING: {missing_count} candidate pairs missing scores!")

    mean_score = float(np.mean(all_scores_flat))
    std_score = float(np.std(all_scores_flat))
    min_score = float(np.min(all_scores_flat))
    max_score = float(np.max(all_scores_flat))

    seal_timestamp_utc = datetime.now(timezone.utc).isoformat()

    manifest = {
        "schema_version": "dsc2026.gemini.huy_d1_frozen_section_public_v1.frozen_section_public_manifest.v1",
        "git_commit": git_info["head_commit"],
        "source_files": get_source_files_sha256(),
        "weights_path": str(WEIGHTS_JINA_FT).replace("\\", "/"),
        "weights_sha256": weights_sha256,
        "parser_sha256": parser_sha256,
        "public_query_fingerprint": pub_query_fp,
        "public_candidate_pool_fingerprint": pub_cand_fp,
        "max_length": 512,
        "max_chunk_words": 220,
        "overlap_words": 60,
        "count_sections": 2,
        "aggregation": "MAX",
        "score_semantics": "raw_frozen_ce_score",
        "total_public_queries": total_public_queries,
        "total_candidate_pairs": total_candidate_pairs,
        "total_section_pairs_scored": total_section_pairs,
        "missing_score_count": missing_count,
        "synthetic_fallback_count": 0,
        "empty_section_count": empty_doc_count,
        "scoring_start_time_utc": start_time_iso,
        "scoring_end_time_utc": end_time_iso,
        "elapsed_seconds": elapsed_total,
        "measured_throughput_pairs_per_sec": total_section_pairs / max(elapsed_total, 1e-4),
        "peak_vram_mb": peak_vram_mb,
        "seal_timestamp_utc": seal_timestamp_utc,
        "fresh_final_run": True,
        "reused_candidate_scores": False,
        "score_distribution": {
            "mean": mean_score,
            "std": std_score,
            "min": min_score,
            "max": max_score,
        },
    }

    payload = {
        "scores": scores,
        "manifest": manifest,
        "fresh_final_run": True,
        "reused_candidate_scores": False,
        "seal_timestamp_utc": seal_timestamp_utc,
    }

    PUB_FROZEN_SECTION_CACHE_PATH.write_bytes(pickle.dumps(payload, protocol=5))
    cache_sha256 = sha256_file(PUB_FROZEN_SECTION_CACHE_PATH)
    manifest["cache_path"] = str(PUB_FROZEN_SECTION_CACHE_PATH.relative_to(ROOT)).replace("\\", "/")
    manifest["cache_sha256"] = cache_sha256

    PUB_FROZEN_SECTION_MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved and sealed {PUB_FROZEN_SECTION_CACHE_PATH} (SHA256: {cache_sha256}) at {seal_timestamp_utc}", flush=True)
    print(f"Wrote {PUB_FROZEN_SECTION_MANIFEST_PATH}", flush=True)

    return scores, manifest, cache_sha256, seal_timestamp_utc


if __name__ == "__main__":
    from .public_d1_control_parity import run_public_d1_control_parity
    _, bundle = run_public_d1_control_parity()
    score_public_frozen_section_ce(bundle, batch_size=64, force_fresh=True)
