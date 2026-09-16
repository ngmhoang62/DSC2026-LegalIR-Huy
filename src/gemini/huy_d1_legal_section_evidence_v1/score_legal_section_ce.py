"""Score CAL600 candidate pool using frozen Jina cross-encoder on preselected structured legal sections."""

from __future__ import annotations

import datetime
import json
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import torch

ROOT = Path("D:/Study/DSC2026/sota")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.gemini.huy_d1_legal_section_evidence_v1.common import (
    RES_DIR,
    SCORE_CACHE_PKL,
    SRC_DIR,
    WEIGHTS_JINA_FT,
    compute_candidate_fingerprint,
    compute_query_fingerprint,
    get_git_commit_sha,
    load_cal_data,
    load_jina_crossencoder,
    seed_everything,
    sha256_file,
)
from src.gemini.huy_d1_legal_section_evidence_v1.legal_section_parser import (
    parse_document_into_sections,
    preselect_legal_sections,
    score_section_lexical,
)


def compute_score_contract(
    all_ids: List[str], extended: Dict[str, List[str]], queries: Dict[str, Any]
) -> Dict[str, Any]:
    """Compute provenance-sealed contract manifest for score cache."""
    return {
        "git_commit_sha": get_git_commit_sha(),
        "source_sha256": {
            "legal_section_parser.py": sha256_file(SRC_DIR / "legal_section_parser.py"),
            "score_legal_section_ce.py": sha256_file(SRC_DIR / "score_legal_section_ce.py"),
            "common.py": sha256_file(SRC_DIR / "common.py"),
        },
        "weights_path": str(WEIGHTS_JINA_FT).replace("\\", "/"),
        "weights_sha256": sha256_file(WEIGHTS_JINA_FT),
        "model_identity": "jinaai/jina-reranker-v2-base-multilingual",
        "max_length": 512,
        "max_chunk_words": 220,
        "overlap_words": 60,
        "count_sections": 2,
        "aggregation": "MAX",
        "candidate_pool_fingerprint": compute_candidate_fingerprint(all_ids, extended),
        "query_population_fingerprint": compute_query_fingerprint(all_ids, queries),
    }


def score_legal_sections(
    batch_size: int = 16,
    count_sections: int = 2,
    fresh_run: bool = False,
) -> dict:
    seed_everything(2026)
    RES_DIR.mkdir(parents=True, exist_ok=True)
    print("=== SCORING CAL600 WITH LEGAL SECTION CROSS-ENCODER ===", flush=True)

    # 1. Load CAL dataset
    print("Loading CAL dataset...", flush=True)
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
    ) = load_cal_data()

    # 2. Compute execution contract manifest
    contract = compute_score_contract(all_ids, extended, queries)
    print(f"Contract Git Commit: {contract['git_commit_sha']}", flush=True)
    print(f"Contract Model Weights SHA256: {contract['weights_sha256']}", flush=True)

    # 3. Handle cache and provenance checks
    saved_scores: Dict[str, Dict[str, float]] = {}
    saved_section_details: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    reused_queries = 0
    fresh_final_run = fresh_run

    if fresh_run:
        print("Fresh final run requested: removing any existing score cache...", flush=True)
        if SCORE_CACHE_PKL.exists():
            SCORE_CACHE_PKL.unlink()
        todo = list(all_ids)
    else:
        if SCORE_CACHE_PKL.exists():
            try:
                cached_data = pickle.loads(SCORE_CACHE_PKL.read_bytes())
                cached_manifest = cached_data.get("manifest", {})
                match = (
                    cached_manifest.get("weights_sha256") == contract["weights_sha256"]
                    and cached_manifest.get("candidate_pool_fingerprint") == contract["candidate_pool_fingerprint"]
                    and cached_manifest.get("query_population_fingerprint") == contract["query_population_fingerprint"]
                    and cached_manifest.get("count_sections") == contract["count_sections"]
                    and cached_manifest.get("max_chunk_words") == contract["max_chunk_words"]
                    and cached_manifest.get("overlap_words") == contract["overlap_words"]
                    and cached_manifest.get("aggregation") == contract["aggregation"]
                )
                if match:
                    saved_scores = cached_data.get("scores", {})
                    saved_section_details = cached_data.get("section_details", {})
                    print(
                        f"Loaded valid provenance-matched cache with {len(saved_scores)} queries from {SCORE_CACHE_PKL}",
                        flush=True,
                    )
                else:
                    print("Cache manifest does not match contract! Refusing cache reuse.", flush=True)
                    saved_scores = {}
                    saved_section_details = {}
                    fresh_final_run = True
            except Exception as e:
                print(f"Warning: could not load existing cache: {e}", flush=True)
                saved_scores = {}
                saved_section_details = {}
                fresh_final_run = True

        todo = [
            q
            for q in all_ids
            if any(d not in saved_scores.get(q, {}) for d in extended[q])
        ]
        reused_queries = len(all_ids) - len(todo)

    newly_scored = len(todo)
    print(f"Total queries: {len(all_ids)}, Reused: {reused_queries}, Newly to score: {newly_scored}", flush=True)

    peak_vram_mb = 0.0
    elapsed_seconds = 0.0
    t0_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()

    if todo:
        # Load frozen cross-encoder
        print("Loading frozen Jina cross-encoder on GPU...", flush=True)
        model, tok, prov = load_jina_crossencoder()
        print(f"Model ready on {prov['device']}.", flush=True)

        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()

        # In-memory document sections cache so each document is parsed once
        doc_sections_cache: Dict[str, List[Any]] = {}

        for i, q in enumerate(todo, 1):
            q_text = queries[q][0]
            cand_docs = extended[q]

            # Preselect sections for all candidates of query q
            pair_records = []
            for d in cand_docs:
                if d not in doc_sections_cache:
                    doc_sections_cache[d] = parse_document_into_sections(d, docs[d])
                d_secs = doc_sections_cache[d]
                selected = preselect_legal_sections(
                    q_text, d_secs, count=count_sections
                )
                for sec in selected:
                    pair_records.append((d, sec, q_text, sec.text))

            q_doc_scores = dict(saved_scores.get(q, {}))
            q_sec_details: Dict[str, List[Dict[str, Any]]] = dict(
                saved_section_details.get(q, {})
            )

            if pair_records:
                pairs = [(r[2], r[3]) for r in pair_records]
                raw_scores = model.compute_score(
                    pairs, batch_size=batch_size, max_length=512
                )
                if isinstance(raw_scores, float):
                    raw_scores = [raw_scores]

                # Map raw scores back to candidate docs and selected sections
                for (d, sec, _, _), s_val in zip(pair_records, raw_scores):
                    s_float = float(s_val)
                    if d not in q_sec_details:
                        q_sec_details[d] = []
                    q_sec_details[d].append({
                        "section_index": sec.section_index,
                        "section_type": sec.section_type,
                        "heading": sec.heading,
                        "excerpt": sec.text[:250] + "..." if len(sec.text) > 250 else sec.text,
                        "lexical_score": score_section_lexical(q_text, sec),
                        "raw_ce_score": s_float,
                    })
                    q_doc_scores[d] = max(q_doc_scores.get(d, -1e9), s_float)

            saved_scores[q] = q_doc_scores
            saved_section_details[q] = q_sec_details

            if i % 25 == 0 or i == len(todo):
                cur_vram = torch.cuda.max_memory_allocated() / (1024 * 1024)
                peak_vram_mb = max(peak_vram_mb, cur_vram)
                t_now = time.perf_counter() - t0
                q_per_sec = i / max(t_now, 1e-4)
                print(
                    f"[{i}/{len(todo)}] Scored query {q} ({q_per_sec:.2f} q/s, peak VRAM: {peak_vram_mb:.1f} MB)",
                    flush=True,
                )
                interim_payload = {
                    "scores": saved_scores,
                    "section_details": saved_section_details,
                    "manifest": contract,
                    "fresh_final_run": fresh_final_run,
                    "newly_scored_queries": newly_scored,
                    "reused_queries": reused_queries,
                }
                SCORE_CACHE_PKL.write_bytes(pickle.dumps(interim_payload, protocol=5))

        elapsed_seconds = time.perf_counter() - t0
        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        print(
            f"Finished scoring {len(todo)} queries in {elapsed_seconds:.1f}s ({elapsed_seconds/60:.2f} min).",
            flush=True,
        )

    t1_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()

    final_payload = {
        "scores": saved_scores,
        "section_details": saved_section_details,
        "manifest": contract,
        "fresh_final_run": fresh_final_run,
        "newly_scored_queries": newly_scored,
        "reused_queries": reused_queries,
        "scoring_start_time": t0_iso,
        "scoring_end_time": t1_iso,
    }
    SCORE_CACHE_PKL.write_bytes(pickle.dumps(final_payload, protocol=5))
    cache_size_bytes = SCORE_CACHE_PKL.stat().st_size
    cache_sha256 = sha256_file(SCORE_CACHE_PKL)
    print(f"Saved {SCORE_CACHE_PKL} ({cache_size_bytes} bytes, sha256={cache_sha256}).", flush=True)

    manifest_path = RES_DIR / "legal_section_ce_manifest.json"
    manifest_doc = {
        "manifest": contract,
        "cache_path": str(SCORE_CACHE_PKL).replace("\\", "/"),
        "cache_sha256": cache_sha256,
        "cache_size_bytes": cache_size_bytes,
        "fresh_final_run": fresh_final_run,
        "total_queries_scored": len(all_ids),
        "newly_scored_queries": newly_scored,
        "reused_queries": reused_queries,
        "scoring_start_time": t0_iso,
        "scoring_end_time": t1_iso,
        "elapsed_seconds": elapsed_seconds,
        "peak_vram_mb": peak_vram_mb,
    }
    manifest_path.write_text(json.dumps(manifest_doc, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {manifest_path}", flush=True)

    missing_scores_count = 0
    for q in all_ids:
        for d in extended[q]:
            if d not in saved_scores.get(q, {}):
                missing_scores_count += 1

    assert (
        missing_scores_count == 0
    ), f"FATAL: Missing {missing_scores_count} candidate scores in cache!"
    print(
        f"Verified complete coverage: 600 queries, 0 missing candidate scores.",
        flush=True,
    )

    return {
        "status": "SUCCESS",
        "cache_path": str(SCORE_CACHE_PKL).replace("\\", "/"),
        "cache_sha256": cache_sha256,
        "cache_size_bytes": cache_size_bytes,
        "manifest": contract,
        "fresh_final_run": fresh_final_run,
        "total_queries_scored": len(all_ids),
        "newly_scored_queries": newly_scored,
        "reused_queries": reused_queries,
        "scoring_start_time": t0_iso,
        "scoring_end_time": t1_iso,
        "elapsed_seconds": elapsed_seconds,
        "peak_vram_mb": peak_vram_mb,
    }


if __name__ == "__main__":
    res = score_legal_sections(batch_size=16, count_sections=2, fresh_run=False)
    print(json.dumps(res, indent=2))
