"""Score CAL600 candidate pool using frozen Jina cross-encoder on preselected structured legal sections."""

from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path
from typing import Dict, List

import torch

ROOT = Path("D:/Study/DSC2026/sota")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.gemini.huy_d1_legal_section_evidence_v1.common import (
    RES_DIR,
    SCORE_CACHE_PKL,
    load_cal_data,
    load_jina_crossencoder,
    seed_everything,
)
from src.gemini.huy_d1_legal_section_evidence_v1.legal_section_parser import (
    parse_document_into_sections,
    preselect_legal_sections,
)


def score_legal_sections(batch_size: int = 16, count_sections: int = 2) -> dict:
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

    # 2. Check existing cache for resumability
    saved_scores: Dict[str, Dict[str, float]] = {}
    if SCORE_CACHE_PKL.exists():
        try:
            saved_scores = pickle.loads(SCORE_CACHE_PKL.read_bytes())
            print(
                f"Loaded existing cache with {len(saved_scores)} queries from {SCORE_CACHE_PKL}",
                flush=True,
            )
        except Exception as e:
            print(f"Warning: could not load existing cache: {e}", flush=True)
            saved_scores = {}

    todo = [
        q
        for q in all_ids
        if any(d not in saved_scores.get(q, {}) for d in extended[q])
    ]
    print(f"Total queries: {len(all_ids)}, Remaining to score: {len(todo)}", flush=True)

    peak_vram_mb = 0.0
    elapsed_seconds = 0.0

    if todo:
        # Load frozen cross-encoder
        print("Loading frozen Jina cross-encoder on GPU...", flush=True)
        model, tok, prov = load_jina_crossencoder()
        print(f"Model ready on {prov['device']}.", flush=True)

        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()

        # In-memory document sections cache so each document is only parsed once
        doc_sections_cache = {}

        for i, q in enumerate(todo, 1):
            q_text = queries[q][0]
            cand_docs = extended[q]

            # Preselect sections for all candidates of query q
            owners = []
            pairs = []
            for d in cand_docs:
                if d not in doc_sections_cache:
                    doc_sections_cache[d] = parse_document_into_sections(
                        d, docs[d]
                    )
                d_secs = doc_sections_cache[d]
                selected = preselect_legal_sections(
                    q_text, d_secs, count=count_sections
                )
                for sec in selected:
                    owners.append(d)
                    pairs.append((q_text, sec.text))

            # Compute cross-encoder scores
            q_doc_scores = dict(saved_scores.get(q, {}))
            if pairs:
                raw_scores = model.compute_score(
                    pairs, batch_size=batch_size, max_length=512
                )
                if isinstance(raw_scores, float):
                    raw_scores = [raw_scores]

                # Aggregate by document MAX score across preselected sections
                for d, s in zip(owners, raw_scores):
                    s_val = float(s)
                    q_doc_scores[d] = max(q_doc_scores.get(d, -1e9), s_val)

            saved_scores[q] = q_doc_scores

            if i % 25 == 0 or i == len(todo):
                cur_vram = torch.cuda.max_memory_allocated() / (1024 * 1024)
                peak_vram_mb = max(peak_vram_mb, cur_vram)
                t_now = time.perf_counter() - t0
                q_per_sec = i / max(t_now, 1e-4)
                print(
                    f"[{i}/{len(todo)}] Scored query {q} ({q_per_sec:.2f} q/s, peak VRAM: {peak_vram_mb:.1f} MB)",
                    flush=True,
                )
                SCORE_CACHE_PKL.write_bytes(pickle.dumps(saved_scores, protocol=5))

        elapsed_seconds = time.perf_counter() - t0
        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        print(
            f"Finished scoring {len(todo)} queries in {elapsed_seconds:.1f}s ({elapsed_seconds/60:.2f} min).",
            flush=True,
        )

    # Final cache save and verification
    SCORE_CACHE_PKL.write_bytes(pickle.dumps(saved_scores, protocol=5))
    cache_size_bytes = SCORE_CACHE_PKL.stat().st_size
    print(f"Saved {SCORE_CACHE_PKL} ({cache_size_bytes} bytes).", flush=True)

    # Verify coverage across all CAL600 candidate pool
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
        "cache_size_bytes": cache_size_bytes,
        "total_queries_scored": len(all_ids),
        "newly_scored_queries": len(todo),
        "elapsed_seconds": elapsed_seconds,
        "peak_vram_mb": peak_vram_mb,
    }


if __name__ == "__main__":
    res = score_legal_sections(batch_size=16, count_sections=2)
    print(json.dumps(res, indent=2))
