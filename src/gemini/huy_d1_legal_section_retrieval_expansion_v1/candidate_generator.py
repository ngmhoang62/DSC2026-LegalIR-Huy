"""Label-free candidate addition generator and sealer for CAL600 and Strict-V2.

Strict anti-contamination:
1. Generation inputs contain ZERO gold labels.
2. Indexes built strictly from corpus passages.
3. Additions generated for top 128 section hits, MAX parent score, cap=8 additions.
4. Additions sealed with SHA256 BEFORE gold labels are ever opened.
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

from .common import (
    CAL_CONTEXTS_DIR,
    CANONICAL_V2_CONTEXTS_JSONL,
    RES_DIR,
    ROOT,
    compute_candidate_fingerprint,
    compute_corpus_fingerprint,
    compute_query_fingerprint,
    get_git_status,
    load_cal_corpus,
    load_cal_generation_inputs,
    load_v2_generation_inputs,
    sha256_file,
)
from .section_retriever import LegalSectionIndex

INDEX_DIR = RES_DIR / "indexes"
CAL_DB_PATH = INDEX_DIR / "cal_sections.db"
V2_DB_PATH = INDEX_DIR / "v2_sections.db"


def generate_cal_additions() -> Tuple[Path, Path]:
    print("[GENERATE] Starting CAL600 section retrieval additions...", flush=True)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Build or load CAL index
    corpus = load_cal_corpus()
    print(f"[GENERATE] Loaded CAL corpus: {len(corpus):,} documents", flush=True)

    index = LegalSectionIndex.build_index(
        corpus=corpus,
        db_path=CAL_DB_PATH,
        max_chunk_words=220,
        overlap_words=60,
        verbose=True,
    )
    idx_stats = index.get_stats()
    idx_stats_file = RES_DIR / "CAL_SECTION_INDEX_STATS.json"
    idx_stats_file.write_text(json.dumps(idx_stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[GENERATE] Saved CAL index stats -> {idx_stats_file}", flush=True)

    # 2. Load generation inputs strictly label-free
    query_texts, blocks, all_ids, extended = load_cal_generation_inputs()
    cand_fp = compute_candidate_fingerprint(all_ids, extended)
    query_fp = compute_query_fingerprint(all_ids, query_texts)
    corpus_fp = compute_corpus_fingerprint(corpus)

    # 3. Generate additions
    out_jsonl = RES_DIR / "CAL_SECTION_RETRIEVAL_ADDITIONS.jsonl"
    out_seal = RES_DIR / "CAL_SECTION_RETRIEVAL_ADDITIONS_SEAL.json"

    t_gen_start = time.perf_counter()
    records: List[Dict[str, Any]] = []
    total_additions = 0
    triggered_queries = 0

    with open(out_jsonl, "w", encoding="utf-8") as f_out:
        for qid in all_ids:
            q_text = query_texts[qid]
            base_pool = set(extended[qid])
            adds = index.retrieve_parent_candidates(
                query_text=q_text,
                existing_pool=base_pool,
                section_hit_depth=128,
                cap=8,
            )
            rec = {
                "qid": qid,
                "query_text": q_text,
                "baseline_pool_size": len(base_pool),
                "addition_count": len(adds),
                "additions": adds,
            }
            f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            records.append(rec)
            if len(adds) > 0:
                triggered_queries += 1
                total_additions += len(adds)

    index.close()
    gen_time = time.perf_counter() - t_gen_start
    print(
        f"[GENERATE] Generated CAL additions in {gen_time:.2f}s: "
        f"{triggered_queries}/{len(all_ids)} queries triggered, {total_additions} total additions",
        flush=True,
    )

    # 4. Seal additions
    git_info = get_git_status()
    seal_data = {
        "schema_version": "dsc2026.gemini.huy_d1_legal_section_retrieval_expansion_v1.additions_seal.v1",
        "experiment_id": "HUY_D1_LEGAL_SECTION_RETRIEVAL_EXPANSION_V1",
        "dataset": "CAL600",
        "git_commit_sha": git_info["head_commit"],
        "query_fingerprint": query_fp,
        "baseline_candidate_pool_fingerprint": cand_fp,
        "corpus_fingerprint": corpus_fp,
        "section_index_sha256": idx_stats["sha256"],
        "section_index_sections": idx_stats["sections_indexed"],
        "section_index_documents": idx_stats["documents_indexed"],
        "generation_policy_config": {
            "section_parser": "legal_section_parser.py (boundary detection + sliding window fallback)",
            "max_chunk_words": 220,
            "overlap_words": 60,
            "tokenizer": "re.UNICODE r'\\w+', lowercased",
            "fts_query_formulation": "deduplicated OR bag-of-words",
            "sqlite_fts_tokenizer": "unicode61",
            "scoring_function": "-bm25(sections_fts)",
            "section_hit_depth": 128,
            "max_additions_cap": 8,
            "parent_aggregation": "MAX section score per parent document",
            "tie_breaking": "score DESC, parent_doc_id NUMERIC ASC",
        },
        "cap": 8,
        "section_hit_depth": 128,
        "generated_additions_artifact_path": str(out_jsonl.relative_to(ROOT)),
        "generated_additions_artifact_sha256": sha256_file(out_jsonl),
        "total_queries_evaluated": len(all_ids),
        "triggered_queries_count": triggered_queries,
        "total_new_additions": total_additions,
        "generation_time_seconds": gen_time,
        "sealed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    out_seal.write_text(json.dumps(seal_data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[GENERATE] Sealed CAL additions -> {out_seal}", flush=True)
    return out_jsonl, out_seal


def generate_v2_additions() -> Tuple[Path, Path]:
    print("[GENERATE] Starting Strict-V2 section retrieval additions...", flush=True)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load V2 generation inputs strictly label-free
    corpus, queries, v2_qids, candidate_pools = load_v2_generation_inputs()
    print(f"[GENERATE] Loaded V2 corpus: {len(corpus):,} documents, {len(v2_qids):,} queries", flush=True)

    # 2. Build or load V2 index
    index = LegalSectionIndex.build_index(
        corpus=corpus,
        db_path=V2_DB_PATH,
        max_chunk_words=220,
        overlap_words=60,
        verbose=True,
    )
    idx_stats = index.get_stats()
    idx_stats_file = RES_DIR / "V2_SECTION_INDEX_STATS.json"
    idx_stats_file.write_text(json.dumps(idx_stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[GENERATE] Saved V2 index stats -> {idx_stats_file}", flush=True)

    # 3. Generate additions
    out_jsonl = RES_DIR / "V2_SECTION_RETRIEVAL_ADDITIONS.jsonl"
    out_seal = RES_DIR / "V2_SECTION_RETRIEVAL_ADDITIONS_SEAL.json"

    t_gen_start = time.perf_counter()
    records: List[Dict[str, Any]] = []
    total_additions = 0
    triggered_queries = 0

    with open(out_jsonl, "w", encoding="utf-8") as f_out:
        for q_idx, qid in enumerate(v2_qids, 1):
            q_text = queries[qid]
            base_pool = set(candidate_pools[qid])
            adds = index.retrieve_parent_candidates(
                query_text=q_text,
                existing_pool=base_pool,
                section_hit_depth=128,
                cap=8,
            )
            rec = {
                "qid": qid,
                "query_text": q_text,
                "baseline_pool_size": len(base_pool),
                "addition_count": len(adds),
                "additions": adds,
            }
            f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            records.append(rec)
            if len(adds) > 0:
                triggered_queries += 1
                total_additions += len(adds)

            if q_idx % 1000 == 0 or q_idx == len(v2_qids):
                print(f"[GENERATE] V2 processed {q_idx}/{len(v2_qids)} queries...", flush=True)

    index.close()
    gen_time = time.perf_counter() - t_gen_start
    print(
        f"[GENERATE] Generated V2 additions in {gen_time:.2f}s: "
        f"{triggered_queries}/{len(v2_qids)} queries triggered, {total_additions} total additions",
        flush=True,
    )

    # 4. Seal additions
    git_info = get_git_status()
    seal_data = {
        "schema_version": "dsc2026.gemini.huy_d1_legal_section_retrieval_expansion_v1.additions_seal.v1",
        "experiment_id": "HUY_D1_LEGAL_SECTION_RETRIEVAL_EXPANSION_V1",
        "dataset": "Strict-V2",
        "git_commit_sha": git_info["head_commit"],
        "section_index_sha256": idx_stats["sha256"],
        "section_index_sections": idx_stats["sections_indexed"],
        "section_index_documents": idx_stats["documents_indexed"],
        "generation_policy_config": {
            "section_parser": "legal_section_parser.py (boundary detection + sliding window fallback)",
            "max_chunk_words": 220,
            "overlap_words": 60,
            "tokenizer": "re.UNICODE r'\\w+', lowercased",
            "fts_query_formulation": "deduplicated OR bag-of-words",
            "sqlite_fts_tokenizer": "unicode61",
            "scoring_function": "-bm25(sections_fts)",
            "section_hit_depth": 128,
            "max_additions_cap": 8,
            "parent_aggregation": "MAX section score per parent document",
            "tie_breaking": "score DESC, parent_doc_id NUMERIC ASC",
        },
        "cap": 8,
        "section_hit_depth": 128,
        "generated_additions_artifact_path": str(out_jsonl.relative_to(ROOT)),
        "generated_additions_artifact_sha256": sha256_file(out_jsonl),
        "total_queries_evaluated": len(v2_qids),
        "triggered_queries_count": triggered_queries,
        "total_new_additions": total_additions,
        "generation_time_seconds": gen_time,
        "sealed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    out_seal.write_text(json.dumps(seal_data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[GENERATE] Sealed V2 additions -> {out_seal}", flush=True)
    return out_jsonl, out_seal


def run_all_candidate_generation() -> None:
    generate_cal_additions()
    generate_v2_additions()


if __name__ == "__main__":
    run_all_candidate_generation()
