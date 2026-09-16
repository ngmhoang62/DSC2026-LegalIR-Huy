"""Label-free candidate addition generator and sealer for CAL600 and Strict-V2.

Strict anti-contamination:
1. Generation inputs contain ZERO gold labels.
2. Indexes built strictly from corpus passages.
3. Additions generated for top 128 section hits, MAX parent score, cap=8 additions.
4. Additions sealed with SHA256 BEFORE gold labels are ever opened.
5. High-throughput concurrent querying over read-only SQLite FTS5 index.
"""

from __future__ import annotations

import concurrent.futures
import json
import os
import sqlite3
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

from .common import (
    CAL_CONTEXTS_DIR,
    CANONICAL_V2_CONTEXTS_JSONL,
    RES_DIR,
    ROOT,
    SRC_DIR,
    compute_candidate_fingerprint,
    compute_corpus_fingerprint,
    compute_query_fingerprint,
    get_git_status,
    load_cal_corpus,
    load_cal_generation_inputs,
    load_v2_generation_inputs,
    sha256_file,
)
from .section_retriever import (
    LegalSectionIndex,
    _doc_sort_key,
    build_index_manifest,
    fts_query,
)

INDEX_DIR = RES_DIR / "indexes"
CAL_DB_PATH = INDEX_DIR / "cal_sections.db"
V2_DB_PATH = INDEX_DIR / "v2_sections.db"

_THREAD_LOCAL = threading.local()


def check_provenance_seal_validity(
    seal_path: Path,
    additions_path: Path,
    db_path: Path,
    prov_path: Path,
    expected_dataset: str,
    expected_query_fp: str,
    expected_cand_fp: str,
    expected_corpus_fp: str,
) -> Tuple[bool, str]:
    """Verify all provenance fields before allowing candidate cache reuse."""
    if not seal_path.exists():
        return False, f"Seal file missing: {seal_path}"
    if not additions_path.exists():
        return False, f"Additions file missing: {additions_path}"
    if not db_path.exists():
        return False, f"Index DB file missing: {db_path}"
    if not prov_path.exists():
        return False, f"Index provenance file missing: {prov_path}"

    try:
        seal = json.loads(seal_path.read_text(encoding="utf-8"))
    except Exception as e:
        return False, f"Corrupted seal JSON: {e}"

    if seal.get("dataset") != expected_dataset:
        return False, f"Dataset mismatch: {seal.get('dataset')} != {expected_dataset}"

    actual_add_sha = sha256_file(additions_path)
    if seal.get("generated_additions_artifact_sha256") != actual_add_sha:
        return False, f"Additions SHA256 mismatch: seal={seal.get('generated_additions_artifact_sha256')} vs actual={actual_add_sha}"

    git_info = get_git_status()
    if seal.get("git_commit_sha") != git_info["head_commit"]:
        return False, f"Git commit mismatch: seal={seal.get('git_commit_sha')} vs head={git_info['head_commit']}"

    cg_sha = sha256_file(SRC_DIR / "candidate_generator.py")
    if seal.get("candidate_generator_source_sha256") != cg_sha:
        return False, f"candidate_generator.py hash mismatch: seal={seal.get('candidate_generator_source_sha256')} vs file={cg_sha}"

    sr_sha = sha256_file(SRC_DIR / "section_retriever.py")
    if seal.get("section_retriever_source_sha256") != sr_sha:
        return False, f"section_retriever.py hash mismatch: seal={seal.get('section_retriever_source_sha256')} vs file={sr_sha}"

    lp_sha = sha256_file(SRC_DIR / "legal_section_parser.py")
    if seal.get("legal_section_parser_source_sha256") != lp_sha:
        return False, f"legal_section_parser.py hash mismatch: seal={seal.get('legal_section_parser_source_sha256')} vs file={lp_sha}"

    if seal.get("query_fingerprint") != expected_query_fp:
        return False, "Query fingerprint mismatch"

    if seal.get("baseline_candidate_pool_fingerprint") != expected_cand_fp:
        return False, "Baseline candidate pool fingerprint mismatch"

    if seal.get("corpus_fingerprint") != expected_corpus_fp:
        return False, "Corpus fingerprint mismatch"

    actual_db_sha = sha256_file(db_path)
    if seal.get("section_index_sha256") != actual_db_sha:
        return False, f"Index DB SHA256 mismatch: seal={seal.get('section_index_sha256')} vs actual={actual_db_sha}"

    actual_prov_sha = sha256_file(prov_path)
    if seal.get("section_index_provenance_sha256") != actual_prov_sha:
        return False, f"Index provenance SHA256 mismatch: seal={seal.get('section_index_provenance_sha256')} vs actual={actual_prov_sha}"

    if seal.get("cap") != 8 or seal.get("section_hit_depth") != 128:
        return False, "Budget mismatch: cap!=8 or depth!=128"

    return True, "OK"


def _get_thread_connection(db_path: Path) -> sqlite3.Connection:
    attr_name = f"conn_{db_path.stem}"
    if not hasattr(_THREAD_LOCAL, attr_name):
        conn = sqlite3.connect(f"file:{db_path.resolve().as_posix()}?mode=ro", uri=True)
        setattr(_THREAD_LOCAL, attr_name, conn)
    return getattr(_THREAD_LOCAL, attr_name)


def _query_worker(
    task: Tuple[str, str, Set[str], str, int, int]
) -> Dict[str, Any]:
    qid, q_text, base_pool, db_path_str, hit_depth, cap = task
    if not q_text or not q_text.strip():
        return {
            "qid": qid,
            "query_text": q_text,
            "baseline_pool_size": len(base_pool),
            "addition_count": 0,
            "additions": [],
        }

    expr = fts_query(q_text)
    if not expr:
        return {
            "qid": qid,
            "query_text": q_text,
            "baseline_pool_size": len(base_pool),
            "addition_count": 0,
            "additions": [],
        }

    conn = _get_thread_connection(Path(db_path_str))
    cur = conn.cursor()
    query_sql = (
        "SELECT s.rowid, -bm25(s.sections_fts) AS score, "
        "m.parent_doc_id, m.section_idx, m.section_type, m.heading, m.snippet "
        "FROM sections_fts s "
        "JOIN section_metadata m ON s.rowid = m.rowid "
        "WHERE s.sections_fts MATCH ? "
        "ORDER BY bm25(s.sections_fts), m.rowid ASC "
        "LIMIT ?"
    )
    rows = cur.execute(query_sql, (expr, hit_depth)).fetchall()
    cur.close()

    parent_docs: Dict[str, Dict[str, Any]] = {}
    for rank, r in enumerate(rows, 1):
        doc_id = str(r[2])
        score = float(r[1])
        if doc_id not in parent_docs:
            parent_docs[doc_id] = {
                "doc_id": doc_id,
                "score": score,
                "best_section_idx": int(r[3]),
                "best_section_type": str(r[4]),
                "best_heading": str(r[5]),
                "best_snippet": str(r[6]),
                "best_section_rank": rank,
                "section_hit_count": 1,
            }
        else:
            entry = parent_docs[doc_id]
            entry["section_hit_count"] += 1
            if score > entry["score"]:
                entry["score"] = score
                entry["best_section_idx"] = int(r[3])
                entry["best_section_type"] = str(r[4])
                entry["best_heading"] = str(r[5])
                entry["best_snippet"] = str(r[6])
                entry["best_section_rank"] = rank

    outside = [info for doc_id, info in parent_docs.items() if doc_id not in base_pool]
    outside.sort(key=lambda x: (-x["score"], _doc_sort_key(x["doc_id"])))
    adds = outside[:cap]

    return {
        "qid": qid,
        "query_text": q_text,
        "baseline_pool_size": len(base_pool),
        "addition_count": len(adds),
        "additions": adds,
    }


def generate_cal_additions(force_rebuild: bool = False) -> Tuple[Path, Path]:
    print("[GENERATE] Starting CAL600 section retrieval additions...", flush=True)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    out_jsonl = RES_DIR / "CAL_SECTION_RETRIEVAL_ADDITIONS.jsonl"
    out_seal = RES_DIR / "CAL_SECTION_RETRIEVAL_ADDITIONS_SEAL.json"
    idx_stats_file = RES_DIR / "CAL_SECTION_INDEX_STATS.json"
    idx_prov_file = RES_DIR / "CAL_SECTION_INDEX_PROVENANCE.json"

    # 1. Load generation inputs strictly label-free
    corpus = load_cal_corpus()
    query_texts, blocks, all_ids, extended = load_cal_generation_inputs()
    cand_fp = compute_candidate_fingerprint(all_ids, extended)
    query_fp = compute_query_fingerprint(all_ids, query_texts)
    corpus_fp = compute_corpus_fingerprint(corpus)

    # 2. Check reuse validity
    if not force_rebuild:
        valid, reason = check_provenance_seal_validity(
            seal_path=out_seal,
            additions_path=out_jsonl,
            db_path=CAL_DB_PATH,
            prov_path=idx_prov_file,
            expected_dataset="CAL600",
            expected_query_fp=query_fp,
            expected_cand_fp=cand_fp,
            expected_corpus_fp=corpus_fp,
        )
        if valid:
            seal_data = json.loads(out_seal.read_text(encoding="utf-8"))
            print(
                f"[GENERATE] Found valid existing sealed CAL additions with full provenance match "
                f"({seal_data.get('triggered_queries_count')}/{seal_data.get('total_queries_evaluated')} triggered, "
                f"{seal_data.get('total_new_additions')} additions) -> Reusing sealed artifact.",
                flush=True,
            )
            return out_jsonl, out_seal
        else:
            print(f"[GENERATE] Cannot reuse CAL candidate additions: {reason}. Running clean generation...", flush=True)

    # Clean previous artifacts for fresh generation
    for p in [CAL_DB_PATH, out_jsonl, out_seal, idx_stats_file, idx_prov_file]:
        if p.exists():
            p.unlink(missing_ok=True)

    # 3. Build CAL index
    print(f"[GENERATE] Building fresh CAL section index from {len(corpus):,} documents...", flush=True)
    index = LegalSectionIndex.build_index(
        corpus=corpus,
        db_path=CAL_DB_PATH,
        max_chunk_words=220,
        overlap_words=60,
        verbose=True,
    )
    idx_stats = index.get_stats()
    idx_stats_file.write_text(json.dumps(idx_stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[GENERATE] Saved CAL index stats -> {idx_stats_file}", flush=True)

    build_index_manifest(
        index_stats=idx_stats,
        corpus_fingerprint=corpus_fp,
        output_path=idx_prov_file,
        dataset_name="CAL600",
        max_chunk_words=220,
        overlap_words=60,
    )
    print(f"[GENERATE] Saved CAL index provenance -> {idx_prov_file}", flush=True)

    # 4. Generate additions concurrently
    t_gen_start = time.perf_counter()
    tasks = [
        (qid, query_texts[qid], set(extended[qid]), str(CAL_DB_PATH), 128, 8)
        for qid in all_ids
    ]

    records: List[Dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        records = list(executor.map(_query_worker, tasks))

    total_additions = sum(r["addition_count"] for r in records)
    triggered_queries = sum(1 for r in records if r["addition_count"] > 0)

    with open(out_jsonl, "w", encoding="utf-8") as f_out:
        for rec in records:
            f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")

    index.close()
    gen_time = time.perf_counter() - t_gen_start
    print(
        f"[GENERATE] Generated CAL additions in {gen_time:.2f}s: "
        f"{triggered_queries}/{len(all_ids)} queries triggered, {total_additions} total additions",
        flush=True,
    )

    # 5. Seal additions with full provenance contract
    git_info = get_git_status()
    seal_data = {
        "schema_version": "dsc2026.gemini.huy_d1_legal_section_retrieval_expansion_v1.additions_seal.v1",
        "experiment_id": "HUY_D1_LEGAL_SECTION_RETRIEVAL_EXPANSION_V1",
        "dataset": "CAL600",
        "git_commit_sha": git_info["head_commit"],
        "candidate_generator_source_sha256": sha256_file(SRC_DIR / "candidate_generator.py"),
        "section_retriever_source_sha256": sha256_file(SRC_DIR / "section_retriever.py"),
        "legal_section_parser_source_sha256": sha256_file(SRC_DIR / "legal_section_parser.py"),
        "query_fingerprint": query_fp,
        "baseline_candidate_pool_fingerprint": cand_fp,
        "corpus_fingerprint": corpus_fp,
        "section_index_path": str(CAL_DB_PATH.relative_to(ROOT)),
        "section_index_sha256": idx_stats["sha256"],
        "section_index_provenance_path": str(idx_prov_file.relative_to(ROOT)),
        "section_index_provenance_sha256": sha256_file(idx_prov_file),
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


def generate_v2_additions(force_rebuild: bool = False) -> Tuple[Path, Path]:
    print("[GENERATE] Starting Strict-V2 section retrieval additions...", flush=True)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    out_jsonl = RES_DIR / "V2_SECTION_RETRIEVAL_ADDITIONS.jsonl"
    out_seal = RES_DIR / "V2_SECTION_RETRIEVAL_ADDITIONS_SEAL.json"
    idx_stats_file = RES_DIR / "V2_SECTION_INDEX_STATS.json"
    idx_prov_file = RES_DIR / "V2_SECTION_INDEX_PROVENANCE.json"

    # 1. Load V2 generation inputs strictly label-free
    corpus, queries, v2_qids, candidate_pools = load_v2_generation_inputs()
    v2_cand_fp = compute_candidate_fingerprint(v2_qids, candidate_pools)
    v2_query_fp = compute_query_fingerprint(v2_qids, queries)
    v2_corpus_fp = compute_corpus_fingerprint(corpus)

    # 2. Check reuse validity
    if not force_rebuild:
        valid, reason = check_provenance_seal_validity(
            seal_path=out_seal,
            additions_path=out_jsonl,
            db_path=V2_DB_PATH,
            prov_path=idx_prov_file,
            expected_dataset="Strict-V2",
            expected_query_fp=v2_query_fp,
            expected_cand_fp=v2_cand_fp,
            expected_corpus_fp=v2_corpus_fp,
        )
        if valid:
            seal_data = json.loads(out_seal.read_text(encoding="utf-8"))
            print(
                f"[GENERATE] Found valid existing sealed V2 additions with full provenance match "
                f"({seal_data.get('triggered_queries_count')}/{seal_data.get('total_queries_evaluated')} triggered, "
                f"{seal_data.get('total_new_additions')} additions) -> Reusing sealed artifact.",
                flush=True,
            )
            return out_jsonl, out_seal
        else:
            print(f"[GENERATE] Cannot reuse V2 candidate additions: {reason}. Running clean generation...", flush=True)

    # Clean previous artifacts for fresh generation
    for p in [V2_DB_PATH, out_jsonl, out_seal, idx_stats_file, idx_prov_file]:
        if p.exists():
            p.unlink(missing_ok=True)

    # 3. Build V2 index
    print(f"[GENERATE] Building fresh V2 section index from {len(corpus):,} documents...", flush=True)
    index = LegalSectionIndex.build_index(
        corpus=corpus,
        db_path=V2_DB_PATH,
        max_chunk_words=220,
        overlap_words=60,
        verbose=True,
    )
    idx_stats = index.get_stats()
    idx_stats_file.write_text(json.dumps(idx_stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[GENERATE] Saved V2 index stats -> {idx_stats_file}", flush=True)

    build_index_manifest(
        index_stats=idx_stats,
        corpus_fingerprint=v2_corpus_fp,
        output_path=idx_prov_file,
        dataset_name="Strict-V2",
        max_chunk_words=220,
        overlap_words=60,
    )
    print(f"[GENERATE] Saved V2 index provenance -> {idx_prov_file}", flush=True)

    # 4. Generate additions concurrently in chunks with progress reporting
    t_gen_start = time.perf_counter()
    tasks = [
        (qid, queries[qid], set(candidate_pools[qid]), str(V2_DB_PATH), 128, 8)
        for qid in v2_qids
    ]

    records: List[Dict[str, Any]] = []
    CHUNK_SIZE = 500
    num_workers = 10
    print(f"[GENERATE] Processing {len(tasks):,} V2 queries using ThreadPoolExecutor(max_workers={num_workers})...", flush=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        for chunk_idx in range(0, len(tasks), CHUNK_SIZE):
            t_chunk_0 = time.perf_counter()
            chunk_tasks = tasks[chunk_idx : chunk_idx + CHUNK_SIZE]
            chunk_results = list(executor.map(_query_worker, chunk_tasks))
            records.extend(chunk_results)

            done_count = len(records)
            chunk_elapsed = time.perf_counter() - t_chunk_0
            total_elapsed = time.perf_counter() - t_gen_start
            rate = len(chunk_results) / max(chunk_elapsed, 0.001)
            eta_seconds = (len(tasks) - done_count) / (done_count / max(total_elapsed, 0.001))
            print(
                f"[GENERATE] V2 processed {done_count}/{len(tasks)} queries "
                f"({rate:.1f} q/s, ETA: {eta_seconds/60:.1f}m)...",
                flush=True,
            )

    total_additions = sum(r["addition_count"] for r in records)
    triggered_queries = sum(1 for r in records if r["addition_count"] > 0)

    with open(out_jsonl, "w", encoding="utf-8") as f_out:
        for rec in records:
            f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")

    index.close()
    gen_time = time.perf_counter() - t_gen_start
    print(
        f"[GENERATE] Generated V2 additions in {gen_time:.2f}s: "
        f"{triggered_queries}/{len(v2_qids)} queries triggered, {total_additions} total additions",
        flush=True,
    )

    # 5. Seal additions with full provenance contract
    git_info = get_git_status()
    seal_data = {
        "schema_version": "dsc2026.gemini.huy_d1_legal_section_retrieval_expansion_v1.additions_seal.v1",
        "experiment_id": "HUY_D1_LEGAL_SECTION_RETRIEVAL_EXPANSION_V1",
        "dataset": "Strict-V2",
        "git_commit_sha": git_info["head_commit"],
        "candidate_generator_source_sha256": sha256_file(SRC_DIR / "candidate_generator.py"),
        "section_retriever_source_sha256": sha256_file(SRC_DIR / "section_retriever.py"),
        "legal_section_parser_source_sha256": sha256_file(SRC_DIR / "legal_section_parser.py"),
        "query_fingerprint": v2_query_fp,
        "baseline_candidate_pool_fingerprint": v2_cand_fp,
        "corpus_fingerprint": v2_corpus_fp,
        "section_index_path": str(V2_DB_PATH.relative_to(ROOT)),
        "section_index_sha256": idx_stats["sha256"],
        "section_index_provenance_path": str(idx_prov_file.relative_to(ROOT)),
        "section_index_provenance_sha256": sha256_file(idx_prov_file),
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


def run_all_candidate_generation(force_rebuild: bool = False) -> None:
    generate_cal_additions(force_rebuild=force_rebuild)
    generate_v2_additions(force_rebuild=force_rebuild)


if __name__ == "__main__":
    force = "--force-rebuild" in sys.argv
    run_all_candidate_generation(force_rebuild=force)
