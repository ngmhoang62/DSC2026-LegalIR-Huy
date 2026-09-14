"""
Execute full Canonical V2 Huy BURST-v4 retrieval for all 6,991 queries.
Retrieves:
- H_FULL: top-500 documents by full-parent BM25
- H_LOCAL: top-2000 chunks by BM25, aggregated to top local documents (best + 0.3 * second)
- H_BURST: historical RRF fusion (local_weight=0.9, second_weight=0.3, rrf_k=20)
- Detailed evidence distributions for all candidates in the canonical pool.

Checkpoints every 250 queries to:
results/gemini/huy_sparse_resurrection_v1/cache/BURST_V2_RETRIEVAL_RESULTS.jsonl
"""

import json
import multiprocessing as mp
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Ensure local imports
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

import common
import burst_retriever
from burst_retriever import BurstIndex

SCRIPT_PATH = Path(__file__).resolve()
DB_PATH = common.CACHE_DIR / "canonical_v2_burst_fts.sqlite"
OUTPUT_FILE = common.CACHE_DIR / "BURST_V2_RETRIEVAL_RESULTS.jsonl"

_WORKER_INDEX: BurstIndex = None


def _init_worker(db_path_str: str):
    global _WORKER_INDEX
    _WORKER_INDEX = BurstIndex(Path(db_path_str))


def _process_single_query(task: Tuple[str, str, List[str]]) -> Dict[str, Any]:
    qid, question, pool = task
    expr = burst_retriever.fts_query(question)

    full_results = _WORKER_INDEX.retrieve_full(expr, limit=500)
    local_results, evidence = _WORKER_INDEX.retrieve_local_distribution(expr, limit=2000, second_weight=0.3)
    burst_results = _WORKER_INDEX.fuse(full_results, local_results, local_weight=0.9, rrf_k=20)

    # Fast lookup maps
    rf_map = {doc_id: (rank, score) for doc_id, rank, score in full_results}
    rl_map = {doc_id: (rank, score) for doc_id, rank, score in local_results}
    rb_map = {doc_id: (rank, score) for doc_id, rank, score in burst_results}

    # Extract pool features for all candidates in locked pool
    pool_data = {}
    for doc in pool:
        rf, sf = rf_map.get(doc, (100000, 0.0))
        rl, sl = rl_map.get(doc, (100000, 0.0))
        rb, sb = rb_map.get(doc, (100000, 0.0))
        ev = evidence.get(doc, {"best": 0.0, "second": 0.0, "third": 0.0, "chunk_count": 0, "local_score": 0.0})
        pool_data[doc] = {
            "h_full_rank": rf,
            "h_full_score": round(sf, 4),
            "h_local_rank": rl,
            "h_local_score": round(sl, 4),
            "h_burst_rank": rb,
            "h_burst_score": round(sb, 6),
            "best_chunk": round(ev["best"], 4),
            "second_chunk": round(ev["second"], 4),
            "third_chunk": round(ev["third"], 4),
            "chunk_count": ev["chunk_count"],
        }

    return {
        "qid": qid,
        "h_full_top100": [[d, r, round(s, 4)] for d, r, s in full_results[:100]],
        "h_local_top100": [[d, r, round(s, 4)] for d, r, s in local_results[:100]],
        "h_burst_top100": [[d, r, round(s, 6)] for d, r, s in burst_results[:100]],
        "h_full_top500_ids": [d for d, _, _ in full_results[:500]],
        "h_local_top500_ids": [d for d, _, _ in local_results[:500]],
        "h_burst_top500_ids": [d for d, _, _ in burst_results[:500]],
        "pool_evidence": pool_data,
    }


def run_all_retrievals(num_workers: int = 8, batch_size: int = 250):
    start_time = time.perf_counter()
    print("=" * 70, flush=True)
    print("EXECUTING CANONICAL V2 HUY BURST-v4 RETRIEVAL FOR ALL 6,991 QUERIES", flush=True)
    print("=" * 70, flush=True)

    git_info = common.get_git_info()
    print(f"Dynamic Git HEAD: {git_info['git_commit_sha']}")

    assert DB_PATH.exists(), f"Missing index database: {DB_PATH}"

    folds, fold_for, pools, questions, golds, e5_orders, e5_scores, dup, base_orders, base_scores = common.load_baseline_data()
    all_qids = sorted(questions.keys(), key=int)
    print(f"Loaded {len(all_qids)} total queries from canonical dataset.")

    # Check resume state
    completed_qids = set()
    if OUTPUT_FILE.exists():
        with OUTPUT_FILE.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        rec = json.loads(line)
                        completed_qids.add(str(rec["qid"]))
                    except Exception:
                        pass
        print(f"Resuming existing run: {len(completed_qids)} queries already processed in {OUTPUT_FILE}.")

    pending_qids = [q for q in all_qids if q not in completed_qids]
    print(f"Pending queries to process: {len(pending_qids)}")

    if not pending_qids:
        print("All queries already completed!")
    else:
        tasks = [(q, questions[q], pools[q]) for q in pending_qids]
        OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

        print(f"Launching worker pool with {num_workers} processes...", flush=True)
        with mp.Pool(processes=num_workers, initializer=_init_worker, initargs=(str(DB_PATH),)) as pool:
            # Process in batches with periodic flush
            for i in range(0, len(tasks), batch_size):
                batch_tasks = tasks[i : i + batch_size]
                t_b0 = time.perf_counter()
                batch_results = pool.map(_process_single_query, batch_tasks)
                t_b1 = time.perf_counter()

                with OUTPUT_FILE.open("a", encoding="utf-8") as f:
                    for rec in batch_results:
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

                done = len(completed_qids) + i + len(batch_results)
                batch_rate = len(batch_results) / (t_b1 - t_b0)
                print(f"[{done}/{len(all_qids)}] Processed batch of {len(batch_results)} in {t_b1-t_b0:.1f}s ({batch_rate:.2f} q/s)", flush=True)

    elapsed = time.perf_counter() - start_time
    file_size_mb = OUTPUT_FILE.stat().st_size / (1024 * 1024)
    file_sha = common.sha256(OUTPUT_FILE)

    print(f"\nAll {len(all_qids)} queries successfully processed and saved!")
    print(f"Output: {OUTPUT_FILE}")
    print(f"Size:   {file_size_mb:.2f} MB")
    print(f"SHA256: {file_sha}")
    print(f"Total wall-clock time: {elapsed:.2f}s")

    # Trace & proof
    common.log_trace(
        stage="HUY_BURST_V2_FULL_RETRIEVAL",
        status="SUCCESS",
        script_path=SCRIPT_PATH,
        input_paths=[DB_PATH, common.core.POOL_PATH],
        output_path=OUTPUT_FILE,
        records_processed=len(all_qids),
        wall_clock_sec=elapsed,
        extra_info={
            "total_queries": len(all_qids),
            "file_size_mb": round(file_size_mb, 2),
            "file_sha256": file_sha,
            "num_workers": num_workers,
        }
    )

    common.update_execution_proof(
        stage="HUY_BURST_V2_FULL_RETRIEVAL",
        stage_data={
            "status": "COMPLETED",
            "total_queries": len(all_qids),
            "retrieval_file": str(OUTPUT_FILE.relative_to(common.REPO_ROOT)),
            "retrieval_sha256": file_sha,
            "file_size_mb": round(file_size_mb, 2),
            "num_workers": num_workers,
            "wall_clock_seconds": round(elapsed, 2),
        }
    )


if __name__ == "__main__":
    run_all_retrievals(num_workers=8, batch_size=250)
