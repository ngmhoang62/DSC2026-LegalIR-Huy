"""Resume-safe exact Huy lexical top-2 Jina-v2 scoring for public candidates."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("HF_MODULES_CACHE", str(ROOT / "cache/huy_fasttrack/hf_modules"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src/research_v2_forensic"))
from benchmark_jina_reranker_holdouts import top_passages
from run_burst_expanded_fusion_submission import DocumentStore, prefetch
from run_evidence_contract_ab import load_model


PUBLIC = ROOT / "DSC2026-LegalIR-main/v4_run/public_test_dataset/public-official.json"
CONTEXTS = ROOT / "DSC2026-LegalIR-main/v4_run/public_test_dataset/selected-contexts"
POOL_DB = ROOT / "cache/research_v2_open_rl/v2_anchor_submission_candidate/public_scores.sqlite"
MODEL = ROOT / "models/jina-reranker-v2-base-multilingual"
OUTPUT = ROOT / "cache/huy_fasttrack/public_frozen_jina_scores.sqlite"


def open_db(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("CREATE TABLE IF NOT EXISTS scores(qid TEXT,doc_id TEXT,score REAL,PRIMARY KEY(qid,doc_id))")
    con.execute("CREATE TABLE IF NOT EXISTS progress(qid TEXT PRIMARY KEY,seconds REAL,pairs INTEGER,peak_mib REAL)")
    con.commit()
    return con


def load_pool():
    public = json.loads(PUBLIC.read_text(encoding="utf-8"))
    db = sqlite3.connect(f"file:{POOL_DB.as_posix()}?mode=ro", uri=True)
    rows = list(db.execute("SELECT qid,rank,doc_id FROM e5 ORDER BY CAST(qid AS INTEGER),rank"))
    db.close()
    pools = {}
    for qid, _, doc in rows:
        pools.setdefault(str(qid), []).append(str(doc))
    if set(pools) != set(public) or any(len(v) != 50 for v in pools.values()):
        raise RuntimeError("public pool contract mismatch")
    return public, pools


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    public, pools = load_pool()
    model, batch_size = load_model(MODEL, args.batch_size)
    docs = DocumentStore(sorted(CONTEXTS.glob("context_*.json")), cache_size=9000)
    con = open_db(OUTPUT)
    complete = {str(x[0]) for x in con.execute("SELECT qid FROM progress")}
    qids = sorted(public, key=int)
    if args.limit is not None:
        qids = qids[:args.limit]
    pending = [qid for qid in qids if qid not in complete]

    def prepare(qid):
        query = str(public[qid]["question"])
        owners, passages = [], []
        for doc in pools[qid]:
            for passage in top_passages(query, docs[doc], count=2):
                owners.append(doc)
                passages.append(passage)
        return qid, query, owners, passages

    started = time.perf_counter()
    done = 0
    torch.cuda.reset_peak_memory_stats()
    with ThreadPoolExecutor(max_workers=2) as executor:
        for qid, query, owners, passages in prefetch(executor, pending, prepare, ahead=2):
            before = time.perf_counter()
            try:
                raw = model.compute_score(
                    [(query, passage) for passage in passages],
                    batch_size=batch_size, max_length=512,
                )
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                batch_size = max(8, batch_size // 2)
                raw = model.compute_score(
                    [(query, passage) for passage in passages],
                    batch_size=batch_size, max_length=512,
                )
            if isinstance(raw, float):
                raw = [raw]
            parent = {}
            for doc, value in zip(owners, raw):
                parent[doc] = max(parent.get(doc, -1e9), float(value))
            if set(parent) != set(pools[qid]):
                raise RuntimeError(f"parent aggregation mismatch: {qid}")
            elapsed = time.perf_counter() - before
            peak = torch.cuda.max_memory_allocated() / 2**20
            with con:
                con.executemany("INSERT OR REPLACE INTO scores VALUES(?,?,?)", [(qid, doc, value) for doc, value in parent.items()])
                con.execute("INSERT OR REPLACE INTO progress VALUES(?,?,?,?)", (qid, elapsed, len(passages), peak))
            done += 1
            if done % 25 == 0:
                rate = (time.perf_counter() - started) / done
                print(json.dumps({"completed_this_run": done, "complete_total": len(complete) + done, "target": len(qids), "seconds_per_query": rate, "eta_seconds": rate * (len(pending) - done), "batch_size": batch_size, "peak_mib": peak}), flush=True)
    integrity = con.execute("PRAGMA integrity_check").fetchone()[0]
    counts = con.execute("SELECT COUNT(*),COUNT(DISTINCT qid) FROM scores").fetchone()
    progress = con.execute("SELECT COUNT(*),SUM(pairs),SUM(seconds),MAX(peak_mib) FROM progress").fetchone()
    con.close()
    expected_qids = len(qids) if args.limit is not None else 1000
    status = "COMPLETE" if integrity == "ok" and counts == (expected_qids * 50, expected_qids) else "PARTIAL"
    print(json.dumps({"status": status, "integrity": integrity, "scores": counts, "progress": progress, "output": str(OUTPUT)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
