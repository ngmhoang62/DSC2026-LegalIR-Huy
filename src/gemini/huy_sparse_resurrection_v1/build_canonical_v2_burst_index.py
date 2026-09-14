"""
Build Canonical V2 Huy BURST-v4 SQLite FTS5 index over the exact 8,507 retained parents.
Enforces G2 geometry: chunk_size=500, overlap=100, step=400.
FTS5 tokenizer: unicode61.
Writes:
- results/gemini/huy_sparse_resurrection_v1/cache/canonical_v2_burst_fts.sqlite
Logs:
- results/gemini/huy_sparse_resurrection_v1/EXECUTION_TRACE.jsonl
- results/gemini/huy_sparse_resurrection_v1/EXECUTION_PROOF.json
"""

import json
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Ensure local imports
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

import common
from burst_retriever import tokens

SCRIPT_PATH = Path(__file__).resolve()
CANONICAL_CONTEXTS = common.REPO_ROOT / "cache/research_v2_forensic/kaggle_input/research-v2-jina-boundary-v4/V2_CONTEXTS.jsonl"
OUTPUT_DB = common.CACHE_DIR / "canonical_v2_burst_fts.sqlite"

CHUNK_SIZE = 500
OVERLAP = 100
STEP = CHUNK_SIZE - OVERLAP  # 400


def build_canonical_v2_index(force: bool = False):
    start_time = time.perf_counter()
    print("=" * 70, flush=True)
    print("BUILDING CANONICAL V2 HUY BURST-v4 SQLITE FTS5 INDEX", flush=True)
    print("=" * 70, flush=True)

    git_info = common.get_git_info()
    print(f"Dynamic Git HEAD: {git_info['git_commit_sha']}")
    print(f"Git dirty: {git_info['is_dirty']}")

    assert CANONICAL_CONTEXTS.exists(), f"Missing canonical contexts: {CANONICAL_CONTEXTS}"

    # Load canonical contexts
    print(f"Loading canonical contexts from {CANONICAL_CONTEXTS}...", flush=True)
    docs = []
    with CANONICAL_CONTEXTS.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                docs.append(json.loads(line))

    assert len(docs) == 8507, f"Expected 8,507 canonical docs, got {len(docs)}"
    print(f"Loaded {len(docs)} canonical parent documents.")

    if OUTPUT_DB.exists():
        if not force:
            # Check metadata
            conn = sqlite3.connect(OUTPUT_DB)
            try:
                meta = dict(conn.execute("SELECT key, value FROM metadata").fetchall())
                if (
                    int(meta.get("documents", -1)) == len(docs)
                    and int(meta.get("chunk_size", -1)) == CHUNK_SIZE
                    and int(meta.get("overlap", -1)) == OVERLAP
                ):
                    print(f"Valid cached canonical index found at {OUTPUT_DB} ({int(meta['chunks']):,} chunks).")
                    conn.close()
                    return
            except Exception:
                pass
            conn.close()
        print(f"Removing old index at {OUTPUT_DB}...", flush=True)
        OUTPUT_DB.unlink(missing_ok=True)

    OUTPUT_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(OUTPUT_DB)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=OFF")
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.execute("PRAGMA cache_size=-262144")

    conn.execute("CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT)")
    conn.execute("CREATE TABLE doc_ids(doc_idx INTEGER PRIMARY KEY, doc_id TEXT UNIQUE)")
    conn.execute("CREATE VIRTUAL TABLE docs_fts USING fts5(text, content='', tokenize='unicode61')")
    conn.execute("CREATE VIRTUAL TABLE chunks_fts USING fts5(text, content='', tokenize='unicode61')")
    conn.execute("CREATE TABLE chunk_owner(rowid INTEGER PRIMARY KEY, doc_idx INTEGER NOT NULL, chunk_idx INTEGER NOT NULL, start_tok INTEGER NOT NULL, end_tok INTEGER NOT NULL)")

    chunk_rowid = 0
    t_index_start = time.perf_counter()

    for doc_idx, doc_item in enumerate(docs):
        doc_id = str(doc_item["doc_id"])
        text = str(doc_item["passage"] or "")

        conn.execute("INSERT INTO doc_ids(doc_idx, doc_id) VALUES(?, ?)", (doc_idx, doc_id))
        conn.execute("INSERT INTO docs_fts(rowid, text) VALUES(?, ?)", (doc_idx + 1, text))

        toks = tokens(text)
        if not toks:
            toks = [""]

        chunk_idx = 0
        for start in range(0, len(toks), STEP):
            part = toks[start : start + CHUNK_SIZE]
            if not part:
                break
            chunk_rowid += 1
            chunk_idx += 1
            end = min(start + CHUNK_SIZE, len(toks))
            conn.execute("INSERT INTO chunks_fts(rowid, text) VALUES(?, ?)", (chunk_rowid, " ".join(part)))
            conn.execute("INSERT INTO chunk_owner(rowid, doc_idx, chunk_idx, start_tok, end_tok) VALUES(?, ?, ?, ?, ?)",
                         (chunk_rowid, doc_idx, chunk_idx, start, end))
            if start + CHUNK_SIZE >= len(toks):
                break

        if (doc_idx + 1) % 500 == 0 or (doc_idx + 1) == len(docs):
            conn.commit()
            print(f"Indexed {doc_idx + 1}/{len(docs)} docs, {chunk_rowid:,} chunks...", flush=True)

    print("Creating indexes on chunk_owner and doc_ids...", flush=True)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_chunk_owner_doc ON chunk_owner(doc_idx)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_doc_ids_id ON doc_ids(doc_id)")

    meta_entries = [
        ("documents", str(len(docs))),
        ("chunks", str(chunk_rowid)),
        ("chunk_size", str(CHUNK_SIZE)),
        ("overlap", str(OVERLAP)),
        ("step", str(STEP)),
        ("git_commit_sha", git_info["git_commit_sha"]),
        ("git_status_porcelain", git_info["git_status_porcelain"]),
        ("built_at", datetime.now(timezone.utc).isoformat()),
    ]
    conn.executemany("INSERT INTO metadata(key, value) VALUES(?, ?)", meta_entries)
    conn.commit()

    print("Running PRAGMA optimize...", flush=True)
    conn.execute("PRAGMA optimize")
    conn.close()

    elapsed = time.perf_counter() - start_time
    db_size_mb = OUTPUT_DB.stat().st_size / (1024 * 1024)
    db_sha = common.sha256(OUTPUT_DB)

    print(f"Successfully built Canonical V2 BURST index at {OUTPUT_DB}!")
    print(f"Total documents: {len(docs):,}")
    print(f"Total chunks:    {chunk_rowid:,}")
    print(f"Database size:   {db_size_mb:.2f} MB")
    print(f"Database SHA256: {db_sha}")
    print(f"Wall-clock time: {elapsed:.2f}s")

    # Trace & proof
    common.log_trace(
        stage="BUILD_CANONICAL_V2_BURST_INDEX",
        status="SUCCESS",
        script_path=SCRIPT_PATH,
        input_paths=[CANONICAL_CONTEXTS],
        output_path=OUTPUT_DB,
        records_processed=len(docs),
        wall_clock_sec=elapsed,
        extra_info={
            "documents": len(docs),
            "chunks": chunk_rowid,
            "chunk_size": CHUNK_SIZE,
            "overlap": OVERLAP,
            "db_size_mb": round(db_size_mb, 2),
            "db_sha256": db_sha,
        }
    )

    common.update_execution_proof(
        stage="CANONICAL_V2_BURST_INDEX",
        stage_data={
            "status": "COMPLETED",
            "db_path": str(OUTPUT_DB.relative_to(common.REPO_ROOT)),
            "db_sha256": db_sha,
            "db_size_mb": round(db_size_mb, 2),
            "documents": len(docs),
            "chunks": chunk_rowid,
            "chunk_size": CHUNK_SIZE,
            "overlap": OVERLAP,
            "tokenizer": "unicode61",
        }
    )


if __name__ == "__main__":
    build_canonical_v2_index()
