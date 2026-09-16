from __future__ import annotations

"""SQLite FTS5 Section Retriever for HUY_D1_LEGAL_SECTION_RETRIEVAL_EXPANSION_V1.

Reuses audited Huy lexical retrieval architecture:
- Tokenizer: regex r"\\w+" (re.UNICODE), lowercase
- FTS query formulation: deduplicated OR bag-of-words
- SQLite FTS5 virtual table with tokenize=\'unicode61\'
- Scoring: native SQLite FTS5 -bm25(...)
- Parent Aggregation: MAX section score per parent document
- Deterministic tie-breaking: score DESC, doc_id numeric ASC
- Addition budget: section_hit_depth=128, cap=8 additions outside baseline pool
"""

import math
import os
import re
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from .common import sha256_file
from .legal_section_parser import LegalSection, parse_document_into_sections

# Huy\'s exact token regex from burst_retriever.py
TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def tokens(text: str) -> List[str]:
    """Huy\'s exact tokenization."""
    return TOKEN_RE.findall((text or "").lower())


def fts_query(text: str) -> str:
    """Huy\'s exact FTS query formulation: deduplicated OR bag-of-words."""
    seen, terms = set(), []
    for term in tokens(text):
        if term not in seen:
            seen.add(term)
            terms.append('"' + term.replace('"', '""') + '"')
    return " OR ".join(terms)


def _doc_sort_key(doc_id: str) -> Any:
    """Deterministic sort key for doc IDs (numeric ascending if digits)."""
    return (0, int(doc_id)) if doc_id.isdigit() else (1, doc_id)


class LegalSectionIndex:
    """Interface to SQLite FTS5 legal section index."""

    def __init__(self, db_path: Path, readonly: bool = False):
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found: {self.db_path}")
        if readonly:
            self.conn = sqlite3.connect(
                f"file:{self.db_path.resolve().as_posix()}?mode=ro", uri=True
            )
        else:
            self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = None

    def close(self) -> None:
        if self.conn:
            self.conn.close()
            self.conn = None

    def __enter__(self) -> LegalSectionIndex:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    @classmethod
    def build_index(
        cls,
        corpus: Dict[str, Dict[str, Any]],
        db_path: Path,
        max_chunk_words: int = 220,
        overlap_words: int = 60,
        verbose: bool = True,
    ) -> LegalSectionIndex:
        """Build SQLite FTS5 index from corpus using legal section parser."""
        db_path = Path(db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        if db_path.exists():
            if verbose:
                print(f"[INDEX] Removing old index at {db_path}...", flush=True)
            db_path.unlink(missing_ok=True)

        conn = sqlite3.connect(str(db_path))
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=OFF")
        conn.execute("PRAGMA temp_store=MEMORY")
        conn.execute("PRAGMA cache_size=-262144")

        conn.execute("CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT)")
        conn.execute(
            "CREATE TABLE section_metadata("
            "rowid INTEGER PRIMARY KEY, "
            "parent_doc_id TEXT NOT NULL, "
            "section_idx INTEGER NOT NULL, "
            "section_type TEXT NOT NULL, "
            "heading TEXT NOT NULL, "
            "word_count INTEGER NOT NULL, "
            "snippet TEXT NOT NULL"
            ")"
        )
        conn.execute(
            "CREATE VIRTUAL TABLE sections_fts USING fts5("
            "text, content=\'\', tokenize=\'unicode61\'"
            ")"
        )

        sorted_doc_ids = sorted(
            corpus.keys(),
            key=lambda x: (0, int(x)) if x.isdigit() else (1, x),
        )

        t_start = time.perf_counter()
        total_sections = 0
        batch_fts: List[Tuple[int, str]] = []
        batch_meta: List[Tuple[int, str, int, str, str, int, str]] = []
        BATCH_SIZE = 10000

        try:
            for doc_count, doc_id in enumerate(sorted_doc_ids, 1):
                raw_text = corpus[doc_id].get("passage", "") or ""
                sections = parse_document_into_sections(
                    doc_id=doc_id,
                    raw_text=raw_text,
                    max_chunk_words=max_chunk_words,
                    overlap_words=overlap_words,
                )
                for sec in sections:
                    total_sections += 1
                    snippet = (sec.text or "")[:250].replace("\n", " ")
                    batch_fts.append((total_sections, sec.text))
                    batch_meta.append((
                        total_sections,
                        sec.doc_id,
                        sec.section_index,
                        sec.section_type,
                        sec.heading,
                        sec.word_count,
                        snippet,
                    ))

                if len(batch_fts) >= BATCH_SIZE:
                    conn.executemany(
                        "INSERT INTO sections_fts(rowid, text) VALUES(?, ?)", batch_fts
                    )
                    conn.executemany(
                        "INSERT INTO section_metadata(rowid, parent_doc_id, section_idx, section_type, heading, word_count, snippet) VALUES(?, ?, ?, ?, ?, ?, ?)",
                        batch_meta,
                    )
                    conn.commit()
                    batch_fts.clear()
                    batch_meta.clear()

                if verbose and (doc_count % 1000 == 0 or doc_count == len(sorted_doc_ids)):
                    elapsed = time.perf_counter() - t_start
                    rate = total_sections / max(elapsed, 0.001)
                    print(
                        f"[INDEX] Processed {doc_count}/{len(sorted_doc_ids)} docs -> {total_sections:,} sections ({rate:.0f} sec/s)...",
                        flush=True,
                    )

            if batch_fts:
                conn.executemany(
                    "INSERT INTO sections_fts(rowid, text) VALUES(?, ?)", batch_fts
                )
                conn.executemany(
                    "INSERT INTO section_metadata(rowid, parent_doc_id, section_idx, section_type, heading, word_count, snippet) VALUES(?, ?, ?, ?, ?, ?, ?)",
                    batch_meta,
                )
                conn.commit()
                batch_fts.clear()
                batch_meta.clear()

            if verbose:
                print("[INDEX] Creating index on section_metadata(parent_doc_id)...", flush=True)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_sec_meta_parent ON section_metadata(parent_doc_id)"
            )

            metadata_entries = [
                ("documents", str(len(sorted_doc_ids))),
                ("total_sections", str(total_sections)),
                ("max_chunk_words", str(max_chunk_words)),
                ("overlap_words", str(overlap_words)),
                ("built_at", datetime.now(timezone.utc).isoformat()),
            ]
            conn.executemany("INSERT INTO metadata(key, value) VALUES(?, ?)", metadata_entries)
            conn.commit()
        finally:
            conn.close()

        if verbose:
            elapsed = time.perf_counter() - t_start
            print(
                f"[INDEX] Complete! {total_sections:,} sections in {elapsed:.2f}s -> {db_path}",
                flush=True,
            )

        return cls(db_path, readonly=True)

    def get_stats(self) -> Dict[str, Any]:
        """Return index statistics and file size/sha256."""
        cur = self.conn.cursor()
        doc_count = cur.execute("SELECT COUNT(DISTINCT parent_doc_id) FROM section_metadata").fetchone()[0]
        sec_count = cur.execute("SELECT COUNT(*) FROM section_metadata").fetchone()[0]
        cur.close()
        size_bytes = self.db_path.stat().st_size if self.db_path.exists() else 0
        sha = sha256_file(self.db_path)
        return {
            "db_path": str(self.db_path),
            "documents_indexed": doc_count,
            "sections_indexed": sec_count,
            "size_bytes": size_bytes,
            "sha256": sha,
        }

    def retrieve_raw_sections(
        self, query_text: str, limit: int = 128
    ) -> List[Dict[str, Any]]:
        """Retrieve top section hits ordered by -bm25 descending."""
        if not query_text or not query_text.strip():
            return []
        expr = fts_query(query_text)
        if not expr:
            return []

        cur = self.conn.cursor()
        query_sql = (
            "SELECT s.rowid, -bm25(s.sections_fts) AS score, "
            "m.parent_doc_id, m.section_idx, m.section_type, m.heading, m.snippet "
            "FROM sections_fts s "
            "JOIN section_metadata m ON s.rowid = m.rowid "
            "WHERE s.sections_fts MATCH ? "
            "ORDER BY bm25(s.sections_fts), m.rowid ASC "
            "LIMIT ?"
        )
        rows = cur.execute(query_sql, (expr, limit)).fetchall()
        cur.close()

        results = []
        for r in rows:
            results.append({
                "rowid": r[0],
                "score": float(r[1]),
                "parent_doc_id": str(r[2]),
                "section_idx": int(r[3]),
                "section_type": str(r[4]),
                "heading": str(r[5]),
                "snippet": str(r[6]),
            })
        return results

    def retrieve_parent_candidates(
        self,
        query_text: str,
        existing_pool: Set[str],
        section_hit_depth: int = 128,
        cap: int = 8,
    ) -> List[Dict[str, Any]]:
        """Retrieve parent candidates via MAX section score aggregation.

        1. Fetches top `section_hit_depth` section matches by bm25.
        2. Collapses to parent documents keeping the MAX section score.
        3. Identifies candidates outside `existing_pool`.
        4. Sorts outside candidates deterministically: score DESC, doc_id numeric ASC.
        5. Takes up to `cap` new parent document additions.
        """
        raw_hits = self.retrieve_raw_sections(query_text, limit=section_hit_depth)
        if not raw_hits:
            return []

        # Collapse to unique parent documents with MAX score
        parent_docs: Dict[str, Dict[str, Any]] = {}
        for rank, hit in enumerate(raw_hits, 1):
            doc_id = hit["parent_doc_id"]
            score = hit["score"]
            if doc_id not in parent_docs:
                parent_docs[doc_id] = {
                    "doc_id": doc_id,
                    "score": score,
                    "best_section_idx": hit["section_idx"],
                    "best_section_type": hit["section_type"],
                    "best_heading": hit["heading"],
                    "best_snippet": hit["snippet"],
                    "best_section_rank": rank,
                    "section_hit_count": 1,
                }
            else:
                entry = parent_docs[doc_id]
                entry["section_hit_count"] += 1
                if score > entry["score"]:
                    entry["score"] = score
                    entry["best_section_idx"] = hit["section_idx"]
                    entry["best_section_type"] = hit["section_type"]
                    entry["best_heading"] = hit["heading"]
                    entry["best_snippet"] = hit["snippet"]
                    entry["best_section_rank"] = rank

        # Filter outside candidates
        outside_candidates: List[Dict[str, Any]] = [
            info for doc_id, info in parent_docs.items() if doc_id not in existing_pool
        ]

        # Deterministic sorting: score DESC, then numeric doc_id ASC
        outside_candidates.sort(
            key=lambda x: (-x["score"], _doc_sort_key(x["doc_id"]))
        )

        # Cap at budget
        return outside_candidates[:cap]
