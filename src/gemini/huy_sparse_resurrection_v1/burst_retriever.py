import math
import re
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Huy's exact token regex
TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def tokens(text: str) -> List[str]:
    """Huy's exact tokenization."""
    return TOKEN_RE.findall((text or "").lower())


def fts_query(text: str) -> str:
    """Huy's exact FTS query formulation: deduplicated OR bag-of-words."""
    seen, terms = set(), []
    for term in tokens(text):
        if term not in seen:
            seen.add(term)
            terms.append('"' + term.replace('"', '""') + '"')
    return " OR ".join(terms)


def _title_from_link(link: str) -> str:
    """Huy's fallback title from URL slug for empty passages."""
    if not link:
        return ""
    from urllib.parse import urlparse
    slug = urlparse(link).path.rsplit("/", 1)[-1]
    slug = re.sub(r"\.aspx$", "", slug, flags=re.I)
    slug = re.sub(r"-\d+$", "", slug)
    return slug.replace("-", " ").strip()


class BurstIndex:
    """Interface to SQLite FTS5 full-document and local-chunk BURST index."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        if not db_path.exists():
            raise FileNotFoundError(f"Database not found: {db_path}")
        self.conn = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)
        self.conn.row_factory = None
        # Load doc_idx to doc_id mapping
        cur = self.conn.cursor()
        self.doc_idx_to_id = {row[0]: row[1] for row in cur.execute("SELECT doc_idx, doc_id FROM doc_ids")}
        self.doc_id_to_idx = {row[1]: row[0] for row in cur.execute("SELECT doc_idx, doc_id FROM doc_ids")}
        self.chunk_to_doc = {row[0]: row[1] for row in cur.execute("SELECT rowid, doc_idx FROM chunk_owner")}
        cur.close()

    def close(self):
        if self.conn:
            self.conn.close()
            self.conn = None

    def retrieve_full(self, expression: str, limit: int = 500) -> List[Tuple[str, int, float]]:
        """
        Retrieve top parent documents by full-document BM25.
        Returns: [(doc_id, rank (1-indexed), score), ...]
        """
        if not expression:
            return []
        cur = self.conn.cursor()
        rows = cur.execute(
            "SELECT rowid, -bm25(docs_fts) AS score FROM docs_fts "
            "WHERE docs_fts MATCH ? ORDER BY bm25(docs_fts) LIMIT ?",
            (expression, limit)
        ).fetchall()
        cur.close()
        results = []
        for rank, (rowid, score) in enumerate(rows, 1):
            doc_idx = int(rowid) - 1
            doc_id = self.doc_idx_to_id[doc_idx]
            results.append((doc_id, rank, float(score)))
        return results

    def retrieve_local_distribution(
        self, expression: str, limit: int = 2000, second_weight: float = 0.3
    ) -> Tuple[List[Tuple[str, int, float]], Dict[str, Dict[str, Any]]]:
        """
        Retrieve top chunks by BM25 and aggregate per document.
        Returns:
            ranked_docs: [(doc_id, rank, local_score), ...]
            doc_evidence: {
                doc_id: {
                    "best": float,
                    "second": float,
                    "third": float,
                    "chunk_count": int,
                    "local_score": float,
                }
            }
        """
        if not expression:
            return [], {}
        cur = self.conn.cursor()
        rows = cur.execute(
            "SELECT rowid, -bm25(chunks_fts) AS score FROM chunks_fts "
            "WHERE chunks_fts MATCH ? ORDER BY bm25(chunks_fts) LIMIT ?",
            (expression, limit)
        ).fetchall()
        cur.close()

        chunk_scores_per_doc: Dict[int, List[float]] = {}
        for chunk_rowid, score in rows:
            doc_idx = self.chunk_to_doc[int(chunk_rowid)]
            chunk_scores_per_doc.setdefault(doc_idx, []).append(float(score))

        doc_evidence: Dict[str, Dict[str, Any]] = {}
        doc_aggregated: List[Tuple[str, float]] = []

        for doc_idx, scores in chunk_scores_per_doc.items():
            doc_id = self.doc_idx_to_id[doc_idx]
            best = scores[0] if len(scores) > 0 else 0.0
            second = scores[1] if len(scores) > 1 else 0.0
            third = scores[2] if len(scores) > 2 else 0.0
            chunk_count = len(scores)
            local_score = best + second_weight * second
            doc_evidence[doc_id] = {
                "best": best,
                "second": second,
                "third": third,
                "chunk_count": chunk_count,
                "local_score": local_score,
            }
            doc_aggregated.append((doc_id, local_score))

        doc_aggregated.sort(key=lambda x: (-x[1], x[0]))
        ranked_docs = [(doc_id, rank, score) for rank, (doc_id, score) in enumerate(doc_aggregated, 1)]
        return ranked_docs, doc_evidence

    def fuse(
        self,
        full_results: List[Tuple[str, int, float]],
        local_results: List[Tuple[str, int, float]],
        local_weight: float = 0.9,
        rrf_k: int = 20,
    ) -> List[Tuple[str, int, float]]:
        """
        Exact historical BURST RRF fusion.
        Returns: [(doc_id, rank, rrf_score), ...]
        """
        rf = {doc_id: rank for doc_id, rank, _ in full_results}
        rl = {doc_id: rank for doc_id, rank, _ in local_results}
        candidates = set(rf) | set(rl)
        global_weight = 1.0 - local_weight

        scored = []
        for d in candidates:
            r_full = rf.get(d, 100000)
            r_local = rl.get(d, 100000)
            score = (global_weight / (rrf_k + r_full)) + (local_weight / (rrf_k + r_local))
            scored.append((d, score))

        # Sort descending by score, tie-break by doc_id ascending
        scored.sort(key=lambda x: (-x[1], x[0]))
        return [(doc_id, rank, score) for rank, (doc_id, score) in enumerate(scored, 1)]
