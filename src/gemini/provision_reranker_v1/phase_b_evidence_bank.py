"""Phase B: Build Label-Free Multi-Resolution Evidence Bank (Parallelized).

Constructs multi-resolution passage evidence for Top-16 parents across all 6,991 queries:
- R0: HUY_LOCKED (Huy top_passages, count=2, window=220, overlap=70)
- R1: LEXICAL_MEDIUM (window=160, overlap=40, count=3, compact prefix)
- R2: LOCAL_FOCUS (up to 3 locally coherent 80-140 word windows around highest query-token anchors)
- R3: STRUCTURAL (V2 structural-v3 chunk sliced from document text)

Deduplicates near-duplicates (Jaccard > 0.8) and caps at 7 unique passages per (query, parent).
Stores passages in evidence_bank.sqlite.
Writes EVIDENCE_BANK_MANIFEST.json.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sqlite3
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import numpy as np
from joblib import Parallel, delayed

# Ensure stdout uses UTF-8
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from common import (
    GIT_COMMIT_SHA,
    REPO_ROOT,
    RESULTS_DIR,
    core,
    load_baseline_data,
    sha256,
)

CACHE_DIR = RESULTS_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
SQLITE_PATH = CACHE_DIR / "evidence_bank.sqlite"

DATA_DIR = REPO_ROOT / "DSC2026-LegalIR-main/v4_run/public_test_dataset/selected-contexts"
STRUCTURAL_SELECTION_FILE = REPO_ROOT / "results/research_v2_forensic/V2_STRUCTURAL_SELECTION.jsonl"

STOPWORDS = {
    "bị", "các", "có", "của", "cho", "được", "để", "đến", "đối", "gì",
    "hay", "khi", "không", "là", "làm", "một", "nào", "những", "như",
    "phải", "ra", "sẽ", "theo", "thì", "thế", "trong", "trên", "từ",
    "và", "về", "với", "việc", "bao", "nhiêu", "người", "quy", "định",
}
SPACE_RE = re.compile(r"\S+", re.UNICODE)
WORD_RE = re.compile(r"\w+", re.UNICODE)


def tokens(text: str) -> List[str]:
    return [m.group(0).lower() for m in WORD_RE.finditer(text or "")]


def token_set_jaccard(tokens_a: Set[str], tokens_b: Set[str]) -> float:
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = len(tokens_a & tokens_b)
    union = len(tokens_a | tokens_b)
    return intersection / union if union > 0 else 0.0


def _title_from_link(link: str) -> str:
    if not link:
        return ""
    slug = link.strip("/").split("/")[-1]
    slug = re.sub(r"\.aspx$", "", slug, flags=re.I)
    slug = re.sub(r"-\d+$", "", slug)
    return slug.replace("-", " ").strip()


class DocumentStore:
    """Thread-safe and process-safe LRU context store."""
    def __init__(self, paths, cache_size=800):
        self.paths = {p.stem[len("context_"):]: p for p in paths}
        self.cache = {}
        self.cache_size = cache_size

    def __getitem__(self, doc: str) -> str:
        text = self.cache.get(doc)
        if text is None:
            p = self.paths.get(doc)
            if p is None:
                return ""
            row = json.loads(p.read_text(encoding="utf-8"))
            text = row.get("passage") or _title_from_link(row.get("link")) or ""
            if len(self.cache) >= self.cache_size:
                self.cache.clear()
            self.cache[doc] = text
        return text


def load_structural_selection() -> Dict[str, Dict[str, str]]:
    print("Loading structural selection mapping from V2_STRUCTURAL_SELECTION.jsonl...", flush=True)
    started = time.perf_counter()
    mapping = defaultdict(dict)
    with STRUCTURAL_SELECTION_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            qid = str(row["qid"])
            doc_ids = [str(x) for x in row["doc_ids"]]
            chunk_ids = row.get("chunk_ids", [])
            for doc_id, chunk_id in zip(doc_ids, chunk_ids):
                if chunk_id:
                    mapping[qid][doc_id] = str(chunk_id)
    elapsed = time.perf_counter() - started
    print(f"Loaded structural selection for {len(mapping)} queries in {elapsed:.2f}s", flush=True)
    return mapping


def render_r3(chunk_id: str, doc_text: str) -> List[Tuple[str, float]]:
    if not chunk_id or ":" not in chunk_id:
        words = SPACE_RE.findall(doc_text or "")
        return [(" ".join(words[:200]), 0.5)]
    parts = chunk_id.split(":")
    if len(parts) >= 5:
        try:
            start = int(parts[2])
            end = int(parts[3])
            if end > start and start < len(doc_text):
                chunk_str = doc_text[start:end].strip()
                if chunk_str:
                    return [(chunk_str, 1.0)]
        except ValueError:
            pass
    words = SPACE_RE.findall(doc_text or "")
    return [(" ".join(words[:200]), 0.5)]


def render_pair_passages(q_text: str, doc_text: str, chunk_id: str) -> List[Tuple[str, str, float]]:
    """Render and deduplicate passages for a single (query, document) pair."""
    words = SPACE_RE.findall(doc_text or "")
    if not words:
        return []

    q_toks = tokens(q_text)
    content = {t for t in q_toks if len(t) >= 3 and t not in STOPWORDS}
    numbers = {t for t in q_toks if any(c.isdigit() for c in t)}
    bigrams = {" ".join(q_toks[i:i + 2]) for i in range(len(q_toks) - 1)}
    targets = content | numbers

    # Pre-tokenize words
    word_tokens = [tokens(w) for w in words]
    has_target = [any(t in targets for t in toks) for toks in word_tokens]

    # R0: HUY_LOCKED
    r0 = []
    if len(words) <= 300:
        r0 = [(" ".join(words), 1.0)]
    else:
        header = " ".join(words[:70])
        scored = []
        for start in range(0, len(words), 150):
            end = min(start + 220, len(words))
            if not any(has_target[start:end]):
                scored.append((0.0, -start, start, end))
                if end == len(words):
                    break
                continue
            part_toks = [t for toks in word_tokens[start:end] for t in toks]
            token_set = set(part_toks)
            norm_text = " ".join(part_toks)
            cov = sum(1.0 + 0.20 * min(part_toks.count(t), 3) for t in content if t in token_set)
            num = 3.0 * sum(t in token_set for t in numbers)
            phr = 1.8 * sum(p in norm_text for p in bigrams)
            den = (cov + num + phr) / math.sqrt(max(len(part_toks), 1))
            scored.append((den, -start, start, end))
            if end == len(words):
                break
        scored.sort(reverse=True)
        seen = set()
        for den, neg_start, start, end in scored:
            part = " ".join(words[start:end])
            cand = part if start < 70 else header + "\n[ĐOẠN PHÙ HỢP]\n" + part
            if cand not in seen:
                seen.add(cand)
                r0.append((cand, float(den)))
            if len(r0) >= 2:
                break

    # R1: LEXICAL_MEDIUM
    r1 = []
    if len(words) <= 200:
        r1 = [(" ".join(words), 1.0)]
    else:
        prefix = " ".join(words[:28])
        scored = []
        for start in range(0, len(words), 120):
            end = min(start + 160, len(words))
            if not any(has_target[start:end]):
                scored.append((0.0, -start, start, end))
                if end == len(words):
                    break
                continue
            part_toks = [t for toks in word_tokens[start:end] for t in toks]
            token_set = set(part_toks)
            norm_text = " ".join(part_toks)
            cov = sum(1.0 + 0.20 * min(part_toks.count(t), 3) for t in content if t in token_set)
            num = 3.0 * sum(t in token_set for t in numbers)
            phr = 1.8 * sum(p in norm_text for p in bigrams)
            den = (cov + num + phr) / math.sqrt(max(len(part_toks), 1))
            scored.append((den, -start, start, end))
            if end == len(words):
                break
        scored.sort(reverse=True)
        seen = set()
        for den, neg_start, start, end in scored:
            part = " ".join(words[start:end])
            cand = part if start < 50 else prefix + "\n[ĐOẠN TRÍCH]\n" + part
            if cand not in seen:
                seen.add(cand)
                r1.append((cand, float(den)))
            if len(r1) >= 3:
                break

    # R2: LOCAL_FOCUS
    r2 = []
    if len(words) <= 130:
        r2 = [(" ".join(words), 1.0)]
    elif targets:
        scores = np.zeros(len(words), dtype=np.float32)
        for i, toks in enumerate(word_tokens):
            scores[i] = sum(3.0 if t in numbers else 1.0 for t in toks if t in targets)
        if np.sum(scores) > 0:
            sorted_anchors = np.argsort(-scores)
            selected_spans = []
            seen_indices = set()
            for idx in sorted_anchors:
                if scores[idx] <= 0:
                    break
                if idx in seen_indices:
                    continue
                start = max(0, idx - 50)
                end = min(len(words), start + 100)
                overlap = False
                for s, e, _ in selected_spans:
                    intersect = max(0, min(end, e) - max(start, s))
                    if intersect > 30:
                        overlap = True
                        break
                if not overlap:
                    window_score = float(np.sum(scores[start:end]) / math.sqrt(max(end - start, 1)))
                    selected_spans.append((start, end, window_score))
                    for k in range(start, end):
                        seen_indices.add(k)
                if len(selected_spans) >= 3:
                    break
            selected_spans.sort(key=lambda x: x[0])
            r2 = [(" ".join(words[s:e]), sc) for s, e, sc in selected_spans]

    # R3: STRUCTURAL
    r3 = render_r3(chunk_id, doc_text)

    # Collect raw candidates
    raw = []
    for p, s in r0:
        raw.append(("R0_HUY_LOCKED", p, s))
    for p, s in r1:
        raw.append(("R1_LEXICAL_MEDIUM", p, s))
    for p, s in r2:
        raw.append(("R2_LOCAL_FOCUS", p, s))
    for p, s in r3:
        raw.append(("R3_STRUCTURAL", p, s))

    # Deduplicate with Token Jaccard > 0.8
    accepted = []
    accepted_toks = []
    for r_name, p_text, p_score in raw:
        if not p_text.strip():
            continue
        toks = set(tokens(p_text))
        is_dup = False
        for acc in accepted_toks:
            if token_set_jaccard(toks, acc) > 0.80:
                is_dup = True
                break
        if not is_dup:
            accepted.append((r_name, p_text, p_score))
            accepted_toks.append(toks)
        if len(accepted) >= 7:
            break
    return accepted


def process_query_batch(qids: List[str], questions: Dict[str, str], base_orders: Dict[str, List[str]], structural_map: Dict[str, Dict[str, str]]) -> List[Tuple]:
    """Worker function to process a batch of queries with an independent DocumentStore."""
    doc_store = DocumentStore(sorted(DATA_DIR.glob("context_*.json")), cache_size=800)
    records = []
    for qid in qids:
        q_text = questions[qid]
        top16 = base_orders[qid][:16]
        for doc_id in top16:
            doc_text = doc_store[doc_id]
            chunk_id = structural_map.get(qid, {}).get(doc_id, "")
            passages = render_pair_passages(q_text, doc_text, chunk_id)
            for idx, (renderer, p_text, p_score) in enumerate(passages):
                h = hashlib.sha256(p_text.encode("utf-8")).hexdigest()[:16]
                p_id = f"{qid}_{doc_id}_{renderer}_{idx}"
                w_count = len(SPACE_RE.findall(p_text))
                records.append((p_id, qid, doc_id, renderer, idx, h, w_count, float(p_score), p_text))
    return records


def build_evidence_bank():
    started = time.perf_counter()
    print("=" * 70, flush=True)
    print("PHASE B: BUILD LABEL-FREE MULTI-RESOLUTION EVIDENCE BANK (PARALLEL)", flush=True)
    print("=" * 70, flush=True)

    folds, fold_for, pools, questions, golds, e5_orders, e5_scores, dup, base_orders, base_scores = load_baseline_data()
    structural_map = load_structural_selection()

    all_qids = sorted(base_orders, key=int)
    print(f"\nProcessing Top-16 parents across {len(all_qids)} queries on 8 CPU cores...", flush=True)

    # Initialize SQLite database
    if SQLITE_PATH.exists():
        SQLITE_PATH.unlink()
    con = sqlite3.connect(SQLITE_PATH)
    cur = con.cursor()
    cur.execute("PRAGMA journal_mode=WAL")
    cur.execute("PRAGMA synchronous=NORMAL")
    cur.execute("""
        CREATE TABLE passages (
            passage_id TEXT PRIMARY KEY,
            qid TEXT,
            doc_id TEXT,
            renderer TEXT,
            passage_index INTEGER,
            text_hash TEXT,
            word_count INTEGER,
            lexical_score REAL,
            text TEXT
        )
    """)
    cur.execute("CREATE INDEX idx_passages_qid_doc ON passages(qid, doc_id)")
    cur.execute("CREATE INDEX idx_passages_qid ON passages(qid)")
    con.commit()

    # Split qids into batches of 25 queries for smooth parallel dispatch
    chunk_size = 25
    query_batches = [all_qids[i:i + chunk_size] for i in range(0, len(all_qids), chunk_size)]
    print(f"Total batches: {len(query_batches)} (chunk size {chunk_size})", flush=True)

    total_pairs = 0
    total_passages = 0
    renderer_counter = Counter()
    passages_per_pair_map = defaultdict(int)

    # Parallel processing across 8 cores
    n_jobs = 8
    t_start = time.perf_counter()
    results_generator = Parallel(n_jobs=n_jobs, return_as="generator", batch_size=2)(
        delayed(process_query_batch)(batch, questions, base_orders, structural_map)
        for batch in query_batches
    )

    batch_buffer = []
    completed_queries = 0

    for batch_records in results_generator:
        batch_buffer.extend(batch_records)
        completed_queries += chunk_size
        if completed_queries > len(all_qids):
            completed_queries = len(all_qids)

        # Write to SQLite in chunks of 5000 records
        if len(batch_buffer) >= 5000:
            cur.executemany("INSERT INTO passages VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", batch_buffer)
            con.commit()
            for r in batch_buffer:
                renderer_counter[r[3]] += 1
                passages_per_pair_map[(r[1], r[2])] += 1
            total_passages += len(batch_buffer)
            batch_buffer = []

        if completed_queries % 500 == 0 or completed_queries == len(all_qids):
            elapsed = time.perf_counter() - t_start
            q_rate = completed_queries / max(elapsed, 0.01)
            remaining_q = len(all_qids) - completed_queries
            eta_min = (remaining_q / max(q_rate, 0.01)) / 60.0
            print(f"Progress: {completed_queries}/{len(all_qids)} queries ({total_passages} passages written) | Elapsed: {elapsed:.1f}s | Rate: {q_rate:.1f} q/s | ETA: {eta_min:.1f} min", flush=True)

    if batch_buffer:
        cur.executemany("INSERT INTO passages VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", batch_buffer)
        con.commit()
        for r in batch_buffer:
            renderer_counter[r[3]] += 1
            passages_per_pair_map[(r[1], r[2])] += 1
        total_passages += len(batch_buffer)
        batch_buffer = []

    con.close()

    total_pairs = len(all_qids) * 16
    passages_per_pair_counts = list(passages_per_pair_map.values())
    total_time = time.perf_counter() - started

    print(f"\nEvidence Bank successfully generated in {total_time:.1f}s ({total_time / 60.0:.2f} min)!", flush=True)
    print(f"Total Pairs: {total_pairs}")
    print(f"Total Passages: {total_passages} (mean {total_passages / total_pairs:.2f} per pair)")
    print(f"Renderer breakdown: {dict(renderer_counter)}")
    print(f"Database Size: {SQLITE_PATH.stat().st_size / (1024 * 1024):.1f} MB")

    # Manifest
    manifest = {
        "schema_version": "dsc2026.gemini.provision_reranker_v1.evidence_bank.v1",
        "git_commit_sha": GIT_COMMIT_SHA,
        "total_queries": len(all_qids),
        "parents_per_query": 16,
        "total_pairs": total_pairs,
        "total_passages": total_passages,
        "passages_per_parent_distribution": {
            "min": int(np.min(passages_per_pair_counts)) if passages_per_pair_counts else 0,
            "max": int(np.max(passages_per_pair_counts)) if passages_per_pair_counts else 0,
            "mean": float(np.mean(passages_per_pair_counts)) if passages_per_pair_counts else 0.0,
            "median": float(np.median(passages_per_pair_counts)) if passages_per_pair_counts else 0.0,
        },
        "renderer_counts": dict(renderer_counter),
        "database_file": str(SQLITE_PATH.relative_to(REPO_ROOT)),
        "database_sha256": sha256(SQLITE_PATH),
        "database_bytes": SQLITE_PATH.stat().st_size,
        "execution_seconds": total_time,
    }

    manifest_path = RESULTS_DIR / "EVIDENCE_BANK_MANIFEST.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest written to {manifest_path}", flush=True)


if __name__ == "__main__":
    build_evidence_bank()
