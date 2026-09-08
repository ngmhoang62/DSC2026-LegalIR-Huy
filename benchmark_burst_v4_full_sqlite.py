"""Full LegalIR benchmark: 8,532 documents and all 7,000 labelled queries.

The disk-backed FTS5 index avoids keeping hundreds of thousands of passage
postings in RAM. BM25 and BURST-v4 use exactly the same tokenizer and corpus.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sqlite3
import statistics
import time
from pathlib import Path


TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def tokens(text):
    return TOKEN_RE.findall((text or "").lower())


def fts_query(text):
    # Deduplication prevents long natural-language questions from needlessly
    # repeating identical posting traversals. OR gives normal bag-of-words IR.
    seen, terms = set(), []
    for term in tokens(text):
        if term not in seen:
            seen.add(term)
            terms.append('"' + term.replace('"', '""') + '"')
    return " OR ".join(terms)


def _title_from_link(link):
    """20/8532 docs have an empty "passage" field; their "link" URL slug
    still carries a readable title -- see title_from_link() in
    run_burst_expanded_fusion_submission.py (duplicated here to avoid a
    circular import)."""
    if not link:
        return ""
    from urllib.parse import urlparse
    slug = urlparse(link).path.rsplit("/", 1)[-1]
    slug = re.sub(r"\.aspx$", "", slug, flags=re.I)
    slug = re.sub(r"-\d+$", "", slug)
    return slug.replace("-", " ").strip()


def load_dataset(data_dir):
    docs = []
    for path in sorted((data_dir / "selected-contexts").glob("context_*.json")):
        with path.open(encoding="utf-8") as f:
            row = json.load(f)
        docs.append((str(row["id"]), row.get("passage") or _title_from_link(row.get("link"))))
    with (data_dir / "train.json").open(encoding="utf-8") as f:
        raw = json.load(f)
    queries = {str(qid): (x["question"], {str(d) for d in x["answer"]})
               for qid, x in raw.items() if x.get("answer")}
    return docs, queries


def build_database(path, docs, chunk_size, overlap):
    if path.exists():
        conn = sqlite3.connect(path)
        meta = dict(conn.execute("SELECT key,value FROM metadata"))
        if (int(meta.get("documents", -1)) == len(docs)
                and int(meta.get("chunk_size", -1)) == chunk_size
                and int(meta.get("overlap", -1)) == overlap):
            print(f"Using cached FTS index: {path}")
            return conn, int(meta["chunks"])
        conn.close()
        path.unlink()

    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=OFF")
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.execute("PRAGMA cache_size=-262144")
    conn.execute("CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT)")
    # Contentless indexes retain postings/statistics but not a second copy of
    # the original 466 MB collection text.
    conn.execute("CREATE VIRTUAL TABLE docs_fts USING fts5(text, content='', tokenize='unicode61')")
    conn.execute("CREATE VIRTUAL TABLE chunks_fts USING fts5(text, content='', tokenize='unicode61')")
    conn.execute("CREATE TABLE chunk_owner(rowid INTEGER PRIMARY KEY, doc_idx INTEGER NOT NULL)")

    step = chunk_size - overlap
    chunk_rowid = 0
    started = time.perf_counter()
    for doc_idx, (_, text) in enumerate(docs):
        conn.execute("INSERT INTO docs_fts(rowid,text) VALUES(?,?)", (doc_idx + 1, text))
        toks = tokens(text)
        if not toks:
            toks = [""]
        for start in range(0, len(toks), step):
            part = toks[start:start + chunk_size]
            if not part:
                break
            chunk_rowid += 1
            conn.execute("INSERT INTO chunks_fts(rowid,text) VALUES(?,?)",
                         (chunk_rowid, " ".join(part)))
            conn.execute("INSERT INTO chunk_owner(rowid,doc_idx) VALUES(?,?)",
                         (chunk_rowid, doc_idx))
            if start + chunk_size >= len(toks):
                break
        if (doc_idx + 1) % 250 == 0:
            conn.commit()
            print(f"Indexed {doc_idx+1}/{len(docs)} docs, {chunk_rowid:,} chunks", flush=True)
    conn.executemany("INSERT INTO metadata(key,value) VALUES(?,?)", [
        ("documents", str(len(docs))), ("chunks", str(chunk_rowid)),
        ("chunk_size", str(chunk_size)), ("overlap", str(overlap))])
    conn.commit()
    print(f"Built FTS index in {time.perf_counter()-started:.1f}s")
    return conn, chunk_rowid


def retrieve_docs(conn, expression, limit):
    if not expression:
        return []
    rows = conn.execute(
        "SELECT rowid,-bm25(docs_fts) AS score FROM docs_fts "
        "WHERE docs_fts MATCH ? ORDER BY bm25(docs_fts) LIMIT ?",
        (expression, limit)).fetchall()
    return [(int(rowid)-1, float(score)) for rowid, score in rows]


def retrieve_local(conn, expression, limit, second_weight):
    if not expression:
        return []
    rows = conn.execute(
        "SELECT o.doc_idx,-bm25(chunks_fts) AS score FROM chunks_fts "
        "JOIN chunk_owner o ON o.rowid=chunks_fts.rowid "
        "WHERE chunks_fts MATCH ? ORDER BY bm25(chunks_fts) LIMIT ?",
        (expression, limit)).fetchall()
    best = {}
    for doc, score in rows:
        doc = int(doc); score = float(score)
        a, b = best.get(doc, (0.0, 0.0))
        if score > a:
            a, b = score, a
        elif score > b:
            b = score
        best[doc] = (a, b)
    return sorted(((d, a + second_weight*b) for d, (a, b) in best.items()),
                  key=lambda x: (-x[1], x[0]))


def fuse(full, local, local_weight=0.9, rrf_k=20):
    rf = {d: i+1 for i, (d, _) in enumerate(full)}
    rl = {d: i+1 for i, (d, _) in enumerate(local)}
    candidates = set(rf) | set(rl)
    return sorted(candidates, key=lambda d: (
        -((1-local_weight)/(rrf_k+rf.get(d, 100000))
          + local_weight/(rrf_k+rl.get(d, 100000))), d))


def metrics(rankings, queries):
    recall5 = recall100 = ndcg10 = mrr10 = precision5 = 0.0
    per_query = []
    for qid, (_, gold) in queries.items():
        got = rankings.get(qid, [])
        h5 = len(gold.intersection(got[:5]))
        r5 = h5 / len(gold)
        recall5 += r5; precision5 += h5 / 5
        recall100 += len(gold.intersection(got[:100])) / len(gold)
        dcg = sum(1/math.log2(i+2) for i, d in enumerate(got[:10]) if d in gold)
        ideal = sum(1/math.log2(i+2) for i in range(min(len(gold), 10)))
        ndcg10 += dcg/ideal if ideal else 0
        mrr10 += next((1/(i+1) for i, d in enumerate(got[:10]) if d in gold), 0)
        per_query.append(r5)
    n = len(queries); p = precision5/n; r = recall5/n
    f2 = 0 if 4*p+r == 0 else 5*p*r/(4*p+r)
    return {"n": n, "Recall@5": r, "Recall@100": recall100/n,
            "nDCG@10": ndcg10/n, "MRR@10": mrr10/n, "F2@5": f2}, per_query


def main():
    root = Path(__file__).resolve().parent
    data = root / "DSC2026-LegalIR-main" / "v4_run" / "public_test_dataset"
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk-size", type=int, default=500)
    ap.add_argument("--overlap", type=int, default=100)
    ap.add_argument("--doc-depth", type=int, default=500)
    ap.add_argument("--chunk-depth", type=int, default=2000)
    ap.add_argument("--checkpoint", type=Path,
                    default=root / "burst_v4_full_checkpoint.json")
    ap.add_argument("--db", type=Path, default=root / "benchmarks" / "legalir_full_fts.sqlite")
    ap.add_argument("--output", type=Path, default=root / "burst_v4_full_7000_results.json")
    args = ap.parse_args()
    docs, queries = load_dataset(data)
    print(f"Full benchmark: {len(docs)} documents, {len(queries)} queries")
    conn, n_chunks = build_database(args.db, docs, args.chunk_size, args.overlap)
    doc_ids = [d for d, _ in docs]
    bm_ranked, burst_ranked = {}, {}
    if args.checkpoint.exists():
        saved = json.loads(args.checkpoint.read_text(encoding="utf-8"))
        if (saved.get("doc_depth") == args.doc_depth
                and saved.get("chunk_depth") == args.chunk_depth):
            bm_ranked = saved.get("bm_ranked", {})
            burst_ranked = saved.get("burst_ranked", {})
            print(f"Resuming checkpoint: {len(bm_ranked)}/{len(queries)} queries", flush=True)
    latencies = []
    started = time.perf_counter()
    for i, (qid, (question, _)) in enumerate(queries.items(), 1):
        if qid in bm_ranked and qid in burst_ranked:
            continue
        qstart = time.perf_counter()
        expression = fts_query(question)
        full = retrieve_docs(conn, expression, args.doc_depth)
        local = retrieve_local(conn, expression, args.chunk_depth, second_weight=0.3)
        bm_ranked[qid] = [doc_ids[d] for d, _ in full[:100]]
        burst_ranked[qid] = [doc_ids[d] for d in fuse(full, local)[:100]]
        latencies.append((time.perf_counter()-qstart)*1000)
        if len(bm_ranked) % 100 == 0:
            args.checkpoint.write_text(json.dumps({"doc_depth": args.doc_depth,
                "chunk_depth": args.chunk_depth, "bm_ranked": bm_ranked,
                "burst_ranked": burst_ranked}), encoding="utf-8")
        if len(bm_ranked) % 250 == 0:
            print(f"Scored {i}/{len(queries)} queries", flush=True)
    bm, bpq = metrics(bm_ranked, queries)
    burst, upq = metrics(burst_ranked, queries)
    report = {"documents": len(docs), "chunks": n_chunks, "queries": len(queries),
              "index": "SQLite FTS5 unicode61; identical corpus/tokenizer for both methods",
              "BM25": {"metrics": bm},
              "BURST-v4": {"params": {"chunk_size": args.chunk_size,
                  "overlap": args.overlap, "second_weight": 0.3,
                  "local_weight": 0.9, "doc_depth": args.doc_depth,
                  "chunk_depth": args.chunk_depth}, "metrics": burst,
                  "mean_pipeline_ms": statistics.fmean(latencies),
                  "p95_pipeline_ms": sorted(latencies)[int(.95*len(latencies))]},
              "paired_Recall@5": {"BURST-v4_wins": sum(a>b for a,b in zip(upq,bpq)),
                  "ties": sum(a==b for a,b in zip(upq,bpq)),
                  "BM25_wins": sum(a<b for a,b in zip(upq,bpq))},
              "scoring_seconds": time.perf_counter()-started}
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    args.checkpoint.unlink(missing_ok=True)
    print(json.dumps(report, indent=2))
    conn.close()


if __name__ == "__main__":
    main()
