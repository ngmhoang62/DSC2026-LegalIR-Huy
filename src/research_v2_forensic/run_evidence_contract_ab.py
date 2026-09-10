"""Prepare and run the zero-shot Jina-v2 full evidence-contract A/B.

Arm lexical: exact Huy top_passages(count=2) over the canonical parent text.
Arm structural: best query-conditioned structural-v3 chunk already selected by
the frozen E5 source, falling back to the frozen BM25 best structural chunk for
BM25-only parents.  Both arms share candidates, model, max aggregation and folds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def patch_transformers_v5() -> None:
    """Restore the helper removed in transformers v5 for Jina-v2 remote code."""
    import transformers.models.xlm_roberta.modeling_xlm_roberta as module
    if hasattr(module, "create_position_ids_from_input_ids"):
        return

    def create_position_ids(input_ids, padding_idx, past_key_values_length=0):
        mask = input_ids.ne(padding_idx).int()
        positions = (torch.cumsum(mask, dim=1) + past_key_values_length) * mask
        return positions.long() + padding_idx

    module.create_position_ids_from_input_ids = create_position_ids


def load_model(path: Path, batch_size: int):
    patch_transformers_v5()
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True,
                                               fix_mistral_regex=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        path, trust_remote_code=True, dtype=torch.float16).eval().to("cuda")
    model._tokenizer = tokenizer
    return model, batch_size


def canonical_answers(train: dict, exclusions: list[dict]) -> dict[str, set[str]]:
    alias = {str(row["doc_id"]): str(row["duplicate_retained_id"])
             for row in exclusions if row.get("duplicate_retained_id")}
    empty = {str(row["doc_id"]) for row in exclusions
             if "empty_passage" in row.get("reasons", [])}
    return {str(q): {alias.get(str(x), str(x)) for x in row["answer"]} - empty
            for q, row in train.items()}


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)


def prepare_selection(args: argparse.Namespace) -> None:
    pools = {str(row["qid"]): row for row in read_jsonl(args.pool)}
    doc_chunks = json.loads(args.doc_to_chunks.read_text(encoding="utf-8"))
    bm25 = {}
    for row in read_jsonl(args.bm25_rankings):
        qid = str(row["qid"])
        if qid not in pools:
            continue
        bm25[qid] = {str(doc["doc_id"]): str(doc["best_chunk_id"])
                     for doc in row["documents"]}
    output = args.output_dir / "V2_STRUCTURAL_SELECTION.jsonl"
    rows = missing = 0
    selector_counts = Counter()
    with output.open("w", encoding="utf-8", newline="\n") as sink:
        for row in read_jsonl(args.historical_candidates):
            qid = str(row["qid"])
            pool = pools.get(qid)
            if pool is None:
                continue
            source = {str(c["doc_id"]): c.get("sources", {}) for c in row["candidates"]}
            chunks = []
            selectors = []
            for doc in pool["doc_ids"]:
                info = source.get(str(doc), {})
                evidence = info.get("e5", {}).get("evidence", [])
                if evidence:
                    chunks.append(str(evidence[0]["chunk_id"]))
                    selectors.append("e5_top1_structural")
                else:
                    chunk_id = bm25.get(qid, {}).get(str(doc))
                    if not chunk_id:
                        fallback = doc_chunks.get(str(doc), [])
                        if not fallback:
                            missing += 1
                            chunks.append(None)
                            selectors.append("missing")
                        else:
                            chunks.append(str(fallback[0]))
                            selectors.append("structural_first_chunk_fallback")
                    else:
                        chunks.append(chunk_id)
                        selectors.append("bm25_top1_structural")
            sink.write(canonical_json({"qid": qid, "doc_ids": pool["doc_ids"],
                                       "chunk_ids": chunks, "selectors": selectors}) + "\n")
            selector_counts.update(selectors)
            rows += 1
    if rows != 6991 or missing:
        raise RuntimeError(f"structural selection incomplete rows={rows} missing={missing}")
    manifest = {
        "schema_version": "dsc2026.research_v2.structural_selection.v1",
        "status": "SEALED", "rows": rows,
        "contract": "E5 top-1 structural-v3 chunk; BM25 top-1 structural-v3; deterministic first structural chunk for residual BM25-only parents",
        "selector_counts": dict(selector_counts),
        "selection_sha256": sha256(output),
        "inputs": {"pool": sha256(args.pool),
                   "historical_candidates": sha256(args.historical_candidates),
                   "bm25_rankings": sha256(args.bm25_rankings),
                   "doc_to_chunks": sha256(args.doc_to_chunks)},
    }
    (args.output_dir / "V2_STRUCTURAL_SELECTION_MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


class ChunkReader:
    def __init__(self, chunks: Path, lookup: Path):
        chunks = chunks.resolve()
        lookup = lookup.resolve()
        self.source = chunks.open("rb")
        self.db = sqlite3.connect(f"file:{lookup.as_posix()}?mode=ro&immutable=1", uri=True)

    def load(self, chunk_ids: list[str]) -> dict[str, dict]:
        unique = sorted(set(chunk_ids))
        offsets = {}
        for start in range(0, len(unique), 900):
            batch = unique[start:start + 900]
            marks = ",".join("?" for _ in batch)
            offsets.update(self.db.execute(
                f"SELECT chunk_id,byte_offset FROM chunk_offsets WHERE chunk_id IN ({marks})",
                batch).fetchall())
        if len(offsets) != len(unique):
            raise RuntimeError(f"missing structural chunks: {len(unique) - len(offsets)}")
        result = {}
        for chunk_id, offset in sorted(offsets.items(), key=lambda x: x[1]):
            self.source.seek(int(offset))
            row = json.loads(self.source.readline())
            if str(row["chunk_id"]) != chunk_id:
                raise RuntimeError("stale structural chunk lookup")
            result[chunk_id] = row
        return result

    def close(self):
        self.db.close()
        self.source.close()


def open_score_db(path: Path) -> sqlite3.Connection:
    db = sqlite3.connect(path)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("CREATE TABLE IF NOT EXISTS scores(qid TEXT,doc_id TEXT,arm TEXT,score REAL,PRIMARY KEY(qid,doc_id,arm))")
    db.execute("CREATE TABLE IF NOT EXISTS progress(qid TEXT,arm TEXT,seconds REAL,pairs INTEGER,peak_mib REAL,PRIMARY KEY(qid,arm))")
    db.commit()
    return db


def score(args: argparse.Namespace) -> None:
    sys.path.insert(0, str(args.huy_root))
    from benchmark_jina_reranker_holdouts import top_passages
    from run_burst_expanded_fusion_submission import DocumentStore, prefetch

    model, batch_size = load_model(args.model, args.batch_size)
    # The canonical corpus is only 8,507 parents (~0.46 GiB serialized).  Keep
    # all parents resident after first access; a 512-entry LRU thrashes under
    # dense-ranking order and wastes hours without changing the experiment.
    docs = DocumentStore(sorted(args.contexts.glob("context_*.json")), cache_size=9000)
    selection = {str(row["qid"]): row for row in read_jsonl(args.selection)}
    db = open_score_db(args.score_db)
    complete = {(q, arm) for q, arm in db.execute("SELECT qid,arm FROM progress")}
    local = threading.local()
    readers = []
    readers_lock = threading.Lock()

    def prepare(row):
        qid, query = str(row["qid"]), str(row["query"])
        docs_for_q = [str(x) for x in row["doc_ids"]]
        items = []
        if (qid, "lexical") not in complete:
            owners, passages = [], []
            for doc in docs_for_q:
                for passage in top_passages(query, docs[doc], count=2):
                    owners.append(doc); passages.append(passage)
            items.append(("lexical", owners, passages))
        if (qid, "structural") not in complete:
            selected = selection[qid]
            if selected["doc_ids"] != docs_for_q:
                raise RuntimeError(f"candidate membership changed for {qid}")
            if not hasattr(local, "reader"):
                local.reader = ChunkReader(args.chunks, args.chunk_lookup)
                with readers_lock:
                    readers.append(local.reader)
            records = local.reader.load([str(x) for x in selected["chunk_ids"]])
            owners = docs_for_q
            passages = [str(records[str(chunk_id)]["retrieval_text"])
                        for chunk_id in selected["chunk_ids"]]
            items.append(("structural", owners, passages))
        return qid, query, docs_for_q, items
    started = time.perf_counter()
    done = 0
    torch.cuda.reset_peak_memory_stats()
    try:
        rows = read_jsonl(args.pool)
        if args.limit is not None:
            rows = (row for index, row in enumerate(rows) if index < args.limit)
        with ThreadPoolExecutor(max_workers=2) as pool:
            for qid, query, docs_for_q, items in prefetch(pool, rows, prepare, ahead=2):
              for arm, owners, passages in items:
                before = time.perf_counter()
                raw = model.compute_score([(query, p) for p in passages],
                                          batch_size=batch_size, max_length=512)
                if isinstance(raw, float):
                    raw = [raw]
                parent = {}
                for doc, value in zip(owners, raw):
                    parent[doc] = max(parent.get(doc, -1e9), float(value))
                if set(parent) != set(docs_for_q):
                    raise RuntimeError(f"parent score mismatch {qid} {arm}")
                elapsed = time.perf_counter() - before
                peak = torch.cuda.max_memory_allocated() / 2**20
                with db:
                    db.executemany("INSERT OR REPLACE INTO scores VALUES(?,?,?,?)",
                                   [(qid, doc, arm, value) for doc, value in parent.items()])
                    db.execute("INSERT OR REPLACE INTO progress VALUES(?,?,?,?,?)",
                               (qid, arm, elapsed, len(passages), peak))
                done += 1
                if done % 50 == 0:
                    rate = (time.perf_counter() - started) / done
                    total = 2 * (args.limit if args.limit is not None else 6991)
                    print(f"completed={done} remaining={max(0, total-len(complete)-done)} sec/item={rate:.3f} peak_mib={peak:.1f}", flush=True)
    finally:
        for reader in readers:
            reader.close()
        db.close()


def metric(pred: list[str], gold: set[str]) -> tuple[float, float]:
    hits = len(set(pred[:5]) & gold)
    return hits / len(gold), hits / 5.0


def evaluate(args: argparse.Namespace) -> None:
    train = json.loads(args.train.read_text(encoding="utf-8"))
    answers = canonical_answers(train, json.loads(args.exclusions.read_text(encoding="utf-8")))
    folds = json.loads(args.folds.read_text(encoding="utf-8"))["folds"]
    fold_for = {str(q): fold for fold, ids in folds.items() for q in ids}
    lengths = {}
    for path in args.contexts.glob("context_*.json"):
        row = json.loads(path.read_text(encoding="utf-8"))
        lengths[str(row["id"])] = len(str(row.get("passage") or ""))
    db = sqlite3.connect(f"file:{args.score_db.as_posix()}?mode=ro", uri=True)
    scores = {(str(q), str(d), str(a)): float(s) for q, d, a, s in
              db.execute("SELECT qid,doc_id,arm,score FROM scores")}
    progress = list(db.execute("SELECT arm,SUM(seconds),SUM(pairs),MAX(peak_mib),COUNT(*) FROM progress GROUP BY arm"))
    db.close()
    arms = ("lexical", "structural")
    values = {a: [] for a in arms}
    fold_values = {a: {f: [] for f in folds} for a in arms}
    long_values = {a: [] for a in arms}
    boundary = {a: [] for a in arms}
    predictions = {a: {} for a in arms}
    for row in read_jsonl(args.pool):
        qid, docs = str(row["qid"]), [str(x) for x in row["doc_ids"]]
        gold = answers[qid]
        if not gold:
            continue
        ordered_by_arm = {}
        for arm in arms:
            ordered = sorted(docs, key=lambda d: (-scores[(qid, d, arm)], d))
            ordered_by_arm[arm] = ordered
            predictions[arm][qid] = ordered[:5]
            r, p = metric(ordered, gold)
            values[arm].append((r, p))
            fold_values[arm][fold_for[qid]].append((r, p))
            if any(lengths.get(g, 0) >= 20000 for g in gold & set(docs)):
                long_values[arm].append((r, p))
        # Compare both renderers on the exact same boundary pair universe: the
        # union of their base-Jina ranks 4--20.  This avoids changing the pair
        # population between arms while still measuring the actual CE boundary.
        band = list(dict.fromkeys(ordered_by_arm["lexical"][3:20] +
                                  ordered_by_arm["structural"][3:20]))
        positives = [d for d in band if d in gold]
        negatives = [d for d in band if d not in gold]
        for arm in arms:
            boundary[arm].extend(scores[(qid, pos, arm)] > scores[(qid, neg, arm)]
                                 for pos in positives for neg in negatives)
    result = {}
    prediction_hashes = {}
    for arm in arms:
        result[arm] = {
            "recall_at_5": float(np.mean([x[0] for x in values[arm]])),
            "precision_at_5": float(np.mean([x[1] for x in values[arm]])),
            "per_fold_recall": {f: float(np.mean([x[0] for x in rows]))
                                for f, rows in fold_values[arm].items()},
            "long_parent_slice": {"definition": "at least one reachable gold parent >=20000 characters",
                                  "queries": len(long_values[arm]),
                                  "recall_at_5": float(np.mean([x[0] for x in long_values[arm]]))},
            "boundary_pair_accuracy": {"band": "base-Jina ranks 4-20",
                                       "shared_pair_universe": "union of lexical and structural base-Jina ranks 4-20",
                                       "pairs": len(boundary[arm]),
                                       "accuracy": float(np.mean(boundary[arm])) if boundary[arm] else None},
        }
        prediction_path = args.output_dir / f"V2_ZERO_SHOT_{arm.upper()}_PREDICTIONS.jsonl"
        with prediction_path.open("w", encoding="utf-8", newline="\n") as sink:
            for qid in sorted(predictions[arm], key=lambda x: int(x)):
                sink.write(canonical_json({"qid": qid, "top5": predictions[arm][qid]}) + "\n")
        prediction_hashes[arm] = sha256(prediction_path)
    query_deltas = {q: metric(predictions["structural"][q], answers[q])[0] -
                       metric(predictions["lexical"][q], answers[q])[0]
                    for q in predictions["lexical"]}
    fold_delta = {f: result["structural"]["per_fold_recall"][f] -
                     result["lexical"]["per_fold_recall"][f] for f in folds}
    delta = result["structural"]["recall_at_5"] - result["lexical"]["recall_at_5"]
    nonnegative = sum(v >= 0 for v in fold_delta.values())
    long_delta = (result["structural"]["long_parent_slice"]["recall_at_5"] -
                  result["lexical"]["long_parent_slice"]["recall_at_5"])
    runtime = {arm: {"seconds": sec, "pairs": pairs, "peak_mib": peak, "query_arm_rows": rows}
               for arm, sec, pairs, peak, rows in progress}
    if delta >= 0.003 and nonnegative >= 4:
        winner, decision = "structural", "PASS_POOLED_AND_FOLD_GATE"
    elif delta > 0 and long_delta > 0 and nonnegative == 5:
        winner, decision = "structural", "PASS_LONG_DOCUMENT_NO_OVERALL_HARM"
    elif (abs(delta) <= 0.001 and min(fold_delta.values()) >= -0.002 and
          runtime.get("structural", {}).get("pairs", float("inf")) <
          0.60 * runtime.get("lexical", {}).get("pairs", 0)):
        winner, decision = "structural", "COST_DOMINANT_STATISTICAL_TIE"
    else:
        winner, decision = "lexical", "SIMPLICITY_TIE_OR_STRUCTURAL_KILL"
    report = {
        "schema_version": "dsc2026.research_v2.evidence_contract_ab.v1",
        "status": "COMPLETE", "causal_scope": "FULL_EVIDENCE_CONTRACT_AB_NOT_CHUNKING_ONLY",
        "held_constant": ["candidate membership", "query-document pairs", "base Jina-v2 weights",
                          "max parent aggregation", "V2 folds", "max_length=512"],
        "changed": {
            "lexical": "Huy whitespace windows + query lexical selector + two passages",
            "structural": "structural-v3 segmentation/prefix composition + frozen E5/BM25 selector + one passage",
        },
        "metrics": result, "structural_minus_lexical": {
            "pooled_recall": delta, "per_fold": fold_delta,
            "long_parent_recall": long_delta,
            "wins": sum(v > 0 for v in query_deltas.values()),
            "losses": sum(v < 0 for v in query_deltas.values()),
            "ties": sum(v == 0 for v in query_deltas.values()),
        },
        "preregistered_gate": "structural wins for delta>=0.003 with >=4 nonnegative folds; positive long-doc delta with 5 nonnegative folds; or |delta|<=0.001, worst fold>=-0.002 and <60% lexical CE pairs. Otherwise lexical wins.",
        "winner": winner, "decision": decision, "runtime": runtime,
        "prediction_artifacts_sha256": prediction_hashes,
        "inputs_sha256": {"pool": sha256(args.pool), "selection": sha256(args.selection),
                          "model_weights": sha256(args.model / "model.safetensors"),
                          "folds": sha256(args.folds), "score_db": sha256(args.score_db)},
    }
    output = args.output_dir / "EVIDENCE_CONTRACT_MATCHED_AB.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lock = {"schema_version": "dsc2026.research_v2.evidence_renderer_lock.v1",
            "status": "SEALED", "winner": winner, "report_sha256": sha256(output),
            "contract": report["changed"][winner] + "; parent aggregation=max; max_length=512"}
    (args.output_dir / "EVIDENCE_RENDERER_LOCK.json").write_text(
        json.dumps(lock, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(output), "winner": winner, "delta": delta,
                      "decision": decision, "runtime": runtime}, indent=2))


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=("prepare-selection", "score", "evaluate"))
    p.add_argument("--output-dir", type=Path, default=Path("results/research_v2_forensic"))
    p.add_argument("--pool", type=Path, default=Path("results/research_v2_forensic/V2_CANDIDATE_POOL.jsonl"))
    p.add_argument("--selection", type=Path, default=Path("results/research_v2_forensic/V2_STRUCTURAL_SELECTION.jsonl"))
    p.add_argument("--historical-candidates", type=Path, required=True)
    p.add_argument("--bm25-rankings", type=Path, required=True)
    p.add_argument("--model", type=Path, default=Path("cache/research_v2_forensic/models/jina-reranker-v2-base-multilingual"))
    p.add_argument("--contexts", type=Path, required=True)
    p.add_argument("--chunks", type=Path, required=True)
    p.add_argument("--doc-to-chunks", type=Path, required=True)
    p.add_argument("--chunk-lookup", type=Path, default=Path("cache/research_v2_forensic/structural_chunk_lookup/chunk_offsets.sqlite"))
    p.add_argument("--score-db", type=Path, default=Path("cache/research_v2_forensic/evidence_ab_scores.sqlite"))
    p.add_argument("--train", type=Path, required=True)
    p.add_argument("--exclusions", type=Path, required=True)
    p.add_argument("--folds", type=Path, default=Path("results/research_v2_forensic/V2_FOLDS.json"))
    p.add_argument("--huy-root", type=Path, default=Path(__file__).resolve().parents[2])
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--limit", type=int, default=None,
                   help="Score only the first N pool rows (smoke tests only).")
    return p


if __name__ == "__main__":
    args = parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.stage == "prepare-selection":
        prepare_selection(args)
    elif args.stage == "score":
        score(args)
    else:
        evaluate(args)
