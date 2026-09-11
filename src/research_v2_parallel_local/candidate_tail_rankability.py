"""Local-only V2 candidate-tail rankability diagnostic.

This program reads sealed Research V2 and historical LegalIR artifacts but
writes only to the research_v2_parallel_local namespace.  It never modifies the
immutable Kaggle pool or any training artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sqlite3
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch


EXPECTED = {
    "folds": "94ad5c6d5e582ced5eec8d2c3c15f938454c17e713614391091e72abea9aba19",
    "historical_candidates": "b86bfbca5837bd4c8d295fd9943e6469580a5aba601efa22c4ab285325d50435",
    "dense_candidates": "dce4a38bae5b14ad37acd73a821b1c04897356fd010da01be1f123ccea3619a4",
    "current_pool": "96a44e66549cc211e1f9d0fabb84fc825db3f21f32d5b349eeca3b1c0413e277",
    "train": "c39cde9e74977e350f1456e7d487aafe67d2bcbaa4fa26fcabd557fe635635b7",
    "exclusions": "d10aef3d891746cd9f874b32e9cb20940cb4fef0928c0fe239fdb7ca16337616",
    "model_weights": "ab2595ab9f34bdeffe645431d64c6e4aabe2ff5a57cfcacfef0727a97434238f",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def json_line(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)


def verify(path: Path, key: str) -> None:
    actual = sha256(path)
    if actual != EXPECTED[key]:
        raise RuntimeError(f"{key} checksum mismatch: {actual}")


def canonical_answers(train: dict, exclusions: list[dict]) -> dict[str, set[str]]:
    alias = {str(row["doc_id"]): str(row["duplicate_retained_id"])
             for row in exclusions if row.get("duplicate_retained_id")}
    empty = {str(row["doc_id"]) for row in exclusions
             if "empty_passage" in row.get("reasons", [])}
    return {str(q): {alias.get(str(x), str(x)) for x in row["answer"]} - empty
            for q, row in train.items()}


def doc_type(text: str) -> str:
    """Exact logic of Huy tune_doctype_features.doc_type, copied for isolation."""
    head = text[:400].upper()
    if re.search(r"LUẬT\s+SỐ|LUẬT\s*[:\n]", head) or "QUỐC HỘI" in head[:100]:
        return "LUAT"
    if "NGHỊ ĐỊNH" in head:
        return "NGHIDINH"
    if "THÔNG TƯ" in head:
        return "THONGTU"
    if "QUYẾT ĐỊNH" in head:
        return "QUYETDINH"
    if re.search(r"V/V|CÔNG VĂN", head):
        return "CONGVAN"
    return "KHAC"


def length_bin(n: int) -> str:
    if n < 5_000:
        return "lt5k"
    if n < 20_000:
        return "5k_20k"
    if n < 100_000:
        return "20k_100k"
    return "ge100k"


def source_lists(row: dict) -> tuple[list[str], list[str], dict[str, dict]]:
    e5, bm25, meta = {}, {}, {}
    for cand in row["candidates"]:
        doc = str(cand["doc_id"])
        sources = cand.get("sources", {})
        info = {}
        if "e5" in sources:
            rank = int(sources["e5"]["rank"])
            e5[rank] = doc
            info["e5_rank"] = rank
        if "bm25" in sources:
            rank = int(sources["bm25"]["rank"])
            bm25[rank] = doc
            info["bm25_rank"] = rank
        meta[doc] = info
    return ([e5[x] for x in sorted(e5)], [bm25[x] for x in sorted(bm25)], meta)


def prepare(args: argparse.Namespace) -> None:
    for path, key in ((args.folds, "folds"), (args.historical_candidates, "historical_candidates"),
                      (args.dense_candidates, "dense_candidates"),
                      (args.current_pool, "current_pool"), (args.train, "train"),
                      (args.exclusions, "exclusions")):
        verify(path, key)
    train = json.loads(args.train.read_text(encoding="utf-8"))
    answers = canonical_answers(train, json.loads(args.exclusions.read_text(encoding="utf-8")))
    current = {str(r["qid"]): r for r in read_jsonl(args.current_pool)}
    folds = json.loads(args.folds.read_text(encoding="utf-8"))["folds"]
    fold_for = {str(q): f for f, ids in folds.items() for q in ids}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pool_path = args.output_dir / "V2_CANDIDATE_TAIL_POOLS.jsonl"

    trace_rows = []
    ceilings = Counter()
    size_values = defaultdict(list)
    fold_sums = defaultdict(Counter)
    fold_counts = Counter()
    query_count = 0
    with pool_path.open("w", encoding="utf-8", newline="\n") as sink:
        historical_rows = read_jsonl(args.historical_candidates)
        dense_rows = read_jsonl(args.dense_candidates)
        for row, dense_row in zip(historical_rows, dense_rows, strict=True):
            qid = str(row["qid"])
            if str(dense_row["qid"]) != qid:
                raise RuntimeError(f"EXP-021/022 qid order mismatch at {qid}")
            if qid not in current:
                continue
            e5, bm25, meta = source_lists(row)
            dense_rank = {str(c["doc_id"]): int(c["rank"])
                          for c in dense_row["candidates"]}
            # EXP-022 deliberately clears the source payload on E5 backfill
            # rows. Recover their exact provenance from its hashed EXP-021
            # parent instead of misclassifying them as absent from all sources.
            for doc, rank in dense_rank.items():
                meta.setdefault(doc, {})["e5_rank"] = rank
            current_docs = [str(x) for x in current[qid]["doc_ids"]]
            rebuilt_current = list(dict.fromkeys(e5[:50] + bm25[:10]))
            if rebuilt_current != current_docs:
                raise RuntimeError(f"current pool contract mismatch at qid={qid}")
            tail50 = list(dict.fromkeys(e5[:50] + bm25[:50]))
            # The historical EXP-022 row is the complete cached bounded tail:
            # E5 anchor, novel BM25@50, then available E5 tail backfill.
            full_cached = [str(c["doc_id"]) for c in row["candidates"]]
            if set(tail50) - set(full_cached) or set(current_docs) - set(tail50):
                raise RuntimeError(f"pool nesting failure at qid={qid}")
            pools = {"current": current_docs, "tail50": tail50,
                     "full_cached": full_cached}
            gold = answers[qid]
            if not gold:
                continue
            for name, docs in pools.items():
                value = len(set(docs) & gold) / len(gold)
                ceilings[name] += value
                fold_sums[fold_for[qid]][name] += value
                size_values[name].append(len(docs))
            outside = gold - set(current_docs)
            for doc in sorted(outside):
                info = meta.get(doc, {})
                er, br = info.get("e5_rank"), info.get("bm25_rank")
                flags = {
                    "bm25_11_50": br is not None and 11 <= br <= 50,
                    "e5_51_100": er is not None and 51 <= er <= 100,
                    "e5_101_150": er is not None and 101 <= er <= 150,
                    "not_in_any_cached_source": er is None and br is None,
                }
                if flags["bm25_11_50"]:
                    exclusive = "bm25_11_50"
                elif flags["e5_51_100"]:
                    exclusive = "e5_51_100"
                elif flags["e5_101_150"]:
                    exclusive = "e5_101_150"
                elif flags["not_in_any_cached_source"]:
                    exclusive = "not_in_any_cached_source"
                else:
                    exclusive = "cached_other_or_truncated"
                context_path = args.contexts / f"context_{doc}.json"
                if context_path.exists():
                    ctx = json.loads(context_path.read_text(encoding="utf-8"))
                    passage = str(ctx.get("passage") or "")
                    n_chars, kind = len(passage), doc_type(passage)
                else:
                    n_chars, kind = None, "MISSING_CONTEXT"
                trace_rows.append({
                    "qid": qid, "doc_id": doc, "fold": fold_for[qid],
                    "gold_count": len(gold), "single_or_multi": "single" if len(gold) == 1 else "multi",
                    "e5_rank": er, "bm25_rank": br, "membership_flags": flags,
                    "exclusive_bucket": exclusive,
                    "in_tail50": doc in set(tail50), "in_full_cached": doc in set(full_cached),
                    "document_chars": n_chars,
                    "document_length_bin": length_bin(n_chars) if n_chars is not None else "missing",
                    "document_type": kind,
                })
            sink.write(json_line({"qid": qid, "fold": fold_for[qid],
                                  "query": str(row["query"]), "pools": pools,
                                  "source_meta": meta}) + "\n")
            query_count += 1
            fold_counts[fold_for[qid]] += 1
    if query_count != 6991:
        raise RuntimeError(f"expected 6991 evaluable rows, got {query_count}")

    def breakdown(key: str) -> dict:
        out = Counter(str(r[key]) for r in trace_rows)
        return dict(sorted(out.items()))

    def cross_breakdown(key: str) -> dict:
        grouped = defaultdict(Counter)
        for trace in trace_rows:
            grouped[trace["exclusive_bucket"]][str(trace[key])] += 1
        return {bucket: dict(sorted(values.items())) for bucket, values in sorted(grouped.items())}

    trace_path = args.output_dir / "V2_CURRENT_POOL_MISSED_GOLD_TRACE.jsonl"
    with trace_path.open("w", encoding="utf-8", newline="\n") as sink:
        for row in trace_rows:
            sink.write(json_line(row) + "\n")
    report = {
        "schema_version": "dsc2026.research_v2.candidate_tail_prepare.v1",
        "status": "PREPARED_NO_NEW_JINA_SCORES_YET", "queries": query_count,
        "pool_definitions": {
            "current": "E5@50 union novel BM25@10",
            "tail50": "E5@50 set-union BM25@50",
            "full_cached": "complete EXP-022 bounded row: E5 anchor + novel BM25@50 + available E5 tail backfill; maximum 150",
        },
        "candidate_counts": {name: {"min": min(v), "mean": float(np.mean(v)),
                                      "max": max(v), "total": sum(v)}
                             for name, v in size_values.items()},
        "candidate_ceiling": {name: ceilings[name] / query_count for name in size_values},
        "per_fold_ceiling": {fold: {name: sums[name] / fold_counts[fold]
                                     for name in size_values}
                             for fold, sums in fold_sums.items()},
        "current_out_of_pool_gold_occurrences": len(trace_rows),
        "missed_gold_breakdown": {
            "exclusive_bucket": breakdown("exclusive_bucket"),
            "single_or_multi": breakdown("single_or_multi"),
            "fold": breakdown("fold"),
            "document_length_bin": breakdown("document_length_bin"),
            "document_type": breakdown("document_type"),
            "exclusive_bucket_by_single_or_multi": cross_breakdown("single_or_multi"),
            "exclusive_bucket_by_fold": cross_breakdown("fold"),
            "exclusive_bucket_by_document_length_bin": cross_breakdown("document_length_bin"),
            "exclusive_bucket_by_document_type": cross_breakdown("document_type"),
            "overlapping_membership_flags": {key: sum(bool(r["membership_flags"][key]) for r in trace_rows)
                                               for key in ("bm25_11_50", "e5_51_100", "e5_101_150", "not_in_any_cached_source")},
        },
        "artifacts": {"pools": str(pool_path), "pools_sha256": sha256(pool_path),
                      "trace": str(trace_path), "trace_sha256": sha256(trace_path)},
        "inputs_sha256": {"folds": sha256(args.folds), "historical_candidates": sha256(args.historical_candidates),
                          "dense_candidates": sha256(args.dense_candidates),
                          "current_pool": sha256(args.current_pool), "train": sha256(args.train),
                          "exclusions": sha256(args.exclusions)},
    }
    output = args.output_dir / "V2_CANDIDATE_TAIL_PREPARE_REPORT.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def patch_transformers_v5() -> None:
    import transformers.models.xlm_roberta.modeling_xlm_roberta as module
    if hasattr(module, "create_position_ids_from_input_ids"):
        return

    def create_position_ids(input_ids, padding_idx, past_key_values_length=0):
        mask = input_ids.ne(padding_idx).int()
        positions = (torch.cumsum(mask, dim=1) + past_key_values_length) * mask
        return positions.long() + padding_idx
    module.create_position_ids_from_input_ids = create_position_ids


def load_model(path: Path, hf_modules_cache: Path):
    verify(path / "model.safetensors", "model_weights")
    hf_modules_cache.mkdir(parents=True, exist_ok=True)
    os.environ["HF_MODULES_CACHE"] = str(hf_modules_cache.resolve())
    patch_transformers_v5()
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True, fix_mistral_regex=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        path, trust_remote_code=True, dtype=torch.float16).eval().to("cuda")
    model._tokenizer = tok
    return model


def open_tail_db(path: Path) -> sqlite3.Connection:
    db = sqlite3.connect(path)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=NORMAL")
    db.execute("CREATE TABLE IF NOT EXISTS scores(qid TEXT,doc_id TEXT,score REAL,PRIMARY KEY(qid,doc_id))")
    db.execute("CREATE TABLE IF NOT EXISTS progress(qid TEXT,target_pool TEXT,seconds REAL,pairs INTEGER,parents INTEGER,peak_mib REAL,PRIMARY KEY(qid,target_pool))")
    db.execute("CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT)")
    db.commit()
    return db


def score(args: argparse.Namespace) -> None:
    verify(args.current_pool, "current_pool")
    model = load_model(args.model, args.hf_modules_cache)
    sys.path.insert(0, str(args.huy_root))
    from benchmark_jina_reranker_holdouts import top_passages
    from run_burst_expanded_fusion_submission import DocumentStore, prefetch

    docs = DocumentStore(sorted(args.contexts.glob("context_*.json")), cache_size=9000)
    db = open_tail_db(args.tail_score_db)
    pool_hash = sha256(args.tail_pools)
    stored = dict(db.execute("SELECT key,value FROM metadata"))
    locks = {"tail_pools_sha256": pool_hash, "model_weights_sha256": EXPECTED["model_weights"],
             "renderer": "Huy top_passages count=2; parent=max; max_length=512"}
    if stored and any(stored.get(k) != v for k, v in locks.items()):
        raise RuntimeError("tail score DB metadata mismatch")
    with db:
        db.executemany("INSERT OR REPLACE INTO metadata VALUES(?,?)", locks.items())
    complete = {str(q) for (q,) in db.execute(
        "SELECT qid FROM progress WHERE target_pool=?", (args.target_pool,))}
    # A score may already exist from the smaller nested pool.  Never recompute it.
    already = defaultdict(set)
    for q, d in db.execute("SELECT qid,doc_id FROM scores"):
        already[str(q)].add(str(d))

    def prepare(row: dict):
        qid, query = str(row["qid"]), str(row["query"])
        if qid in complete:
            return qid, query, [], [], []
        current = set(row["pools"]["current"])
        targets = [str(x) for x in row["pools"][args.target_pool]]
        new_docs = [d for d in targets if d not in current and d not in already[qid]]
        owners, passages = [], []
        for doc in new_docs:
            selected = top_passages(query, docs[doc], count=2)
            if not selected:
                raise RuntimeError(f"no lexical passage qid={qid} doc={doc}")
            for passage in selected:
                owners.append(doc); passages.append(passage)
        return qid, query, new_docs, owners, passages

    started = time.perf_counter()
    newly_scored = 0
    rows_done = 0
    torch.cuda.reset_peak_memory_stats()
    try:
        rows = read_jsonl(args.tail_pools)
        if args.limit is not None:
            rows = (row for index, row in enumerate(rows) if index < args.limit)
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            for qid, query, new_docs, owners, passages in prefetch(
                    executor, rows, prepare, ahead=args.workers):
                if qid in complete:
                    continue
                before = time.perf_counter()
                parent = {}
                if passages:
                    raw = model.compute_score([(query, p) for p in passages],
                                              batch_size=args.batch_size, max_length=512)
                    if isinstance(raw, float):
                        raw = [raw]
                    for doc, value in zip(owners, raw):
                        parent[doc] = max(parent.get(doc, -math.inf), float(value))
                if set(parent) != set(new_docs):
                    raise RuntimeError(f"parent score mismatch qid={qid}")
                elapsed = time.perf_counter() - before
                peak = torch.cuda.max_memory_allocated() / 2**20
                with db:
                    db.executemany("INSERT OR REPLACE INTO scores VALUES(?,?,?)",
                                   [(qid, doc, value) for doc, value in parent.items()])
                    db.execute("INSERT OR REPLACE INTO progress VALUES(?,?,?,?,?,?)",
                               (qid, args.target_pool, elapsed, len(passages), len(new_docs), peak))
                newly_scored += len(new_docs)
                rows_done += 1
                if rows_done % args.print_every == 0:
                    wall = time.perf_counter() - started
                    print(f"target={args.target_pool} rows={rows_done} parents={newly_scored} "
                          f"parents_per_sec={newly_scored/max(wall,1e-9):.2f} peak_mib={peak:.1f}", flush=True)
    finally:
        db.close()
    print(json.dumps({"status": "COMPLETE", "target_pool": args.target_pool,
                      "new_parents_scored_this_run": newly_scored,
                      "rows_completed_this_run": rows_done,
                      "wall_seconds": time.perf_counter() - started}, indent=2))


def metric(pred: list[str], gold: set[str]) -> tuple[float, float]:
    hits = len(set(pred[:5]) & gold)
    return hits / len(gold), hits / 5.0


def evaluate(args: argparse.Namespace) -> None:
    for path, key in ((args.folds, "folds"), (args.current_pool, "current_pool"),
                      (args.train, "train"), (args.exclusions, "exclusions")):
        verify(path, key)
    train = json.loads(args.train.read_text(encoding="utf-8"))
    answers = canonical_answers(train, json.loads(args.exclusions.read_text(encoding="utf-8")))
    folds = json.loads(args.folds.read_text(encoding="utf-8"))["folds"]
    fold_for = {str(q): f for f, ids in folds.items() for q in ids}
    base = sqlite3.connect(f"file:{args.base_score_db.resolve().as_posix()}?mode=ro", uri=True)
    base_scores = {(str(q), str(d)): float(s) for q, d, s in
                   base.execute("SELECT qid,doc_id,score FROM scores WHERE arm='lexical'")}
    base.close()
    tail = sqlite3.connect(f"file:{args.tail_score_db.resolve().as_posix()}?mode=ro", uri=True)
    tail_scores = {(str(q), str(d)): float(s) for q, d, s in tail.execute("SELECT qid,doc_id,score FROM scores")}
    runtime = [{"target_pool": str(pool), "seconds": float(sec or 0), "pairs": int(pairs or 0),
                "parents": int(parents or 0), "queries": int(n), "peak_mib": float(peak or 0)}
               for pool, sec, pairs, parents, n, peak in tail.execute(
                   "SELECT target_pool,SUM(seconds),SUM(pairs),SUM(parents),COUNT(*),MAX(peak_mib) FROM progress GROUP BY target_pool")]
    tail.close()
    all_scores = dict(base_scores)
    all_scores.update(tail_scores)

    names = tuple(args.evaluation_pools)
    if not names or names[0] != "current":
        raise RuntimeError("evaluation pools must start with current")
    vals = {n: [] for n in names}
    fold_vals = {n: defaultdict(list) for n in names}
    predictions = {n: {} for n in names}
    counts = defaultdict(list)
    recovered = {name: [] for name in names if name != "current"}
    pool_rows = list(read_jsonl(args.tail_pools))
    for row in pool_rows:
        qid, gold = str(row["qid"]), answers[str(row["qid"])]
        for name in names:
            docs = [str(x) for x in row["pools"][name]]
            missing = [d for d in docs if (qid, d) not in all_scores]
            if missing:
                raise RuntimeError(f"missing {len(missing)} scores for qid={qid} pool={name}")
            ordered = sorted(docs, key=lambda d: (-all_scores[(qid, d)], d))
            predictions[name][qid] = ordered[:5]
            values = metric(ordered, gold)
            vals[name].append(values)
            fold_vals[name][fold_for[qid]].append(values)
            counts[name].append(len(docs))
        current_set = set(row["pools"]["current"])
        fifth = all_scores[(qid, predictions["current"][qid][4])]
        for name in recovered:
            docs = [str(x) for x in row["pools"][name]]
            ordered = sorted(docs, key=lambda d: (-all_scores[(qid, d)], d))
            positions = {d: i + 1 for i, d in enumerate(ordered)}
            for doc in sorted((gold & set(docs)) - current_set):
                recovered[name].append({
                    "qid": qid, "doc_id": doc, "fold": fold_for[qid],
                    "gold_count": len(gold), "expanded_rank": positions[doc],
                    "score": all_scores[(qid, doc)], "incumbent_current_fifth_score": fifth,
                    "score_margin_vs_current_fifth": all_scores[(qid, doc)] - fifth,
                    "enters_top5": positions[doc] <= 5,
                })

    metrics = {}
    current_q = {q: metric(predictions["current"][q], answers[q])[0] for q in predictions["current"]}
    for name in names:
        qvals = {q: metric(predictions[name][q], answers[q])[0] for q in predictions[name]}
        delta = {q: qvals[q] - current_q[q] for q in qvals}
        oracle = np.mean([len(set(row["pools"][name]) & answers[str(row["qid"])]) /
                          len(answers[str(row["qid"])]) for row in pool_rows])
        metrics[name] = {
            "candidate_count": {"min": min(counts[name]), "mean": float(np.mean(counts[name])),
                                "max": max(counts[name]), "total": sum(counts[name])},
            "candidate_ceiling": float(oracle),
            "recall_at_5": float(np.mean([x[0] for x in vals[name]])),
            "precision_at_5": float(np.mean([x[1] for x in vals[name]])),
            "delta_recall_vs_current": float(np.mean(list(delta.values()))),
            "wins_losses_ties_vs_current": {"wins": sum(x > 0 for x in delta.values()),
                                             "losses": sum(x < 0 for x in delta.values()),
                                             "ties": sum(x == 0 for x in delta.values())},
            "changed_top5_sets_vs_current": sum(set(predictions[name][q]) != set(predictions["current"][q])
                                                 for q in predictions[name]),
            "per_fold_recall": {f: float(np.mean([x[0] for x in fold_vals[name][f]])) for f in folds},
            "single_gold_recall": float(np.mean([qvals[q] for q in qvals if len(answers[q]) == 1])),
            "multi_gold_recall": float(np.mean([qvals[q] for q in qvals if len(answers[q]) > 1])),
        }
    rankability = {}
    for name, rows in recovered.items():
        margins = [r["score_margin_vs_current_fifth"] for r in rows]
        ranks = [r["expanded_rank"] for r in rows]
        rankability[name] = {
            "newly_recovered_gold_occurrences": len(rows),
            "enters_top5": sum(r["enters_top5"] for r in rows),
            "rank_distribution": {"top5": sum(x <= 5 for x in ranks), "rank6_10": sum(6 <= x <= 10 for x in ranks),
                                  "rank11_20": sum(11 <= x <= 20 for x in ranks), "rank_gt20": sum(x > 20 for x in ranks)},
            "margin_vs_current_fifth": {"mean": float(np.mean(margins)) if margins else None,
                                        "median": float(np.median(margins)) if margins else None,
                                        "p90": float(np.quantile(margins, .9)) if margins else None,
                                        "nonnegative": sum(x >= 0 for x in margins),
                                        "within_minus_0_5": sum(x >= -0.5 for x in margins)},
            "rows": rows,
        }

    expanded = [metrics[name] for name in names if name != "current"]
    if not expanded:
        raise RuntimeError("at least one expanded pool is required")
    best = max(expanded, key=lambda x: x["delta_recall_vs_current"])
    headroom = max(x["candidate_ceiling"] for x in expanded) - metrics["current"]["candidate_ceiling"]
    if headroom >= .003 and best["delta_recall_vs_current"] >= .002:
        decision = "POOL_EXPANSION_CANDIDATE_FOR_NEXT_CAMPAIGN"
    else:
        decision = "REJECT_RETRIEVAL_TAIL_AS_NEXT_ACTION"
    report = {
        "schema_version": "dsc2026.research_v2.candidate_tail_rankability.v1",
        "status": "COMPLETE", "decision": decision,
        "interpretation_gate": "candidate headroom >=0.003 and deterministic Jina-v2 delta >=0.002; no quota tuning",
        "metrics": metrics, "new_gold_rankability": rankability, "runtime": runtime,
        "inference_cost": {"base_current_parent_rows": len(base_scores), "additional_unique_parent_rows": len(tail_scores),
                           "additional_fraction_vs_current": len(tail_scores) / len(base_scores)},
        "inputs_sha256": {"tail_pools": sha256(args.tail_pools), "base_score_db": sha256(args.base_score_db),
                          "tail_score_db": sha256(args.tail_score_db), "folds": sha256(args.folds)},
    }
    suffix = "" if names == ("current", "tail50", "full_cached") else "_" + "_".join(names[1:])
    output = args.output_dir / f"V2_CANDIDATE_TAIL_RANKABILITY{suffix}.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    pred_path = args.output_dir / "V2_CANDIDATE_TAIL_TOP5.jsonl"
    with pred_path.open("w", encoding="utf-8", newline="\n") as sink:
        for qid in sorted(predictions["current"], key=int):
            sink.write(json_line({"qid": qid, **{n: predictions[n][qid] for n in names}}) + "\n")
    print(json.dumps({"decision": decision, "metrics": metrics,
                      "rankability": {k: {x: y for x, y in v.items() if x != "rows"}
                                      for k, v in rankability.items()},
                      "runtime": runtime}, ensure_ascii=False, indent=2))


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=("prepare", "score", "evaluate"))
    p.add_argument("--output-dir", type=Path, default=Path("results/research_v2_parallel_local"))
    p.add_argument("--tail-pools", type=Path, default=Path("results/research_v2_parallel_local/V2_CANDIDATE_TAIL_POOLS.jsonl"))
    p.add_argument("--historical-candidates", type=Path, default=Path("../LegalIR/cache/exp022_e5_bm25_union/train_oof_candidates.jsonl"))
    p.add_argument("--dense-candidates", type=Path, default=Path("../LegalIR/cache/exp021_e5_dense_candidates/train_oof_candidates.jsonl"))
    p.add_argument("--current-pool", type=Path, default=Path("results/research_v2_forensic/V2_CANDIDATE_POOL.jsonl"))
    p.add_argument("--folds", type=Path, default=Path("results/research_v2_forensic/V2_FOLDS.json"))
    p.add_argument("--train", type=Path, default=Path("../LegalIR/public_test_dataset/train.json"))
    p.add_argument("--exclusions", type=Path, default=Path("../LegalIR/cache/final_preprocessed_v2/exclusions.json"))
    p.add_argument("--contexts", type=Path, default=Path("../LegalIR/cache/final_preprocessed_v2/contexts"))
    p.add_argument("--model", type=Path, default=Path("cache/research_v2_forensic/models/jina-reranker-v2-base-multilingual"))
    p.add_argument("--base-score-db", type=Path, default=Path("cache/research_v2_forensic/evidence_ab_scores.sqlite"))
    p.add_argument("--tail-score-db", type=Path, default=Path("cache/research_v2_parallel_local/candidate_tail_scores.sqlite"))
    p.add_argument("--hf-modules-cache", type=Path, default=Path("cache/research_v2_parallel_local/hf_modules"))
    p.add_argument("--target-pool", choices=("tail50", "full_cached"), default="tail50")
    p.add_argument("--evaluation-pools", nargs="+", choices=("current", "tail50", "full_cached"),
                   default=["current", "tail50", "full_cached"])
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--print-every", type=int, default=250)
    p.add_argument("--limit", type=int, default=None, help="Smoke-test first N pool rows only.")
    p.add_argument("--huy-root", type=Path, default=Path(__file__).resolve().parents[2])
    return p


if __name__ == "__main__":
    args = parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.tail_score_db.parent.mkdir(parents=True, exist_ok=True)
    if args.stage == "prepare":
        prepare(args)
    elif args.stage == "score":
        score(args)
    else:
        evaluate(args)
