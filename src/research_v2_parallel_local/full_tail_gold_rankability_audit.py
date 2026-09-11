"""Optimistic gold-only rankability audit for deep cached V2 tails.

This is explicitly label-diagnostic and not a deployable expanded-pool metric.
It is run only after broad BM25 tail50 append failed decisively, avoiding the
low-EV cost of scoring another ~503k lower-ranked distractors.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from candidate_tail_rankability import (  # noqa: E402
    canonical_answers, load_model, open_tail_db, read_jsonl, sha256, verify,
)


def run(args: argparse.Namespace) -> None:
    for path, key in ((args.current_pool, "current_pool"), (args.train, "train"),
                      (args.exclusions, "exclusions"), (args.folds, "folds")):
        verify(path, key)
    train = json.loads(args.train.read_text(encoding="utf-8"))
    golds = canonical_answers(train, json.loads(args.exclusions.read_text(encoding="utf-8")))
    current_rows = {str(r["qid"]): r for r in read_jsonl(args.current_pool)}
    pool_rows = {str(r["qid"]): r for r in read_jsonl(args.tail_pools)}

    db = open_tail_db(args.tail_score_db)
    existing = {(str(q), str(d)) for q, d in db.execute("SELECT qid,doc_id FROM scores")}
    targets = []
    for qid, row in pool_rows.items():
        current = set(current_rows[qid]["doc_ids"])
        full = set(row["pools"]["full_cached"])
        for doc in sorted((golds[qid] & full) - current):
            if (qid, doc) not in existing:
                targets.append((qid, doc, str(row["query"])))
    db.close()

    if targets:
        model = load_model(args.model, args.hf_modules_cache)
        sys.path.insert(0, str(args.huy_root))
        from benchmark_jina_reranker_holdouts import top_passages
        from run_burst_expanded_fusion_submission import DocumentStore
        docs = DocumentStore(sorted(args.contexts.glob("context_*.json")), cache_size=9000)

        def prepare(item):
            qid, doc, query = item
            return qid, doc, query, top_passages(query, docs[doc], count=2)

        db = open_tail_db(args.tail_score_db)
        started = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        with ThreadPoolExecutor(max_workers=2) as executor:
            prepared = list(executor.map(prepare, targets))
        pairs, owners = [], []
        for qid, doc, query, passages in prepared:
            for passage in passages:
                pairs.append((query, passage)); owners.append((qid, doc))
        raw = model.compute_score(pairs, batch_size=args.batch_size, max_length=512)
        if isinstance(raw, float):
            raw = [raw]
        parent = {}
        for owner, value in zip(owners, raw):
            parent[owner] = max(parent.get(owner, -1e30), float(value))
        if set(parent) != {(q, d) for q, d, _ in targets}:
            raise RuntimeError("gold-only parent aggregation mismatch")
        elapsed = time.perf_counter() - started
        peak = torch.cuda.max_memory_allocated() / 2**20
        with db:
            db.executemany("INSERT OR REPLACE INTO scores VALUES(?,?,?)",
                           [(q, d, s) for (q, d), s in parent.items()])
            for qid in sorted({q for q, _, _ in targets}, key=int):
                n = sum(q == qid for q, _, _ in targets)
                db.execute("INSERT OR REPLACE INTO progress VALUES(?,?,?,?,?,?)",
                           (qid, "full_cached_gold_only", elapsed, 2 * n, n, peak))
        db.close()

    base = sqlite3.connect(f"file:{args.base_score_db.resolve().as_posix()}?mode=ro", uri=True)
    scores = {(str(q), str(d)): float(s) for q, d, s in
              base.execute("SELECT qid,doc_id,score FROM scores WHERE arm='lexical'")}
    base.close()
    tail = sqlite3.connect(f"file:{args.tail_score_db.resolve().as_posix()}?mode=ro", uri=True)
    scores.update({(str(q), str(d)): float(s) for q, d, s in tail.execute("SELECT qid,doc_id,score FROM scores")})
    progress = tail.execute(
        "SELECT SUM(parents),SUM(pairs),MAX(peak_mib) FROM progress WHERE target_pool='full_cached_gold_only'").fetchone()
    tail.close()
    trace = {(str(r["qid"]), str(r["doc_id"])): r for r in read_jsonl(args.trace)}
    rows = []
    for qid, row in pool_rows.items():
        current = [str(x) for x in current_rows[qid]["doc_ids"]]
        added_gold = sorted((golds[qid] & set(row["pools"]["full_cached"])) - set(current))
        if not added_gold:
            continue
        incumbent = sorted(current, key=lambda d: (-scores[(qid, d)], d))
        fifth_score = scores[(qid, incumbent[4])]
        optimistic = sorted(current + added_gold, key=lambda d: (-scores[(qid, d)], d))
        positions = {d: i + 1 for i, d in enumerate(optimistic)}
        for doc in added_gold:
            meta = trace[(qid, doc)]
            rows.append({
                "qid": qid, "doc_id": doc, "fold": meta["fold"],
                "single_or_multi": meta["single_or_multi"],
                "exclusive_bucket": meta["exclusive_bucket"],
                "score": scores[(qid, doc)], "current_fifth_score": fifth_score,
                "score_margin_vs_current_fifth": scores[(qid, doc)] - fifth_score,
                "optimistic_rank_ignoring_added_distractors": positions[doc],
                "optimistic_enters_top5": positions[doc] <= 5,
            })
    margins = [r["score_margin_vs_current_fifth"] for r in rows]
    report = {
        "schema_version": "dsc2026.research_v2.full_tail_gold_rankability.v1",
        "status": "COMPLETE_DIAGNOSTIC_ONLY",
        "warning": "Gold labels selected the scored deep-tail rows. This is an optimistic rankability audit, not a deployable expanded-pool Recall metric; unscored tail distractors can only worsen ranks.",
        "gold_occurrences_recovered_by_full_cached_over_current": len(rows),
        "already_scored_by_tail50": len(rows) - len(targets),
        "new_deep_tail_gold_rows_scored": len(targets),
        "optimistic_enters_top5": sum(r["optimistic_enters_top5"] for r in rows),
        "margin_vs_current_fifth": {"mean": float(np.mean(margins)), "median": float(np.median(margins)),
                                    "nonnegative": sum(x >= 0 for x in margins),
                                    "p10": float(np.quantile(margins, .1)), "p90": float(np.quantile(margins, .9))},
        "by_exclusive_bucket": dict(Counter(r["exclusive_bucket"] for r in rows)),
        "by_single_or_multi": dict(Counter(r["single_or_multi"] for r in rows)),
        "by_fold": dict(Counter(r["fold"] for r in rows)),
        "runtime": {"new_parents": int(progress[0] or 0), "pairs": int(progress[1] or 0),
                    "peak_mib": float(progress[2] or 0)},
        "rows": rows,
        "inputs_sha256": {"tail_pools": sha256(args.tail_pools), "trace": sha256(args.trace),
                          "base_score_db": sha256(args.base_score_db), "tail_score_db": sha256(args.tail_score_db)},
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "FULL_CACHED_TAIL_GOLD_ONLY_RANKABILITY.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=Path, default=Path("results/research_v2_parallel_local"))
    p.add_argument("--tail-pools", type=Path, default=Path("results/research_v2_parallel_local/V2_CANDIDATE_TAIL_POOLS.jsonl"))
    p.add_argument("--trace", type=Path, default=Path("results/research_v2_parallel_local/V2_CURRENT_POOL_MISSED_GOLD_TRACE.jsonl"))
    p.add_argument("--current-pool", type=Path, default=Path("results/research_v2_forensic/V2_CANDIDATE_POOL.jsonl"))
    p.add_argument("--folds", type=Path, default=Path("results/research_v2_forensic/V2_FOLDS.json"))
    p.add_argument("--train", type=Path, default=Path("../LegalIR/public_test_dataset/train.json"))
    p.add_argument("--exclusions", type=Path, default=Path("../LegalIR/cache/final_preprocessed_v2/exclusions.json"))
    p.add_argument("--contexts", type=Path, default=Path("../LegalIR/cache/final_preprocessed_v2/contexts"))
    p.add_argument("--model", type=Path, default=Path("cache/research_v2_forensic/models/jina-reranker-v2-base-multilingual"))
    p.add_argument("--hf-modules-cache", type=Path, default=Path("cache/research_v2_parallel_local/hf_modules"))
    p.add_argument("--base-score-db", type=Path, default=Path("cache/research_v2_forensic/evidence_ab_scores.sqlite"))
    p.add_argument("--tail-score-db", type=Path, default=Path("cache/research_v2_parallel_local/candidate_tail_scores.sqlite"))
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--huy-root", type=Path, default=Path(__file__).resolve().parents[2])
    run(p.parse_args())
