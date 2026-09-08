"""Fixed-membership, boundary-focused within-parent BM25 evidence pilot.

Candidate selection is the immutable Round-0 LOBO order, ranks 5--20.  BM25
selects two windows from the complete parent document without labels.  The same
frozen Jina-FT checkpoint scores both Huy's lexical selector and BM25 selector,
so the experiment isolates evidence selection despite provenance drift in the
old document-level score cache.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from benchmark_burst_v4_full_sqlite import tokens
from benchmark_jina_reranker_holdouts import SPACE_RE, STOPWORDS, top_passages
from jina_ft_local import load_model_and_tokenizer, score_pairs
from run_burst_expanded_fusion_submission import DocumentStore
from tune_corpus_cap32_fusion import build_training_cap


CACHE_PATH = ROOT / "cache/sol_high_rl/bm25_evidence_boundary.pkl"
REPORT_PATH = ROOT / "results/sol_high_rl/bm25_evidence_boundary_report.json"
WINDOW = 220
OVERLAP = 70
K1 = 1.2
B = 0.75


def windows(text: str):
    words = SPACE_RE.findall(text or "")
    if len(words) <= WINDOW + 80:
        return [(0, len(words), " ".join(words))]
    out = []
    step = WINDOW - OVERLAP
    for start in range(0, len(words), step):
        end = min(start + WINDOW, len(words))
        out.append((start, end, " ".join(words[start:end])))
        if end == len(words):
            break
    return out


def bm25_passages(question: str, text: str, count: int = 2):
    chunks = windows(text)
    if len(chunks) == 1:
        return [chunks[0][2]]
    query = [t for t in tokens(question) if len(t) >= 2 and t not in STOPWORDS]
    query = list(dict.fromkeys(query))
    chunk_tokens = [tokens(part) for _, _, part in chunks]
    lengths = [len(x) for x in chunk_tokens]
    avgdl = sum(lengths) / len(lengths)
    df = Counter()
    for row in chunk_tokens:
        df.update(set(row))
    n = len(chunks)
    scored = []
    for (start, end, part), row, dl in zip(chunks, chunk_tokens, lengths):
        tf = Counter(row)
        score = 0.0
        for term in query:
            freq = tf.get(term, 0)
            if not freq:
                continue
            idf = math.log(1.0 + (n - df[term] + 0.5) / (df[term] + 0.5))
            denom = freq + K1 * (1.0 - B + B * dl / max(avgdl, 1.0))
            score += idf * freq * (K1 + 1.0) / denom
        scored.append((score, -start, start, end, part))
    scored.sort(reverse=True)
    header = " ".join(SPACE_RE.findall(text or "")[:70])
    result = []
    for _, _, start, _, part in scored[:count]:
        result.append(part if start < 70 else header + "\n[ĐOẠN PHÙ HỢP]\n" + part)
    return result


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def evaluate(saved, queries, blocks, all_ids, ranking):
    rows = []
    for qid in all_ids:
        gold = queries[qid][1]
        record = saved[qid]
        for rank, docid in enumerate(ranking[qid][4:20], 5):
            row = record[docid]
            rows.append(
                {
                    "qid": qid,
                    "docid": docid,
                    "rank": rank,
                    "gold": docid in gold,
                    "changed": row["lexical_hashes"] != row["bm25_hashes"],
                    "lexical": row["lexical_score"],
                    "bm25": row["bm25_score"],
                    "delta": row["bm25_score"] - row["lexical_score"],
                }
            )
    gold_rows = [x for x in rows if x["gold"]]
    nongold_rows = [x for x in rows if not x["gold"]]

    opportunities = []
    defenses = []
    for qid in all_ids:
        qrows = [x for x in rows if x["qid"] == qid]
        defender = next(x for x in qrows if x["rank"] == 5)
        challengers = [x for x in qrows if x["rank"] >= 6]
        gold_challengers = [x for x in challengers if x["gold"]]
        nongold_challengers = [x for x in challengers if not x["gold"]]
        if not defender["gold"] and gold_challengers:
            best_l = max(x["lexical"] for x in gold_challengers)
            best_b = max(x["bm25"] for x in gold_challengers)
            opportunities.append(
                {
                    "qid": qid,
                    "lexical_margin": best_l - defender["lexical"],
                    "bm25_margin": best_b - defender["bm25"],
                }
            )
        if defender["gold"] and nongold_challengers:
            best_l = max(x["lexical"] for x in nongold_challengers)
            best_b = max(x["bm25"] for x in nongold_challengers)
            defenses.append(
                {
                    "qid": qid,
                    "lexical_margin": defender["lexical"] - best_l,
                    "bm25_margin": defender["bm25"] - best_b,
                }
            )

    def margin_summary(items):
        return {
            "queries": len(items),
            "margin_improved": sum(x["bm25_margin"] > x["lexical_margin"] for x in items),
            "margin_worsened": sum(x["bm25_margin"] < x["lexical_margin"] for x in items),
            "crossed_negative_to_positive": sum(x["lexical_margin"] <= 0 < x["bm25_margin"] for x in items),
            "crossed_positive_to_negative": sum(x["lexical_margin"] > 0 >= x["bm25_margin"] for x in items),
            "mean_margin_delta": float(np.mean([x["bm25_margin"] - x["lexical_margin"] for x in items])) if items else None,
            "details": items,
        }

    by_block = {}
    for block, ids in blocks.items():
        subset = [x for x in rows if x["qid"] in set(ids)]
        by_block[block] = {
            "documents": len(subset),
            "selector_changed": sum(x["changed"] for x in subset),
            "gold_documents": sum(x["gold"] for x in subset),
            "gold_score_up": sum(x["gold"] and x["delta"] > 0 for x in subset),
            "gold_score_down": sum(x["gold"] and x["delta"] < 0 for x in subset),
        }
    return {
        "scope": "Round-0 LOBO ranks 5-20; candidate membership unchanged",
        "documents": len(rows),
        "selector_changed": sum(x["changed"] for x in rows),
        "selector_changed_fraction": sum(x["changed"] for x in rows) / len(rows),
        "gold_documents": len(gold_rows),
        "nongold_documents": len(nongold_rows),
        "gold_score_up": sum(x["delta"] > 0 for x in gold_rows),
        "gold_score_down": sum(x["delta"] < 0 for x in gold_rows),
        "gold_score_tie": sum(x["delta"] == 0 for x in gold_rows),
        "nongold_score_up": sum(x["delta"] > 0 for x in nongold_rows),
        "nongold_score_down": sum(x["delta"] < 0 for x in nongold_rows),
        "mean_gold_delta": float(np.mean([x["delta"] for x in gold_rows])) if gold_rows else None,
        "mean_nongold_delta": float(np.mean([x["delta"] for x in nongold_rows])),
        "rank6_20_gold_promotion_opportunities": margin_summary(opportunities),
        "rank5_gold_defenses": margin_summary(defenses),
        "per_block": by_block,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    started = time.perf_counter()
    queries, blocks, all_ids, _, _, _ = build_training_cap(
        ROOT, 32, "results/corpus_index/holdout_extended_scores_cap32.pkl", depth=20
    )
    ranking = json.loads((ROOT / "results/sol_high_rl/BASELINE_LOBO_PREDICTIONS.json").read_text(encoding="utf-8"))
    docs = DocumentStore(
        sorted((ROOT / "DSC2026-LegalIR-main/v4_run/public_test_dataset/selected-contexts").glob("context_*.json")),
        cache_size=128,
    )
    saved = pickle.loads(CACHE_PATH.read_bytes()) if CACHE_PATH.exists() else {}
    todo = [q for q in all_ids if q not in saved]
    if args.limit:
        todo = todo[: args.limit]
    print(f"resume={len(saved)}/600 todo={len(todo)}", flush=True)
    model = tokenizer = None
    if todo:
        model, tokenizer = load_model_and_tokenizer()
    for index, qid in enumerate(todo, 1):
        question = queries[qid][0]
        record = {}
        pairs = []
        owners = []
        for docid in ranking[qid][4:20]:
            text = docs[docid]
            lexical = top_passages(question, text, count=2)
            bm25 = bm25_passages(question, text, count=2)
            unique = []
            for selector, passages in (("lexical", lexical), ("bm25", bm25)):
                for passage in passages:
                    key = (selector, digest(passage))
                    unique.append((selector, passage, key[1]))
                    pairs.append((question, passage))
                    owners.append((docid, selector, key[1]))
            record[docid] = {
                "lexical_hashes": [digest(x) for x in lexical],
                "bm25_hashes": [digest(x) for x in bm25],
                "scores": {"lexical": [], "bm25": []},
            }
        raw = score_pairs(model, tokenizer, pairs, batch_size=args.batch_size)
        for (docid, selector, passage_hash), score in zip(owners, raw):
            record[docid]["scores"][selector].append(
                {"sha256": passage_hash, "score": float(score)}
            )
        for docid in record:
            for selector in ("lexical", "bm25"):
                record[docid][f"{selector}_score"] = max(
                    x["score"] for x in record[docid]["scores"][selector]
                )
        saved[qid] = record
        if index % 10 == 0 or index == len(todo):
            CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            CACHE_PATH.write_bytes(pickle.dumps(saved, protocol=5))
        if index % 25 == 0 or index == len(todo):
            elapsed = time.perf_counter() - started
            rate = elapsed / index
            print(f"{index}/{len(todo)} {rate:.2f}s/q eta={rate*(len(todo)-index)/60:.1f}m", flush=True)

    if all(q in saved for q in all_ids):
        report = evaluate(saved, queries, blocks, all_ids, ranking)
        report["runtime_seconds_this_invocation"] = time.perf_counter() - started
        report["cache"] = str(CACHE_PATH.relative_to(ROOT))
        REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({k: v for k, v in report.items() if k not in {"rank6_20_gold_promotion_opportunities", "rank5_gold_defenses"}}, ensure_ascii=False, indent=2))
        print(REPORT_PATH)


if __name__ == "__main__":
    main()
