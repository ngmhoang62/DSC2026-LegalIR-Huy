"""Within-parent dense evidence retrieval on fixed Round-0 boundary candidates."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
from numpy.lib.format import open_memmap

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from benchmark_jina_reranker_holdouts import SPACE_RE
from full_corpus_title_retrieval import encode, load_model
from jina_ft_local import load_model_and_tokenizer, score_pairs
from run_burst_expanded_fusion_submission import DocumentStore
from tune_corpus_cap32_fusion import build_training_cap


CACHE = ROOT / "cache/sol_high_rl/dense_evidence_boundary"
RESULT = ROOT / "results/sol_high_rl/dense_evidence_boundary_report.json"
RANKING_PATH = ROOT / "results/sol_high_rl/BASELINE_LOBO_PREDICTIONS.json"
BM25_CACHE = ROOT / "cache/sol_high_rl/bm25_evidence_boundary.pkl"
META_PATH = CACHE / "window_meta.json"
VECTOR_PATH = CACHE / "window_vectors.f16.npy"
QUERY_PATH = CACHE / "query_vectors.f32.npy"
SELECTED_PATH = CACHE / "selected_passages.pkl"
SCORES_PATH = CACHE / "jina_scores.pkl"
SUCCESS_PATH = CACHE / "ENCODING_SUCCESS.json"
WINDOW = 220
STEP = 150
SCAN_LIMIT = 300_000


def doc_windows(text: str):
    words = SPACE_RE.findall((text or "")[: SCAN_LIMIT * 12])
    del words[SCAN_LIMIT:]
    if not words:
        return [""], [0]
    starts = list(range(0, max(len(words) - 70, 1), STEP))
    return [" ".join(words[s : s + WINDOW]) for s in starts], starts


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def prepare_meta(docs, unique_docs):
    if META_PATH.exists():
        return json.loads(META_PATH.read_text(encoding="utf-8"))
    counts = []
    max_chunks = 0
    for i, docid in enumerate(unique_docs, 1):
        passages, _ = doc_windows(docs[docid])
        counts.append(len(passages))
        max_chunks = max(max_chunks, len(passages))
        if i % 500 == 0:
            print(f"counted {i}/{len(unique_docs)} docs", flush=True)
    meta = {
        "documents": unique_docs,
        "counts": counts,
        "total_windows": int(sum(counts)),
        "max_windows_one_document": max_chunks,
        "window_words": WINDOW,
        "step_words": STEP,
        "scan_limit_words": SCAN_LIMIT,
        "embedding_dim": 1024,
        "dtype": "float16",
    }
    META_PATH.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def encode_windows(docs, meta, queries, all_ids, batch_size):
    expected_bytes = meta["total_windows"] * meta["embedding_dim"] * 2
    if SUCCESS_PATH.exists() and VECTOR_PATH.exists() and QUERY_PATH.exists():
        print("dense encoding cache complete", flush=True)
        return
    model, tokenizer = load_model()
    print(f"AITeamVN-FT ready; encoding {meta['total_windows']} windows", flush=True)
    vectors = open_memmap(
        VECTOR_PATH,
        mode="w+",
        dtype=np.float16,
        shape=(meta["total_windows"], meta["embedding_dim"]),
    )
    cursor = 0
    started = time.perf_counter()
    for i, docid in enumerate(meta["documents"], 1):
        passages, _ = doc_windows(docs[docid])
        batch_vectors = encode(model, tokenizer, passages, batch_size=batch_size, max_length=512)
        vectors[cursor : cursor + len(passages)] = batch_vectors.astype(np.float16)
        cursor += len(passages)
        if i % 100 == 0 or i == len(meta["documents"]):
            vectors.flush()
            rate = (time.perf_counter() - started) / i
            print(
                f"encoded docs {i}/{len(meta['documents'])}, windows {cursor}/{meta['total_windows']}, "
                f"eta={rate*(len(meta['documents'])-i)/60:.1f}m",
                flush=True,
            )
    query_vectors = encode(
        model, tokenizer, [queries[q][0] for q in all_ids], batch_size=64, max_length=512
    )
    np.save(QUERY_PATH, query_vectors.astype(np.float32))
    SUCCESS_PATH.write_text(
        json.dumps(
            {
                "documents": len(meta["documents"]),
                "windows": cursor,
                "expected_vector_bytes": expected_bytes,
                "runtime_seconds": time.perf_counter() - started,
                "max_gpu_memory_mib": torch.cuda.max_memory_allocated() / 2**20,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    del vectors, model
    gc.collect()
    torch.cuda.empty_cache()


def select_passages(docs, meta, queries, all_ids, ranking):
    if SELECTED_PATH.exists():
        return pickle.loads(SELECTED_PATH.read_bytes())
    vectors = np.load(VECTOR_PATH, mmap_mode="r")
    qvectors = np.load(QUERY_PATH, mmap_mode="r")
    offsets = {}
    cursor = 0
    for docid, count in zip(meta["documents"], meta["counts"]):
        offsets[docid] = (cursor, cursor + count)
        cursor += count
    selected = {}
    for qi, qid in enumerate(all_ids):
        row = {}
        for docid in ranking[qid][4:20]:
            begin, end = offsets[docid]
            sims = np.asarray(vectors[begin:end], dtype=np.float32) @ qvectors[qi]
            k = min(2, len(sims))
            picked = np.argpartition(-sims, k - 1)[:k]
            picked = sorted(picked.tolist(), key=lambda j: (-float(sims[j]), j))
            passages, starts = doc_windows(docs[docid])
            header = " ".join(SPACE_RE.findall(docs[docid])[:70])
            chosen = []
            for j in picked:
                part = passages[j]
                text = part if starts[j] < 70 else header + "\n[ĐOẠN PHÙ HỢP]\n" + part
                chosen.append({"text": text, "sha256": digest(text), "cosine": float(sims[j])})
            row[docid] = chosen
        selected[qid] = row
        if (qi + 1) % 100 == 0:
            print(f"selected dense passages {qi+1}/{len(all_ids)}", flush=True)
    SELECTED_PATH.write_bytes(pickle.dumps(selected, protocol=5))
    return selected


def score_dense(selected, queries, all_ids, batch_size):
    saved = pickle.loads(SCORES_PATH.read_bytes()) if SCORES_PATH.exists() else {}
    todo = [q for q in all_ids if q not in saved]
    print(f"Jina dense-evidence scoring resume={len(saved)}/{len(all_ids)}", flush=True)
    if not todo:
        return saved
    model, tokenizer = load_model_and_tokenizer()
    started = time.perf_counter()
    for i, qid in enumerate(todo, 1):
        pairs, owners = [], []
        for docid, passages in selected[qid].items():
            for passage in passages:
                pairs.append((queries[qid][0], passage["text"]))
                owners.append(docid)
        raw = score_pairs(model, tokenizer, pairs, batch_size=batch_size)
        row = {}
        for docid, score in zip(owners, raw):
            row[docid] = max(row.get(docid, float("-inf")), float(score))
        saved[qid] = row
        if i % 10 == 0 or i == len(todo):
            SCORES_PATH.write_bytes(pickle.dumps(saved, protocol=5))
        if i % 50 == 0 or i == len(todo):
            rate = (time.perf_counter() - started) / i
            print(f"Jina {i}/{len(todo)} {rate:.2f}s/q eta={rate*(len(todo)-i)/60:.1f}m", flush=True)
    return saved


def variant_report(name, values, bm25_saved, queries, blocks, all_ids, ranking, selected):
    rows = []
    for qid in all_ids:
        gold = queries[qid][1]
        for rank, docid in enumerate(ranking[qid][4:20], 5):
            base = bm25_saved[qid][docid]
            lexical = float(base["lexical_score"])
            candidate = float(values[qid][docid])
            rows.append(
                {
                    "qid": qid,
                    "docid": docid,
                    "rank": rank,
                    "gold": docid in gold,
                    "lexical": lexical,
                    "candidate": candidate,
                    "delta": candidate - lexical,
                    "changed": base["lexical_hashes"] != [x["sha256"] for x in selected[qid][docid]],
                }
            )

    def summarize_margin(items):
        return {
            "queries": len(items),
            "margin_improved": sum(x["candidate_margin"] > x["lexical_margin"] for x in items),
            "margin_worsened": sum(x["candidate_margin"] < x["lexical_margin"] for x in items),
            "crossed_negative_to_positive": sum(x["lexical_margin"] <= 0 < x["candidate_margin"] for x in items),
            "crossed_positive_to_negative": sum(x["lexical_margin"] > 0 >= x["candidate_margin"] for x in items),
            "mean_margin_delta": float(np.mean([x["candidate_margin"] - x["lexical_margin"] for x in items])) if items else None,
            "details": items,
        }

    opportunities, defenses = [], []
    for qid in all_ids:
        qrows = [x for x in rows if x["qid"] == qid]
        defender = next(x for x in qrows if x["rank"] == 5)
        challengers = [x for x in qrows if x["rank"] >= 6]
        gold_ch = [x for x in challengers if x["gold"]]
        nongold_ch = [x for x in challengers if not x["gold"]]
        if not defender["gold"] and gold_ch:
            opportunities.append(
                {
                    "qid": qid,
                    "lexical_margin": max(x["lexical"] for x in gold_ch) - defender["lexical"],
                    "candidate_margin": max(x["candidate"] for x in gold_ch) - defender["candidate"],
                }
            )
        if defender["gold"] and nongold_ch:
            defenses.append(
                {
                    "qid": qid,
                    "lexical_margin": defender["lexical"] - max(x["lexical"] for x in nongold_ch),
                    "candidate_margin": defender["candidate"] - max(x["candidate"] for x in nongold_ch),
                }
            )
    gold_rows = [x for x in rows if x["gold"]]
    by_block = {}
    for block, ids in blocks.items():
        idset = set(ids)
        subset = [x for x in rows if x["qid"] in idset]
        g = [x for x in subset if x["gold"]]
        by_block[block] = {
            "gold_documents": len(g),
            "gold_score_up": sum(x["delta"] > 0 for x in g),
            "gold_score_down": sum(x["delta"] < 0 for x in g),
            "selector_changed": sum(x["changed"] for x in subset),
        }
    return {
        "name": name,
        "documents": len(rows),
        "selector_changed": sum(x["changed"] for x in rows),
        "gold_documents": len(gold_rows),
        "gold_score_up": sum(x["delta"] > 0 for x in gold_rows),
        "gold_score_down": sum(x["delta"] < 0 for x in gold_rows),
        "mean_gold_delta": float(np.mean([x["delta"] for x in gold_rows])),
        "mean_nongold_delta": float(np.mean([x["delta"] for x in rows if not x["gold"]])),
        "rank6_20_gold_promotion_opportunities": summarize_margin(opportunities),
        "rank5_gold_defenses": summarize_margin(defenses),
        "per_block": by_block,
    }


def main():
    global CACHE, RESULT, META_PATH, VECTOR_PATH, QUERY_PATH, SELECTED_PATH, SCORES_PATH, SUCCESS_PATH
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare-only", action="store_true")
    ap.add_argument("--per-block-limit", type=int, default=0)
    ap.add_argument("--benchmark-windows", type=int, default=0)
    ap.add_argument("--encoder-batch-size", type=int, default=24)
    ap.add_argument("--jina-batch-size", type=int, default=8)
    args = ap.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    queries, blocks, all_ids, _, _, _ = build_training_cap(
        ROOT, 32, "results/corpus_index/holdout_extended_scores_cap32.pkl", depth=20
    )
    if args.per_block_limit:
        blocks = {name: ids[: args.per_block_limit] for name, ids in blocks.items()}
        all_ids = [qid for name in blocks for qid in blocks[name]]
        CACHE = ROOT / f"cache/sol_high_rl/dense_evidence_boundary_pb{args.per_block_limit}"
        RESULT = ROOT / f"results/sol_high_rl/dense_evidence_boundary_pb{args.per_block_limit}_report.json"
        META_PATH = CACHE / "window_meta.json"
        VECTOR_PATH = CACHE / "window_vectors.f16.npy"
        QUERY_PATH = CACHE / "query_vectors.f32.npy"
        SELECTED_PATH = CACHE / "selected_passages.pkl"
        SCORES_PATH = CACHE / "jina_scores.pkl"
        SUCCESS_PATH = CACHE / "ENCODING_SUCCESS.json"
        CACHE.mkdir(parents=True, exist_ok=True)
    ranking = json.loads(RANKING_PATH.read_text(encoding="utf-8"))
    unique_docs = sorted({d for q in all_ids for d in ranking[q][4:20]})
    docs = DocumentStore(
        sorted((ROOT / "DSC2026-LegalIR-main/v4_run/public_test_dataset/selected-contexts").glob("context_*.json")),
        cache_size=64,
    )
    meta = prepare_meta(docs, unique_docs)
    print(json.dumps({"queries": len(all_ids), "unique_docs": len(unique_docs), **{k: meta[k] for k in ("total_windows", "max_windows_one_document")}, "estimated_vector_mib": meta["total_windows"]*1024*2/2**20}, indent=2), flush=True)
    if args.prepare_only:
        return
    if args.benchmark_windows:
        sample = []
        for docid in meta["documents"]:
            passages, _ = doc_windows(docs[docid])
            sample.extend(passages)
            if len(sample) >= args.benchmark_windows:
                break
        sample = sample[: args.benchmark_windows]
        model, tokenizer = load_model()
        mark = time.perf_counter()
        encode(model, tokenizer, sample, batch_size=args.encoder_batch_size, max_length=512)
        seconds = time.perf_counter() - mark
        print(json.dumps({"benchmark_windows": len(sample), "seconds": seconds, "windows_per_second": len(sample)/seconds, "projected_full_minutes": meta["total_windows"]/max(len(sample)/seconds, 1e-9)/60}, indent=2), flush=True)
        return
    encode_windows(docs, meta, queries, all_ids, args.encoder_batch_size)
    selected = select_passages(docs, meta, queries, all_ids, ranking)
    dense_scores = score_dense(selected, queries, all_ids, args.jina_batch_size)
    bm25_saved = pickle.loads(BM25_CACHE.read_bytes())
    union_scores = {
        q: {d: max(float(dense_scores[q][d]), float(bm25_saved[q][d]["bm25_score"])) for d in dense_scores[q]}
        for q in all_ids
    }
    report = {
        "scope": "Round-0 LOBO ranks 5-20; fixed candidate membership",
        "model_contract": {"selector_encoder": "AITeamVN fine-tuned CLS+L2", "prefixes": "none", "max_length": 512, "window_words": WINDOW, "step_words": STEP, "cross_encoder": "local Jina-FT strict state load"},
        "meta": meta,
        "dense": variant_report("dense", dense_scores, bm25_saved, queries, blocks, all_ids, ranking, selected),
        "dense_bm25_union": variant_report("dense_bm25_union", union_scores, bm25_saved, queries, blocks, all_ids, ranking, selected),
        "runtime_seconds_this_invocation": time.perf_counter() - started,
    }
    RESULT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    compact = {}
    for name in ("dense", "dense_bm25_union"):
        x = report[name]
        compact[name] = {k: v for k, v in x.items() if k not in {"rank6_20_gold_promotion_opportunities", "rank5_gold_defenses"}}
        compact[name]["promotion"] = {k: v for k, v in x["rank6_20_gold_promotion_opportunities"].items() if k != "details"}
        compact[name]["defense"] = {k: v for k, v in x["rank5_gold_defenses"].items() if k != "details"}
    print(json.dumps(compact, ensure_ascii=False, indent=2), flush=True)
    print(RESULT)


if __name__ == "__main__":
    main()
