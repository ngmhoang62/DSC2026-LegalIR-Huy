"""Expand BURST candidates with Vietnamese dense scoring over raw union top-50."""

from __future__ import annotations

import json
import pickle
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from benchmark_aiteamvn_holdouts import encode_cls
from benchmark_jina_reranker_holdouts import load_cache, load_documents, top_passages
from tune_burst_kernel_posterior import load_queries
from tune_burst_multistage_posterior import weighted_rrf
from tune_burst_pairwise import fixed_metrics


DEPTH = 50
RRF_K_RAW = 20


def raw_union(cache_row, depth=DEPTH):
    """Union candidates plus an RRF lexical rank across raw retrieval branches."""
    scores = {}
    for branch in cache_row:
        for rank, item in enumerate(branch[:depth], 1):
            doc = str(item[0])
            scores[doc] = scores.get(doc, 0.0) + 1.0 / (RRF_K_RAW + rank)
    return sorted(scores, key=lambda d: (-scores[d], d))


def main():
    root = Path(__file__).resolve().parent
    queries = load_queries(root)
    qids = list(queries)
    blocks = {
        "validation_a": qids[750:850],
        "fresh_1251_1350": qids[1250:1350],
        "fresh_1351_1450": qids[1350:1450],
    }
    raw = {}
    for tag, ids in blocks.items():
        cache = load_cache(root, tag)
        raw.update({q: raw_union(cache[q]) for q in ids})
    documents = load_documents(root / "DSC2026-LegalIR-main/v4_run/public_test_dataset")

    output = root / "results/dense_expansion"
    output.mkdir(parents=True, exist_ok=True)
    cache_path = output / "union50_scores.pkl"
    scores = {}
    if cache_path.exists():
        saved = pickle.loads(cache_path.read_bytes())
        if saved.get("depth") == DEPTH:
            scores = saved.get("scores", {})

    model_path = root / "models/AITeamVN_Vietnamese_Embedding"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path, dtype=torch.float16).eval().to("cuda")
    print(f"Dense expansion on {torch.cuda.get_device_name(0)}", flush=True)
    all_ids = sum(blocks.values(), [])
    started = time.perf_counter()
    for qi, q in enumerate(all_ids, 1):
        if q in scores:
            continue
        question = queries[q][0]
        docs = raw[q]
        passages = [top_passages(question, documents[doc], count=1)[0] for doc in docs]
        qvec = encode_cls(model, tokenizer, [question], 1, 512)[0]
        dvec = encode_cls(model, tokenizer, passages, 32, 512)
        similarities = dvec @ qvec
        scores[q] = {doc: float(score) for doc, score in zip(docs, similarities)}
        if qi % 5 == 0:
            cache_path.write_bytes(pickle.dumps(
                {"depth": DEPTH, "scores": scores}, protocol=5
            ))
            print(f"Dense-expanded {qi}/{len(all_ids)} "
                  f"({time.perf_counter()-started:.1f}s)", flush=True)
    cache_path.write_bytes(pickle.dumps({"depth": DEPTH, "scores": scores}, protocol=5))

    dense = {q: sorted(raw[q], key=lambda d: (-scores[q][d], d)) for q in all_ids}
    tune_ids = blocks["validation_a"] + blocks["fresh_1251_1350"]
    gold = {q: queries[q] for q in tune_ids}
    trials = []
    for wr in np.arange(.15, .91, .05):
        wd = 1.0 - float(wr)
        for k in (0, 2, 5, 10, 20, 40, 80):
            ranked = weighted_rrf(
                [{q: raw[q] for q in tune_ids}, {q: dense[q] for q in tune_ids}],
                (float(wr), wd), k,
            )
            m, _ = fixed_metrics(ranked, gold)
            trials.append((m["Recall@5"], m["Precision@5"], m["nDCG@10"],
                           float(wr), wd, k, m))
    trials.sort(reverse=True, key=lambda x: (x[0], x[1], x[2]))
    _, _, _, wr, wd, k, tune_metrics = trials[0]
    report = {"model": "AITeamVN/Vietnamese_Embedding", "raw_union_depth": DEPTH,
              "passages_per_doc": 1, "max_length": 512,
              "best_weights": {"raw_rrf": wr, "dense": wd, "rrf_k": k,
                               "tune": tune_metrics}, "blocks": {}}
    for tag, ids in blocks.items():
        block_gold = {q: queries[q] for q in ids}
        raw_m, _ = fixed_metrics({q: raw[q] for q in ids}, block_gold)
        dense_m, _ = fixed_metrics({q: dense[q] for q in ids}, block_gold)
        fused = weighted_rrf(
            [{q: raw[q] for q in ids}, {q: dense[q] for q in ids}], (wr, wd), k
        )
        fused_m, _ = fixed_metrics(fused, block_gold)
        report["blocks"][tag] = {"raw_rrf": raw_m, "dense_only": dense_m,
                                  "expanded_fusion": fused_m,
                                  "Recall@20_ceiling": sum(
                                      len(set(fused[q][:20]) & queries[q][1]) /
                                      len(queries[q][1]) for q in ids) / len(ids)}
    path = root / "burst_dense_expansion_validation.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    print(f"Saved {path}", flush=True)


if __name__ == "__main__":
    main()
