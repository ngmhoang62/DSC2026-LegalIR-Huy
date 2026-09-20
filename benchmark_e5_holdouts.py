"""Benchmark multilingual-E5 as an independent semantic BURST reranker."""

from __future__ import annotations

import json
import pickle
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from benchmark_jina_reranker_holdouts import load_documents, top_passages
from tune_burst_kernel_posterior import load_queries
from tune_burst_multistage_posterior import weighted_rrf
from tune_burst_pairwise import fixed_metrics


def average_pool(last_hidden, attention_mask):
    masked = last_hidden.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return masked.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


@torch.inference_mode()
def encode(model, tokenizer, texts, batch_size=32, max_length=384):
    output = []
    for start in range(0, len(texts), batch_size):
        batch = tokenizer(texts[start:start+batch_size], max_length=max_length,
                          padding=True, truncation=True, return_tensors="pt")
        batch = {key: value.to("cuda", non_blocking=True)
                 for key, value in batch.items()}
        hidden = model(**batch).last_hidden_state
        vectors = F.normalize(average_pool(hidden, batch["attention_mask"]), p=2, dim=1)
        output.append(vectors.float().cpu().numpy())
    return np.vstack(output)


def main():
    root = Path(__file__).resolve().parent
    queries = load_queries(root)
    qids = list(queries)
    blocks = {
        "validation_a": qids[750:850],
        "fresh_1251_1350": qids[1250:1350],
        "fresh_1351_1450": qids[1350:1450],
    }
    jina_saved = pickle.loads(
        (root / "results/jina_reranker/holdout_scores_finetuned.pkl").read_bytes()
    )["scores"]
    base = {qid: list(jina_saved[qid]) for ids in blocks.values() for qid in ids}
    jina = {qid: sorted(base[qid], key=lambda d: (-jina_saved[qid][d], d))
            for qid in base}
    documents = load_documents(
        root / "DSC2026-LegalIR-main/v4_run/public_test_dataset"
    )

    output = root / "results/e5_dense"
    output.mkdir(parents=True, exist_ok=True)
    cache_path = output / "holdout_scores.pkl"
    scores = {}
    if cache_path.exists():
        scores = pickle.loads(cache_path.read_bytes()).get("scores", {})

    model_path = root / "models/multilingual-e5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path, dtype=torch.float16).eval().to("cuda")
    print(f"E5 on {torch.cuda.get_device_name(0)}", flush=True)
    started = time.perf_counter()
    all_ids = sum(blocks.values(), [])
    for qi, qid in enumerate(all_ids, 1):
        if qid in scores:
            continue
        question = queries[qid][0]
        passages, owners = [], []
        for doc in base[qid]:
            for passage in top_passages(question, documents[doc], count=2):
                passages.append("passage: " + passage)
                owners.append(doc)
        query_vector = encode(model, tokenizer, ["query: " + question], 1)[0]
        passage_vectors = encode(model, tokenizer, passages, 32)
        similarities = passage_vectors @ query_vector
        scores[qid] = {doc: -1.0 for doc in base[qid]}
        for doc, score in zip(owners, similarities):
            scores[qid][doc] = max(scores[qid][doc], float(score))
        if qi % 10 == 0:
            print(f"Dense scored {qi}/{len(all_ids)} "
                  f"({time.perf_counter()-started:.1f}s)", flush=True)
            cache_path.write_bytes(pickle.dumps({"scores": scores}, protocol=5))
    cache_path.write_bytes(pickle.dumps({"scores": scores}, protocol=5))

    e5 = {qid: sorted(base[qid], key=lambda d: (-scores[qid][d], d))
          for qid in base}
    tune_ids = blocks["validation_a"] + blocks["fresh_1251_1350"]
    configs = []
    # Three-view RRF: lexical/statistical base + cross encoder + dense encoder.
    for wb in np.arange(.50, .91, .05):
        for wj in np.arange(0.0, .31, .05):
            we = 1.0 - float(wb) - float(wj)
            if we < -.001:
                continue
            for k in (0, 2, 5, 10, 20, 40):
                rankings = weighted_rrf(
                    [{q: base[q] for q in tune_ids},
                     {q: jina[q] for q in tune_ids},
                     {q: e5[q] for q in tune_ids}],
                    (float(wb), float(wj), max(we, 0.0)), k,
                )
                metrics, _ = fixed_metrics(rankings,
                                           {q: queries[q] for q in tune_ids})
                configs.append((metrics["Recall@5"], metrics["Precision@5"],
                                metrics["nDCG@10"], float(wb), float(wj),
                                max(we, 0.0), k, metrics))
    configs.sort(reverse=True, key=lambda x: (x[0], x[1], x[2]))
    _, _, _, wb, wj, we, k, tune_metrics = configs[0]

    report = {"model": "intfloat/multilingual-e5-small", "candidate_depth": 20,
              "best_weights": {"base": wb, "jina": wj, "e5": we,
                               "rrf_k": k, "tune": tune_metrics}, "blocks": {}}
    for tag, ids in blocks.items():
        gold = {q: queries[q] for q in ids}
        base_m, base_p = fixed_metrics({q: base[q] for q in ids}, gold)
        e5_m, _ = fixed_metrics({q: e5[q] for q in ids}, gold)
        fused = weighted_rrf(
            [{q: base[q] for q in ids}, {q: jina[q] for q in ids},
             {q: e5[q] for q in ids}], (wb, wj, we), k)
        fused_m, fused_p = fixed_metrics(fused, gold)
        report["blocks"][tag] = {
            "Multistage": base_m, "E5_only": e5_m, "ThreeViewFusion": fused_m,
            "paired_vs_multistage": {
                "wins": sum(a > b for a, b in zip(fused_p, base_p)),
                "ties": sum(a == b for a, b in zip(fused_p, base_p)),
                "losses": sum(a < b for a, b in zip(fused_p, base_p)),
            },
        }
    path = root / "burst_e5_dense_validation.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    print(f"Saved {path}", flush=True)


if __name__ == "__main__":
    main()
