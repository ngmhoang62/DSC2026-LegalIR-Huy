"""Benchmark AITeamVN Vietnamese_Embedding on BURST LegalIR holdouts."""

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


@torch.inference_mode()
def encode_cls(model, tokenizer, texts, batch_size=16, max_length=512):
    vectors = []
    for start in range(0, len(texts), batch_size):
        batch = tokenizer(texts[start:start+batch_size], max_length=max_length,
                          padding=True, truncation=True, return_tensors="pt")
        batch = {key: value.to("cuda", non_blocking=True)
                 for key, value in batch.items()}
        cls = model(**batch).last_hidden_state[:, 0]
        vectors.append(F.normalize(cls.float(), p=2, dim=1).cpu().numpy())
    return np.vstack(vectors)


def main():
    root = Path(__file__).resolve().parent
    queries = load_queries(root)
    qids = list(queries)
    blocks = {
        "validation_a": qids[750:850],
        "fresh_1251_1350": qids[1250:1350],
        "fresh_1351_1450": qids[1350:1450],
    }
    jina_scores = pickle.loads(
        (root / "results/jina_reranker/holdout_scores_finetuned.pkl").read_bytes()
    )["scores"]
    e5_scores = pickle.loads(
        (root / "results/e5_dense/holdout_scores.pkl").read_bytes()
    )["scores"]
    base = {q: list(jina_scores[q]) for ids in blocks.values() for q in ids}
    jina = {q: sorted(base[q], key=lambda d: (-jina_scores[q][d], d)) for q in base}
    e5 = {q: sorted(base[q], key=lambda d: (-e5_scores[q][d], d)) for q in base}
    documents = load_documents(root / "DSC2026-LegalIR-main/v4_run/public_test_dataset")

    output = root / "results/aiteamvn_dense"
    output.mkdir(parents=True, exist_ok=True)
    cache_path = output / "holdout_scores_512.pkl"
    scores = {}
    if cache_path.exists():
        scores = pickle.loads(cache_path.read_bytes()).get("scores", {})

    model_path = root / "models/AITeamVN_Vietnamese_Embedding"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path, dtype=torch.float16).eval().to("cuda")
    print(f"AITeamVN Vietnamese_Embedding on {torch.cuda.get_device_name(0)}", flush=True)
    started = time.perf_counter()
    all_ids = sum(blocks.values(), [])
    for qi, q in enumerate(all_ids, 1):
        if q in scores:
            continue
        question = queries[q][0]
        passages, owners = [], []
        for doc in base[q]:
            for passage in top_passages(question, documents[doc], count=2):
                passages.append(passage)
                owners.append(doc)
        qvector = encode_cls(model, tokenizer, [question], 1)[0]
        pvectors = encode_cls(model, tokenizer, passages, 16)
        similarities = pvectors @ qvector
        scores[q] = {doc: -1.0 for doc in base[q]}
        for doc, score in zip(owners, similarities):
            scores[q][doc] = max(scores[q][doc], float(score))
        if qi % 10 == 0:
            cache_path.write_bytes(pickle.dumps({"scores": scores}, protocol=5))
            print(f"AITeamVN scored {qi}/{len(all_ids)} "
                  f"({time.perf_counter()-started:.1f}s) "
                  f"VRAM={torch.cuda.max_memory_allocated()/2**30:.2f}GB", flush=True)
    cache_path.write_bytes(pickle.dumps({"scores": scores}, protocol=5))

    vi = {q: sorted(base[q], key=lambda d: (-scores[q][d], d)) for q in base}
    tune_ids = blocks["validation_a"] + blocks["fresh_1251_1350"]
    tune_gold = {q: queries[q] for q in tune_ids}
    trials = []
    # Four complementary views. All weights sum exactly to one.
    values = [i / 20 for i in range(0, 9)]
    for wb in [i / 20 for i in range(8, 17)]:
        for wj in values:
            for we in values:
                wv = 1.0 - wb - wj - we
                if wv < -1e-8 or wv > .40 + 1e-8:
                    continue
                for k in (0, 2, 5, 10, 20, 40):
                    ranked = weighted_rrf(
                        [{q: base[q] for q in tune_ids},
                         {q: jina[q] for q in tune_ids},
                         {q: e5[q] for q in tune_ids},
                         {q: vi[q] for q in tune_ids}],
                        (wb, wj, we, max(0.0, wv)), k,
                    )
                    metrics, _ = fixed_metrics(ranked, tune_gold)
                    trials.append((metrics["Recall@5"], metrics["Precision@5"],
                                   metrics["nDCG@10"], wb, wj, we,
                                   max(0.0, wv), k, metrics))
    trials.sort(reverse=True, key=lambda x: (x[0], x[1], x[2]))
    _, _, _, wb, wj, we, wv, k, tune_metrics = trials[0]

    report = {
        "model": "AITeamVN/Vietnamese_Embedding", "parameters": "~568M",
        "embedding_dimensions": 1024, "max_length_tested": 512,
        "pooling": "CLS+L2", "candidate_depth": 20,
        "best_weights": {"base": wb, "jina": wj, "e5_small": we,
                         "aiteamvn": wv, "rrf_k": k, "tune": tune_metrics},
        "blocks": {},
    }
    for tag, ids in blocks.items():
        gold = {q: queries[q] for q in ids}
        bm, bp = fixed_metrics({q: base[q] for q in ids}, gold)
        vm, _ = fixed_metrics({q: vi[q] for q in ids}, gold)
        fused = weighted_rrf(
            [{q: base[q] for q in ids}, {q: jina[q] for q in ids},
             {q: e5[q] for q in ids}, {q: vi[q] for q in ids}],
            (wb, wj, we, wv), k,
        )
        fm, fp = fixed_metrics(fused, gold)
        report["blocks"][tag] = {
            "Multistage": bm, "AITeamVN_only": vm, "FourViewFusion": fm,
            "paired_vs_multistage": {
                "wins": sum(a > b for a, b in zip(fp, bp)),
                "ties": sum(a == b for a, b in zip(fp, bp)),
                "losses": sum(a < b for a, b in zip(fp, bp)),
            },
        }
    path = root / "burst_aiteamvn_dense_validation.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    print(f"Saved {path}", flush=True)


if __name__ == "__main__":
    main()
