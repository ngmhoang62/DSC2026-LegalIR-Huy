"""Rerank the union of BURST top-20 and dense-expanded top-20 candidates."""

from __future__ import annotations

import json
import pickle
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

from benchmark_aiteamvn_holdouts import encode_cls
from benchmark_dense_expansion_holdouts import raw_union
from benchmark_jina_reranker_holdouts import load_cache, load_documents, top_passages
from tune_burst_kernel_posterior import load_queries
from tune_burst_multistage_posterior import weighted_rrf
from tune_burst_pairwise import fixed_metrics


def load_expanded(root, queries, blocks, raw_scores):
    raw = {}
    for tag, ids in blocks.items():
        cache = load_cache(root, tag)
        raw.update({q: raw_union(cache[q]) for q in ids})
    dense = {q: sorted(raw[q], key=lambda d: (-raw_scores[q][d], d)) for q in raw}
    expanded = weighted_rrf([raw, dense], (.55, .45), 10)
    return raw, dense, expanded


def main():
    root = Path(__file__).resolve().parent
    queries = load_queries(root)
    qids = list(queries)
    blocks = {
        "validation_a": qids[750:850],
        "fresh_1251_1350": qids[1250:1350],
        "fresh_1351_1450": qids[1350:1450],
    }
    all_ids = sum(blocks.values(), [])
    old_jina = pickle.loads(
        (root / "results/jina_reranker/holdout_scores_finetuned.pkl").read_bytes()
    )["scores"]
    old_dense = pickle.loads(
        (root / "results/aiteamvn_dense/holdout_scores_512.pkl").read_bytes()
    )["scores"]
    expansion_scores = pickle.loads(
        (root / "results/dense_expansion/union50_scores.pkl").read_bytes()
    )["scores"]
    base = {q: list(old_jina[q]) for q in all_ids}
    _, _, expanded = load_expanded(root, queries, blocks, expansion_scores)
    candidates = {q: list(dict.fromkeys(base[q] + expanded[q][:20])) for q in all_ids}
    print("Candidate sizes", np.min([len(candidates[q]) for q in all_ids]),
          np.mean([len(candidates[q]) for q in all_ids]),
          np.max([len(candidates[q]) for q in all_ids]), flush=True)

    documents = load_documents(root / "DSC2026-LegalIR-main/v4_run/public_test_dataset")
    output = root / "results/expanded_rerank"
    output.mkdir(parents=True, exist_ok=True)
    cache_path = output / "scores.pkl"
    saved = {"jina": {}, "dense": {}}
    if cache_path.exists():
        saved = pickle.loads(cache_path.read_bytes())

    jina_path = root / "models/jina-reranker-v2-base-multilingual"
    jina_tok = AutoTokenizer.from_pretrained(jina_path, trust_remote_code=True,
                                             fix_mistral_regex=True)
    jina_model = AutoModelForSequenceClassification.from_pretrained(
        jina_path, trust_remote_code=True, dtype=torch.bfloat16
    )
    finetuned = torch.load(root / "results/jina_reranker/burst_pairwise_state.pt",
                           map_location="cpu", weights_only=True)
    jina_model.load_state_dict(finetuned["state_dict"], strict=False)
    jina_model._tokenizer = jina_tok
    jina_model.eval().to("cuda")
    dense_path = root / "models/AITeamVN_Vietnamese_Embedding"
    dense_tok = AutoTokenizer.from_pretrained(dense_path)
    dense_model = AutoModel.from_pretrained(dense_path, dtype=torch.float16).eval().to("cuda")
    print(f"Models on {torch.cuda.get_device_name(0)}", flush=True)

    started = time.perf_counter()
    for qi, q in enumerate(all_ids, 1):
        if q in saved["jina"] and q in saved["dense"]:
            continue
        question = queries[q][0]
        js = dict(old_jina[q])
        ds = dict(old_dense[q])
        missing = [d for d in candidates[q] if d not in js or d not in ds]
        owners, passages, pairs = [], [], []
        for doc in missing:
            for passage in top_passages(question, documents[doc], count=2):
                owners.append(doc)
                passages.append(passage)
                pairs.append((question, passage))
        if pairs:
            jraw = jina_model.compute_score(pairs, batch_size=16, max_length=512)
            qvec = encode_cls(dense_model, dense_tok, [question], 1, 512)[0]
            pvec = encode_cls(dense_model, dense_tok, passages, 32, 512)
            draw = pvec @ qvec
            for doc, jscore, dscore in zip(owners, jraw, draw):
                js[doc] = max(js.get(doc, -1e9), float(jscore))
                ds[doc] = max(ds.get(doc, -1e9), float(dscore))
        saved["jina"][q] = {d: js[d] for d in candidates[q]}
        saved["dense"][q] = {d: ds[d] for d in candidates[q]}
        if qi % 5 == 0:
            cache_path.write_bytes(pickle.dumps(saved, protocol=5))
            print(f"Expanded rerank scored {qi}/{len(all_ids)} "
                  f"({time.perf_counter()-started:.1f}s)", flush=True)
    cache_path.write_bytes(pickle.dumps(saved, protocol=5))

    jina = {q: sorted(candidates[q], key=lambda d: (-saved["jina"][q][d], d))
            for q in all_ids}
    dense = {q: sorted(candidates[q], key=lambda d: (-saved["dense"][q][d], d))
             for q in all_ids}
    tune_ids = blocks["validation_a"] + blocks["fresh_1251_1350"]
    gold = {q: queries[q] for q in tune_ids}
    trials = []
    values = [i / 20 for i in range(0, 9)]
    for wb in [i / 20 for i in range(5, 15)]:
        for we in values:
            for wj in values:
                wd = 1.0 - wb - we - wj
                if wd < -1e-8 or wd > .50 + 1e-8:
                    continue
                for k in (0, 2, 5, 10, 20, 40):
                    ranked = weighted_rrf(
                        [{q: base[q] for q in tune_ids},
                         {q: expanded[q] for q in tune_ids},
                         {q: jina[q] for q in tune_ids},
                         {q: dense[q] for q in tune_ids}],
                        (wb, we, wj, max(0., wd)), k,
                    )
                    m, _ = fixed_metrics(ranked, gold)
                    trials.append((m["Recall@5"], m["Precision@5"], m["nDCG@10"],
                                   wb, we, wj, max(0., wd), k, m))
    trials.sort(reverse=True, key=lambda x: (x[0], x[1], x[2]))
    _, _, _, wb, we, wj, wd, k, tune_metrics = trials[0]
    report = {"candidate_union": "burst20 + expanded20", "best_weights": {
        "burst": wb, "expanded": we, "jina": wj, "dense": wd, "rrf_k": k,
        "tune": tune_metrics}, "blocks": {}}
    for tag, ids in blocks.items():
        block_gold = {q: queries[q] for q in ids}
        base_m, base_p = fixed_metrics({q: base[q] for q in ids}, block_gold)
        candidate_ceiling = sum(len(set(candidates[q]) & queries[q][1]) /
                                len(queries[q][1]) for q in ids) / len(ids)
        fused = weighted_rrf(
            [{q: base[q] for q in ids}, {q: expanded[q] for q in ids},
             {q: jina[q] for q in ids}, {q: dense[q] for q in ids}],
            (wb, we, wj, wd), k,
        )
        fm, fp = fixed_metrics(fused, block_gold)
        report["blocks"][tag] = {
            "Multistage": base_m, "CandidateRecall": candidate_ceiling,
            "ExpandedRerank": fm,
            "paired_vs_multistage": {"wins": sum(a > b for a, b in zip(fp, base_p)),
                                      "ties": sum(a == b for a, b in zip(fp, base_p)),
                                      "losses": sum(a < b for a, b in zip(fp, base_p))},
        }
    path = root / "burst_expanded_rerank_validation.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    print(f"Saved {path}", flush=True)


if __name__ == "__main__":
    main()
