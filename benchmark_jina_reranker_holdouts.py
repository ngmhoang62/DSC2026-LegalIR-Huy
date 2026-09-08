"""GPU Jina passage reranking over clean BURST Multistage holdouts."""

from __future__ import annotations

import argparse
import json
import math
import pickle
import re
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from benchmark_burst_v4_full_sqlite import tokens
from tune_burst_empirical_bayes_ltr import label_frequency
from tune_burst_empirical_pairwise import features_for, rank as pairwise_rank
from tune_burst_graph_posterior import build_graph, graph_rerank
from tune_burst_kernel_posterior import load_queries, robust_rankings
from tune_burst_multistage_posterior import weighted_rrf
from tune_burst_pairwise import blend_rankings, fixed_metrics
from tune_burst_supervised_profile_bm25 import build_profiles, profile_rank


STOPWORDS = {
    "bị", "các", "có", "của", "cho", "được", "để", "đến", "đối", "gì",
    "hay", "khi", "không", "là", "làm", "một", "nào", "những", "như",
    "phải", "ra", "sẽ", "theo", "thì", "thế", "trong", "trên", "từ",
    "và", "về", "với", "việc", "bao", "nhiêu", "người", "quy", "định",
}
SPACE_RE = re.compile(r"\S+", re.UNICODE)


def top_passages(question, text, count=2, window=220, overlap=70):
    """Cheap lexical preselection; return original-Unicode windows for Jina."""
    words = SPACE_RE.findall(text or "")
    if len(words) <= window + 80:
        return [" ".join(words)]
    query_tokens = tokens(question)
    content = {t for t in query_tokens if len(t) >= 3 and t not in STOPWORDS}
    numbers = {t for t in query_tokens if any(c.isdigit() for c in t)}
    bigrams = {" ".join(query_tokens[i:i+2]) for i in range(len(query_tokens)-1)}
    header = " ".join(words[:70])
    scored = []
    step = window - overlap
    for start in range(0, len(words), step):
        end = min(start + window, len(words))
        part_words = words[start:end]
        part = " ".join(part_words)
        normalized = tokens(part)
        token_set = set(normalized)
        norm_text = " ".join(normalized)
        coverage = sum(1.0 + .20 * min(normalized.count(t), 3)
                       for t in content if t in token_set)
        numeric = 3.0 * sum(t in token_set for t in numbers)
        phrase = 1.8 * sum(p in norm_text for p in bigrams)
        density = (coverage + numeric + phrase) / math.sqrt(max(len(normalized), 1))
        scored.append((density, coverage + numeric + phrase, -start, part))
        if end == len(words):
            break
    scored.sort(reverse=True)
    passages = []
    for _, _, neg_start, part in scored:
        candidate = part if -neg_start < 70 else header + "\n[ĐOẠN PHÙ HỢP]\n" + part
        if candidate not in passages:
            passages.append(candidate)
        if len(passages) >= count:
            break
    return passages


def load_cache(root, tag):
    large = root / "results" / "burst_large_ltr"
    if tag == "validation_a":
        return pickle.loads(
            (large / "retrieval_train1000_tune50_val100.pkl").read_bytes()
        )["cache"]
    return pickle.loads((large / f"{tag}_retrieval.pkl").read_bytes())["cache"]


def build_multistage(root, queries, ids, tag, pair_saved):
    cache_all = load_cache(root, tag)
    cache = {q: cache_all[q] for q in ids}
    robust = robust_rankings(root, queries, ids)
    frequency = label_frequency(queries, set(ids))
    pair_features = features_for(cache, queries, ids, frequency)
    pair = pairwise_rank(pair_saved["model"], pair_saved["scaler"], pair_features, ids)
    memory = [q for q in queries if q not in set(ids)]
    profile_model = build_profiles(queries, memory)
    profile = {q: profile_rank(queries[q][0], profile_model, 2, 1.2, .75, .3)
               for q in ids}
    graph_frequency, adjacency, _ = build_graph(queries, memory)
    graph = {q: graph_rerank(robust[q], graph_frequency, adjacency,
                             3, .5, "conditional", 3, .4, 0) for q in ids}
    final = weighted_rrf([robust, pair, profile, graph], (.40, .30, .15, .15), 0)
    return final


def _title_from_link(link):
    """Same fallback as run_burst_expanded_fusion_submission.title_from_link
    (duplicated locally to avoid a circular import -- that module imports
    top_passages from here): 20/8532 docs have an empty "passage" field, but
    their "link" URL slug still carries a readable title."""
    if not link:
        return ""
    from urllib.parse import urlparse
    slug = urlparse(link).path.rsplit("/", 1)[-1]
    slug = re.sub(r"\.aspx$", "", slug, flags=re.I)
    slug = re.sub(r"-\d+$", "", slug)
    return slug.replace("-", " ").strip()


def load_documents(data):
    documents = {}
    for i, path in enumerate(sorted((data / "selected-contexts").glob("context_*.json")), 1):
        row = json.loads(path.read_text(encoding="utf-8"))
        documents[str(row["id"])] = row.get("passage") or _title_from_link(row.get("link"))
        if i % 1000 == 0:
            print(f"Documents {i}/8532", flush=True)
    return documents


def score_block(model, queries, ids, rankings, documents, score_cache, depth=20):
    started = time.perf_counter()
    for qi, q in enumerate(ids, 1):
        if q in score_cache:
            continue
        owners, pairs = [], []
        for doc in rankings[q][:depth]:
            for passage in top_passages(queries[q][0], documents[doc], count=2):
                owners.append(doc)
                pairs.append((queries[q][0], passage))
        raw_scores = model.compute_score(pairs, batch_size=16, max_length=512)
        doc_scores = {doc: 0.0 for doc in rankings[q][:depth]}
        for doc, score in zip(owners, raw_scores):
            doc_scores[doc] = max(doc_scores[doc], float(score))
        score_cache[q] = doc_scores
        if qi % 10 == 0:
            print(f"GPU scored {qi}/{len(ids)} ({time.perf_counter()-started:.1f}s)",
                  flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--cache-name", default="holdout_scores.pkl")
    parser.add_argument("--report-name", default="burst_jina_reranker_validation.json")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    data = root / "DSC2026-LegalIR-main" / "v4_run" / "public_test_dataset"
    queries = load_queries(root)
    qids = list(queries)
    blocks = {
        "validation_a": qids[750:850],
        "fresh_1251_1350": qids[1250:1350],
        "fresh_1351_1450": qids[1350:1450],
    }
    pair_saved = pickle.loads(
        (root / "results" / "burst_empirical_pairwise" / "model.pkl").read_bytes()
    )
    multistage = {}
    for tag, ids in blocks.items():
        print(f"Building Multistage candidates: {tag}", flush=True)
        multistage.update(build_multistage(root, queries, ids, tag, pair_saved))
    documents = load_documents(data)

    output = root / "results" / "jina_reranker"
    output.mkdir(parents=True, exist_ok=True)
    cache_path = output / args.cache_name
    score_cache = {}
    if cache_path.exists():
        saved = pickle.loads(cache_path.read_bytes())
        if saved.get("depth") == 20:
            score_cache = saved.get("scores", {})

    model_path = root / "models" / "jina-reranker-v2-base-multilingual"
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, fix_mistral_regex=True
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, trust_remote_code=True, dtype=torch.bfloat16,
    )
    if args.checkpoint:
        saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        missing, unexpected = model.load_state_dict(saved["state_dict"], strict=False)
        print(f"Loaded fine-tuned checkpoint: {args.checkpoint}", flush=True)
        print(f"Partial state load: missing={len(missing)} unexpected={len(unexpected)}",
              flush=True)
    model._tokenizer = tokenizer
    model.eval().to("cuda")
    print(f"Jina on {torch.cuda.get_device_name(0)}", flush=True)

    for tag, ids in blocks.items():
        score_block(model, queries, ids, multistage, documents, score_cache, 20)
        cache_path.write_bytes(pickle.dumps(
            {"depth": 20, "scores": score_cache}, protocol=5
        ))

    semantic = {
        q: sorted(score_cache[q], key=lambda d: (-score_cache[q][d], d))
        for q in score_cache
    }
    tune_ids = blocks["validation_a"] + blocks["fresh_1251_1350"]
    tune_gold = {q: queries[q] for q in tune_ids}
    trials = []
    for alpha in np.linspace(.02, 1.0, 50):
        for k in (0, 2, 5, 10, 20, 40, 80):
            fused = blend_rankings({q: semantic[q] for q in tune_ids},
                                   {q: multistage[q] for q in tune_ids},
                                   float(alpha), k)
            metrics, _ = fixed_metrics(fused, tune_gold)
            trials.append((metrics["Recall@5"], metrics["Precision@5"],
                           metrics["nDCG@10"], float(alpha), k, metrics))
    trials.sort(reverse=True, key=lambda x: (x[0], x[1], x[2]))
    _, _, _, alpha, k, tune_metrics = trials[0]

    model_name = "jina-reranker-v2-base-multilingual"
    if args.checkpoint:
        model_name += "+burst-pairwise"
    report = {"model": model_name,
              "candidate_depth": 20, "passages_per_doc": 2,
              "best_fusion": {"semantic_alpha": alpha, "rrf_k": k,
                              "tune": tune_metrics}, "blocks": {}}
    for tag, ids in blocks.items():
        gold = {q: queries[q] for q in ids}
        base_m, base_p = fixed_metrics({q: multistage[q] for q in ids}, gold)
        sem_m, _ = fixed_metrics({q: semantic[q] for q in ids}, gold)
        fused = blend_rankings({q: semantic[q] for q in ids},
                               {q: multistage[q] for q in ids}, alpha, k)
        fused_m, fused_p = fixed_metrics(fused, gold)
        ceiling = sum(len(set(multistage[q][:20]) & queries[q][1]) / len(queries[q][1])
                      for q in ids) / len(ids)
        report["blocks"][tag] = {
            "candidate_Recall@20": ceiling, "Multistage": base_m,
            "Jina_only": sem_m, "Jina_fusion": fused_m,
            "paired_vs_multistage": {
                "wins": sum(a > b for a, b in zip(fused_p, base_p)),
                "ties": sum(a == b for a, b in zip(fused_p, base_p)),
                "losses": sum(a < b for a, b in zip(fused_p, base_p)),
            },
        }
    report_path = root / args.report_name
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    print(f"Saved {report_path}", flush=True)


if __name__ == "__main__":
    main()
