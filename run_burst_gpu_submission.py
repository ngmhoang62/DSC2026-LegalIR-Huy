"""Generate the top-5 BURST + fine-tuned Jina + multilingual-E5 submission."""

from __future__ import annotations

import argparse
import json
import pickle
import sqlite3
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

from benchmark_burst_v4_full_sqlite import tokens
from benchmark_e5_holdouts import encode
from benchmark_jina_reranker_holdouts import top_passages
from run_burst_multistage_submission import load_metadata
from tune_burst_empirical_bayes_ltr import label_frequency
from tune_burst_empirical_pairwise import features_for, rank as pairwise_rank
from tune_burst_graph_posterior import build_graph, graph_rerank
from tune_burst_legal_features import enhanced_features
from tune_burst_memory import build_query_memory
from tune_burst_multistage_posterior import weighted_rrf
from tune_burst_pairwise import blend_rankings
from tune_burst_score_ltr import retrieve, score_features
from tune_burst_supervised_profile_bm25 import build_profiles, profile_rank


CPU_CONFIG = {
    "version": "BURST-MultistagePosterior-v1", "robust_alpha": .275,
    "robust_k": 40, "stack_weights": (.40, .30, .15, .15), "stack_k": 0,
}
GPU_CONFIG = {
    "version": "BURST-ThreeViewGPU-v1", "candidate_depth": 20,
    "passages_per_doc": 2, "max_length": 384,
    "weights_base_jina_e5": (.50, .20, .30), "rrf_k": 20, "top_k": 5,
}


def main():
    root = Path(__file__).resolve().parent
    default_data = root / "DSC2026-LegalIR-main/v4_run/public_test_dataset"
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=default_data)
    ap.add_argument("--db", type=Path,
                    default=root / "benchmarks/legalir_full_fts.sqlite")
    ap.add_argument("--output-dir", type=Path,
                    default=root / "results/burst_gpu_threeview")
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    paths, doc_ids, train, public = load_metadata(args.data_dir)
    public_ids = list(public)
    valid = set(doc_ids)

    retrieval_path = root / "results/burst_multistage/public_retrieval.pkl"
    if not retrieval_path.exists():
        retrieval_path = root / "results/burst_robust_fusion/public_retrieval.pkl"
    saved_retrieval = pickle.loads(retrieval_path.read_bytes())
    retrieval_cache = saved_retrieval.get("cache", {})
    if any(q not in retrieval_cache for q in public_ids):
        print("Completing missing public retrieval cache", flush=True)
        conn = sqlite3.connect(args.db)
        build_query_memory(conn, train)
        conn.close()
        missing = [q for q in public_ids if q not in retrieval_cache]
        local = threading.local()

        def one(qid):
            if not hasattr(local, "conn"):
                local.conn = sqlite3.connect(args.db)
            return qid, retrieve(local.conn, doc_ids, None, public[qid])

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for i, (qid, result) in enumerate(pool.map(one, missing), 1):
                retrieval_cache[qid] = result
                if i % 25 == 0:
                    print(f"Retrieved {i}/{len(missing)}", flush=True)

    documents, normalized = {}, {}
    print("Loading 8,532 documents", flush=True)
    for i, path in enumerate(paths, 1):
        row = json.loads(path.read_text(encoding="utf-8"))
        doc = str(row["id"])
        text = row.get("passage") or ""
        documents[doc] = text
        normalized[doc] = " " + " ".join(tokens(text)) + " "
        if i % 1000 == 0:
            print(f"Documents {i}/{len(paths)}", flush=True)

    cpu_path = args.output_dir / "cpu_top20.pkl"
    cpu_rankings = {}
    if cpu_path.exists():
        saved = pickle.loads(cpu_path.read_bytes())
        if saved.get("config") == CPU_CONFIG:
            cpu_rankings = saved.get("rankings", {})
    if len(cpu_rankings) < len(public_ids):
        print("Building CPU multistage top-20 candidates", flush=True)
        train_ids = list(train)
        profile_model = build_profiles(train, train_ids)
        graph_frequency, graph_adjacency, _ = build_graph(train, train_ids)
        frequency = label_frequency(train, set())
        large_model = pickle.loads(
            (root / "results/burst_large_ltr/best_model.pkl").read_bytes()
        )["model"]
        legal_model = pickle.loads(
            (root / "results/burst_legal_features/validation_model.pkl").read_bytes()
        )["model"]
        pair_saved = pickle.loads(
            (root / "results/burst_empirical_pairwise/model.pkl").read_bytes()
        )
        public_queries = {q: (text, set()) for q, text in public.items()}
        pair_features = features_for(retrieval_cache, public_queries,
                                     public_ids, frequency)
        started = time.perf_counter()
        for qid in public_ids:
            if qid in cpu_rankings:
                continue
            lists = retrieval_cache[qid]
            candidates, x = score_features(lists)
            large = [candidates[i] for i in
                     np.argsort(-large_model.predict(x))[:100]]
            candidates, x = enhanced_features(
                lists, public[qid], normalized, lexical_depth=20
            )
            legal = [candidates[i] for i in
                     np.argsort(-legal_model.decision_function(x))[:100]]
            robust = blend_rankings({qid: large}, {qid: legal[:10]}, .275, 40)[qid]
            pair = pairwise_rank(pair_saved["model"], pair_saved["scaler"],
                                 {qid: pair_features[qid]}, [qid])[qid]
            profile = profile_rank(public[qid], profile_model, 2, 1.2, .75, .3)
            graph = graph_rerank(robust, graph_frequency, graph_adjacency,
                                 3, .5, "conditional", 3, .4, 0)
            cpu_rankings[qid] = weighted_rrf(
                [{qid: robust}, {qid: pair}, {qid: profile}, {qid: graph}],
                CPU_CONFIG["stack_weights"], CPU_CONFIG["stack_k"]
            )[qid][:GPU_CONFIG["candidate_depth"]]
            if len(cpu_rankings) % 25 == 0:
                cpu_path.write_bytes(pickle.dumps(
                    {"config": CPU_CONFIG, "rankings": cpu_rankings}, protocol=5
                ))
                print(f"CPU ranked {len(cpu_rankings)}/1000 "
                      f"({time.perf_counter()-started:.1f}s)", flush=True)
        cpu_path.write_bytes(pickle.dumps(
            {"config": CPU_CONFIG, "rankings": cpu_rankings}, protocol=5
        ))

    print("Loading fine-tuned Jina and multilingual-E5", flush=True)
    jina_path = root / "models/jina-reranker-v2-base-multilingual"
    jina_tokenizer = AutoTokenizer.from_pretrained(
        jina_path, trust_remote_code=True, fix_mistral_regex=True
    )
    jina_model = AutoModelForSequenceClassification.from_pretrained(
        jina_path, trust_remote_code=True, dtype=torch.bfloat16
    )
    checkpoint = torch.load(root / "results/jina_reranker/burst_pairwise_state.pt",
                            map_location="cpu", weights_only=True)
    jina_model.load_state_dict(checkpoint["state_dict"], strict=False)
    jina_model._tokenizer = jina_tokenizer
    jina_model.eval().to("cuda")

    e5_path = root / "models/multilingual-e5-small"
    e5_tokenizer = AutoTokenizer.from_pretrained(e5_path)
    e5_model = AutoModel.from_pretrained(
        e5_path, dtype=torch.float16
    ).eval().to("cuda")
    print(f"Models ready on {torch.cuda.get_device_name(0)}", flush=True)

    score_path = args.output_dir / "gpu_scores.checkpoint.pkl"
    gpu_scores = {}
    if score_path.exists():
        saved = pickle.loads(score_path.read_bytes())
        if saved.get("config") == GPU_CONFIG:
            gpu_scores = saved.get("scores", {})
    started = time.perf_counter()
    for qi, qid in enumerate(public_ids, 1):
        if qid in gpu_scores:
            continue
        question = public[qid]
        owners, passages, jina_pairs = [], [], []
        for doc in cpu_rankings[qid]:
            for passage in top_passages(question, documents[doc], count=2):
                owners.append(doc)
                passages.append("passage: " + passage)
                jina_pairs.append((question, passage))
        jina_raw = jina_model.compute_score(
            jina_pairs, batch_size=16, max_length=GPU_CONFIG["max_length"]
        )
        qvector = encode(e5_model, e5_tokenizer, ["query: " + question], 1,
                         GPU_CONFIG["max_length"])[0]
        pvectors = encode(e5_model, e5_tokenizer, passages, 32,
                          GPU_CONFIG["max_length"])
        e5_raw = pvectors @ qvector
        js = {doc: 0.0 for doc in cpu_rankings[qid]}
        es = {doc: -1.0 for doc in cpu_rankings[qid]}
        for doc, jscore, escore in zip(owners, jina_raw, e5_raw):
            js[doc] = max(js[doc], float(jscore))
            es[doc] = max(es[doc], float(escore))
        gpu_scores[qid] = {"jina": js, "e5": es}
        if qi % 10 == 0:
            score_path.write_bytes(pickle.dumps(
                {"config": GPU_CONFIG, "scores": gpu_scores}, protocol=5
            ))
            print(f"GPU scored {qi}/1000 ({time.perf_counter()-started:.1f}s)",
                  flush=True)
    score_path.write_bytes(pickle.dumps(
        {"config": GPU_CONFIG, "scores": gpu_scores}, protocol=5
    ))

    predictions = {}
    for qid in public_ids:
        candidates = cpu_rankings[qid]
        jina = sorted(candidates,
                      key=lambda d: (-gpu_scores[qid]["jina"][d], d))
        e5 = sorted(candidates, key=lambda d: (-gpu_scores[qid]["e5"][d], d))
        final = weighted_rrf(
            [{qid: candidates}, {qid: jina}, {qid: e5}],
            GPU_CONFIG["weights_base_jina_e5"], GPU_CONFIG["rrf_k"]
        )[qid][:5]
        if len(final) < 5:
            final.extend(d for d in doc_ids if d not in final and len(final) < 5)
        predictions[qid] = {"answer": final}

    if set(predictions) != set(public):
        raise RuntimeError("QID mismatch")
    for qid, row in predictions.items():
        answers = row["answer"]
        if len(answers) != 5 or len(set(answers)) != 5 or any(
                doc not in valid for doc in answers):
            raise RuntimeError(f"Invalid prediction: {qid}")
    out_json = args.output_dir / "submission.json"
    out_zip = args.output_dir / "submission.zip"
    out_json.write_text(json.dumps(predictions, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.write(out_json, arcname="submission.json")
    metadata = {"cpu": CPU_CONFIG, "gpu": GPU_CONFIG,
                "queries": len(predictions), "documents_per_query": 5}
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Saved: {out_zip}", flush=True)


if __name__ == "__main__":
    main()
