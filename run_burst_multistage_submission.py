"""Generate the CPU-only BURST Multistage-Posterior submission (top-5)."""

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

from benchmark_burst_v4_full_sqlite import tokens
from tune_burst_empirical_bayes_ltr import label_frequency
from tune_burst_empirical_pairwise import features_for, rank as pairwise_rank
from tune_burst_graph_posterior import build_graph, graph_rerank
from tune_burst_legal_features import enhanced_features
from tune_burst_memory import build_query_memory
from tune_burst_multistage_posterior import weighted_rrf
from tune_burst_pairwise import blend_rankings
from tune_burst_score_ltr import retrieve, score_features
from tune_burst_supervised_profile_bm25 import build_profiles, profile_rank


def load_metadata(data):
    paths = sorted((data / "selected-contexts").glob("context_*.json"))
    doc_ids = [p.stem[len("context_"):] for p in paths]
    raw = json.loads((data / "train.json").read_text(encoding="utf-8"))
    train = {str(q): (x["question"], {str(d) for d in x["answer"]})
             for q, x in raw.items() if x.get("answer")}
    raw_public = json.loads((data / "public-official.json").read_text(encoding="utf-8"))
    public = {str(q): x["question"] for q, x in raw_public.items()}
    return paths, doc_ids, train, public


def main():
    root = Path(__file__).resolve().parent
    default_data = root / "DSC2026-LegalIR-main" / "v4_run" / "public_test_dataset"
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=default_data)
    ap.add_argument("--db", type=Path, default=root / "benchmarks" / "legalir_full_fts.sqlite")
    ap.add_argument("--output-dir", type=Path,
                    default=root / "results" / "burst_multistage")
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths, doc_ids, train, public = load_metadata(args.data_dir)
    valid = set(doc_ids)
    public_ids = list(public)

    retrieval_path = args.output_dir / "public_retrieval.pkl"
    old_retrieval = root / "results" / "burst_robust_fusion" / "public_retrieval.pkl"
    retrieval_checkpoint = args.output_dir / "public_retrieval.checkpoint.pkl"
    retrieval_cache = {}
    for path in (retrieval_path, old_retrieval, retrieval_checkpoint):
        if path.exists():
            saved = pickle.loads(path.read_bytes())
            if saved.get("qids") == public_ids:
                retrieval_cache.update(saved.get("cache", {}))
    print(f"Public retrieval cache {len(retrieval_cache)}/{len(public_ids)}", flush=True)
    if len(retrieval_cache) < len(public_ids):
        conn = sqlite3.connect(args.db)
        build_query_memory(conn, train)
        conn.close()
        missing = [q for q in public_ids if q not in retrieval_cache]
        local = threading.local()

        def one(q):
            if not hasattr(local, "conn"):
                local.conn = sqlite3.connect(args.db)
            return q, retrieve(local.conn, doc_ids, None, public[q])

        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for i, (q, result) in enumerate(pool.map(one, missing), 1):
                retrieval_cache[q] = result
                if i % 25 == 0:
                    retrieval_checkpoint.write_bytes(pickle.dumps(
                        {"qids": public_ids, "cache": retrieval_cache}, protocol=5
                    ))
                    print(f"Retrieved {i}/{len(missing)} "
                          f"({time.perf_counter()-started:.1f}s)", flush=True)
        retrieval_path.write_bytes(pickle.dumps(
            {"qids": public_ids, "cache": retrieval_cache}, protocol=5
        ))
        retrieval_checkpoint.unlink(missing_ok=True)

    print("Building supervised train profiles and co-relevance graph", flush=True)
    train_ids = list(train)
    profile_model = build_profiles(train, train_ids)
    graph_frequency, graph_adjacency, _ = build_graph(train, train_ids)
    frequency = label_frequency(train, set())

    print("Loading and normalizing 8,532 documents", flush=True)
    normalized = {}
    for i, path in enumerate(paths, 1):
        row = json.loads(path.read_text(encoding="utf-8"))
        normalized[str(row["id"])] = " " + " ".join(tokens(row.get("passage") or "")) + " "
        if i % 1000 == 0:
            print(f"Documents {i}/{len(paths)}", flush=True)

    large_saved = pickle.loads(
        (root / "results" / "burst_large_ltr" / "best_model.pkl").read_bytes()
    )
    large_model = large_saved["model"]
    legal_model = pickle.loads(
        (root / "results" / "burst_legal_features" / "validation_model.pkl").read_bytes()
    )["model"]
    pair_saved = pickle.loads(
        (root / "results" / "burst_empirical_pairwise" / "model.pkl").read_bytes()
    )
    public_queries = {q: (text, set()) for q, text in public.items()}
    pair_features = features_for(retrieval_cache, public_queries, public_ids, frequency)

    config = {
        "version": "BURST-MultistagePosterior-v1",
        "robust_alpha": .275, "robust_k": 40,
        "pair_alpha": .30, "pair_k": 2,
        "profile": {"max_ngram": 2, "k1": 1.2, "b": .75, "prior_power": .3},
        "graph": {"seed_depth": 3, "seed_power": .5, "min_edge": 3,
                  "alpha": .4},
        "stack_weights": [.40, .30, .15, .15], "stack_k": 0,
        "lexical_depth": 20, "top_k": 5,
    }
    checkpoint = args.output_dir / "submission.checkpoint.pkl"
    predictions = {}
    if checkpoint.exists():
        saved = pickle.loads(checkpoint.read_bytes())
        if saved.get("config") == config:
            predictions = saved.get("predictions", {})
    print(f"Prediction cache {len(predictions)}/{len(public_ids)}", flush=True)

    started = time.perf_counter()
    for q in public_ids:
        if q in predictions:
            continue
        lists = retrieval_cache[q]
        candidates, x = score_features(lists)
        large = [candidates[i] for i in np.argsort(-large_model.predict(x))[:100]]

        candidates, x = enhanced_features(
            lists, public[q], normalized, lexical_depth=20
        )
        legal = [candidates[i] for i in
                 np.argsort(-legal_model.decision_function(x))[:100]]
        robust = blend_rankings({q: large}, {q: legal[:10]}, .275, 40)[q]

        pair = pairwise_rank(pair_saved["model"], pair_saved["scaler"],
                             {q: pair_features[q]}, [q])[q]
        profile = profile_rank(public[q], profile_model, 2, 1.2, .75, .3)
        graph = graph_rerank(robust, graph_frequency, graph_adjacency,
                             3, .5, "conditional", 3, .4, 0)
        final = weighted_rrf(
            [{q: robust}, {q: pair}, {q: profile}, {q: graph}],
            (.40, .30, .15, .15), 0,
        )[q][:5]
        if len(final) < 5:
            final.extend(d for d in doc_ids if d not in final and len(final) < 5)
        predictions[q] = {"answer": final}
        if len(predictions) % 25 == 0:
            checkpoint.write_bytes(pickle.dumps(
                {"config": config, "predictions": predictions}, protocol=5
            ))
            print(f"Scored {len(predictions)}/1000 "
                  f"({time.perf_counter()-started:.1f}s)", flush=True)

    if set(predictions) != set(public):
        raise RuntimeError("QID mismatch")
    for q, row in predictions.items():
        answers = row["answer"]
        if len(answers) != 5 or len(set(answers)) != 5 or any(d not in valid for d in answers):
            raise RuntimeError(f"Invalid prediction: {q}")
    out_json = args.output_dir / "submission.json"
    out_zip = args.output_dir / "submission.zip"
    out_json.write_text(json.dumps(predictions, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(out_json, arcname="submission.json")
    checkpoint.unlink(missing_ok=True)
    print(f"Saved: {out_zip}", flush=True)


if __name__ == "__main__":
    main()
