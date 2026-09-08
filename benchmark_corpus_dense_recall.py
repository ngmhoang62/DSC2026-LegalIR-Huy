"""What does full-corpus dense retrieval add to the candidate pool?

Ranks all 8,532 documents for each holdout query by max chunk similarity, then
reports how much of the gold set the dense branch reaches on its own and how far
it lifts the ceiling of the shipped candidate pool.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from benchmark_aiteamvn_holdouts import encode_cls
from tune_expanded_fusion_robust import build_views


def load_index(root, cap):
    directory = root / "results/corpus_index"
    meta = json.loads((directory / f"chunks_cap{cap}.json").read_text(encoding="utf-8"))
    vectors = np.fromfile(directory / f"chunks_cap{cap}.f16", dtype=np.float16)
    vectors = vectors.reshape(-1, 1024)
    counts = np.asarray(meta["counts"])
    if vectors.shape[0] != counts.sum():
        raise RuntimeError(f"Index mismatch: {vectors.shape[0]} vs {counts.sum()}")
    owners = np.repeat(np.arange(len(counts)), counts)
    return meta["documents"], vectors, owners


def rank_documents(vectors, owners, doc_count, query_vectors, top_k=100):
    """Max-pool chunk similarity per document, then take the top documents."""
    ranked = []
    for qvec in query_vectors:
        similarity = vectors @ qvec.astype(np.float16)
        best = np.full(doc_count, -np.inf, dtype=np.float32)
        np.maximum.at(best, owners, similarity.astype(np.float32))
        order = np.argpartition(-best, min(top_k, doc_count - 1))[:top_k]
        order = order[np.argsort(-best[order])]
        ranked.append((order, best))
    return ranked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, default=16)
    args = ap.parse_args()
    root = Path(__file__).resolve().parent
    queries, blocks, all_ids, candidates, views = build_views(root)
    documents, vectors, owners = load_index(root, args.cap)
    index = {doc: i for i, doc in enumerate(documents)}
    print(f"Index: {len(documents)} documents, {vectors.shape[0]} chunks", flush=True)

    model_path = root / "models/AITeamVN_Vietnamese_Embedding"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(
        model_path, dtype=torch.float16).eval().to("cuda")
    qvectors = encode_cls(model, tokenizer, [queries[q][0] for q in all_ids], 16, 512)
    del model
    torch.cuda.empty_cache()

    ranked = rank_documents(vectors, owners, len(documents), qvectors)
    dense_rank = {q: [documents[i] for i in order]
                  for q, (order, _) in zip(all_ids, ranked)}

    output = root / "results/corpus_index"
    (output / f"holdout_dense_rank_cap{args.cap}.pkl").write_bytes(
        pickle.dumps({"cap": args.cap, "ranking": dense_rank,
                      "scores": {q: {documents[i]: float(best[i]) for i in order}
                                 for q, (order, best) in zip(all_ids, ranked)}},
                     protocol=5))

    report = {"cap": args.cap, "chunks": int(vectors.shape[0]), "blocks": {}}
    for name, ids in blocks.items():
        gold = {q: queries[q][1] for q in ids}
        entry = {}
        for k in (5, 20, 50, 100):
            entry[f"dense_recall@{k}"] = float(np.mean(
                [len(set(dense_rank[q][:k]) & gold[q]) / len(gold[q]) for q in ids]))
        entry["current_ceiling"] = float(np.mean(
            [len(set(candidates[q]) & gold[q]) / len(gold[q]) for q in ids]))
        for k in (10, 20, 50):
            entry[f"ceiling_plus_dense{k}"] = float(np.mean(
                [len((set(candidates[q]) | set(dense_rank[q][:k])) & gold[q]) /
                 len(gold[q]) for q in ids]))
        # Gold documents the lexical pipeline can never reach, that dense finds.
        entry["rescued_by_dense20"] = int(sum(
            len((set(dense_rank[q][:20]) - set(candidates[q])) & gold[q]) for q in ids))
        entry["unreachable_gold"] = int(sum(
            len(gold[q] - set(candidates[q]) - set(dense_rank[q][:100])) for q in ids))
        report["blocks"][name] = entry
        print(f"{name}: dense R@20={entry['dense_recall@20']:.4f} "
              f"ceiling {entry['current_ceiling']:.4f} -> "
              f"{entry['ceiling_plus_dense20']:.4f} "
              f"(rescued {entry['rescued_by_dense20']})", flush=True)

    path = root / "burst_corpus_dense_validation.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {path}", flush=True)


if __name__ == "__main__":
    main()
