#!/usr/bin/env python
"""
FAST GPU full-corpus dense recall benchmark for arbitrary cap.

Compatible output:
  results/corpus_index/holdout_dense_rank_cap<CAP>.pkl

Unlike historical benchmark_corpus_dense_recall.py, this keeps the chunk bank on
GPU and batches query x chunk matrix multiplication, then scatter-reduces chunks
to parent documents with MAX. This preserves the exact retrieval semantics.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


@torch.inference_mode()
def encode_cls(model, tokenizer, texts, batch_size=16, max_length=512):
    import torch.nn.functional as F
    rows = []
    for s in range(0, len(texts), batch_size):
        batch = tokenizer(
            texts[s:s+batch_size],
            max_length=max_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        batch = {k: v.to("cuda", non_blocking=True) for k, v in batch.items()}
        cls = model(**batch).last_hidden_state[:, 0]
        rows.append(F.normalize(cls.float(), p=2, dim=1).cpu())
    return torch.cat(rows, dim=0)


def load_index(root, cap):
    d = root / "results/corpus_index"
    meta_path = d / f"chunks_cap{cap}.json"
    vec_path = d / f"chunks_cap{cap}.f16"
    if not meta_path.is_file() or not vec_path.is_file():
        raise FileNotFoundError(
            f"Missing cap{cap} index. Need {meta_path} and {vec_path}"
        )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    counts = np.asarray(meta["counts"], dtype=np.int64)
    vec = np.fromfile(vec_path, dtype=np.float16)
    if vec.size % 1024:
        raise RuntimeError("Vector file is not divisible by embedding dim 1024")
    vec = vec.reshape(-1, 1024)
    if vec.shape[0] != int(counts.sum()):
        raise RuntimeError(
            f"Index mismatch: vectors={vec.shape[0]} counts={counts.sum()}"
        )
    owners = np.repeat(np.arange(len(counts), dtype=np.int64), counts)
    return meta["documents"], vec, owners


@torch.inference_mode()
def gpu_rank(vec_np, owners_np, qvec_cpu, doc_count, batch_size=24, top_k=100):
    device = torch.device("cuda")

    print(
        f"  uploading chunk bank to GPU: {vec_np.shape[0]:,} x {vec_np.shape[1]}",
        flush=True,
    )
    chunks = torch.from_numpy(vec_np).to(device=device, dtype=torch.float16)
    owners = torch.from_numpy(owners_np).to(device=device, dtype=torch.long)

    out = []
    for s in range(0, len(qvec_cpu), batch_size):
        q = qvec_cpu[s:s+batch_size].to(device=device, dtype=torch.float16)
        sim = q @ chunks.T  # [B, chunks]

        B = sim.shape[0]
        best = torch.full(
            (B, doc_count),
            -torch.inf,
            dtype=sim.dtype,
            device=device,
        )
        idx = owners.unsqueeze(0).expand(B, -1)

        if not hasattr(best, "scatter_reduce_"):
            raise RuntimeError(
                "Installed PyTorch lacks scatter_reduce_. "
                "Use historical benchmark script as fallback."
            )

        best.scatter_reduce_(
            1, idx, sim, reduce="amax", include_self=True
        )
        vals, inds = torch.topk(best, k=min(top_k, doc_count), dim=1)

        vals = vals.float().cpu().numpy()
        inds = inds.cpu().numpy()
        for i in range(B):
            out.append((inds[i], vals[i]))

        print(
            f"  ranked queries {min(s+batch_size,len(qvec_cpu))}/{len(qvec_cpu)} "
            f"| VRAM={torch.cuda.memory_allocated()/2**30:.2f} GB",
            flush=True,
        )

        del sim, best, q, idx, vals, inds

    del chunks, owners
    torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=Path, required=True)
    ap.add_argument("--cap", type=int, required=True)
    ap.add_argument("--query-batch", type=int, default=24)
    args = ap.parse_args()

    root = args.repo_root.resolve()
    sys.path.insert(0, str(root))

    from tune_expanded_fusion_robust import build_views

    print("[1/4] Loading CAL600 queries and candidate baseline...", flush=True)
    queries, blocks, ids, candidates, _ = build_views(root)

    print(f"[2/4] Loading cap{args.cap} chunk index...", flush=True)
    documents, vectors, owners = load_index(root, args.cap)
    print(
        f"  docs={len(documents):,} chunks={len(vectors):,} "
        f"bank={vectors.nbytes/2**30:.2f} GiB",
        flush=True,
    )

    print("[3/4] Encoding queries + GPU max-parent retrieval...", flush=True)
    model_path = root / "models/AITeamVN_Vietnamese_Embedding"
    tok = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModel.from_pretrained(
        model_path,
        dtype=torch.float16,
        local_files_only=True,
    ).eval().to("cuda")

    qvec = encode_cls(
        model, tok, [queries[q][0] for q in ids], batch_size=16, max_length=512
    )
    del model
    torch.cuda.empty_cache()

    ranked = gpu_rank(
        vectors,
        owners,
        qvec,
        len(documents),
        batch_size=args.query_batch,
        top_k=100,
    )

    dense_rank = {
        q: [documents[int(i)] for i in inds]
        for q, (inds, vals) in zip(ids, ranked)
    }
    dense_scores = {
        q: {
            documents[int(i)]: float(v)
            for i, v in zip(inds, vals)
        }
        for q, (inds, vals) in zip(ids, ranked)
    }

    output = root / "results/corpus_index"
    cache_path = output / f"holdout_dense_rank_cap{args.cap}.pkl"
    cache_path.write_bytes(
        pickle.dumps(
            {
                "cap": args.cap,
                "ranking": dense_rank,
                "scores": dense_scores,
                "backend": "gpu_exact_max_parent_v1",
            },
            protocol=5,
        )
    )

    print("[4/4] RESULT", flush=True)
    report = {"cap": args.cap, "chunks": int(vectors.shape[0]), "blocks": {}}

    for name, qids in blocks.items():
        g = {q: set(queries[q][1]) for q in qids}
        entry = {}
        for k in (5, 20, 50, 100):
            entry[f"dense_recall@{k}"] = float(np.mean([
                len(set(dense_rank[q][:k]) & g[q]) / len(g[q])
                for q in qids
            ]))
        entry["current_ceiling"] = float(np.mean([
            len(set(candidates[q]) & g[q]) / len(g[q])
            for q in qids
        ]))
        for k in (10, 20, 50):
            entry[f"ceiling_plus_dense{k}"] = float(np.mean([
                len((set(candidates[q]) | set(dense_rank[q][:k])) & g[q])
                / len(g[q])
                for q in qids
            ]))
        entry["rescued_by_dense20"] = int(sum(
            len((set(dense_rank[q][:20]) - set(candidates[q])) & g[q])
            for q in qids
        ))
        report["blocks"][name] = entry
        print(
            f"  {name}: dense R@20={entry['dense_recall@20']:.6f} "
            f"ceiling {entry['current_ceiling']:.6f} -> "
            f"{entry['ceiling_plus_dense20']:.6f} "
            f"rescued={entry['rescued_by_dense20']}",
            flush=True,
        )

    report_path = (
        root
        / f"results/manual/huy_cap{args.cap}_dense_gpu_benchmark_v1/REPORT.json"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("Cache:", cache_path)
    print("Report:", report_path)


if __name__ == "__main__":
    main()
