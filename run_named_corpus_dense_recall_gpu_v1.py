#!/usr/bin/env python
"""
FAST GPU benchmark for a named full-corpus chunk index.

Example:
  python ../run_named_corpus_dense_recall_gpu_v1.py \
    --repo-root /d/Study/DSC2026/sota \
    --name structured_chunks --cap 32 --query-batch 24

Loads:
  results/corpus_index/<name>_cap<CAP>.f16/.json
Writes:
  results/corpus_index/holdout_dense_rank_<name>_cap<CAP>.pkl
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


DIM = 1024


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
        batch = {k:v.to("cuda", non_blocking=True) for k,v in batch.items()}
        cls = model(**batch).last_hidden_state[:,0]
        rows.append(F.normalize(cls.float(), p=2, dim=1).cpu())
    return torch.cat(rows, dim=0)


def load_index(root, name, cap):
    d = root / "results/corpus_index"
    mp = d / f"{name}_cap{cap}.json"
    vp = d / f"{name}_cap{cap}.f16"
    if not mp.is_file() or not vp.is_file():
        raise FileNotFoundError(f"Missing {mp} / {vp}")
    meta = json.loads(mp.read_text(encoding="utf-8"))
    counts = np.asarray(meta["counts"], dtype=np.int64)
    vec = np.fromfile(vp, dtype=np.float16).reshape(-1, DIM)
    if vec.shape[0] != counts.sum():
        raise RuntimeError("vector/count mismatch")
    owners = np.repeat(np.arange(len(counts), dtype=np.int64), counts)
    return meta["documents"], vec, owners


@torch.inference_mode()
def rank_gpu(vec_np, owners_np, qvec, n_docs, batch=24, topk=100):
    chunks = torch.from_numpy(vec_np).to("cuda", dtype=torch.float16)
    owners = torch.from_numpy(owners_np).to("cuda", dtype=torch.long)
    out = []

    for s in range(0, len(qvec), batch):
        q = qvec[s:s+batch].to("cuda", dtype=torch.float16)
        sim = q @ chunks.T
        B = sim.shape[0]
        best = torch.full(
            (B, n_docs), -torch.inf, dtype=sim.dtype, device="cuda"
        )
        idx = owners.unsqueeze(0).expand(B,-1)
        best.scatter_reduce_(1, idx, sim, reduce="amax", include_self=True)
        vals, inds = torch.topk(best, k=min(topk,n_docs), dim=1)
        vals = vals.float().cpu().numpy()
        inds = inds.cpu().numpy()
        out.extend((inds[i], vals[i]) for i in range(B))
        print(
            f"  ranked {min(s+batch,len(qvec))}/{len(qvec)} | "
            f"VRAM={torch.cuda.memory_allocated()/2**30:.2f}GB",
            flush=True,
        )
        del q,sim,best,idx,vals,inds

    del chunks,owners
    torch.cuda.empty_cache()
    return out


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--repo-root",type=Path,required=True)
    ap.add_argument("--name",required=True)
    ap.add_argument("--cap",type=int,required=True)
    ap.add_argument("--query-batch",type=int,default=24)
    args=ap.parse_args()

    root=args.repo_root.resolve(); sys.path.insert(0,str(root))
    from tune_expanded_fusion_robust import build_views

    queries,blocks,ids,pre_corpus,_=build_views(root,expanded_depth=20)
    docs,vec,owners=load_index(root,args.name,args.cap)
    print(
        f"Index {args.name} cap{args.cap}: docs={len(docs)} chunks={len(vec)} "
        f"bank={vec.nbytes/2**30:.2f}GiB",
        flush=True,
    )

    mp=root/"models/AITeamVN_Vietnamese_Embedding"
    tok=AutoTokenizer.from_pretrained(mp,local_files_only=True)
    model=AutoModel.from_pretrained(
        mp,dtype=torch.float16,local_files_only=True
    ).eval().to("cuda")
    qvec=encode_cls(model,tok,[queries[q][0] for q in ids])
    del model; torch.cuda.empty_cache()

    ranked=rank_gpu(vec,owners,qvec,len(docs),args.query_batch,100)
    ranking={
        q:[docs[int(i)] for i in inds]
        for q,(inds,vals) in zip(ids,ranked)
    }
    scores={
        q:{docs[int(i)]:float(v) for i,v in zip(inds,vals)}
        for q,(inds,vals) in zip(ids,ranked)
    }

    out=root/"results/corpus_index"/f"holdout_dense_rank_{args.name}_cap{args.cap}.pkl"
    out.write_bytes(pickle.dumps({
        "name":args.name,"cap":args.cap,"ranking":ranking,"scores":scores,
        "backend":"gpu_exact_max_parent_v1",
    },protocol=5))

    print("RESULT")
    for b,qids in blocks.items():
        def rec(k):
            return float(np.mean([
                len(set(ranking[q][:k])&set(queries[q][1]))/len(queries[q][1])
                for q in qids
            ]))
        ceiling=float(np.mean([
            len((set(pre_corpus[q])|set(ranking[q][:20]))&set(queries[q][1]))
            /len(queries[q][1]) for q in qids
        ]))
        print(f"  {b}: R20={rec(20):.6f} ceiling+20={ceiling:.6f}")
    print("Cache:",out)


if __name__=="__main__":
    main()
