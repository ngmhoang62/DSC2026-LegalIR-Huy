#!/usr/bin/env python
"""
HUY PIPELINE AUDIT J — FULL-CORPUS PARENT AGGREGATION V1
========================================================

GPU audit on the EXISTING cap32 AITeamVN corpus index.

Current production:
    parent_score = max(chunk cosine)

Preregistered alternatives:
    TOP2_MEAN = mean of best 2 available chunk cosines
    TOP3_MEAN = mean of best 3 available chunk cosines

No tuning beyond these three formulas.

Measures:
- standalone dense Recall@5/20/50/100
- candidate ceiling when adding top20/top50 to pre-corpus pool
- per-block deltas
- retrieval churn and unique golds

Run after other GPU jobs finish.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


DEFAULT_CAP = 48
DIM = 1024


@torch.inference_mode()
def encode_cls(model, tok, texts, batch=16, max_length=512):
    import torch.nn.functional as F
    out = []
    for s in range(0, len(texts), batch):
        enc = tok(
            texts[s:s+batch],
            max_length=max_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        enc = {k: v.to("cuda") for k, v in enc.items()}
        cls = model(**enc).last_hidden_state[:, 0]
        out.append(F.normalize(cls.float(), p=2, dim=1).cpu())
    return torch.cat(out, dim=0)


def load_padded_index(root: Path, cap: int):
    d = root / "results/corpus_index"
    meta_path = d / f"chunks_cap{cap}.json"
    vec_path = d / f"chunks_cap{cap}.f16"
    if not meta_path.is_file() or not vec_path.is_file():
        raise FileNotFoundError(
            f"Missing cap{cap} index: {meta_path} / {vec_path}"
        )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    counts = np.asarray(meta["counts"], dtype=np.int64)
    raw = np.fromfile(vec_path, dtype=np.float16).reshape(-1, DIM)

    if raw.shape[0] != counts.sum():
        raise RuntimeError(f"cap{cap} vector/count mismatch")
    if counts.max() > cap:
        raise RuntimeError(f"count exceeds cap{cap}: {counts.max()}")

    n_docs = len(counts)
    padded = np.zeros((n_docs, cap, DIM), dtype=np.float16)
    mask = np.zeros((n_docs, cap), dtype=np.bool_)

    pos = 0
    for i, c in enumerate(counts):
        padded[i, :c] = raw[pos:pos+c]
        mask[i, :c] = True
        pos += c

    return meta["documents"], padded, mask, counts


@torch.inference_mode()
def retrieve_all(padded_np, mask_np, qvec_cpu, batch_size=4, top_k=100):
    device = torch.device("cuda")
    docs_tensor = torch.from_numpy(padded_np).to(device=device, dtype=torch.float16)
    mask = torch.from_numpy(mask_np).to(device=device)

    result = {"MAX": [], "TOP2_MEAN": [], "TOP3_MEAN": []}

    for s in range(0, len(qvec_cpu), batch_size):
        q = qvec_cpu[s:s+batch_size].to(device=device, dtype=torch.float16)

        # [B, docs, cap]
        sim = torch.einsum("bd,nkd->bnk", q, docs_tensor)
        sim = sim.masked_fill(~mask.unsqueeze(0), -torch.inf)

        max_score = sim.max(dim=2).values

        top2 = torch.topk(sim, k=2, dim=2).values
        valid2 = torch.isfinite(top2)
        top2_score = (
            torch.where(valid2, top2, torch.zeros_like(top2)).sum(dim=2)
            / valid2.sum(dim=2).clamp_min(1)
        )

        top3 = torch.topk(sim, k=3, dim=2).values
        valid3 = torch.isfinite(top3)
        top3_score = (
            torch.where(valid3, top3, torch.zeros_like(top3)).sum(dim=2)
            / valid3.sum(dim=2).clamp_min(1)
        )

        for name, score in (
            ("MAX", max_score),
            ("TOP2_MEAN", top2_score),
            ("TOP3_MEAN", top3_score),
        ):
            vals, inds = torch.topk(score, k=top_k, dim=1)
            inds = inds.cpu().numpy()
            vals = vals.float().cpu().numpy()
            for j in range(len(inds)):
                result[name].append((inds[j], vals[j]))

        print(
            f"  aggregation queries {min(s+batch_size,len(qvec_cpu))}/"
            f"{len(qvec_cpu)} | VRAM={torch.cuda.memory_allocated()/2**30:.2f} GB",
            flush=True,
        )

        del q, sim, max_score, top2, top3, top2_score, top3_score

    del docs_tensor, mask
    torch.cuda.empty_cache()
    return result


def macro_recall(queries, ranking, ids, k):
    return float(np.mean([
        len(set(ranking[q][:k]) & set(queries[q][1])) / len(queries[q][1])
        for q in ids
    ]))


def ceiling(queries, pre, ranking, ids, k):
    return float(np.mean([
        len((set(pre[q]) | set(ranking[q][:k])) & set(queries[q][1]))
        / len(queries[q][1])
        for q in ids
    ]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=Path, required=True)
    ap.add_argument("--cap", type=int, default=DEFAULT_CAP)
    ap.add_argument("--query-batch", type=int, default=4)
    args = ap.parse_args()
    root = args.repo_root.resolve()
    sys.path.insert(0, str(root))

    from tune_expanded_fusion_robust import build_views

    print(f"[1/4] Loading CAL600 and cap{args.cap} padded chunk index...", flush=True)
    queries, blocks, ids, pre_corpus, _ = build_views(root, expanded_depth=20)
    documents, padded, mask, counts = load_padded_index(root, args.cap)
    print(
        f"  docs={len(documents)} chunks={int(counts.sum())} "
        f"padded_bank={padded.nbytes/2**30:.2f} GiB",
        flush=True,
    )

    # Current production candidate ceiling from exact cap32 pool.
    from tune_corpus_cap32_fusion import build_training_cap
    qcur, bcur, idcur, current_pool, _, _ = build_training_cap(
        root,
        32,
        "results/corpus_index/holdout_extended_scores_cap32.pkl",
        depth=20,
        expanded_depth=20,
    )
    if idcur != ids:
        raise RuntimeError("CAL id mismatch vs current production pool")
    current_ceiling = float(np.mean([
        len(set(current_pool[q]) & set(queries[q][1])) / len(queries[q][1])
        for q in ids
    ]))

    print("[2/4] Encoding queries...", flush=True)
    model_path = root / "models/AITeamVN_Vietnamese_Embedding"
    tok = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModel.from_pretrained(
        model_path, dtype=torch.float16, local_files_only=True
    ).eval().to("cuda")

    qvec = encode_cls(
        model, tok, [queries[q][0] for q in ids], batch=16, max_length=512
    )
    del model
    torch.cuda.empty_cache()

    print("[3/4] Retrieving under MAX/TOP2/TOP3 parent aggregation...", flush=True)
    raw = retrieve_all(
        padded, mask, qvec, batch_size=args.query_batch, top_k=100
    )

    ranks = {}
    for name, rows in raw.items():
        ranks[name] = {
            q: [documents[int(i)] for i in inds]
            for q, (inds, vals) in zip(ids, rows)
        }

    summary = {}
    for name in ranks:
        summary[name] = {
            "standalone": {
                f"R@{k}": macro_recall(queries, ranks[name], ids, k)
                for k in (5, 20, 50, 100)
            },
            "ceiling": {
                f"precorpus_plus_top{k}": ceiling(
                    queries, pre_corpus, ranks[name], ids, k
                )
                for k in (10, 20, 50, 100)
            },
            "blocks_R20": {
                b: macro_recall(queries, ranks[name], qids, 20)
                for b, qids in blocks.items()
            },
            "blocks_ceiling20": {
                b: ceiling(queries, pre_corpus, ranks[name], qids, 20)
                for b, qids in blocks.items()
            },
        }

    control = summary["MAX"]
    comparisons = {}
    for name in ("TOP2_MEAN", "TOP3_MEAN"):
        comparisons[name] = {
            "delta_R20": (
                summary[name]["standalone"]["R@20"]
                - control["standalone"]["R@20"]
            ),
            "delta_ceiling20": (
                summary[name]["ceiling"]["precorpus_plus_top20"]
                - control["ceiling"]["precorpus_plus_top20"]
            ),
            "block_R20_deltas": {
                b: summary[name]["blocks_R20"][b] - control["blocks_R20"][b]
                for b in blocks
            },
            "block_ceiling20_deltas": {
                b: (
                    summary[name]["blocks_ceiling20"][b]
                    - control["blocks_ceiling20"][b]
                )
                for b in blocks
            },
            "mean_top20_overlap_with_max": float(np.mean([
                len(set(ranks[name][q][:20]) & set(ranks["MAX"][q][:20])) / 20
                for q in ids
            ])),
        }

    report = {
        "schema": "manual.parent_aggregation_ablation_v2",
        "contract": {
            "cap": args.cap,
            "chunk_pooling": "CLS+L2",
            "formulas": ["MAX", "TOP2_MEAN", "TOP3_MEAN"],
            "public_labels_used": False,
        },
        "summary": summary,
        "comparisons_vs_max": comparisons,
        "current_production_cap32_candidate_ceiling": current_ceiling,
    }

    out = root / f"results/manual/huy_parent_aggregation_ablation_cap{args.cap}_v2"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "REPORT.json"
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("[4/4] RESULT")
    print("=" * 112)
    for name in ("MAX", "TOP2_MEAN", "TOP3_MEAN"):
        s = summary[name]
        print(
            f"{name:<10s} R20={s['standalone']['R@20']:.10f} "
            f"ceiling20={s['ceiling']['precorpus_plus_top20']:.10f}"
        )
        if name != "MAX":
            c = comparisons[name]
            print(
                f"           dR20={c['delta_R20']:+.10f} "
                f"dCeil20={c['delta_ceiling20']:+.10f} "
                f"blockR={c['block_R20_deltas']} "
                f"overlap20={c['mean_top20_overlap_with_max']:.4f}"
            )
    print(f"Current production cap32 candidate ceiling={current_ceiling:.10f}")
    print("Report:", path)
    print("=" * 112)


if __name__ == "__main__":
    main()
