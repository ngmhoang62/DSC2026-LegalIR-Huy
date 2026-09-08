"""Score the CV candidate pool with any drop-in bi-encoder.

Same convention as the production `dense` channel and score_cv_vnlegal_lal.py:
CLS pooling, 2 lexical-density passages per document, document score = max over
its passages. Generic in the model path so a newly supplied checkpoint can be
evaluated without writing a new script each time.
"""

from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

from benchmark_aiteamvn_holdouts import encode_cls
from benchmark_jina_reranker_holdouts import top_passages
from run_burst_expanded_fusion_submission import DocumentStore
from tune_corpus_cap32_fusion import build_training_cap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--passages", type=int, default=2)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    root = Path(__file__).resolve().parent
    docs = DocumentStore(sorted(
        (root / "DSC2026-LegalIR-main/v4_run/public_test_dataset/selected-contexts")
        .glob("context_*.json")))
    queries, blocks, all_ids, extended, local, _ = build_training_cap(
        root, 32, "results/corpus_index/holdout_extended_scores_cap32.pkl", depth=20)

    out_path = root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    saved = pickle.loads(out_path.read_bytes()) if out_path.exists() else {}
    todo = [q for q in all_ids if any(d not in saved.get(q, {}) for d in extended[q])]
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(saved)} cached, scoring {len(todo)} queries with {args.model_path}",
          flush=True)
    if not todo:
        return

    path = root / args.model_path
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModel.from_pretrained(path, dtype=torch.float16,
                                      low_cpu_mem_usage=True).eval().to("cuda")
    print(f"ready on {torch.cuda.get_device_name(0)}", flush=True)

    started = time.time()
    for i, q in enumerate(todo, 1):
        text = queries[q][0]
        qvec = encode_cls(model, tokenizer, [text], 1, args.max_length)[0]
        owners, passages = [], []
        for d in extended[q]:
            for p in top_passages(text, docs[d], count=args.passages):
                owners.append(d)
                passages.append(p)
        ds = dict(saved.get(q, {}))
        if passages:
            pvec = encode_cls(model, tokenizer, passages, args.batch_size,
                              args.max_length)
            for d, s in zip(owners, pvec @ qvec):
                ds[d] = max(ds.get(d, -1e9), float(s))
        saved[q] = ds
        if i % 25 == 0:
            out_path.write_bytes(pickle.dumps(saved, protocol=5))
            rate = (time.time() - started) / i
            print(f"  {i}/{len(todo)} {rate:.2f}s/q eta {rate*(len(todo)-i)/60:.1f}m",
                  flush=True)
    out_path.write_bytes(pickle.dumps(saved, protocol=5))
    print(f"Saved {out_path} ({len(saved)} queries)", flush=True)


if __name__ == "__main__":
    main()
