"""Score the CV pool with vietlegal-harrier-0.6b + the supplied LoRA adapter.

The base is a Qwen3Model (1024x28) packaged as a SentenceTransformer whose own
config specifies LAST-TOKEN pooling plus L2 normalisation -- not the CLS pooling
every other channel in this pipeline uses, so the SentenceTransformer stack is
used directly to get the pooling right. The LoRA adapter (r=16,
task_type=FEATURE_EXTRACTION) is applied to the underlying transformer.

Same convention as the other channels otherwise: 2 lexical-density passages per
document, document score = max over its passages.
"""

from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

import numpy as np
import torch

from benchmark_jina_reranker_holdouts import top_passages
from run_burst_expanded_fusion_submission import DocumentStore
from tune_corpus_cap32_fusion import build_training_cap

BASE = "models/vietlegal-harrier-0.6b"
ADAPTER = ("models/from_drive/vietlegal_finetuned_results_HNSW/"
           "vietlegal_finetuned/best_adapter")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--adapter", default=ADAPTER)
    ap.add_argument("--no-adapter", action="store_true",
                    help="score with the base model only, as a control")
    ap.add_argument("--out", default="results/from_drive/harrier_ft_cv.pkl")
    ap.add_argument("--passages", type=int, default=2)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=16)
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
    print(f"{len(saved)} cached, scoring {len(todo)} queries", flush=True)
    if not todo:
        return

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(str(root / args.base), device="cuda",
                                model_kwargs={"dtype": torch.float16})
    if not args.no_adapter:
        from peft import PeftModel
        inner = model[0].auto_model
        model[0].auto_model = PeftModel.from_pretrained(
            inner, str(root / args.adapter), is_trainable=False)
        print(f"LoRA adapter applied from {args.adapter}", flush=True)
    model.max_seq_length = args.max_length
    model.eval()
    pooling = model[1].get_pooling_mode_str() if hasattr(model[1], "get_pooling_mode_str") else "?"
    print(f"ready on {torch.cuda.get_device_name(0)}; pooling={pooling}", flush=True)

    started = time.time()
    for i, q in enumerate(todo, 1):
        text = queries[q][0]
        owners, passages = [], []
        for d in extended[q]:
            for p in top_passages(text, docs[d], count=args.passages):
                owners.append(d)
                passages.append(p)
        ds = dict(saved.get(q, {}))
        if passages:
            with torch.inference_mode():
                qv = model.encode([text], batch_size=1, convert_to_numpy=True,
                                  show_progress_bar=False)[0]
                pv = model.encode(passages, batch_size=args.batch_size,
                                  convert_to_numpy=True, show_progress_bar=False)
            for d, s in zip(owners, pv @ qv):
                ds[d] = max(ds.get(d, -1e9), float(s))
        saved[q] = ds
        if i % 25 == 0:
            out_path.write_bytes(pickle.dumps(saved, protocol=5))
            rate = (time.time() - started) / i
            print(f"  {i}/{len(todo)} {rate:.2f}s/q "
                  f"eta {rate * (len(todo) - i) / 60:.1f}m", flush=True)
    out_path.write_bytes(pickle.dumps(saved, protocol=5))
    print(f"Saved {out_path} ({len(saved)} queries)", flush=True)


if __name__ == "__main__":
    main()
