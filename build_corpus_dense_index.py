"""Embed the whole corpus so candidates can come from meaning, not just words.

Roughly 2% of gold documents never appear in any lexical branch at any depth, so
no amount of reranking can reach them.  This builds a query-independent chunk
index over all 8,532 contexts with the Vietnamese dense encoder.

Documents are long (mean ~8,500 words), so full coverage would need ~486k chunks.
Chunks per document are capped and sampled evenly across the text, which keeps
the build near an hour while still covering about half of an average document and
all of a short one.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from benchmark_aiteamvn_holdouts import encode_cls
from benchmark_jina_reranker_holdouts import SPACE_RE
from run_burst_expanded_fusion_submission import DocumentStore


WINDOW = 220
STEP = 150
CAP = 16


SCAN_LIMIT = 300_000


def document_chunks(text, window=WINDOW, step=STEP, cap=CAP):
    """Evenly spread windows over the document, always including the header.

    Extremely long contexts are scanned only up to SCAN_LIMIT words; materialising
    every word of a million-word document costs more memory than the tail is worth.
    """
    words = SPACE_RE.findall(text[:SCAN_LIMIT * 12] if text else "")
    if not words:
        return [""]
    del words[SCAN_LIMIT:]
    starts = list(range(0, max(len(words) - 70, 1), step))
    if len(starts) > cap:
        picked = np.linspace(0, len(starts) - 1, cap).round().astype(int)
        starts = [starts[i] for i in dict.fromkeys(picked.tolist())]
    return [" ".join(words[s:s + window]) for s in starts]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, default=CAP)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--output", type=Path)
    ap.add_argument("--model-path", default="models/AITeamVN_Vietnamese_Embedding",
                    help="encoder to index with")
    ap.add_argument("--name", default="chunks",
                    help="output basename: <name>_cap<CAP>.f16/.json")
    args = ap.parse_args()
    root = Path(__file__).resolve().parent
    output = args.output or root / "results/corpus_index"
    output.mkdir(parents=True, exist_ok=True)
    vectors_path = output / f"{args.name}_cap{args.cap}.f16"
    meta_path = output / f"{args.name}_cap{args.cap}.json"

    paths = sorted((root / "DSC2026-LegalIR-main/v4_run/public_test_dataset"
                    "/selected-contexts").glob("context_*.json"))
    documents = DocumentStore(paths, cache_size=64)
    doc_ids = [p.stem[len("context_"):] for p in paths]

    done_docs, counts = [], []
    if meta_path.exists() and vectors_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("cap") == args.cap:
            done_docs = meta["documents"]
            counts = meta["counts"]
            print(f"Resuming with {len(done_docs)} documents, "
                  f"{sum(counts)} chunks", flush=True)
    done = set(done_docs)
    remaining = [d for d in doc_ids if d not in done]
    if not remaining:
        print("Index already complete", flush=True)
        return

    model_path = root / args.model_path
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(
        model_path, dtype=torch.float16).eval().to("cuda")
    print(f"Indexing {len(remaining)} documents with {args.model_path} on "
          f"{torch.cuda.get_device_name(0)} -> {vectors_path.name}", flush=True)

    started = time.perf_counter()
    pending_texts = []
    # Vectors stream straight to disk so the build stays memory-flat.
    handle = open(vectors_path, "ab" if done_docs else "wb")

    def flush():
        if not pending_texts:
            return
        vectors = encode_cls(model, tokenizer, pending_texts, args.batch_size,
                             args.max_length)
        handle.write(vectors.astype(np.float16).tobytes())
        pending_texts.clear()

    for i, doc in enumerate(remaining, 1):
        chunks = document_chunks(documents[doc], cap=args.cap)
        pending_texts.extend(chunks)
        done_docs.append(doc)
        counts.append(len(chunks))
        if len(pending_texts) >= 512:
            flush()
        if i % 500 == 0 or i == len(remaining):
            flush()
            handle.flush()
            meta_path.write_text(json.dumps(
                {"cap": args.cap, "window": WINDOW, "step": STEP,
                 "documents": done_docs, "counts": counts},
                ensure_ascii=False), encoding="utf-8")
            total = sum(counts)
            rate = (time.perf_counter() - started) / i
            print(f"Indexed {i}/{len(remaining)} documents, {total} chunks "
                  f"({rate:.2f}s/doc, eta {rate*(len(remaining)-i)/60:.1f}m)",
                  flush=True)
    print(f"Saved {vectors_path}", flush=True)


if __name__ == "__main__":
    main()
