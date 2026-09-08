"""Resume-safe cap-32 full-corpus index for the local AITeamVN fine-tune."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_corpus_dense_index import STEP, WINDOW, document_chunks
from full_corpus_title_retrieval import encode, load_model
from run_burst_expanded_fusion_submission import DocumentStore


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--max-length", type=int, default=512)
    args = ap.parse_args()
    output = ROOT / "cache/sol_high_rl/aiteam_ft_full_corpus"
    output.mkdir(parents=True, exist_ok=True)
    vectors_path = output / f"chunks_cap{args.cap}.f16"
    meta_path = output / f"chunks_cap{args.cap}.json"
    paths = sorted((ROOT / "DSC2026-LegalIR-main/v4_run/public_test_dataset/selected-contexts").glob("context_*.json"))
    documents = DocumentStore(paths, cache_size=64)
    doc_ids = [p.stem[len("context_") :] for p in paths]

    done_docs, counts = [], []
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("contract") != "aiteam_ft_cls_l2_cap32_v1" or meta.get("cap") != args.cap:
            raise RuntimeError("Existing isolated index has a different contract")
        done_docs, counts = meta["documents"], meta["counts"]
    expected_bytes = sum(counts) * 1024 * 2
    if vectors_path.exists():
        actual = vectors_path.stat().st_size
        if actual < expected_bytes:
            raise RuntimeError(f"Vector file shorter than manifest: {actual} < {expected_bytes}")
        if actual > expected_bytes:
            with vectors_path.open("r+b") as handle:
                handle.truncate(expected_bytes)
            print(f"truncated uncommitted vector tail {actual-expected_bytes} bytes", flush=True)
    elif expected_bytes:
        raise RuntimeError("Manifest exists but vector file is missing")

    remaining = [d for d in doc_ids if d not in set(done_docs)]
    print(f"resume={len(done_docs)}/{len(doc_ids)} remaining={len(remaining)} chunks={sum(counts)}", flush=True)
    if not remaining:
        return
    model, tokenizer = load_model()
    print(f"AITeamVN-FT ready; writing {vectors_path}", flush=True)
    started = time.perf_counter()
    handle = vectors_path.open("ab")
    pending, pending_docs, pending_counts = [], [], []

    def flush_commit():
        nonlocal pending
        if not pending_docs:
            return
        if pending:
            vec = encode(model, tokenizer, pending, batch_size=args.batch_size, max_length=args.max_length)
            handle.write(vec.astype(np.float16).tobytes())
            pending = []
        handle.flush()
        os.fsync(handle.fileno())
        done_docs.extend(pending_docs)
        counts.extend(pending_counts)
        pending_docs.clear()
        pending_counts.clear()
        tmp = meta_path.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(
                {
                    "contract": "aiteam_ft_cls_l2_cap32_v1",
                    "model": "fine_tune/AITeamVN_Vietnamese_Embedding",
                    "pooling": "CLS+L2",
                    "prefixes": "none",
                    "max_length": args.max_length,
                    "cap": args.cap,
                    "window": WINDOW,
                    "step": STEP,
                    "documents": done_docs,
                    "counts": counts,
                    "embedding_dim": 1024,
                    "dtype": "float16",
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        tmp.replace(meta_path)

    for i, docid in enumerate(remaining, 1):
        chunks = document_chunks(documents[docid], cap=args.cap)
        pending.extend(chunks)
        pending_docs.append(docid)
        pending_counts.append(len(chunks))
        if len(pending) >= 512 or len(pending_docs) >= 100:
            flush_commit()
        if i % 500 == 0 or i == len(remaining):
            flush_commit()
            elapsed = time.perf_counter() - started
            rate = elapsed / i
            print(
                f"indexed {i}/{len(remaining)} this run; total_docs={len(done_docs)}, "
                f"chunks={sum(counts)}, eta={rate*(len(remaining)-i)/60:.1f}m",
                flush=True,
            )
    handle.close()
    print(f"complete bytes={vectors_path.stat().st_size}", flush=True)


if __name__ == "__main__":
    main()
