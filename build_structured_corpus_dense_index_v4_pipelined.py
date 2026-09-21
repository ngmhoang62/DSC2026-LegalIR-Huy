#!/usr/bin/env python
"""
STRUCTURE-AWARE FULL-CORPUS AITeamVN INDEX V3 — PARALLEL PARSER
===============================================================

Same scientific policy/output as V2:
  parse_document_into_sections(max_chunk_words=220, overlap_words=60)
  keep all sections if <=cap; otherwise evenly sample section indices
  same AITeamVN encoder, CLS+L2, max_length=512

Only runtime changes:
  - CPU document parsing is parallelized with ProcessPoolExecutor;
  - CPU parsing and GPU encoding overlap;
  - orphan trailing vectors from an interrupted V2/V3 run are detected and
    truncated back to the last metadata checkpoint before resuming.

Safe to resume the existing:
  results/corpus_index/structured_chunks_cap32.{json,f16}
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

WINDOW = 220
STEP = 150
CAP = 32
SCAN_LIMIT = 300_000
DIM = 1024
SPACE_RE = re.compile(r"\S+", re.UNICODE)

_PARSER = None


def raw_fallback(text: str, cap: int):
    words = SPACE_RE.findall(text[:SCAN_LIMIT * 12] if text else "")
    if not words:
        return [""]
    del words[SCAN_LIMIT:]
    starts = list(range(0, max(len(words) - 70, 1), STEP))
    if len(starts) > cap:
        picked = np.linspace(0, len(starts) - 1, cap).round().astype(int)
        starts = [starts[i] for i in dict.fromkeys(picked.tolist())]
    return [" ".join(words[s:s + WINDOW]) for s in starts]


def _worker_init(repo_root: str):
    global _PARSER
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from src.gemini.huy_d1_legal_section_evidence_v1.legal_section_parser import (
        parse_document_into_sections,
    )
    _PARSER = parse_document_into_sections


def _parse_one(task):
    doc_id, path_str, cap = task
    path = Path(path_str)
    row = json.loads(path.read_text(encoding="utf-8"))
    text = row.get("passage") or ""

    sections = _PARSER(
        doc_id,
        text,
        max_chunk_words=220,
        overlap_words=60,
    )
    if not sections:
        chunks = raw_fallback(text, cap)
        audit = {
            "mode": "RAW_FALLBACK_EMPTY",
            "types": [],
        }
    else:
        if len(sections) <= cap:
            chosen = list(sections)
        else:
            picked = np.linspace(0, len(sections) - 1, cap).round().astype(int)
            idx = list(dict.fromkeys(picked.tolist()))
            chosen = [sections[i] for i in idx]
        chunks = [s.text for s in chosen]
        audit = {
            "mode": "STRUCTURED",
            "types": [s.section_type for s in chosen],
        }

    return doc_id, chunks, audit


@torch.inference_mode()
def encode_cls_pipelined(model, tokenizer, texts, batch_size=96, max_length=512):
    """Same CLS+L2 encoder contract as benchmark_aiteamvn_holdouts.encode_cls,
    but overlaps CPU tokenization of batch N+1 with GPU forward of batch N.

    Scientific semantics are unchanged:
      same tokenizer args, same text order, same max_length/truncation/padding,
      same model, same CLS pooling, same L2 normalization.
    """
    if not texts:
        return np.empty((0, DIM), dtype=np.float32)

    batches = [
        texts[s:s + batch_size]
        for s in range(0, len(texts), batch_size)
    ]

    def tokenize(batch_texts):
        return tokenizer(
            batch_texts,
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

    vectors = []
    with ThreadPoolExecutor(max_workers=1) as ex:
        future = ex.submit(tokenize, batches[0])

        for i in range(len(batches)):
            encoded = future.result()

            # Launch tokenization of the next batch BEFORE the current GPU
            # forward, so CPU tokenization overlaps CUDA compute.
            if i + 1 < len(batches):
                future = ex.submit(tokenize, batches[i + 1])

            batch = {
                k: v.to("cuda", non_blocking=True)
                for k, v in encoded.items()
            }
            cls = model(**batch).last_hidden_state[:, 0]
            vec = F.normalize(cls.float(), p=2, dim=1)
            vectors.append(vec.cpu().numpy())

    return np.vstack(vectors)



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=Path, required=True)
    ap.add_argument("--cap", type=int, default=CAP)
    ap.add_argument("--batch-size", type=int, default=96)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--parse-workers", type=int, default=8)
    ap.add_argument("--checkpoint-docs", type=int, default=250)
    ap.add_argument("--gpu-flush-chunks", type=int, default=2048)
    args = ap.parse_args()

    root = args.repo_root.resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


    out = root / "results/corpus_index"
    out.mkdir(parents=True, exist_ok=True)
    vec_path = out / f"structured_chunks_cap{args.cap}.f16"
    meta_path = out / f"structured_chunks_cap{args.cap}.json"

    paths = sorted(
        (
            root
            / "DSC2026-LegalIR-main/v4_run/public_test_dataset/selected-contexts"
        ).glob("context_*.json")
    )
    id_to_path = {
        p.stem[len("context_"):]: p for p in paths
    }
    doc_ids = list(id_to_path)

    done_docs, counts = [], []
    mode_counts = Counter()
    chosen_type_counts = Counter()

    if meta_path.exists() and vec_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if (
            meta.get("cap") == args.cap
            and meta.get("policy") == "LEGAL_SECTION_INDEX_EVEN_V1"
        ):
            done_docs = list(meta["documents"])
            counts = list(meta["counts"])
            mode_counts.update(meta.get("mode_counts", {}))
            chosen_type_counts.update(meta.get("chosen_type_counts", {}))

            expected_bytes = int(sum(counts)) * DIM * 2
            actual_bytes = vec_path.stat().st_size
            if actual_bytes < expected_bytes:
                raise RuntimeError(
                    f"Vector file shorter than metadata: "
                    f"{actual_bytes} < {expected_bytes}"
                )
            if actual_bytes > expected_bytes:
                orphan_chunks = (actual_bytes - expected_bytes) // (DIM * 2)
                print(
                    f"Repairing interrupted run: truncating "
                    f"{orphan_chunks} orphan trailing vectors "
                    f"back to metadata checkpoint.",
                    flush=True,
                )
                with vec_path.open("r+b") as f:
                    f.truncate(expected_bytes)

            print(
                f"Resuming with {len(done_docs)} docs / "
                f"{sum(counts)} chunks",
                flush=True,
            )

    done = set(done_docs)
    remaining = [d for d in doc_ids if d not in done]
    if not remaining:
        print("Structured index already complete", flush=True)
        return

    model_path = root / "models/AITeamVN_Vietnamese_Embedding"
    tok = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModel.from_pretrained(
        model_path,
        dtype=torch.float16,
        local_files_only=True,
    ).eval().to("cuda")

    print(
        f"Building structure-aware cap{args.cap}: remaining={len(remaining)} | "
        f"parse_workers={args.parse_workers} | batch={args.batch_size} | "
        f"GPU={torch.cuda.get_device_name(0)}",
        flush=True,
    )

    handle = open(vec_path, "ab" if done_docs else "wb")
    pending = []
    started = time.perf_counter()

    def flush_gpu():
        if not pending:
            return
        vec = encode_cls_pipelined(
            model,
            tok,
            pending,
            args.batch_size,
            args.max_length,
        )
        handle.write(vec.astype(np.float16).tobytes())
        pending.clear()

    def write_meta():
        handle.flush()
        meta = {
            "policy": "LEGAL_SECTION_INDEX_EVEN_V1",
            "cap": args.cap,
            "parser_max_chunk_words": 220,
            "parser_overlap_words": 60,
            "selection": (
                "all parsed sections if <=cap; otherwise np.linspace "
                "over ordered section indices"
            ),
            "documents": done_docs,
            "counts": counts,
            "mode_counts": dict(mode_counts),
            "chosen_type_counts": dict(chosen_type_counts),
            "runtime_backend": {
                "version": "V4_PARALLEL_PARSER_PIPELINED_TOKENIZER",
                "parse_workers": args.parse_workers,
                "batch_size": args.batch_size,
            },
        }
        meta_path.write_text(
            json.dumps(meta, ensure_ascii=False),
            encoding="utf-8",
        )

    tasks = [
        (d, str(id_to_path[d]), args.cap)
        for d in remaining
    ]

    # ProcessPool map preserves input order, so document order/vector ownership
    # remains exactly deterministic while workers parse ahead of GPU encoding.
    try:
        with ProcessPoolExecutor(
            max_workers=args.parse_workers,
            initializer=_worker_init,
            initargs=(str(root),),
        ) as ex:
            iterator = ex.map(_parse_one, tasks, chunksize=4)

            for i, (d, chunks, audit) in enumerate(iterator, 1):
                if d != remaining[i - 1]:
                    raise RuntimeError(
                        f"Parser order drift at {i}: {d} != {remaining[i-1]}"
                    )

                pending.extend(chunks)
                done_docs.append(d)
                counts.append(len(chunks))
                mode_counts[audit["mode"]] += 1
                chosen_type_counts.update(audit.get("types", []))

                if len(pending) >= args.gpu_flush_chunks:
                    flush_gpu()

                if i % args.checkpoint_docs == 0 or i == len(remaining):
                    flush_gpu()
                    write_meta()
                    elapsed = time.perf_counter() - started
                    rate = elapsed / i
                    eta = rate * (len(remaining) - i) / 60
                    print(
                        f"Indexed {i}/{len(remaining)} new docs | "
                        f"total_docs={len(done_docs)} | "
                        f"chunks={sum(counts)} | "
                        f"{rate:.3f}s/doc | eta={eta:.1f}m | "
                        f"VRAM={torch.cuda.memory_allocated()/2**30:.2f}GB",
                        flush=True,
                    )
    except KeyboardInterrupt:
        # Do NOT flush pending vectors on interrupt: metadata remains the source
        # of truth. Any already-written complete GPU flush since the last
        # metadata checkpoint will be automatically truncated on next resume.
        print(
            "\nInterrupted. Safe to rerun: next start will reconcile the vector "
            "file to the last metadata checkpoint.",
            flush=True,
        )
        raise
    finally:
        handle.close()

    print("Saved:", vec_path)
    print("Metadata:", meta_path)


if __name__ == "__main__":
    main()
