#!/usr/bin/env python
"""
STRUCTURE-AWARE FULL-CORPUS AITeamVN INDEX V1
==============================================

Query-independent replacement for the current evenly-spaced raw-word cap32 index.

Current production:
  raw 220-word windows, stride 150;
  if >32 windows, sample 32 starts evenly over raw text.

Audit findings:
  cap32 mean word coverage 82.34%, p05 25.38%, p95 uncovered gap 680 words.
  cap48 improves standalone dense R@20 but NOT union candidate ceiling.

Hypothesis:
  the problem is information placement, not simply chunk count.

This index keeps the SAME cap32 and SAME encoder, but aligns chunks to
deterministic Vietnamese legal structure:
  parse_document_into_sections(max_chunk_words=220, overlap_words=60)
  if <=32 parsed sections: keep all
  if >32: evenly sample 32 SECTION INDICES (always includes first/last)
  empty/unparseable docs fall back to current raw-window policy.

No query labels, no CAL/public data influence chunk selection.

Output:
  results/corpus_index/structured_chunks_cap32.f16
  results/corpus_index/structured_chunks_cap32.json
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from benchmark_aiteamvn_holdouts import encode_cls
from benchmark_jina_reranker_holdouts import SPACE_RE
from run_burst_expanded_fusion_submission import DocumentStore
from src.gemini.huy_d1_legal_section_evidence_v1.legal_section_parser import (
    parse_document_into_sections,
)

WINDOW = 220
STEP = 150
CAP = 32
SCAN_LIMIT = 300_000


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


def structure_chunks(doc_id: str, text: str, cap: int):
    sections = parse_document_into_sections(
        doc_id,
        text,
        max_chunk_words=220,
        overlap_words=60,
    )
    if not sections:
        return raw_fallback(text, cap), {
            "mode": "RAW_FALLBACK_EMPTY",
            "parsed_sections": 0,
            "kept_sections": 1,
        }

    # Parser already creates <=220-word bodies and contextualized legal headings.
    # Sampling is by semantic/legal-section index, not raw-word start.
    if len(sections) <= cap:
        chosen = list(sections)
    else:
        picked = np.linspace(0, len(sections) - 1, cap).round().astype(int)
        idx = list(dict.fromkeys(picked.tolist()))
        chosen = [sections[i] for i in idx]

    return [s.text for s in chosen], {
        "mode": "STRUCTURED",
        "parsed_sections": len(sections),
        "kept_sections": len(chosen),
        "types": [s.section_type for s in chosen],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=Path, required=True)
    ap.add_argument("--cap", type=int, default=CAP)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-length", type=int, default=512)
    args = ap.parse_args()

    root = args.repo_root.resolve()
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
    docs = DocumentStore(paths, cache_size=64)
    doc_ids = [p.stem[len("context_"):] for p in paths]

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
            print(
                f"Resuming with {len(done_docs)} docs / {sum(counts)} chunks",
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
        f"Building structure-aware cap{args.cap}: remaining={len(remaining)} "
        f"GPU={torch.cuda.get_device_name(0)}",
        flush=True,
    )

    handle = open(vec_path, "ab" if done_docs else "wb")
    pending = []
    started = time.perf_counter()

    def flush():
        if not pending:
            return
        vec = encode_cls(
            model,
            tok,
            pending,
            args.batch_size,
            args.max_length,
        )
        handle.write(vec.astype(np.float16).tobytes())
        pending.clear()

    try:
        for i, d in enumerate(remaining, 1):
            chunks, audit = structure_chunks(d, docs[d], args.cap)
            pending.extend(chunks)
            done_docs.append(d)
            counts.append(len(chunks))
            mode_counts[audit["mode"]] += 1
            chosen_type_counts.update(audit.get("types", []))

            if len(pending) >= 512:
                flush()

            if i % 250 == 0 or i == len(remaining):
                flush()
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
                }
                meta_path.write_text(
                    json.dumps(meta, ensure_ascii=False),
                    encoding="utf-8",
                )
                rate = (time.perf_counter() - started) / i
                print(
                    f"Indexed {i}/{len(remaining)} docs | "
                    f"chunks={sum(counts)} | {rate:.2f}s/doc | "
                    f"eta={rate*(len(remaining)-i)/60:.1f}m",
                    flush=True,
                )
    finally:
        flush()
        handle.close()

    print("Saved:", vec_path)
    print("Metadata:", meta_path)


if __name__ == "__main__":
    main()
