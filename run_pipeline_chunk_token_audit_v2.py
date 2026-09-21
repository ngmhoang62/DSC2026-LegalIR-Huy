#!/usr/bin/env python
"""
HUY PIPELINE MICRO-AUDIT A2 — EXACT CHUNK / TOKEN CONTRACT
==========================================================

Corrected production-path audit.

Fixes versus v1:
- batch tokenization instead of one tokenizer call per text;
- parse each legal document into sections ONCE and cache it;
- audit only the exact top-2 structured sections that production actually
  selects with preselect_legal_sections(..., count=2);
- progress logs immediately after candidate membership is loaded;
- suppresses HuggingFace's model_max_length warning while still measuring the
  true untruncated token lengths (no model forward pass is performed).

No public labels and no model inference.

Run:
  python ../run_pipeline_chunk_token_audit_v2.py \
    --repo-root /d/Study/DSC2026/sota
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer


WINDOW = 220
STEP = 150
CAP = 32
SCAN_LIMIT = 300_000
MAXLEN = 512
SPACE_RE = re.compile(r"\S+", re.UNICODE)


def title_from_link(link: str) -> str:
    if not link:
        return ""
    from urllib.parse import urlparse
    slug = urlparse(link).path.rsplit("/", 1)[-1]
    slug = re.sub(r"\.aspx$", "", slug, flags=re.I)
    slug = re.sub(r"-\d+$", "", slug)
    return slug.replace("-", " ").strip()


def load_doc(path: Path) -> str:
    row = json.loads(path.read_text(encoding="utf-8"))
    return row.get("passage") or title_from_link(row.get("link")) or ""


def shipped_starts(n_words: int, cap: int = CAP):
    starts = list(range(0, max(n_words - 70, 1), STEP))
    possible = list(starts)
    if len(starts) > cap:
        picked = np.linspace(0, len(starts) - 1, cap).round().astype(int)
        starts = [starts[i] for i in dict.fromkeys(picked.tolist())]
    return possible, starts


def interval_stats(n_words: int, starts):
    if n_words <= 0:
        return {
            "covered_fraction": 0.0,
            "max_uncovered_gap": 0,
            "tail_uncovered": 0,
            "covered_words": 0,
        }

    intervals = sorted((s, min(s + WINDOW, n_words)) for s in starts)
    merged = []
    for a, b in intervals:
        if not merged or a > merged[-1][1]:
            merged.append([a, b])
        else:
            merged[-1][1] = max(merged[-1][1], b)

    covered = sum(b - a for a, b in merged)
    gaps = []
    prev = 0
    for a, b in merged:
        if a > prev:
            gaps.append(a - prev)
        prev = max(prev, b)
    if prev < n_words:
        gaps.append(n_words - prev)

    return {
        "covered_fraction": covered / n_words,
        "max_uncovered_gap": max(gaps, default=0),
        "tail_uncovered": max(0, n_words - (merged[-1][1] if merged else 0)),
        "covered_words": covered,
    }


def summarize(values):
    x = np.asarray(values, dtype=np.float64)
    if not len(x):
        return {}
    return {
        "n": int(len(x)),
        "mean": float(np.mean(x)),
        "p50": float(np.percentile(x, 50)),
        "p75": float(np.percentile(x, 75)),
        "p90": float(np.percentile(x, 90)),
        "p95": float(np.percentile(x, 95)),
        "p99": float(np.percentile(x, 99)),
        "max": float(np.max(x)),
    }


def batch_single_lengths(tok, texts, batch_size=512):
    out = []
    for s in range(0, len(texts), batch_size):
        enc = tok(
            texts[s:s + batch_size],
            add_special_tokens=True,
            truncation=False,
            padding=False,
        )
        out.extend(len(x) for x in enc["input_ids"])
    return out


def batch_pair_lengths(tok, qs, ps, batch_size=256):
    assert len(qs) == len(ps)
    out = []
    for s in range(0, len(qs), batch_size):
        enc = tok(
            qs[s:s + batch_size],
            ps[s:s + batch_size],
            add_special_tokens=True,
            truncation=False,
            padding=False,
        )
        out.extend(len(x) for x in enc["input_ids"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=Path, required=True)
    ap.add_argument(
        "--max-candidate-pairs",
        type=int,
        default=0,
        help="0 = all exact CAL candidate pairs; positive value = deterministic sample",
    )
    args = ap.parse_args()
    root = args.repo_root.resolve()
    sys.path.insert(0, str(root))

    # Exact production helpers.
    from benchmark_jina_reranker_holdouts import top_passages
    from tune_corpus_cap32_fusion import build_training_cap
    from src.gemini.huy_d1_legal_section_evidence_v1.legal_section_parser import (
        parse_document_into_sections,
        preselect_legal_sections,
    )

    out = root / "results/manual/huy_pipeline_chunk_token_audit_v2"
    out.mkdir(parents=True, exist_ok=True)

    contexts = (
        root
        / "DSC2026-LegalIR-main/v4_run/public_test_dataset/selected-contexts"
    )
    paths = sorted(contexts.glob("context_*.json"))
    if len(paths) != 8532:
        raise RuntimeError(f"Expected 8532 corpus docs, got {len(paths)}")

    print("[1/5] Auditing shipped cap32 word coverage...", flush=True)

    coverage_rows = []
    retained_chunks = []
    doc_text = {}

    for i, p in enumerate(paths, 1):
        did = p.stem[len("context_"):]
        text = load_doc(p)
        doc_text[did] = text

        scanned = text[:SCAN_LIMIT * 12] if text else ""
        words = SPACE_RE.findall(scanned)
        del words[SCAN_LIMIT:]
        n = len(words)

        possible, starts = shipped_starts(n, CAP)
        st = interval_stats(n, starts)

        coverage_rows.append({
            "doc_id": did,
            "words_scanned": n,
            "possible_windows": len(possible),
            "retained_windows": len(starts),
            **st,
        })
        retained_chunks.extend(" ".join(words[s:s + WINDOW]) for s in starts)

        if i % 1000 == 0:
            print(f"  documents {i}/{len(paths)}", flush=True)

    print("[2/5] Loading local tokenizers...", flush=True)

    ai_tok = AutoTokenizer.from_pretrained(
        root / "models/AITeamVN_Vietnamese_Embedding",
        local_files_only=True,
    )
    jina_tok = AutoTokenizer.from_pretrained(
        root / "models/jina-reranker-v2-base-multilingual",
        trust_remote_code=True,
        fix_mistral_regex=True,
        local_files_only=True,
    )

    # We deliberately request untruncated lengths. The warning at e.g. 1030>1024
    # is irrelevant here because there is NO model forward. Raise tokenizer's
    # warning threshold only; this does not alter token IDs when truncation=False.
    ai_original_max = getattr(ai_tok, "model_max_length", None)
    jina_original_max = getattr(jina_tok, "model_max_length", None)
    ai_tok.model_max_length = 10**9
    jina_tok.model_max_length = 10**9

    print(
        f"[3/5] Batch-tokenizing {len(retained_chunks):,} shipped corpus chunks...",
        flush=True,
    )
    ai_chunk_lengths = batch_single_lengths(ai_tok, retained_chunks, batch_size=512)

    print("[4/5] Loading exact CAL candidate membership...", flush=True)

    queries, blocks, ids, extended, _, _ = build_training_cap(
        root,
        32,
        "results/corpus_index/holdout_extended_scores_cap32.pkl",
        depth=20,
    )

    pairs = [(q, d) for q in ids for d in extended[q]]
    if args.max_candidate_pairs and len(pairs) > args.max_candidate_pairs:
        idx = np.linspace(
            0, len(pairs) - 1, args.max_candidate_pairs
        ).round().astype(int)
        pairs = [pairs[i] for i in idx]

    unique_docs = sorted({d for _, d in pairs})

    print(
        f"  loaded {len(ids)} queries, {len(pairs):,} candidate pairs, "
        f"{len(unique_docs):,} unique candidate documents",
        flush=True,
    )

    print(
        "  parsing each unique candidate document into legal sections ONCE...",
        flush=True,
    )
    sections_cache = {}
    for i, d in enumerate(unique_docs, 1):
        sections_cache[d] = parse_document_into_sections(
            d,
            doc_text.get(d, ""),
            max_chunk_words=220,
            overlap_words=60,
        )
        if i % 500 == 0 or i == len(unique_docs):
            print(f"    parsed docs {i}/{len(unique_docs)}", flush=True)

    jina_pair_lengths = []
    ai_passage_lengths = []
    section_pair_lengths = []
    passages_per_pair = []
    selected_sections_per_pair = []

    # Batch buffers avoid one tokenizer call per passage/section.
    pq, pp = [], []
    aq = []
    sq, sp = [], []

    def flush_passages():
        nonlocal pq, pp, aq
        if not pp:
            return
        jina_pair_lengths.extend(batch_pair_lengths(jina_tok, pq, pp, 256))
        ai_passage_lengths.extend(batch_single_lengths(ai_tok, aq, 512))
        pq, pp, aq = [], [], []

    def flush_sections():
        nonlocal sq, sp
        if not sp:
            return
        section_pair_lengths.extend(batch_pair_lengths(jina_tok, sq, sp, 256))
        sq, sp = [], []

    started = time.perf_counter()

    for n, (q, d) in enumerate(pairs, 1):
        question = queries[q][0]
        text = doc_text.get(d, "")

        passages = top_passages(question, text, count=2)
        passages_per_pair.append(len(passages))
        for p in passages:
            pq.append(question)
            pp.append(p)
            aq.append(p)

        chosen = preselect_legal_sections(
            question,
            sections_cache[d],
            count=2,
        )
        selected_sections_per_pair.append(len(chosen))
        for sec in chosen:
            sq.append(question)
            sp.append(sec.text)

        if len(pp) >= 1024:
            flush_passages()
        if len(sp) >= 1024:
            flush_sections()

        if n % 1000 == 0 or n == len(pairs):
            rate = n / max(time.perf_counter() - started, 1e-9)
            print(
                f"  candidate pairs {n}/{len(pairs)} ({rate:.1f}/s)",
                flush=True,
            )

    flush_passages()
    flush_sections()

    cov_frac = [r["covered_fraction"] for r in coverage_rows]
    max_gaps = [r["max_uncovered_gap"] for r in coverage_rows]
    word_counts = [r["words_scanned"] for r in coverage_rows]
    possible = [r["possible_windows"] for r in coverage_rows]
    retained = [r["retained_windows"] for r in coverage_rows]

    def trunc_stats(lengths):
        arr = np.asarray(lengths, dtype=np.int64)
        return {
            **summarize(lengths),
            "gt512": int(np.sum(arr > MAXLEN)),
            "gt512_rate": float(np.mean(arr > MAXLEN)) if len(arr) else 0.0,
            "gt1024": int(np.sum(arr > 1024)),
            "gt1024_rate": float(np.mean(arr > 1024)) if len(arr) else 0.0,
        }

    report = {
        "schema": "manual.pipeline_chunk_token_audit_v2",
        "contract": {
            "corpus_chunking": {
                "window_words": WINDOW,
                "step_words": STEP,
                "cap": CAP,
                "scan_limit_words": SCAN_LIMIT,
                "aggregation": "max chunk cosine",
            },
            "production_passage_selection": "top_passages(..., count=2)",
            "production_section_selection": (
                "parse_document_into_sections(220,60) + "
                "preselect_legal_sections(..., count=2)"
            ),
            "model_inference_max_length": MAXLEN,
            "candidate_pairs_audited": len(pairs),
            "unique_candidate_docs": len(unique_docs),
            "gold_used_in_statistics": False,
            "model_forward_performed": False,
        },
        "corpus_coverage": {
            "documents": len(coverage_rows),
            "word_counts": summarize(word_counts),
            "possible_windows": summarize(possible),
            "retained_windows": summarize(retained),
            "coverage_fraction": summarize(cov_frac),
            "coverage_fraction_p05": float(np.percentile(cov_frac, 5)),
            "max_uncovered_gap_words": summarize(max_gaps),
            "docs_with_any_uncovered_gap": int(sum(x > 0 for x in max_gaps)),
            "docs_coverage_below_90pct": int(sum(x < .90 for x in cov_frac)),
            "docs_coverage_below_75pct": int(sum(x < .75 for x in cov_frac)),
            "docs_hitting_cap32": int(sum(x > CAP for x in possible)),
        },
        "tokenization": {
            "aiteam_corpus_chunk_tokens": trunc_stats(ai_chunk_lengths),
            "jina_query_top_passage_pair_tokens": trunc_stats(jina_pair_lengths),
            "aiteam_top_passage_tokens": trunc_stats(ai_passage_lengths),
            "jina_query_selected_structured_section_pair_tokens": trunc_stats(
                section_pair_lengths
            ),
            "passages_per_candidate_pair": summarize(passages_per_pair),
            "selected_sections_per_candidate_pair": summarize(
                selected_sections_per_pair
            ),
        },
        "tokenizer_metadata": {
            "aiteam_original_model_max_length": ai_original_max,
            "jina_original_model_max_length": jina_original_max,
            "note": (
                "Audit temporarily raises tokenizer.model_max_length only to "
                "suppress warnings while truncation=False. Production inference "
                "still uses max_length=512."
            ),
        },
        "interpretation_flags": {
            "corpus_sampling_gap_material": bool(
                np.percentile(max_gaps, 95) >= 100
                or np.mean(np.asarray(cov_frac) < .90) >= .05
            ),
            "aiteam_chunk_truncation_material": bool(
                np.mean(np.asarray(ai_chunk_lengths) > MAXLEN) >= .01
            ),
            "jina_passage_pair_truncation_material": bool(
                np.mean(np.asarray(jina_pair_lengths) > MAXLEN) >= .01
            ),
            "selected_section_pair_truncation_material": bool(
                np.mean(np.asarray(section_pair_lengths) > MAXLEN) >= .01
            ),
        },
    }

    (out / "REPORT.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out / "CORPUS_COVERAGE_ROWS.json").write_text(
        json.dumps(coverage_rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("[5/5] RESULT", flush=True)
    print("=" * 108)
    print(
        f"Corpus coverage: mean={np.mean(cov_frac):.4f} "
        f"p05={np.percentile(cov_frac,5):.4f}; "
        f"p95 max-gap={np.percentile(max_gaps,95):.1f} words"
    )
    print(
        f"AITeam corpus chunk >512: "
        f"{report['tokenization']['aiteam_corpus_chunk_tokens']['gt512']}/"
        f"{report['tokenization']['aiteam_corpus_chunk_tokens']['n']} "
        f"({100*report['tokenization']['aiteam_corpus_chunk_tokens']['gt512_rate']:.2f}%)"
    )
    print(
        f"Jina query+top-passage >512: "
        f"{report['tokenization']['jina_query_top_passage_pair_tokens']['gt512']}/"
        f"{report['tokenization']['jina_query_top_passage_pair_tokens']['n']} "
        f"({100*report['tokenization']['jina_query_top_passage_pair_tokens']['gt512_rate']:.2f}%)"
    )
    print(
        f"AITeam top-passage >512: "
        f"{report['tokenization']['aiteam_top_passage_tokens']['gt512']}/"
        f"{report['tokenization']['aiteam_top_passage_tokens']['n']} "
        f"({100*report['tokenization']['aiteam_top_passage_tokens']['gt512_rate']:.2f}%)"
    )
    print(
        f"Jina query+SELECTED legal section >512: "
        f"{report['tokenization']['jina_query_selected_structured_section_pair_tokens']['gt512']}/"
        f"{report['tokenization']['jina_query_selected_structured_section_pair_tokens']['n']} "
        f"({100*report['tokenization']['jina_query_selected_structured_section_pair_tokens']['gt512_rate']:.2f}%)"
    )
    print("Flags:", report["interpretation_flags"])
    print("Report:", out / "REPORT.json")
    print("=" * 108)


if __name__ == "__main__":
    main()
