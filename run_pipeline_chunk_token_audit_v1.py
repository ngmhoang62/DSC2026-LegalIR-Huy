#!/usr/bin/env python
"""
HUY PIPELINE MICRO-AUDIT A — CHUNK / TOKEN CONTRACT V1
======================================================

Pure diagnostic. No model inference and no public labels.

Audits:
1) Current full-corpus dense cap32 chunk coverage:
   - actual document word counts under the shipped whitespace tokenizer
   - number of possible 220/150 windows vs retained cap32 windows
   - covered-word fraction
   - maximum uncovered interior gap
   - tail coverage
2) Tokenizer mismatch:
   - AITeam embedding tokenizer token count for shipped 220-word corpus chunks
   - Jina pair-token count (query, top_passage) for D1 candidate passages
   - AITeam token count for the same top_passages
   - structured legal-section Jina pair-token count
3) Reports how often max_length=512 necessarily truncates.

No gold labels are inspected by the statistics below. build_training_cap() is used
only to recover CAL query text + exact candidate membership; answer/gold fields
are not referenced after loading.

Run:
  python ../run_pipeline_chunk_token_audit_v1.py \
    --repo-root /d/Study/DSC2026/sota
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer


WINDOW = 220
STEP = 150
CAP = 32
SCAN_LIMIT = 300_000
MAXLEN = 512
SPACE_RE = re.compile(r"\S+", re.UNICODE)

STOPWORDS = {
    "bị", "các", "có", "của", "cho", "được", "để", "đến", "đối", "gì",
    "hay", "khi", "không", "là", "làm", "một", "nào", "những", "như",
    "phải", "ra", "sẽ", "theo", "thì", "thế", "trong", "trên", "từ",
    "và", "về", "với", "việc", "bao", "nhiêu", "người", "quy", "định",
}

TOKEN_RE = re.compile(r"\w+", re.UNICODE)


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
    intervals = [(s, min(s + WINDOW, n_words)) for s in starts]
    intervals.sort()
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


def tok_len_single(tok, text: str) -> int:
    return len(tok(text, add_special_tokens=True, truncation=False)["input_ids"])


def tok_len_pair(tok, q: str, p: str) -> int:
    return len(
        tok(
            q,
            p,
            add_special_tokens=True,
            truncation=False,
        )["input_ids"]
    )


def tokens(text):
    return TOKEN_RE.findall((text or "").lower())


def top_passages(question, text, count=2, window=220, overlap=70):
    """Exact shipped lexical preselection logic, copied locally."""
    words = SPACE_RE.findall(text or "")
    if len(words) <= window + 80:
        return [" ".join(words)]
    query_tokens = tokens(question)
    content = {t for t in query_tokens if len(t) >= 3 and t not in STOPWORDS}
    numbers = {t for t in query_tokens if any(c.isdigit() for c in t)}
    bigrams = {" ".join(query_tokens[i:i+2]) for i in range(len(query_tokens)-1)}
    header = " ".join(words[:70])
    scored = []
    step = window - overlap
    for start in range(0, len(words), step):
        end = min(start + window, len(words))
        part_words = words[start:end]
        part = " ".join(part_words)
        normalized = tokens(part)
        token_set = set(normalized)
        norm_text = " ".join(normalized)
        coverage = sum(
            1.0 + .20 * min(normalized.count(t), 3)
            for t in content if t in token_set
        )
        numeric = 3.0 * sum(t in token_set for t in numbers)
        phrase = 1.8 * sum(p in norm_text for p in bigrams)
        density = (coverage + numeric + phrase) / math.sqrt(max(len(normalized), 1))
        scored.append((density, coverage + numeric + phrase, -start, part))
        if end == len(words):
            break
    scored.sort(reverse=True)
    passages = []
    for _, _, neg_start, part in scored:
        candidate = (
            part if -neg_start < 70
            else header + "\n[ĐOẠN PHÙ HỢP]\n" + part
        )
        if candidate not in passages:
            passages.append(candidate)
        if len(passages) >= count:
            break
    return passages


# Legal section parser copied from frozen parser to avoid importing experiment common.py.
BOUNDARY_RE = re.compile(
    r'(?:\r?\n)+(?='
    r'^\s*(?:Điều\s+\d+[\.\:\s\–\-]|Chương\s+[IVXLCDM\d]+|Mục\s+\d+|Phụ\s+lục\s+[IVXLCDM\d]+)'
    r')',
    re.IGNORECASE | re.MULTILINE,
)
HEADING_RE = re.compile(
    r'^\s*(Điều\s+\d+[\.\:\s\–\-][^\n]*|Chương\s+[IVXLCDM\d]+[^\n]*|Mục\s+\d+[^\n]*|Phụ\s+lục\s+[IVXLCDM\d]+[^\n]*)',
    re.IGNORECASE,
)


def parse_sections(raw_text: str, max_chunk_words=220, overlap_words=60):
    if not raw_text or not raw_text.strip():
        return []
    words = SPACE_RE.findall(raw_text)
    if len(words) <= max_chunk_words + 50:
        return [raw_text.strip()]
    doc_header = " ".join(words[:60])
    parts = [p.strip() for p in BOUNDARY_RE.split(raw_text) if p.strip()]
    if len(parts) <= 1:
        out = []
        step = max_chunk_words - overlap_words
        for i in range(0, len(words), step):
            sub = " ".join(words[i:i+max_chunk_words])
            out.append(
                f"{doc_header}\n[MỤC]\n{sub}" if i >= 60 else sub
            )
            if i + max_chunk_words >= len(words):
                break
        return out

    out = []
    preamble_words = parts[0].split()
    if len(preamble_words) > 30 and len(parts) > 1:
        out.append(
            f"{doc_header}\n[PHẦN MỞ ĐẦU / CĂN CỨ]\n{parts[0]}"
        )

    for part in parts[1:]:
        p_words = part.split()
        if not p_words:
            continue
        first_line = part.split("\n")[0].strip()
        m = HEADING_RE.match(part)
        heading = m.group(1).strip() if m else first_line[:120]
        if len(p_words) <= max_chunk_words:
            out.append(
                f"{doc_header}\n[ĐIỀU KHOẢN: {heading}]\n{part}"
            )
        else:
            step = max_chunk_words - overlap_words
            for i in range(0, len(p_words), step):
                sub = " ".join(p_words[i:i+max_chunk_words])
                sub_heading = (
                    f"{heading} (Đoạn {i // step + 1})" if i > 0 else heading
                )
                out.append(
                    f"{doc_header}\n[ĐIỀU KHOẢN: {sub_heading}]\n{sub}"
                )
                if i + max_chunk_words >= len(p_words):
                    break
    return out


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=Path, required=True)
    ap.add_argument(
        "--max-candidate-pairs",
        type=int,
        default=0,
        help="0 = audit all CAL D1 candidate pairs",
    )
    args = ap.parse_args()
    root = args.repo_root.resolve()

    out = root / "results/manual/huy_pipeline_chunk_token_audit_v1"
    out.mkdir(parents=True, exist_ok=True)

    contexts = (
        root
        / "DSC2026-LegalIR-main/v4_run/public_test_dataset/selected-contexts"
    )
    paths = sorted(contexts.glob("context_*.json"))
    if len(paths) != 8532:
        raise RuntimeError(f"Expected 8532 corpus docs, got {len(paths)}")

    print("[1/5] Auditing current cap32 full-corpus word coverage...", flush=True)
    coverage_rows = []
    retained_chunk_texts = []
    doc_text = {}

    for i, p in enumerate(paths, 1):
        d = p.stem[len("context_"):]
        text = load_doc(p)
        doc_text[d] = text
        # Match shipped builder's pre-scan approximation and hard word cap.
        scanned = text[:SCAN_LIMIT * 12] if text else ""
        words = SPACE_RE.findall(scanned)
        del words[SCAN_LIMIT:]
        n = len(words)
        possible, starts = shipped_starts(n, CAP)
        st = interval_stats(n, starts)
        row = {
            "doc_id": d,
            "words_scanned": n,
            "possible_windows": len(possible),
            "retained_windows": len(starts),
            **st,
        }
        coverage_rows.append(row)
        retained_chunk_texts.extend(
            [" ".join(words[s:s+WINDOW]) for s in starts]
        )
        if i % 1000 == 0:
            print(f"  docs {i}/{len(paths)}", flush=True)

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

    print(
        f"[3/5] Tokenizing {len(retained_chunk_texts):,} shipped corpus chunks...",
        flush=True,
    )
    ai_chunk_lengths = []
    for i, t in enumerate(retained_chunk_texts, 1):
        ai_chunk_lengths.append(tok_len_single(ai_tok, t))
        if i % 25000 == 0:
            print(f"  corpus chunks {i}/{len(retained_chunk_texts)}", flush=True)

    # Candidate-pair audit. Uses no gold statistics.
    print("[4/5] Loading exact CAL candidate membership (gold not inspected)...", flush=True)
    sys.path.insert(0, str(root))
    from tune_corpus_cap32_fusion import build_training_cap

    queries, blocks, ids, extended, _, _ = build_training_cap(
        root,
        32,
        "results/corpus_index/holdout_extended_scores_cap32.pkl",
        depth=20,
    )

    pairs = [(q, d) for q in ids for d in extended[q]]
    if args.max_candidate_pairs and len(pairs) > args.max_candidate_pairs:
        # Deterministic evenly-spaced diagnostic sample.
        idx = np.linspace(
            0, len(pairs) - 1, args.max_candidate_pairs
        ).round().astype(int)
        pairs = [pairs[i] for i in idx]

    jina_pair_lengths = []
    ai_passage_lengths = []
    section_pair_lengths = []
    passages_per_pair = []
    sections_per_pair = []

    started = time.perf_counter()
    for n, (q, d) in enumerate(pairs, 1):
        question = queries[q][0]
        text = doc_text.get(d, "")

        passages = top_passages(question, text, count=2)
        passages_per_pair.append(len(passages))
        for passage in passages:
            jina_pair_lengths.append(tok_len_pair(jina_tok, question, passage))
            ai_passage_lengths.append(tok_len_single(ai_tok, passage))

        sections = parse_sections(text)
        # Audit ALL constructed sections' token lengths because lexical selection
        # happens before Jina and truncation risk is a section construction issue.
        sections_per_pair.append(len(sections))
        for sec in sections:
            section_pair_lengths.append(tok_len_pair(jina_tok, question, sec))

        if n % 2000 == 0:
            rate = n / max(time.perf_counter() - started, 1e-9)
            print(f"  pairs {n}/{len(pairs)} ({rate:.1f}/s)", flush=True)

    cov_frac = [r["covered_fraction"] for r in coverage_rows]
    max_gaps = [r["max_uncovered_gap"] for r in coverage_rows]
    word_counts = [r["words_scanned"] for r in coverage_rows]
    possible = [r["possible_windows"] for r in coverage_rows]
    retained = [r["retained_windows"] for r in coverage_rows]

    report = {
        "schema": "manual.pipeline_chunk_token_audit_v1",
        "contract": {
            "corpus_chunking": {
                "window_words": WINDOW,
                "step_words": STEP,
                "cap": CAP,
                "scan_limit_words": SCAN_LIMIT,
                "aggregation": "max chunk cosine downstream",
            },
            "max_length": MAXLEN,
            "candidate_pairs_audited": len(pairs),
            "gold_used_in_statistics": False,
        },
        "corpus_coverage": {
            "documents": len(coverage_rows),
            "word_counts": summarize(word_counts),
            "possible_windows": summarize(possible),
            "retained_windows": summarize(retained),
            "coverage_fraction": summarize(cov_frac),
            "max_uncovered_gap_words": summarize(max_gaps),
            "docs_with_any_uncovered_gap": int(sum(x > 0 for x in max_gaps)),
            "docs_coverage_below_90pct": int(sum(x < .90 for x in cov_frac)),
            "docs_coverage_below_75pct": int(sum(x < .75 for x in cov_frac)),
            "docs_hitting_cap32": int(sum(x > CAP for x in possible)),
        },
        "tokenization": {
            "aiteam_corpus_chunk_tokens": {
                **summarize(ai_chunk_lengths),
                "gt512": int(sum(x > MAXLEN for x in ai_chunk_lengths)),
                "gt512_rate": float(
                    np.mean(np.asarray(ai_chunk_lengths) > MAXLEN)
                ),
            },
            "jina_query_top_passage_pair_tokens": {
                **summarize(jina_pair_lengths),
                "gt512": int(sum(x > MAXLEN for x in jina_pair_lengths)),
                "gt512_rate": float(
                    np.mean(np.asarray(jina_pair_lengths) > MAXLEN)
                ),
            },
            "aiteam_top_passage_tokens": {
                **summarize(ai_passage_lengths),
                "gt512": int(sum(x > MAXLEN for x in ai_passage_lengths)),
                "gt512_rate": float(
                    np.mean(np.asarray(ai_passage_lengths) > MAXLEN)
                ),
            },
            "jina_query_structured_section_pair_tokens": {
                **summarize(section_pair_lengths),
                "gt512": int(sum(x > MAXLEN for x in section_pair_lengths)),
                "gt512_rate": float(
                    np.mean(np.asarray(section_pair_lengths) > MAXLEN)
                ),
            },
            "passages_per_candidate_pair": summarize(passages_per_pair),
            "sections_per_candidate_pair": summarize(sections_per_pair),
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
            "section_pair_truncation_material": bool(
                np.mean(np.asarray(section_pair_lengths) > MAXLEN) >= .01
            ),
        },
    }

    (out / "REPORT.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    # Keep per-doc coverage rows for forensic follow-up, but not all token rows.
    (out / "CORPUS_COVERAGE_ROWS.json").write_text(
        json.dumps(coverage_rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("[5/5] RESULT", flush=True)
    print("=" * 104)
    print(
        "Corpus coverage mean/p5? "
        f"mean={np.mean(cov_frac):.4f} "
        f"p05={np.percentile(cov_frac,5):.4f} "
        f"p95 max-gap={np.percentile(max_gaps,95):.1f} words"
    )
    print(
        f"AITeam corpus chunks >512: "
        f"{sum(x>MAXLEN for x in ai_chunk_lengths)}/{len(ai_chunk_lengths)} "
        f"({100*np.mean(np.asarray(ai_chunk_lengths)>MAXLEN):.2f}%)"
    )
    print(
        f"Jina query+top_passage >512: "
        f"{sum(x>MAXLEN for x in jina_pair_lengths)}/{len(jina_pair_lengths)} "
        f"({100*np.mean(np.asarray(jina_pair_lengths)>MAXLEN):.2f}%)"
    )
    print(
        f"AITeam top_passage >512: "
        f"{sum(x>MAXLEN for x in ai_passage_lengths)}/{len(ai_passage_lengths)} "
        f"({100*np.mean(np.asarray(ai_passage_lengths)>MAXLEN):.2f}%)"
    )
    print(
        f"Jina query+structured_section >512: "
        f"{sum(x>MAXLEN for x in section_pair_lengths)}/{len(section_pair_lengths)} "
        f"({100*np.mean(np.asarray(section_pair_lengths)>MAXLEN):.2f}%)"
    )
    print("Flags:", report["interpretation_flags"])
    print("Report:", out / "REPORT.json")
    print("=" * 104)


if __name__ == "__main__":
    main()
