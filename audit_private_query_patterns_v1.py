#!/usr/bin/env python
"""
PRIVATE QUERY PATTERN AUDIT — DSC2026 LEGAL IR
==============================================

Label-free / CPU-only.

Primary goal
------------
Inspect the *scored* private query population after removing the 103 warmup
overlaps that organizers exclude from private scoring.

Default expected population:
    private total       = 2080
    excluded warmup     = 103
    scored-pattern set  = 1977

The script deliberately separates:
  A) query intent/archetype heuristics;
  B) structural/legal-reference cues;
  C) lexical distribution;
  D) optional D1 boundary uncertainty from EXISTING private decision-score cache.

No private labels are read. No model inference is performed.

Primary archetypes (heuristic, mutually exclusive)
---------------------------------------------------
  DIRECT_REF
  ARTICLE_LOOKUP
  DEFINITION
  PROCEDURE
  SANCTION
  CONDITION
  CROSS_REFERENCE
  GENERAL_SEMANTIC

A query may also have many independent cue flags, e.g.:
  HAS_ARTICLE, HAS_CLAUSE, HAS_POINT, HAS_INSTRUMENT, HAS_DOC_NUMBER,
  HAS_YEAR, HAS_EXPLICIT_CITATION, HAS_HYPOTHETICAL, HAS_NEGATION,
  HAS_DURATION, HAS_MONEY, HAS_PERCENT, HAS_QUANTITY, etc.

Important
---------
Archetypes are deterministic *routing heuristics*, not gold semantic labels.
The report preserves examples so we can inspect whether a category is coherent
before using it in any retrieval/ranking experiment.

Outputs
-------
results/manual/huy_private_query_patterns_v1/
  PRIVATE_OVERLAP_AUDIT.json
  PRIVATE_QUERY_PATTERNS.jsonl
  PRIVATE_QUERY_PATTERN_REPORT.json
  PRIVATE_QUERY_PATTERN_SUMMARY.txt
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


EXPECTED_PRIVATE = 2080
EXPECTED_EXCLUDED = 103
EXPECTED_SCORED = EXPECTED_PRIVATE - EXPECTED_EXCLUDED

# ---------------------------------------------------------------------------
# Normalization / loading
# ---------------------------------------------------------------------------

WS_RE = re.compile(r"\s+", re.UNICODE)
WORD_RE = re.compile(r"\b[\wÀ-ỹĐđ]+\b", re.UNICODE)
PUNCT_RE = re.compile(r"[^\wÀ-ỹĐđ]+", re.UNICODE)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def norm_text(s: str) -> str:
    return WS_RE.sub(
        " ",
        unicodedata.normalize("NFKC", str(s or "")).strip().lower(),
    )


def loose_text(s: str) -> str:
    s = norm_text(s)
    s = PUNCT_RE.sub(" ", s)
    return WS_RE.sub(" ", s).strip()


def load_questions(path: Path) -> tuple[list[str], dict[str, str]]:
    obj = json.loads(path.read_text(encoding="utf-8"))

    ids: list[str] = []
    questions: dict[str, str] = {}

    if isinstance(obj, dict):
        for k, v in obj.items():
            qid = str(k)
            if isinstance(v, str):
                text = v
            elif isinstance(v, dict):
                text = (
                    v.get("question")
                    or v.get("query")
                    or v.get("text")
                    or v.get("content")
                )
            else:
                text = None
            if text is None:
                continue
            ids.append(qid)
            questions[qid] = str(text)

    elif isinstance(obj, list):
        for row in obj:
            if not isinstance(row, dict):
                continue
            qid = row.get("id", row.get("qid", row.get("query_id")))
            text = (
                row.get("question")
                or row.get("query")
                or row.get("text")
                or row.get("content")
            )
            if qid is None or text is None:
                continue
            qid = str(qid)
            ids.append(qid)
            questions[qid] = str(text)

    if not ids:
        raise RuntimeError(f"Unsupported/no-question schema: {path}")
    return ids, questions


def resolve_file(
    explicit: Path | None,
    candidates: list[Path],
    *,
    required: bool,
    label: str,
) -> Path | None:
    if explicit is not None:
        p = explicit.expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(f"{label}: {p}")
        return p

    hits = [p.resolve() for p in candidates if p.is_file()]
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        # Prefer first listed canonical location deterministically.
        return hits[0]
    if required:
        raise FileNotFoundError(
            f"Could not autodiscover {label}. Tried:\n"
            + "\n".join(f"  {p}" for p in candidates)
        )
    return None


# ---------------------------------------------------------------------------
# Warmup overlap audit
# ---------------------------------------------------------------------------

def overlap_audit(
    private_ids: list[str],
    private_q: dict[str, str],
    warm_ids: list[str],
    warm_q: dict[str, str],
):
    warm_id_set = set(warm_ids)

    warm_norm_to_ids = defaultdict(list)
    warm_loose_to_ids = defaultdict(list)
    for q in warm_ids:
        warm_norm_to_ids[norm_text(warm_q[q])].append(q)
        warm_loose_to_ids[loose_text(warm_q[q])].append(q)

    excluded = set()
    rows = []

    counts = Counter()

    for q in private_ids:
        t = private_q[q]
        n = norm_text(t)
        l = loose_text(t)

        same_id = q in warm_id_set
        same_id_same_text = same_id and norm_text(warm_q[q]) == n
        same_id_diff_text = same_id and not same_id_same_text

        exact_text_ids = warm_norm_to_ids.get(n, [])
        cross_id_exact = [x for x in exact_text_ids if x != q]

        loose_ids = warm_loose_to_ids.get(l, [])
        loose_only = [
            x for x in loose_ids
            if x not in exact_text_ids and x != q
        ]

        # Organizer exclusion proxy:
        # same QID OR exact normalized question text.
        is_excluded = bool(same_id or exact_text_ids)
        if is_excluded:
            excluded.add(q)

        if same_id_same_text:
            counts["same_qid_same_text"] += 1
        if same_id_diff_text:
            counts["same_qid_diff_text"] += 1
        if cross_id_exact:
            counts["cross_qid_exact_text"] += 1
        if loose_only:
            counts["loose_text_only_not_excluded"] += 1

        if same_id or exact_text_ids or loose_only:
            rows.append({
                "private_qid": q,
                "question": t,
                "excluded": is_excluded,
                "same_qid": same_id,
                "same_qid_same_text": same_id_same_text,
                "same_qid_diff_text": same_id_diff_text,
                "exact_text_warmup_qids": exact_text_ids,
                "cross_qid_exact_text_warmup_qids": cross_id_exact,
                "loose_only_warmup_qids": loose_only,
            })

    return excluded, {
        "private_queries": len(private_ids),
        "warmup_queries": len(warm_ids),
        "excluded_union_count": len(excluded),
        "scored_private_count": len(private_ids) - len(excluded),
        "counts": dict(counts),
        "definition": (
            "excluded iff private qid exists in warmup OR normalized exact "
            "question text exists in warmup; loose punctuation-only matches "
            "are diagnostic and are NOT excluded"
        ),
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# Query cues / archetypes
# ---------------------------------------------------------------------------

# Vietnamese legal-structure patterns. Keep these intentionally interpretable.
ARTICLE_RE = re.compile(r"\bđiều\s+\d+[a-zđ]?\b", re.I | re.UNICODE)
CLAUSE_RE = re.compile(r"\bkhoản\s+\d+\b", re.I | re.UNICODE)
POINT_RE = re.compile(
    r"\bđiểm\s+(?:[a-zđ]|[a-zđ]\s*,\s*[a-zđ]|\d+)\b",
    re.I | re.UNICODE,
)
CHAPTER_RE = re.compile(r"\bchương\s+(?:[ivxlcdm]+|\d+)\b", re.I | re.UNICODE)
SECTION_RE = re.compile(r"\bmục\s+(?:[ivxlcdm]+|\d+)\b", re.I | re.UNICODE)

DOC_NUMBER_RE = re.compile(
    r"\b\d{1,4}\s*/\s*\d{4}\s*/\s*"
    r"(?:nđ-cp|nd-cp|tt-[a-zđ]+|tt-btc|tt-bca|tt-byt|"
    r"qđ-[a-zđ]+|qd-[a-zđ]+|qh\d*|ubtvqh\d*|cp|btc|bca|byt|"
    r"bgtvt|blđtbxh|bldtbxh|bkhđt|bkhdt|btnmt)\b",
    re.I | re.UNICODE,
)
YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
MONEY_RE = re.compile(
    r"\b\d+(?:[.,]\d+)?\s*(?:đồng|vnđ|vnd|triệu|tỷ)\b",
    re.I | re.UNICODE,
)
PERCENT_RE = re.compile(r"\b\d+(?:[.,]\d+)?\s*%")
DURATION_RE = re.compile(
    r"\b\d+\s*(?:ngày|tháng|năm|giờ|phút|tuần)\b",
    re.I | re.UNICODE,
)

INSTRUMENT_WORDS = (
    "luật ", "bộ luật", "nghị định", "thông tư", "quyết định",
    "nghị quyết", "pháp lệnh", "hiến pháp", "quy chuẩn",
    "tiêu chuẩn", "văn bản", "quy chế",
)

SANCTION_PHRASES = (
    "xử phạt", "mức phạt", "phạt bao nhiêu", "tiền phạt",
    "bị phạt", "hình phạt", "xử lý vi phạm", "chế tài",
    "phạt tiền", "trách nhiệm hình sự", "truy cứu",
)
DEFINITION_PHRASES = (
    "là gì", "được hiểu là", "thế nào là", "khái niệm",
    "định nghĩa", "có nghĩa là", "được gọi là",
)
PROCEDURE_PHRASES = (
    "thủ tục", "hồ sơ", "trình tự", "nộp hồ sơ", "đăng ký",
    "cấp giấy", "cấp phép", "xin phép", "giải quyết",
    "thẩm quyền", "cơ quan nào", "cơ quan có thẩm quyền",
    "thời hạn giải quyết", "gia hạn", "cấp lại", "cấp đổi",
    "thu hồi", "đề nghị", "yêu cầu cấp",
)
CONDITION_PHRASES = (
    "điều kiện", "khi nào", "trường hợp nào", "trong trường hợp",
    "có được", "được phép", "không được", "phải đáp ứng",
    "cần đáp ứng", "yêu cầu gì", "cần những gì",
    "đủ điều kiện", "điều kiện để", "điều kiện nào",
)
CROSSREF_PHRASES = (
    "theo quy định tại", "quy định tại điều", "quy định tại khoản",
    "căn cứ", "dẫn chiếu", "theo điều", "theo khoản", "theo điểm",
    "tại điều", "tại khoản", "tại điểm", "chiếu theo",
)
ARTICLE_LOOKUP_PHRASES = (
    "quy định gì", "nội dung của điều", "điều này quy định",
    "theo điều", "tại điều", "điều nào quy định",
)
HYPOTHETICAL_PHRASES = (
    "nếu ", "giả sử", "trong trường hợp", "trường hợp ",
    "khi ", "khi mà",
)
NEGATION_PHRASES = (
    "không ", "chưa ", "không được", "không phải", "không có",
)
COMPARISON_PHRASES = (
    "khác nhau", "phân biệt", "so với", "khác gì", "giống nhau",
)
OBLIGATION_PHRASES = (
    "phải ", "bắt buộc", "có nghĩa vụ", "nghĩa vụ", "trách nhiệm",
)
RIGHT_PHRASES = (
    "quyền ", "được quyền", "có quyền", "quyền lợi",
)


def contains_any(text: str, phrases) -> bool:
    return any(p in text for p in phrases)


def cue_features(question: str) -> dict[str, Any]:
    t = norm_text(question)
    words = WORD_RE.findall(t)
    digits = re.findall(r"\d+", t)

    article_count = len(ARTICLE_RE.findall(t))
    clause_count = len(CLAUSE_RE.findall(t))
    point_count = len(POINT_RE.findall(t))
    chapter_count = len(CHAPTER_RE.findall(t))
    section_count = len(SECTION_RE.findall(t))

    has_instrument = contains_any(t, INSTRUMENT_WORDS)
    has_doc_number = bool(DOC_NUMBER_RE.search(t))
    explicit_citation = bool(
        article_count
        or clause_count
        or point_count
        or chapter_count
        or section_count
        or has_doc_number
        or contains_any(t, CROSSREF_PHRASES)
    )

    # Question form / semantic cues.
    sanction = contains_any(t, SANCTION_PHRASES)
    definition = contains_any(t, DEFINITION_PHRASES)
    procedure = contains_any(t, PROCEDURE_PHRASES)
    condition = contains_any(t, CONDITION_PHRASES)
    crossref = contains_any(t, CROSSREF_PHRASES)

    # ARTICLE_LOOKUP is intentionally narrow: explicit article reference plus
    # wording that asks for content/provision, without overriding task intent
    # such as sanction/procedure.
    article_lookup = bool(
        article_count > 0
        and (
            contains_any(t, ARTICLE_LOOKUP_PHRASES)
            or re.search(
                r"\bđiều\s+\d+[a-zđ]?\b.{0,45}"
                r"(?:quy định|nội dung|nói về|áp dụng)",
                t,
                re.I | re.UNICODE,
            )
        )
    )

    direct_ref = bool(
        explicit_citation or has_instrument or has_doc_number
    )

    # Intent first, structural lookup later. This makes e.g. "mức phạt theo
    # Điều 15" SANCTION with HAS_ARTICLE rather than hiding intent as DIRECT_REF.
    if sanction:
        primary = "SANCTION"
    elif definition:
        primary = "DEFINITION"
    elif procedure:
        primary = "PROCEDURE"
    elif condition:
        primary = "CONDITION"
    elif article_lookup:
        primary = "ARTICLE_LOOKUP"
    elif crossref and explicit_citation:
        primary = "CROSS_REFERENCE"
    elif direct_ref:
        primary = "DIRECT_REF"
    else:
        primary = "GENERAL_SEMANTIC"

    cue = {
        "HAS_ARTICLE": article_count > 0,
        "HAS_CLAUSE": clause_count > 0,
        "HAS_POINT": point_count > 0,
        "HAS_CHAPTER": chapter_count > 0,
        "HAS_SECTION": section_count > 0,
        "HAS_INSTRUMENT": has_instrument,
        "HAS_DOC_NUMBER": has_doc_number,
        "HAS_YEAR": bool(YEAR_RE.search(t)),
        "HAS_EXPLICIT_CITATION": explicit_citation,
        "HAS_HYPOTHETICAL": contains_any(t, HYPOTHETICAL_PHRASES),
        "HAS_NEGATION": contains_any(t, NEGATION_PHRASES),
        "HAS_COMPARISON": contains_any(t, COMPARISON_PHRASES),
        "HAS_OBLIGATION": contains_any(t, OBLIGATION_PHRASES),
        "HAS_RIGHT": contains_any(t, RIGHT_PHRASES),
        "HAS_DURATION": bool(DURATION_RE.search(t)),
        "HAS_MONEY": bool(MONEY_RE.search(t)),
        "HAS_PERCENT": bool(PERCENT_RE.search(t)),
        "HAS_QUESTION_MARK": "?" in question,
        "HAS_MULTIPLE_SENTENCES": (
            len(re.findall(r"[.!?]+", question)) >= 2
        ),
        "NUMERIC_HEAVY": len(digits) >= 3,
    }

    return {
        "primary_archetype": primary,
        "cues": cue,
        "article_count": article_count,
        "clause_count": clause_count,
        "point_count": point_count,
        "word_count": len(words),
        "char_count": len(question),
        "digit_token_count": len(digits),
    }


# ---------------------------------------------------------------------------
# D1 private boundary uncertainty (optional)
# ---------------------------------------------------------------------------

def load_d1_uncertainty(path: Path | None, private_ids: list[str]):
    if path is None or not path.is_file():
        return {}, {
            "status": "SKIPPED",
            "reason": "d1_private_scores.pkl not found",
        }

    obj = pickle.loads(path.read_bytes())
    if not isinstance(obj, dict):
        return {}, {"status": "SKIPPED_BAD_SCHEMA"}

    decision = obj.get("decision_scores")
    top5 = obj.get("top5")

    if not isinstance(decision, dict):
        return {}, {
            "status": "SKIPPED_BAD_SCHEMA",
            "keys": sorted(map(str, obj.keys())),
        }

    out = {}
    missing = []

    for q in private_ids:
        row = decision.get(q)
        if not isinstance(row, dict) or len(row) < 6:
            missing.append(q)
            continue

        ordered = sorted(
            ((str(d), float(s)) for d, s in row.items()),
            key=lambda x: (-x[1], x[0]),
        )
        scores = np.asarray([s for _, s in ordered], dtype=np.float64)

        s1 = float(scores[0])
        s5 = float(scores[4])
        s6 = float(scores[5])
        med = float(np.median(scores))
        sd = float(np.std(scores))
        if sd <= 1e-12:
            sd = 1.0

        out[q] = {
            "d1_rank5_minus_rank6": s5 - s6,
            "d1_rank1_minus_rank5": s1 - s5,
            "d1_rank5_z": (s5 - med) / sd,
            "d1_score_std": float(np.std(scores)),
            "d1_candidate_count": int(len(scores)),
            "d1_rank5_doc": ordered[4][0],
            "d1_rank6_doc": ordered[5][0],
        }

        if isinstance(top5, dict) and q in top5:
            expected = [str(x) for x in top5[q]]
            got = [d for d, _ in ordered[:5]]
            if expected != got:
                raise RuntimeError(
                    f"D1 cache internal parity failed qid={q}: "
                    f"top5 cache={expected}, decision={got}"
                )

    return out, {
        "status": "OK",
        "path": str(path),
        "sha256": sha256(path),
        "queries_available": len(out),
        "queries_missing": len(missing),
        "missing_sample": missing[:20],
    }


# ---------------------------------------------------------------------------
# Summary / comparison helpers
# ---------------------------------------------------------------------------

ARCHETYPES = [
    "DIRECT_REF",
    "ARTICLE_LOOKUP",
    "DEFINITION",
    "PROCEDURE",
    "SANCTION",
    "CONDITION",
    "CROSS_REFERENCE",
    "GENERAL_SEMANTIC",
]


def stat(values):
    vals = np.asarray(list(values), dtype=np.float64)
    if vals.size == 0:
        return None
    return {
        "n": int(vals.size),
        "mean": float(np.mean(vals)),
        "median": float(np.median(vals)),
        "p10": float(np.quantile(vals, .10)),
        "p25": float(np.quantile(vals, .25)),
        "p75": float(np.quantile(vals, .75)),
        "p90": float(np.quantile(vals, .90)),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
    }


def prevalence(records):
    n = max(1, len(records))
    arch = Counter(r["primary_archetype"] for r in records)
    cue_names = sorted({
        k for r in records for k in r["cues"]
    })
    cue = {
        k: sum(bool(r["cues"].get(k)) for r in records) / n
        for k in cue_names
    }
    return {
        "queries": len(records),
        "archetype_counts": {a: int(arch.get(a, 0)) for a in ARCHETYPES},
        "archetype_fraction": {
            a: float(arch.get(a, 0) / n) for a in ARCHETYPES
        },
        "cue_fraction": cue,
        "word_count": stat(r["word_count"] for r in records),
        "char_count": stat(r["char_count"] for r in records),
    }


def js_divergence(p: dict[str, float], q: dict[str, float]) -> float:
    keys = sorted(set(p) | set(q))
    pv = np.asarray([p.get(k, 0.0) for k in keys], dtype=np.float64)
    qv = np.asarray([q.get(k, 0.0) for k in keys], dtype=np.float64)

    if pv.sum() <= 0 or qv.sum() <= 0:
        return 0.0
    pv /= pv.sum()
    qv /= qv.sum()
    m = 0.5 * (pv + qv)

    def kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

    return 0.5 * kl(pv, m) + 0.5 * kl(qv, m)


def compare_distribution(private_summary, other_summary):
    arch_delta = {
        a: (
            private_summary["archetype_fraction"].get(a, 0.0)
            - other_summary["archetype_fraction"].get(a, 0.0)
        )
        for a in ARCHETYPES
    }

    cues = sorted(
        set(private_summary["cue_fraction"])
        | set(other_summary["cue_fraction"])
    )
    cue_delta = {
        c: (
            private_summary["cue_fraction"].get(c, 0.0)
            - other_summary["cue_fraction"].get(c, 0.0)
        )
        for c in cues
    }

    return {
        "archetype_js_divergence_bits": js_divergence(
            private_summary["archetype_fraction"],
            other_summary["archetype_fraction"],
        ),
        "archetype_fraction_delta_private_minus_other": arch_delta,
        "cue_fraction_delta_private_minus_other": cue_delta,
        "largest_archetype_shifts": sorted(
            (
                {"name": k, "delta": v}
                for k, v in arch_delta.items()
            ),
            key=lambda x: -abs(x["delta"]),
        ),
        "largest_cue_shifts": sorted(
            (
                {"name": k, "delta": v}
                for k, v in cue_delta.items()
            ),
            key=lambda x: -abs(x["delta"]),
        ),
    }


def make_records(
    ids,
    questions,
    *,
    d1_uncertainty=None,
):
    d1_uncertainty = d1_uncertainty or {}
    rows = []
    for q in ids:
        f = cue_features(questions[q])
        row = {
            "qid": q,
            "question": questions[q],
            **f,
        }
        if q in d1_uncertainty:
            row["d1"] = d1_uncertainty[q]
        rows.append(row)
    return rows


def examples_by_archetype(records, n=8):
    out = {}
    for a in ARCHETYPES:
        rows = [r for r in records if r["primary_archetype"] == a]
        # Deterministic spread: shortest/median/longest-ish rather than only first.
        rows = sorted(rows, key=lambda r: (r["word_count"], r["qid"]))
        if len(rows) <= n:
            chosen = rows
        else:
            idx = np.linspace(0, len(rows) - 1, n, dtype=int)
            chosen = [rows[int(i)] for i in idx]
        out[a] = [
            {
                "qid": r["qid"],
                "question": r["question"],
                "word_count": r["word_count"],
                "cues": [k for k, v in r["cues"].items() if v],
            }
            for r in chosen
        ]
    return out


def uncertainty_by_group(records):
    rows = [r for r in records if "d1" in r]
    if not rows:
        return {"status": "SKIPPED"}

    global_margin = np.asarray(
        [r["d1"]["d1_rank5_minus_rank6"] for r in rows],
        dtype=np.float64,
    )
    q10 = float(np.quantile(global_margin, .10))
    q25 = float(np.quantile(global_margin, .25))

    by_arch = {}
    for a in ARCHETYPES:
        rr = [r for r in rows if r["primary_archetype"] == a]
        margins = [r["d1"]["d1_rank5_minus_rank6"] for r in rr]
        by_arch[a] = {
            "queries": len(rr),
            "rank5_minus_rank6": stat(margins),
            "very_ambiguous_fraction_global_p10": (
                float(np.mean(np.asarray(margins) <= q10))
                if margins else None
            ),
            "ambiguous_fraction_global_p25": (
                float(np.mean(np.asarray(margins) <= q25))
                if margins else None
            ),
        }

    cue_names = sorted({k for r in rows for k in r["cues"]})
    by_cue = {}
    for c in cue_names:
        yes = [
            r["d1"]["d1_rank5_minus_rank6"]
            for r in rows if r["cues"].get(c)
        ]
        no = [
            r["d1"]["d1_rank5_minus_rank6"]
            for r in rows if not r["cues"].get(c)
        ]
        by_cue[c] = {
            "queries_with_cue": len(yes),
            "mean_margin_with_cue": (
                float(np.mean(yes)) if yes else None
            ),
            "mean_margin_without_cue": (
                float(np.mean(no)) if no else None
            ),
            "delta_mean_margin_yes_minus_no": (
                float(np.mean(yes) - np.mean(no))
                if yes and no else None
            ),
        }

    ambiguous = sorted(
        rows,
        key=lambda r: (
            r["d1"]["d1_rank5_minus_rank6"],
            r["qid"],
        ),
    )

    return {
        "status": "OK",
        "queries": len(rows),
        "global_rank5_minus_rank6": stat(global_margin),
        "global_p10_threshold": q10,
        "global_p25_threshold": q25,
        "by_archetype": by_arch,
        "by_cue": by_cue,
        "most_ambiguous_50": [
            {
                "qid": r["qid"],
                "question": r["question"],
                "primary_archetype": r["primary_archetype"],
                "active_cues": [
                    k for k, v in r["cues"].items() if v
                ],
                **r["d1"],
            }
            for r in ambiguous[:50]
        ],
    }


# ---------------------------------------------------------------------------
# Lexical n-gram shifts
# ---------------------------------------------------------------------------

STOPWORDS = {
    "và", "là", "có", "của", "được", "cho", "trong", "theo", "với",
    "thì", "khi", "các", "một", "những", "này", "đó", "về", "để",
    "từ", "tại", "hay", "phải", "không", "như", "nào", "gì", "bao",
    "nhiêu", "quy", "định",
}


def tokens_for_ngrams(text: str):
    return [
        w for w in WORD_RE.findall(norm_text(text))
        if len(w) >= 2 and w not in STOPWORDS
    ]


def ngram_docfreq(records, n):
    df = Counter()
    for r in records:
        toks = tokens_for_ngrams(r["question"])
        grams = {
            " ".join(toks[i:i+n])
            for i in range(len(toks) - n + 1)
        }
        df.update(grams)
    return df


def distinctive_ngrams(private_records, other_records, n, topk=30):
    a = ngram_docfreq(private_records, n)
    b = ngram_docfreq(other_records, n)
    na = max(1, len(private_records))
    nb = max(1, len(other_records))

    rows = []
    vocab = set(a) | set(b)
    for g in vocab:
        ca, cb = a[g], b[g]
        if ca < 5:
            continue
        pa = (ca + 1.0) / (na + 2.0)
        pb = (cb + 1.0) / (nb + 2.0)
        log2_ratio = math.log2(pa / pb)
        rows.append({
            "ngram": g,
            "private_df": int(ca),
            "private_fraction": float(ca / na),
            "other_df": int(cb),
            "other_fraction": float(cb / nb),
            "log2_prevalence_ratio": float(log2_ratio),
        })

    rows.sort(
        key=lambda x: (
            -x["log2_prevalence_ratio"],
            -x["private_df"],
            x["ngram"],
        )
    )
    return rows[:topk]


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------

def fmt_pct(x):
    return f"{100*x:.1f}%"


def build_summary_text(report):
    p = report["private_scored_summary"]
    lines = []
    lines.append("PRIVATE QUERY PATTERN AUDIT — LABEL FREE")
    lines.append("=" * 88)
    lines.append(
        f"Private: {report['population']['private_total']} | "
        f"warmup-excluded: {report['population']['warmup_excluded']} | "
        f"scored-pattern population: {report['population']['private_scored']}"
    )
    lines.append("")

    lines.append("PRIMARY ARCHETYPES")
    lines.append("-" * 88)
    for a in ARCHETYPES:
        lines.append(
            f"{a:<20} "
            f"{p['archetype_counts'][a]:>5} "
            f"{fmt_pct(p['archetype_fraction'][a]):>8}"
        )

    lines.append("")
    lines.append("MOST COMMON STRUCTURAL CUES")
    lines.append("-" * 88)
    for k, v in sorted(
        p["cue_fraction"].items(),
        key=lambda kv: (-kv[1], kv[0]),
    ):
        lines.append(f"{k:<28} {fmt_pct(v):>8}")

    if report["comparisons"]:
        lines.append("")
        lines.append("LARGEST DISTRIBUTION SHIFTS VS REFERENCE")
        lines.append("-" * 88)
        for name, cmp in report["comparisons"].items():
            lines.append(
                f"{name}: archetype JS={cmp['archetype_js_divergence_bits']:.4f} bits"
            )
            for row in cmp["largest_archetype_shifts"][:4]:
                lines.append(
                    f"  archetype {row['name']:<20} "
                    f"{row['delta']:+.1%}"
                )
            for row in cmp["largest_cue_shifts"][:5]:
                lines.append(
                    f"  cue       {row['name']:<20} "
                    f"{row['delta']:+.1%}"
                )

    u = report["d1_uncertainty_analysis"]
    if u.get("status") == "OK":
        lines.append("")
        lines.append("D1 BOUNDARY AMBIGUITY BY ARCHETYPE")
        lines.append("-" * 88)
        lines.append(
            "Smaller rank5-rank6 decision-score margin = more ambiguous boundary."
        )
        for a in ARCHETYPES:
            row = u["by_archetype"][a]
            s = row["rank5_minus_rank6"]
            if s is None:
                continue
            lines.append(
                f"{a:<20} n={row['queries']:>4} "
                f"mean={s['mean']:.4f} "
                f"p25={s['p25']:.4f} "
                f"ambig<=globalP25={fmt_pct(row['ambiguous_fraction_global_p25'])}"
            )

        cue_rows = [
            (k, v)
            for k, v in u["by_cue"].items()
            if v["delta_mean_margin_yes_minus_no"] is not None
            and v["queries_with_cue"] >= 20
        ]
        cue_rows.sort(
            key=lambda kv: kv[1]["delta_mean_margin_yes_minus_no"]
        )
        lines.append("")
        lines.append("CUES ASSOCIATED WITH SMALLER D1 BOUNDARY MARGIN")
        for k, v in cue_rows[:10]:
            lines.append(
                f"{k:<28} n={v['queries_with_cue']:>4} "
                f"Δmargin={v['delta_mean_margin_yes_minus_no']:+.4f}"
            )

    lines.append("")
    lines.append("NOTE")
    lines.append("-" * 88)
    lines.append(
        "Archetypes are deterministic routing heuristics, not semantic gold labels."
    )
    lines.append(
        "No private answer labels or Codabench per-query outcomes were used."
    )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=Path, required=True)
    ap.add_argument("--private-file", type=Path, default=None)
    ap.add_argument("--warmup-file", type=Path, default=None)
    ap.add_argument("--public-file", type=Path, default=None)
    ap.add_argument("--cal-file", type=Path, default=None)
    ap.add_argument("--d1-private-scores", type=Path, default=None)
    ap.add_argument("--expected-private", type=int, default=EXPECTED_PRIVATE)
    ap.add_argument("--expected-excluded", type=int, default=EXPECTED_EXCLUDED)
    ap.add_argument("--examples-per-archetype", type=int, default=8)
    args = ap.parse_args()

    root = args.repo_root.expanduser().resolve()
    data = root / "DSC2026-LegalIR-main/v4_run/public_test_dataset"

    private_path = resolve_file(
        args.private_file,
        [data / "private-official.json"],
        required=True,
        label="private file",
    )
    warmup_path = resolve_file(
        args.warmup_file,
        [
            data / "warmup.json",
            root / "DSC2026-LegalIR-main/data/warmup.json",
            root / "data/warmup.json",
        ],
        required=True,
        label="warmup file",
    )
    public_path = resolve_file(
        args.public_file,
        [data / "public-official.json"],
        required=False,
        label="public file",
    )
    cal_path = resolve_file(
        args.cal_file,
        [
            root
            / "src/gemini/huy_d1_frozen_section_public_v1/"
              "CAL_QUESTIONS_LABEL_FREE.json",
        ],
        required=False,
        label="CAL label-free question file",
    )

    if args.d1_private_scores is not None:
        d1_path = args.d1_private_scores.expanduser().resolve()
    else:
        d1_path = (
            root
            / "results/manual/huy_private_d1_rel_l0_exact_v1/"
              "cache/d1_private_scores.pkl"
        )
        if not d1_path.is_file():
            d1_path = None

    outdir = root / "results/manual/huy_private_query_patterns_v1"
    outdir.mkdir(parents=True, exist_ok=True)

    print("[1/7] Loading private + warmup questions...", flush=True)
    private_ids, private_q = load_questions(private_path)
    warm_ids, warm_q = load_questions(warmup_path)

    if len(private_ids) != args.expected_private:
        raise RuntimeError(
            f"Expected {args.expected_private} private queries, "
            f"got {len(private_ids)}"
        )

    print(
        f"  private={len(private_ids)} warmup={len(warm_ids)}",
        flush=True,
    )

    print("[2/7] Auditing organizer-excluded warmup overlap...", flush=True)
    excluded, overlap = overlap_audit(
        private_ids, private_q, warm_ids, warm_q
    )

    overlap_doc = {
        "schema": "manual.private_warmup_overlap_audit.v1",
        "private_file": str(private_path),
        "private_sha256": sha256(private_path),
        "warmup_file": str(warmup_path),
        "warmup_sha256": sha256(warmup_path),
        "expected_excluded": args.expected_excluded,
        **overlap,
    }
    overlap_path = outdir / "PRIVATE_OVERLAP_AUDIT.json"
    overlap_path.write_text(
        json.dumps(overlap_doc, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(
        f"  excluded={len(excluded)} "
        f"scored={len(private_ids)-len(excluded)}",
        flush=True,
    )
    print(
        "  "
        + " ".join(
            f"{k}={v}" for k, v in sorted(overlap["counts"].items())
        ),
        flush=True,
    )

    if len(excluded) != args.expected_excluded:
        raise RuntimeError(
            "Warmup exclusion count mismatch: "
            f"got {len(excluded)}, expected {args.expected_excluded}. "
            f"Inspect {overlap_path}; refusing to silently analyze the wrong "
            "private population."
        )

    scored_ids = [q for q in private_ids if q not in excluded]
    expected_scored = args.expected_private - args.expected_excluded
    if len(scored_ids) != expected_scored:
        raise RuntimeError(
            f"Expected scored population {expected_scored}, got {len(scored_ids)}"
        )

    print("[3/7] Extracting query patterns/archetypes...", flush=True)
    d1_uncertainty, d1_audit = load_d1_uncertainty(
        d1_path,
        private_ids,
    )
    private_records = make_records(
        scored_ids,
        private_q,
        d1_uncertainty=d1_uncertainty,
    )

    # Internal duplicate diagnostics *within scored private population*.
    norm_groups = defaultdict(list)
    for q in scored_ids:
        norm_groups[norm_text(private_q[q])].append(q)
    duplicate_groups = [
        {"qids": ids, "question": private_q[ids[0]]}
        for ids in norm_groups.values()
        if len(ids) > 1
    ]

    print("[4/7] Loading optional public/CAL reference distributions...", flush=True)
    reference_records = {}
    reference_sources = {}

    if public_path is not None:
        ids, qq = load_questions(public_path)
        reference_records["public1000"] = make_records(ids, qq)
        reference_sources["public1000"] = {
            "path": str(public_path),
            "sha256": sha256(public_path),
            "queries": len(ids),
        }
        print(f"  public1000={len(ids)}", flush=True)

    if cal_path is not None:
        ids, qq = load_questions(cal_path)
        reference_records["cal600"] = make_records(ids, qq)
        reference_sources["cal600"] = {
            "path": str(cal_path),
            "sha256": sha256(cal_path),
            "queries": len(ids),
        }
        print(f"  cal600={len(ids)}", flush=True)

    print("[5/7] Computing distribution + lexical shifts...", flush=True)
    private_summary = prevalence(private_records)
    refs_summary = {
        name: prevalence(rows)
        for name, rows in reference_records.items()
    }
    comparisons = {
        name: compare_distribution(private_summary, summary)
        for name, summary in refs_summary.items()
    }

    lexical = {}
    for name, rows in reference_records.items():
        lexical[name] = {
            "private_distinctive_unigrams": distinctive_ngrams(
                private_records, rows, 1, 35
            ),
            "private_distinctive_bigrams": distinctive_ngrams(
                private_records, rows, 2, 35
            ),
            "private_distinctive_trigrams": distinctive_ngrams(
                private_records, rows, 3, 25
            ),
        }

    print("[6/7] Aggregating optional D1 boundary uncertainty...", flush=True)
    uncertainty = uncertainty_by_group(private_records)
    if uncertainty.get("status") == "OK":
        s = uncertainty["global_rank5_minus_rank6"]
        print(
            f"  D1 margin5-6 n={s['n']} "
            f"mean={s['mean']:.4f} "
            f"p10={s['p10']:.4f} "
            f"p25={s['p25']:.4f}",
            flush=True,
        )
    else:
        print(f"  D1 uncertainty: {uncertainty['status']}", flush=True)

    print("[7/7] Writing detailed artifacts...", flush=True)

    patterns_path = outdir / "PRIVATE_QUERY_PATTERNS.jsonl"
    with patterns_path.open("w", encoding="utf-8") as f:
        for row in private_records:
            f.write(
                json.dumps(row, ensure_ascii=False) + "\n"
            )

    report = {
        "schema": "manual.private_query_patterns_v1",
        "label_free": True,
        "private_labels_used": False,
        "codabench_per_query_outcomes_used": False,
        "population": {
            "private_total": len(private_ids),
            "warmup_excluded": len(excluded),
            "private_scored": len(scored_ids),
            "expected_private": args.expected_private,
            "expected_excluded": args.expected_excluded,
            "expected_scored": expected_scored,
            "internal_exact_text_duplicate_groups_after_exclusion": (
                len(duplicate_groups)
            ),
            "internal_exact_text_duplicate_extra_qids_after_exclusion": (
                sum(len(x["qids"]) - 1 for x in duplicate_groups)
            ),
        },
        "sources": {
            "private": {
                "path": str(private_path),
                "sha256": sha256(private_path),
            },
            "warmup": {
                "path": str(warmup_path),
                "sha256": sha256(warmup_path),
            },
            "references": reference_sources,
            "d1_private_scores": d1_audit,
        },
        "warmup_overlap_audit_file": str(overlap_path),
        "archetype_definition": {
            "classes": ARCHETYPES,
            "priority": [
                "SANCTION",
                "DEFINITION",
                "PROCEDURE",
                "CONDITION",
                "ARTICLE_LOOKUP",
                "CROSS_REFERENCE",
                "DIRECT_REF",
                "GENERAL_SEMANTIC",
            ],
            "note": (
                "Primary archetype is a deterministic routing heuristic. "
                "Independent cue flags retain overlapping structure."
            ),
        },
        "private_scored_summary": private_summary,
        "reference_summaries": refs_summary,
        "comparisons": comparisons,
        "lexical_shifts": lexical,
        "examples_by_archetype": examples_by_archetype(
            private_records,
            n=args.examples_per_archetype,
        ),
        "internal_duplicate_groups_after_exclusion": duplicate_groups[:100],
        "d1_uncertainty_analysis": uncertainty,
    }

    report_path = outdir / "PRIVATE_QUERY_PATTERN_REPORT.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    summary = build_summary_text(report)
    summary_path = outdir / "PRIVATE_QUERY_PATTERN_SUMMARY.txt"
    summary_path.write_text(summary, encoding="utf-8")

    print()
    print(summary)
    print("=" * 88)
    print("PATTERNS:", patterns_path)
    print("REPORT  :", report_path)
    print("SUMMARY :", summary_path)
    print("OVERLAP :", overlap_path)
    print("=" * 88)


if __name__ == "__main__":
    main()
