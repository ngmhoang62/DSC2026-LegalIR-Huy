"""Deterministic, label-free legal section parser and preselector for Vietnamese legal documents."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import List, Set, Tuple

STOPWORDS = {
    "bị", "các", "có", "của", "cho", "được", "để", "đến", "đối", "gì",
    "hay", "khi", "không", "là", "làm", "một", "nào", "những", "như",
    "phải", "ra", "sẽ", "theo", "thì", "thế", "trong", "trên", "từ",
    "và", "về", "với", "việc", "bao", "nhiêu", "người", "quy", "định",
}

TOKEN_RE = re.compile(r"\w+", re.UNICODE)
SPACE_RE = re.compile(r"\S+", re.UNICODE)

# Regex to detect legal structural boundaries
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


def tokens(text: str) -> List[str]:
    return TOKEN_RE.findall((text or "").lower())


@dataclass
class LegalSection:
    doc_id: str
    section_index: int
    section_type: str
    heading: str
    text: str
    word_count: int


def parse_document_into_sections(
    doc_id: str,
    raw_text: str,
    max_chunk_words: int = 220,
    overlap_words: int = 60,
) -> List[LegalSection]:
    """Parse raw Vietnamese legal document into structured, contextualized legal sections."""
    if not raw_text or not raw_text.strip():
        return []

    words = SPACE_RE.findall(raw_text)
    if len(words) <= max_chunk_words + 50:
        # Document is short enough to fit in a single section
        first_line = raw_text.strip().split("\n")[0].strip()
        sec = LegalSection(
            doc_id=doc_id,
            section_index=0,
            section_type="FULL_DOC",
            heading=first_line[:120],
            text=raw_text.strip(),
            word_count=len(words),
        )
        return [sec]

    # Extract preamble / doc header (issuing authority, number, title) from first 60 words
    doc_header = " ".join(words[:60])

    parts = BOUNDARY_RE.split(raw_text)
    parts = [p.strip() for p in parts if p.strip()]

    # If no structural boundary detected, fall back to sliding windows
    if len(parts) <= 1:
        sections = []
        step = max_chunk_words - overlap_words
        sec_idx = 0
        for i in range(0, len(words), step):
            sub = " ".join(words[i : i + max_chunk_words])
            sec_text = f"{doc_header}\n[MỤC]\n{sub}" if i >= 60 else sub
            sections.append(
                LegalSection(
                    doc_id=doc_id,
                    section_index=sec_idx,
                    section_type="FALLBACK_WINDOW",
                    heading=f"Đoạn {sec_idx + 1}",
                    text=sec_text,
                    word_count=len(sub.split()),
                )
            )
            sec_idx += 1
            if i + max_chunk_words >= len(words):
                break
        return sections

    sections: List[LegalSection] = []
    sec_idx = 0

    # parts[0] is often the document preamble/metadata
    preamble_words = parts[0].split()
    if len(preamble_words) > 30 and len(parts) > 1:
        # Keep preamble as a section if substantial
        preamble_text = f"{doc_header}\n[PHẦN MỞ ĐẦU / CĂN CỨ]\n{parts[0]}"
        sections.append(
            LegalSection(
                doc_id=doc_id,
                section_index=sec_idx,
                section_type="PREAMBLE",
                heading="Phần mở đầu / Căn cứ pháp lý",
                text=preamble_text,
                word_count=len(preamble_words),
            )
        )
        sec_idx += 1

    # Process all subsequent structural parts
    for part in parts[1:]:
        p_words = part.split()
        if not p_words:
            continue

        first_line = part.split("\n")[0].strip()
        m = HEADING_RE.match(part)
        if m:
            heading = m.group(1).strip()
            if heading.lower().startswith("điều"):
                sec_type = "DIEU"
            elif heading.lower().startswith("chương"):
                sec_type = "CHUONG"
            elif heading.lower().startswith("mục"):
                sec_type = "MUC"
            elif heading.lower().startswith("phụ"):
                sec_type = "PHU_LUC"
            else:
                sec_type = "OTHER"
        else:
            heading = first_line[:120]
            sec_type = "OTHER"

        # If the article fits in a single chunk, keep it intact
        if len(p_words) <= max_chunk_words:
            formatted_text = f"{doc_header}\n[ĐIỀU KHOẢN: {heading}]\n{part}"
            sections.append(
                LegalSection(
                    doc_id=doc_id,
                    section_index=sec_idx,
                    section_type=sec_type,
                    heading=heading,
                    text=formatted_text,
                    word_count=len(p_words),
                )
            )
            sec_idx += 1
        else:
            # Subchunk large article deterministically, retaining article heading and doc header in every chunk
            step = max_chunk_words - overlap_words
            for i in range(0, len(p_words), step):
                sub = " ".join(p_words[i : i + max_chunk_words])
                sub_heading = f"{heading} (Đoạn {i // step + 1})" if i > 0 else heading
                formatted_text = f"{doc_header}\n[ĐIỀU KHOẢN: {sub_heading}]\n{sub}"
                sections.append(
                    LegalSection(
                        doc_id=doc_id,
                        section_index=sec_idx,
                        section_type=sec_type,
                        heading=sub_heading,
                        text=formatted_text,
                        word_count=len(sub.split()),
                    )
                )
                sec_idx += 1
                if i + max_chunk_words >= len(p_words):
                    break

    return sections


def score_section_lexical(query: str, section: LegalSection) -> float:
    """Compute deterministic, label-free lexical score between query and legal section."""
    q_tokens = tokens(query)
    content_tokens = {t for t in q_tokens if len(t) >= 3 and t not in STOPWORDS}
    number_tokens = {t for t in q_tokens if any(c.isdigit() for c in t)}
    bigrams = {" ".join(q_tokens[i : i + 2]) for i in range(len(q_tokens) - 1)}

    sec_toks = tokens(section.text)
    token_set = set(sec_toks)
    norm_text = " ".join(sec_toks)

    # Content token coverage
    coverage = sum(
        1.0 + 0.20 * min(sec_toks.count(t), 3)
        for t in content_tokens
        if t in token_set
    )
    # Exact numeric token matches (article numbers, penalties, years)
    numeric = 3.0 * sum(t in token_set for t in number_tokens)
    # Bigram phrase matches
    phrase = 1.8 * sum(p in norm_text for p in bigrams)

    # Heading match bonus
    heading_bonus = 0.0
    if section.heading:
        h_toks = set(tokens(section.heading))
        for num in number_tokens:
            if num in h_toks:
                heading_bonus += 4.0

    score = (coverage + numeric + phrase + heading_bonus) / math.sqrt(
        max(len(sec_toks), 1)
    )
    return float(score)


def preselect_legal_sections(
    query: str,
    sections: List[LegalSection],
    count: int = 2,
) -> List[LegalSection]:
    """Deterministically preselect top-k legal sections for a query."""
    if not sections:
        return []
    if len(sections) <= count:
        return sections

    scored = []
    for s in sections:
        sc = score_section_lexical(query, s)
        # Tie-breaking by negative section index (earlier sections preferred on tie)
        scored.append((sc, -s.section_index, s))

    scored.sort(reverse=True)
    return [s for _, _, s in scored[:count]]
