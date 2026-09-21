#!/usr/bin/env python
"""
HUY PIPELINE AUDIT O — SECTION PARSER / PRESELECTOR / PACKAGING V1
==================================================================

CPU-only, no model inference/public labels.

Audits production legal-section path:
  parse_document_into_sections(max_chunk_words=220, overlap=60)
  preselect_legal_sections(query, count=2)
  Jina pair max_length=512 downstream

Questions:
1) How often parser uses true legal structure vs fallback windows?
2) If a query explicitly names "Điều N" and a candidate document actually
   contains parsed "Điều N", does top-2 lexical preselection select it?
3) For selected pair overflow >512, is the 60-word document header prepend
   the main cause?
4) If header is removed, how much residual overflow remains?

No CAL gold is used in any statistic.
"""

from __future__ import annotations

import argparse, json, re, sys, time
from collections import Counter
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

ARTICLE_Q_RE = re.compile(r"\bĐiều\s+(\d+[A-Za-z]?)\b", re.I | re.UNICODE)
ARTICLE_H_RE = re.compile(r"^\s*Điều\s+(\d+[A-Za-z]?)\b", re.I | re.UNICODE)


def summarize(x):
    a=np.asarray(x,dtype=np.float64)
    if not len(a): return {}
    return {
        "n":int(len(a)),"mean":float(a.mean()),
        "p50":float(np.percentile(a,50)),"p90":float(np.percentile(a,90)),
        "p95":float(np.percentile(a,95)),"p99":float(np.percentile(a,99)),
        "max":float(a.max())
    }


def pair_lengths(tok, qs, ps, batch=256):
    out=[]
    for s in range(0,len(ps),batch):
        e=tok(qs[s:s+batch],ps[s:s+batch],add_special_tokens=True,truncation=False,padding=False)
        out.extend(len(v) for v in e["input_ids"])
    return out


def drop_doc_header(text):
    lines=text.splitlines()
    if len(lines)>=2 and lines[1].lstrip().startswith("["):
        return "\n".join(lines[1:])
    return text


def body_only(text):
    lines=text.splitlines()
    # formatted structural sections: header / marker / raw body
    if len(lines)>=3 and lines[1].lstrip().startswith("["):
        return "\n".join(lines[2:])
    return text


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--repo-root",type=Path,required=True)
    args=ap.parse_args()
    root=args.repo_root.resolve(); sys.path.insert(0,str(root))

    from tune_corpus_cap32_fusion import build_training_cap
    from src.gemini.huy_d1_legal_section_evidence_v1.legal_section_parser import (
        parse_document_into_sections, preselect_legal_sections
    )

    print("[1/5] Loading CAL candidate membership and corpus text...",flush=True)
    queries,blocks,ids,cand,_,_=build_training_cap(
        root,32,"results/corpus_index/holdout_extended_scores_cap32.pkl",depth=20
    )

    ctx=root/"DSC2026-LegalIR-main/v4_run/public_test_dataset/selected-contexts"
    text={}
    for p in sorted(ctx.glob("context_*.json")):
        r=json.loads(p.read_text(encoding="utf-8"))
        text[str(r["id"])]=r.get("passage") or ""

    unique_docs=sorted({d for q in ids for d in cand[q]})
    print(f"  candidate pairs={sum(len(cand[q]) for q in ids):,} unique_docs={len(unique_docs):,}",flush=True)

    print("[2/5] Parsing each unique document once...",flush=True)
    sections={}
    type_counts=Counter()
    docs_fallback=0; docs_structured=0; docs_empty=0
    section_counts=[]

    for i,d in enumerate(unique_docs,1):
        ss=parse_document_into_sections(d,text.get(d,""),max_chunk_words=220,overlap_words=60)
        sections[d]=ss
        section_counts.append(len(ss))
        if not ss:
            docs_empty+=1
        types={s.section_type for s in ss}
        type_counts.update(s.section_type for s in ss)
        if types and types <= {"FALLBACK_WINDOW","FULL_DOC"}:
            docs_fallback+=1
        elif types:
            docs_structured+=1
        if i%500==0 or i==len(unique_docs):
            print(f"  parsed {i}/{len(unique_docs)}",flush=True)

    print("[3/5] Auditing explicit-Điều preselection consistency...",flush=True)
    explicit_queries=0
    applicable_pairs=0
    hit_pairs=0
    missed_examples=[]

    # Also collect selected sections for packaging audit.
    sel_q=[]; sel_full=[]; sel_nohdr=[]; sel_body=[]; sel_types=[]
    started=time.perf_counter()

    n_pairs=0
    for qi,q in enumerate(ids,1):
        question=queries[q][0]
        qarts={m.upper() for m in ARTICLE_Q_RE.findall(question)}
        if qarts:
            explicit_queries+=1

        for d in cand[q]:
            n_pairs+=1
            ss=sections[d]
            chosen=preselect_legal_sections(question,ss,count=2)

            for s in chosen:
                sel_q.append(question)
                sel_full.append(s.text)
                sel_nohdr.append(drop_doc_header(s.text))
                sel_body.append(body_only(s.text))
                sel_types.append(s.section_type)

            if qarts and ss:
                available=set()
                for s in ss:
                    m=ARTICLE_H_RE.match(s.heading or "")
                    if m:
                        available.add(m.group(1).upper())
                target=qarts & available
                if target:
                    applicable_pairs+=1
                    chosen_articles=set()
                    for s in chosen:
                        m=ARTICLE_H_RE.match(s.heading or "")
                        if m:
                            chosen_articles.add(m.group(1).upper())
                    ok=bool(target & chosen_articles)
                    hit_pairs+=int(ok)
                    if not ok and len(missed_examples)<100:
                        missed_examples.append({
                            "qid":q,"doc_id":d,
                            "query_articles":sorted(qarts),
                            "available_target_articles":sorted(target),
                            "chosen_headings":[s.heading for s in chosen],
                        })

        if qi%50==0 or qi==len(ids):
            print(
                f"  queries {qi}/{len(ids)} applicable_article_pairs={applicable_pairs} "
                f"selection_hit_rate={(hit_pairs/applicable_pairs if applicable_pairs else 0):.3f}",
                flush=True
            )

    print("[4/5] Batch-tokenizing selected production sections...",flush=True)
    tok=AutoTokenizer.from_pretrained(
        root/"models/jina-reranker-v2-base-multilingual",
        trust_remote_code=True,fix_mistral_regex=True,local_files_only=True
    )
    tok.model_max_length=10**9

    full_len=pair_lengths(tok,sel_q,sel_full)
    nohdr_len=pair_lengths(tok,sel_q,sel_nohdr)
    body_len=pair_lengths(tok,sel_q,sel_body)

    full=np.asarray(full_len)
    nohdr=np.asarray(nohdr_len)
    body=np.asarray(body_len)

    overflow=full>512
    fixed_by_header=overflow & (nohdr<=512)
    residual_nohdr=overflow & (nohdr>512)
    fixed_by_body=overflow & (body<=512)

    report={
        "schema":"manual.section_parser_preselector_packaging_v1",
        "population":{
            "queries":len(ids),"candidate_pairs":n_pairs,"unique_candidate_docs":len(unique_docs),
            "selected_sections":len(sel_full),
        },
        "parser":{
            "docs_structured":docs_structured,
            "docs_fallback_or_full_only":docs_fallback,
            "docs_empty":docs_empty,
            "section_type_counts":dict(type_counts),
            "sections_per_doc":summarize(section_counts),
        },
        "explicit_article_preselection":{
            "queries_with_explicit_article":explicit_queries,
            "candidate_pairs_where_named_article_exists":applicable_pairs,
            "top2_selected_named_article_pairs":hit_pairs,
            "top2_article_selection_recall":float(hit_pairs/applicable_pairs) if applicable_pairs else None,
            "missed_examples":missed_examples,
        },
        "packaging":{
            "full_pair_tokens":summarize(full_len),
            "no_doc_header_pair_tokens":summarize(nohdr_len),
            "body_only_pair_tokens":summarize(body_len),
            "full_over512":int(overflow.sum()),
            "full_over512_rate":float(overflow.mean()),
            "overflow_fixed_by_dropping_doc_header":int(fixed_by_header.sum()),
            "overflow_fixed_by_dropping_doc_header_fraction":float(fixed_by_header.sum()/overflow.sum()) if overflow.sum() else 0.0,
            "overflow_still_over512_without_doc_header":int(residual_nohdr.sum()),
            "overflow_fixed_if_body_only":int(fixed_by_body.sum()),
            "selected_section_type_counts":dict(Counter(sel_types)),
        },
        "public_labels_used":False,
        "cal_gold_used":False,
    }

    out=root/"results/manual/huy_section_parser_preselector_packaging_v1"
    out.mkdir(parents=True,exist_ok=True)
    path=out/"REPORT.json"; path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")

    print("[5/5] RESULT"); print("="*116)
    print(
        f"Parser docs: structured={docs_structured} fallback/full-only={docs_fallback} empty={docs_empty}; "
        f"section types={dict(type_counts)}"
    )
    print(
        f"Explicit Điều: queries={explicit_queries} applicable_pairs={applicable_pairs} "
        f"top2_hit={hit_pairs} recall={(hit_pairs/applicable_pairs if applicable_pairs else 0):.4f}"
    )
    print(
        f"Selected sections >512: {int(overflow.sum())}/{len(full)} ({100*overflow.mean():.2f}%) | "
        f"fixed by drop-header={int(fixed_by_header.sum())} "
        f"({100*(fixed_by_header.sum()/overflow.sum() if overflow.sum() else 0):.2f}% of overflow) | "
        f"still-over-noheader={int(residual_nohdr.sum())}"
    )
    print("Report:",path); print("="*116)


if __name__=="__main__":
    main()
