#!/usr/bin/env python
"""
HUY PIPELINE AUDIT N — SPARSE FTS QUERY TOKENIZATION V1
========================================================

CPU-only, existing SQLite FTS index, no model inference/public labels.

Audits whether current sparse retrieval is hurt by feeding every unique question
token (including Vietnamese function words) into FTS5 OR queries.

Fixed variants:
  CURRENT_ALL_UNIQUE
  DROP_STOPWORDS
  CONTENT3_OR_NUMERIC

For each variant evaluate:
  FULL document BM25
  LOCAL chunk BM25 (current second_weight=.3)
  FULL+LOCAL current RRF fusion (local_weight=.9, rrf_k=20)

Metrics:
  Recall@5/20/100 on CAL600
  D1 missed-gold recovery @5/20/100
  current outside-pool gold recovery @100
  query token/posting-term counts

This is an acquisition/tokenization audit; it does not port sparse scores into
D1 LTR.
"""

from __future__ import annotations

import argparse, json, re, sqlite3, sys
from pathlib import Path

import numpy as np

TOKEN_RE = re.compile(r"\w+", re.UNICODE)
STOPWORDS = {
    "bị","các","có","của","cho","được","để","đến","đối","gì","hay","khi","không",
    "là","làm","một","nào","những","như","phải","ra","sẽ","theo","thì","thế",
    "trong","trên","từ","và","về","với","việc","bao","nhiêu","người","quy","định",
}


def toks(text):
    return TOKEN_RE.findall((text or "").lower())


def make_terms(text, mode):
    raw = toks(text)
    seen, out = set(), []
    for t in raw:
        keep = True
        if mode == "DROP_STOPWORDS":
            keep = t not in STOPWORDS
        elif mode == "CONTENT3_OR_NUMERIC":
            keep = (
                (len(t) >= 3 and t not in STOPWORDS)
                or any(c.isdigit() for c in t)
            )
        elif mode != "CURRENT_ALL_UNIQUE":
            raise ValueError(mode)
        if keep and t not in seen:
            seen.add(t)
            out.append(t)
    return out


def expr_from_terms(terms):
    return " OR ".join('"' + t.replace('"','""') + '"' for t in terms)


def recall_at(rank, gold, k):
    return len(set(rank[:k]) & gold) / len(gold)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--repo-root",type=Path,required=True)
    args=ap.parse_args()
    root=args.repo_root.resolve(); sys.path.insert(0,str(root))

    from benchmark_burst_v4_full_sqlite import retrieve_docs, retrieve_local, fuse
    from tune_corpus_cap32_fusion import build_training_cap

    print("[1/5] Loading CAL600 + current D1/current pool...",flush=True)
    queries,blocks,ids,current,_,_=build_training_cap(
        root,32,"results/corpus_index/holdout_extended_scores_cap32.pkl",depth=20
    )
    gold={q:set(map(str,queries[q][1])) for q in ids}

    pred_path=root/"results/gemini/huy_d1_legal_section_evidence_v1/S0_S1_CAL_PREDICTIONS.jsonl"
    d1={}
    with pred_path.open("r",encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r=json.loads(line); q=str(r["qid"])
                if q in gold:
                    d1[q]=[str(x) for x in r["s0_top5"]]
    if set(d1)!=set(ids):
        raise RuntimeError("D1 population mismatch")

    d1_miss={(q,d) for q in ids for d in gold[q] if d not in set(d1[q])}
    outside={(q,d) for q in ids for d in gold[q] if d not in set(current[q])}

    print("[2/5] Opening existing FTS5 database + document-id map...",flush=True)
    db=root/"benchmarks/legalir_full_fts.sqlite"
    if not db.is_file():
        raise FileNotFoundError(db)
    conn=sqlite3.connect(f"file:{db.as_posix()}?mode=ro",uri=True)

    ctx=root/"DSC2026-LegalIR-main/v4_run/public_test_dataset/selected-contexts"
    doc_ids=[]
    for p in sorted(ctx.glob("context_*.json")):
        row=json.loads(p.read_text(encoding="utf-8"))
        doc_ids.append(str(row["id"]))
    if len(doc_ids)!=8532:
        raise RuntimeError(f"doc map size={len(doc_ids)}")

    modes=["CURRENT_ALL_UNIQUE","DROP_STOPWORDS","CONTENT3_OR_NUMERIC"]
    methods=["FULL","LOCAL","FUSED"]
    ranks={m:{meth:{} for meth in methods} for m in modes}
    qterm_stats={m:[] for m in modes}

    print("[3/5] Retrieving 600 queries × 3 tokenization variants...",flush=True)
    try:
        for i,q in enumerate(ids,1):
            question=queries[q][0]
            for mode in modes:
                terms=make_terms(question,mode)
                qterm_stats[mode].append(len(terms))
                expr=expr_from_terms(terms)

                full=retrieve_docs(conn,expr,500)
                local=retrieve_local(conn,expr,2000,second_weight=.3)
                fused=fuse(full,local,local_weight=.9,rrf_k=20)

                ranks[mode]["FULL"][q]=[doc_ids[d] for d,_ in full[:100]]
                ranks[mode]["LOCAL"][q]=[doc_ids[d] for d,_ in local[:100]]
                ranks[mode]["FUSED"][q]=[doc_ids[d] for d in fused[:100]]

            if i%50==0 or i==len(ids):
                print(f"  queries {i}/{len(ids)}",flush=True)
    finally:
        conn.close()

    print("[4/5] Computing retrieval/tokenization diagnostics...",flush=True)
    report={
        "schema":"manual.sparse_fts_query_tokenization_v1",
        "population":{
            "queries":len(ids),
            "d1_missed_gold_occurrences":len(d1_miss),
            "outside_current_pool_gold_occurrences":len(outside),
        },
        "variants":{},
        "public_labels_used":False,
    }

    for mode in modes:
        entry={
            "query_unique_terms":{
                "mean":float(np.mean(qterm_stats[mode])),
                "p50":float(np.percentile(qterm_stats[mode],50)),
                "p90":float(np.percentile(qterm_stats[mode],90)),
                "max":int(max(qterm_stats[mode])),
            },
            "methods":{},
        }
        for meth in methods:
            rr=ranks[mode][meth]
            metrics={}
            for k in (5,20,100):
                metrics[f"recall@{k}"]=float(np.mean([
                    recall_at(rr[q],gold[q],k) for q in ids
                ]))
                recovered={(q,d) for q,d in d1_miss if d in set(rr[q][:k])}
                metrics[f"d1_missed_gold_recovered@{k}"]=len(recovered)
            metrics["outside_current_pool_gold_recovered@100"]=len({
                (q,d) for q,d in outside if d in set(rr[q][:100])
            })
            metrics["blocks_recall@20"]={
                b:float(np.mean([recall_at(rr[q],gold[q],20) for q in qids]))
                for b,qids in blocks.items()
            }
            entry["methods"][meth]=metrics
        report["variants"][mode]=entry

    # Strict tokenization winner only if FUSED R20 improves with no block regression.
    ctrl=report["variants"]["CURRENT_ALL_UNIQUE"]["methods"]["FUSED"]
    comparisons={}
    for mode in modes[1:]:
        x=report["variants"][mode]["methods"]["FUSED"]
        bd={b:x["blocks_recall@20"][b]-ctrl["blocks_recall@20"][b] for b in blocks}
        comparisons[mode]={
            "delta_fused_R5":x["recall@5"]-ctrl["recall@5"],
            "delta_fused_R20":x["recall@20"]-ctrl["recall@20"],
            "delta_fused_R100":x["recall@100"]-ctrl["recall@100"],
            "block_R20_deltas":bd,
            "strict_acquisition_improvement":bool(
                x["recall@20"]>ctrl["recall@20"]+1e-12
                and all(v>=-1e-12 for v in bd.values())
            ),
        }
    report["comparisons_vs_current"]=comparisons

    out=root/"results/manual/huy_sparse_fts_query_tokenization_v1"
    out.mkdir(parents=True,exist_ok=True)
    path=out/"REPORT.json"; path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")

    print("[5/5] RESULT"); print("="*116)
    for mode in modes:
        e=report["variants"][mode]
        f=e["methods"]["FUSED"]
        print(
            f"{mode:<22s} terms={e['query_unique_terms']['mean']:.1f} "
            f"FUSED R5/R20/R100={f['recall@5']:.6f}/{f['recall@20']:.6f}/{f['recall@100']:.6f} "
            f"D1miss@100={f['d1_missed_gold_recovered@100']} "
            f"outside@100={f['outside_current_pool_gold_recovered@100']}"
        )
        if mode!="CURRENT_ALL_UNIQUE":
            print("  vs current:",comparisons[mode])
    print("Report:",path); print("="*116)


if __name__=="__main__":
    main()
