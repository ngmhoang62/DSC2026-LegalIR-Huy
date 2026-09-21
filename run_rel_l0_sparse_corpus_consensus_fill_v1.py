#!/usr/bin/env python
"""
HUY DEADLINE — REL_L0 SAFE-SLOT MULTI-SOURCE CONSENSUS FILL V1
===============================================================

CPU-only fallback after the single-source stopword safe-fill audit.

Use the same REL_L0 safe queries, but fill the free rank5 slot only with a
candidate supported by BOTH:
  - DROP_STOPWORDS sparse FTS fused ranking
  - production cap32 full-corpus dense ranking

No D1 retraining. No neural inference. No public labels.

Policies:
  SW_CORPUS_20   candidate must be in sparse top100 AND corpus top20
  SW_CORPUS_50   candidate must be in sparse top100 AND corpus top50
  SW_CORPUS_100  candidate must be in sparse top100 AND corpus top100

Within the intersection, choose by two-source RRF:
  1/(20+sparse_rank) + 1/(20+corpus_rank)

Exclude all original D1 Top5 docs, because the slot is intended to introduce
new recall rather than reorder incumbents.

Gold is opened only after actions are sealed.
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
import sqlite3
import sys
from pathlib import Path

import numpy as np

REL_L0 = -3.0393552780151367
EXPECTED_R = 0.9569444444444444
TOKEN_RE = re.compile(r"\w+", re.UNICODE)
STOPWORDS = {
    "bị","các","có","của","cho","được","để","đến","đối","gì","hay","khi","không",
    "là","làm","một","nào","những","như","phải","ra","sẽ","theo","thì","thế",
    "trong","trên","từ","và","về","với","việc","bao","nhiêu","người","quy","định",
}


def terms(text):
    seen=set(); out=[]
    for t in TOKEN_RE.findall((text or "").lower()):
        if t in STOPWORDS or t in seen:
            continue
        seen.add(t); out.append(t)
    return out


def expr(ts):
    return " OR ".join('"' + t.replace('"','""') + '"' for t in ts)


def locate_db(root):
    for p in [
        root/"benchmarks/legalir_full_fts.sqlite",
        root.parent/"LegalIR/benchmarks/legalir_full_fts.sqlite",
        root/"results/manual/huy_sparse_fts_query_tokenization_v2/legalir_full_fts_audit.sqlite",
    ]:
        if p.is_file():
            return p
    raise FileNotFoundError("No FTS DB found")


def score_file(root,q):
    b=root/"results/manual/huy_d1_cal_ce_rank5_veto_transfer_v2_exactd1/cal_exact_d1_top5_ce_scores_v1"
    for p in (b/f"{q}.json",b/f"{q}.jsonl"):
        if p.is_file(): return p
    return None


def read_scores(p):
    obj=json.loads(p.read_text(encoding="utf-8"))
    if isinstance(obj,dict) and isinstance(obj.get("scores"),dict):
        return {str(k):float(v) for k,v in obj["scores"].items()}
    vals={}
    if isinstance(obj,dict):
        for k,v in obj.items():
            try: vals[str(k)]=float(v)
            except Exception: pass
    if vals:return vals
    raise RuntimeError(f"Unknown CE schema {p}")


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--repo-root",type=Path,required=True)
    args=ap.parse_args()
    root=args.repo_root.resolve(); sys.path.insert(0,str(root))

    from benchmark_burst_v4_full_sqlite import retrieve_docs,retrieve_local,fuse,load_dataset
    from tune_corpus_cap32_fusion import build_training_cap

    print("[1/4] Loading D1/CAL/corpus ranks...",flush=True)
    queries,blocks,ids,current,_,_=build_training_cap(
        root,32,"results/corpus_index/holdout_extended_scores_cap32.pkl",depth=20
    )
    gold={q:set(map(str,queries[q][1])) for q in ids}

    p=root/"results/gemini/huy_d1_legal_section_evidence_v1/S0_S1_CAL_PREDICTIONS.jsonl"
    d1={}
    with p.open("r",encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r=json.loads(line); q=str(r["qid"])
                if q in gold:d1[q]=[str(x) for x in r["s0_top5"]]
    if set(d1)!=set(ids):raise RuntimeError("D1 population mismatch")

    dense=pickle.loads(
        (root/"results/corpus_index/holdout_dense_rank_cap32.pkl").read_bytes()
    )["ranking"]

    print("[2/4] Sealing REL_L0 safe population...",flush=True)
    safe=set()
    for q in ids:
        sf=score_file(root,q)
        if sf is None:continue
        sc=read_scores(sf); top=d1[q]
        if any(d not in sc for d in top):continue
        rel=float(sc[top[4]]-np.median([sc[d] for d in top[:4]]))
        if rel<REL_L0:safe.add(q)
    print(f"  safe={len(safe)}",flush=True)

    print("[3/4] Building stopword sparse ranks and sealing consensus fills...",flush=True)
    docs_all,_=load_dataset(root/"DSC2026-LegalIR-main/v4_run/public_test_dataset")
    doc_ids=[str(d) for d,_ in docs_all]
    db=locate_db(root)
    conn=sqlite3.connect(f"file:{db.resolve().as_posix()}?mode=ro",uri=True)

    policies={k:{q:list(d1[q]) for q in ids} for k in (20,50,100)}
    action_meta={k:[] for k in policies}

    try:
        for i,q in enumerate(ids,1):
            if q not in safe:
                continue
            e=expr(terms(queries[q][0]))
            full=retrieve_docs(conn,e,500)
            local=retrieve_local(conn,e,2000,second_weight=.3)
            fused=fuse(full,local,local_weight=.9,rrf_k=20)
            sw=[doc_ids[d] for d in fused[:100]]
            sr={d:r+1 for r,d in enumerate(sw)}
            cr={d:r+1 for r,d in enumerate(dense[q][:100])}
            forbidden=set(d1[q])

            for depth in policies:
                common=[
                    d for d in sw
                    if d not in forbidden and cr.get(d,10**9)<=depth
                ]
                if not common:
                    policies[depth][q]=d1[q][:4]
                    continue
                best=max(
                    common,
                    key=lambda d:(
                        1/(20+sr[d])+1/(20+cr[d]),
                        -sr[d],
                        -cr[d],
                        d,
                    )
                )
                policies[depth][q]=d1[q][:4]+[best]
                action_meta[depth].append({
                    "qid":q,"challenger":best,
                    "sparse_rank":sr[best],"corpus_rank":cr[best],
                })
            if i%100==0:
                print(f"  processed up to q {i}/{len(ids)}",flush=True)
    finally:
        conn.close()

    print("[4/4] Opening gold for evaluation...",flush=True)
    def metric(pred):
        per={}
        for q in ids:
            per[q]=len(set(pred[q])&gold[q])/len(gold[q])
        return float(np.mean([per[q] for q in ids])),per

    br,bper=metric(d1)
    if abs(br-EXPECTED_R)>1e-12:raise RuntimeError("D1 parity fail")

    results={}
    for depth,pred in policies.items():
        r,per=metric(pred)
        dd=np.asarray([per[q]-bper[q] for q in ids])
        blockd={
            b:float(np.mean([per[q] for q in qs])-np.mean([bper[q] for q in qs]))
            for b,qs in blocks.items()
        }
        results[str(depth)]={
            "recall":r,"delta_recall":r-br,
            "wins":int((dd>1e-12).sum()),
            "losses":int((dd<-1e-12).sum()),
            "ties":int((np.abs(dd)<=1e-12).sum()),
            "filled_actions":len(action_meta[depth]),
            "block_deltas":blockd,
            "strict_promote":bool(
                r>br+1e-12 and int((dd<-1e-12).sum())==0
                and all(v>=-1e-12 for v in blockd.values())
            ),
        }

    out=root/"results/manual/huy_rel_l0_sparse_corpus_consensus_fill_v1"
    out.mkdir(parents=True,exist_ok=True)
    path=out/"REPORT.json"
    path.write_text(json.dumps({
        "schema":"manual.rel_l0_sparse_corpus_consensus_fill_v1",
        "safe_queries":len(safe),
        "baseline_recall":br,
        "results":results,
        "actions":action_meta,
        "construction_gold_free":True,
        "public_labels_used":False,
    },ensure_ascii=False,indent=2),encoding="utf-8")

    print("="*116)
    print(f"D1 R={br:.10f} | safe={len(safe)}")
    for depth,x in results.items():
        print(
            f"SW_CORPUS_{depth:<3s} R={x['recall']:.10f} "
            f"dR={x['delta_recall']:+.10f} "
            f"W/L/T={x['wins']}/{x['losses']}/{x['ties']} "
            f"fills={x['filled_actions']} blocks={x['block_deltas']} "
            f"promote={x['strict_promote']}"
        )
    print("Report:",path);print("="*116)


if __name__=="__main__":
    main()
