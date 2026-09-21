#!/usr/bin/env python
"""
HUY DEADLINE — REL_L0 SAFE-SLOT + STOPWORD SPARSE FILL V1
==========================================================

Goal:
Use REL_L0's empirically safe rank5 removals as FREE recall slots.

Baseline D1:
  [d1,d2,d3,d4,d5]

REL_L0-safe query:
  d5 is dropped if CE(d5)-median(CE(d1..d4)) < REL_L0.
  On CAL600 the existing policy has 96 actions and ZERO gold removed.

Instead of returning K=4:
  [d1,d2,d3,d4] + [sparse challenger]

Challenger policies are label-free and preregistered:
  SW_POOL:
    first DROP_STOPWORDS FTS fused doc in CURRENT D1 candidate pool,
    excluding top4 and excluding the dropped d5.
  SW_ANY:
    first DROP_STOPWORDS FTS fused doc from the whole 8,532-doc corpus,
    excluding top4 and dropped d5.
  SW_POOL_RANK20:
    first SW doc among current D1 candidate docs whose D1 full-ranker rank
    is 6..20.  (Requires recomputing exact D1 full ranks.)

Gold is used only after all actions are sealed.

No public labels.
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
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

REL_L0 = -3.0393552780151367
EXPECTED_R = 0.9569444444444444
EXPECTED_P = 0.20566666666666666
D1_VIEWS = ["base","expanded","jina","dense","corpus"]
TOKEN_RE = re.compile(r"\w+", re.UNICODE)
STOPWORDS = {
    "bị","các","có","của","cho","được","để","đến","đối","gì","hay","khi","không",
    "là","làm","một","nào","những","như","phải","ra","sẽ","theo","thì","thế",
    "trong","trên","từ","và","về","với","việc","bao","nhiêu","người","quy","định",
}


def terms(text):
    out=[]; seen=set()
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
        if p.is_file(): return p
    raise FileNotFoundError("No FTS DB found")


def score_file(root,q):
    b=root/"results/manual/huy_d1_cal_ce_rank5_veto_transfer_v2_exactd1/cal_exact_d1_top5_ce_scores_v1"
    for p in (b/f"{q}.json", b/f"{q}.jsonl"):
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


def loadp(root,rel):
    obj=pickle.loads((root/rel).read_bytes())
    if isinstance(obj,dict) and isinstance(obj.get("scores"),dict):
        return obj["scores"]
    return obj


def align(raw,cand,ids,floor=None):
    if floor is None:
        vals=[v for q in raw.values() for v in q.values()]
        floor=min(vals) if vals else -1e9
    return {q:{d:float(raw.get(q,{}).get(d,floor)) for d in cand[q]} for q in ids}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--repo-root",type=Path,required=True)
    args=ap.parse_args()
    root=args.repo_root.resolve(); sys.path.insert(0,str(root))

    from benchmark_burst_v4_full_sqlite import retrieve_docs,retrieve_local,fuse,load_dataset
    from run_burst_expanded_fusion_submission import DocumentStore
    from tune_citation_graph import build_citation_table,citation_features
    from tune_corpus_cap32_fusion import build_training_cap
    from tune_doctype_features import build_type_table,type_features
    from tune_expanded_fusion_selection import ltr_features

    print("[1/5] Loading exact D1 + candidate pool...",flush=True)
    queries,blocks,ids,cand,views,base_scores=build_training_cap(
        root,32,"results/corpus_index/holdout_extended_scores_cap32.pkl",depth=20
    )
    gold={q:set(map(str,queries[q][1])) for q in ids}

    pred_path=root/"results/gemini/huy_d1_legal_section_evidence_v1/S0_S1_CAL_PREDICTIONS.jsonl"
    d1={}
    with pred_path.open("r",encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r=json.loads(line); q=str(r["qid"])
                if q in gold:d1[q]=[str(x) for x in r["s0_top5"]]
    if set(d1)!=set(ids):raise RuntimeError("D1 population mismatch")

    print("[2/5] Sealing REL_L0 safe queries...",flush=True)
    safe={}
    abstain=[]
    for q in ids:
        sf=score_file(root,q)
        if sf is None:
            abstain.append(q); continue
        sc=read_scores(sf); top=d1[q]
        if any(d not in sc for d in top):
            abstain.append(q); continue
        rel=float(sc[top[4]]-np.median([sc[d] for d in top[:4]]))
        if rel<REL_L0:
            safe[q]={"dropped":top[4],"rel":rel}

    print(f"  safe={len(safe)} abstain={len(abstain)}",flush=True)

    print("[3/5] Building DROP_STOPWORDS sparse fused rankings...",flush=True)
    db=locate_db(root)
    docs_all,_=load_dataset(root/"DSC2026-LegalIR-main/v4_run/public_test_dataset")
    doc_ids=[str(d) for d,_ in docs_all]
    conn=sqlite3.connect(f"file:{db.resolve().as_posix()}?mode=ro",uri=True)
    sw={}
    try:
        for i,q in enumerate(ids,1):
            e=expr(terms(queries[q][0]))
            full=retrieve_docs(conn,e,500)
            local=retrieve_local(conn,e,2000,second_weight=.3)
            fused=fuse(full,local,local_weight=.9,rrf_k=20)
            sw[q]=[doc_ids[d] for d in fused[:200]]
            if i%100==0: print(f"  sparse {i}/{len(ids)}",flush=True)
    finally:
        conn.close()

    # Recompute exact D1 full ranks only for the conservative rank6..20 policy.
    print("[4/5] Recomputing exact D1 full ranks for conservative selector...",flush=True)
    EXTRA={
        "aiteamvn_ft":"results/from_drive/aiteamvn_ft_cv.pkl",
        "jina_ft":"results/from_drive/jina_ft_cv.pkl",
        "title_embed":"results/burst_fresh_block/title_embed_scores.pkl",
    }
    channels={
        **base_scores,
        "vnlegal_lal":align(loadp(root,"results/embedding_finetune/vnlegal_lal_cv_scores.pkl"),cand,ids),
        "crossenc":align(loadp(root,"results/crossenc_fullpool/cv_scores.pkl"),cand,ids,-11.5),
        **{k:align(loadp(root,v),cand,ids) for k,v in EXTRA.items()},
    }
    docs=DocumentStore(sorted(
        (root/"DSC2026-LegalIR-main/v4_run/public_test_dataset/selected-contexts").glob("context_*.json")
    ))
    tt=build_type_table(root,docs,ids,cand); tr=type_features(cand,tt,queries,ids)
    own,cited=build_citation_table(docs,ids,cand); cr=citation_features(cand,own,cited,ids)
    rows,groups=ltr_features(views,D1_VIEWS,cand,ids,channels)
    for q in ids: rows[q]=np.concatenate([rows[q],tr[q],cr[q]],axis=1)

    fullrank={}
    for held in sorted(blocks):
        train=sum((blocks[b] for b in blocks if b!=held),[])
        X=np.vstack([rows[q] for q in train])
        y=np.concatenate([[d in gold[q] for d in groups[q]] for q in train]).astype(np.int8)
        scaler=StandardScaler().fit(X)
        m=LogisticRegression(C=.15,class_weight="balanced",solver="liblinear",max_iter=3000,random_state=2026)
        m.fit(scaler.transform(X),y)
        for q in blocks[held]:
            s=m.decision_function(scaler.transform(rows[q]))
            order=np.argsort(-s,kind="stable")
            fullrank[q]=[groups[q][i] for i in order]
    if any(fullrank[q][:5]!=d1[q] for q in ids):
        bad=[q for q in ids if fullrank[q][:5]!=d1[q]]
        raise RuntimeError(f"D1 full-rank parity failed on {bad[:5]}")

    # Seal candidates WITHOUT looking at gold.
    policies={"SW_ANY":{},"SW_POOL":{},"SW_POOL_RANK20":{}}
    for q in ids:
        for name in policies:
            policies[name][q]=list(d1[q])
        if q not in safe:continue

        top4=d1[q][:4]; dropped=d1[q][4]
        forbidden=set(top4+[dropped])
        pool=set(cand[q])

        any_c=next((d for d in sw[q] if d not in forbidden),None)
        pool_c=next((d for d in sw[q] if d not in forbidden and d in pool),None)

        d1rank={d:i+1 for i,d in enumerate(fullrank[q])}
        rank20_c=next((
            d for d in sw[q]
            if d not in forbidden and d in pool and 6<=d1rank.get(d,10**9)<=20
        ),None)

        for name,c in [("SW_ANY",any_c),("SW_POOL",pool_c),("SW_POOL_RANK20",rank20_c)]:
            if c is not None:
                policies[name][q]=top4+[c]
            else:
                policies[name][q]=top4  # honest abstain K4 if no challenger

    print("[5/5] Opening gold and evaluating sealed fills...",flush=True)
    def metrics(pred):
        rec=[]; prec=[]; per={}
        for q in ids:
            h=len(set(pred[q])&gold[q])
            r=h/len(gold[q]); rec.append(r); prec.append(h/len(pred[q])); per[q]=r
        return float(np.mean(rec)),float(np.mean(prec)),per

    br,bp,bper=metrics(d1)
    if abs(br-EXPECTED_R)>1e-12 or abs(bp-EXPECTED_P)>1e-12:
        raise RuntimeError("D1 metric parity fail")

    results={}
    for name,pred in policies.items():
        r,p,per=metrics(pred)
        delta=np.asarray([per[q]-bper[q] for q in ids])
        actions=sum(pred[q]!=d1[q] for q in ids)
        fills=sum(len(pred[q])==5 and pred[q]!=d1[q] for q in ids)
        blockd={
            b:float(np.mean([per[q] for q in qs])-np.mean([bper[q] for q in qs]))
            for b,qs in blocks.items()
        }
        result={
            "recall":r,"precision_variableK":p,
            "delta_recall":r-br,
            "wins":int((delta>1e-12).sum()),
            "losses":int((delta<-1e-12).sum()),
            "ties":int((np.abs(delta)<=1e-12).sum()),
            "actions":actions,"fills":fills,
            "block_deltas":blockd,
            "strict_promote":bool(
                r>br+1e-12 and int((delta<-1e-12).sum())==0
                and all(v>=-1e-12 for v in blockd.values())
            ),
        }
        results[name]=result

    out=root/"results/manual/huy_rel_l0_stopword_safe_fill_v1"
    out.mkdir(parents=True,exist_ok=True)
    path=out/"REPORT.json"
    path.write_text(json.dumps({
        "schema":"manual.rel_l0_stopword_safe_fill_v1",
        "rel_l0":REL_L0,
        "safe_queries":len(safe),
        "abstained_ce_queries":abstain,
        "baseline":{"recall":br,"precision":bp},
        "results":results,
        "public_labels_used":False,
        "construction_gold_free":True,
    },ensure_ascii=False,indent=2),encoding="utf-8")

    print("="*116)
    print(f"D1 R={br:.10f} P={bp:.10f} | REL_L0 safe={len(safe)}")
    for name,x in results.items():
        print(
            f"{name:<16s} R={x['recall']:.10f} dR={x['delta_recall']:+.10f} "
            f"W/L/T={x['wins']}/{x['losses']}/{x['ties']} "
            f"actions={x['actions']} fills={x['fills']} blocks={x['block_deltas']} "
            f"promote={x['strict_promote']}"
        )
    print("Report:",path); print("="*116)


if __name__=="__main__":
    main()
