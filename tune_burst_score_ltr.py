"""Score-aware BURST-LTR with a genuinely unseen 701-800 holdout."""

from __future__ import annotations

import json, sqlite3
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from benchmark_burst_v4_full_sqlite import fts_query, load_dataset, metrics, retrieve_docs, retrieve_local
from tune_burst_memory import build_query_memory
from tune_burst_phrases import multi_rrf, phrase_query


def memory_scores(conn,text,exclude_qid=None,depth=100):
    expr=fts_query(text)
    if not expr:return []
    rows=conn.execute("SELECT m.qid,m.answers,-bm25(query_memory_fts) score FROM query_memory_fts "
        "JOIN query_memory_meta m ON m.rowid=query_memory_fts.rowid WHERE query_memory_fts MATCH ? "
        "ORDER BY bm25(query_memory_fts) LIMIT ?",(expr,depth+1)).fetchall()
    scores=defaultdict(float); rank=0
    for qid,answers,score in rows:
        if qid==exclude_qid:continue
        rank+=1
        # combine absolute lexical similarity and reciprocal neighbor rank
        vote=max(float(score),0.0)/(10+rank)
        for d in json.loads(answers):scores[d]+=vote
        if rank>=depth:break
    return sorted(scores.items(),key=lambda x:(-x[1],x[0]))


def retrieve(conn,doc_ids,qid,text):
    uq=fts_query(text)
    full=[(doc_ids[d],s) for d,s in retrieve_docs(conn,uq,500)]
    local=[(doc_ids[d],s) for d,s in retrieve_local(conn,uq,2000,.6)]
    tri=[(doc_ids[d],s) for d,s in retrieve_local(conn,phrase_query(text,3),1000,.6)]
    mem=memory_scores(conn,text,qid,100)
    return full,local,tri,mem


def score_features(lists,max_each=150):
    rankmaps=[]; scoremaps=[]
    for lst in lists:
        rankmaps.append({d:i+1 for i,(d,_) in enumerate(lst)})
        top=max((s for _,s in lst),default=0.0)
        scoremaps.append({d:(s/top if top>0 else 0.0) for d,s in lst})
    candidates=[]; seen=set()
    for lst in lists:
        for d,_ in lst[:max_each]:
            if d not in seen:seen.add(d);candidates.append(d)
    X=[]
    for d in candidates:
        rs=np.asarray([m.get(d,10000) for m in rankmaps],np.float32)
        present=(rs<10000).astype(np.float32)
        rr=np.where(present,1/(5+rs),0)
        sc=np.asarray([m.get(d,0.0) for m in scoremaps],np.float32)
        X.append(np.concatenate([rr,sc,present,[present.sum(),rr.sum(),sc.sum(),
            rr.max(),sc.max(),present[1]*present[2],present[2]*present[3],
            present[1]*present[3],sc[1]*sc[2],sc[2]*sc[3],sc[1]*sc[3]]]))
    return candidates,np.asarray(X,np.float32)


def fixed_metrics(ranked,queries):
    m,pq=metrics(ranked,queries)
    hits=sum(len(set(ranked[q][:5])&queries[q][1]) for q in queries)
    m["Precision@5"]=hits/(5*len(queries));return m,pq


def main():
    root=Path(__file__).resolve().parent
    data=root/"DSC2026-LegalIR-main"/"v4_run"/"public_test_dataset"
    docs,allq=load_dataset(data);doc_ids=[d for d,_ in docs];qids=list(allq)
    # 300 queries are sufficient for the linear ranker and keep fitting within
    # RAM; the validation block remains completely unseen.
    train_ids=qids[400:700];val_ids=qids[700:800];valset=set(val_ids)
    # Realistic validation: the whole holdout is absent from labelled memory.
    memory={q:v for q,v in allq.items() if q not in valset}
    conn=sqlite3.connect(root/"benchmarks"/"legalir_full_fts.sqlite");build_query_memory(conn,memory)
    cache={}
    for i,q in enumerate(train_ids+val_ids,1):
        cache[q]=retrieve(conn,doc_ids,q,allq[q][0])
        if i%50==0:print(f"Retrieved {i}/800",flush=True)
    conn.close()
    Xs=[];ys=[]
    for q in train_ids:
        cand,X=score_features(cache[q]);gold=allq[q][1]
        Xs.append(X);ys.extend(1 if d in gold else 0 for d in cand)
    X=np.vstack(Xs);y=np.asarray(ys);print(f"Examples={len(y):,}, positives={y.sum()}",flush=True)
    model=LogisticRegression(C=.2,class_weight="balanced",max_iter=1000,
                             solver="liblinear").fit(X,y)
    valq={q:allq[q] for q in val_ids};ranked={}
    for q in val_ids:
        cand,F=score_features(cache[q]);ranked[q]=[cand[i] for i in np.argsort(-model.predict_proba(F)[:,1])[:100]]
    new,npq=fixed_metrics(ranked,valq)
    base={q:multi_rrf([[d for d,_ in x] for x in cache[q]],[.063,.357,.28,.30],5)[:100] for q in val_ids}
    old,opq=fixed_metrics(base,valq)
    report={"train":"401-700","validation":"701-800 excluded entirely from query memory",
      "BURST_QM":old,"BURST_ScoreLTR":new,"paired":{
       "ScoreLTR_wins":sum(a>b for a,b in zip(npq,opq)),"ties":sum(a==b for a,b in zip(npq,opq)),
       "BURST_QM_wins":sum(a<b for a,b in zip(npq,opq))}}
    (root/"burst_score_ltr_validation.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))


if __name__=="__main__":main()
