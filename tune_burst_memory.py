"""Develop CPU-only BURST Query-Memory and validate without self leakage."""

from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from pathlib import Path

from benchmark_burst_v4_full_sqlite import fts_query, load_dataset, metrics, retrieve_docs, retrieve_local
from tune_burst_phrases import multi_rrf, phrase_query


def build_query_memory(conn, queries):
    conn.execute("DROP TABLE IF EXISTS query_memory_fts")
    conn.execute("DROP TABLE IF EXISTS query_memory_meta")
    conn.execute("CREATE VIRTUAL TABLE query_memory_fts USING fts5(text, tokenize='unicode61')")
    conn.execute("CREATE TABLE query_memory_meta(rowid INTEGER PRIMARY KEY,qid TEXT,answers TEXT)")
    for rowid,(qid,(text,gold)) in enumerate(queries.items(),1):
        conn.execute("INSERT INTO query_memory_fts(rowid,text) VALUES(?,?)",(rowid,text))
        conn.execute("INSERT INTO query_memory_meta(rowid,qid,answers) VALUES(?,?,?)",
                     (rowid,qid,json.dumps(sorted(gold))))
    conn.commit()


def memory_rank(conn, text, exclude_qid=None, depth=100):
    expr=fts_query(text)
    if not expr: return []
    rows=conn.execute(
        "SELECT m.qid,m.answers,-bm25(query_memory_fts) score "
        "FROM query_memory_fts JOIN query_memory_meta m ON m.rowid=query_memory_fts.rowid "
        "WHERE query_memory_fts MATCH ? ORDER BY bm25(query_memory_fts) LIMIT ?",
        (expr,depth+1)).fetchall()
    scores=defaultdict(float); rank=0
    for qid,answers,_ in rows:
        if qid==exclude_qid: continue
        rank+=1
        # Multiple related labelled questions voting for the same law document
        # is stronger than a single accidental lexical neighbor.
        for doc_id in json.loads(answers):
            scores[doc_id]+=1.0/(10+rank)
        if rank>=depth: break
    return sorted(scores,key=lambda d:(-scores[d],d))


def retrieve_burst(conn,doc_ids,qid,text,memory_depth):
    uq=fts_query(text)
    full=retrieve_docs(conn,uq,500)
    local=retrieve_local(conn,uq,2000,.6)
    tri=retrieve_local(conn,phrase_query(text,3),1000,.6)
    mem=memory_rank(conn,text,qid,memory_depth)
    # Convert integer document positions to IDs for a common fusion space.
    return ([doc_ids[d] for d,_ in full], [doc_ids[d] for d,_ in local],
            [doc_ids[d] for d,_ in tri], mem)


def main():
    root=Path(__file__).resolve().parent
    data=root/"DSC2026-LegalIR-main"/"v4_run"/"public_test_dataset"
    docs,allq=load_dataset(data); doc_ids=[d for d,_ in docs]
    conn=sqlite3.connect(root/"benchmarks"/"legalir_full_fts.sqlite")
    build_query_memory(conn,allq)
    # 201-300 tune; 301-400 untouched validation. Previous experiments only
    # used queries 1-200.
    all_ids=list(allq); tune_ids=all_ids[200:300]; val_ids=all_ids[300:400]
    cache={}
    for label,qids in (("tune",tune_ids),("validation",val_ids)):
        for i,qid in enumerate(qids,1):
            text=allq[qid][0]
            cache[qid]=retrieve_burst(conn,doc_ids,qid,text,100)
            if i%20==0: print(f"{label}: {i}/{len(qids)}",flush=True)

    trials=[]
    # Non-memory weights retain the validated phrase configuration ratios.
    for mw in (0.05,0.1,0.15,0.2,0.3,0.4,0.5,0.6):
        for k in (5,10,20,40):
            weights=[.09*(1-mw),.51*(1-mw),.4*(1-mw),mw]
            ranked={q:[x for x in multi_rrf(cache[q],weights,k)[:100]] for q in tune_ids}
            m,_=metrics(ranked,{q:allq[q] for q in tune_ids})
            trials.append((m["Recall@5"],m["nDCG@10"],mw,k,m))
    trials.sort(reverse=True,key=lambda x:(x[0],x[1]))
    _,_,mw,k,tune_m=trials[0]
    weights=[.09*(1-mw),.51*(1-mw),.4*(1-mw),mw]
    val_ranked={q:[x for x in multi_rrf(cache[q],weights,k)[:100]] for q in val_ids}
    val_m,_=metrics(val_ranked,{q:allq[q] for q in val_ids})
    phrase_ranked={q:[x for x in multi_rrf(cache[q][:3],[.09,.51,.4],10)[:100]] for q in val_ids}
    phrase_m,_=metrics(phrase_ranked,{q:allq[q] for q in val_ids})
    report={"best_params":{"memory_weight":mw,"rrf_k":k,"memory_depth":100},
            "tune_201_300":tune_m,"validation_301_400":{
                "BURST_QS_phrase":phrase_m,"BURST_QM":val_m}}
    (root/"burst_memory_validation.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2)); conn.close()


if __name__=="__main__": main()
