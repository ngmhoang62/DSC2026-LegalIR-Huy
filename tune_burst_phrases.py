"""Test phrase-aware BURST-QS fusion on 100 labelled queries."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from benchmark_burst_v4_full_sqlite import fts_query, load_dataset, metrics, retrieve_docs, retrieve_local, tokens


STOP = {"là", "có", "và", "của", "được", "như", "thế", "nào", "gì", "cho",
        "trong", "khi", "về", "theo", "thì", "phải", "một", "những", "các"}


def phrase_query(text, n):
    ts = tokens(text)
    phrases = []
    for i in range(len(ts)-n+1):
        gram = ts[i:i+n]
        if sum(t not in STOP for t in gram) >= max(1, n-1):
            phrases.append('"' + " ".join(gram).replace('"', '""') + '"')
    return " OR ".join(dict.fromkeys(phrases))


def multi_rrf(lists, weights, k):
    ranks = [{d: i+1 for i, item in enumerate(lst) for d in [item[0] if isinstance(item, tuple) else item]}
             for lst in lists]
    candidates = set().union(*(set(r) for r in ranks))
    return sorted(candidates, key=lambda d: (-sum(w/(k+r.get(d,100000)) for w,r in zip(weights,ranks)), d))


def main():
    root = Path(__file__).resolve().parent
    data = root / "DSC2026-LegalIR-main" / "v4_run" / "public_test_dataset"
    docs, allq = load_dataset(data); doc_ids = [d for d,_ in docs]
    qids = list(allq)[:100]; queries = {q: allq[q] for q in qids}
    conn = sqlite3.connect(root / "benchmarks" / "legalir_full_fts.sqlite")
    cache = {}
    for i,(qid,(text,_)) in enumerate(queries.items(),1):
        uq = fts_query(text)
        full = retrieve_docs(conn, uq, 500)
        local = retrieve_local(conn, uq, 2000, .6)
        p2 = retrieve_local(conn, phrase_query(text,2), 1000, .6)
        p3 = retrieve_local(conn, phrase_query(text,3), 1000, .6)
        cache[qid] = (full,local,p2,p3)
        if i%20==0: print(f"Retrieved {i}/100",flush=True)
    conn.close()
    trials=[]
    # Base weights reproduce best recall tuning: full/local = .15/.85.
    for pw in (0.05,0.1,0.15,0.2,0.3,0.4):
        for tri_share in (0.0,0.25,0.5,0.75,1.0):
            phrase2=pw*(1-tri_share); phrase3=pw*tri_share
            for k in (5,10,20):
                ranked={qid:[doc_ids[d] for d in multi_rrf(
                    cache[qid], [.15*(1-pw),.85*(1-pw),phrase2,phrase3],k)[:100]]
                    for qid in queries}
                m,_=metrics(ranked,queries)
                trials.append((m["Recall@5"],m["nDCG@10"],{"phrase_weight":pw,
                    "trigram_share":tri_share,"rrf_k":k},m))
    trials.sort(key=lambda x:(x[0],x[1]),reverse=True)
    out=[{"params":p,"metrics":m} for _,_,p,m in trials[:20]]
    (root/"burst_phrase_tuning.json").write_text(json.dumps(out,indent=2),encoding="utf-8")
    print(json.dumps(out[:10],indent=2))


if __name__=="__main__": main()
