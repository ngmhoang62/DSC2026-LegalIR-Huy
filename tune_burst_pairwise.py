"""Pairwise CPU learning-to-rank for maximizing hits in BURST's fixed top 5."""

from __future__ import annotations

import json, pickle, sqlite3
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from benchmark_burst_v4_full_sqlite import load_dataset, metrics
from tune_burst_memory import build_query_memory
from tune_burst_phrases import multi_rrf
from tune_burst_score_ltr import retrieve, score_features


def fixed_metrics(ranked, queries):
    m,pq=metrics(ranked,queries)
    hits=sum(len(set(ranked[q][:5])&queries[q][1]) for q in queries)
    m["Precision@5"]=hits/(5*len(queries));return m,pq


def pairwise_examples(cache, allq, train_ids, negatives_per_positive=40):
    Xs=[];ys=[]
    base_w=[.063,.357,.28,.30]
    rng=np.random.default_rng(2026)
    for q in train_ids:
        candidates,X=score_features(cache[q]);gold=allq[q][1]
        pos=[i for i,d in enumerate(candidates) if d in gold]
        if not pos:continue
        # Focus on hard negatives ranked highest by the strong existing system.
        base_rank=multi_rrf([[d for d,_ in x] for x in cache[q]],base_w,5)
        hard_ids=[d for d in base_rank[:100] if d not in gold]
        position={d:i for i,d in enumerate(candidates)}
        neg=[position[d] for d in hard_ids if d in position][:negatives_per_positive]
        if not neg:continue
        for pi in pos:
            for ni in neg:
                diff=X[pi]-X[ni]
                # Symmetric samples force score(pos)>score(neg), with no query bias.
                Xs.extend((diff,-diff));ys.extend((1,0))
    order=rng.permutation(len(ys))
    return np.asarray(Xs,np.float32)[order],np.asarray(ys,np.int8)[order]


def rank_model(model,cache,qids):
    ranked={}
    coef=model.coef_[0]
    for q in qids:
        candidates,X=score_features(cache[q])
        score=X@coef
        ranked[q]=[candidates[i] for i in np.argsort(-score)[:100]]
    return ranked


def blend_rankings(a, b, alpha, rrf_k):
    """RRF ensemble: alpha=1 keeps ranking a; alpha=0 keeps ranking b."""
    out={}
    for q in a:
        ra={doc:rank for rank,doc in enumerate(a[q],1)}
        rb={doc:rank for rank,doc in enumerate(b[q],1)}
        docs=set(ra)|set(rb)
        out[q]=sorted(docs,key=lambda doc:(
            -(alpha/(rrf_k+ra.get(doc,100000))
              +(1-alpha)/(rrf_k+rb.get(doc,100000))),doc))[:100]
    return out


def main():
    root=Path(__file__).resolve().parent
    data=root/"DSC2026-LegalIR-main"/"v4_run"/"public_test_dataset"
    docs,allq=load_dataset(data);doc_ids=[d for d,_ in docs];qids=list(allq)
    train_ids=qids[400:700];tune_ids=qids[700:750];val_ids=qids[750:850]
    excluded=set(tune_ids)|set(val_ids)
    memory={q:v for q,v in allq.items() if q not in excluded}
    needed=train_ids+tune_ids+val_ids
    cache_dir=root/"results"/"burst_pairwise";cache_dir.mkdir(parents=True,exist_ok=True)
    cache_path=cache_dir/"retrieval_401_850_exclude_701_850.pkl"
    cache={}
    if cache_path.exists():
        saved=pickle.loads(cache_path.read_bytes())
        if saved.get("qids")==needed:
            cache=saved["cache"]
            print(f"Loaded retrieval cache: {len(cache)} queries",flush=True)
    if not cache:
        conn=sqlite3.connect(root/"benchmarks"/"legalir_full_fts.sqlite");build_query_memory(conn,memory)
        for i,q in enumerate(needed,1):
            cache[q]=retrieve(conn,doc_ids,q,allq[q][0])
            if i%50==0:print(f"Retrieved {i}/{len(needed)}",flush=True)
        conn.close()
        cache_path.write_bytes(pickle.dumps({"qids":needed,"cache":cache},protocol=5))
        print(f"Saved retrieval cache: {cache_path}",flush=True)
    X,y=pairwise_examples(cache,allq,train_ids)
    print(f"Pairwise examples={len(y):,}",flush=True)
    tuneq={q:allq[q] for q in tune_ids};trials=[]
    for C in (.01,.03,.1,.3,1.0,3.0,10.0):
        model=LogisticRegression(C=C,fit_intercept=False,solver="liblinear",max_iter=1000).fit(X,y)
        ranked=rank_model(model,cache,tune_ids);m,_=fixed_metrics(ranked,tuneq)
        trials.append((m["Recall@5"],m["Precision@5"],m["nDCG@10"],C,model,m))
    trials.sort(reverse=True,key=lambda z:(z[0],z[1],z[2]));_,_,_,C,model,tune_m=trials[0]
    # Pointwise ScoreLTR, trained from the identical candidates/features.
    Xs=[];ys=[]
    for q in train_ids:
        cand,F=score_features(cache[q]);Xs.append(F)
        ys.extend(1 if d in allq[q][1] else 0 for d in cand)
    point_X=np.vstack(Xs);point_y=np.asarray(ys,np.int8)
    point_trials=[]
    for point_C in (.03,.1,.2,.5,1.0):
        point=LogisticRegression(C=point_C,class_weight="balanced",max_iter=1000,
                                 solver="liblinear").fit(point_X,point_y)
        ranked=rank_model(point,cache,tune_ids);m,_=fixed_metrics(ranked,tuneq)
        point_trials.append((m["Recall@5"],m["Precision@5"],m["nDCG@10"],point_C,point,m))
    point_trials.sort(reverse=True,key=lambda z:(z[0],z[1],z[2]))
    _,_,_,point_C,point,point_tune_m=point_trials[0]

    # Tune pointwise/pairwise fusion only on the 50-query tuning split.
    tune_pair=rank_model(model,cache,tune_ids);tune_point=rank_model(point,cache,tune_ids)
    blend_trials=[]
    for alpha in np.linspace(0,1,11):
        for rrf_k in (0,5,20,60):
            ranked=blend_rankings(tune_point,tune_pair,float(alpha),rrf_k)
            m,_=fixed_metrics(ranked,tuneq)
            blend_trials.append((m["Recall@5"],m["Precision@5"],m["nDCG@10"],
                                 float(alpha),rrf_k,m))
    blend_trials.sort(reverse=True,key=lambda z:(z[0],z[1],z[2]))
    _,_,_,alpha,rrf_k,blend_tune_m=blend_trials[0]

    valq={q:allq[q] for q in val_ids}
    pair_rank=rank_model(model,cache,val_ids);new,npq=fixed_metrics(pair_rank,valq)
    point_rank=rank_model(point,cache,val_ids);point_m,point_pq=fixed_metrics(point_rank,valq)
    blend_rank=blend_rankings(point_rank,pair_rank,alpha,rrf_k)
    blend_m,blend_pq=fixed_metrics(blend_rank,valq)
    base={q:multi_rrf([[d for d,_ in x] for x in cache[q]],[.063,.357,.28,.30],5)[:100] for q in val_ids}
    old,opq=fixed_metrics(base,valq)
    report={"train":"401-700","C_tune":"701-750","validation":"751-850 excluded from memory",
      "best_pair_C":C,"pair_tune_metrics":tune_m,
      "best_point_C":point_C,"point_tune_metrics":point_tune_m,
      "blend":{"point_alpha":alpha,"rrf_k":rrf_k,"tune_metrics":blend_tune_m},
      "validation":{"BURST_QM":old,"BURST_PairLTR":new,"BURST_ScoreLTR":point_m,
      "BURST_Ensemble":blend_m,
      "pair_vs_qm":{"PairLTR_wins":sum(a>b for a,b in zip(npq,opq)),
      "ties":sum(a==b for a,b in zip(npq,opq)),
      "BURST_QM_wins":sum(a<b for a,b in zip(npq,opq))},
      "ensemble_vs_score":{"Ensemble_wins":sum(a>b for a,b in zip(blend_pq,point_pq)),
      "ties":sum(a==b for a,b in zip(blend_pq,point_pq)),
      "ScoreLTR_wins":sum(a<b for a,b in zip(blend_pq,point_pq))}}}
    (root/"burst_pairwise_validation.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))


if __name__=="__main__":main()
