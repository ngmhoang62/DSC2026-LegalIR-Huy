"""Train BURST rankers on 1,000 labelled queries with a clean holdout."""

from __future__ import annotations

import json
import pickle
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from xgboost import XGBRanker

from tune_burst_legal_features import rank_cached
from tune_burst_memory import build_query_memory
from tune_burst_pairwise import blend_rankings, fixed_metrics
from tune_burst_phrases import multi_rrf
from tune_burst_score_ltr import retrieve, score_features


def load_metadata(data):
    doc_ids=[p.stem[len("context_"):] for p in sorted((data/"selected-contexts").glob("context_*.json"))]
    raw=json.loads((data/"train.json").read_text(encoding="utf-8"))
    queries={str(q):(x["question"],{str(d) for d in x["answer"]})
             for q,x in raw.items() if x.get("answer")}
    return doc_ids,queries


def matrices(cache,queries,qids):
    features={q:score_features(cache[q]) for q in qids}
    X=np.vstack([features[q][1] for q in qids])
    y=np.asarray([1 if d in queries[q][1] else 0 for q in qids
                  for d in features[q][0]],np.int8)
    groups=[len(features[q][0]) for q in qids]
    return features,X,y,groups


def rank_linear(model,features,qids):
    out={}
    for q in qids:
        candidates,X=features[q];score=model.decision_function(X)
        out[q]=[candidates[i] for i in np.argsort(-score)[:100]]
    return out


def rank_xgb(model,features,qids):
    out={}
    for q in qids:
        candidates,X=features[q];score=model.predict(X)
        out[q]=[candidates[i] for i in np.argsort(-score)[:100]]
    return out


def hard_matrix(cache,features,queries,qids,depth):
    Xs=[];ys=[];weights=[]
    for q in qids:
        candidates,X=features[q];gold=queries[q][1]
        posmap={d:i for i,d in enumerate(candidates)}
        positives=[posmap[d] for d in gold if d in posmap]
        base=multi_rrf([[d for d,_ in source] for source in cache[q]],
                       [.063,.357,.28,.30],5)
        negatives=[posmap[d] for d in base if d not in gold and d in posmap][:depth]
        if not positives or not negatives:continue
        chosen=positives+negatives;Xs.append(X[chosen])
        ys.extend([1]*len(positives)+[0]*len(negatives))
        weights.extend([.5/len(positives)]*len(positives))
        weights.extend([.5/len(negatives)]*len(negatives))
    return np.vstack(Xs),np.asarray(ys,np.int8),np.asarray(weights,np.float32)


def main():
    root=Path(__file__).resolve().parent
    data=root/"DSC2026-LegalIR-main"/"v4_run"/"public_test_dataset"
    doc_ids,queries=load_metadata(data);qids=list(queries)
    train_ids=qids[:700]+qids[850:1150]
    tune_ids=qids[700:750];val_ids=qids[750:850]
    needed=train_ids+tune_ids+val_ids
    excluded=set(tune_ids)|set(val_ids)
    output=root/"results"/"burst_large_ltr";output.mkdir(parents=True,exist_ok=True)
    final_cache=output/"retrieval_train1000_tune50_val100.pkl"
    checkpoint=output/"retrieval.checkpoint.pkl"

    cache={}
    old=pickle.loads((root/"results"/"burst_pairwise"/
        "retrieval_401_850_exclude_701_850.pkl").read_bytes())
    cache.update({q:v for q,v in old["cache"].items() if q in set(needed)})
    if final_cache.exists():
        saved=pickle.loads(final_cache.read_bytes())
        if saved.get("qids")==needed:cache=saved["cache"]
    elif checkpoint.exists():
        saved=pickle.loads(checkpoint.read_bytes())
        if saved.get("qids")==needed:cache.update(saved["cache"])
    print(f"Retrieval cache {len(cache)}/{len(needed)}",flush=True)

    if len(cache)<len(needed):
        memory={q:v for q,v in queries.items() if q not in excluded}
        conn=sqlite3.connect(root/"benchmarks"/"legalir_full_fts.sqlite")
        build_query_memory(conn,memory)
        conn.close()
        missing=[q for q in needed if q not in cache]
        local=threading.local()
        db_path=root/"benchmarks"/"legalir_full_fts.sqlite"
        def retrieve_one(q):
            if not hasattr(local,"conn"):
                local.conn=sqlite3.connect(db_path)
            return q,retrieve(local.conn,doc_ids,q,queries[q][0])
        with ThreadPoolExecutor(max_workers=2) as pool:
            for i,(q,result) in enumerate(pool.map(retrieve_one,missing),1):
                cache[q]=result
                if i%25==0:
                    checkpoint.write_bytes(pickle.dumps({"qids":needed,"cache":cache},protocol=5))
                    print(f"Retrieved new {i}/{len(missing)}; total {len(cache)}/{len(needed)}",flush=True)
        final_cache.write_bytes(pickle.dumps({"qids":needed,"cache":cache},protocol=5))
        checkpoint.unlink(missing_ok=True)
        print(f"Saved {final_cache}",flush=True)

    train_features,trainX,trainy,train_groups=matrices(cache,queries,train_ids)
    tune_features,tuneX,tuney,tune_groups=matrices(cache,queries,tune_ids)
    val_features,_,_,_=matrices(cache,queries,val_ids)
    tuneq={q:queries[q] for q in tune_ids};trials=[]

    for C in (.01,.03,.1,.2,.5,1.0,2.0):
        model=LogisticRegression(C=C,class_weight="balanced",solver="liblinear",
                                 max_iter=1500).fit(trainX,trainy)
        ranked=rank_linear(model,tune_features,tune_ids);m,_=fixed_metrics(ranked,tuneq)
        trials.append((m["Recall@5"],m["Precision@5"],m["nDCG@10"],
                       "logistic",{"C":C},model,m))

    for depth in (40,80,150):
        X,y,w=hard_matrix(cache,train_features,queries,train_ids,depth)
        for C in (.1,.3,1.0,3.0):
            model=LogisticRegression(C=C,solver="liblinear",max_iter=1500).fit(
                X,y,sample_weight=w)
            ranked=rank_linear(model,tune_features,tune_ids);m,_=fixed_metrics(ranked,tuneq)
            trials.append((m["Recall@5"],m["Precision@5"],m["nDCG@10"],
                           "hard_logistic",{"depth":depth,"C":C},model,m))

    xgb_configs=[(2,.03,180,10),(2,.05,140,10),(3,.03,200,10),
                 (3,.05,160,10),(4,.03,180,10),(3,.03,220,20)]
    for depth,rate,trees,pairs in xgb_configs:
        model=XGBRanker(objective="rank:ndcg",eval_metric="ndcg@5",tree_method="hist",
            n_estimators=trees,max_depth=depth,learning_rate=rate,min_child_weight=3,
            subsample=.85,colsample_bytree=.9,reg_lambda=5.0,n_jobs=2,
            lambdarank_pair_method="topk",lambdarank_num_pair_per_sample=pairs,
            random_state=2026)
        model.fit(trainX,trainy,group=train_groups,eval_set=[(tuneX,tuney)],
                  eval_group=[tune_groups],verbose=False)
        ranked=rank_xgb(model,tune_features,tune_ids);m,_=fixed_metrics(ranked,tuneq)
        trials.append((m["Recall@5"],m["Precision@5"],m["nDCG@10"],"xgb",
                       {"depth":depth,"rate":rate,"trees":trees,"pairs":pairs},model,m))

    trials.sort(reverse=True,key=lambda x:(x[0],x[1],x[2]))
    _,_,_,kind,params,best,tune_m=trials[0]
    rank_best=lambda fs,ids:(rank_xgb(best,fs,ids) if kind=="xgb" else rank_linear(best,fs,ids))

    legal_saved=pickle.loads((root/"results"/"burst_legal_features"/
        "features_401_850.pkl").read_bytes())["features"]
    legal_model=pickle.loads((root/"results"/"burst_legal_features"/
        "validation_model.pkl").read_bytes())["model"]
    tune_legal=rank_cached(legal_model,legal_saved,tune_ids)
    val_legal=rank_cached(legal_model,legal_saved,val_ids)
    tune_new=rank_best(tune_features,tune_ids)
    blends=[]
    for alpha in np.linspace(0,1,21):
        for k in (0,5,20,60):
            ranked=blend_rankings(tune_new,tune_legal,float(alpha),k)
            m,_=fixed_metrics(ranked,tuneq)
            blends.append((m["Recall@5"],m["Precision@5"],m["nDCG@10"],float(alpha),k,m))
    blends.sort(reverse=True,key=lambda x:(x[0],x[1],x[2]))
    _,_,_,alpha,k,blend_tune=blends[0]

    valq={q:queries[q] for q in val_ids}
    legal_m,lpq=fixed_metrics(val_legal,valq)
    val_new=rank_best(val_features,val_ids);new_m,npq=fixed_metrics(val_new,valq)
    val_blend=blend_rankings(val_new,val_legal,alpha,k);blend_m,bpq=fixed_metrics(val_blend,valq)
    report={"train_queries":len(train_ids),"train":"1-700 plus 851-1150",
      "tune":"701-750","validation":"751-850 excluded from query memory",
      "best":{"kind":kind,"params":params,"tune":tune_m,"validation":new_m},
      "legal_baseline":legal_m,
      "blend":{"new_alpha":alpha,"rrf_k":k,"tune":blend_tune,"validation":blend_m},
      "new_vs_legal":{"New_wins":sum(a>b for a,b in zip(npq,lpq)),
      "ties":sum(a==b for a,b in zip(npq,lpq)),"Legal_wins":sum(a<b for a,b in zip(npq,lpq))},
      "blend_vs_legal":{"Blend_wins":sum(a>b for a,b in zip(bpq,lpq)),
      "ties":sum(a==b for a,b in zip(bpq,lpq)),"Legal_wins":sum(a<b for a,b in zip(bpq,lpq))},
      "all_tune_trials":[{"kind":x[3],"params":x[4],"metrics":x[6]} for x in trials]}
    (root/"burst_large_ltr_validation.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    (output/"best_model.pkl").write_bytes(pickle.dumps({"kind":kind,"params":params,
        "model":best,"blend_alpha":alpha,"blend_k":k,"version":"large-ltr-v1"},protocol=5))
    print(json.dumps(report,indent=2))


if __name__=="__main__":main()
