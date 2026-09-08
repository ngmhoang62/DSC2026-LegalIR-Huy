"""CPU-only legal/lexical feature reranker on the cached BURST candidates."""

from __future__ import annotations

import json
import math
import pickle
import re
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from benchmark_burst_v4_full_sqlite import load_dataset, tokens
from tune_burst_pairwise import fixed_metrics, rank_model
from tune_burst_score_ltr import score_features


STOPWORDS = {
    "bị", "các", "có", "của", "cho", "được", "để", "đến", "đối", "gì",
    "hay", "khi", "không", "là", "làm", "một", "nào", "những", "như",
    "phải", "ra", "sẽ", "theo", "thì", "thế", "trong", "trên", "từ",
    "và", "về", "với", "việc", "bao", "nhiêu",
}
LEGAL_RE = re.compile(
    r"(?:điều|khoản|điểm|mục|chương)\s+[\w.-]+|"
    r"\b\d{1,4}/\d{2,4}/[\w-]+\b",
    re.IGNORECASE,
)


def unique(items):
    return list(dict.fromkeys(items))


def lexical_features(question, candidates, normalized_docs):
    qt=tokens(question)
    uq=unique(qt)
    content=unique(t for t in qt if len(t)>=3 and t not in STOPWORDS)
    numbers=unique(t for t in qt if any(ch.isdigit() for ch in t))
    bigrams=unique(" ".join(qt[i:i+2]) for i in range(max(0,len(qt)-1)))
    trigrams=unique(" ".join(qt[i:i+3]) for i in range(max(0,len(qt)-2)))
    citations=unique(x.lower() for x in LEGAL_RE.findall(question.lower()))

    rows=[]
    for docid in candidates:
        doc=normalized_docs[docid]
        # Python's substring search runs in optimized C and is substantially
        # faster here than a large alternation regex on 50k+-character laws.
        present={t:(" "+t+" ") in doc for t in uq}
        term_hits=sum(present.values())
        content_hits=sum(present[t] for t in content)
        number_hits=sum(present[t] for t in numbers)
        bi_hits=sum((" "+p+" ") in doc for p in bigrams)
        tri_hits=sum((" "+p+" ") in doc for p in trigrams)
        cite_hits=sum(c in doc for c in citations)
        rows.append([
            term_hits/max(len(uq),1),
            content_hits/max(len(content),1),
            number_hits/max(len(numbers),1) if numbers else 0.0,
            float(bool(numbers) and number_hits==len(numbers)),
            bi_hits/max(len(bigrams),1),
            tri_hits/max(len(trigrams),1),
            float(tri_hits>0),
            cite_hits/max(len(citations),1) if citations else 0.0,
            float(bool(citations) and cite_hits==len(citations)),
            math.log1p(len(doc))/15.0,
        ])
    return np.asarray(rows,np.float32)


def enhanced_features(lists, question, normalized_docs, lexical_depth=None):
    candidates,base=score_features(lists)
    if lexical_depth is None:
        lexical=lexical_features(question,candidates,normalized_docs)
    else:
        eligible={doc for source in lists for doc,_ in source[:lexical_depth]}
        selected=[doc for doc in candidates if doc in eligible]
        selected_features=lexical_features(question,selected,normalized_docs)
        selected_map={doc:row for doc,row in zip(selected,selected_features)}
        lexical=np.zeros((len(candidates),10),np.float32)
        for i,doc in enumerate(candidates):
            if doc in selected_map:
                lexical[i]=selected_map[doc]
            else:
                # Document length is cheap and was retained in the validation
                # simulation; only term/phrase scans are restricted to top-N.
                lexical[i,9]=math.log1p(len(normalized_docs[doc]))/15.0
    # Interactions let coverage matter differently for strong local/phrase
    # candidates without requiring a nonlinear/GPU model.
    interactions=np.column_stack((
        lexical[:,1]*base[:,4],   # content coverage x full-document score
        lexical[:,1]*base[:,5],   # content coverage x local-passage score
        lexical[:,5]*base[:,6],   # trigram coverage x phrase score
        lexical[:,2]*base[:,7],   # number coverage x query-memory score
    )).astype(np.float32)
    return candidates,np.hstack((base,lexical,interactions))


def fit_rank(cache, questions, gold, train_ids, C, normalized_docs, enhanced):
    Xs=[];ys=[]
    for i,q in enumerate(train_ids,1):
        if enhanced:
            cand,X=enhanced_features(cache[q],questions[q],normalized_docs)
        else:
            cand,X=score_features(cache[q])
        Xs.append(X);ys.extend(1 if d in gold[q] else 0 for d in cand)
        if enhanced and i%50==0:print(f"Features {i}/{len(train_ids)}",flush=True)
    X=np.vstack(Xs);y=np.asarray(ys,np.int8)
    return LogisticRegression(C=C,class_weight="balanced",max_iter=1000,
                              solver="liblinear").fit(X,y)


def rank_enhanced(model,cache,qids,questions,normalized_docs):
    ranked={}
    for i,q in enumerate(qids,1):
        cand,X=enhanced_features(cache[q],questions[q],normalized_docs)
        score=model.decision_function(X)
        ranked[q]=[cand[j] for j in np.argsort(-score)[:100]]
        if i%50==0:print(f"Ranked enhanced {i}/{len(qids)}",flush=True)
    return ranked


def fit_cached(feature_cache,gold,train_ids,C):
    X=np.vstack([feature_cache[q][1] for q in train_ids])
    y=np.asarray([1 if d in gold[q] else 0 for q in train_ids
                  for d in feature_cache[q][0]],np.int8)
    return LogisticRegression(C=C,class_weight="balanced",max_iter=1000,
                              solver="liblinear").fit(X,y)


def rank_cached(model,feature_cache,qids):
    ranked={}
    for q in qids:
        cand,X=feature_cache[q]
        ranked[q]=[cand[j] for j in np.argsort(-model.decision_function(X))[:100]]
    return ranked


def main():
    root=Path(__file__).resolve().parent
    data=root/"DSC2026-LegalIR-main"/"v4_run"/"public_test_dataset"
    docs,allq=load_dataset(data);qids=list(allq)
    questions={q:v[0] for q,v in allq.items()};gold={q:v[1] for q,v in allq.items()}
    train_ids=qids[400:700];tune_ids=qids[700:750];val_ids=qids[750:850]
    saved=pickle.loads((root/"results"/"burst_pairwise"/
        "retrieval_401_850_exclude_701_850.pkl").read_bytes())
    cache=saved["cache"]
    if saved.get("qids")!=train_ids+tune_ids+val_ids:raise RuntimeError("Cache split mismatch")

    output_dir=root/"results"/"burst_legal_features";output_dir.mkdir(parents=True,exist_ok=True)
    feature_path=output_dir/"features_401_850.pkl"

    # Current ScoreLTR baseline, identical train/tune/validation candidates.
    base_trials=[];tuneq={q:allq[q] for q in tune_ids}
    for C in (.03,.1,.2,.5,1.0):
        model=fit_rank(cache,questions,gold,train_ids,C,None,False)
        ranked=rank_model(model,cache,tune_ids);m,_=fixed_metrics(ranked,tuneq)
        base_trials.append((m["Recall@5"],m["Precision@5"],m["nDCG@10"],C,model,m))
    base_trials.sort(reverse=True,key=lambda x:(x[0],x[1],x[2]))
    _,_,_,base_C,base,base_tune=base_trials[0]

    feature_cache={}
    if feature_path.exists():
        saved_features=pickle.loads(feature_path.read_bytes())
        if saved_features.get("qids")==train_ids+tune_ids+val_ids:
            feature_cache=saved_features["features"]
            print(f"Loaded legal feature cache: {len(feature_cache)} queries",flush=True)
    if not feature_cache:
        print("Normalizing 8,532 documents once",flush=True)
        normalized_docs={docid:" "+" ".join(tokens(text))+" " for docid,text in docs}
        print("Documents normalized",flush=True)
        print("Building legal features once",flush=True)
        for i,q in enumerate(train_ids+tune_ids+val_ids,1):
            feature_cache[q]=enhanced_features(cache[q],questions[q],normalized_docs)
            if i%50==0:print(f"Legal features {i}/{len(train_ids+tune_ids+val_ids)}",flush=True)
        feature_path.write_bytes(pickle.dumps({"qids":train_ids+tune_ids+val_ids,
                                               "features":feature_cache},protocol=5))
        print(f"Saved legal feature cache: {feature_path}",flush=True)
        del normalized_docs

    enhanced_trials=[]
    for C in (.03,.1,.2,.5,1.0):
        model=fit_cached(feature_cache,gold,train_ids,C)
        ranked=rank_cached(model,feature_cache,tune_ids)
        m,_=fixed_metrics(ranked,tuneq)
        enhanced_trials.append((m["Recall@5"],m["Precision@5"],m["nDCG@10"],C,model,m))
    enhanced_trials.sort(reverse=True,key=lambda x:(x[0],x[1],x[2]))
    _,_,_,enh_C,enhanced,enh_tune=enhanced_trials[0]
    (output_dir/"validation_model.pkl").write_bytes(pickle.dumps({
        "C":enh_C,"model":enhanced,"train":"401-700",
        "feature_version":"legal-v1"},protocol=5))

    valq={q:allq[q] for q in val_ids}
    base_rank=rank_model(base,cache,val_ids);base_m,base_pq=fixed_metrics(base_rank,valq)
    enh_rank=rank_cached(enhanced,feature_cache,val_ids)
    enh_m,enh_pq=fixed_metrics(enh_rank,valq)
    report={
        "train":"401-700","tune":"701-750","validation":"751-850 excluded from memory",
        "base":{"C":base_C,"tune":base_tune,"validation":base_m},
        "legal_features":{"C":enh_C,"tune":enh_tune,"validation":enh_m},
        "paired":{"LegalFeature_wins":sum(a>b for a,b in zip(enh_pq,base_pq)),
                  "ties":sum(a==b for a,b in zip(enh_pq,base_pq)),
                  "ScoreLTR_wins":sum(a<b for a,b in zip(enh_pq,base_pq))},
    }
    (root/"burst_legal_features_validation.json").write_text(
        json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=="__main__":main()
