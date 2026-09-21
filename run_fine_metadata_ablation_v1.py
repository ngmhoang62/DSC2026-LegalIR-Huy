#!/usr/bin/env python
"""
HUY PIPELINE AUDIT M — FINE-GRAINED DOCTYPE / CITATION ABLATION V1
==================================================================

CPU-only exact D1 LOBO.

Current metadata:
  doctype 12D
    [6 one-hots, pool_share, 5 query-only hints]
  citation 4D
    [cites_another, n_cited_in_pool, is_cited_by_another, n_citing_it]

Important structural facts:
- 6 one-hots sum to 1 => one category is linearly redundant with intercept.
- 5 question hints are constant across all documents of the same query, so
  they cannot directly alter within-query ordering for a linear scorer.
- citation binary/count pairs may be partially redundant.

Preregistered variants:
  CONTROL_METADATA_16D
  DROP_QUERY_HINTS              -> doctype 7D
  DROP_KHAC_REFERENCE           -> doctype 11D
  COMPACT_DOCTYPE               -> 5 one-hot(ref=KHAC dropped)+pool_share = 6D
  CITATION_COUNTS_ONLY          -> 2D citation
  CITATION_BINARY_ONLY          -> 2D citation
  CITATION_OUTGOING_ONLY        -> 2D citation
  CITATION_INCOMING_ONLY        -> 2D citation
  COMPACT_BOTH                  -> compact doctype + citation counts only

No public labels.
"""

from __future__ import annotations

import argparse, json, pickle, sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

EXPECTED_R = 0.9569444444444444
D1_VIEWS = ["base","expanded","jina","dense","corpus"]
EXTRA = {
    "aiteamvn_ft":"results/from_drive/aiteamvn_ft_cv.pkl",
    "jina_ft":"results/from_drive/jina_ft_cv.pkl",
    "title_embed":"results/burst_fresh_block/title_embed_scores.pkl",
}


def loadp(root, rel):
    return pickle.loads((root/rel).read_bytes())


def align(raw, cand, ids, floor=None):
    if isinstance(raw, dict) and isinstance(raw.get("scores"),dict):
        raw=raw["scores"]
    if floor is None:
        vals=[v for q in raw.values() for v in q.values()]
        floor=min(vals) if vals else -1e9
    return {q:{d:float(raw.get(q,{}).get(d,floor)) for d in cand[q]} for q in ids}


def prepare(root):
    from run_burst_expanded_fusion_submission import DocumentStore
    from tune_citation_graph import build_citation_table, citation_features
    from tune_corpus_cap32_fusion import build_training_cap
    from tune_doctype_features import build_type_table, type_features
    from tune_expanded_fusion_selection import ltr_features

    docs=DocumentStore(sorted(
        (root/"DSC2026-LegalIR-main/v4_run/public_test_dataset/selected-contexts")
        .glob("context_*.json")
    ))
    queries,blocks,ids,cand,views,base_scores=build_training_cap(
        root,32,"results/corpus_index/holdout_extended_scores_cap32.pkl",depth=20
    )
    gold={q:set(map(str,queries[q][1])) for q in ids}
    channels={
        **base_scores,
        "vnlegal_lal":align(loadp(root,"results/embedding_finetune/vnlegal_lal_cv_scores.pkl"),cand,ids),
        "crossenc":align(loadp(root,"results/crossenc_fullpool/cv_scores.pkl"),cand,ids,-11.5),
        **{k:align(loadp(root,rel),cand,ids) for k,rel in EXTRA.items()},
    }
    base_rows, groups=ltr_features(views,D1_VIEWS,cand,ids,channels)
    tt=build_type_table(root,docs,ids,cand)
    tr=type_features(cand,tt,queries,ids)
    own,cited=build_citation_table(docs,ids,cand)
    cr=citation_features(cand,own,cited,ids)
    return queries,blocks,ids,cand,gold,base_rows,groups,tr,cr


def select_meta(tr,cr,variant,q):
    t=tr[q]; c=cr[q]
    if variant=="CONTROL_METADATA_16D":
        return np.concatenate([t,c],axis=1)
    if variant=="DROP_QUERY_HINTS":
        return np.concatenate([t[:,:7],c],axis=1)
    if variant=="DROP_KHAC_REFERENCE":
        # one-hot columns 0..5, KHAC is col5
        keep=[0,1,2,3,4,6,7,8,9,10,11]
        return np.concatenate([t[:,keep],c],axis=1)
    if variant=="COMPACT_DOCTYPE":
        # five explicit types, KHAC is reference category; keep pool_share
        keep=[0,1,2,3,4,6]
        return np.concatenate([t[:,keep],c],axis=1)
    if variant=="CITATION_COUNTS_ONLY":
        return np.concatenate([t,c[:,[1,3]]],axis=1)
    if variant=="CITATION_BINARY_ONLY":
        return np.concatenate([t,c[:,[0,2]]],axis=1)
    if variant=="CITATION_OUTGOING_ONLY":
        return np.concatenate([t,c[:,[0,1]]],axis=1)
    if variant=="CITATION_INCOMING_ONLY":
        return np.concatenate([t,c[:,[2,3]]],axis=1)
    if variant=="COMPACT_BOTH":
        keep=[0,1,2,3,4,6]
        return np.concatenate([t[:,keep],c[:,[1,3]]],axis=1)
    raise ValueError(variant)


def run(world,variant):
    queries,blocks,ids,cand,gold,base_rows,groups,tr,cr=world
    rows={}
    for q in ids:
        rows[q]=np.concatenate([base_rows[q],select_meta(tr,cr,variant,q)],axis=1)

    pred={}; perq={}
    for held in sorted(blocks):
        train=sum((blocks[b] for b in blocks if b!=held),[])
        X=np.vstack([rows[q] for q in train])
        y=np.concatenate([[d in gold[q] for d in groups[q]] for q in train]).astype(np.int8)
        sc=StandardScaler().fit(X)
        m=LogisticRegression(C=.15,class_weight="balanced",solver="liblinear",max_iter=3000,random_state=2026)
        m.fit(sc.transform(X),y)
        for q in blocks[held]:
            s=m.decision_function(sc.transform(rows[q]))
            order=np.argsort(-s,kind="stable")
            pred[q]=[groups[q][i] for i in order[:5]]

    for q in ids:
        perq[q]=len(set(pred[q])&gold[q])/len(gold[q])

    return {
        "variant":variant,
        "dim":int(rows[ids[0]].shape[1]),
        "metadata_dim":int(select_meta(tr,cr,variant,ids[0]).shape[1]),
        "recall":float(np.mean([perq[q] for q in ids])),
        "precision":float(np.mean([len(set(pred[q])&gold[q])/5 for q in ids])),
        "blocks":{b:float(np.mean([perq[q] for q in blocks[b]])) for b in sorted(blocks)},
        "pred":pred,"perq":perq,
    }


def cmp(base,x,ids):
    d=np.asarray([x["perq"][q]-base["perq"][q] for q in ids])
    return {
        "dR":x["recall"]-base["recall"],
        "dP":x["precision"]-base["precision"],
        "W":int((d>1e-12).sum()),"L":int((d<-1e-12).sum()),"T":int((np.abs(d)<=1e-12).sum()),
        "blocks":{b:x["blocks"][b]-base["blocks"][b] for b in base["blocks"]},
        "exact_top5":int(sum(x["pred"][q]==base["pred"][q] for q in ids)),
    }


def slim(x):
    return {k:v for k,v in x.items() if k not in ("pred","perq")}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--repo-root",type=Path,required=True)
    args=ap.parse_args()
    root=args.repo_root.resolve(); sys.path.insert(0,str(root))

    print("[1/3] Loading exact D1 feature world...",flush=True)
    w=prepare(root); ids=w[2]
    variants=[
        "CONTROL_METADATA_16D",
        "DROP_QUERY_HINTS",
        "DROP_KHAC_REFERENCE",
        "COMPACT_DOCTYPE",
        "CITATION_COUNTS_ONLY",
        "CITATION_BINARY_ONLY",
        "CITATION_OUTGOING_ONLY",
        "CITATION_INCOMING_ONLY",
        "COMPACT_BOTH",
    ]
    print("[2/3] Running fine-grained metadata ablations...",flush=True)
    rr={}
    for v in variants:
        r=run(w,v); rr[v]=r
        print(f"  {v:<28s} dim={r['dim']:2d} meta={r['metadata_dim']:2d} R={r['recall']:.10f}",flush=True)

    base=rr["CONTROL_METADATA_16D"]
    if abs(base["recall"]-EXPECTED_R)>1e-12 or base["dim"]!=48:
        raise RuntimeError(f"D1 parity failed dim={base['dim']} R={base['recall']}")

    rows=[]
    for v in variants[1:]:
        c=cmp(base,rr[v],ids)
        strict=(
            c["dR"]>=-1e-12
            and all(z>=-1e-12 for z in c["blocks"].values())
            and (c["W"]>c["L"] or (rr[v]["dim"]<48 and c["exact_top5"]==len(ids)))
        )
        rows.append({**slim(rr[v]),"comparison":c,"strict_promote":bool(strict)})

    report={
        "schema":"manual.fine_metadata_ablation_v1",
        "control":slim(base),
        "variants":rows,
        "feature_semantics":{
            "doctype":["LUAT","NGHIDINH","THONGTU","QUYETDINH","CONGVAN","KHAC","pool_share","hint_luat","hint_nghidinh","hint_thongtu","hint_quyetdinh","hint_congvan_or_huongdan"],
            "citation":["cites_another","n_cited_in_pool","is_cited_by_another","n_citing_it"],
        },
        "public_labels_used":False,
    }
    out=root/"results/manual/huy_fine_metadata_ablation_v1"; out.mkdir(parents=True,exist_ok=True)
    path=out/"REPORT.json"; path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")

    print("[3/3] RESULT"); print("="*112)
    print(f"CONTROL 48D R={base['recall']:.10f} P={base['precision']:.10f}")
    for x in rows:
        c=x["comparison"]
        print(
            f"{x['variant']:<28s} dim={x['dim']:2d} dR={c['dR']:+.10f} dP={c['dP']:+.10f} "
            f"W/L/T={c['W']}/{c['L']}/{c['T']} blocks={c['blocks']} strict={x['strict_promote']}"
        )
    print("Report:",path); print("="*112)


if __name__=="__main__":
    main()
