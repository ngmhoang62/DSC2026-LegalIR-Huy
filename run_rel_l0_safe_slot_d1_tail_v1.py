#!/usr/bin/env python
"""
HUY DEADLINE — REL_L0 SAFE-SLOT D1 TAIL / DIRECT-REF FILL V1
=============================================================

For REL_L0-safe queries only, replace the known-safe dropped rank5 with:
  R6: D1 full-rank position 6
  R7: D1 full-rank position 7
  R8: D1 full-rank position 8
  DIRECT_REF_THEN_R6:
      first pre-generated DIRECT_REFERENCE_MATCH candidate outside D1 top5,
      otherwise D1 rank6

Why this differs from prior rank6/rank7 rescue:
  prior swaps could evict a gold rank5;
  here actions occur ONLY in the REL_L0-safe population, for which the
  existing CAL contract observed zero gold rank5 removals.

All action policies are sealed before gold evaluation.
No public labels.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

REL_L0 = -3.0393552780151367
EXPECTED_R = 0.9569444444444444
EXPECTED_P = 0.20566666666666666
D1_VIEWS = ["base","expanded","jina","dense","corpus"]
EXTRA = {
    "aiteamvn_ft":"results/from_drive/aiteamvn_ft_cv.pkl",
    "jina_ft":"results/from_drive/jina_ft_cv.pkl",
    "title_embed":"results/burst_fresh_block/title_embed_scores.pkl",
}


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

    from run_burst_expanded_fusion_submission import DocumentStore
    from tune_citation_graph import build_citation_table,citation_features
    from tune_corpus_cap32_fusion import build_training_cap
    from tune_doctype_features import build_type_table,type_features
    from tune_expanded_fusion_selection import ltr_features

    print("[1/5] Loading exact D1 world...",flush=True)
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

    print("[2/5] Sealing REL_L0 safe population...",flush=True)
    safe=set()
    for q in ids:
        sf=score_file(root,q)
        if sf is None: continue
        sc=read_scores(sf); top=d1[q]
        if any(d not in sc for d in top): continue
        rel=float(sc[top[4]]-np.median([sc[d] for d in top[:4]]))
        if rel<REL_L0:safe.add(q)
    print(f"  safe={len(safe)}",flush=True)

    print("[3/5] Recomputing exact D1 full LOBO ranks...",flush=True)
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
        sc=StandardScaler().fit(X)
        m=LogisticRegression(C=.15,class_weight="balanced",solver="liblinear",
                             max_iter=3000,random_state=2026)
        m.fit(sc.transform(X),y)
        for q in blocks[held]:
            s=m.decision_function(sc.transform(rows[q]))
            order=np.argsort(-s,kind="stable")
            fullrank[q]=[groups[q][i] for i in order]

    bad=[q for q in ids if fullrank[q][:5]!=d1[q]]
    if bad: raise RuntimeError(f"D1 parity fail on {len(bad)} qids sample={bad[:5]}")

    print("[4/5] Loading pre-generated legal-reference additions...",flush=True)
    legal_path=root/"results/gemini/huy_d1_query_anchored_legal_ref_expansion_v1/CAL_PER_QUERY_EXPANSION.jsonl"
    direct={}
    if legal_path.is_file():
        with legal_path.open("r",encoding="utf-8") as f:
            for line in f:
                if not line.strip():continue
                r=json.loads(line); q=str(r["qid"])
                direct[q]=[
                    str(x["doc_id"])
                    for x in r.get("addition_details",[])
                    if x.get("addition_type")=="DIRECT_REFERENCE_MATCH"
                ]
        print(f"  loaded direct-ref artifact for {len(direct)} qids",flush=True)
    else:
        print("  direct-ref artifact missing; DIRECT_REF_THEN_R6 will equal R6",flush=True)

    # Seal policies before gold evaluation.
    policies={name:{q:list(d1[q]) for q in ids} for name in
              ("R6","R7","R8","DIRECT_REF_THEN_R6")}
    provenance={name:[] for name in policies}

    for q in ids:
        if q not in safe:continue
        top4=d1[q][:4]; forbidden=set(d1[q])

        for name,pos in (("R6",5),("R7",6),("R8",7)):
            if len(fullrank[q])>pos:
                c=fullrank[q][pos]
                policies[name][q]=top4+[c]
                provenance[name].append({"qid":q,"challenger":c,"source":name})

        c=next((d for d in direct.get(q,[]) if d not in forbidden),None)
        source="DIRECT_REFERENCE_MATCH"
        if c is None and len(fullrank[q])>5:
            c=fullrank[q][5]; source="R6_FALLBACK"
        if c is not None:
            policies["DIRECT_REF_THEN_R6"][q]=top4+[c]
            provenance["DIRECT_REF_THEN_R6"].append(
                {"qid":q,"challenger":c,"source":source}
            )

    print("[5/5] Opening gold and evaluating sealed policies...",flush=True)
    def metrics(pred):
        per={}; rec=[]; prec=[]
        for q in ids:
            h=len(set(pred[q])&gold[q])
            per[q]=h/len(gold[q]); rec.append(per[q]); prec.append(h/len(pred[q]))
        return float(np.mean(rec)),float(np.mean(prec)),per

    br,bp,bper=metrics(d1)
    if abs(br-EXPECTED_R)>1e-12 or abs(bp-EXPECTED_P)>1e-12:
        raise RuntimeError(f"D1 metric parity fail R={br} P={bp}")

    results={}
    for name,pred in policies.items():
        r,p,per=metrics(pred)
        diff=np.asarray([per[q]-bper[q] for q in ids])
        blockd={
            b:float(np.mean([per[q] for q in qs])-np.mean([bper[q] for q in qs]))
            for b,qs in blocks.items()
        }
        wins=[q for q in ids if per[q]>bper[q]+1e-12]
        losses=[q for q in ids if per[q]<bper[q]-1e-12]
        results[name]={
            "recall":r,"precision":p,
            "delta_recall":r-br,
            "wins":len(wins),"losses":len(losses),
            "ties":len(ids)-len(wins)-len(losses),
            "win_qids":wins,"loss_qids":losses,
            "block_deltas":blockd,
            "actions":len(provenance[name]),
            "strict_promote":bool(
                r>br+1e-12 and not losses and
                all(v>=-1e-12 for v in blockd.values())
            ),
        }

    out=root/"results/manual/huy_rel_l0_safe_slot_d1_tail_v1"
    out.mkdir(parents=True,exist_ok=True)
    path=out/"REPORT.json"
    path.write_text(json.dumps({
        "schema":"manual.rel_l0_safe_slot_d1_tail_v1",
        "safe_queries":len(safe),
        "baseline":{"recall":br,"precision":bp},
        "results":results,
        "action_provenance":provenance,
        "construction_gold_free":True,
        "public_labels_used":False,
    },ensure_ascii=False,indent=2),encoding="utf-8")

    print("="*118)
    print(f"D1 R={br:.10f} P={bp:.10f} | REL_L0 safe={len(safe)}")
    for name,x in results.items():
        print(
            f"{name:<19s} R={x['recall']:.10f} dR={x['delta_recall']:+.10f} "
            f"W/L/T={x['wins']}/{x['losses']}/{x['ties']} "
            f"actions={x['actions']} blocks={x['block_deltas']} "
            f"promote={x['strict_promote']}"
        )
    print("Report:",path); print("="*118)


if __name__=="__main__":
    main()
