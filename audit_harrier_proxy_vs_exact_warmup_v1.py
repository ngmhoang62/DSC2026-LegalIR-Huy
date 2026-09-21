#!/usr/bin/env python
"""
AUDIT HAR(R)IER PUBLIC PROXY VS SAVED EXACT WARMUP RETRIEVAL
============================================================

Compares the newly generated PUBLIC_HARRIER_GLOBAL_TOP50 rankings against the
saved warmup_finetuned_best.json on identical question texts.

Saved warmup results were produced by the checkpoint's original all-chunk HNSW
pipeline.  The public proxy used one vector per selected-context document
(prefix/first tokenizer window).  If overlap is low, the proxy is not a faithful
reproduction of Harrier retrieval and a zero-gain public result should NOT be
interpreted as evidence against the checkpoint itself.
"""

from __future__ import annotations
import argparse, json, re, unicodedata
from pathlib import Path
import numpy as np

WS=re.compile(r"\s+")

def norm(s):
    return WS.sub(" ", unicodedata.normalize("NFKC", str(s or "")).lower().strip())

def find_named(base:Path,name:str):
    p=base/name
    if p.is_file(): return p
    hits=list(base.rglob(name))
    return hits[0] if hits else None

def qtext(v):
    if isinstance(v,dict):
        return str(v.get("question") or v.get("query") or v.get("text") or "")
    return str(v)

def dedup_parents(results, limit=200):
    out=[]
    for x in results:
        if isinstance(x,dict):
            d=str(x.get("ctx_id", x.get("doc_id", "")))
        else:
            d=str(x)
        if d and d not in out:
            out.append(d)
        if len(out)>=limit: break
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--repo-root",type=Path,required=True)
    ap.add_argument("--model-root",type=Path,required=True)
    args=ap.parse_args()
    root=args.repo_root.resolve()
    model_root=args.model_root.resolve()

    public_path=root/"DSC2026-LegalIR-main/v4_run/public_test_dataset/public-official.json"
    pub=json.loads(public_path.read_text(encoding="utf-8"))

    rank_path=root/"results/manual/huy_public_rel_l0_harrier_safe_fill_v1/PUBLIC_HARRIER_GLOBAL_TOP50.json"
    fresh=json.loads(rank_path.read_text(encoding="utf-8"))

    warm_path=find_named(model_root,"warmup_finetuned_best.json")
    if not warm_path: raise FileNotFoundError("warmup_finetuned_best.json")
    warm=json.loads(warm_path.read_text(encoding="utf-8"))

    # match by normalized content, not qid
    warm_by_text={}
    for q,v in warm.items():
        nt=norm(qtext(v))
        if nt:
            warm_by_text.setdefault(nt,[]).append((str(q),v))

    rows=[]
    for pq,pv in pub.items():
        nt=norm(qtext(pv))
        if nt not in warm_by_text: continue
        if str(pq) not in fresh: continue
        fresh_rank=dedup_parents(fresh[str(pq)].get("results",[]),50)
        for wq,wv in warm_by_text[nt]:
            exact_rank=dedup_parents(wv.get("results",[]),200)
            if not exact_rank or not fresh_rank: continue
            rec={"public_qid":str(pq),"warmup_qid":str(wq)}
            for k in (1,5,10,20,50):
                a=fresh_rank[:k]
                b=exact_rank[:k]
                rec[f"overlap_at_{k}"]=len(set(a)&set(b))/max(1,k)
            rec["fresh_top1_in_exact_top5"]=fresh_rank[0] in set(exact_rank[:5])
            rec["fresh_top1_exact_rank"]=(exact_rank.index(fresh_rank[0])+1
                                          if fresh_rank[0] in exact_rank else None)
            rec["exact_top1_fresh_rank"]=(fresh_rank.index(exact_rank[0])+1
                                          if exact_rank[0] in fresh_rank else None)
            rec["fresh_top5"]=fresh_rank[:5]
            rec["exact_parent_top5"]=exact_rank[:5]
            rows.append(rec)

    if not rows:
        raise RuntimeError("No identical public↔warmup questions matched")

    summary={
        "matched_question_pairs":len(rows),
        "mean_overlap":{
            f"at_{k}":float(np.mean([r[f"overlap_at_{k}"] for r in rows]))
            for k in (1,5,10,20,50)
        },
        "fresh_top1_in_exact_top5_rate":float(np.mean([
            r["fresh_top1_in_exact_top5"] for r in rows
        ])),
        "fresh_top1_exact_rank_median":float(np.median([
            r["fresh_top1_exact_rank"] for r in rows
            if r["fresh_top1_exact_rank"] is not None
        ])) if any(r["fresh_top1_exact_rank"] is not None for r in rows) else None,
        "pairs":rows,
    }

    out=root/"results/manual/huy_harrier_proxy_vs_exact_warmup_audit_v1"
    out.mkdir(parents=True,exist_ok=True)
    rp=out/"REPORT.json"
    rp.write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")

    print("="*110)
    print("Matched identical questions:",len(rows))
    for k in (1,5,10,20,50):
        print(f"Mean dedup-parent Top{k:>2} overlap: {summary['mean_overlap'][f'at_{k}']:.4f}")
    print("Fresh top1 ∈ exact top5 rate:",
          f"{summary['fresh_top1_in_exact_top5_rate']:.4f}")
    print("Median exact rank of fresh top1:",
          summary["fresh_top1_exact_rank_median"])
    print("Report:",rp)
    print("="*110)

if __name__=="__main__":
    main()
