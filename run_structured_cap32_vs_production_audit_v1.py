#!/usr/bin/env python
"""
Compare structure-aware cap32 full-corpus retrieval against production cap32.

Uses ranking caches only; production raw cap32 chunk vectors are not required.
"""

from __future__ import annotations
import argparse,json,pickle,sys
from pathlib import Path
import numpy as np


def rec(queries,rank,ids,k):
    return float(np.mean([
        len(set(rank[q][:k])&set(queries[q][1]))/len(queries[q][1])
        for q in ids
    ]))


def ceil(queries,pre,rank,ids,k):
    return float(np.mean([
        len((set(pre[q])|set(rank[q][:k]))&set(queries[q][1]))
        /len(queries[q][1]) for q in ids
    ]))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--repo-root",type=Path,required=True)
    args=ap.parse_args()
    root=args.repo_root.resolve(); sys.path.insert(0,str(root))

    from tune_expanded_fusion_robust import build_views
    from tune_corpus_cap32_fusion import build_training_cap

    queries,blocks,ids,pre,_=build_views(root,expanded_depth=20)
    _,_,_,current,_,_=build_training_cap(
        root,32,"results/corpus_index/holdout_extended_scores_cap32.pkl",
        depth=20,expanded_depth=20
    )

    prod=pickle.loads(
        (root/"results/corpus_index/holdout_dense_rank_cap32.pkl").read_bytes()
    )["ranking"]
    sp=root/"results/corpus_index/holdout_dense_rank_structured_chunks_cap32.pkl"
    if not sp.is_file(): raise FileNotFoundError(sp)
    struct=pickle.loads(sp.read_bytes())["ranking"]

    current_ceiling=float(np.mean([
        len(set(current[q])&set(queries[q][1]))/len(queries[q][1])
        for q in ids
    ]))
    outside={(q,str(d)) for q in ids for d in queries[q][1] if str(d) not in set(current[q])}

    summary={}
    for name,r in (("production_raw_cap32",prod),("structured_cap32",struct)):
        summary[name]={
            "R5":rec(queries,r,ids,5),
            "R20":rec(queries,r,ids,20),
            "R50":rec(queries,r,ids,50),
            "R100":rec(queries,r,ids,100),
            "ceiling20":ceil(queries,pre,r,ids,20),
            "ceiling50":ceil(queries,pre,r,ids,50),
            "blocks_R20":{b:rec(queries,r,qids,20) for b,qids in blocks.items()},
            "blocks_ceiling20":{b:ceil(queries,pre,r,qids,20) for b,qids in blocks.items()},
            "outside_reached100":len({
                (q,d) for q,d in outside if d in set(r[q][:100])
            }),
        }

    a=summary["production_raw_cap32"]; b=summary["structured_cap32"]
    delta={
        "R20":b["R20"]-a["R20"],
        "ceiling20":b["ceiling20"]-a["ceiling20"],
        "ceiling50":b["ceiling50"]-a["ceiling50"],
        "blocks_R20":{x:b["blocks_R20"][x]-a["blocks_R20"][x] for x in blocks},
        "blocks_ceiling20":{x:b["blocks_ceiling20"][x]-a["blocks_ceiling20"][x] for x in blocks},
    }
    overlap=float(np.mean([
        len(set(prod[q][:20])&set(struct[q][:20]))/20 for q in ids
    ]))

    report={
        "schema":"manual.structured_cap32_vs_production_v1",
        "current_production_union_ceiling":current_ceiling,
        "summary":summary,"delta":delta,"top20_overlap":overlap,
        "verdict":(
            "PROMOTE_STRUCTURE_AWARE_ACQUISITION"
            if delta["ceiling20"]>1e-12 and all(v>=-1e-12 for v in delta["blocks_ceiling20"].values())
            else "KILL_OR_DEFER_STRUCTURE_AWARE_ACQUISITION"
        ),
        "public_labels_used":False,
    }

    out=root/"results/manual/huy_structured_cap32_vs_production_v1"
    out.mkdir(parents=True,exist_ok=True)
    path=out/"REPORT.json"; path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")

    print("RESULT"); print("="*112)
    print(
        f"RAW32    R20={a['R20']:.10f} ceiling20={a['ceiling20']:.10f} "
        f"ceiling50={a['ceiling50']:.10f} outside@100={a['outside_reached100']}"
    )
    print(
        f"STRUCT32 R20={b['R20']:.10f} ({delta['R20']:+.10f}) "
        f"ceiling20={b['ceiling20']:.10f} ({delta['ceiling20']:+.10f}) "
        f"ceiling50={b['ceiling50']:.10f} ({delta['ceiling50']:+.10f}) "
        f"outside@100={b['outside_reached100']}"
    )
    print("Block R20 deltas:",delta["blocks_R20"])
    print("Block ceiling20 deltas:",delta["blocks_ceiling20"])
    print(f"Top20 overlap={overlap:.4f}")
    print("VERDICT:",report["verdict"])
    print("Report:",path); print("="*112)


if __name__=="__main__":
    main()
