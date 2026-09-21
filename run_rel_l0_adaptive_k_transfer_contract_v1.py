#!/usr/bin/env python
"""
HUY PIPELINE AUDIT Q — REL_L0 ADAPTIVE-K TRANSFER CONTRACT V1
==============================================================

CPU-only final audit of the production adaptive-K arm.

Recomputes CAL behavior from:
  exact D1 Top5 + frozen per-query BGE CE score caches + REL_L0 threshold.

Audits public materialization from:
  exact public D1 champion + submission_REL_L0.zip.

Scientific contract:
  REL_L0 = -3.0393552780151367
  drop rank5 iff CE(rank5) - median(CE(rank1..4)) < REL_L0
  never reorder top1..4, never add docs, K in {4,5} only.

No public labels are accessed. Optional leaderboard numbers supplied on the
command line are recorded as external observations, not recomputed.
"""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import numpy as np

REL_L0 = -3.0393552780151367


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def score_file_candidates(root, q):
    bases = [
        root / "results/manual/huy_d1_cal_ce_rank5_veto_transfer_v2_exactd1/cal_exact_d1_top5_ce_scores_v1",
        root / "results/manual/huy_d1_cal_ce_rank5_veto_transfer_v2_exactd1/ce_scores",
    ]
    out=[]
    for b in bases:
        out += [
            b / f"{q}.json",
            b / f"{q}.jsonl",
        ]
    return out


def read_scores(path):
    obj=load_json(path)
    if isinstance(obj, dict) and isinstance(obj.get("scores"), dict):
        return {str(k):float(v) for k,v in obj["scores"].items()}
    if isinstance(obj, dict):
        # tolerate direct doc->score map
        vals={}
        for k,v in obj.items():
            try: vals[str(k)]=float(v)
            except Exception: pass
        if vals: return vals
    raise RuntimeError(f"Unknown CE score file schema: {path}")


def load_cal(root):
    import sys
    sys.path.insert(0,str(root))
    from tune_corpus_cap32_fusion import build_training_cap
    queries,blocks,ids,_,_,_=build_training_cap(
        root,32,"results/corpus_index/holdout_extended_scores_cap32.pkl",depth=20
    )
    gold={q:set(map(str,queries[q][1])) for q in ids}

    p=root/"results/gemini/huy_d1_legal_section_evidence_v1/S0_S1_CAL_PREDICTIONS.jsonl"
    d1={}
    with p.open("r",encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r=json.loads(line); q=str(r["qid"])
                if q in gold: d1[q]=[str(x) for x in r["s0_top5"]]
    if set(d1)!=set(ids): raise RuntimeError("CAL D1 population mismatch")
    return ids,blocks,gold,d1,p


def metrics(pred,gold,ids):
    rec=[]; prec=[]
    for q in ids:
        p=pred[q]
        h=len(set(p)&gold[q])
        rec.append(h/len(gold[q]))
        prec.append(h/len(p))
    return float(np.mean(rec)),float(np.mean(prec))


def load_zip_submission(path):
    with zipfile.ZipFile(path,"r") as zf:
        names=zf.namelist()
        if "submission.json" not in names:
            raise RuntimeError(f"submission.json missing from {path}: {names}")
        return json.loads(zf.read("submission.json").decode("utf-8"))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--repo-root",type=Path,required=True)
    ap.add_argument("--public-d1-recall",type=float,default=None)
    ap.add_argument("--public-d1-precision",type=float,default=None)
    ap.add_argument("--public-rel-l0-recall",type=float,default=None)
    ap.add_argument("--public-rel-l0-precision",type=float,default=None)
    args=ap.parse_args()

    root=args.repo_root.resolve()

    print("[1/5] Recomputing exact CAL REL_L0 actions...",flush=True)
    ids,blocks,gold,d1,d1_path=load_cal(root)
    candidate={}
    rows=[]
    missing=[]

    for q in ids:
        sf=next((p for p in score_file_candidates(root,q) if p.is_file()),None)
        if sf is None:
            missing.append(q); candidate[q]=list(d1[q]); continue
        scores=read_scores(sf)
        top=d1[q]
        if any(d not in scores for d in top):
            missing.append(q); candidate[q]=list(top); continue
        med=float(np.median([scores[d] for d in top[:4]]))
        rel=float(scores[top[4]]-med)
        drop=rel<REL_L0
        candidate[q]=top[:4] if drop else list(top)
        rows.append({
            "qid":q,"rank5":top[4],
            "rank5_score":scores[top[4]],
            "median_top4":med,"rel":rel,"drop":bool(drop),
            "rank5_is_gold":bool(top[4] in gold[q]),
        })

    base_r,base_p=metrics(d1,gold,ids)
    cand_r,cand_p=metrics(candidate,gold,ids)
    actions=sum(r["drop"] for r in rows)
    gold_removed=sum(r["drop"] and r["rank5_is_gold"] for r in rows)

    block_metrics={}
    for b,qids in blocks.items():
        br,bp=metrics({q:d1[q] for q in qids},gold,qids)
        cr,cp=metrics({q:candidate[q] for q in qids},gold,qids)
        block_metrics[b]={
            "base_recall":br,"candidate_recall":cr,"delta_recall":cr-br,
            "base_precision":bp,"candidate_precision":cp,"delta_precision":cp-bp,
            "actions":sum(candidate[q]!=d1[q] for q in qids),
        }

    print(
        f"  CAL actions={actions} gold_removed={gold_removed} "
        f"R={base_r:.10f}->{cand_r:.10f} P={base_p:.10f}->{cand_p:.10f}",
        flush=True
    )

    print("[2/5] Auditing exact public REL_L0 materialization...",flush=True)
    pub_d1_path=root/"results/gemini/huy_vnlegal_rank_ablation_v1/CANDIDATE_D1_VNLEGAL_SCORE_ONLY.json"
    pub_zip=root/"results/manual/huy_public_d1_ce_rank5_veto_v1/REL_L0/submission_REL_L0.zip"
    if not pub_d1_path.is_file(): raise FileNotFoundError(pub_d1_path)
    if not pub_zip.is_file(): raise FileNotFoundError(pub_zip)

    pub_d1_raw=load_json(pub_d1_path)
    pub_d1={str(q):[str(x) for x in row["answer"]] for q,row in pub_d1_raw.items()}
    pub_rel_raw=load_zip_submission(pub_zip)
    pub_rel={str(q):[str(x) for x in row["answer"]] for q,row in pub_rel_raw.items()}

    if set(pub_d1)!=set(pub_rel) or len(pub_d1)!=1000:
        raise RuntimeError("Public D1/REL_L0 population mismatch")

    violations=[]
    public_actions=[]
    k_counts={}
    for q in pub_d1:
        a=pub_d1[q]; b=pub_rel[q]
        k_counts[len(b)]=k_counts.get(len(b),0)+1
        if b==a:
            continue
        public_actions.append(q)
        ok=(
            len(a)==5 and len(b)==4
            and b==a[:4]
            and len(set(b))==4
        )
        if not ok:
            violations.append({"qid":q,"d1":a,"rel_l0":b})

    print(
        f"  public actions={len(public_actions)} "
        f"rate={len(public_actions)/len(pub_d1):.3%} "
        f"K_counts={k_counts} violations={len(violations)}",
        flush=True
    )

    print("[3/5] Checking distribution shift of action rate / K...",flush=True)
    cal_action_rate=actions/len(ids)
    public_action_rate=len(public_actions)/len(pub_d1)
    action_rate_ratio=public_action_rate/cal_action_rate if cal_action_rate else None

    print("[4/5] Sealing transfer interpretation...",flush=True)
    external={}
    if None not in (
        args.public_d1_recall,args.public_d1_precision,
        args.public_rel_l0_recall,args.public_rel_l0_precision
    ):
        external={
            "source":"USER_SUPPLIED_LEADERBOARD_OBSERVATION",
            "d1":{"recall":args.public_d1_recall,"precision":args.public_d1_precision},
            "rel_l0":{"recall":args.public_rel_l0_recall,"precision":args.public_rel_l0_precision},
            "delta":{
                "recall":args.public_rel_l0_recall-args.public_d1_recall,
                "precision":args.public_rel_l0_precision-args.public_d1_precision,
            },
        }

    transfer_pass=(
        not missing
        and gold_removed==0
        and abs(cand_r-base_r)<1e-12
        and not violations
        and set(k_counts)<= {4,5}
        and len(public_actions)>0
    )
    if external:
        transfer_pass = transfer_pass and (
            abs(external["delta"]["recall"])<1e-12
            and external["delta"]["precision"]>0
        )

    report={
        "schema":"manual.rel_l0_adaptive_k_transfer_contract_v1",
        "policy":{
            "threshold":REL_L0,
            "rule":"drop rank5 iff CE(rank5)-median(CE(rank1..4)) < REL_L0",
            "allowed_K":[4,5],
            "reordering_allowed":False,
            "additions_allowed":False,
        },
        "cal":{
            "queries":len(ids),"scorable_rows":len(rows),"missing_score_qids":missing,
            "actions":actions,"action_rate":cal_action_rate,
            "gold_rank5_removed":gold_removed,
            "base_recall":base_r,"candidate_recall":cand_r,
            "base_precision_macro_variableK":base_p,
            "candidate_precision_macro_variableK":cand_p,
            "blocks":block_metrics,
        },
        "public_materialization":{
            "queries":len(pub_d1),"actions":len(public_actions),
            "action_rate":public_action_rate,"K_counts":k_counts,
            "contract_violations":violations,
            "action_qids":public_actions,
        },
        "shift":{
            "public_minus_cal_action_rate":public_action_rate-cal_action_rate,
            "public_over_cal_action_rate_ratio":action_rate_ratio,
        },
        "external_public_leaderboard":external or None,
        "verdict":"PASS_PRODUCTION_TRANSFER_CONTRACT" if transfer_pass else "FAIL_OR_INCOMPLETE_TRANSFER_CONTRACT",
        "provenance":{
            "cal_d1":str(d1_path),"public_d1":str(pub_d1_path),"public_rel_l0_zip":str(pub_zip)
        },
        "public_labels_accessed_by_script":False,
    }

    out=root/"results/manual/huy_rel_l0_adaptive_k_transfer_contract_v1"
    out.mkdir(parents=True,exist_ok=True)
    path=out/"REPORT.json"; path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")

    print("[5/5] RESULT"); print("="*116)
    print(
        f"CAL: actions={actions}/{len(ids)} ({100*cal_action_rate:.2f}%) "
        f"gold_removed={gold_removed} R={base_r:.10f}->{cand_r:.10f} "
        f"P={base_p:.10f}->{cand_p:.10f}"
    )
    print(
        f"PUBLIC MATERIALIZATION: actions={len(public_actions)}/{len(pub_d1)} "
        f"({100*public_action_rate:.2f}%) K={k_counts} violations={len(violations)}"
    )
    if external:
        print(
            "PUBLIC LEADERBOARD (external): "
            f"R {external['d1']['recall']:.9f}->{external['rel_l0']['recall']:.9f} "
            f"dR={external['delta']['recall']:+.9f}; "
            f"P {external['d1']['precision']:.9f}->{external['rel_l0']['precision']:.9f} "
            f"dP={external['delta']['precision']:+.9f}"
        )
    print("VERDICT:",report["verdict"])
    print("Report:",path); print("="*116)


if __name__=="__main__":
    main()
