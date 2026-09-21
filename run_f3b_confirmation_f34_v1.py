#!/usr/bin/env python
"""
F3B exploratory confirmation on untouched CAL folds 3-4.

IMPORTANT:
- Original F3 preregistered DEV gate failed because +0.00370 < +0.005.
- This script does NOT retroactively claim that gate passed.
- It performs a new, explicitly exploratory confirmation using the previously
  untouched outcome folds 3-4, with ZERO change to model/feature/config.
- No thresholds or hyperparameters are changed after DEV.
"""
from __future__ import annotations
import argparse, hashlib, json, sys, time
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

CONF_FOLDS = ("fold_3","fold_4")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--repo-root",type=Path,required=True)
    args=ap.parse_args()
    root=args.repo_root.resolve()
    sys.path[:0]=[str(root),str(root/"src"),str(root/"src/sol_high_rl")]

    import noncal_boundary_metric_dev as f3
    from forensic_world_model import metrics, prepare_contract
    from tune_expanded_fusion_selection import ltr_features

    started=time.perf_counter()
    out=root/"results/sol_high_rl"

    dev_report=json.loads((out/"F3_NONCAL_BOUNDARY_METRIC_DEV_REPORT.json").read_text(encoding="utf-8"))
    if tuple(dev_report["protocol"]["development_folds"]) != ("fold_0","fold_1","fold_2"):
        raise RuntimeError("Unexpected historical F3 DEV protocol")
    if dev_report["protocol"]["confirmation_outcomes_inspected"] is not False:
        raise RuntimeError("Historical report says confirmation already inspected")

    split_path=out/"CAL600_STRATIFIED_5FOLD_SEED42.json"
    split_bytes=split_path.read_bytes()
    split=json.loads(split_bytes)
    base_report=json.loads((out/"CAL600_CANONICAL_BASELINE_REPORT.json").read_text(encoding="utf-8"))
    if hashlib.sha256(split_bytes).hexdigest()!=base_report["protocol"]["split_sha256"]:
        raise RuntimeError("split mismatch")
    baseline=json.loads((out/"CAL600_CANONICAL_BASELINE_OOF_PREDICTIONS.json").read_text(encoding="utf-8"))

    queries,_,all_ids,candidates,views,scores,names,_,extras,_=prepare_contract()
    cbytes=(json.dumps({q:candidates[q] for q in all_ids},ensure_ascii=False,indent=2)+"\n").encode()
    if hashlib.sha256(cbytes).hexdigest()!=base_report["candidate_contract_sha256"]:
        raise RuntimeError("candidate mismatch")

    offsets,vectors=f3.load_index()
    weight,training=f3.train_metric(offsets,vectors)
    specialist=f3.score_cal(weight,offsets,vectors,candidates,all_ids)

    new_views=dict(views)
    new_scores=dict(scores)
    new_names=list(names)+["noncal_boundary_diag"]
    new_scores["noncal_boundary_diag"]=specialist
    new_views["noncal_boundary_diag"]={
        q:sorted(candidates[q],key=lambda d:(-specialist[q][d],d))
        for q in all_ids
    }

    rows,groups=ltr_features(new_views,new_names,candidates,all_ids,new_scores)
    for extra in extras:
        for q in all_ids:
            rows[q]=np.concatenate([rows[q],extra[q]],axis=1)

    pred={}
    fold_reports={}
    for fold in CONF_FOLDS:
        test=list(map(str,split["folds"][fold]))
        held=set(test)
        train=[q for q in all_ids if q not in held]

        x=np.vstack([rows[q] for q in train])
        y=np.concatenate([
            [d in queries[q][1] for d in groups[q]]
            for q in train
        ]).astype(np.int8)

        scaler=StandardScaler().fit(x)
        model=LogisticRegression(
            C=.15,class_weight="balanced",solver="liblinear",
            max_iter=3000,random_state=2026
        )
        model.fit(scaler.transform(x),y)

        for q in test:
            s=model.predict_proba(scaler.transform(rows[q]))[:,1]
            pred[q]=[groups[q][i] for i in np.argsort(-s,kind="stable")]

        bm=metrics(baseline,queries,test)[0]
        cm=metrics(pred,queries,test)[0]
        fold_reports[fold]={
            "baseline":bm,
            "candidate":cm,
            "delta":cm["recall_at_5"]-bm["recall_at_5"],
            "paired":f3.paired(pred,baseline,queries,test),
        }

    conf_ids=sum((list(map(str,split["folds"][f])) for f in CONF_FOLDS),[])
    bm=metrics(baseline,queries,conf_ids)[0]
    cm=metrics(pred,queries,conf_ids)[0]
    pp=f3.paired(pred,baseline,queries,conf_ids)

    multi=[q for q in conf_ids if len(queries[q][1])>1]
    bmulti=metrics(baseline,queries,multi)[0]
    cmulti=metrics(pred,queries,multi)[0]

    delta=cm["recall_at_5"]-bm["recall_at_5"]
    multi_delta=cmulti["recall_at_5"]-bmulti["recall_at_5"]

    # New exploratory confirmation gate, declared here before seeing outputs.
    # Require positive aggregate, wins>losses, BOTH folds nonnegative,
    # and no material multi-gold regression.
    confirm_pass=(
        delta>0
        and pp["wins"]>pp["losses"]
        and all(x["delta"]>=-1e-12 for x in fold_reports.values())
        and multi_delta>=-0.01
    )

    report={
        "status":"EXPLORATORY_CONFIRM_PASS" if confirm_pass else "EXPLORATORY_CONFIRM_FAIL",
        "family":"F3B_same_frozen_noncal_diagonal_metric_no_config_change",
        "disclosure":{
            "original_F3_dev_gate_passed":bool(dev_report["promotion_gate_passed"]),
            "original_F3_dev_delta":dev_report["dev_delta"],
            "reason_original_failed":"original preregistered delta gate required >= +0.005",
            "claim":"new exploratory confirmation; not a retroactive preregistration pass",
        },
        "confirmation_folds":list(CONF_FOLDS),
        "training":training,
        "baseline_confirm":bm,
        "candidate_confirm":cm,
        "delta_confirm":delta,
        "paired_confirm":pp,
        "folds":fold_reports,
        "multi_gold":{
            "baseline":bmulti,"candidate":cmulti,"delta":multi_delta
        },
        "confirmation_gate":{
            "aggregate_delta_gt_0":delta>0,
            "wins_gt_losses":pp["wins"]>pp["losses"],
            "both_folds_nonnegative":all(x["delta"]>=-1e-12 for x in fold_reports.values()),
            "multi_delta_ge_minus_0.01":multi_delta>=-0.01,
            "passed":confirm_pass,
        },
        "runtime_seconds":time.perf_counter()-started,
    }

    path=out/"F3B_NONCAL_BOUNDARY_METRIC_CONFIRM_F34_REPORT.json"
    path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")

    print("="*96)
    print("F3B EXPLORATORY CONFIRMATION — SAME CONFIG, FOLDS 3-4")
    print(f"baseline R@5={bm['recall_at_5']:.10f}")
    print(f"candidate R@5={cm['recall_at_5']:.10f}")
    print(f"delta={delta:+.10f}")
    print(f"W/L/T={pp['wins']}/{pp['losses']}/{pp['ties']}")
    print("fold deltas:",{f:round(x["delta"],10) for f,x in fold_reports.items()})
    print(f"multi_delta={multi_delta:+.10f}")
    print("VERDICT:",report["status"])
    print("Report:",path)
    print("="*96)

if __name__=="__main__":
    main()
