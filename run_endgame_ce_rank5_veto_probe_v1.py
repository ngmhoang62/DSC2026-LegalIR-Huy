#!/usr/bin/env python
"""
ENDGAME CE RANK5 VETO PROBE V1
------------------------------
Zero-training diagnostic using the completed held-out Fold0 CE scores.

Question:
Can the CE checkpoint safely identify obviously non-relevant rank-5 docs?

Protocol:
- fold0 is unseen during CE training.
- deterministic DEV/CONFIRM split.
- test two shift-resistant CE confidence features:
    ABS_LOGIT
    REL_TOP4_MED = score(rank5) - median(score(rank1..4))
- DEV threshold comes ONLY from gold rank5 lower envelope:
    t = min(gold_feature_DEV) - lambda * robust_scale(gold_feature_DEV)
  for preregistered lambdas [0,.25,.5,1,1.5,2,3].
- DROP iff feature < t.
- report CONFIRM gold_removed/actions/precision proxy.
No CAL gold is read.
"""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, math, sys
from pathlib import Path
from typing import Dict, List
import numpy as np

LAMBDAS = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0]
SALT = "dsc2026-endgame-ce-rank5-veto-v1"

def loadmod(path: Path):
    spec = importlib.util.spec_from_file_location("cebase", path)
    m = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(m)
    return m

def scale(x):
    x=np.asarray(x,dtype=float)
    if len(x)<2: return 1e-6
    med=np.median(x)
    mad=np.median(np.abs(x-med))*1.4826
    q75,q25=np.percentile(x,[75,25])
    iqr=(q75-q25)/1.349
    std=np.std(x)
    return max(float(mad),float(iqr),float(std),1e-6)

def split(ids):
    dev=[]; conf=[]
    for q in ids:
        h=int(hashlib.sha256(f"{SALT}|{q}".encode()).hexdigest(),16)
        (dev if h%2==0 else conf).append(q)
    return dev,conf

def eval_drop(rows, ids, feat, threshold):
    actions=gold_removed=non_gold_removed=0
    removed_hit_mass=0.0
    for q in ids:
        r=rows[q]
        if r[feat] < threshold:
            actions += 1
            if r["rank5_gold"]:
                gold_removed += 1
                removed_hit_mass += 1.0/r["gold_count"]
            else:
                non_gold_removed += 1
    return {
        "actions":actions,
        "gold_removed":gold_removed,
        "non_gold_removed":non_gold_removed,
        "recall_delta":-removed_hit_mass/len(ids),
        "safe_action_precision":non_gold_removed/max(actions,1),
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--repo-root",type=Path,required=True)
    args=ap.parse_args()
    root=args.repo_root.resolve()
    base_script=root.parent/"run_noncal_trainable_ce_boundary_v3_fixed.py"
    if not base_script.is_file():
        raise FileNotFoundError(base_script)
    m=loadmod(base_script)
    sibling=root.parent/"LegalIR"
    sys.path[:0]=[str(root),str(root/"src"),str(sibling),str(sibling/"src")]

    cal_ids,_=m.get_cal_ids_label_free(root)
    world=m.load_noncal_world(root,sibling,set(cal_ids))
    fold0=[q for q in world["noncal"] if world["folds"][q]=="fold_0"]
    dev,conf=split(fold0)

    score_dir=root/"results/manual/huy_noncal_trainable_ce_boundary_v1/oof/fold_0/scores"
    rows={}
    missing=[]
    for q in fold0:
        p=score_dir/f"{q}.json"
        if not p.is_file():
            missing.append(q); continue
        obj=json.loads(p.read_text(encoding="utf-8"))
        scores={str(d):float(s) for d,s in obj["scores"].items()}
        top5=world["base"][q][:5]
        if any(d not in scores for d in top5):
            raise RuntimeError(f"Missing top5 CE score q={q}")
        s5=scores[top5[4]]
        top4=[scores[d] for d in top5[:4]]
        rows[q]={
            "ABS_LOGIT":s5,
            "REL_TOP4_MED":s5-float(np.median(top4)),
            "rank5_gold":top5[4] in world["gold"][q],
            "gold_count":len(world["gold"][q]),
        }
    if missing:
        raise RuntimeError(f"Missing {len(missing)} score files sample={missing[:10]}")

    report={
        "fold0":len(fold0),"dev":len(dev),"confirm":len(conf),
        "dev_rank5_gold":sum(rows[q]["rank5_gold"] for q in dev),
        "confirm_rank5_gold":sum(rows[q]["rank5_gold"] for q in conf),
        "families":{}
    }
    print(f"fold0={len(fold0)} dev={len(dev)} confirm={len(conf)}")
    print(f"rank5 gold: DEV={report['dev_rank5_gold']} CONFIRM={report['confirm_rank5_gold']}")

    for feat in ["ABS_LOGIT","REL_TOP4_MED"]:
        gold_dev=[rows[q][feat] for q in dev if rows[q]["rank5_gold"]]
        if not gold_dev:
            raise RuntimeError(f"No DEV gold rank5 for {feat}")
        mn=min(gold_dev); sc=scale(gold_dev)
        fam=[]
        print(f"\n[{feat}] gold_dev_min={mn:+.6f} scale={sc:.6f}")
        for lam in LAMBDAS:
            t=mn-lam*sc
            de=eval_drop(rows,dev,feat,t)
            co=eval_drop(rows,conf,feat,t)
            rec={"lambda":lam,"threshold":t,"dev":de,"confirm":co}
            fam.append(rec)
            print(
                f"  λ={lam:<4} t={t:+.5f} | "
                f"DEV a={de['actions']:3d} loss={de['gold_removed']} | "
                f"CONF a={co['actions']:3d} loss={co['gold_removed']} "
                f"RΔ={co['recall_delta']:+.8f}"
            )
        report["families"][feat]=fam

    # Rank safe confirmed configurations by CONFIRM coverage.
    safe=[]
    for feat,fam in report["families"].items():
        for r in fam:
            if r["dev"]["gold_removed"]==0 and r["confirm"]["gold_removed"]==0:
                safe.append({
                    "feature":feat,"lambda":r["lambda"],"threshold":r["threshold"],
                    "dev_actions":r["dev"]["actions"],
                    "confirm_actions":r["confirm"]["actions"],
                })
    safe.sort(key=lambda x:(x["confirm_actions"],x["dev_actions"]),reverse=True)
    report["safe_confirmed"]=safe
    out=root/"results/manual/huy_endgame_ce_rank5_veto_probe_v1/REPORT.json"
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("\nTop safe-confirmed configs:")
    for x in safe[:10]:
        print(x)
    print(f"\nReport: {out}")

if __name__=="__main__":
    main()
