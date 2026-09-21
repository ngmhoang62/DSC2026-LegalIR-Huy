#!/usr/bin/env python
"""
D1 CAL600 CE RANK5 VETO TRANSFER AUDIT V1
=========================================

Purpose
-------
Test whether the already-trained Fold0 BGE CE can safely DROP D1 rank-5 docs.
No additional training.

Scientific contract
-------------------
1) Thresholds are derived ONLY from held-out nonCAL Fold0 DEV.
2) Candidate thresholds are pre-registered:
   - PRIMARY: REL_TOP4_MED lambda=0.0
   - CONSERVATIVE: REL_TOP4_MED lambda=0.25, 0.5, 1.0
   - ABS controls: ABS_LOGIT lambda=0.5, 1.0
3) Score D1 CAL Top5 label-free and seal ALL actions/predictions.
4) Only then reveal CAL gold and report Recall / macro Precision / actions /
   gold_removed.
5) CAL query outside the V2-evaluable representation automatically abstains.

This specifically tests transfer:
    Fold0 base (E5+LAL RRF32) -> production D1 rank5.
"""

from __future__ import annotations
import argparse, gc, hashlib, importlib.util, json, sys, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch

SALT = "dsc2026-endgame-ce-rank5-veto-v1"
PRESET = {
    "REL_L0": ("REL_TOP4_MED", 0.0),
    "REL_L025": ("REL_TOP4_MED", 0.25),
    "REL_L05": ("REL_TOP4_MED", 0.5),
    "REL_L10": ("REL_TOP4_MED", 1.0),
    "ABS_L05": ("ABS_LOGIT", 0.5),
    "ABS_L10": ("ABS_LOGIT", 1.0),
}
EXPECTED_D1_R = 0.9569444444444444
EXPECTED_D1_P = 0.20566666666666666


def loadmod(path: Path):
    spec = importlib.util.spec_from_file_location("cebase", path)
    m = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(m)
    return m


def dump(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def sha256(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def robust_scale(x):
    x = np.asarray(x, dtype=np.float64)
    med = np.median(x)
    mad = np.median(np.abs(x-med))*1.4826
    q75,q25 = np.percentile(x,[75,25])
    iqr = (q75-q25)/1.349
    std = np.std(x)
    return max(float(mad), float(iqr), float(std), 1e-6)


def split_fold0(qids):
    dev=[]; confirm=[]
    for q in qids:
        h=int(hashlib.sha256(f"{SALT}|{q}".encode()).hexdigest(),16)
        (dev if h%2==0 else confirm).append(q)
    return dev, confirm


def derive_thresholds(m, world, root: Path):
    fold0=[q for q in world["noncal"] if world["folds"][q]=="fold_0"]
    dev,_=split_fold0(fold0)
    score_dir=root/"results/manual/huy_noncal_trainable_ce_boundary_v1/oof/fold_0/scores"

    feats={q:{} for q in dev}
    for q in dev:
        p=score_dir/f"{q}.json"
        if not p.is_file():
            raise FileNotFoundError(p)
        obj=json.loads(p.read_text(encoding="utf-8"))
        scores={str(d):float(s) for d,s in obj["scores"].items()}
        top5=world["base"][q][:5]
        s5=scores[top5[4]]
        s14=[scores[d] for d in top5[:4]]
        feats[q]["ABS_LOGIT"]=s5
        feats[q]["REL_TOP4_MED"]=s5-float(np.median(s14))
        feats[q]["rank5_gold"]=top5[4] in world["gold"][q]

    stats={}
    for feature in ("ABS_LOGIT","REL_TOP4_MED"):
        vals=[feats[q][feature] for q in dev if feats[q]["rank5_gold"]]
        if not vals:
            raise RuntimeError(f"No Fold0 DEV gold rank5 for {feature}")
        stats[feature]={
            "gold_dev_n":len(vals),
            "gold_dev_min":float(min(vals)),
            "robust_scale":robust_scale(vals),
        }

    arms={}
    for arm,(feature,lam) in PRESET.items():
        s=stats[feature]
        threshold=s["gold_dev_min"]-lam*s["robust_scale"]
        arms[arm]={"feature":feature,"lambda":lam,"threshold":float(threshold)}
    return stats, arms


@torch.inference_mode()
def score_cal_top5(m, root, sibling, world, model_path, pair_microbatch, out):
    # Precision-veto only needs the exact frozen D1 Top5.  Do NOT reuse the
    # recall-frontier loader here: that loader reconstructs a cap32 candidate
    # universe and enforces pool-containment assumptions that are irrelevant to
    # rank5 pruning (and are not true for every D1 query).
    baseline_path = root / "results/sol_high_rl/BASELINE_LOBO_PREDICTIONS.json"
    if not baseline_path.is_file():
        raise FileNotFoundError(baseline_path)

    raw_base = json.loads(baseline_path.read_text(encoding="utf-8"))
    ids = [str(q) for q in raw_base.keys()]
    if len(ids) != 600 or len(set(ids)) != 600:
        raise RuntimeError(f"Expected 600 unique CAL D1 queries, got {len(ids)}")

    base = {}
    for q, row in raw_base.items():
        q = str(q)
        if isinstance(row, dict):
            if "answer" in row:
                row = row["answer"]
            elif "predictions" in row:
                row = row["predictions"]
            else:
                raise RuntimeError(f"Unexpected D1 row schema qid={q}: keys={list(row)[:10]}")
        docs = [str(d) for d in row]
        if len(docs) < 5 or len(set(docs[:5])) != 5:
            raise RuntimeError(f"Invalid D1 Top5 qid={q}: {docs[:5]}")
        base[q] = docs

    cal_overlap = set(world["cal_overlap"])
    eligible=[q for q in ids if q in cal_overlap]
    abstain=[q for q in ids if q not in cal_overlap]

    if set(eligible) != cal_overlap:
        raise RuntimeError(
            f"D1/V2 CAL overlap mismatch: D1 eligible={len(eligible)} world={len(cal_overlap)}"
        )

    sys.path[:0]=[str(sibling),str(sibling/"src")]
    from exp_final.cross_encoder import CrossEncoder
    from exp_final.evidence import Evidence

    model=CrossEncoder(model_path)
    model.eval()
    evidence=Evidence(world["render"],model.tokenizer)

    score_dir=out/"cal_d1_top5_ce_scores"
    score_dir.mkdir(parents=True,exist_ok=True)
    rows={}
    t0=time.perf_counter()

    try:
        for i,q in enumerate(eligible,1):
            top5=base[q][:5]
            p=score_dir/f"{q}.json"
            sig=hashlib.sha256(
                json.dumps([sha256(model_path),q,top5,"d1-top5-ce-v1"],sort_keys=True).encode()
            ).hexdigest()

            if p.is_file():
                obj=json.loads(p.read_text(encoding="utf-8"))
                if obj["signature"]!=sig:
                    raise RuntimeError(f"Score cache signature mismatch q={q}")
                scores={str(d):float(s) for d,s in obj["scores"].items()}
            else:
                vals=[]
                for st in range(0,5,pair_microbatch):
                    docs=top5[st:st+pair_microbatch]
                    pairs=[evidence.package(q,d) for d in docs]
                    vals.extend(model(pairs).detach().cpu().tolist())
                scores={d:float(s) for d,s in zip(top5,vals)}
                dump(p,{"signature":sig,"scores":scores})

            s5=scores[top5[4]]
            s14=[scores[d] for d in top5[:4]]
            rows[q]={
                "ABS_LOGIT":s5,
                "REL_TOP4_MED":s5-float(np.median(s14)),
                "rank5":top5[4],
                "top5":top5,
                "scores":scores,
            }

            if i%50==0 or i==len(eligible):
                print(
                    f"  CE score {i}/{len(eligible)} "
                    f"qps={i/max(time.perf_counter()-t0,1e-9):.3f}",
                    flush=True
                )
    finally:
        evidence.db.close()
        del evidence,model
        gc.collect()
        torch.cuda.empty_cache()

    return ids,base,eligible,abstain,rows


def variable_metrics(pred,gold,ids):
    rs=[]; ps=[]; hits=returned=0
    for q in ids:
        ans=pred[q]
        h=len(set(ans)&gold[q])
        hits+=h; returned+=len(ans)
        rs.append(h/len(gold[q]))
        ps.append(h/len(ans))
    return {
        "recall":float(np.mean(rs)),
        "precision_macro":float(np.mean(ps)),
        "precision_micro":hits/returned,
        "hits":hits,
        "returned":returned,
        "mean_k":returned/len(ids),
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--repo-root",type=Path,required=True)
    ap.add_argument("--pair-microbatch",type=int,default=4)
    args=ap.parse_args()

    root=args.repo_root.resolve()
    sibling=root.parent/"LegalIR"
    base_script=root.parent/"run_noncal_trainable_ce_boundary_v3_fixed.py"
    if not base_script.is_file():
        raise FileNotFoundError(base_script)
    m=loadmod(base_script)
    sys.path[:0]=[str(root),str(root/"src"),str(sibling),str(sibling/"src")]

    out=root/"results/manual/huy_d1_cal_ce_rank5_veto_transfer_v1"
    model_path=root/"results/manual/huy_noncal_trainable_ce_boundary_v1/oof/fold_0/training/model.pt"
    if not model_path.is_file():
        raise FileNotFoundError(model_path)

    print("[1/5] Loading sanitized world...")
    cal_ids,_=m.get_cal_ids_label_free(root)
    world=m.load_noncal_world(root,sibling,set(cal_ids))

    print("[2/5] Re-deriving frozen thresholds from Fold0 DEV only...")
    fold_stats,arms=derive_thresholds(m,world,root)
    print(json.dumps({"fold0_dev_stats":fold_stats,"arms":arms},indent=2))

    print("[3/5] Scoring exact D1 CAL Top5 label-free...")
    ids,base,eligible,abstain,rows=score_cal_top5(
        m,root,sibling,world,model_path,args.pair_microbatch,out
    )

    predictions={}
    actions={}
    for arm,cfg in arms.items():
        pred={q:list(base[q][:5]) for q in ids}
        act={}
        for q in eligible:
            if rows[q][cfg["feature"]] < cfg["threshold"]:
                removed=pred[q].pop()
                act[q]={
                    "qid":q,
                    "removed":removed,
                    "feature":cfg["feature"],
                    "value":rows[q][cfg["feature"]],
                    "threshold":cfg["threshold"],
                    "new_answer":pred[q],
                }
        predictions[arm]=pred
        actions[arm]=act

    seal={
        "schema":"manual.d1_cal_ce_rank5_veto_transfer_v1.actions",
        "status":"SEALED_BEFORE_CAL_GOLD",
        "model_sha256":sha256(model_path),
        "fold0_dev_stats":fold_stats,
        "arms":arms,
        "eligible_cal_v2":len(eligible),
        "cal_outside_v2_abstentions":abstain,
        "action_counts":{a:len(x) for a,x in actions.items()},
        "actions":actions,
        "predictions":predictions,
    }
    seal_path=out/"CAL_ACTIONS_LABEL_FREE.json"
    dump(seal_path,seal)
    print("  sealed action counts:",seal["action_counts"])

    print("[4/5] Revealing CAL gold AFTER seal...")
    gold_path=root/"DSC2026-LegalIR-main/v4_run/public_test_dataset/train.json"
    raw=json.loads(gold_path.read_text(encoding="utf-8"))
    gold={q:{str(d) for d in raw[q]["answer"]} for q in ids}

    baseline=variable_metrics({q:base[q][:5] for q in ids},gold,ids)
    if abs(baseline["recall"]-EXPECTED_D1_R)>1e-12:
        raise RuntimeError(f"D1 recall parity failed: {baseline}")
    if abs(baseline["precision_macro"]-EXPECTED_D1_P)>5e-10:
        raise RuntimeError(f"D1 precision parity failed: {baseline}")

    report={
        "schema":"manual.d1_cal_ce_rank5_veto_transfer_v1.report",
        "baseline":baseline,
        "arms":{},
        "action_seal_sha256":sha256(seal_path),
        "gold_reveal_time":datetime.now(timezone.utc).isoformat(),
    }

    for arm in PRESET:
        met=variable_metrics(predictions[arm],gold,ids)
        harmful=[]
        for q,a in actions[arm].items():
            if a["removed"] in gold[q]:
                harmful.append({
                    "qid":q,
                    "removed":a["removed"],
                    "gold_count":len(gold[q]),
                    "feature_value":a["value"],
                })
        report["arms"][arm]={
            "config":arms[arm],
            "metrics":met,
            "delta_recall":met["recall"]-baseline["recall"],
            "delta_precision_macro":met["precision_macro"]-baseline["precision_macro"],
            "actions":len(actions[arm]),
            "gold_removed":len(harmful),
            "harmful":harmful,
        }

    dump(out/"REPORT.json",report)

    print("[5/5] RESULTS")
    print("="*104)
    print(
        f"D1 R={baseline['recall']:.10f} Pmacro={baseline['precision_macro']:.10f} "
        f"Pmicro={baseline['precision_micro']:.10f} meanK={baseline['mean_k']:.4f}"
    )
    for arm,r in report["arms"].items():
        x=r["metrics"]
        print(
            f"{arm:9s} R={x['recall']:.10f} ({r['delta_recall']:+.10f}) "
            f"Pmacro={x['precision_macro']:.10f} ({r['delta_precision_macro']:+.10f}) "
            f"actions={r['actions']:3d} gold_removed={r['gold_removed']} meanK={x['mean_k']:.4f}"
        )
    print(f"Report: {out/'REPORT.json'}")
    print("="*104)


if __name__=="__main__":
    main()
