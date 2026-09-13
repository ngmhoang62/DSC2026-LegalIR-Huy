"""Evaluate the preregistered bounded MonoT5 domain-adaptation pilot."""

from __future__ import annotations
import hashlib,json,sqlite3
from collections import Counter
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]; WORKSPACE=ROOT.parent; OUT=ROOT/"results/research_v2_open_rl"
POOL=ROOT/"results/research_v2_forensic/V2_CANDIDATE_POOL.jsonl"
FROZEN_DB=ROOT/"cache/research_v2_open_rl/monot5_fold0_scores.sqlite"
ADAPTED_DB=ROOT/"cache/research_v2_open_rl/monot5_adapted_fold0_scores.sqlite"
SOURCE_DB=WORKSPACE/"LegalIR/cache/exp112_task_adaptive_retrieval/sources.sqlite"

def rows(p):
    with p.open("r",encoding="utf-8") as f:
        for line in f: yield json.loads(line)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda:f.read(8<<20),b""):h.update(b)
    return h.hexdigest()
def answers():
    tr=json.loads((WORKSPACE/"LegalIR/public_test_dataset/train.json").read_text(encoding="utf-8")); ex=json.loads((WORKSPACE/"LegalIR/cache/final_preprocessed_v2/exclusions.json").read_text(encoding="utf-8")); alias={str(x["doc_id"]):str(x["duplicate_retained_id"]) for x in ex if x.get("duplicate_retained_id")}; empty={str(x["doc_id"]) for x in ex if "empty_passage" in x.get("reasons",[])}; return {str(q):{alias.get(str(d),str(d)) for d in v["answer"]}-empty for q,v in tr.items()}
def db_scores(path):
    db=sqlite3.connect(f"file:{Path(path).resolve().as_posix()}?mode=ro",uri=True); integrity=db.execute("pragma integrity_check").fetchone()[0]; progress=db.execute("select count(*),sum(sequences),sum(seconds),max(peak_mib) from progress").fetchone(); scores={(str(q),str(d)):float(s) for q,d,s in db.execute("select qid,doc_id,max(score) from scores group by qid,doc_id")}; db.close(); return scores,integrity,progress
def rec(pred,gold,k=5):return len(set(pred[:k])&gold)/len(gold)
def bucket(r):return "missing" if r is None else "1-5" if r<=5 else "6-10" if r<=10 else "11-20" if r<=20 else "21-50" if r<=50 else "51+"

def main():
    pool={str(x["qid"]):list(map(str,x["doc_ids"])) for x in rows(POOL) if x["fold"]=="fold_0"}; golds=answers(); frozen,fi,fp=db_scores(FROZEN_DB); ft,ai,ap=db_scores(ADAPTED_DB)
    if fi!="ok" or ai!="ok" or fp[0]!=1398 or ap[0]!=1398:raise RuntimeError("DB integrity/completeness")
    anchor={str(x["qid"]):list(map(str,x["fused_top5"])) for x in rows(ROOT/"results/research_v2_post_e5/V2_ADAPTED_E5_LAL_EQUAL_RRF32_PREDICTIONS.jsonl") if x["fold"]=="fold_0"}
    e5={str(x["qid"]):x for x in rows(ROOT/"results/research_v2_e5_confirmation/fold0_runner_parity/E5_CONFIRMATION_FOLD_0_PREDICTIONS.jsonl")}; jv2={str(x["qid"]):list(map(str,x["top5"])) for x in rows(ROOT/"results/research_v2_forensic/V2_ZERO_SHOT_LEXICAL_PREDICTIONS.jsonl") if str(x["qid"]) in pool}
    db=sqlite3.connect(f"file:{SOURCE_DB.as_posix()}?mode=ro",uri=True); native={}
    for q,s,payload in db.execute("select q,source,payload from sources where source in ('lal','jina')"):
        q=str(q)
        if q in pool:native[(q,str(s))]=[str(x["doc_id"]) for x in json.loads(payload) if str(x["doc_id"]) in set(pool[q])][:5]
    db.close()
    base5=[];ft5=[];basek={k:[] for k in (10,20,50)};ftk={k:[] for k in (10,20,50)};clean=[];cleanft=[];anchorft=[]
    wins=losses=changed=into=out=0; movements=Counter(); slices={"single":[[],[]],"multi":[[],[]]}; preds=[]
    for q,docs in pool.items():
        g=golds[q]; br=sorted(docs,key=lambda d:(-frozen[(q,d)],d)); fr=sorted(docs,key=lambda d:(-ft[(q,d)],d)); b=rec(br,g);f=rec(fr,g);base5.append(b);ft5.append(f);wins+=f>b;losses+=f<b;changed+=br[:5]!=fr[:5]
        for k in basek:basek[k].append(rec(br,g,k));ftk[k].append(rec(fr,g,k))
        for d in g:
            rb=br.index(d)+1 if d in br else None;rf=fr.index(d)+1 if d in fr else None;movements[f"{bucket(rb)}->{bucket(rf)}"]+=1;into+=not(rb and rb<=5) and bool(rf and rf<=5);out+=bool(rb and rb<=5) and not(rf and rf<=5)
        key="single" if len(g)==1 else "multi";slices[key][0].append(b);slices[key][1].append(f)
        clean_set=set(e5[q]["ft_order"][:5])|set(e5[q]["base_order"][:5])|set(jv2[q])|set(native[(q,"lal")])|set(native[(q,"jina")]);clean.append(len(clean_set&g)/len(g));cleanft.append(len((clean_set|set(fr[:5]))&g)/len(g));anchorft.append(len((set(anchor[q])|set(fr[:5]))&g)/len(g));preds.append({"qid":q,"frozen_top5":br[:5],"adapted_top5":fr[:5],"frozen_order":br,"adapted_order":fr})
    base=float(np.mean(base5));adapt=float(np.mean(ft5));delta=adapt-base;multi_delta=float(np.mean(slices["multi"][1])-np.mean(slices["multi"][0]));increment=float(np.mean(cleanft)-np.mean(clean))
    if delta>=.05 and adapt>=.84 and wins>losses and multi_delta>=-.005 and increment>=.005: verdict="PASS_BOUNDED_PILOT"
    elif delta<.02 or increment<.003 or multi_delta<-.01:verdict="KILL"
    else:verdict="INCONCLUSIVE_NO_TUNING"
    pp=OUT/"MONOT5_ADAPTED_FOLD0_PREDICTIONS.jsonl"
    with pp.open("w",encoding="utf-8",newline="\n") as f:
        for x in sorted(preds,key=lambda z:int(z["qid"])):f.write(json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(",",":"))+"\n")
    report={"schema_version":"dsc2026.research_v2.monot5_adaptation_fold0_report.v1","status":"COMPLETE","verdict":verdict,"metrics":{"queries":1398,"frozen_recall_at_5":base,"adapted_recall_at_5":adapt,"delta_recall_at_5":delta,"frozen_precision_at_5":float(np.mean(base5)*0+sum(len(set(x["frozen_top5"])&golds[x["qid"]]) for x in preds)/(1398*5)),"adapted_precision_at_5":sum(len(set(x["adapted_top5"])&golds[x["qid"]]) for x in preds)/(1398*5),"wins":wins,"losses":losses,"ties":1398-wins-losses,"changed_top5_sets":changed,"gold_crossings_in":into,"gold_crossings_out":out,"multi_gold":{"queries":len(slices["multi"][0]),"frozen":float(np.mean(slices["multi"][0])),"adapted":float(np.mean(slices["multi"][1])),"delta":multi_delta},"single_gold":{"queries":len(slices["single"][0]),"frozen":float(np.mean(slices["single"][0])),"adapted":float(np.mean(slices["single"][1])),"delta":float(np.mean(slices["single"][1])-np.mean(slices["single"][0]))},"recall_depth":{"frozen":{str(k):float(np.mean(v)) for k,v in basek.items()},"adapted":{str(k):float(np.mean(ftk[k])) for k in ftk}},"rank_movements":dict(movements),"existing_clean_union":float(np.mean(clean)),"existing_clean_plus_adapted_union":float(np.mean(cleanft)),"increment_over_existing_clean_union":increment,"current_anchor_plus_adapted_union":float(np.mean(anchorft))},"runtime":{"frozen":fp,"adapted":ap},"integrity":{"frozen_db":fi,"adapted_db":ai,"frozen_db_sha256":sha(FROZEN_DB),"adapted_db_sha256":sha(ADAPTED_DB),"adapter_sha256":sha(OUT/"monot5_domain_adapter_fold0_pilot/epoch_2/adapter/adapter_model.safetensors"),"predictions_sha256":sha(pp),"pool_sha256":sha(POOL),"training_selection_sha256":sha(OUT/"monot5_domain_adapter_fold0_pilot/TRAINING_SELECTION.json")},"anti_rescue":"No loss/mining/LoRA/epoch/seed/prompt/aggregation/fusion tuning."}
    (OUT/"MONOT5_DOMAIN_ADAPTATION_FOLD0_REPORT.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8");print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=="__main__":main()
