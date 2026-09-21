#!/usr/bin/env python
from __future__ import annotations
import argparse, hashlib, json, sqlite3, sys, time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Set, Tuple
import numpy as np, torch

DEPTH=50
EXPECTED_D1_R5=0.9569444444444444
EXPECTED_D1_P5=0.20566666666666666
V2_MIN_DELTA=0.001
V2_MIN_NONNEG_FOLDS=4
V2_WORST_FOLD=-0.005
V2_MIN_INTERVENTION_PRECISION=0.60

def sha256_file(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(8<<20),b""): h.update(block)
    return h.hexdigest()

def write_json(path:Path,obj:Any)->None:
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    tmp.replace(path)

def read_jsonl(path:Path)->List[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]

def recall(top5:Sequence[str],gold:Set[str])->float:
    return len(set(top5[:5])&gold)/len(gold)

def precision(top5:Sequence[str],gold:Set[str])->float:
    return len(set(top5[:5])&gold)/5.0

def metric_map(pred,gold,ids):
    return {"recall_at_5":float(np.mean([recall(pred[q],gold[q]) for q in ids])),
            "precision_at_5":float(np.mean([precision(pred[q],gold[q]) for q in ids]))}

def patch_transformers_v5():
    import transformers.models.xlm_roberta.modeling_xlm_roberta as module
    if hasattr(module,"create_position_ids_from_input_ids"): return
    def create_position_ids(input_ids,padding_idx,past_key_values_length=0):
        mask=input_ids.ne(padding_idx).int()
        positions=(torch.cumsum(mask,dim=1)+past_key_values_length)*mask
        return positions.long()+padding_idx
    module.create_position_ids_from_input_ids=create_position_ids

def load_base_jina(model_path:Path):
    patch_transformers_v5()
    from transformers import AutoModelForSequenceClassification,AutoTokenizer
    tok=AutoTokenizer.from_pretrained(model_path,trust_remote_code=True,fix_mistral_regex=True)
    model=AutoModelForSequenceClassification.from_pretrained(model_path,trust_remote_code=True,dtype=torch.float16).eval().to("cuda")
    model._tokenizer=tok
    return model

def open_score_db(path:Path):
    db=sqlite3.connect(path)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("CREATE TABLE IF NOT EXISTS scores(dataset TEXT,qid TEXT,doc_id TEXT,score REAL,PRIMARY KEY(dataset,qid,doc_id))")
    db.execute("CREATE TABLE IF NOT EXISTS progress(dataset TEXT,qid TEXT,docs INTEGER,pairs INTEGER,seconds REAL,peak_mib REAL,PRIMARY KEY(dataset,qid))")
    db.commit()
    return db

def overlap_novel(a,b,pool):
    pool=set(map(str,pool)); rb={str(d):i+1 for i,d in enumerate(b[:DEPTH])}; ra={str(d):i+1 for i,d in enumerate(a[:DEPTH])}
    c=[str(d) for d in a[:DEPTH] if str(d) in rb and str(d) not in pool]
    return sorted(set(c),key=lambda d:(ra[d]+rb[d],max(ra[d],rb[d]),d))

def required_universe(ids,pools,base,a,b):
    universes={}; overlap={}
    for q in ids:
        novels=overlap_novel(a[q],b[q],pools[q]); overlap[q]=novels
        universes[q]=list(dict.fromkeys(base[q][:5]+novels))
    return universes,overlap

def score_universes(*,root,dataset,ids,questions,universes,cache_path,model_path,batch_size):
    if not torch.cuda.is_available(): raise RuntimeError("CUDA required.")
    if str(root) not in sys.path: sys.path.insert(0,str(root))
    from benchmark_jina_reranker_holdouts import top_passages
    from run_burst_expanded_fusion_submission import DocumentStore
    context_dir=root/"DSC2026-LegalIR-main/v4_run/public_test_dataset/selected-contexts"
    docs=DocumentStore(sorted(context_dir.glob("context_*.json")),cache_size=9000)
    db=open_score_db(cache_path)
    cached={(str(q),str(d)):float(s) for q,d,s in db.execute("SELECT qid,doc_id,score FROM scores WHERE dataset=?",(dataset,))}
    todo=[q for q in ids if any((q,d) not in cached for d in universes[q])]
    if todo:
        model=load_base_jina(model_path); torch.cuda.reset_peak_memory_stats(); started=time.perf_counter()
        try:
            for index,q in enumerate(todo,1):
                qtext=questions[q]; missing=[d for d in universes[q] if (q,d) not in cached]
                owners=[]; passages=[]
                for d in missing:
                    text=docs[d]
                    ps=top_passages(qtext,text,count=2,window=220,overlap=70) or [text]
                    for p in ps: owners.append(d); passages.append(p)
                before=time.perf_counter()
                raw=model.compute_score([(qtext,p) for p in passages],batch_size=batch_size,max_length=512)
                if isinstance(raw,(float,int)): raw=[float(raw)]
                parent={}
                for d,s in zip(owners,[float(x) for x in raw]): parent[d]=max(parent.get(d,-1e9),s)
                if set(parent)!=set(missing): raise RuntimeError(f"{dataset}/{q}: coverage mismatch")
                elapsed=time.perf_counter()-before; peak=float(torch.cuda.max_memory_allocated()/2**20)
                with db:
                    db.executemany("INSERT OR REPLACE INTO scores VALUES(?,?,?,?)",[(dataset,q,d,s) for d,s in parent.items()])
                    db.execute("INSERT OR REPLACE INTO progress VALUES(?,?,?,?,?,?)",(dataset,q,len(missing),len(passages),elapsed,peak))
                for d,s in parent.items(): cached[(q,d)]=s
                if index%50==0 or index==len(todo):
                    print(f"    {dataset}: {index}/{len(todo)} q={q} docs={len(missing)} pairs={len(passages)} sec/q={(time.perf_counter()-started)/index:.3f} peak={peak:.0f} MiB",flush=True)
        finally:
            del model; torch.cuda.empty_cache()
    else:
        print(f"  {dataset}: semantic cache complete.",flush=True)
    result={}
    for q in ids:
        result[q]={}
        for d in universes[q]:
            row=db.execute("SELECT score FROM scores WHERE dataset=? AND qid=? AND doc_id=?",(dataset,q,d)).fetchone()
            if row is None: db.close(); raise RuntimeError(f"missing score {dataset}/{q}/{d}")
            result[q][d]=float(row[0])
    db.close()
    return result

def apply_crossover(ids,base,overlap,semantic):
    modified={q:list(base[q][:5]) for q in ids}; actions={}
    for q in ids:
        defender=base[q][4]; novels=overlap[q]
        if not novels: continue
        universe=list(dict.fromkeys(base[q][:5]+novels)); scores=semantic[q]
        ordered=sorted(universe,key=lambda d:(-scores[d],str(d))); ranks={d:i+1 for i,d in enumerate(ordered)}
        def_rank=ranks[defender]; def_score=scores[defender]; eligible=[]; evals=[]
        for c in novels:
            passed=bool(scores[c]>def_score and ranks[c]<=5 and def_rank>5)
            if passed: eligible.append(c)
            evals.append({"doc_id":c,"semantic_score":scores[c],"semantic_rank":ranks[c],"eligible":passed})
        if len(eligible)==1:
            c=eligible[0]; modified[q]=list(base[q][:4])+[c]
            actions[q]={"qid":q,"defender":defender,"challenger":c,"defender_semantic_score":def_score,"defender_semantic_rank":def_rank,
                        "overlap_novel_count":len(novels),"candidate_evaluations":evals,"new_top5":modified[q]}
    return modified,actions

def outcomes(ids,base,modified,gold):
    beneficial=harmful=neutral=0; weighted=0.0; details=[]
    for q in ids:
        if base[q][:5]==modified[q][:5]: continue
        before=recall(base[q],gold[q]); after=recall(modified[q],gold[q]); delta=after-before; weighted+=delta
        if delta>0: beneficial+=1; label="beneficial"
        elif delta<0: harmful+=1; label="harmful"
        else: neutral+=1; label="neutral"
        details.append({"qid":q,"before":before,"after":after,"delta":delta,"outcome":label,"base_top5":base[q][:5],"modified_top5":modified[q][:5]})
    return {"actions":len(details),"beneficial":beneficial,"harmful":harmful,"neutral":neutral,"weighted_query_recall_mass":weighted,
            "intervention_precision_excluding_neutral":beneficial/max(beneficial+harmful,1),"details":details}

def overlap_oracle(ids,pools,base,overlap,gold):
    recoverable=[]; mass=0.0
    for q in ids:
        for d in overlap[q]:
            if d in gold[q] and d not in set(base[q][:5]): recoverable.append({"qid":q,"doc_id":d})
        defender=base[q][4]; best=0.0
        for c in overlap[q]:
            util=(int(c in gold[q])-int(defender in gold[q]))/len(gold[q]); best=max(best,util)
        mass+=best
    return {"recoverable_gold_occurrences":len(recoverable),"recoverable":recoverable,"one_swap_oracle_delta":mass/len(ids),
            "queries_with_overlap_novel":sum(bool(overlap[q]) for q in ids),"total_overlap_novel_candidates":sum(len(overlap[q]) for q in ids),
            "mean_overlap_novel_candidates":float(np.mean([len(overlap[q]) for q in ids]))}

def load_v2(root):
    if str(root/"src") not in sys.path: sys.path.insert(0,str(root/"src"))
    from research_v2_e5_transfer.e5_transfer_runner import TransferData
    data=TransferData(root/"cache/research_v2_e5_confirmation/bundle-v1")
    ids=sorted(data.questions,key=int); questions=dict(data.questions); gold={q:set(data.gold[q]) for q in ids}; pools={q:list(data.pool[q]) for q in ids}; folds=dict(data.fold_for)
    anchor_path=root/"results/research_v2_post_e5/V2_ADAPTED_E5_LAL_EQUAL_RRF32_PREDICTIONS.jsonl"
    anchor={str(r["qid"]):r for r in read_jsonl(anchor_path)}; base={q:[str(d) for d in anchor[q]["fused_top5"]] for q in ids}
    e5={}
    for i in range(5):
        p=root/f"results/research_v2_open_rl/fold_{i}/FULL_CORPUS_PREDICTIONS.jsonl"
        if not p.is_file(): raise FileNotFoundError(p)
        for r in read_jsonl(p): e5[str(r["qid"])]=[str(d) for d in r["adapted_order_top150"]]
    lal={q:[str(d) for d in data.sources[q]["lal"]] for q in ids}
    if not(set(e5)==set(lal)==set(base)==set(ids)): raise RuntimeError("V2 population mismatch")
    return ids,questions,gold,pools,folds,base,e5,lal

def v2_stage(root,out,score_db,model_path,batch_size):
    ids,questions,gold,pools,folds,base,e5,lal=load_v2(root)
    universes,overlap=required_universe(ids,pools,base,e5,lal)
    oracle=overlap_oracle(ids,pools,base,overlap,gold)
    print(f"  V2 overlap: queries={oracle['queries_with_overlap_novel']} candidates={oracle['total_overlap_novel_candidates']} recoverable_gold={oracle['recoverable_gold_occurrences']} oracle_delta={oracle['one_swap_oracle_delta']:+.6f}",flush=True)
    semantic=score_universes(root=root,dataset="v2",ids=ids,questions=questions,universes=universes,cache_path=score_db,model_path=model_path,batch_size=batch_size)
    modified,actions=apply_crossover(ids,base,overlap,semantic)
    bm=metric_map(base,gold,ids); mm=metric_map(modified,gold,ids); paired=outcomes(ids,base,modified,gold)
    fold_report={}; deltas=[]
    for f in sorted(set(folds.values())):
        qids=[q for q in ids if folds[q]==f]; fb=metric_map(base,gold,qids); fm=metric_map(modified,gold,qids); d=fm["recall_at_5"]-fb["recall_at_5"]; deltas.append(d)
        po=outcomes(qids,base,modified,gold); po.pop("details")
        fold_report[f]={"baseline":fb,"modified":fm,"recall_delta":d,"paired_actions":po}
    delta=mm["recall_at_5"]-bm["recall_at_5"]
    checks={"delta_gte_0_001":delta>=V2_MIN_DELTA,"wins_gt_losses":paired["beneficial"]>paired["harmful"],
            "at_least_4_of_5_folds_nonnegative":sum(d>=0 for d in deltas)>=V2_MIN_NONNEG_FOLDS,
            "worst_fold_gte_minus_0_005":min(deltas)>=V2_WORST_FOLD,
            "intervention_precision_gte_0_60":paired["intervention_precision_excluding_neutral"]>=V2_MIN_INTERVENTION_PRECISION}
    report={"schema":"manual.overlap_semantic_crossover_v1.v2","status":"PASS_V2_GATE" if all(checks.values()) else "KILL_AT_V2_GATE",
            "mechanism":{"depth":DEPTH,"candidate_prior":"intersection of two independent full-corpus Top50 sources minus current pool",
                         "semantic_model":str(model_path),"semantic_contract":"Huy lexical top_passages count=2; frozen base Jina-v2; MAX parent aggregation",
                         "action_rule":"exactly one novel Jina crossover of rank5 defender; otherwise abstain","training":"none","thresholds":"none"},
            "overlap_oracle":oracle,"baseline":bm,"modified":mm,"delta_recall_at_5":delta,"paired_actions":paired,"folds":fold_report,
            "gate_checks":checks,"gate_pass":all(checks.values())}
    write_json(out/"V2_REPORT.json",report); return report

def load_cal_label_free(root):
    if str(root) not in sys.path: sys.path.insert(0,str(root))
    from src.gemini.huy_d1_aiteam50_soft_admission_v1.common import load_cal_data_label_free
    _docs,queries,blocks,ids,pools,_views,_scores,_types,_cite=load_cal_data_label_free()
    ids=[str(q) for q in ids]; questions={q:str(queries[q][0]) for q in ids}; pools={q:[str(d) for d in pools[q]] for q in ids}; blocks={str(k):[str(q) for q in v] for k,v in blocks.items()}
    base={str(q):[str(d) for d in r] for q,r in json.loads((root/"results/sol_high_rl/BASELINE_LOBO_PREDICTIONS.json").read_text(encoding="utf-8")).items()}
    e5={str(q):[str(d) for d in r] for q,r in json.loads((root/"results/manual/huy_cal600_adapted_e5_full_corpus_v1/CAL600_ADAPTED_E5_FULL_CORPUS_TOP150.json").read_text(encoding="utf-8")).items()}
    ai={str(q):[str(d) for d in r] for q,r in json.loads((root/"results/sol_high_rl/AITEAM_FT_FULL_CORPUS_TOP50.json").read_text(encoding="utf-8")).items()}
    if not(set(base)==set(e5)==set(ai)==set(pools)==set(ids)): raise RuntimeError("CAL population mismatch")
    return ids,questions,blocks,pools,base,e5,ai

def cal_stage(root,out,score_db,model_path,batch_size):
    ids,questions,blocks,pools,base,e5,ai=load_cal_label_free(root); universes,overlap=required_universe(ids,pools,base,e5,ai)
    semantic=score_universes(root=root,dataset="cal",ids=ids,questions=questions,universes=universes,cache_path=score_db,model_path=model_path,batch_size=batch_size)
    modified,actions=apply_crossover(ids,base,overlap,semantic)
    action_doc={"schema":"manual.overlap_semantic_crossover_v1.cal_actions","status":"SEALED_BEFORE_CAL_GOLD","candidate_depth":DEPTH,
                "sources":["OOF adapted VietLegal-E5 full corpus","AITeamVN-FT full corpus"],"semantic_model":str(model_path),
                "action_rule":"intersection-novel + frozen base-Jina lexical crossover; exactly one eligible else abstain",
                "actions_count":len(actions),"queries_with_overlap_novel":sum(bool(overlap[q]) for q in ids),
                "total_overlap_novel_candidates":sum(len(overlap[q]) for q in ids),"overlap_novel":overlap,"actions":actions,"predictions":modified}
    write_json(out/"CAL_ACTIONS_LABEL_FREE.json",action_doc); action_sha=sha256_file(out/"CAL_ACTIONS_LABEL_FREE.json")
    from src.gemini.huy_d1_aiteam50_soft_admission_v1.common import load_cal_gold_labels
    gold,reveal=load_cal_gold_labels(ids)
    bm=metric_map(base,gold,ids); mm=metric_map(modified,gold,ids)
    if abs(bm["recall_at_5"]-EXPECTED_D1_R5)>1e-12: raise RuntimeError(f"D1 R parity fail {bm}")
    if abs(bm["precision_at_5"]-EXPECTED_D1_P5)>5e-10: raise RuntimeError(f"D1 P parity fail {bm}")
    paired=outcomes(ids,base,modified,gold); oracle=overlap_oracle(ids,pools,base,overlap,gold)
    block_delta={}; block_metrics={}
    for b,qids in blocks.items():
        bb=metric_map(base,gold,qids); mm_b=metric_map(modified,gold,qids); block_metrics[b]={"baseline":bb,"modified":mm_b}; block_delta[b]=mm_b["recall_at_5"]-bb["recall_at_5"]
    single=[q for q in ids if len(gold[q])==1]; multi=[q for q in ids if len(gold[q])>1]
    sb,sm=metric_map(base,gold,single),metric_map(modified,gold,single); mb,mmu=metric_map(base,gold,multi),metric_map(modified,gold,multi)
    dr=mm["recall_at_5"]-bm["recall_at_5"]; dp=mm["precision_at_5"]-bm["precision_at_5"]
    gates={"recall_positive":dr>0,"precision_no_decrease":dp>=-1e-12,"wins_gt_losses":paired["beneficial"]>paired["harmful"],"no_block_decrease":all(d>=-1e-12 for d in block_delta.values())}
    verdict="STRONG_PROMOTE_OVERLAP_SEMANTIC_CROSSOVER_V1" if all(gates.values()) and mm["recall_at_5"]>=0.96 else ("PROMISING_OVERLAP_SEMANTIC_CROSSOVER_V1" if dr>0 and dp>=-1e-12 and paired["beneficial"]>paired["harmful"] else "KILL_OVERLAP_SEMANTIC_CROSSOVER_V1")
    report={"schema":"manual.overlap_semantic_crossover_v1.cal","gold_reveal_time":reveal,"action_artifact_sha256":action_sha,"overlap_oracle":oracle,
            "baseline":bm,"modified":mm,"delta":{"recall_at_5":dr,"precision_at_5":dp,"single_gold":sm["recall_at_5"]-sb["recall_at_5"],
            "multi_gold":mmu["recall_at_5"]-mb["recall_at_5"],"blocks":block_delta},"paired_actions":paired,"promotion_gates":gates,"verdict":verdict}
    write_json(out/"CAL_REPORT.json",report); return report

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--repo-root",type=Path,required=True); ap.add_argument("--batch-size",type=int,default=64); args=ap.parse_args()
    root=args.repo_root.resolve()
    if str(root) not in sys.path: sys.path.insert(0,str(root))
    if str(root/"src") not in sys.path: sys.path.insert(0,str(root/"src"))
    out=root/"results/manual/huy_overlap_semantic_crossover_v1"; out.mkdir(parents=True,exist_ok=True)
    model_path=root/"cache/research_v2_forensic/models/jina-reranker-v2-base-multilingual"
    if not (model_path/"model.safetensors").is_file(): raise FileNotFoundError(model_path)
    score_db=out/"SEMANTIC_SCORES.sqlite"
    print("[1/6] Strict-V2 overlap frontier + semantic scoring...",flush=True)
    v2=v2_stage(root,out,score_db,model_path,args.batch_size)
    p=v2["paired_actions"]
    print(f"[2/6] V2 {v2['baseline']['recall_at_5']:.10f} -> {v2['modified']['recall_at_5']:.10f} ({v2['delta_recall_at_5']:+.10f})",flush=True)
    print(f"  actions={p['actions']} beneficial/harmful/neutral={p['beneficial']}/{p['harmful']}/{p['neutral']} precision={p['intervention_precision_excluding_neutral']:.3f}",flush=True)
    print(f"  gate={v2['gate_pass']} {v2['gate_checks']}",flush=True)
    if not v2["gate_pass"]:
        print("[3/6] STOP: Strict-V2 gate failed; CAL gold NOT read.")
        print("="*92); print("Verdict : KILL_AT_STRICT_V2_GATE"); print(f"Report  : {out/'V2_REPORT.json'}"); print("="*92); return
    print("[3/6] Strict-V2 gate PASS.",flush=True)
    print("[4/6] Scoring CAL overlap universe label-free...",flush=True)
    cal=cal_stage(root,out,score_db,model_path,args.batch_size)
    print("[5/6] CAL actions sealed before gold reveal.",flush=True); print("[6/6] DONE"); print("="*92)
    print(f"D1        R@5={cal['baseline']['recall_at_5']:.10f} P@5={cal['baseline']['precision_at_5']:.10f}")
    print(f"Crossover R@5={cal['modified']['recall_at_5']:.10f} P@5={cal['modified']['precision_at_5']:.10f}")
    print(f"Delta     R={cal['delta']['recall_at_5']:+.10f} P={cal['delta']['precision_at_5']:+.10f}")
    print(f"Single Δ  {cal['delta']['single_gold']:+.10f} | Multi Δ {cal['delta']['multi_gold']:+.10f}")
    print("Blocks    "+" ".join(f"{b}:{d:+.6f}" for b,d in sorted(cal["delta"]["blocks"].items())))
    p=cal["paired_actions"]; print(f"Actions   {p['actions']} | beneficial={p['beneficial']} harmful={p['harmful']} neutral={p['neutral']}")
    print(f"Overlap   recoverable_gold={cal['overlap_oracle']['recoverable_gold_occurrences']} one_swap_oracle_delta={cal['overlap_oracle']['one_swap_oracle_delta']:+.10f}")
    print(f"Verdict   {cal['verdict']}"); print(f"Report    {out/'CAL_REPORT.json'}"); print("="*92)

if __name__=="__main__":
    main()
