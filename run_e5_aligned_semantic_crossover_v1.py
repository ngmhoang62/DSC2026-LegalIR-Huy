#!/usr/bin/env python
from __future__ import annotations
import argparse, gc, hashlib, json, sqlite3, sys, time
from pathlib import Path
from typing import Any, Sequence, Set
import numpy as np, torch

DEPTH=50
D1_R=0.9569444444444444
D1_P=0.20566666666666666

def sha(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(8<<20),b""): h.update(b)
    return h.hexdigest()

def dump(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_suffix(p.suffix+".tmp")
    t.write_text(json.dumps(x,ensure_ascii=False,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    t.replace(p)

def readl(p):
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]

def rec(a,g): return len(set(a[:5])&g)/len(g)
def prec(a,g): return len(set(a[:5])&g)/5.0
def metrics(pred,gold,ids):
    return {"recall_at_5":float(np.mean([rec(pred[q],gold[q]) for q in ids])),
            "precision_at_5":float(np.mean([prec(pred[q],gold[q]) for q in ids]))}

def ckpt(root,fold):
    i=int(fold[-1])
    return (root/"results/research_v2_e5_transfer/research_v2_e5_transfer_fold0/training/epoch-2.pt"
            if i==0 else root/f"results/research_v2_e5_confirmation/fold_{i}/training/epoch-2.pt")

def overlap(a,b,pool):
    pa={str(d):i+1 for i,d in enumerate(a[:DEPTH])}
    pb={str(d):i+1 for i,d in enumerate(b[:DEPTH])}
    ps=set(map(str,pool))
    c=[d for d in pa if d in pb and d not in ps]
    return sorted(c,key=lambda d:(pa[d]+pb[d],max(pa[d],pb[d]),d))

def universes(ids,pools,base,a,b):
    u,o={},{}
    for q in ids:
        o[q]=overlap(a[q],b[q],pools[q])
        u[q]=list(dict.fromkeys(base[q][:5]+o[q]))
    return u,o

def open_cache(p):
    db=sqlite3.connect(p); db.execute("PRAGMA journal_mode=WAL")
    db.execute("CREATE TABLE IF NOT EXISTS ev(dataset TEXT,qid TEXT,doc TEXT,i1 INTEGER,i2 INTEGER,s1 REAL,s2 REAL,PRIMARY KEY(dataset,qid,doc))")
    db.execute("CREATE TABLE IF NOT EXISTS sem(dataset TEXT,qid TEXT,doc TEXT,score REAL,PRIMARY KEY(dataset,qid,doc))")
    db.commit(); return db

def stable_top2(idx,s):
    order=np.lexsort((idx,-s))
    i1=int(idx[order[0]]); s1=float(s[order[0]])
    if len(order)==1: return i1,i1,s1,s1
    return i1,int(idx[order[1]]),s1,float(s[order[1]])

def select_evidence(root,bundle,data,dataset,fold,qids,questions,u,cache,batch,full_rows=None):
    from research_v2_e5_transfer import e5_transfer_runner as core
    db=open_cache(cache)
    got={(str(q),str(d)):(int(i1),int(i2),float(s1),float(s2))
         for q,d,i1,i2,s1,s2 in db.execute("SELECT qid,doc,i1,i2,s1,s2 FROM ev WHERE dataset=?",(dataset,))}
    todo=[q for q in qids if any((q,d) not in got for d in u[q])]
    if todo:
        model=core.QueryEncoder(bundle/"vietlegal-e5",checkpoint_path=ckpt(root,fold)).eval()
        errs=[]; t0=time.perf_counter()
        try:
            for st in range(0,len(todo),batch):
                qs=todo[st:st+batch]
                with torch.inference_mode():
                    Q=model([questions[q] for q in qs]).detach().cpu().numpy()
                for q,qv in zip(qs,Q):
                    for d in u[q]:
                        if (q,d) in got: continue
                        if d not in data.doc_row: raise RuntimeError(f"{dataset}/{q}: {d} absent E5 bank")
                        pos=data.positions[data.doc_row[d]]
                        v=np.asarray(data.vectors[pos],dtype=np.float32)
                        v/=np.maximum(np.linalg.norm(v,axis=1,keepdims=True),1e-12)
                        got[(q,d)]=stable_top2(pos,v@np.asarray(qv,dtype=np.float32))
                        if full_rows and q in full_rows and d in full_rows[q]["adapted_order_top150"]:
                            j=full_rows[q]["adapted_order_top150"].index(d)
                            ref=float(full_rows[q]["adapted_scores_top150"][j])
                            x=got[(q,d)]
                            errs.append(abs((x[2]+x[3])/2-ref))
                    with db:
                        db.executemany("INSERT OR REPLACE INTO ev VALUES(?,?,?,?,?,?,?)",
                                       [(dataset,q,d,*got[(q,d)]) for d in u[q]])
                done=min(st+len(qs),len(todo))
                if done%(batch*5)==0 or done==len(todo):
                    print(f"    E5 evidence {dataset}/{fold}: {done}/{len(todo)} sec/q={(time.perf_counter()-t0)/done:.3f}",flush=True)
        finally:
            del model; gc.collect(); torch.cuda.empty_cache()
        me=max(errs) if errs else 0.0
        if me>1e-4: db.close(); raise RuntimeError(f"E5 parent parity fail {dataset}/{fold}: {me}")
        print(f"    E5 parent parity {dataset}/{fold}: max_abs={me:.3e}",flush=True)
    out={q:{d:got[(q,d)] for d in u[q]} for q in qids}
    db.close(); return out

def chunk_texts(dbpath,data,idxs):
    con=sqlite3.connect(f"file:{dbpath.resolve().as_posix()}?mode=ro&immutable=1",uri=True)
    if con.execute("PRAGMA integrity_check").fetchone()[0]!="ok": raise RuntimeError("evidence.sqlite corrupt")
    out={}
    arr=sorted(idxs)
    for st in range(0,len(arr),800):
        b=arr[st:st+800]; marks=",".join("?"*len(b))
        for idx,payload in con.execute(f"SELECT idx,payload FROM chunks WHERE idx IN ({marks})",b):
            row=json.loads(payload); idx=int(idx)
            if str(row["chunk_id"])!=str(data.chunk_ids[idx]): raise RuntimeError(f"chunk alignment fail idx={idx}")
            out[idx]=str(row["raw_text"])
    con.close()
    if set(out)!=set(arr): raise RuntimeError("missing aligned chunk texts")
    return out

def patch_tf():
    import transformers.models.xlm_roberta.modeling_xlm_roberta as m
    if hasattr(m,"create_position_ids_from_input_ids"): return
    def f(ids,padding_idx,past_key_values_length=0):
        mask=ids.ne(padding_idx).int(); p=(torch.cumsum(mask,dim=1)+past_key_values_length)*mask
        return p.long()+padding_idx
    m.create_position_ids_from_input_ids=f

def load_jina(path):
    patch_tf()
    from transformers import AutoModelForSequenceClassification,AutoTokenizer
    tok=AutoTokenizer.from_pretrained(path,trust_remote_code=True,fix_mistral_regex=True)
    model=AutoModelForSequenceClassification.from_pretrained(path,trust_remote_code=True,dtype=torch.float16).eval().to("cuda")
    model._tokenizer=tok; return model

def jina_score(dataset,ids,questions,u,ev,texts,cache,model_path,batch):
    db=open_cache(cache)
    got={(str(q),str(d)):float(s) for q,d,s in db.execute("SELECT qid,doc,score FROM sem WHERE dataset=?",(dataset,))}
    todo=[q for q in ids if any((q,d) not in got for d in u[q])]
    if todo:
        model=load_jina(model_path); t0=time.perf_counter(); torch.cuda.reset_peak_memory_stats()
        try:
            for n,q in enumerate(todo,1):
                docs=[d for d in u[q] if (q,d) not in got]; owners=[]; passages=[]
                for d in docs:
                    i1,i2,_,_=ev[q][d]
                    owners.append(d); passages.append(texts[i1])
                    if i2!=i1: owners.append(d); passages.append(texts[i2])
                raw=model.compute_score([(questions[q],p) for p in passages],batch_size=batch,max_length=512)
                if isinstance(raw,(float,int)): raw=[raw]
                par={}
                for d,s in zip(owners,[float(x) for x in raw]): par[d]=max(par.get(d,-1e9),s)
                if set(par)!=set(docs): raise RuntimeError(f"Jina coverage {dataset}/{q}")
                with db: db.executemany("INSERT OR REPLACE INTO sem VALUES(?,?,?,?)",[(dataset,q,d,s) for d,s in par.items()])
                got.update({(q,d):s for d,s in par.items()})
                if n%50==0 or n==len(todo):
                    print(f"    Jina {dataset}: {n}/{len(todo)} sec/q={(time.perf_counter()-t0)/n:.3f} peak={torch.cuda.max_memory_allocated()/2**20:.0f} MiB",flush=True)
        finally:
            del model; gc.collect(); torch.cuda.empty_cache()
    out={q:{d:got[(q,d)] for d in u[q]} for q in ids}; db.close(); return out

def apply(ids,base,o,sem):
    out={q:list(base[q][:5]) for q in ids}; actions={}
    for q in ids:
        if not o[q]: continue
        defender=base[q][4]; U=list(dict.fromkeys(base[q][:5]+o[q])); s=sem[q]
        order=sorted(U,key=lambda d:(-s[d],d)); r={d:i+1 for i,d in enumerate(order)}
        eligible=[c for c in o[q] if s[c]>s[defender] and r[c]<=5 and r[defender]>5]
        if len(eligible)==1:
            c=eligible[0]; out[q]=base[q][:4]+[c]
            actions[q]={"defender":defender,"challenger":c,"def_score":s[defender],"chal_score":s[c],"def_rank":r[defender],"chal_rank":r[c]}
    return out,actions

def paired(ids,base,mod,gold):
    b=h=n=0; details=[]
    for q in ids:
        if base[q][:5]==mod[q][:5]: continue
        d=rec(mod[q],gold[q])-rec(base[q],gold[q])
        if d>0: b+=1; typ="beneficial"
        elif d<0: h+=1; typ="harmful"
        else: n+=1; typ="neutral"
        details.append({"qid":q,"delta":d,"outcome":typ})
    return {"actions":len(details),"beneficial":b,"harmful":h,"neutral":n,
            "precision_ex_neutral":b/max(b+h,1),"details":details}

def oracle(ids,base,o,gold):
    mass=0.; rows=[]
    for q in ids:
        defender=base[q][4]; best=0.
        for c in o[q]:
            u=(int(c in gold[q])-int(defender in gold[q]))/len(gold[q]); best=max(best,u)
            if c in gold[q] and c not in set(base[q][:5]): rows.append({"qid":q,"doc_id":c})
        mass+=best
    return {"recoverable_gold_occurrences":len(rows),"one_swap_oracle_delta":mass/len(ids),"rows":rows}

def load_v2(root):
    from research_v2_e5_transfer.e5_transfer_runner import TransferData
    bundle=root/"cache/research_v2_e5_confirmation/bundle-v1"; data=TransferData(bundle)
    ids=sorted(data.questions,key=int); questions=dict(data.questions); gold={q:set(data.gold[q]) for q in ids}
    pools={q:list(data.pool[q]) for q in ids}; folds=dict(data.fold_for)
    a={str(r["qid"]):r for r in readl(root/"results/research_v2_post_e5/V2_ADAPTED_E5_LAL_EQUAL_RRF32_PREDICTIONS.jsonl")}
    base={q:[str(d) for d in a[q]["fused_top5"]] for q in ids}
    e5={}; full={}
    for i in range(5):
        for r in readl(root/f"results/research_v2_open_rl/fold_{i}/FULL_CORPUS_PREDICTIONS.jsonl"):
            q=str(r["qid"]); e5[q]=[str(d) for d in r["adapted_order_top150"]]; full[q]=r
    lal={q:[str(d) for d in data.sources[q]["lal"]] for q in ids}
    return bundle,data,ids,questions,gold,pools,folds,base,e5,lal,full

def run_v2(root,out,cache,model_path,e5bs,jbs):
    bundle,data,ids,questions,gold,pools,folds,base,e5,lal,full=load_v2(root)
    U,O=universes(ids,pools,base,e5,lal); ev={}
    for fold in [f"fold_{i}" for i in range(5)]:
        qs=[q for q in ids if folds[q]==fold]
        ev.update(select_evidence(root,bundle,data,"v2",fold,qs,questions,U,cache,e5bs,full))
    idxs={i for q in ids for d in U[q] for i in ev[q][d][:2]}
    edb=root.parent/"LegalIR/cache/exp112_task_adaptive_retrieval/evidence.sqlite"
    texts=chunk_texts(edb,data,idxs)
    print(f"  V2 overlap: queries={sum(bool(O[q]) for q in ids)} candidates={sum(len(O[q]) for q in ids)} oracle={oracle(ids,base,O,gold)}",flush=True)
    sem=jina_score("v2",ids,questions,U,ev,texts,cache,model_path,jbs)
    mod,_=apply(ids,base,O,sem); bm=metrics(base,gold,ids); mm=metrics(mod,gold,ids); po=paired(ids,base,mod,gold)
    ds=[]; fr={}
    for fold in [f"fold_{i}" for i in range(5)]:
        qs=[q for q in ids if folds[q]==fold]; fb=metrics(base,gold,qs); fm=metrics(mod,gold,qs); d=fm["recall_at_5"]-fb["recall_at_5"]; ds.append(d)
        fr[fold]={"delta":d,"paired":{k:v for k,v in paired(qs,base,mod,gold).items() if k!="details"}}
    dr=mm["recall_at_5"]-bm["recall_at_5"]
    checks={"delta_gte_0_001":dr>=.001,"wins_gt_losses":po["beneficial"]>po["harmful"],
            "four_folds_nonneg":sum(d>=0 for d in ds)>=4,"worst_fold_gte_minus_0_005":min(ds)>=-.005,
            "precision_gte_0_60":po["precision_ex_neutral"]>=.60}
    rep={"baseline":bm,"modified":mm,"delta_recall_at_5":dr,"paired":po,"folds":fr,"gate_checks":checks,"gate_pass":all(checks.values()),"oracle":oracle(ids,base,O,gold)}
    dump(out/"V2_REPORT.json",rep); return rep

def load_cal(root):
    from src.gemini.huy_d1_aiteam50_soft_admission_v1.common import load_cal_data_label_free
    _,queries,blocks,ids,pools,*_=load_cal_data_label_free()
    ids=[str(q) for q in ids]; questions={q:str(queries[q][0]) for q in ids}; pools={q:[str(d) for d in pools[q]] for q in ids}
    blocks={str(k):[str(q) for q in v] for k,v in blocks.items()}
    base={str(q):[str(d) for d in r] for q,r in json.loads((root/"results/sol_high_rl/BASELINE_LOBO_PREDICTIONS.json").read_text()).items()}
    e5={str(q):[str(d) for d in r] for q,r in json.loads((root/"results/manual/huy_cal600_adapted_e5_full_corpus_v1/CAL600_ADAPTED_E5_FULL_CORPUS_TOP150.json").read_text()).items()}
    ai={str(q):[str(d) for d in r] for q,r in json.loads((root/"results/sol_high_rl/AITEAM_FT_FULL_CORPUS_TOP50.json").read_text()).items()}
    return ids,questions,blocks,pools,base,e5,ai

def cal_full_rows(root):
    R={}
    b=root/"results/manual/huy_cal600_adapted_e5_full_corpus_v1"
    for i in range(5):
        p=b/f"fold_{i}/FULL_CORPUS_TOP150.jsonl"
        if p.is_file():
            for r in readl(p):
                R[str(r["qid"])]={"adapted_order_top150":[str(d) for d in r["order_top150"]],
                                  "adapted_scores_top150":[float(x) for x in r["scores_top150"]]}
    return R

def run_cal(root,out,cache,model_path,e5bs,jbs):
    from research_v2_e5_transfer.e5_transfer_runner import TransferData
    ids,questions,blocks,pools,base,e5,ai=load_cal(root)
    bundle=root/"cache/research_v2_e5_confirmation/bundle-v1"; data=TransferData(bundle)
    U,O=universes(ids,pools,base,e5,ai)
    fo=json.loads((bundle/"V2_FOLDS.json").read_text())["folds"]; fmap={str(q):f for f,qs in fo.items() for q in qs}
    full=cal_full_rows(root); ev={}
    for fold in [f"fold_{i}" for i in range(5)]:
        qs=[q for q in ids if fmap[q]==fold]
        ev.update(select_evidence(root,bundle,data,"cal",fold,qs,questions,U,cache,e5bs,full if full else None))
    idxs={i for q in ids for d in U[q] for i in ev[q][d][:2]}
    texts=chunk_texts(root.parent/"LegalIR/cache/exp112_task_adaptive_retrieval/evidence.sqlite",data,idxs)
    sem=jina_score("cal",ids,questions,U,ev,texts,cache,model_path,jbs); mod,actions=apply(ids,base,O,sem)
    dump(out/"CAL_ACTIONS_LABEL_FREE.json",{"status":"SEALED_BEFORE_GOLD","actions":actions,"predictions":mod})
    from src.gemini.huy_d1_aiteam50_soft_admission_v1.common import load_cal_gold_labels
    gold,_=load_cal_gold_labels(ids); bm=metrics(base,gold,ids); mm=metrics(mod,gold,ids)
    if abs(bm["recall_at_5"]-D1_R)>1e-12 or abs(bm["precision_at_5"]-D1_P)>5e-10: raise RuntimeError("D1 parity fail")
    po=paired(ids,base,mod,gold); dr=mm["recall_at_5"]-bm["recall_at_5"]; dp=mm["precision_at_5"]-bm["precision_at_5"]
    bd={}
    for b,qs in blocks.items(): bd[b]=metrics(mod,gold,qs)["recall_at_5"]-metrics(base,gold,qs)["recall_at_5"]
    s=[q for q in ids if len(gold[q])==1]; m=[q for q in ids if len(gold[q])>1]
    rep={"baseline":bm,"modified":mm,"delta":{"recall":dr,"precision":dp,"single":metrics(mod,gold,s)["recall_at_5"]-metrics(base,gold,s)["recall_at_5"],
         "multi":metrics(mod,gold,m)["recall_at_5"]-metrics(base,gold,m)["recall_at_5"],"blocks":bd},"paired":po,"oracle":oracle(ids,base,O,gold)}
    rep["verdict"]="STRONG_PROMOTE" if dr>0 and dp>=0 and po["beneficial"]>po["harmful"] and mm["recall_at_5"]>=.96 else ("PROMISING" if dr>0 and dp>=0 and po["beneficial"]>po["harmful"] else "KILL")
    dump(out/"CAL_REPORT.json",rep); return rep

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--repo-root",type=Path,required=True); ap.add_argument("--e5-batch-size",type=int,default=16); ap.add_argument("--jina-batch-size",type=int,default=64); a=ap.parse_args()
    root=a.repo_root.resolve(); sys.path[:0]=[str(root),str(root/"src")]
    if not torch.cuda.is_available(): raise RuntimeError("CUDA required")
    out=root/"results/manual/huy_e5_aligned_semantic_crossover_v1"; out.mkdir(parents=True,exist_ok=True); cache=out/"CACHE.sqlite"
    model=root/"cache/research_v2_forensic/models/jina-reranker-v2-base-multilingual"
    if not (model/"model.safetensors").is_file(): raise FileNotFoundError(model)
    if not (root.parent/"LegalIR/cache/exp112_task_adaptive_retrieval/evidence.sqlite").is_file(): raise FileNotFoundError("LegalIR evidence.sqlite")
    print("[1/5] Strict-V2 E5-aligned semantic crossover...",flush=True)
    v=run_v2(root,out,cache,model,a.e5_batch_size,a.jina_batch_size); p=v["paired"]
    print(f"[2/5] V2 {v['baseline']['recall_at_5']:.10f} -> {v['modified']['recall_at_5']:.10f} ({v['delta_recall_at_5']:+.10f})")
    print(f"  actions={p['actions']} beneficial/harmful/neutral={p['beneficial']}/{p['harmful']}/{p['neutral']} precision={p['precision_ex_neutral']:.3f}")
    print(f"  gate={v['gate_pass']} {v['gate_checks']}")
    if not v["gate_pass"]:
        print("[3/5] STOP: V2 gate failed; CAL gold NOT read."); print("="*92); print("Verdict : KILL_AT_STRICT_V2_GATE"); print(f"Report  : {out/'V2_REPORT.json'}"); print("="*92); return
    print("[3/5] V2 PASS; running CAL label-free evidence + Jina...",flush=True)
    c=run_cal(root,out,cache,model,a.e5_batch_size,a.jina_batch_size); p=c["paired"]
    print("[4/5] CAL actions sealed before gold reveal."); print("[5/5] DONE"); print("="*92)
    print(f"D1      R@5={c['baseline']['recall_at_5']:.10f} P@5={c['baseline']['precision_at_5']:.10f}")
    print(f"Aligned R@5={c['modified']['recall_at_5']:.10f} P@5={c['modified']['precision_at_5']:.10f}")
    print(f"Delta   R={c['delta']['recall']:+.10f} P={c['delta']['precision']:+.10f}")
    print(f"Single  {c['delta']['single']:+.10f} Multi {c['delta']['multi']:+.10f}")
    print("Blocks  "+" ".join(f"{k}:{v:+.6f}" for k,v in sorted(c["delta"]["blocks"].items())))
    print(f"Actions {p['actions']} beneficial={p['beneficial']} harmful={p['harmful']} neutral={p['neutral']}")
    print(f"Oracle  {c['oracle']}")
    print(f"Verdict {c['verdict']}"); print(f"Report  {out/'CAL_REPORT.json'}"); print("="*92)

if __name__=="__main__": main()
