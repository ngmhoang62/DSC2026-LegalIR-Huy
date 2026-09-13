"""Bounded strict-V2 Fold-0 MonoT5 domain-adaptation pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


NO_ID = 375
YES_ID = 36339
PROMPT = "Query: {query} Document: {document} Relevant:"


def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda:stream.read(8<<20),b""): h.update(block)
    return h.hexdigest()


def read_jsonl(path: Path):
    with path.open("r",encoding="utf-8") as stream:
        for line in stream: yield json.loads(line)


def seed_all(seed:int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def select_groups(args):
    folds=json.loads(args.folds.read_text(encoding="utf-8"))["folds"]
    fold0=set(map(str,folds["fold_0"]))
    manifest=json.loads(args.groups_manifest.read_text(encoding="utf-8"))
    excluded=set(map(str,manifest["held_fold_duplicate_exclusions"]["fold_0"]))
    eligible=[]
    for row in read_jsonl(args.groups):
        qid=str(row["qid"])
        if qid not in fold0 and qid not in excluded:
            eligible.append(row)
    if len(eligible)!=5586: raise RuntimeError(f"train isolation mismatch {len(eligible)}")
    eligible.sort(key=lambda row:hashlib.sha256(str(row["qid"]).encode()).hexdigest())
    selected=eligible[:args.max_groups]
    if set(str(x["qid"]) for x in selected)&(fold0|excluded): raise RuntimeError("held/duplicate leakage")
    return selected,excluded


def load_model(args):
    from peft import LoraConfig,get_peft_model
    from transformers import AutoModelForSeq2SeqLM,AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(args.model,use_fast=False)
    vocab=tokenizer.get_vocab()
    if vocab.get("\u2581no")!=NO_ID or vocab.get("\u2581yes")!=YES_ID: raise RuntimeError("token contract")
    # FP16 is numerically unsafe for this checkpoint even at frozen inference
    # (it changes parent Top-5).  Keep the scientific computation in FP32.
    base=AutoModelForSeq2SeqLM.from_pretrained(args.model,dtype=torch.float32)
    base.config.use_cache=False
    if args.gradient_checkpointing:
        base.gradient_checkpointing_enable(); base.enable_input_require_grads()
    config=LoraConfig(r=8,lora_alpha=16,lora_dropout=.05,bias="none",target_modules=["q","v"],task_type="SEQ_2_SEQ_LM")
    model=get_peft_model(base,config).to("cuda"); model.train()
    return model,tokenizer


def group_loss(model,tokenizer,row,epoch):
    positives=list(row["positives"])
    negatives=list(row["negatives"])[2*epoch:2*epoch+2]
    if not positives or len(negatives)!=2: raise RuntimeError(f"invalid group {row['qid']}")
    parents=[]; texts=[]; labels=[]
    for label,items in ((1,positives),(0,negatives)):
        for parent in items:
            owner=len(parents); passages=list(parent["passages"])
            if not passages: raise RuntimeError("empty passages")
            parents.append((label,owner,len(passages)))
            for passage in passages:
                labels.append((label,owner)); texts.append(PROMPT.format(query=row["query"],document=passage))
    batch=tokenizer(texts,padding="longest",truncation=True,max_length=512,return_tensors="pt")
    batch={k:v.to("cuda") for k,v in batch.items()}
    decoder=torch.full((len(texts),1),int(model.config.decoder_start_token_id),dtype=torch.long,device="cuda")
    logits=model(**batch,decoder_input_ids=decoder,use_cache=False).logits[:,0,[NO_ID,YES_ID]].float()
    margins=logits[:,1]-logits[:,0]
    parent_margins=[]; parent_labels=[]
    for label,owner,_ in parents:
        indices=[i for i,(_,o) in enumerate(labels) if o==owner]
        parent_margins.append(margins[indices].max()); parent_labels.append(label)
    pos=torch.stack([m for m,y in zip(parent_margins,parent_labels) if y==1])
    neg=torch.stack([m for m,y in zip(parent_margins,parent_labels) if y==0])
    loss=.5*F.softplus(-pos).mean()+.5*F.softplus(neg).mean()
    return loss,{"positive_parents":len(pos),"negative_parents":len(neg),"passages":len(texts),"max_tokens":int(batch["attention_mask"].sum(1).max())}


def save_checkpoint(model,optimizer,scheduler,scaler,args,epoch,step,selection_sha,stats):
    path=args.output_dir/f"epoch_{epoch+1}"; adapter=path/"adapter"; adapter.mkdir(parents=True,exist_ok=True)
    model.save_pretrained(adapter,safe_serialization=True)
    torch.save({"epoch":epoch+1,"microstep":step,"optimizer":optimizer.state_dict(),"scheduler":scheduler.state_dict(),"scaler":scaler.state_dict(),"torch_rng":torch.get_rng_state(),"cuda_rng":torch.cuda.get_rng_state_all()},path/"training_state.pt")
    report={"status":"COMPLETE","epoch":epoch+1,"microsteps":step,"selection_sha256":selection_sha,"adapter_files":{p.name:sha256(p) for p in adapter.iterdir() if p.is_file()},"stats":stats}
    (path/"CHECKPOINT_MANIFEST.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    return path


def main():
    root=Path(__file__).resolve().parents[2]
    p=argparse.ArgumentParser()
    p.add_argument("--model",type=Path,default=root/"cache/research_v2_open_rl/models/mt5-base-mmarco-v2")
    p.add_argument("--groups",type=Path,default=root/"results/research_v2_forensic/V2_BOUNDARY_GROUPS.jsonl")
    p.add_argument("--groups-manifest",type=Path,default=root/"results/research_v2_forensic/V2_BOUNDARY_GROUPS_MANIFEST.json")
    p.add_argument("--folds",type=Path,default=root/"results/research_v2_forensic/V2_FOLDS.json")
    p.add_argument("--output-dir",type=Path,default=root/"results/research_v2_open_rl/monot5_domain_adapter_fold0_pilot")
    p.add_argument("--max-groups",type=int,default=1000);p.add_argument("--epochs",type=int,default=2)
    p.add_argument("--accumulation",type=int,default=8);p.add_argument("--lr",type=float,default=5e-5)
    p.add_argument("--seed",type=int,default=112);p.add_argument("--gradient-checkpointing",action=argparse.BooleanOptionalAction,default=True)
    args=p.parse_args(); args.output_dir.mkdir(parents=True,exist_ok=True)
    if args.epochs!=2 and args.max_groups>4: raise RuntimeError("scientific pilot requires exactly two epochs")
    seed_all(args.seed); groups,excluded=select_groups(args)
    selection={"schema_version":"dsc2026.research_v2.monot5_pilot_selection.v1","status":"SEALED","qids":[str(x["qid"]) for x in groups],"held_fold":"fold_0","duplicate_exclusions":sorted(excluded),"groups":len(groups),"source_sha256":sha256(args.groups)}
    selection_path=args.output_dir/"TRAINING_SELECTION.json"; selection_path.write_text(json.dumps(selection,ensure_ascii=False,indent=2),encoding="utf-8"); selection_sha=sha256(selection_path)
    model,tokenizer=load_model(args)
    trainable=sum(p.numel() for p in model.parameters() if p.requires_grad); total=sum(p.numel() for p in model.parameters())
    optimizer=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=args.lr,weight_decay=.01)
    updates=math.ceil(len(groups)*args.epochs/args.accumulation); warmup=max(1,int(.1*updates))
    from transformers import get_linear_schedule_with_warmup
    scheduler=get_linear_schedule_with_warmup(optimizer,warmup,updates)
    scaler=torch.amp.GradScaler("cuda",enabled=False); torch.cuda.reset_peak_memory_stats()
    started=time.perf_counter(); microstep=0; losses=[]; last=None
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(args.epochs):
        order=list(groups); random.Random(args.seed+epoch).shuffle(order)
        epoch_losses=[]; passages=0; max_tokens=0
        for index,row in enumerate(order):
            loss,info=group_loss(model,tokenizer,row,epoch)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss qid={row['qid']} epoch={epoch+1}")
            scaler.scale(loss/args.accumulation).backward(); microstep+=1
            epoch_losses.append(float(loss.detach())); passages+=info["passages"];max_tokens=max(max_tokens,info["max_tokens"])
            if microstep%args.accumulation==0 or (epoch==args.epochs-1 and index==len(order)-1):
                scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad),1.0)
                scaler.step(optimizer);scaler.update();scheduler.step();optimizer.zero_grad(set_to_none=True)
            if microstep%25==0:
                elapsed=time.perf_counter()-started; print(f"epoch={epoch+1} step={microstep} loss={np.mean(epoch_losses[-25:]):.4f} sec_step={elapsed/microstep:.2f} peak_mib={torch.cuda.max_memory_allocated()/2**20:.1f}",flush=True)
        stats={"mean_loss":float(np.mean(epoch_losses)),"passages":passages,"max_tokens":max_tokens,"elapsed_seconds":time.perf_counter()-started,"peak_allocated_mib":torch.cuda.max_memory_allocated()/2**20}
        losses.append(stats); last=save_checkpoint(model,optimizer,scheduler,scaler,args,epoch,microstep,selection_sha,stats)
    report={"schema_version":"dsc2026.research_v2.monot5_domain_training.v1","status":"COMPLETE","held_fold":"fold_0","train_groups":len(groups),"epochs":args.epochs,"trainable_parameters":trainable,"total_parameters":total,"trainable_fraction":trainable/total,"selection_sha256":selection_sha,"epoch_stats":losses,"final_checkpoint":str(last),"final_manifest_sha256":sha256(last/"CHECKPOINT_MANIFEST.json"),"scientific_contract":{"lora":"q/v r8 alpha16 dropout.05","loss":"balanced independent parent-max binary softplus","lr":args.lr,"accumulation":args.accumulation,"seed":args.seed}}
    (args.output_dir/"TRAINING_REPORT.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=="__main__": main()
