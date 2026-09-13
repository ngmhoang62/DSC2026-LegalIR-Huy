"""Frozen mT5-mMARCO scorer for the sealed Research V2 Fold-0 falsifier.

The implementation follows PyGaggle MonoT5: one decoder step, log-softmax over
the checkpoint-specific `no` and `yes` SentencePiece tokens, with the exact T5
query/document prompt.  SQLite makes the expensive inference resumable.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import sqlite3
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch


NO_TOKEN = "\u2581no"
YES_TOKEN = "\u2581yes"
PROMPT = "Query: {query} Document: {document} Relevant:"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)


def canonical_answers(train: dict, exclusions: list[dict]) -> dict[str, set[str]]:
    alias = {str(row["doc_id"]): str(row["duplicate_retained_id"])
             for row in exclusions if row.get("duplicate_retained_id")}
    empty = {str(row["doc_id"]) for row in exclusions
             if "empty_passage" in row.get("reasons", [])}
    return {str(q): {alias.get(str(x), str(x)) for x in row["answer"]} - empty
            for q, row in train.items()}


def load_model(model_path: Path, dtype: torch.dtype, adapter_path: Path | None = None):
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    vocab = tokenizer.get_vocab()
    if vocab.get(NO_TOKEN) != 375 or vocab.get(YES_TOKEN) != 36339:
        raise RuntimeError("checkpoint prediction-token contract changed")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path, dtype=dtype)
    if adapter_path is not None:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)
    model.eval().to("cuda")
    return model, tokenizer


@torch.inference_mode()
def score_texts(model, tokenizer, texts: list[str], batch_size: int) -> list[float]:
    out: list[float] = []
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    for start in range(0, len(texts), batch_size):
        encoded = tokenizer(texts[start:start + batch_size], padding="longest",
                            truncation=True, max_length=512, return_tensors="pt")
        encoded = {key: value.to(device) for key, value in encoded.items()}
        decoder_input_ids = torch.full(
            (encoded["input_ids"].shape[0], 1),
            int(model.config.decoder_start_token_id), dtype=torch.long, device=device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=dtype == torch.float16):
            logits = model(**encoded, decoder_input_ids=decoder_input_ids,
                           use_cache=False).logits[:, 0, [375, 36339]]
        scores = torch.log_softmax(logits.float(), dim=1)[:, 1]
        out.extend(float(x) for x in scores.cpu())
    return out


def fold0_rows(args: argparse.Namespace) -> list[dict]:
    folds = json.loads(args.folds.read_text(encoding="utf-8"))
    fold0 = set(map(str, folds["folds"]["fold_0"]))
    rows = [row for row in read_jsonl(args.pool) if str(row["qid"]) in fold0]
    if len(rows) != 1398 or any(str(row["fold"]) != "fold_0" for row in rows):
        raise RuntimeError(f"Fold0 pool mismatch: {len(rows)}")
    return rows


def build_inputs(args: argparse.Namespace, rows: list[dict]):
    sys.path.insert(0, str(args.huy_root))
    from benchmark_jina_reranker_holdouts import top_passages
    from run_burst_expanded_fusion_submission import DocumentStore
    docs = DocumentStore(sorted(args.contexts.glob("context_*.json")), cache_size=9000)
    for row in rows:
        query = str(row["query"])
        owners, inputs = [], []
        for doc_id in map(str, row["doc_ids"]):
            passages = list(top_passages(query, docs[doc_id], count=2))
            if not passages:
                raise RuntimeError(f"no passage for {row['qid']} {doc_id}")
            for passage_index, passage in enumerate(passages):
                owners.append((doc_id, passage_index))
                inputs.append(PROMPT.format(query=query, document=str(passage)))
        yield row, owners, inputs


def deterministic_smoke_rows(rows: list[dict], n: int) -> list[dict]:
    return sorted(rows, key=lambda row: hashlib.sha256(str(row["qid"]).encode()).hexdigest())[:n]


def parity_and_throughput(args: argparse.Namespace) -> None:
    rows = deterministic_smoke_rows(fold0_rows(args), args.smoke_queries)
    rendered = list(build_inputs(args, rows))
    flat = [text for _, _, texts in rendered for text in texts]
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    fp32_model, tokenizer = load_model(args.model, torch.float32)
    fp32 = score_texts(fp32_model, tokenizer, flat, args.fp32_batch)
    fp32_seconds = time.perf_counter() - started
    del fp32_model
    gc.collect(); torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    fp16_model, tokenizer = load_model(args.model, torch.float16)
    fp16 = score_texts(fp16_model, tokenizer, flat, args.batch_size)
    fp16_seconds = time.perf_counter() - started
    peak = torch.cuda.max_memory_allocated() / 2**20
    max_abs = max(abs(a - b) for a, b in zip(fp32, fp16))
    mean_abs = float(np.mean(np.abs(np.asarray(fp32) - np.asarray(fp16))))
    parent_top5_equal = True
    cursor = 0
    for row, owners, texts in rendered:
        n = len(texts)
        ranks = []
        for values in (fp32[cursor:cursor+n], fp16[cursor:cursor+n]):
            parent = {}
            for (doc, _), value in zip(owners, values):
                parent[doc] = max(parent.get(doc, -math.inf), value)
            ranks.append(sorted(parent, key=lambda doc: (-parent[doc], doc))[:5])
        parent_top5_equal &= ranks[0] == ranks[1]
        cursor += n
    fp16_sequences_per_second = len(flat) / fp16_seconds
    fp32_sequences_per_second = len(flat) / fp32_seconds
    total_sequences = sum(len(row["doc_ids"]) * 2 for row in fold0_rows(args))
    fp16_safe = parent_top5_equal and max_abs <= 0.01
    report = {
        "schema_version": "dsc2026.research_v2.monot5_parity_throughput.v1",
        "status": "PASS_FP16" if fp16_safe else "PASS_FP32_REQUIRED",
        "authorized_dtype": "float16" if fp16_safe else "float32",
        "smoke_queries": len(rows), "sequences": len(flat),
        "fp32_seconds_including_load": fp32_seconds,
        "fp16_seconds_including_load": fp16_seconds,
        "fp16_sequences_per_second_including_load": fp16_sequences_per_second,
        "fp32_sequences_per_second_including_load": fp32_sequences_per_second,
        "estimated_full_fold0_seconds_conservative": total_sequences /
            (fp16_sequences_per_second if fp16_safe else fp32_sequences_per_second),
        "fp16_peak_allocated_mib": peak,
        "score_max_abs_error": max_abs, "score_mean_abs_error": mean_abs,
        "parent_top5_identical": parent_top5_equal,
        "token_ids": {"no": 375, "yes": 36339},
        "model_files": {p.name: sha256(p) for p in sorted(args.model.iterdir()) if p.is_file()},
        "inputs": {"folds": sha256(args.folds), "pool": sha256(args.pool)},
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / "MONOT5_PARITY_THROUGHPUT.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


def fp32_batch_parity(args: argparse.Namespace) -> None:
    """Prove that a larger inference batch preserves the FP32 ranking."""
    rows = deterministic_smoke_rows(fold0_rows(args), args.smoke_queries)
    rendered = list(build_inputs(args, rows))
    flat = [text for _, _, texts in rendered for text in texts]
    model, tokenizer = load_model(args.model, torch.float32)
    reference = score_texts(model, tokenizer, flat, 4)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    candidate = score_texts(model, tokenizer, flat, args.fp32_batch)
    seconds = time.perf_counter() - started
    peak = torch.cuda.max_memory_allocated() / 2**20
    cursor = 0; top5_equal = True
    for row, owners, texts in rendered:
        n = len(texts); rankings = []
        for values in (reference[cursor:cursor+n], candidate[cursor:cursor+n]):
            parent = {}
            for (doc, _), value in zip(owners, values):
                parent[doc] = max(parent.get(doc, -math.inf), value)
            rankings.append(sorted(parent, key=lambda doc: (-parent[doc], doc))[:5])
        top5_equal &= rankings[0] == rankings[1]; cursor += n
    errors = np.abs(np.asarray(reference) - np.asarray(candidate))
    total_sequences = sum(len(row["doc_ids"]) * 2 for row in fold0_rows(args))
    report = {
        "schema_version": "dsc2026.research_v2.monot5_fp32_batch_parity.v1",
        "status": "PASS" if top5_equal and float(errors.max()) <= 1e-4 else "FAIL",
        "reference_batch": 4, "candidate_batch": args.fp32_batch,
        "queries": len(rows), "sequences": len(flat),
        "score_max_abs_error": float(errors.max()),
        "score_mean_abs_error": float(errors.mean()),
        "parent_top5_identical": top5_equal,
        "seconds_excluding_load": seconds,
        "sequences_per_second": len(flat) / seconds,
        "estimated_full_fold0_seconds": total_sequences / (len(flat) / seconds),
        "peak_allocated_mib": peak,
    }
    path = args.output_dir / "MONOT5_FP32_BATCH_PARITY.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


def adapter_smoke(args: argparse.Namespace) -> None:
    if args.adapter is None:
        raise RuntimeError("--adapter is required")
    rendered = list(build_inputs(args, deterministic_smoke_rows(fold0_rows(args), 1)))
    row, owners, texts = rendered[0]
    model, tokenizer = load_model(args.model, torch.float32, args.adapter)
    torch.cuda.reset_peak_memory_stats(); started=time.perf_counter()
    values=score_texts(model,tokenizer,texts,args.fp32_batch)
    parent={}
    for (doc,_),value in zip(owners,values): parent[doc]=max(parent.get(doc,-math.inf),value)
    report={"schema_version":"dsc2026.research_v2.monot5_adapter_smoke.v1",
            "status":"PASS" if len(parent)==len(row["doc_ids"]) and all(math.isfinite(x) for x in values) else "FAIL",
            "qid":str(row["qid"]),"passages":len(values),"parents":len(parent),
            "seconds":time.perf_counter()-started,"peak_allocated_mib":torch.cuda.max_memory_allocated()/2**20,
            "top5":sorted(parent,key=lambda d:(-parent[d],d))[:5],
            "adapter_files":{p.name:sha256(p) for p in args.adapter.iterdir() if p.is_file()}}
    path=args.output_dir/"MONOT5_ADAPTER_INFERENCE_SMOKE.json";path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False,indent=2))


def open_db(path: Path) -> sqlite3.Connection:
    db = sqlite3.connect(path)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("CREATE TABLE IF NOT EXISTS scores(qid TEXT,doc_id TEXT,passage_index INTEGER,score REAL,PRIMARY KEY(qid,doc_id,passage_index))")
    db.execute("CREATE TABLE IF NOT EXISTS progress(qid TEXT PRIMARY KEY,seconds REAL,sequences INTEGER,peak_mib REAL)")
    db.commit()
    return db


def run_score(args: argparse.Namespace) -> None:
    rows = fold0_rows(args)
    parity = json.loads((args.output_dir / "MONOT5_PARITY_THROUGHPUT.json").read_text(encoding="utf-8"))
    if parity["status"] not in {"PASS_FP16", "PASS_FP32_REQUIRED"} or parity["estimated_full_fold0_seconds_conservative"] > 14400:
        raise RuntimeError("parity/runtime gate does not authorize full Fold0")
    db = open_db(args.score_db)
    complete = {str(row[0]) for row in db.execute("SELECT qid FROM progress")}
    dtype = torch.float16 if parity["authorized_dtype"] == "float16" else torch.float32
    batch_size = args.batch_size if dtype == torch.float16 else args.fp32_batch
    model, tokenizer = load_model(args.model, dtype, args.adapter)
    started = time.perf_counter(); newly_done = 0
    torch.cuda.reset_peak_memory_stats()
    for row, owners, inputs in build_inputs(args, (row for row in rows if str(row["qid"]) not in complete)):
        before = time.perf_counter()
        values = score_texts(model, tokenizer, inputs, batch_size)
        if len(values) != len(owners) or not all(math.isfinite(x) for x in values):
            raise RuntimeError(f"invalid scores for {row['qid']}")
        elapsed = time.perf_counter() - before
        peak = torch.cuda.max_memory_allocated() / 2**20
        qid = str(row["qid"])
        with db:
            db.executemany("INSERT INTO scores VALUES(?,?,?,?)",
                           [(qid, doc, index, value) for (doc, index), value in zip(owners, values)])
            db.execute("INSERT INTO progress VALUES(?,?,?,?)", (qid, elapsed, len(inputs), peak))
        newly_done += 1
        if newly_done % 10 == 0:
            total_done = len(complete) + newly_done
            rate = (time.perf_counter() - started) / newly_done
            print(f"done={total_done}/1398 qsec={rate:.2f} eta_min={(1398-total_done)*rate/60:.1f} peak_mib={peak:.1f}", flush=True)
    counts = db.execute("SELECT COUNT(DISTINCT qid),COUNT(*),SUM(sequences),SUM(seconds),MAX(peak_mib) FROM progress").fetchone()
    score_rows = db.execute("SELECT COUNT(*) FROM scores").fetchone()[0]
    db.close()
    print(json.dumps({"progress": counts, "score_rows": score_rows}, indent=2), flush=True)


def query_metric(pred: list[str], gold: set[str]) -> tuple[float, float]:
    hits = len(set(pred[:5]) & gold)
    return hits / len(gold), hits / 5.0


def evaluate(args: argparse.Namespace) -> None:
    rows = fold0_rows(args)
    train = json.loads(args.train.read_text(encoding="utf-8"))
    answers = canonical_answers(train, json.loads(args.exclusions.read_text(encoding="utf-8")))
    anchor = {}
    for row in read_jsonl(args.anchor):
        top5 = row.get("top5", row.get("fused_top5"))
        if top5 is None:
            raise RuntimeError("unsupported anchor prediction schema")
        anchor[str(row["qid"])] = list(map(str, top5))
    db = sqlite3.connect(f"file:{args.score_db.resolve().as_posix()}?mode=ro", uri=True)
    complete = set(str(x[0]) for x in db.execute("SELECT qid FROM progress"))
    if complete != {str(row["qid"]) for row in rows}:
        raise RuntimeError("incomplete Fold0 scoring")
    scores = {(str(q), str(d)): float(s) for q, d, s in db.execute(
        "SELECT qid,doc_id,MAX(score) FROM scores GROUP BY qid,doc_id")}
    runtime = db.execute("SELECT SUM(seconds),SUM(sequences),MAX(peak_mib) FROM progress").fetchone()
    db.close()
    mono_values=[]; anchor_values=[]; union_values=[]; predictions=[]
    wins=losses=changed=0; exclusive_mono=exclusive_anchor=0
    by_gold_count={"single": [[],[],[]], "multi": [[],[],[]]}
    gold_rank_buckets=Counter()
    for row in rows:
        qid=str(row["qid"]); docs=list(map(str,row["doc_ids"])); gold=answers[qid]
        ordered=sorted(docs,key=lambda d:(-scores[(qid,d)],d)); pred=ordered[:5]; base=anchor[qid]
        mr,mp=query_metric(pred,gold); ar,ap=query_metric(base,gold)
        union=set(base)|set(pred)
        ur=len(union & gold)/len(gold)
        mono_values.append((mr,mp)); anchor_values.append((ar,ap)); union_values.append(ur)
        wins += mr>ar; losses += mr<ar; changed += pred!=base
        exclusive_mono += bool(set(pred)&gold) and not bool(set(base)&gold)
        exclusive_anchor += bool(set(base)&gold) and not bool(set(pred)&gold)
        key="single" if len(gold)==1 else "multi"
        by_gold_count[key][0].append(mr); by_gold_count[key][1].append(ar); by_gold_count[key][2].append(ur)
        for doc in gold:
            rank=ordered.index(doc)+1 if doc in ordered else None
            bucket="missing" if rank is None else "1-5" if rank<=5 else "6-10" if rank<=10 else "11-20" if rank<=20 else "21-50" if rank<=50 else "51+"
            gold_rank_buckets[bucket]+=1
        predictions.append({"qid":qid,"top5":pred,"ranking":ordered,"scores":[scores[(qid,d)] for d in ordered]})
    mono_r=float(np.mean([x[0] for x in mono_values])); anchor_r=float(np.mean([x[0] for x in anchor_values])); union_r=float(np.mean(union_values))
    union_delta=union_r-anchor_r
    if mono_r>=0.900 and union_delta>=0.006:
        verdict="PASS_STANDALONE"
    elif mono_r>=0.860 and union_delta>=0.012:
        verdict="PASS_ORTHOGONAL"
    elif union_delta<0.005 or mono_r<0.840:
        verdict="KILL"
    else:
        verdict="INCONCLUSIVE_NO_TUNING"
    args.output_dir.mkdir(parents=True,exist_ok=True)
    pred_path=args.output_dir/"MONOT5_FOLD0_PREDICTIONS.jsonl"
    with pred_path.open("w",encoding="utf-8",newline="\n") as sink:
        for row in sorted(predictions,key=lambda x:int(x["qid"])):
            sink.write(canonical_json(row)+"\n")
    report={
        "schema_version":"dsc2026.research_v2.monot5_fold0_report.v1","status":"COMPLETE","verdict":verdict,
        "metrics":{"queries":len(rows),"monot5_recall_at_5":mono_r,"monot5_precision_at_5":float(np.mean([x[1] for x in mono_values])),
                   "current_anchor_recall_at_5":anchor_r,"top5_set_union_oracle":union_r,"union_oracle_delta":union_delta,
                   "wins_vs_anchor":wins,"losses_vs_anchor":losses,"ties":len(rows)-wins-losses,"changed_top5_sets":changed,
                   "exclusive_query_hits":{"monot5_only":exclusive_mono,"anchor_only":exclusive_anchor},
                   "gold_rank_buckets":dict(gold_rank_buckets),
                   "single_multi":{k:{"queries":len(v[0]),"monot5":float(np.mean(v[0])),"anchor":float(np.mean(v[1])),"union":float(np.mean(v[2]))} for k,v in by_gold_count.items()}},
        "runtime":{"score_seconds":runtime[0],"sequences":runtime[1],"peak_allocated_mib":runtime[2]},
        "integrity":{"folds_sha256":sha256(args.folds),"pool_sha256":sha256(args.pool),"score_db_sha256":sha256(args.score_db),
                     "predictions_sha256":sha256(pred_path),"model_weights_sha256":sha256(args.model/"pytorch_model.bin"),
                     "preregistration_sha256":sha256(args.output_dir/"MONOT5_FOLD0_PREREGISTRATION.json"),
                     "amendment_sha256":sha256(args.output_dir/"MONOT5_PREMETRIC_CONTRACT_AMENDMENT.json")},
        "anti_rescue":"No prompt/model/length/passage/aggregation/fusion grid is authorized."
    }
    out=args.output_dir/"MONOT5_FOLD0_REPORT.json"; out.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)


def parser() -> argparse.ArgumentParser:
    root=Path(__file__).resolve().parents[2]; workspace=root.parent
    p=argparse.ArgumentParser(); p.add_argument("stage",choices=("parity","batch-parity","adapter-smoke","score","evaluate"))
    p.add_argument("--model",type=Path,default=root/"cache/research_v2_open_rl/models/mt5-base-mmarco-v2")
    p.add_argument("--adapter",type=Path,default=None)
    p.add_argument("--output-dir",type=Path,default=root/"results/research_v2_open_rl")
    p.add_argument("--score-db",type=Path,default=root/"cache/research_v2_open_rl/monot5_fold0_scores.sqlite")
    p.add_argument("--pool",type=Path,default=root/"results/research_v2_forensic/V2_CANDIDATE_POOL.jsonl")
    p.add_argument("--folds",type=Path,default=root/"results/research_v2_forensic/V2_FOLDS.json")
    p.add_argument("--train",type=Path,default=workspace/"LegalIR/public_test_dataset/train.json")
    p.add_argument("--exclusions",type=Path,default=workspace/"LegalIR/cache/final_preprocessed_v2/exclusions.json")
    p.add_argument("--contexts",type=Path,default=workspace/"LegalIR/cache/final_preprocessed_v2/contexts")
    p.add_argument("--anchor",type=Path,default=root/"results/research_v2_post_e5/V2_ADAPTED_E5_LAL_EQUAL_RRF32_PREDICTIONS.jsonl")
    p.add_argument("--huy-root",type=Path,default=root); p.add_argument("--batch-size",type=int,default=16)
    p.add_argument("--fp32-batch",type=int,default=4); p.add_argument("--smoke-queries",type=int,default=3)
    return p


if __name__ == "__main__":
    args=parser().parse_args(); args.output_dir.mkdir(parents=True,exist_ok=True); args.score_db.parent.mkdir(parents=True,exist_ok=True)
    if args.stage=="parity": parity_and_throughput(args)
    elif args.stage=="batch-parity": fp32_batch_parity(args)
    elif args.stage=="adapter-smoke": adapter_smoke(args)
    elif args.stage=="score": run_score(args)
    else: evaluate(args)
