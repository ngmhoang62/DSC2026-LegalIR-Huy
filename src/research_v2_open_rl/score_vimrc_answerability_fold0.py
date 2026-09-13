"""Frozen Vietnamese extractive-QA answerability scorer for V2 Fold 0.

One sealed mechanism only: locked lexical top-2 evidence, SQuAD2-style
best-context-span minus CLS no-answer margin, parent max, deterministic Top-5.
The SQLite cache makes the single expensive local pass resumable.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import sqlite3
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch


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


def fold0_rows(args: argparse.Namespace) -> list[dict]:
    folds = json.loads(args.folds.read_text(encoding="utf-8"))
    fold0 = set(map(str, folds["folds"]["fold_0"]))
    rows = [row for row in read_jsonl(args.pool) if str(row["qid"]) in fold0]
    if len(rows) != 1398 or any(str(row["fold"]) != "fold_0" for row in rows):
        raise RuntimeError(f"Fold0 pool mismatch: {len(rows)}")
    return rows


def deterministic_rows(rows: list[dict], n: int) -> list[dict]:
    return sorted(rows, key=lambda row: hashlib.sha256(str(row["qid"]).encode()).hexdigest())[:n]


def build_inputs(args: argparse.Namespace, rows: list[dict]):
    sys.path.insert(0, str(args.huy_root))
    from benchmark_jina_reranker_holdouts import top_passages
    from run_burst_expanded_fusion_submission import DocumentStore
    docs = DocumentStore(sorted(args.contexts.glob("context_*.json")), cache_size=9000)
    for row in rows:
        query = str(row["query"])
        owners: list[tuple[str, int]] = []
        passages: list[str] = []
        for doc_id in map(str, row["doc_ids"]):
            selected = list(top_passages(query, docs[doc_id], count=2))
            if not selected:
                raise RuntimeError(f"no passage for {row['qid']} {doc_id}")
            for passage_index, passage in enumerate(selected):
                owners.append((doc_id, passage_index))
                passages.append(str(passage))
        yield row, owners, [query] * len(passages), passages


def load_model(model_path: Path, dtype: torch.dtype, attention_implementation: str = "eager"):
    from transformers import AutoModelForQuestionAnswering, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if not tokenizer.is_fast:
        raise RuntimeError("fast tokenizer required for context-token masking")
    model = AutoModelForQuestionAnswering.from_pretrained(
        model_path, dtype=dtype, attn_implementation=attention_implementation)
    model.eval().to("cuda")
    return model, tokenizer


@torch.inference_mode()
def score_pairs(model, tokenizer, questions: list[str], passages: list[str],
                batch_size: int, max_answer_length: int = 64) -> list[float]:
    out: list[float] = []
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    for start in range(0, len(questions), batch_size):
        q_batch = questions[start:start + batch_size]
        p_batch = passages[start:start + batch_size]
        encoded = tokenizer(q_batch, p_batch, padding="longest", truncation="only_second",
                            max_length=512, return_tensors="pt")
        context_masks = torch.tensor(
            [[token == 1 for token in encoded.sequence_ids(i)]
             for i in range(len(q_batch))], dtype=torch.bool, device=device)
        tensors = {key: value.to(device) for key, value in encoded.items()}
        with torch.autocast("cuda", dtype=torch.float16, enabled=dtype == torch.float16):
            outputs = model(**tensors)
        start_logits = outputs.start_logits.float()
        end_logits = outputs.end_logits.float()
        input_ids = tensors["input_ids"]
        cls_positions = (input_ids == int(tokenizer.cls_token_id)).float().argmax(dim=1)
        batch_indices = torch.arange(input_ids.shape[0], device=device)
        null_scores = start_logits[batch_indices, cls_positions] + end_logits[batch_indices, cls_positions]
        length = input_ids.shape[1]
        positions = torch.arange(length, device=device)
        valid_shape = ((positions[None, :] >= positions[:, None]) &
                       (positions[None, :] - positions[:, None] < max_answer_length))
        pair_scores = start_logits[:, :, None] + end_logits[:, None, :]
        pair_mask = context_masks[:, :, None] & context_masks[:, None, :] & valid_shape[None, :, :]
        answer_scores = pair_scores.masked_fill(~pair_mask, -torch.inf).flatten(1).max(dim=1).values
        margins = answer_scores - null_scores
        if not torch.isfinite(margins).all():
            raise RuntimeError("non-finite answerability margin")
        out.extend(float(x) for x in margins.cpu())
    return out


def parent_ranking(owners, values):
    parent: dict[str, float] = {}
    for (doc, _), value in zip(owners, values):
        parent[doc] = max(parent.get(doc, -math.inf), value)
    return sorted(parent, key=lambda doc: (-parent[doc], doc)), parent


def parity(args: argparse.Namespace) -> None:
    rows = deterministic_rows(fold0_rows(args), args.smoke_queries)
    rendered = list(build_inputs(args, rows))
    questions = [q for _, _, qs, _ in rendered for q in qs]
    passages = [p for _, _, _, ps in rendered for p in ps]
    torch.cuda.reset_peak_memory_stats(); started = time.perf_counter()
    fp32_model, tokenizer = load_model(args.model, torch.float32)
    fp32 = score_pairs(fp32_model, tokenizer, questions, passages, 1)
    fp32_seconds = time.perf_counter() - started
    del fp32_model; gc.collect(); torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(); started = time.perf_counter()
    fp16_model, tokenizer = load_model(args.model, torch.float16)
    fp16 = score_pairs(fp16_model, tokenizer, questions, passages, args.batch_size)
    fp16_seconds = time.perf_counter() - started
    peak = torch.cuda.max_memory_allocated() / 2**20
    errors = np.abs(np.asarray(fp32) - np.asarray(fp16))
    cursor = 0; top5_equal = True
    for _, owners, qs, _ in rendered:
        n = len(qs)
        rank32, _ = parent_ranking(owners, fp32[cursor:cursor+n])
        rank16, _ = parent_ranking(owners, fp16[cursor:cursor+n])
        top5_equal &= rank32[:5] == rank16[:5]
        cursor += n
    total_sequences = sum(len(row["doc_ids"]) * 2 for row in fold0_rows(args))
    # Loading time makes this deliberately conservative for the tiny smoke.
    fp16_rate = len(questions) / fp16_seconds
    fp32_rate = len(questions) / fp32_seconds
    fp16_safe = top5_equal and float(errors.max()) <= 0.05
    authorized = "float16" if fp16_safe else "float32"
    estimate = total_sequences / (fp16_rate if fp16_safe else fp32_rate)
    status = "PASS_FP16" if fp16_safe else "PASS_FP32_REQUIRED"
    if estimate > 14400:
        status = "BLOCKED_LOCAL_RUNTIME"
    report = {
        "schema_version": "dsc2026.research_v2.vimrc_answerability_parity.v1",
        "status": status, "authorized_dtype": authorized,
        "queries": len(rows), "sequences": len(questions),
        "fp32_seconds_including_load": fp32_seconds,
        "fp16_seconds_including_load": fp16_seconds,
        "estimated_full_fold0_seconds_conservative": estimate,
        "fp16_peak_allocated_mib": peak,
        "score_max_abs_error": float(errors.max()),
        "score_mean_abs_error": float(errors.mean()),
        "parent_top5_identical": top5_equal,
        "model_files": {p.name: sha256(p) for p in sorted(args.model.iterdir()) if p.is_file()},
        "inputs": {"folds": sha256(args.folds), "pool": sha256(args.pool)},
        "score_contract": "best_context_span_len64_minus_cls_no_answer"
    }
    path = args.output_dir / "VIMRC_ANSWERABILITY_PARITY.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


def fp32_batch_parity(args: argparse.Namespace) -> None:
    """Authorize a throughput-only FP32 batch increase without changing scores."""
    rows = deterministic_rows(fold0_rows(args), args.smoke_queries)
    rendered = list(build_inputs(args, rows))
    questions = [q for _, _, qs, _ in rendered for q in qs]
    passages = [p for _, _, _, ps in rendered for p in ps]
    model, tokenizer = load_model(args.model, torch.float32)
    reference = score_pairs(model, tokenizer, questions, passages, 1)
    torch.cuda.reset_peak_memory_stats(); started = time.perf_counter()
    candidate = score_pairs(model, tokenizer, questions, passages, args.fp32_batch)
    elapsed = time.perf_counter() - started
    peak = torch.cuda.max_memory_allocated() / 2**20
    errors = np.abs(np.asarray(reference) - np.asarray(candidate))
    cursor = 0; top5_equal = True
    for _, owners, qs, _ in rendered:
        n = len(qs)
        rank1, _ = parent_ranking(owners, reference[cursor:cursor+n])
        rankn, _ = parent_ranking(owners, candidate[cursor:cursor+n])
        top5_equal &= rank1[:5] == rankn[:5]
        cursor += n
    total_sequences = sum(len(row["doc_ids"]) * 2 for row in fold0_rows(args))
    estimate = total_sequences / (len(questions) / elapsed)
    status = "PASS" if top5_equal and float(errors.max()) <= 1e-4 and estimate <= 14400 else "FAIL"
    report = {"schema_version":"dsc2026.research_v2.vimrc_fp32_batch_parity.v1",
              "status":status,"reference_batch":1,"candidate_batch":args.fp32_batch,
              "queries":len(rows),"sequences":len(questions),"seconds_excluding_load":elapsed,
              "estimated_full_fold0_seconds":estimate,"peak_allocated_mib":peak,
              "score_max_abs_error":float(errors.max()),"score_mean_abs_error":float(errors.mean()),
              "parent_top5_identical":top5_equal}
    path=args.output_dir/"VIMRC_FP32_BATCH_PARITY.json";path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)


def sdpa_parity(args: argparse.Namespace) -> None:
    """Compare FP32 SDPA against the authorized eager FP32 scorer."""
    rows = deterministic_rows(fold0_rows(args), args.smoke_queries)
    rendered = list(build_inputs(args, rows))
    questions = [q for _, _, qs, _ in rendered for q in qs]
    passages = [p for _, _, _, ps in rendered for p in ps]
    eager, tokenizer = load_model(args.model, torch.float32, "eager")
    reference = score_pairs(eager, tokenizer, questions, passages, args.fp32_batch)
    del eager; gc.collect(); torch.cuda.empty_cache()
    sdpa, tokenizer = load_model(args.model, torch.float32, "sdpa")
    torch.cuda.reset_peak_memory_stats(); started=time.perf_counter()
    candidate = score_pairs(sdpa, tokenizer, questions, passages, args.fp32_batch)
    elapsed=time.perf_counter()-started; peak=torch.cuda.max_memory_allocated()/2**20
    errors=np.abs(np.asarray(reference)-np.asarray(candidate));cursor=0;top5_equal=True
    for _,owners,qs,_ in rendered:
        n=len(qs); rank_a,_=parent_ranking(owners,reference[cursor:cursor+n]);rank_b,_=parent_ranking(owners,candidate[cursor:cursor+n])
        top5_equal &= rank_a[:5]==rank_b[:5];cursor+=n
    total_sequences=sum(len(row["doc_ids"])*2 for row in fold0_rows(args));estimate=total_sequences/(len(questions)/elapsed)
    status="PASS" if top5_equal and float(errors.max())<=1e-4 and estimate<=14400 else "FAIL"
    report={"schema_version":"dsc2026.research_v2.vimrc_sdpa_parity.v1","status":status,
            "dtype":"float32","batch":args.fp32_batch,"queries":len(rows),"sequences":len(questions),
            "seconds_excluding_load":elapsed,"estimated_full_fold0_seconds":estimate,"peak_allocated_mib":peak,
            "score_max_abs_error":float(errors.max()),"score_mean_abs_error":float(errors.mean()),
            "parent_top5_identical":top5_equal}
    path=args.output_dir/"VIMRC_SDPA_PARITY.json";path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)


def open_db(path: Path) -> sqlite3.Connection:
    db = sqlite3.connect(path)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("CREATE TABLE IF NOT EXISTS scores(qid TEXT,doc_id TEXT,passage_index INTEGER,score REAL,PRIMARY KEY(qid,doc_id,passage_index))")
    db.execute("CREATE TABLE IF NOT EXISTS progress(qid TEXT PRIMARY KEY,seconds REAL,sequences INTEGER,peak_mib REAL)")
    db.commit()
    return db


def run_score(args: argparse.Namespace) -> None:
    rows = fold0_rows(args)
    gate = json.loads((args.output_dir / "VIMRC_ANSWERABILITY_PARITY.json").read_text(encoding="utf-8"))
    if gate["status"] not in {"PASS_FP16", "PASS_FP32_REQUIRED"}:
        raise RuntimeError(f"parity/runtime gate closed: {gate['status']}")
    dtype = torch.float16 if gate["authorized_dtype"] == "float16" else torch.float32
    if dtype == torch.float32:
        batch_gate = json.loads((args.output_dir/"VIMRC_FP32_BATCH_PARITY.json").read_text(encoding="utf-8"))
        batch_size = args.fp32_batch if (batch_gate["status"] == "PASS" and
            batch_gate["candidate_batch"] == args.fp32_batch) else 1
    else:
        batch_size = args.batch_size
    db = open_db(args.score_db)
    complete = {str(row[0]) for row in db.execute("SELECT qid FROM progress")}
    sdpa_path=args.output_dir/"VIMRC_SDPA_PARITY.json"
    attention="sdpa" if sdpa_path.exists() and json.loads(sdpa_path.read_text(encoding="utf-8"))["status"]=="PASS" else "eager"
    model, tokenizer = load_model(args.model, dtype, attention)
    started = time.perf_counter(); newly_done = 0
    for row, owners, questions, passages in build_inputs(args, rows):
        qid = str(row["qid"])
        if qid in complete:
            continue
        torch.cuda.reset_peak_memory_stats(); qstarted = time.perf_counter()
        values = score_pairs(model, tokenizer, questions, passages, batch_size)
        elapsed = time.perf_counter() - qstarted
        peak = torch.cuda.max_memory_allocated() / 2**20
        with db:
            db.executemany("INSERT INTO scores VALUES(?,?,?,?)",
                           [(qid, doc, idx, value) for (doc, idx), value in zip(owners, values)])
            db.execute("INSERT INTO progress VALUES(?,?,?,?)", (qid, elapsed, len(values), peak))
        newly_done += 1
        if newly_done % 10 == 0:
            total_done = len(complete) + newly_done
            qsec = (time.perf_counter() - started) / newly_done
            print(f"done={total_done}/1398 qsec={qsec:.2f} eta_min={(1398-total_done)*qsec/60:.1f} peak_mib={peak:.1f}", flush=True)
    counts = db.execute("SELECT COUNT(DISTINCT qid),COUNT(*),SUM(sequences),SUM(seconds),MAX(peak_mib) FROM progress").fetchone()
    score_rows = db.execute("SELECT COUNT(*) FROM scores").fetchone()[0]
    integrity = db.execute("PRAGMA integrity_check").fetchone()[0]
    db.close()
    print(json.dumps({"progress": counts, "score_rows": score_rows, "integrity": integrity}, indent=2), flush=True)


def recall_at(docs, gold, k=5):
    return len(set(docs[:k]) & gold) / len(gold)


def load_clean_expert_sets(args, pool):
    e5 = {str(x["qid"]): set(map(str, x["ft_order"][:5])) |
          set(map(str, x["base_order"][:5])) for x in read_jsonl(args.e5_predictions)
          if str(x["qid"]) in pool}
    jina_v2 = {str(x["qid"]): set(map(str, x["top5"])) for x in read_jsonl(args.jina_predictions) if str(x["qid"]) in pool}
    native: dict[tuple[str, str], list[str]] = {}
    db = sqlite3.connect(f"file:{args.sources_db.resolve().as_posix()}?mode=ro", uri=True)
    for qid, source, payload in db.execute("SELECT q,source,payload FROM sources WHERE source IN ('lal','jina')"):
        qid = str(qid)
        if qid in pool:
            native[(qid, str(source))] = [str(x["doc_id"]) for x in json.loads(payload)
                                          if str(x["doc_id"]) in pool[qid]][:5]
    db.close()
    return {qid: e5[qid] | jina_v2[qid] | set(native[(qid, "lal")]) |
            set(native[(qid, "jina")]) for qid in pool}


def evaluate(args: argparse.Namespace) -> None:
    rows = fold0_rows(args)
    pool = {str(row["qid"]): set(map(str, row["doc_ids"])) for row in rows}
    answers = canonical_answers(json.loads(args.train.read_text(encoding="utf-8")),
                                json.loads(args.exclusions.read_text(encoding="utf-8")))
    anchor = {str(x["qid"]): list(map(str, x["fused_top5"])) for x in read_jsonl(args.anchor) if str(x["qid"]) in pool}
    clean = load_clean_expert_sets(args, pool)
    db = sqlite3.connect(f"file:{args.score_db.resolve().as_posix()}?mode=ro", uri=True)
    if db.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
        raise RuntimeError("score DB integrity failure")
    complete = {str(x[0]) for x in db.execute("SELECT qid FROM progress")}
    if complete != set(pool):
        raise RuntimeError(f"incomplete scoring: {len(complete)}/{len(pool)}")
    scores = {(str(q), str(d)): float(s) for q, d, s in db.execute("SELECT qid,doc_id,MAX(score) FROM scores GROUP BY qid,doc_id")}
    runtime = db.execute("SELECT SUM(seconds),SUM(sequences),MAX(peak_mib) FROM progress").fetchone()
    db.close()
    qa_r=[]; qa_p=[]; anchor_r=[]; anchor_union=[]; clean_r=[]; clean_plus=[]
    single=[[], []]; multi=[[], []]; depth={k:[] for k in (5,10,20,50)}
    wins=losses=changed=0; qa_only=anchor_only=0; buckets=Counter(); predictions=[]
    for row in rows:
        qid=str(row["qid"]); docs=list(map(str,row["doc_ids"])); gold=answers[qid]
        ordered=sorted(docs,key=lambda d:(-scores[(qid,d)],d)); pred=ordered[:5]; base=anchor[qid]
        qr=recall_at(pred,gold); ar=recall_at(base,gold)
        qa_r.append(qr); qa_p.append(len(set(pred)&gold)/5); anchor_r.append(ar)
        anchor_union.append(recall_at(list(set(pred)|set(base)),gold,k=10_000))
        clean_r.append(recall_at(list(clean[qid]),gold,k=10_000))
        clean_plus.append(recall_at(list(clean[qid]|set(pred)),gold,k=10_000))
        wins += qr>ar; losses += qr<ar; changed += pred!=base
        qa_only += bool(set(pred)&gold) and not bool(set(base)&gold)
        anchor_only += bool(set(base)&gold) and not bool(set(pred)&gold)
        target=single if len(gold)==1 else multi; target[0].append(qr); target[1].append(ar)
        for k in depth: depth[k].append(recall_at(ordered,gold,k))
        for doc in gold:
            rank=ordered.index(doc)+1 if doc in ordered else None
            bucket="missing" if rank is None else "1-5" if rank<=5 else "6-10" if rank<=10 else "11-20" if rank<=20 else "21-50" if rank<=50 else "51+"
            buckets[bucket]+=1
        predictions.append({"qid":qid,"top5":pred,"ranking":ordered,"scores":[scores[(qid,d)] for d in ordered]})
    qa=float(np.mean(qa_r)); clean_base=float(np.mean(clean_r)); clean_expanded=float(np.mean(clean_plus)); clean_delta=clean_expanded-clean_base
    if qa>=.90 and clean_delta>=.006: verdict="PASS_STANDALONE"
    elif qa>=.84 and clean_delta>=.005: verdict="PASS_ORTHOGONAL"
    elif qa<.78 or clean_delta<.003: verdict="KILL"
    else: verdict="INCONCLUSIVE_NO_TUNING"
    pred_path=args.output_dir/"VIMRC_ANSWERABILITY_FOLD0_PREDICTIONS.jsonl"
    with pred_path.open("w",encoding="utf-8",newline="\n") as sink:
        for item in sorted(predictions,key=lambda x:int(x["qid"])): sink.write(canonical_json(item)+"\n")
    report={
      "schema_version":"dsc2026.research_v2.vimrc_answerability_fold0_report.v1","status":"COMPLETE","verdict":verdict,
      "metrics":{"queries":len(rows),"recall_at_5":qa,"precision_at_5":float(np.mean(qa_p)),
        "current_anchor_recall_at_5":float(np.mean(anchor_r)),"anchor_plus_qa_union":float(np.mean(anchor_union)),
        "anchor_union_delta":float(np.mean(anchor_union)-np.mean(anchor_r)),
        "existing_clean_experts_union":clean_base,"clean_experts_plus_qa_union":clean_expanded,"clean_union_delta":clean_delta,
        "wins_vs_anchor":wins,"losses_vs_anchor":losses,"ties":len(rows)-wins-losses,"changed_top5_sets":changed,
        "exclusive_query_hits":{"qa_only":qa_only,"anchor_only":anchor_only},
        "single":{"queries":len(single[0]),"qa":float(np.mean(single[0])),"anchor":float(np.mean(single[1]))},
        "multi":{"queries":len(multi[0]),"qa":float(np.mean(multi[0])),"anchor":float(np.mean(multi[1]))},
        "recall_depth":{str(k):float(np.mean(v)) for k,v in depth.items()},"gold_rank_buckets":dict(buckets)},
      "runtime":{"score_seconds":runtime[0],"sequences":runtime[1],"peak_allocated_mib":runtime[2]},
      "integrity":{"folds_sha256":sha256(args.folds),"pool_sha256":sha256(args.pool),"score_db_sha256":sha256(args.score_db),
        "predictions_sha256":sha256(pred_path),"preregistration_sha256":sha256(args.output_dir/"VIMRC_ANSWERABILITY_FOLD0_PREREGISTRATION.json")},
      "anti_rescue":"No model prompt span evidence aggregation fusion or threshold tuning."
    }
    path=args.output_dir/"VIMRC_ANSWERABILITY_FOLD0_REPORT.json";path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)


def parser():
    root=Path(__file__).resolve().parents[2]; workspace=root.parent
    p=argparse.ArgumentParser(); p.add_argument("stage",choices=("parity","batch-parity","sdpa-parity","score","evaluate"))
    p.add_argument("--model",type=Path,default=root/"cache/research_v2_open_rl/models/vi-mrc-large")
    p.add_argument("--output-dir",type=Path,default=root/"results/research_v2_open_rl")
    p.add_argument("--score-db",type=Path,default=root/"cache/research_v2_open_rl/vimrc_answerability_fold0_scores.sqlite")
    p.add_argument("--pool",type=Path,default=root/"results/research_v2_forensic/V2_CANDIDATE_POOL.jsonl")
    p.add_argument("--folds",type=Path,default=root/"results/research_v2_forensic/V2_FOLDS.json")
    p.add_argument("--train",type=Path,default=workspace/"LegalIR/public_test_dataset/train.json")
    p.add_argument("--exclusions",type=Path,default=workspace/"LegalIR/cache/final_preprocessed_v2/exclusions.json")
    p.add_argument("--contexts",type=Path,default=workspace/"LegalIR/cache/final_preprocessed_v2/contexts")
    p.add_argument("--anchor",type=Path,default=root/"results/research_v2_post_e5/V2_ADAPTED_E5_LAL_EQUAL_RRF32_PREDICTIONS.jsonl")
    p.add_argument("--e5-predictions",type=Path,default=root/"results/research_v2_e5_confirmation/fold0_runner_parity/E5_CONFIRMATION_FOLD_0_PREDICTIONS.jsonl")
    p.add_argument("--jina-predictions",type=Path,default=root/"results/research_v2_forensic/V2_ZERO_SHOT_LEXICAL_PREDICTIONS.jsonl")
    p.add_argument("--sources-db",type=Path,default=workspace/"LegalIR/cache/exp112_task_adaptive_retrieval/sources.sqlite")
    p.add_argument("--huy-root",type=Path,default=root)
    p.add_argument("--batch-size",type=int,default=8); p.add_argument("--fp32-batch",type=int,default=4); p.add_argument("--smoke-queries",type=int,default=2)
    return p


if __name__=="__main__":
    args=parser().parse_args();args.output_dir.mkdir(parents=True,exist_ok=True);args.score_db.parent.mkdir(parents=True,exist_ok=True)
    if args.stage=="parity": parity(args)
    elif args.stage=="batch-parity": fp32_batch_parity(args)
    elif args.stage=="sdpa-parity": sdpa_parity(args)
    elif args.stage=="score": run_score(args)
    else: evaluate(args)
