"""Strict Fold-0 adapted-E5 token-MaxSim pilot for Research V2."""

from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import json
import math
import os
import sqlite3
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import legal_mlm_query_likelihood_fold0 as common


EXPECTED = {
    "folds": common.EXPECTED["folds"], "pool": common.EXPECTED["pool"],
    "contexts": common.EXPECTED["contexts"], "anchor": common.EXPECTED["anchor"],
    "e5_predictions": common.EXPECTED["e5_predictions"],
    "jina_predictions": common.EXPECTED["jina_predictions"],
    "sources_db": common.EXPECTED["sources_db"], "selector": common.EXPECTED["selector"],
    "base_model": "afa0f907c7e1d8290854b8c295cd7d77521591b4c2f2a27c261258de92333ced",
    "adapter": "ef4c293ba78917522c81fa119a7406c3c936330c0985b53e5a0188cf91e36cc6",
    "core": "b674c9756b26d79966734d8acb928013880c80a150f5326462056055b3d3fd9b",
    "preregistration": "ffe5354e7931beffc3cf617001d9812bb1eb782ca66c25f418237a28d51cc219",
}
BASE_PARAMETERS = 559_890_432
DEPLOYED_PARAMETERS = 3_397_795_840
PARAMETER_LIMIT = 4_000_000_000
EXPECTED_QUERIES = 1398
EXPECTED_PAIRS = 73128


def require_hash(name: str, path: Path) -> str:
    observed = common.sha256(path)
    if observed != EXPECTED[name]:
        raise RuntimeError(f"{name} hash mismatch: {observed} != {EXPECTED[name]}")
    return observed


def paths(args: argparse.Namespace, include_eval: bool = False) -> dict[str, Path]:
    value = {
        "folds": args.folds, "pool": args.pool, "contexts": args.contexts,
        "selector": args.selector, "base_model": args.model / "model.safetensors",
        "adapter": args.adapter, "core": args.core, "preregistration": args.preregistration,
    }
    if include_eval:
        value.update({"anchor": args.anchor, "e5_predictions": args.e5_predictions,
                      "jina_predictions": args.jina_predictions, "sources_db": args.sources_db})
    return value


def validate(args: argparse.Namespace, include_eval: bool = False) -> dict:
    hashes = {name: require_hash(name, path) for name, path in paths(args, include_eval).items()}
    prereg = json.loads(args.preregistration.read_text(encoding="utf-8"))
    checkpoint = torch.load(args.adapter, map_location="cpu", weights_only=False)
    checks = {
        "sealed": prereg.get("status") == "SEALED_BEFORE_INFERENCE_OR_V2_METRIC",
        "new_training_false": prereg["model"].get("new_training") is False,
        "adapter_contract": checkpoint.get("contract_hash") == "1a0a2db621d773d401527c540b9bc8d3648f7d34c3424f0049dda6b0f5fea42d",
        "adapter_epoch2": checkpoint.get("epoch") == 2 and checkpoint.get("updates") == 700,
        "adapter_parameter_count": sum(t.numel() for t in checkpoint["adapter"].values()) == 1_572_864,
        "no_augmentation": prereg["inference"].get("augmentation") is False,
        "adaptive_k_disabled": prereg["inference"].get("adaptive_k") is False,
        "fixed_geometry": prereg["inference"].get("score") == "mean_query_token_max_document_token_dot",
        "parameter_budget": prereg["parameter_budget"].get("pass") is True and DEPLOYED_PARAMETERS < PARAMETER_LIMIT,
    }
    if not all(checks.values()):
        raise RuntimeError(f"contract failure: {checks}")
    return {"hashes": hashes, "checks": checks}


def fold_rows(args: argparse.Namespace) -> list[dict]:
    return common.fold0_rows(args)


def load_documents(args: argparse.Namespace, rows: list[dict]) -> dict[str, str]:
    return common.load_documents(args, rows)


def evidence_for_row(args: argparse.Namespace, row: dict, documents: dict[str, str]):
    sys.path.insert(0, str(args.root))
    from benchmark_jina_reranker_holdouts import top_passages
    query = str(row["query"]); docs = list(map(str, row["doc_ids"])); passages = []
    for doc in docs:
        selected = list(top_passages(query, documents[doc], count=1))
        if len(selected) != 1:
            raise RuntimeError(f"top1 evidence mismatch: {row['qid']} {doc} {len(selected)}")
        passages.append(str(selected[0]))
    return query, docs, passages


def load_encoder(args: argparse.Namespace, dtype: torch.dtype):
    sys.path.insert(0, str(args.core.parent.parent))
    from research_v2_e5_transfer.e5_transfer_runner import QueryEncoder
    encoder = QueryEncoder(args.model, checkpoint_path=args.adapter)
    encoder.eval()
    if dtype == torch.float16:
        encoder.half()
    count = sum(parameter.numel() for name, parameter in encoder.named_parameters() if "lora_" not in name)
    if count != BASE_PARAMETERS:
        raise RuntimeError(f"base parameter mismatch: {count}")
    return encoder


def token_batch(encoder, texts: list[str], prefix: str, max_length: int, adapter_enabled: bool):
    batch = encoder.tokenizer(
        [prefix + text for text in texts], padding=True, truncation=True, max_length=max_length,
        return_special_tokens_mask=True, return_tensors="pt",
    )
    special = batch.pop("special_tokens_mask").bool()
    device = next(encoder.parameters()).device
    batch = {key: value.to(device) for key, value in batch.items()}
    special = special.to(device)
    context = contextlib.nullcontext() if adapter_enabled else encoder.adapter_disabled()
    with context, torch.inference_mode():
        hidden = encoder.model(**batch, return_dict=True).last_hidden_state.float()
    hidden = F.normalize(hidden, dim=-1)
    valid = batch["attention_mask"].bool() & ~special
    if not valid.any(dim=1).all():
        raise RuntimeError("empty non-special token sequence")
    return hidden, valid


@torch.inference_mode()
def score_documents(encoder, query: str, passages: list[str], batch_size: int) -> list[float]:
    query_hidden, query_valid = token_batch(encoder, [query], "query: ", 64, True)
    query_tokens = query_hidden[0, query_valid[0]]
    output: list[float] = []
    for start in range(0, len(passages), batch_size):
        docs, valid = token_batch(encoder, passages[start:start + batch_size], "passage: ", 256, False)
        similarity = torch.einsum("qd,bkd->bqk", query_tokens, docs)
        similarity = similarity.masked_fill(~valid[:, None, :], -torch.inf)
        values = similarity.max(dim=-1).values.mean(dim=-1)
        if not torch.isfinite(values).all():
            raise RuntimeError("non-finite MaxSim score")
        output.extend(float(value) for value in values.cpu())
    return output


def deterministic_rows(rows: list[dict], count: int) -> list[dict]:
    return sorted(rows, key=lambda row: hashlib.sha256(str(row["qid"]).encode()).hexdigest())[:count]


def preflight(args: argparse.Namespace) -> None:
    contract = validate(args); rows = fold_rows(args); documents = load_documents(args, rows)
    report = {
        "schema_version": "dsc2026.research_v2.adapted_e5_token_maxsim_input_manifest.v1",
        "status": "PASS_INPUT_ONLY_NO_LABEL_METRIC_READ", "queries": len(rows),
        "candidate_pairs": sum(len(row["doc_ids"]) for row in rows),
        "unique_candidate_parents": len(documents), "base_parameters": BASE_PARAMETERS,
        "deployed_original_parameters": DEPLOYED_PARAMETERS, "parameter_limit": PARAMETER_LIMIT,
        "contract": contract, "labels_read": False, "metric_read": False, "augmentation": False,
    }
    if args.input_manifest.exists():
        if common.canonical_json(json.loads(args.input_manifest.read_text(encoding="utf-8"))) != common.canonical_json(report):
            raise RuntimeError("input manifest differs")
    else:
        common.write_json(args.input_manifest, report)
    print(json.dumps(report, indent=2), flush=True)


def parity(args: argparse.Namespace) -> None:
    validate(args)
    if not args.input_manifest.exists():
        raise RuntimeError("preflight missing")
    rows = deterministic_rows(fold_rows(args), 2); documents = load_documents(args, rows)
    rendered = [evidence_for_row(args, row, documents) for row in rows]
    total = sum(len(docs) for _, docs, _ in rendered)
    common.set_determinism(); torch.cuda.reset_peak_memory_stats(); started = time.perf_counter()
    fp32_model = load_encoder(args, torch.float32); fp32 = []
    for query, _, passages in rendered: fp32.extend(score_documents(fp32_model, query, passages, 8))
    fp32_seconds = time.perf_counter() - started
    del fp32_model; gc.collect(); torch.cuda.empty_cache()
    common.set_determinism(); torch.cuda.reset_peak_memory_stats(); started = time.perf_counter()
    fp16_model = load_encoder(args, torch.float16); fp16 = []
    for query, _, passages in rendered: fp16.extend(score_documents(fp16_model, query, passages, 32))
    fp16_seconds = time.perf_counter() - started; peak = torch.cuda.max_memory_allocated() / 2**20
    errors = np.abs(np.asarray(fp32) - np.asarray(fp16)); cursor = 0; top5_equal = True
    for _, docs, _ in rendered:
        count = len(docs)
        rank32 = sorted(range(count), key=lambda i: (-fp32[cursor+i], docs[i]))[:5]
        rank16 = sorted(range(count), key=lambda i: (-fp16[cursor+i], docs[i]))[:5]
        top5_equal &= [docs[i] for i in rank32] == [docs[i] for i in rank16]; cursor += count
    estimate16 = EXPECTED_PAIRS * fp16_seconds / total; estimate32 = EXPECTED_PAIRS * fp32_seconds / total
    pass16 = top5_equal and float(errors.max()) <= 0.01 and estimate16 <= 7200
    pass32 = estimate32 <= 7200
    report = {
        "schema_version": "dsc2026.research_v2.adapted_e5_token_maxsim_parity.v1",
        "status": "PASS_FP16" if pass16 else "PASS_FP32" if pass32 else "BLOCKED_LOCAL_RUNTIME",
        "authorized_dtype": "float16" if pass16 else "float32" if pass32 else None,
        "authorized_batch": 32 if pass16 else 8 if pass32 else None,
        "queries": len(rows), "sequences": total, "score_max_abs_error": float(errors.max()),
        "score_mean_abs_error": float(errors.mean()), "parent_top5_identical": top5_equal,
        "fp32_seconds_including_load": fp32_seconds, "fp16_seconds_including_load": fp16_seconds,
        "fp32_estimated_full_seconds": estimate32, "fp16_estimated_full_seconds": estimate16,
        "fp16_peak_allocated_mib": peak, "labels_read": False, "metric_read": False,
    }
    if args.parity_report.exists():
        raise RuntimeError("refusing to overwrite parity")
    common.write_json(args.parity_report, report); print(json.dumps(report, indent=2), flush=True)


def fingerprint() -> str:
    return hashlib.sha256(common.canonical_json({
        "preregistration": EXPECTED["preregistration"], "base_model": EXPECTED["base_model"],
        "adapter": EXPECTED["adapter"], "pool": EXPECTED["pool"], "contexts": EXPECTED["contexts"],
        "query": ["query: ", 64, "adapter_on"], "passage": ["passage: ", 256, "adapter_off"],
        "tokens": "attention_non_special_l2", "score": "mean_query_max_doc_dot", "evidence": "lexical_top1",
    }).encode()).hexdigest()


def open_db(path: Path) -> sqlite3.Connection:
    db = sqlite3.connect(path); db.execute("PRAGMA journal_mode=WAL")
    db.execute("CREATE TABLE IF NOT EXISTS scores(qid TEXT,doc_id TEXT,score REAL,PRIMARY KEY(qid,doc_id))")
    db.execute("CREATE TABLE IF NOT EXISTS progress(qid TEXT PRIMARY KEY,seconds REAL,sequences INTEGER,peak_mib REAL)")
    db.execute("CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT)"); db.commit(); return db


def score(args: argparse.Namespace) -> None:
    validate(args); parity_report = json.loads(args.parity_report.read_text(encoding="utf-8"))
    if parity_report.get("status") not in {"PASS_FP16", "PASS_FP32"}:
        raise RuntimeError("parity gate closed")
    dtype = torch.float16 if parity_report["authorized_dtype"] == "float16" else torch.float32
    batch_size = int(parity_report["authorized_batch"])
    rows = fold_rows(args); documents = load_documents(args, rows); args.score_db.parent.mkdir(parents=True, exist_ok=True)
    db = open_db(args.score_db); expected_meta = {"fingerprint": fingerprint(), "folds_sha256": EXPECTED["folds"], "pool_sha256": EXPECTED["pool"]}
    stored = dict(db.execute("SELECT key,value FROM metadata"))
    if stored and stored != expected_meta: raise RuntimeError(f"metadata mismatch: {stored}")
    if not stored:
        with db: db.executemany("INSERT INTO metadata VALUES(?,?)", expected_meta.items())
    complete = {str(row[0]) for row in db.execute("SELECT qid FROM progress")}
    common.set_determinism(); encoder = load_encoder(args, dtype); process_started = time.perf_counter(); newly_done = 0
    for row in rows:
        qid = str(row["qid"])
        if qid in complete: continue
        query, docs, passages = evidence_for_row(args, row, documents)
        torch.cuda.reset_peak_memory_stats(); started = time.perf_counter()
        values = score_documents(encoder, query, passages, batch_size); elapsed = time.perf_counter() - started
        if len(values) != len(docs): raise RuntimeError("cardinality mismatch")
        with db:
            db.executemany("INSERT INTO scores VALUES(?,?,?)", [(qid, doc, value) for doc, value in zip(docs, values)])
            db.execute("INSERT INTO progress VALUES(?,?,?,?)", (qid, elapsed, len(values), torch.cuda.max_memory_allocated()/2**20))
        newly_done += 1
        if newly_done % 10 == 0:
            done = len(complete) + newly_done; qps = newly_done / (time.perf_counter() - process_started)
            print(f"score={done}/{EXPECTED_QUERIES} qps={qps:.3f} eta_min={(EXPECTED_QUERIES-done)/qps/60:.1f}", flush=True)
    stats = db.execute("SELECT COUNT(*),COUNT(DISTINCT qid),SUM(sequences),SUM(seconds),MAX(peak_mib) FROM progress").fetchone()
    count = db.execute("SELECT COUNT(*) FROM scores").fetchone()[0]; integrity = db.execute("PRAGMA integrity_check").fetchone()[0]
    db.execute("PRAGMA wal_checkpoint(TRUNCATE)"); db.close()
    print(json.dumps({"progress": stats, "score_rows": count, "integrity": integrity}, indent=2), flush=True)


def evaluate(args: argparse.Namespace) -> None:
    if args.report.exists(): raise RuntimeError("refusing second evaluation")
    contract = validate(args, include_eval=True); rows = fold_rows(args)
    pool = {str(row["qid"]): set(map(str, row["doc_ids"])) for row in rows}
    anchor = {str(row["qid"]): row for row in common.read_jsonl(args.anchor) if str(row["qid"]) in pool}
    clean = common.load_clean_sets(args, pool)
    db = sqlite3.connect(f"file:{args.score_db.resolve().as_posix()}?mode=ro&immutable=1", uri=True)
    integrity = db.execute("PRAGMA integrity_check").fetchone()[0]
    progress = db.execute("SELECT COUNT(*),COUNT(DISTINCT qid),SUM(sequences),SUM(seconds),MAX(peak_mib) FROM progress").fetchone()
    scores = {(str(q),str(d)):float(s) for q,d,s in db.execute("SELECT qid,doc_id,score FROM scores")}
    metadata = dict(db.execute("SELECT key,value FROM metadata")); db.close()
    expected = {(qid,doc) for qid,docs in pool.items() for doc in docs}
    integrity_ok = integrity == "ok" and progress[0] == progress[1] == EXPECTED_QUERIES and progress[2] == len(scores) == EXPECTED_PAIRS and set(scores) == expected and metadata.get("fingerprint") == fingerprint()
    if not integrity_ok: raise RuntimeError("score cache incomplete")
    expert_r=[];expert_p=[];anchor_r=[];clean_r=[];clean_plus=[];candidate=[];single=[[],[]];multi=[[],[]];depth={k:[] for k in (5,10,20,50)}
    wins=losses=churn=cross_in=cross_out=0; predictions=[]; buckets=Counter()
    for row in rows:
        qid=str(row["qid"]);docs=list(map(str,row["doc_ids"]));gold=set(map(str,anchor[qid]["gold"]));ordered=sorted(docs,key=lambda d:(-scores[(qid,d)],d));top5=ordered[:5];base=list(map(str,anchor[qid]["fused_top5"]));er=common.recall(top5,gold);ar=common.recall(base,gold)
        expert_r.append(er);expert_p.append(len(set(top5)&gold)/5);anchor_r.append(ar);clean_r.append(common.recall(clean[qid],gold));clean_plus.append(common.recall(clean[qid]|set(top5),gold));candidate.append(common.recall(pool[qid],gold));target=single if len(gold)==1 else multi;target[0].append(er);target[1].append(ar)
        wins+=er>ar;losses+=er<ar;churn+=top5!=base;cross_in+=len((set(top5)-set(base))&gold);cross_out+=len((set(base)-set(top5))&gold)
        for k in depth:depth[k].append(common.recall(ordered[:k],gold))
        for doc in gold:
            rank=ordered.index(doc)+1 if doc in pool[qid] else None;bucket="missing" if rank is None else "1-5" if rank<=5 else "6-10" if rank<=10 else "11-20" if rank<=20 else "21-50" if rank<=50 else "51+";buckets[bucket]+=1
        predictions.append({"qid":qid,"top5":top5,"ranking":ordered,"scores":[scores[(qid,d)] for d in ordered]})
    metrics={"recall_at_5":float(np.mean(expert_r)),"precision_at_5":float(np.mean(expert_p)),"current_anchor_recall_at_5":float(np.mean(anchor_r)),"existing_clean_experts_union":float(np.mean(clean_r)),"clean_experts_plus_maxsim_union":float(np.mean(clean_plus)),"clean_union_delta":float(np.mean(clean_plus)-np.mean(clean_r)),"candidate_ceiling":float(np.mean(candidate)),"single_gold":{"queries":len(single[0]),"expert_recall_at_5":float(np.mean(single[0])),"anchor_recall_at_5":float(np.mean(single[1]))},"multi_gold":{"queries":len(multi[0]),"expert_recall_at_5":float(np.mean(multi[0])),"anchor_recall_at_5":float(np.mean(multi[1]))},"per_fold":{"fold_0":{"queries":len(rows),"expert_recall_at_5":float(np.mean(expert_r)),"anchor_recall_at_5":float(np.mean(anchor_r)),"delta":float(np.mean(expert_r)-np.mean(anchor_r))}},"wins_losses_ties_vs_anchor":{"wins":wins,"losses":losses,"ties":len(rows)-wins-losses},"gold_crossings":{"into_top5":cross_in,"out_of_top5":cross_out},"top5_churn_queries":churn,"recall_depth":{str(k):float(np.mean(v)) for k,v in depth.items()},"gold_rank_buckets":dict(buckets)}
    pass_s=metrics["recall_at_5"]>=0.88 and metrics["clean_union_delta"]>=0.004;pass_o=metrics["recall_at_5"]>=0.82 and metrics["clean_union_delta"]>=0.005;kill=metrics["recall_at_5"]<0.80 or metrics["clean_union_delta"]<0.0025 or not integrity_ok or not all(contract["checks"].values());verdict="KILL" if kill else "PASS_STANDALONE" if pass_s else "PASS_ORTHOGONAL" if pass_o else "INCONCLUSIVE_NO_TUNING"
    with args.predictions.open("x",encoding="utf-8",newline="\n") as stream:
        for item in predictions:stream.write(common.canonical_json(item)+"\n")
    parity_report=json.loads(args.parity_report.read_text(encoding="utf-8"));report={"schema_version":"dsc2026.research_v2.adapted_e5_token_maxsim_fold0_report.v1","status":"COMPLETE","verdict":verdict,"metrics":metrics,"gate":{"pass_standalone":pass_s,"pass_orthogonal":pass_o,"kill_recall_lt_0_80":metrics["recall_at_5"]<0.80,"kill_clean_delta_lt_0_0025":metrics["clean_union_delta"]<0.0025,"integrity":integrity_ok,"contract":all(contract["checks"].values())},"runtime":{"seconds":progress[3],"sequences":progress[2],"peak_mib":progress[4],"dtype":parity_report["authorized_dtype"],"batch":parity_report["authorized_batch"]},"hashes":{**contract["hashes"],"score_db":common.sha256(args.score_db),"predictions":common.sha256(args.predictions)},"fixed_interface":{"evidence_count":1,"query_max_length":64,"passage_max_length":256,"score":"mean_query_token_max_document_token_dot","query_adapter":"enabled","document_adapter":"disabled","ranking":"score_desc_doc_id_string_asc_top5","adaptive_k":False},"anti_rescue":"No training, model, direction, length, evidence, pooling, normalization, fusion, threshold, swap, route, rule, or adaptive-K grid."}
    common.write_json(args.report,report);top5_rows=[common.canonical_json({"qid":r["qid"],"top5":r["top5"]}) for r in predictions];lock={"schema_version":"dsc2026.research_v2.adapted_e5_token_maxsim_prediction_lock.v1","status":"LOCKED","verdict":verdict,"queries":len(predictions),"predictions_sha256":common.sha256(args.predictions),"canonical_qid_top5_sha256":hashlib.sha256(("\n".join(top5_rows)+"\n").encode()).hexdigest()};common.write_json(args.prediction_lock,lock)
    files=[args.input_manifest,args.parity_report,args.score_db,args.report,args.predictions,args.prediction_lock,args.preregistration];manifest={"schema_version":"dsc2026.research_v2.adapted_e5_token_maxsim_output_manifest.v1","status":"COMPLETE","verdict":verdict,"files":{p.name:{"path":str(p),"sha256":common.sha256(p),"bytes":p.stat().st_size} for p in files},"reproduce":f'"{sys.executable}" "{Path(__file__).resolve()}" verify'};common.write_json(args.output_manifest,manifest);print(json.dumps({"verdict":verdict,"metrics":metrics},ensure_ascii=False,indent=2),flush=True)


def verify(args: argparse.Namespace) -> None:
    manifest=json.loads(args.output_manifest.read_text(encoding="utf-8"));fail=[]
    for name,item in manifest["files"].items():
        path=Path(item["path"]);observed=common.sha256(path) if path.exists() else None
        if observed!=item["sha256"]:fail.append({"file":name,"expected":item["sha256"],"observed":observed})
    db=sqlite3.connect(f"file:{args.score_db.resolve().as_posix()}?mode=ro&immutable=1",uri=True);integrity=db.execute("pragma integrity_check").fetchone()[0];progress=db.execute("select count(*),sum(sequences) from progress").fetchone();count=db.execute("select count(*) from scores").fetchone()[0];db.close();status="PASS" if not fail and integrity=="ok" and progress==(EXPECTED_QUERIES,EXPECTED_PAIRS) and count==EXPECTED_PAIRS else "FAIL";result={"status":status,"manifest_sha256":common.sha256(args.output_manifest),"hash_failures":fail,"database":{"integrity":integrity,"queries":progress[0],"sequences":progress[1],"score_rows":count}};print(json.dumps(result,indent=2));
    if status!="PASS":raise RuntimeError("verification failed")


def parser() -> argparse.ArgumentParser:
    root=Path(__file__).resolve().parents[2];output=root/"results/research_v2_open_rl";cache=root/"cache/research_v2_open_rl/adapted_e5_token_maxsim_fold0";p=argparse.ArgumentParser();p.add_argument("stage",choices=["preflight","parity","score","evaluate","verify","all"]);p.add_argument("--root",type=Path,default=root);p.add_argument("--model",type=Path,default=root/"cache/research_v2_e5_confirmation/bundle-v1/vietlegal-e5");p.add_argument("--adapter",type=Path,default=root/"results/research_v2_e5_transfer/research_v2_e5_transfer_fold0/training/epoch-2.pt");p.add_argument("--core",type=Path,default=root/"src/research_v2_e5_transfer/e5_transfer_runner.py");p.add_argument("--folds",type=Path,default=root/"results/research_v2_forensic/V2_FOLDS.json");p.add_argument("--pool",type=Path,default=root/"results/research_v2_forensic/V2_CANDIDATE_POOL.jsonl");p.add_argument("--contexts",type=Path,default=root/"cache/research_v2_forensic/kaggle_input/research-v2-jina-boundary-v4/V2_CONTEXTS.jsonl");p.add_argument("--selector",type=Path,default=root/"benchmark_jina_reranker_holdouts.py");p.add_argument("--preregistration",type=Path,default=output/"ADAPTED_E5_TOKEN_MAXSIM_FOLD0_PREREGISTRATION.json");p.add_argument("--anchor",type=Path,default=root/"results/research_v2_post_e5/V2_ADAPTED_E5_LAL_EQUAL_RRF32_PREDICTIONS.jsonl");p.add_argument("--e5-predictions",type=Path,default=root/"results/research_v2_e5_confirmation/fold0_runner_parity/E5_CONFIRMATION_FOLD_0_PREDICTIONS.jsonl");p.add_argument("--jina-predictions",type=Path,default=root/"results/research_v2_forensic/V2_ZERO_SHOT_LEXICAL_PREDICTIONS.jsonl");p.add_argument("--sources-db",type=Path,default=root.parent/"LegalIR/cache/exp112_task_adaptive_retrieval/sources.sqlite");p.add_argument("--score-db",type=Path,default=cache/"scores.sqlite");p.add_argument("--input-manifest",type=Path,default=output/"ADAPTED_E5_TOKEN_MAXSIM_FOLD0_INPUT_MANIFEST.json");p.add_argument("--parity-report",type=Path,default=output/"ADAPTED_E5_TOKEN_MAXSIM_FOLD0_PARITY.json");p.add_argument("--report",type=Path,default=output/"ADAPTED_E5_TOKEN_MAXSIM_FOLD0_REPORT.json");p.add_argument("--predictions",type=Path,default=output/"ADAPTED_E5_TOKEN_MAXSIM_FOLD0_PREDICTIONS.jsonl");p.add_argument("--prediction-lock",type=Path,default=output/"ADAPTED_E5_TOKEN_MAXSIM_FOLD0_PREDICTION_LOCK.json");p.add_argument("--output-manifest",type=Path,default=output/"ADAPTED_E5_TOKEN_MAXSIM_FOLD0_OUTPUT_MANIFEST.json");return p


def main() -> None:
    args=parser().parse_args();args.score_db.parent.mkdir(parents=True,exist_ok=True);args.report.parent.mkdir(parents=True,exist_ok=True)
    if args.stage in {"preflight","all"}:preflight(args)
    if args.stage in {"parity","all"}:parity(args)
    if args.stage in {"score","all"}:score(args)
    if args.stage in {"evaluate","all"}:evaluate(args)
    if args.stage=="verify":verify(args)


if __name__=="__main__":main()
