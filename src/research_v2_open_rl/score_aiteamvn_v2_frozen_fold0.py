"""Sealed Fold-0 frozen AITeamVN Vietnamese_Embedding_v2 falsifier."""

from __future__ import annotations

import argparse
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
import torch.nn.functional as F

import legal_mlm_query_likelihood_fold0 as common

EXPECTED_QUERIES = 1398
EXPECTED_PAIRS = 73128
EXPECTED_PARAMETERS = 567754752
MODEL_SHA256 = "2fa082ead5ade68225327b913339bbd5aa1e14bcd7888ff9b09d69752a8d1cee"
MODEL_REVISION = "18b44161e041bf1d3a333ab5144b5b7b93f914d2"
FOLDS_SHA256 = "94ad5c6d5e582ced5eec8d2c3c15f938454c17e713614391091e72abea9aba19"
POOL_SHA256 = "96a44e66549cc211e1f9d0fabb84fc825db3f21f32d5b349eeca3b1c0413e277"
CONTEXTS_SHA256 = "55c77371edd3b4f28e3e8ca548447e27424e57ce219e3d6da7e9f51238a22291"
ANCHOR_SHA256 = "1854494964f2258243bc00896c76b11d56a3c02752a23af0ee04bac8e260de4d"
SOURCES_SHA256 = "ef763bf8c5e3da91fb6447f8a03ab321fdaca5362bfde0213057a15311824f1e"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8", newline="\n")


def set_determinism() -> None:
    torch.manual_seed(20260913)
    torch.cuda.manual_seed_all(20260913)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def rows_and_documents(args):
    rows = common.fold0_rows(args)
    documents = common.load_documents(args, rows)
    return rows, documents


def evidence(args, row, documents):
    return common.evidence_for_row(args, row, documents)


def load_model(args, dtype: torch.dtype):
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, use_fast=True)
    model = AutoModel.from_pretrained(args.model, local_files_only=True, dtype=dtype, attn_implementation="eager")
    count = sum(parameter.numel() for parameter in model.parameters())
    if count != EXPECTED_PARAMETERS:
        raise RuntimeError(f"parameter mismatch: {count}")
    if model.config.hidden_size != 1024 or model.config.num_hidden_layers != 24:
        raise RuntimeError("architecture mismatch")
    model.eval().to("cuda")
    return model, tokenizer


@torch.inference_mode()
def encode(model, tokenizer, texts: list[str], max_length: int, batch_size: int) -> np.ndarray:
    output = []
    device = next(model.parameters()).device
    for start in range(0, len(texts), batch_size):
        batch = tokenizer(texts[start:start + batch_size], max_length=max_length, truncation=True,
                          padding=True, return_tensors="pt")
        batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
        hidden = model(**batch, return_dict=True).last_hidden_state[:, 0]
        vectors = F.normalize(hidden.float(), p=2, dim=1)
        if not torch.isfinite(vectors).all():
            raise RuntimeError("non-finite embedding")
        output.append(vectors.cpu().numpy())
    return np.concatenate(output, axis=0)


def score_row(model, tokenizer, query: str, passages: list[str], batch_size: int):
    qvec = encode(model, tokenizer, [query], 256, 1)[0]
    pvec = encode(model, tokenizer, passages, 512, batch_size)
    return (pvec @ qvec).astype(np.float32).tolist()


def parent_scores(owners, values):
    scores = {}
    for (doc, _), value in zip(owners, values):
        scores[doc] = max(scores.get(doc, -math.inf), float(value))
    return scores


def input_hashes(args):
    return {
        "folds": sha256(args.folds), "pool": sha256(args.pool), "contexts": sha256(args.contexts),
        "anchor": sha256(args.anchor), "sources_db": sha256(args.sources_db),
        "model_safetensors": sha256(args.model / "model.safetensors"),
        "preregistration": sha256(args.preregistration), "renderer_source": sha256(args.renderer),
    }


def validate(args, include_eval=False):
    hashes = input_hashes(args)
    checks = {
        "folds": hashes["folds"] == FOLDS_SHA256,
        "pool": hashes["pool"] == POOL_SHA256,
        "contexts": hashes["contexts"] == CONTEXTS_SHA256,
        "anchor": hashes["anchor"] == ANCHOR_SHA256,
        "sources_db": hashes["sources_db"] == SOURCES_SHA256,
        "model": hashes["model_safetensors"] == MODEL_SHA256,
        "prereg_status": json.loads(args.preregistration.read_text(encoding="utf-8"))["status"] == "SEALED_BEFORE_WEIGHT_DOWNLOAD_OR_INFERENCE_OR_V2_METRIC",
    }
    if include_eval:
        checks["e5_predictions"] = args.e5_predictions.exists()
        checks["jina_predictions"] = args.jina_predictions.exists()
    if not all(checks.values()):
        raise RuntimeError(f"contract failure: {checks}")
    return {"hashes": hashes, "checks": checks}


def fingerprint(args):
    payload = {"revision": MODEL_REVISION, "hashes": input_hashes(args), "query_max": 256,
               "passage_max": 512, "evidence": "lexical_top2_220_words", "pooling": "cls_l2"}
    return hashlib.sha256(canonical_json(payload).encode()).hexdigest()


def preflight(args):
    contract = validate(args)
    rows, documents = rows_and_documents(args)
    sequences = 0
    for row in rows:
        _, owners, passages = evidence(args, row, documents)
        if len(owners) != len(passages):
            raise RuntimeError("owner/passage mismatch")
        sequences += len(passages)
    manifest = {
        "schema_version": "dsc2026.research_v2.aiteamvn_v2_frozen_fold0_input_manifest.v1",
        "status": "PASS_INPUT_ONLY_NO_LABEL_METRIC_READ", "queries": len(rows),
        "candidate_pairs": sum(len(row["doc_ids"]) for row in rows), "evidence_sequences": sequences,
        "unique_candidate_parents": len(documents), "model_parameters": EXPECTED_PARAMETERS,
        "combined_parameter_total": 3965550592, "fingerprint": fingerprint(args), **contract,
    }
    write_json(args.input_manifest, manifest)
    print(json.dumps(manifest, indent=2))


def parity_scores(args, dtype, batch, sample_rows, documents):
    set_determinism(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    model, tokenizer = load_model(args, dtype)
    values = []
    started = time.perf_counter()
    for row in sample_rows:
        query, owners, passages = evidence(args, row, documents)
        raw = score_row(model, tokenizer, query, passages, batch)
        parent = parent_scores(owners, raw)
        values.append((str(row["qid"]), parent))
    runtime = time.perf_counter() - started
    peak = torch.cuda.max_memory_allocated() / 2**20
    del model, tokenizer; torch.cuda.empty_cache()
    return values, runtime, peak


def parity(args):
    validate(args); rows, documents = rows_and_documents(args); sample = rows[:2]
    fp32, fp32_seconds, fp32_peak = parity_scores(args, torch.float32, 1, sample, documents)
    fp16, fp16_seconds, fp16_peak = parity_scores(args, torch.float16, 16, sample, documents)
    errors = []; exact = True; sequences = 0
    for row, (qid32, s32), (qid16, s16) in zip(sample, fp32, fp16):
        if qid32 != qid16: raise RuntimeError("parity qid mismatch")
        docs = list(map(str, row["doc_ids"])); sequences += len(evidence(args, row, documents)[2])
        errors.extend(abs(s32[d] - s16[d]) for d in docs)
        top32 = sorted(docs, key=lambda d: (-s32[d], d))[:5]
        top16 = sorted(docs, key=lambda d: (-s16[d], d))[:5]
        exact &= top32 == top16
    max_error = max(errors); fp16_ok = max_error <= 0.002 and exact
    if fp16_ok:
        authorized_dtype, authorized_batch, sample_seconds, peak = "fp16", 16, fp16_seconds, fp16_peak
    else:
        authorized_dtype, authorized_batch, sample_seconds, peak = "fp32", 1, fp32_seconds, fp32_peak
    estimated = sample_seconds / sequences * json.loads(args.input_manifest.read_text(encoding="utf-8"))["evidence_sequences"]
    status = "PASS" if estimated <= 7200 and peak < 5600 else "FAIL_COST_OR_MEMORY"
    report = {
        "schema_version": "dsc2026.research_v2.aiteamvn_v2_frozen_fold0_parity.v1",
        "status": status, "sample_queries": len(sample), "sample_sequences": sequences,
        "fp16_vs_fp32_max_abs_error": max_error, "exact_parent_top5": exact,
        "fp32_seconds": fp32_seconds, "fp16_seconds": fp16_seconds,
        "fp32_peak_mib": fp32_peak, "fp16_peak_mib": fp16_peak,
        "authorized_dtype": authorized_dtype, "authorized_batch": authorized_batch,
        "estimated_full_seconds": estimated, "cost_limit_seconds": 7200,
    }
    write_json(args.parity_report, report); print(json.dumps(report, indent=2))
    if status != "PASS": raise RuntimeError(status)


def init_db(args):
    args.score_db.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(args.score_db); db.execute("PRAGMA journal_mode=WAL"); db.execute("PRAGMA synchronous=FULL")
    db.execute("CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL)")
    db.execute("CREATE TABLE IF NOT EXISTS scores(qid TEXT NOT NULL,doc_id TEXT NOT NULL,score REAL NOT NULL,PRIMARY KEY(qid,doc_id))")
    db.execute("CREATE TABLE IF NOT EXISTS progress(qid TEXT PRIMARY KEY,seconds REAL NOT NULL,sequences INTEGER NOT NULL,peak_mib REAL NOT NULL)")
    existing = dict(db.execute("SELECT key,value FROM metadata")); fp = fingerprint(args)
    if existing and existing.get("fingerprint") != fp: raise RuntimeError("database fingerprint mismatch")
    if not existing:
        db.execute("INSERT INTO metadata VALUES('fingerprint',?)", (fp,)); db.execute("INSERT INTO metadata VALUES('folds_sha256',?)", (FOLDS_SHA256,)); db.execute("INSERT INTO metadata VALUES('pool_sha256',?)", (POOL_SHA256,)); db.commit()
    return db


def score(args):
    validate(args); parity_report = json.loads(args.parity_report.read_text(encoding="utf-8"))
    if parity_report["status"] != "PASS": raise RuntimeError("parity not authorized")
    dtype = torch.float16 if parity_report["authorized_dtype"] == "fp16" else torch.float32
    batch = int(parity_report["authorized_batch"]); rows, documents = rows_and_documents(args); db = init_db(args)
    complete = {str(row[0]) for row in db.execute("SELECT qid FROM progress")}; set_determinism(); model, tokenizer = load_model(args, dtype)
    process_started = time.perf_counter(); newly_done = 0
    for row in rows:
        qid = str(row["qid"])
        if qid in complete: continue
        started = time.perf_counter(); query, owners, passages = evidence(args, row, documents)
        values = score_row(model, tokenizer, query, passages, batch); parent = parent_scores(owners, values)
        docs = list(map(str, row["doc_ids"]))
        if set(parent) != set(docs) or any(not math.isfinite(parent[d]) for d in docs): raise RuntimeError(f"score contract: {qid}")
        elapsed = time.perf_counter() - started
        with db:
            db.executemany("INSERT INTO scores VALUES(?,?,?)", [(qid, doc, parent[doc]) for doc in docs])
            db.execute("INSERT INTO progress VALUES(?,?,?,?)", (qid, elapsed, len(passages), torch.cuda.max_memory_allocated()/2**20))
        newly_done += 1
        if newly_done % 10 == 0:
            done = len(complete) + newly_done; qps = newly_done / (time.perf_counter() - process_started)
            print(f"score={done}/{EXPECTED_QUERIES} qps={qps:.3f} eta_min={(EXPECTED_QUERIES-done)/qps/60:.1f}", flush=True)
    stats = db.execute("SELECT COUNT(*),COUNT(DISTINCT qid),SUM(sequences),SUM(seconds),MAX(peak_mib) FROM progress").fetchone()
    count = db.execute("SELECT COUNT(*) FROM scores").fetchone()[0]; integrity = db.execute("PRAGMA integrity_check").fetchone()[0]
    db.execute("PRAGMA wal_checkpoint(TRUNCATE)"); db.close(); print(json.dumps({"progress":stats,"score_rows":count,"integrity":integrity}, indent=2))


def evaluate(args):
    if args.report.exists(): raise RuntimeError("refusing second evaluation")
    contract = validate(args, include_eval=True); rows = common.fold0_rows(args)
    pool = {str(row["qid"]): set(map(str, row["doc_ids"])) for row in rows}
    anchor = {str(row["qid"]): row for row in common.read_jsonl(args.anchor) if str(row["qid"]) in pool}
    clean = common.load_clean_sets(args, pool)
    db = sqlite3.connect(f"file:{args.score_db.resolve().as_posix()}?mode=ro&immutable=1", uri=True)
    integrity = db.execute("PRAGMA integrity_check").fetchone()[0]
    progress = db.execute("SELECT COUNT(*),COUNT(DISTINCT qid),SUM(sequences),SUM(seconds),MAX(peak_mib) FROM progress").fetchone()
    scores = {(str(q),str(d)):float(s) for q,d,s in db.execute("SELECT qid,doc_id,score FROM scores")}; metadata = dict(db.execute("SELECT key,value FROM metadata")); db.close()
    expected = {(qid,doc) for qid,docs in pool.items() for doc in docs}; input_manifest = json.loads(args.input_manifest.read_text(encoding="utf-8"))
    integrity_ok = integrity == "ok" and progress[0] == progress[1] == EXPECTED_QUERIES and progress[2] == input_manifest["evidence_sequences"] and len(scores) == EXPECTED_PAIRS and set(scores) == expected and metadata.get("fingerprint") == fingerprint(args)
    if not integrity_ok: raise RuntimeError("score cache incomplete")
    expert_r=[];expert_p=[];anchor_r=[];clean_r=[];clean_plus=[];candidate=[];single=[[],[]];multi=[[],[]];depth={k:[] for k in (5,10,20,50)}
    wins=losses=churn=cross_in=cross_out=0; predictions=[]; buckets=Counter()
    for row in rows:
        qid=str(row["qid"]);docs=list(map(str,row["doc_ids"]));gold=set(map(str,anchor[qid]["gold"]));ordered=sorted(docs,key=lambda d:(-scores[(qid,d)],d));top5=ordered[:5];base=list(map(str,anchor[qid]["fused_top5"]));er=common.recall(top5,gold);ar=common.recall(base,gold)
        expert_r.append(er);expert_p.append(len(set(top5)&gold)/5);anchor_r.append(ar);clean_r.append(common.recall(clean[qid],gold));clean_plus.append(common.recall(clean[qid]|set(top5),gold));candidate.append(common.recall(pool[qid],gold));target=single if len(gold)==1 else multi;target[0].append(er);target[1].append(ar)
        wins+=er>ar;losses+=er<ar;churn+=top5!=base;cross_in+=len((set(top5)-set(base))&gold);cross_out+=len((set(base)-set(top5))&gold)
        for k in depth: depth[k].append(common.recall(ordered[:k],gold))
        for doc in gold:
            rank=ordered.index(doc)+1 if doc in pool[qid] else None; bucket="missing" if rank is None else "1-5" if rank<=5 else "6-10" if rank<=10 else "11-20" if rank<=20 else "21-50" if rank<=50 else "51+"; buckets[bucket]+=1
        predictions.append({"qid":qid,"top5":top5,"ranking":ordered,"scores":[scores[(qid,d)] for d in ordered]})
    metrics={"recall_at_5":float(np.mean(expert_r)),"precision_at_5":float(np.mean(expert_p)),"current_anchor_recall_at_5":float(np.mean(anchor_r)),"existing_clean_experts_union":float(np.mean(clean_r)),"clean_experts_plus_aiteamvn_union":float(np.mean(clean_plus)),"clean_union_delta":float(np.mean(clean_plus)-np.mean(clean_r)),"candidate_ceiling":float(np.mean(candidate)),"single_gold":{"queries":len(single[0]),"expert_recall_at_5":float(np.mean(single[0])),"anchor_recall_at_5":float(np.mean(single[1]))},"multi_gold":{"queries":len(multi[0]),"expert_recall_at_5":float(np.mean(multi[0])),"anchor_recall_at_5":float(np.mean(multi[1]))},"per_fold":{"fold_0":{"queries":len(rows),"expert_recall_at_5":float(np.mean(expert_r)),"anchor_recall_at_5":float(np.mean(anchor_r)),"delta":float(np.mean(expert_r)-np.mean(anchor_r))}},"wins_losses_ties_vs_anchor":{"wins":wins,"losses":losses,"ties":len(rows)-wins-losses},"gold_crossings":{"into_top5":cross_in,"out_of_top5":cross_out},"top5_churn_queries":churn,"recall_depth":{str(k):float(np.mean(v)) for k,v in depth.items()},"gold_rank_buckets":dict(buckets)}
    pass_s=metrics["recall_at_5"]>=0.90 and metrics["clean_union_delta"]>=0.005;pass_o=metrics["recall_at_5"]>=0.86 and metrics["clean_union_delta"]>=0.006;kill=metrics["recall_at_5"]<0.84 or metrics["clean_union_delta"]<0.003 or not integrity_ok or not all(contract["checks"].values());verdict="KILL" if kill else "PASS_STANDALONE" if pass_s else "PASS_ORTHOGONAL" if pass_o else "INCONCLUSIVE_NO_TUNING"
    with args.predictions.open("x",encoding="utf-8",newline="\n") as stream:
        for item in predictions: stream.write(canonical_json(item)+"\n")
    parity_report=json.loads(args.parity_report.read_text(encoding="utf-8")); report={"schema_version":"dsc2026.research_v2.aiteamvn_v2_frozen_fold0_report.v1","status":"COMPLETE","verdict":verdict,"metrics":metrics,"gate":{"pass_standalone":pass_s,"pass_orthogonal":pass_o,"kill_recall_lt_0_84":metrics["recall_at_5"]<0.84,"kill_clean_delta_lt_0_003":metrics["clean_union_delta"]<0.003,"integrity":integrity_ok,"contract":all(contract["checks"].values())},"runtime":{"seconds":progress[3],"sequences":progress[2],"peak_mib":progress[4],"dtype":parity_report["authorized_dtype"],"batch":parity_report["authorized_batch"]},"hashes":{**contract["hashes"],"score_db":sha256(args.score_db),"predictions":sha256(args.predictions)},"fixed_interface":{"query_max_length":256,"passage_max_length":512,"evidence":"lexical_top2_220_words","pooling":"cls_l2","score":"dot_product_parent_max","ranking":"score_desc_parent_id_string_asc_top5","adaptive_k":False},"anti_rescue":"No alternative length, evidence count, pooling, prefix, dtype outside parity, model, score transform, fusion, threshold, route, rule, augmentation or adaptive-K grid."}
    write_json(args.report,report); top5_rows=[canonical_json({"qid":r["qid"],"top5":r["top5"]}) for r in predictions]; lock={"schema_version":"dsc2026.research_v2.aiteamvn_v2_frozen_prediction_lock.v1","status":"LOCKED","verdict":verdict,"queries":len(predictions),"predictions_sha256":sha256(args.predictions),"canonical_qid_top5_sha256":hashlib.sha256(("\n".join(top5_rows)+"\n").encode()).hexdigest()}; write_json(args.prediction_lock,lock)
    files=[args.input_manifest,args.parity_report,args.score_db,args.report,args.predictions,args.prediction_lock,args.preregistration,args.provenance]; manifest={"schema_version":"dsc2026.research_v2.aiteamvn_v2_frozen_output_manifest.v1","status":"COMPLETE","verdict":verdict,"files":{p.name:{"path":str(p),"sha256":sha256(p),"bytes":p.stat().st_size} for p in files},"reproduce":f'"{sys.executable}" "{Path(__file__).resolve()}" verify'}; write_json(args.output_manifest,manifest); print(json.dumps({"verdict":verdict,"metrics":metrics},ensure_ascii=False,indent=2))


def verify(args):
    manifest=json.loads(args.output_manifest.read_text(encoding="utf-8")); fail=[]
    for name,item in manifest["files"].items():
        path=Path(item["path"]); observed=sha256(path) if path.exists() else None
        if observed!=item["sha256"]: fail.append({"file":name,"expected":item["sha256"],"observed":observed})
    db=sqlite3.connect(f"file:{args.score_db.resolve().as_posix()}?mode=ro&immutable=1",uri=True); integrity=db.execute("pragma integrity_check").fetchone()[0]; progress=db.execute("select count(*),sum(sequences) from progress").fetchone(); count=db.execute("select count(*) from scores").fetchone()[0]; db.close(); expected_sequences=json.loads(args.input_manifest.read_text(encoding="utf-8"))["evidence_sequences"]
    status="PASS" if not fail and integrity=="ok" and progress==(EXPECTED_QUERIES,expected_sequences) and count==EXPECTED_PAIRS else "FAIL"; result={"status":status,"manifest_sha256":sha256(args.output_manifest),"hash_failures":fail,"database":{"integrity":integrity,"queries":progress[0],"sequences":progress[1],"score_rows":count}}; print(json.dumps(result,indent=2))
    if status!="PASS": raise RuntimeError("verification failed")


def parser():
    root=Path(__file__).resolve().parents[2]; output=root/"results/research_v2_open_rl"; cache=root/"cache/research_v2_open_rl/aiteamvn_v2_frozen_fold0"; p=argparse.ArgumentParser(); p.add_argument("stage",choices=["preflight","parity","score","evaluate","verify","all"]); p.add_argument("--root",type=Path,default=root); p.add_argument("--model",type=Path,default=root/"cache/research_v2_open_rl/models/aiteamvn-vietnamese-embedding-v2"); p.add_argument("--folds",type=Path,default=root/"results/research_v2_forensic/V2_FOLDS.json"); p.add_argument("--pool",type=Path,default=root/"results/research_v2_forensic/V2_CANDIDATE_POOL.jsonl"); p.add_argument("--contexts",type=Path,default=root/"cache/research_v2_forensic/kaggle_input/research-v2-jina-boundary-v4/V2_CONTEXTS.jsonl"); p.add_argument("--renderer",type=Path,default=root/"benchmark_jina_reranker_holdouts.py"); p.add_argument("--anchor",type=Path,default=root/"results/research_v2_post_e5/V2_ADAPTED_E5_LAL_EQUAL_RRF32_PREDICTIONS.jsonl"); p.add_argument("--e5-predictions",type=Path,default=root/"results/research_v2_e5_confirmation/fold0_runner_parity/E5_CONFIRMATION_FOLD_0_PREDICTIONS.jsonl"); p.add_argument("--jina-predictions",type=Path,default=root/"results/research_v2_forensic/V2_ZERO_SHOT_LEXICAL_PREDICTIONS.jsonl"); p.add_argument("--sources-db",type=Path,default=root.parent/"LegalIR/cache/exp112_task_adaptive_retrieval/sources.sqlite"); p.add_argument("--preregistration",type=Path,default=output/"AITEAMVN_V2_FROZEN_FOLD0_PREREGISTRATION.json"); p.add_argument("--provenance",type=Path,default=output/"AITEAMVN_V2_FROZEN_FOLD0_PROVENANCE_AUDIT.md"); p.add_argument("--score-db",type=Path,default=cache/"scores.sqlite"); p.add_argument("--input-manifest",type=Path,default=output/"AITEAMVN_V2_FROZEN_FOLD0_INPUT_MANIFEST.json"); p.add_argument("--parity-report",type=Path,default=output/"AITEAMVN_V2_FROZEN_FOLD0_PARITY.json"); p.add_argument("--report",type=Path,default=output/"AITEAMVN_V2_FROZEN_FOLD0_REPORT.json"); p.add_argument("--predictions",type=Path,default=output/"AITEAMVN_V2_FROZEN_FOLD0_PREDICTIONS.jsonl"); p.add_argument("--prediction-lock",type=Path,default=output/"AITEAMVN_V2_FROZEN_FOLD0_PREDICTION_LOCK.json"); p.add_argument("--output-manifest",type=Path,default=output/"AITEAMVN_V2_FROZEN_FOLD0_OUTPUT_MANIFEST.json"); return p


def main():
    args=parser().parse_args(); args.score_db.parent.mkdir(parents=True,exist_ok=True); args.report.parent.mkdir(parents=True,exist_ok=True)
    if args.stage in {"preflight","all"}: preflight(args)
    if args.stage in {"parity","all"}: parity(args)
    if args.stage in {"score","all"}: score(args)
    if args.stage in {"evaluate","all"}: evaluate(args)
    if args.stage=="verify": verify(args)


if __name__ == "__main__": main()
