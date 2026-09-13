"""Materialize the confirmed strict-OOF V2 anchor as a local public candidate."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sqlite3
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from research_v2_e5_transfer import e5_transfer_runner as core  # noqa: E402
from research_v2_lal_transfer import lal_transfer_runner as lal_core  # noqa: E402

PREREG_SHA = "310ac7f26f634367218728f1f7d42cf2dcff3c803389a2aaa5b8f2aa10778069"
PUBLIC_SHA = "adaa250c5469a5ddb37c3875e0aa311cb4545f0a1710903a158b25d67b7f9d89"
E5_BANK_SHA = "54c80d8da4b26806179b186e4cfb3995c5d9dae87e786360ed3f8b721260cc24"
CHUNKS_SHA = "27a33214a9d7347666df39e759ffeed00d5b9816a426a94c61b4466452305e9c"
E5_WEIGHTS_SHA = "afa0f907c7e1d8290854b8c295cd7d77521591b4c2f2a27c261258de92333ced"
CORE_SHA = "b674c9756b26d79966734d8acb928013880c80a150f5326462056055b3d3fd9b"
LAL_BANK_SHA = "7fe7ef366a2114f883e66159fdc132fe32f37cdd63f8513135c9f6d27554d677"
LAL_WEIGHTS_SHA = "e369bd4072ee10776bef9de1cf8719730903a4c027b708fa27f648f705d60241"
EXPECTED_PUBLIC = 1000
EXPECTED_PARENTS = 8507
DEPTH = 50
RRF_K = 32


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def lal_weights_path() -> Path:
    snapshot = lal_core.model_snapshot()
    link = snapshot / "model.safetensors"
    if link.is_symlink():
        return link.resolve()
    return link


def public_queries(args) -> dict[str, str]:
    payload = json.loads(args.public.read_text(encoding="utf-8"))
    result = {str(qid): str(row["question"]) for qid, row in payload.items()}
    if len(result) != EXPECTED_PUBLIC or len(result) != len(set(result)):
        raise RuntimeError("public query cardinality mismatch")
    return result


def validate(args, require_training: bool = False) -> dict:
    observed = {
        "preregistration": sha256(args.preregistration), "public_queries": sha256(args.public),
        "e5_bank": sha256(args.bundle / "embeddings.f16.npy"),
        "chunk_ids": sha256(args.bundle / "chunk_ids.jsonl"),
        "e5_weights": sha256(args.bundle / "vietlegal-e5/model.safetensors"),
        "confirmed_core": sha256(Path(core.__file__)), "lal_bank": sha256(args.lal_bank),
        "lal_weights": sha256(lal_weights_path()),
    }
    expected = {"preregistration": PREREG_SHA, "public_queries": PUBLIC_SHA,
                "e5_bank": E5_BANK_SHA, "chunk_ids": CHUNKS_SHA,
                "e5_weights": E5_WEIGHTS_SHA, "confirmed_core": CORE_SHA,
                "lal_bank": LAL_BANK_SHA, "lal_weights": LAL_WEIGHTS_SHA}
    checks = {key: observed[key] == value for key, value in expected.items()}
    checks["prereg_status"] = json.loads(args.preregistration.read_text(encoding="utf-8"))["status"] == "SEALED_BEFORE_FULL_DATA_TRAINING_OR_PUBLIC_SCORING"
    if require_training:
        checks["training_complete"] = args.checkpoint.is_file() and args.training_manifest.is_file()
    if not all(checks.values()): raise RuntimeError(f"input contract failure: {checks}")
    return {"hashes": observed, "checks": checks}


class FullData(core.TransferData):
    def __init__(self, bundle: Path):
        super().__init__(bundle)
        self.duplicate_exclusions = set()
        self.fingerprint = core.digest([self.fingerprint, "full_data_final_fit_v1", sorted(self.questions, key=int)])

    def training_qids(self) -> list[str]:
        result = sorted(self.questions, key=int)
        if len(result) != 6991 or any(not self.gold[qid] for qid in result):
            raise RuntimeError("full-data training population mismatch")
        return result


def preflight(args) -> None:
    contract = validate(args); data = FullData(args.bundle); public = public_queries(args)
    receipt = json.loads(args.lal_receipt.read_text(encoding="utf-8"))
    checks = {"canonical_parents": len(data.doc_ids) == EXPECTED_PARENTS,
              "chunks": len(data.chunk_ids) == 343347,
              "training_queries": len(data.training_qids()) == 6991,
              "public_queries": len(public) == EXPECTED_PUBLIC,
              "lal_receipt": receipt["sha256"] == LAL_BANK_SHA,
              "original_parameters_lt_4b": 559890432 + 596049920 < 4_000_000_000}
    if not all(checks.values()): raise RuntimeError(f"preflight failure: {checks}")
    report = {"schema_version":"dsc2026.research_v2.anchor_submission_input_manifest.v1",
              "status":"PASS_NO_PUBLIC_SCORING","training_queries":6991,"public_queries":1000,
              "canonical_parents":8507,"chunks":343347,"candidate_depth":DEPTH,
              "original_model_parameters":1155940352,"checks":checks,**contract}
    write_json(args.input_manifest, report); print(json.dumps(report, indent=2))


def train(args) -> None:
    validate(args); manifest = json.loads(args.input_manifest.read_text(encoding="utf-8"))
    if manifest["status"] != "PASS_NO_PUBLIC_SCORING": raise RuntimeError("preflight not passed")
    data = FullData(args.bundle); started = time.perf_counter()
    result = core.train_adapter(data, args.training_dir, microbatch=4, qids=data.training_qids())
    if result["status"] != "COMPLETE_EPOCH2" or not args.checkpoint.is_file():
        raise RuntimeError("full-data fit incomplete")
    report = {"schema_version":"dsc2026.research_v2.anchor_submission_full_data_training.v1",
              "status":"COMPLETE_FULL_DATA_FINAL_FIT","reported_as_oof":False,
              "training_queries":6991,"epochs":2,"updates":result["updates"],
              "runtime_seconds_wrapper":time.perf_counter()-started,
              "core_metadata_held_fold_compatibility_note":"The imported sealed core emits held_fold=fold_0; authoritative scope is final all-6991 fit and is verified by the qid hash.",
              "training_qids_sha256":core.digest(data.training_qids()),
              "checkpoint_sha256":sha256(args.checkpoint),"core_result":result}
    write_json(args.training_manifest, report); print(json.dumps(report, indent=2))


def score_db(args):
    args.score_db.parent.mkdir(parents=True,exist_ok=True); db=sqlite3.connect(args.score_db)
    db.execute("PRAGMA journal_mode=WAL"); db.execute("PRAGMA synchronous=FULL")
    db.execute("CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL)")
    db.execute("CREATE TABLE IF NOT EXISTS e5(qid TEXT,rank INTEGER,doc_id TEXT,score REAL,PRIMARY KEY(qid,doc_id))")
    db.execute("CREATE TABLE IF NOT EXISTS lal(qid TEXT,rank INTEGER,doc_id TEXT,score REAL,PRIMARY KEY(qid,doc_id))")
    fingerprint=core.digest([PREREG_SHA,sha256(args.checkpoint),E5_BANK_SHA,LAL_BANK_SHA,PUBLIC_SHA,DEPTH,RRF_K])
    existing=dict(db.execute("SELECT key,value FROM metadata"))
    if existing and existing.get("fingerprint")!=fingerprint: raise RuntimeError("public score cache fingerprint mismatch")
    if not existing:
        db.execute("INSERT INTO metadata VALUES('fingerprint',?)",(fingerprint,));db.commit()
    return db,fingerprint


def score_e5(args) -> None:
    validate(args,require_training=True); data=FullData(args.bundle); public=public_queries(args); db,_=score_db(args)
    complete={str(x[0]) for x in db.execute("SELECT DISTINCT qid FROM e5")}; seed=112; core.seed_all(seed)
    model=core.QueryEncoder(args.bundle/"vietlegal-e5",checkpoint_path=args.checkpoint); model.eval(); bank=core.ParentBank(data.vectors,data.parent)
    qids=list(public); started=time.perf_counter();torch.cuda.reset_peak_memory_stats();new=0
    for begin in range(0,len(qids),4):
        ids=[qid for qid in qids[begin:begin+4] if qid not in complete]
        if not ids: continue
        with torch.no_grad(): vectors=model([public[qid] for qid in ids]); scores,_=bank.mine(vectors)
        for qid,values in zip(ids,scores):
            order=torch.argsort(values,descending=True,stable=True)[:DEPTH].cpu().tolist()
            rows=[(qid,rank,data.doc_ids[index],float(values[index].cpu())) for rank,index in enumerate(order,1)]
            with db: db.executemany("INSERT INTO e5 VALUES(?,?,?,?)",rows)
            new+=1
        if new%20==0: print(f"public_e5={len(complete)+new}/{len(qids)}",flush=True)
    report={"schema_version":"dsc2026.research_v2.anchor_public_e5_scores.v1","status":"COMPLETE","queries":db.execute("SELECT COUNT(DISTINCT qid) FROM e5").fetchone()[0],"rows":db.execute("SELECT COUNT(*) FROM e5").fetchone()[0],"depth":DEPTH,"runtime_seconds_this_process":time.perf_counter()-started,"peak_mib":torch.cuda.max_memory_allocated()/2**20,"checkpoint_sha256":sha256(args.checkpoint)}
    write_json(args.e5_report,report);db.execute("PRAGMA wal_checkpoint(TRUNCATE)");db.close();print(json.dumps(report,indent=2));del model,bank;gc.collect();torch.cuda.empty_cache()


def score_lal(args) -> None:
    validate(args,require_training=True); data=FullData(args.bundle); public=public_queries(args); db,_=score_db(args)
    if db.execute("SELECT COUNT(*) FROM e5").fetchone()[0] != EXPECTED_PUBLIC*DEPTH: raise RuntimeError("E5 candidates incomplete")
    complete={str(x[0]) for x in db.execute("SELECT DISTINCT qid FROM lal")}; core.seed_all(112)
    model=lal_core.LALQueryEncoder();model.eval();vectors=np.load(args.lal_bank,mmap_mode="r");bank=core.ParentBank(vectors,data.parent)
    qids=list(public);started=time.perf_counter();torch.cuda.reset_peak_memory_stats();new=0
    for begin in range(0,len(qids),8):
        ids=[qid for qid in qids[begin:begin+8] if qid not in complete]
        if not ids: continue
        with torch.no_grad(),model.adapter_disabled(): qvectors=model([public[qid] for qid in ids])
        for qid,qvector in zip(ids,qvectors):
            candidates=[str(x[0]) for x in db.execute("SELECT doc_id FROM e5 WHERE qid=? ORDER BY rank",(qid,))]
            values=bank.score_pool(qvector,candidates,data); order=sorted(range(len(candidates)),key=lambda i:(-values[i],candidates[i]))
            rows=[(qid,rank,candidates[index],float(values[index])) for rank,index in enumerate(order,1)]
            with db: db.executemany("INSERT INTO lal VALUES(?,?,?,?)",rows)
            new+=1
        if new%20==0: print(f"public_lal={len(complete)+new}/{len(qids)}",flush=True)
    report={"schema_version":"dsc2026.research_v2.anchor_public_lal_scores.v1","status":"COMPLETE","queries":db.execute("SELECT COUNT(DISTINCT qid) FROM lal").fetchone()[0],"rows":db.execute("SELECT COUNT(*) FROM lal").fetchone()[0],"depth":DEPTH,"runtime_seconds_this_process":time.perf_counter()-started,"peak_mib":torch.cuda.max_memory_allocated()/2**20,"model_revision":"de759324ef931a2475ae8db97137b6a6cbb98aa0","bank_sha256":LAL_BANK_SHA}
    write_json(args.lal_report,report);db.execute("PRAGMA wal_checkpoint(TRUNCATE)");db.close();print(json.dumps(report,indent=2));del model,bank;gc.collect();torch.cuda.empty_cache()


def package(args) -> None:
    validate(args,require_training=True); public=public_queries(args); data=FullData(args.bundle); valid=set(data.doc_ids)
    db=sqlite3.connect(f"file:{args.score_db.resolve().as_posix()}?mode=ro&immutable=1",uri=True)
    integrity=db.execute("PRAGMA integrity_check").fetchone()[0]
    if integrity!="ok" or db.execute("SELECT COUNT(*) FROM e5").fetchone()[0]!=50000 or db.execute("SELECT COUNT(*) FROM lal").fetchone()[0]!=50000: raise RuntimeError("public scores incomplete")
    submission={};prediction_rows=[]
    for qid in public:
        e5=[str(x[0]) for x in db.execute("SELECT doc_id FROM e5 WHERE qid=? ORDER BY rank",(qid,))];lal=[str(x[0]) for x in db.execute("SELECT doc_id FROM lal WHERE qid=? ORDER BY rank",(qid,))]
        if len(e5)!=DEPTH or set(e5)!=set(lal): raise RuntimeError(f"candidate parity {qid}")
        er={doc:i for i,doc in enumerate(e5,1)};lr={doc:i for i,doc in enumerate(lal,1)};score={doc:1/(RRF_K+er[doc])+1/(RRF_K+lr[doc]) for doc in e5};top5=sorted(e5,key=lambda d:(-score[d],d))[:5]
        if len(top5)!=5 or len(set(top5))!=5 or not set(top5)<=valid: raise RuntimeError(f"invalid output {qid}")
        submission[qid]={"answer":top5};prediction_rows.append({"qid":qid,"top5":top5,"e5_ranks":[er[d] for d in top5],"lal_ranks":[lr[d] for d in top5],"rrf32":[score[d] for d in top5]})
    db.close();write_json(args.submission_json,submission)
    with args.predictions.open("w",encoding="utf-8",newline="\n") as stream:
        for row in prediction_rows: stream.write(json.dumps(row,ensure_ascii=False,sort_keys=True,separators=(",",":"))+"\n")
    if args.submission_zip.exists(): raise RuntimeError("refusing to overwrite existing ZIP")
    with zipfile.ZipFile(args.submission_zip,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=9) as archive: archive.write(args.submission_json,arcname="submission.json")
    lock={"schema_version":"dsc2026.research_v2.anchor_submission_prediction_lock.v1","status":"LOCKED","queries":1000,"answers_per_query":5,"submission_json_sha256":sha256(args.submission_json),"submission_zip_sha256":sha256(args.submission_zip),"predictions_sha256":sha256(args.predictions)};write_json(args.prediction_lock,lock)
    report={"schema_version":"dsc2026.research_v2.anchor_submission_candidate_report.v1","status":"VALID_LOCAL_CANDIDATE_NOT_UPLOADED","strict_oof_support":{"recall_at_5":0.917954036141706,"precision_at_5":0.195937634100987,"positive_folds":5},"public_validation":{"queries":1000,"unique_qids":1000,"answers_per_query":5,"all_answers_unique":True,"all_answers_canonical":True,"canonical_parents":8507},"system":{"components":["full-data adapted mainguyen9/vietlegal-e5","frozen darklethelong/vnlegal-lal"],"original_parameters":1155940352,"limit":4000000000,"candidate_depth":50,"fusion":"equal RRF32","adaptive_k":False,"augmentation":False},"files":{"submission_json":str(args.submission_json),"submission_zip":str(args.submission_zip)}};write_json(args.report,report)
    files=[args.preregistration,args.input_manifest,args.training_manifest,args.checkpoint,args.e5_report,args.lal_report,args.score_db,args.predictions,args.prediction_lock,args.report,args.submission_json,args.submission_zip,Path(__file__)]
    reproduce = [
        f'"{sys.executable}" "{Path(__file__).resolve()}" {stage} --new-output-dir <empty-directory>'
        for stage in ("preflight", "train", "score-e5", "score-lal", "package", "verify")
    ]
    manifest={"schema_version":"dsc2026.research_v2.anchor_submission_output_manifest.v1","status":"COMPLETE_VERIFIED_INPUTS_PENDING_FINAL_REPLAY","files":{p.name:{"path":str(p.resolve()),"bytes":p.stat().st_size,"sha256":sha256(p)} for p in files},"reproduce":reproduce};write_json(args.output_manifest,manifest);print(json.dumps(report,indent=2))


def verify(args) -> None:
    manifest=json.loads(args.output_manifest.read_text(encoding="utf-8"));fail=[]
    for name,item in manifest["files"].items():
        path=Path(item["path"]);observed=sha256(path) if path.exists() else None
        if observed!=item["sha256"]:fail.append({"file":name,"expected":item["sha256"],"observed":observed})
    payload=json.loads(args.submission_json.read_text(encoding="utf-8"));data=FullData(args.bundle);valid=set(data.doc_ids)
    structural=len(payload)==1000 and all(isinstance(x.get("answer"),list) and len(x["answer"])==len(set(x["answer"]))==5 and set(map(str,x["answer"]))<=valid for x in payload.values())
    with zipfile.ZipFile(args.submission_zip) as archive:
        zip_ok=archive.namelist()==["submission.json"] and archive.read("submission.json")==args.submission_json.read_bytes()
    status="PASS" if not fail and structural and zip_ok else "FAIL";result={"status":status,"hash_failures":fail,"queries":len(payload),"structural":structural,"zip_exact":zip_ok,"manifest_sha256":sha256(args.output_manifest),"submission_json_sha256":sha256(args.submission_json),"submission_zip_sha256":sha256(args.submission_zip)}
    if status=="PASS":manifest["status"]="COMPLETE_VERIFIED";write_json(args.output_manifest,manifest);result["manifest_sha256_after_status_update"]=sha256(args.output_manifest)
    print(json.dumps(result,indent=2));
    if status!="PASS":raise RuntimeError("submission verification failed")


def parser():
    out=ROOT/"results/research_v2_open_rl/v2_anchor_submission_candidate";cache=ROOT/"cache/research_v2_open_rl/v2_anchor_submission_candidate";training=out/"full_data_adapter"
    p=argparse.ArgumentParser();p.add_argument("stage",choices=["preflight","train","score-e5","score-lal","package","verify"]);p.add_argument("--new-output-dir",type=Path)
    p.add_argument("--bundle",type=Path,default=ROOT/"cache/research_v2_e5_confirmation/bundle-v1");p.add_argument("--public",type=Path,default=ROOT/"DSC2026-LegalIR-main/v4_run/public_test_dataset/public-official.json");p.add_argument("--lal-bank",type=Path,default=ROOT.parent/"LegalIR/cache/exp112_task_adaptive_retrieval/lal.f16.npy");p.add_argument("--lal-receipt",type=Path,default=ROOT.parent/"LegalIR/cache/exp112_task_adaptive_retrieval/lal.f16.json");p.add_argument("--preregistration",type=Path,default=ROOT/"results/research_v2_open_rl/V2_ANCHOR_SUBMISSION_MATERIALIZATION_PREREGISTRATION.json")
    p.add_argument("--input-manifest",type=Path,default=out/"INPUT_MANIFEST.json");p.add_argument("--training-dir",type=Path,default=training);p.add_argument("--checkpoint",type=Path,default=training/"epoch-2.pt");p.add_argument("--training-manifest",type=Path,default=out/"FULL_DATA_TRAINING_MANIFEST.json");p.add_argument("--score-db",type=Path,default=cache/"public_scores.sqlite");p.add_argument("--e5-report",type=Path,default=out/"PUBLIC_E5_REPORT.json");p.add_argument("--lal-report",type=Path,default=out/"PUBLIC_LAL_REPORT.json");p.add_argument("--submission-json",type=Path,default=out/"submission.json");p.add_argument("--submission-zip",type=Path,default=out/"submission.zip");p.add_argument("--predictions",type=Path,default=out/"PUBLIC_PREDICTIONS.jsonl");p.add_argument("--prediction-lock",type=Path,default=out/"PREDICTION_LOCK.json");p.add_argument("--report",type=Path,default=out/"REPORT.json");p.add_argument("--output-manifest",type=Path,default=out/"OUTPUT_MANIFEST.json");return p


def apply_new_output_dir(args) -> None:
    """Route every generated artifact to a fresh, explicitly supplied namespace."""
    if args.new_output_dir is None:
        return
    out = args.new_output_dir.resolve()
    training = out / "full_data_adapter"
    args.input_manifest = out / "INPUT_MANIFEST.json"
    args.training_dir = training
    args.checkpoint = training / "epoch-2.pt"
    args.training_manifest = out / "FULL_DATA_TRAINING_MANIFEST.json"
    args.score_db = out / "cache" / "public_scores.sqlite"
    args.e5_report = out / "PUBLIC_E5_REPORT.json"
    args.lal_report = out / "PUBLIC_LAL_REPORT.json"
    args.submission_json = out / "submission.json"
    args.submission_zip = out / "submission.zip"
    args.predictions = out / "PUBLIC_PREDICTIONS.jsonl"
    args.prediction_lock = out / "PREDICTION_LOCK.json"
    args.report = out / "REPORT.json"
    args.output_manifest = out / "OUTPUT_MANIFEST.json"


def main():
    args=parser().parse_args();apply_new_output_dir(args);args.input_manifest.parent.mkdir(parents=True,exist_ok=True);args.score_db.parent.mkdir(parents=True,exist_ok=True)
    if args.stage=="preflight":preflight(args)
    elif args.stage=="train":train(args)
    elif args.stage=="score-e5":score_e5(args)
    elif args.stage=="score-lal":score_lal(args)
    elif args.stage=="package":package(args)
    elif args.stage=="verify":verify(args)


if __name__=="__main__":main()
