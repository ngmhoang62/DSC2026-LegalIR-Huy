"""Sealed strict Fold-0 E5 passage-side alignment experiment."""

from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import json
import math
import os
import random
import sqlite3
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import legal_mlm_query_likelihood_fold0 as common

SEED = 113
EXPECTED_QUERIES = 1398
EXPECTED_PAIRS = 73128
BASE_PARAMETERS = 559890432
LAL_PARAMETERS = 596049920
ORIGINAL_SYSTEM_PARAMETERS = BASE_PARAMETERS + LAL_PARAMETERS
FOLDS_SHA = "94ad5c6d5e582ced5eec8d2c3c15f938454c17e713614391091e72abea9aba19"
POOL_SHA = "96a44e66549cc211e1f9d0fabb84fc825db3f21f32d5b349eeca3b1c0413e277"
GROUPS_SHA = "681ca1340dac9e6b498cbde30fa8ac7fee98840664ae51a126491d9173009b2f"
GROUPS_MANIFEST_SHA = "4a6270b9f3e0f9f7f260443107b9868d3fcf634de6a8c0d93e32a20d95a59644"
QUERY_ADAPTER_SHA = "ef4c293ba78917522c81fa119a7406c3c936330c0985b53e5a0188cf91e36cc6"
MODEL_SHA = "afa0f907c7e1d8290854b8c295cd7d77521591b4c2f2a27c261258de92333ced"
QUERY_CORE_SHA = "b674c9756b26d79966734d8acb928013880c80a150f5326462056055b3d3fd9b"
CONTEXTS_SHA = "55c77371edd3b4f28e3e8ca548447e27424e57ce219e3d6da7e9f51238a22291"
ANCHOR_SHA = "1854494964f2258243bc00896c76b11d56a3c02752a23af0ee04bac8e260de4d"
SOURCES_SHA = "ef763bf8c5e3da91fb6447f8a03ab321fdaca5362bfde0213057a15311824f1e"
PREREG_SHA = "10e30921bce67a6276355be5fc43cb65c0d98a84d477341a83618111c8b7c192"
TEMPERATURE = 0.05
DRIFT_WEIGHT = 0.05
EPOCHS = 2
EFFECTIVE_BATCH = 16
LR = 5e-5
WEIGHT_DECAY = 0.01
MAX_LENGTH = 512


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def records(path: Path):
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def seed_all() -> None:
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def input_hashes(args) -> dict[str, str]:
    return {
        "folds": sha256(args.folds), "pool": sha256(args.pool),
        "boundary_groups": sha256(args.groups),
        "boundary_manifest": sha256(args.groups_manifest),
        "query_adapter": sha256(args.query_adapter),
        "model_safetensors": sha256(args.model / "model.safetensors"),
        "query_core": sha256(args.query_core), "contexts": sha256(args.contexts),
        "anchor": sha256(args.anchor), "sources_db": sha256(args.sources_db),
        "preregistration": sha256(args.preregistration),
        "renderer": sha256(args.renderer),
    }


def validate(args, require_checkpoint: bool = False, require_query_cache: bool = False) -> dict:
    hashes = input_hashes(args)
    expected = {
        "folds": FOLDS_SHA, "pool": POOL_SHA, "boundary_groups": GROUPS_SHA,
        "boundary_manifest": GROUPS_MANIFEST_SHA, "query_adapter": QUERY_ADAPTER_SHA,
        "model_safetensors": MODEL_SHA, "query_core": QUERY_CORE_SHA,
        "contexts": CONTEXTS_SHA, "anchor": ANCHOR_SHA, "sources_db": SOURCES_SHA,
        "preregistration": PREREG_SHA,
    }
    checks = {key: hashes[key] == value for key, value in expected.items()}
    prereg = json.loads(args.preregistration.read_text(encoding="utf-8"))
    checks["preregistered"] = prereg["status"] == "SEALED_AFTER_INPUT_CARDINALITY_ONLY_BEFORE_TRAINING_OR_V2_METRIC"
    checks["no_augmentation"] = prereg["training"]["augmentation"] is False
    checks["parameter_limit"] = ORIGINAL_SYSTEM_PARAMETERS < 4_000_000_000
    if require_query_cache:
        checks["query_cache"] = args.query_vectors.is_file() and args.query_ids.is_file() and args.query_cache_manifest.is_file()
    if require_checkpoint:
        checks["passage_checkpoint"] = args.checkpoint.is_file() and args.training_success.is_file()
    if not all(checks.values()):
        raise RuntimeError(f"contract failure: {checks}")
    return {"hashes": hashes, "checks": checks}


def load_groups(args) -> list[dict]:
    rows = list(records(args.groups))
    if len(rows) != 6991 or len({str(row["qid"]) for row in rows}) != 6991:
        raise RuntimeError("boundary group cardinality mismatch")
    for row in rows:
        if not row["positives"] or len(row["negatives"]) != 4:
            raise RuntimeError(f"boundary group structure mismatch: {row['qid']}")
        for parent in row["positives"] + row["negatives"]:
            if len(parent["passages"]) not in (1, 2):
                raise RuntimeError(f"passage count mismatch: {row['qid']} {parent['doc_id']}")
        if set(str(x["doc_id"]) for x in row["positives"]) & set(str(x["doc_id"]) for x in row["negatives"]):
            raise RuntimeError(f"sibling gold negative: {row['qid']}")
    return rows


def training_rows(args, rows: list[dict]) -> list[dict]:
    manifest = json.loads(args.groups_manifest.read_text(encoding="utf-8"))
    excluded = set(map(str, manifest["held_fold_duplicate_exclusions"]["fold_0"]))
    selected = [row for row in rows if row["fold"] != "fold_0" and str(row["qid"]) not in excluded]
    if len(selected) != 5586 or any(row["fold"] == "fold_0" for row in selected):
        raise RuntimeError("strict Fold-0 training isolation failure")
    return selected


class PassageEncoder(torch.nn.Module):
    def __init__(self, model_path: Path, dtype: torch.dtype = torch.float32, checkpoint: Path | None = None):
        super().__init__()
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModel, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
        base = AutoModel.from_pretrained(str(model_path), local_files_only=True,
                                         torch_dtype=dtype, attn_implementation="eager")
        count = sum(value.numel() for value in base.parameters())
        if count != BASE_PARAMETERS: raise RuntimeError(f"base parameter mismatch: {count}")
        self.model = get_peft_model(base, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                                                     target_modules=["query", "value"], bias="none"))
        self.model.config.use_cache = False
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.model.enable_input_require_grads()
        if checkpoint is not None:
            state = torch.load(checkpoint, map_location="cpu", weights_only=False)
            self.load_state_dict(state["adapter"], strict=False)
        self.to("cuda")

    def tokenize(self, texts: list[str]):
        batch = self.tokenizer(["passage: " + text for text in texts], padding=True,
                               truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
        return {key: value.to("cuda", non_blocking=True) for key, value in batch.items()}

    def encode_batch(self, batch) -> torch.Tensor:
        hidden = self.model(**batch).last_hidden_state.float()
        mask = batch["attention_mask"].unsqueeze(-1)
        return F.normalize((hidden * mask).sum(1) / mask.sum(1).clamp_min(1), dim=-1)

    @contextlib.contextmanager
    def adapter_disabled(self):
        with self.model.disable_adapter(): yield

    def adapter_state(self) -> dict[str, torch.Tensor]:
        return {name: value.detach().cpu() for name, value in self.named_parameters() if value.requires_grad}


def prepare_query_cache(args) -> None:
    validate(args); rows = load_groups(args)
    if args.query_cache_manifest.exists():
        manifest = json.loads(args.query_cache_manifest.read_text(encoding="utf-8"))
        if (sha256(args.query_vectors) == manifest["vectors_sha256"] and
                sha256(args.query_ids) == manifest["ids_sha256"]):
            print(json.dumps(manifest, indent=2)); return
        raise RuntimeError("query cache manifest exists but hashes differ")
    import importlib.util
    spec = importlib.util.spec_from_file_location("sealed_e5_core", args.query_core)
    core = importlib.util.module_from_spec(spec); assert spec.loader is not None; spec.loader.exec_module(core)
    seed_all(); model = core.QueryEncoder(args.model, checkpoint_path=args.query_adapter); model.eval()
    qids = [str(row["qid"]) for row in rows]; texts = {str(row["qid"]): str(row["query"]) for row in rows}
    vectors = []
    started = time.perf_counter(); torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for start in range(0, len(qids), 8):
            vectors.append(model([texts[qid] for qid in qids[start:start + 8]]).cpu().numpy())
            if start % 400 == 0: print(f"query_cache={min(start + 8, len(qids))}/{len(qids)}", flush=True)
    matrix = np.concatenate(vectors).astype(np.float32)
    if matrix.shape != (6991, 1024) or not np.isfinite(matrix).all(): raise RuntimeError("query cache invalid")
    args.query_vectors.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.query_vectors, matrix, allow_pickle=False)
    write_json(args.query_ids, qids)
    manifest = {"schema_version": "dsc2026.research_v2.e5_passage_alignment_query_cache.v1",
                "status": "COMPLETE_NO_LABEL_METRIC", "queries": len(qids), "shape": list(matrix.shape),
                "runtime_seconds": time.perf_counter() - started,
                "peak_mib": torch.cuda.max_memory_allocated() / 2**20,
                "query_adapter_sha256": QUERY_ADAPTER_SHA, "vectors_sha256": sha256(args.query_vectors),
                "ids_sha256": sha256(args.query_ids)}
    write_json(args.query_cache_manifest, manifest); print(json.dumps(manifest, indent=2))
    del model; gc.collect(); torch.cuda.empty_cache()


def query_cache(args) -> tuple[np.ndarray, dict[str, int]]:
    ids = list(map(str, json.loads(args.query_ids.read_text(encoding="utf-8"))))
    matrix = np.load(args.query_vectors, mmap_mode="r")
    if matrix.shape != (6991, 1024) or len(ids) != len(set(ids)) != 6991: raise RuntimeError("query cache shape")
    return matrix, {qid: index for index, qid in enumerate(ids)}


def group_loss(model: PassageEncoder, row: dict, query: torch.Tensor) -> torch.Tensor:
    parents = row["positives"] + row["negatives"]
    texts = [text for parent in parents for text in parent["passages"]]
    batch = model.tokenize(texts)
    with torch.no_grad(), model.adapter_disabled(): frozen = model.encode_batch(batch).detach()
    adapted = model.encode_batch(batch)
    raw_scores = (adapted * query[None]).sum(1)
    parent_scores = []
    offset = 0
    for parent in parents:
        count = len(parent["passages"])
        parent_scores.append(raw_scores[offset:offset + count].mean())
        offset += count
    scores = torch.stack(parent_scores)
    positive_count = len(row["positives"])
    positives, negatives = scores[:positive_count] / TEMPERATURE, scores[positive_count:] / TEMPERATURE
    contrastive = (torch.logaddexp(positives, torch.logsumexp(negatives, dim=0)) - positives).mean()
    drift = (1 - F.cosine_similarity(adapted, frozen, dim=1)).mean()
    return contrastive + DRIFT_WEIGHT * drift


def smoke(args) -> None:
    validate(args, require_query_cache=True); rows = training_rows(args, load_groups(args))[:2]
    vectors, lookup = query_cache(args); results = []
    for replay in range(2):
        seed_all(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        model = PassageEncoder(args.model); model.train()
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=WEIGHT_DECAY)
        started = time.perf_counter(); optimizer.zero_grad(set_to_none=True); losses = []
        for row in rows:
            q = torch.as_tensor(np.array(vectors[lookup[str(row["qid"])]], copy=True), device="cuda")
            loss = group_loss(model, row, q) / len(rows); loss.backward(); losses.append(float(loss.detach()))
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step(); state_hash = digest({k: hashlib.sha256(v.numpy().tobytes()).hexdigest()
                                              for k, v in model.adapter_state().items()})
        results.append({"losses": losses, "gradient_norm": float(norm), "adapter_state_hash": state_hash,
                        "seconds": time.perf_counter() - started,
                        "peak_mib": torch.cuda.max_memory_allocated() / 2**20})
        del model, optimizer; gc.collect(); torch.cuda.empty_cache()
    replay_exact = results[0]["adapter_state_hash"] == results[1]["adapter_state_hash"]
    projected_training = results[0]["seconds"] / len(rows) * 5586 * EPOCHS
    # Same 146,249-sequence encoder family; use a conservative 5,000 second scoring allowance.
    projected_total = projected_training + 5000
    status = "PASS" if replay_exact and results[0]["gradient_norm"] > 0 and max(x["peak_mib"] for x in results) < 5600 and projected_total < 10800 else "FAIL"
    report = {"schema_version": "dsc2026.research_v2.e5_passage_alignment_smoke.v1",
              "status": status, "groups": len(rows), "replays": results,
              "deterministic_replay_exact": replay_exact, "projected_training_seconds": projected_training,
              "projected_train_plus_score_seconds": projected_total, "cost_limit_seconds": 10800}
    write_json(args.smoke_report, report); print(json.dumps(report, indent=2))
    if status != "PASS": raise RuntimeError("smoke/cost gate failed")


def save_training(path: Path, model, optimizer, scheduler, epoch: int, position: int, updates: int, contract_hash: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); temporary = path.with_suffix(".tmp")
    torch.save({"adapter": model.adapter_state(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "epoch": epoch, "position": position,
                "updates": updates, "contract_hash": contract_hash}, temporary)
    os.replace(temporary, path)


def train(args) -> None:
    from transformers import get_cosine_schedule_with_warmup
    validate(args, require_query_cache=True)
    smoke_report = json.loads(args.smoke_report.read_text(encoding="utf-8"))
    if smoke_report["status"] != "PASS": raise RuntimeError("smoke not authorized")
    rows = training_rows(args, load_groups(args)); vectors, lookup = query_cache(args)
    contract = {"preregistration": PREREG_SHA, "groups": GROUPS_SHA,
                "query_adapter": QUERY_ADAPTER_SHA, "qids": [str(row["qid"]) for row in rows],
                "epochs": EPOCHS, "effective_batch": EFFECTIVE_BATCH, "seed": SEED}
    contract_hash = digest(contract)
    if args.training_success.exists():
        done = json.loads(args.training_success.read_text(encoding="utf-8"))
        if done["contract_hash"] != contract_hash: raise RuntimeError("completed contract mismatch")
        print(json.dumps(done, indent=2)); return
    seed_all(); model = PassageEncoder(args.model); model.train()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=WEIGHT_DECAY)
    steps_epoch = math.ceil(len(rows) / EFFECTIVE_BATCH)
    scheduler = get_cosine_schedule_with_warmup(optimizer, max(1, int(0.1 * steps_epoch * EPOCHS)), steps_epoch * EPOCHS)
    start_epoch = start_position = updates = 0
    if args.resume.exists():
        state = torch.load(args.resume, map_location="cpu", weights_only=False)
        if state["contract_hash"] != contract_hash: raise RuntimeError("resume contract mismatch")
        model.load_state_dict(state["adapter"], strict=False); optimizer.load_state_dict(state["optimizer"]); scheduler.load_state_dict(state["scheduler"])
        start_epoch, start_position, updates = int(state["epoch"]), int(state["position"]), int(state["updates"])
    started = time.perf_counter(); torch.cuda.reset_peak_memory_stats()
    for epoch in range(start_epoch, EPOCHS):
        order = list(range(len(rows))); random.Random(SEED + epoch).shuffle(order)
        begin = start_position if epoch == start_epoch else 0
        for position in range(begin, len(order), EFFECTIVE_BATCH):
            chosen = order[position:position + EFFECTIVE_BATCH]; optimizer.zero_grad(set_to_none=True); total_loss = 0.0
            for index in chosen:
                row = rows[index]; q = torch.as_tensor(np.array(vectors[lookup[str(row["qid"])]], copy=True), device="cuda")
                loss = group_loss(model, row, q) / len(chosen)
                if not torch.isfinite(loss): raise FloatingPointError("nonfinite loss")
                loss.backward(); total_loss += float(loss.detach())
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            if float(norm) == 0: raise RuntimeError("zero passage-adapter gradient")
            optimizer.step(); scheduler.step(); updates += 1; next_position = position + len(chosen)
            if updates % 10 == 0 or next_position >= len(order):
                save_training(args.resume, model, optimizer, scheduler, epoch, next_position, updates, contract_hash)
                print(json.dumps({"stage": "train", "epoch": epoch + 1, "position": next_position,
                                  "queries": len(order), "updates": updates, "loss": total_loss,
                                  "gradient_norm": float(norm)}), flush=True)
        save_training(args.training_dir / f"epoch-{epoch + 1}.pt", model, optimizer, scheduler,
                      epoch + 1, 0, updates, contract_hash)
        start_position = 0
    if args.checkpoint != args.training_dir / "epoch-2.pt": raise RuntimeError("checkpoint path contract")
    done = {"schema_version": "dsc2026.research_v2.e5_passage_alignment_training.v1",
            "status": "COMPLETE_EPOCH2", "held_fold": "fold_0", "training_queries": len(rows),
            "epochs": EPOCHS, "updates": updates, "contract_hash": contract_hash,
            "runtime_seconds_this_process": time.perf_counter() - started,
            "peak_mib": torch.cuda.max_memory_allocated() / 2**20,
            "checkpoint_sha256": sha256(args.checkpoint),
            "training_qids_sha256": digest(sorted((str(row["qid"]) for row in rows), key=int))}
    write_json(args.training_success, done); print(json.dumps(done, indent=2))
    del model, optimizer, scheduler; gc.collect(); torch.cuda.empty_cache()


@torch.inference_mode()
def encode_passages(model: PassageEncoder, texts: list[str], batch_size: int) -> np.ndarray:
    output = []
    for start in range(0, len(texts), batch_size):
        output.append(model.encode_batch(model.tokenize(texts[start:start + batch_size])).cpu().numpy())
    return np.concatenate(output)


def parent_mean(owners, values) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    for (doc, _), value in zip(owners, values): grouped.setdefault(str(doc), []).append(float(value))
    if any(len(items) not in (1, 2) for items in grouped.values()): raise RuntimeError("evaluation passage count")
    return {doc: float(np.mean(items)) for doc, items in grouped.items()}


def init_db(args):
    args.score_db.parent.mkdir(parents=True, exist_ok=True); db = sqlite3.connect(args.score_db)
    db.execute("PRAGMA journal_mode=WAL"); db.execute("PRAGMA synchronous=FULL")
    db.execute("CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL)")
    db.execute("CREATE TABLE IF NOT EXISTS scores(qid TEXT,doc_id TEXT,score REAL,PRIMARY KEY(qid,doc_id))")
    db.execute("CREATE TABLE IF NOT EXISTS progress(qid TEXT PRIMARY KEY,seconds REAL,sequences INTEGER,peak_mib REAL)")
    fingerprint = digest([PREREG_SHA, sha256(args.checkpoint), sha256(args.query_vectors), POOL_SHA, CONTEXTS_SHA])
    existing = dict(db.execute("SELECT key,value FROM metadata"))
    if existing and existing.get("fingerprint") != fingerprint: raise RuntimeError("score DB fingerprint mismatch")
    if not existing:
        db.executemany("INSERT INTO metadata VALUES(?,?)", [("fingerprint", fingerprint), ("folds_sha256", FOLDS_SHA), ("pool_sha256", POOL_SHA)]); db.commit()
    return db, fingerprint


def score(args) -> None:
    validate(args, require_checkpoint=True, require_query_cache=True)
    rows = common.fold0_rows(args); documents = common.load_documents(args, rows)
    vectors, lookup = query_cache(args); db, _ = init_db(args); complete = {str(x[0]) for x in db.execute("SELECT qid FROM progress")}
    seed_all(); model = PassageEncoder(args.model, dtype=torch.float16, checkpoint=args.checkpoint); model.eval()
    process_started = time.perf_counter(); newly = 0; torch.cuda.reset_peak_memory_stats()
    for row in rows:
        qid = str(row["qid"])
        if qid in complete: continue
        started = time.perf_counter(); _, owners, passages = common.evidence_for_row(args, row, documents)
        pvec = encode_passages(model, passages, 16)
        qvec = np.array(vectors[lookup[qid]], dtype=np.float32, copy=True)
        scores = parent_mean(owners, pvec.astype(np.float32) @ qvec)
        docs = list(map(str, row["doc_ids"]))
        if set(scores) != set(docs) or any(not math.isfinite(scores[d]) for d in docs): raise RuntimeError(f"score contract {qid}")
        elapsed = time.perf_counter() - started
        with db:
            db.executemany("INSERT INTO scores VALUES(?,?,?)", [(qid, doc, scores[doc]) for doc in docs])
            db.execute("INSERT INTO progress VALUES(?,?,?,?)", (qid, elapsed, len(passages), torch.cuda.max_memory_allocated()/2**20))
        newly += 1
        if newly % 10 == 0:
            done = len(complete) + newly; qps = newly / (time.perf_counter() - process_started)
            print(f"score={done}/{EXPECTED_QUERIES} qps={qps:.3f} eta_min={(EXPECTED_QUERIES-done)/qps/60:.1f}", flush=True)
    stats = db.execute("SELECT COUNT(*),SUM(sequences),SUM(seconds),MAX(peak_mib) FROM progress").fetchone()
    count = db.execute("SELECT COUNT(*) FROM scores").fetchone()[0]; integrity = db.execute("PRAGMA integrity_check").fetchone()[0]
    db.execute("PRAGMA wal_checkpoint(TRUNCATE)"); db.close(); print(json.dumps({"progress": stats, "score_rows": count, "integrity": integrity}, indent=2))
    del model; gc.collect(); torch.cuda.empty_cache()


def evaluate(args) -> None:
    if args.report.exists(): raise RuntimeError("refusing second evaluation")
    contract = validate(args, require_checkpoint=True, require_query_cache=True); rows = common.fold0_rows(args)
    pool = {str(row["qid"]): set(map(str, row["doc_ids"])) for row in rows}
    anchor = {str(row["qid"]): row for row in common.read_jsonl(args.anchor) if str(row["qid"]) in pool}
    clean = common.load_clean_sets(args, pool); db = sqlite3.connect(f"file:{args.score_db.resolve().as_posix()}?mode=ro&immutable=1", uri=True)
    integrity = db.execute("PRAGMA integrity_check").fetchone()[0]
    progress = db.execute("SELECT COUNT(*),SUM(sequences),SUM(seconds),MAX(peak_mib) FROM progress").fetchone()
    scores = {(str(q), str(d)): float(s) for q, d, s in db.execute("SELECT qid,doc_id,score FROM scores")}; metadata = dict(db.execute("SELECT key,value FROM metadata")); db.close()
    expected = {(qid, doc) for qid, docs in pool.items() for doc in docs}
    temp, fingerprint = init_db(args); temp.close()
    integrity_ok = integrity == "ok" and progress[0] == EXPECTED_QUERIES and len(scores) == EXPECTED_PAIRS and set(scores) == expected and metadata.get("fingerprint") == fingerprint
    if not integrity_ok: raise RuntimeError("completed cache audit failed")
    expert=[];precision=[];base_r=[];clean_r=[];clean_plus=[];ceiling=[];single=[[],[]];multi=[[],[]];depth={k:[] for k in (5,10,20,50)}
    wins=losses=inside=outside=churn=0; buckets=Counter(); predictions=[]
    for row in rows:
        qid=str(row["qid"]); docs=list(map(str,row["doc_ids"])); gold=set(map(str,anchor[qid]["gold"])); order=sorted(docs,key=lambda d:(-scores[(qid,d)],d)); top5=order[:5]; base=list(map(str,anchor[qid]["fused_top5"])); er=common.recall(top5,gold); br=common.recall(base,gold)
        expert.append(er);precision.append(len(set(top5)&gold)/5);base_r.append(br);clean_r.append(common.recall(clean[qid],gold));clean_plus.append(common.recall(clean[qid]|set(top5),gold));ceiling.append(common.recall(pool[qid],gold));target=single if len(gold)==1 else multi;target[0].append(er);target[1].append(br)
        wins+=er>br;losses+=er<br;inside+=len((set(top5)-set(base))&gold);outside+=len((set(base)-set(top5))&gold);churn+=top5!=base
        for k in depth: depth[k].append(common.recall(order[:k],gold))
        for doc in gold:
            rank=order.index(doc)+1 if doc in pool[qid] else None; bucket="missing" if rank is None else "1-5" if rank<=5 else "6-10" if rank<=10 else "11-20" if rank<=20 else "21-50" if rank<=50 else "51+"; buckets[bucket]+=1
        predictions.append({"qid":qid,"top5":top5,"ranking":order,"scores":[scores[(qid,d)] for d in order]})
    metrics={"recall_at_5":float(np.mean(expert)),"precision_at_5":float(np.mean(precision)),"current_anchor_recall_at_5":float(np.mean(base_r)),"clean_expert_union":float(np.mean(clean_r)),"clean_plus_passage_alignment_union":float(np.mean(clean_plus)),"clean_union_delta":float(np.mean(clean_plus)-np.mean(clean_r)),"candidate_ceiling":float(np.mean(ceiling)),"single_gold":{"queries":len(single[0]),"expert":float(np.mean(single[0])),"anchor":float(np.mean(single[1])),"delta":float(np.mean(single[0])-np.mean(single[1]))},"multi_gold":{"queries":len(multi[0]),"expert":float(np.mean(multi[0])),"anchor":float(np.mean(multi[1])),"delta":float(np.mean(multi[0])-np.mean(multi[1]))},"per_fold":{"fold_0":{"queries":len(rows),"expert":float(np.mean(expert)),"anchor":float(np.mean(base_r)),"delta":float(np.mean(expert)-np.mean(base_r))}},"wins_losses_ties":{"wins":wins,"losses":losses,"ties":len(rows)-wins-losses},"gold_crossings":{"into_top5":inside,"out_of_top5":outside},"top5_churn_queries":churn,"recall_depth":{str(k):float(np.mean(v)) for k,v in depth.items()},"gold_rank_buckets":dict(buckets)}
    directional=wins>losses and inside>outside and metrics["multi_gold"]["delta"]>=-0.005
    pass_s=metrics["recall_at_5"]>=0.90 and metrics["clean_union_delta"]>=0.006 and directional
    pass_o=metrics["recall_at_5"]>=0.86 and metrics["clean_union_delta"]>=0.008 and directional
    kill=metrics["recall_at_5"]<0.84 or metrics["clean_union_delta"]<0.003 or wins<=losses or inside<=outside or metrics["multi_gold"]["delta"] < -0.02 or not integrity_ok or not all(contract["checks"].values())
    verdict="KILL" if kill else "PASS_STANDALONE" if pass_s else "PASS_ORTHOGONAL" if pass_o else "INCONCLUSIVE_NO_TUNING"
    with args.predictions.open("x",encoding="utf-8",newline="\n") as stream:
        for item in predictions: stream.write(json.dumps(item,ensure_ascii=False,sort_keys=True,separators=(",",":"))+"\n")
    report={"schema_version":"dsc2026.research_v2.e5_passage_alignment_fold0_report.v1","status":"COMPLETE","verdict":verdict,"metrics":metrics,"gate":{"pass_standalone":pass_s,"pass_orthogonal":pass_o,"kill":kill,"directional":directional,"integrity":integrity_ok,"contract":all(contract["checks"].values())},"runtime":{"training":json.loads(args.training_success.read_text(encoding="utf-8")),"scoring_seconds":progress[2],"scoring_sequences":progress[1],"scoring_peak_mib":progress[3]},"hashes":{**contract["hashes"],"checkpoint":sha256(args.checkpoint),"query_vectors":sha256(args.query_vectors),"score_db":sha256(args.score_db),"predictions":sha256(args.predictions)},"fixed_interface":{"query_adapter":"confirmed_fold0_frozen","passage_adapter":"qv_lora_r16_epoch2","evidence":"lexical_top2","parent_aggregation":"arithmetic_mean","ranking":"score_desc_parent_id_string_asc_top5","adaptive_k":False},"anti_rescue":"No passage-count, aggregation, LoRA, loss, LR, epoch, evidence, fusion, threshold, routing, rule, model or adaptive-K grid."}
    write_json(args.report,report); canonical=[json.dumps({"qid":r["qid"],"top5":r["top5"]},sort_keys=True,separators=(",",":")) for r in predictions]
    write_json(args.prediction_lock,{"schema_version":"dsc2026.research_v2.e5_passage_alignment_prediction_lock.v1","status":"LOCKED","verdict":verdict,"queries":len(predictions),"predictions_sha256":sha256(args.predictions),"canonical_qid_top5_sha256":hashlib.sha256(("\n".join(canonical)+"\n").encode()).hexdigest()})
    files=[args.preregistration,args.input_manifest,args.query_cache_manifest,args.smoke_report,args.training_success,args.checkpoint,args.score_db,args.report,args.predictions,args.prediction_lock]
    manifest={"schema_version":"dsc2026.research_v2.e5_passage_alignment_output_manifest.v1","status":"COMPLETE","verdict":verdict,"files":{p.name:{"path":str(p),"bytes":p.stat().st_size,"sha256":sha256(p)} for p in files},"reproduce":f'"{sys.executable}" "{Path(__file__).resolve()}" verify'}
    write_json(args.output_manifest,manifest); print(json.dumps({"verdict":verdict,"metrics":metrics},ensure_ascii=False,indent=2))


def preflight(args) -> None:
    contract=validate(args); rows=load_groups(args); train_rows=training_rows(args,rows)
    held=sum(row["fold"]=="fold_0" for row in rows); positives=sum(len(row["positives"]) for row in train_rows); negatives=sum(len(row["negatives"]) for row in train_rows)
    fold_rows=common.fold0_rows(args); documents=common.load_documents(args,fold_rows); sequences=sum(len(common.evidence_for_row(args,row,documents)[2]) for row in fold_rows)
    manifest={"schema_version":"dsc2026.research_v2.e5_passage_alignment_input_manifest.v1","status":"PASS_INPUT_ONLY_NO_LABEL_METRIC","all_groups":len(rows),"held_queries":held,"training_queries":len(train_rows),"training_positive_parents":positives,"training_negative_parents":negatives,"evaluation_queries":len(fold_rows),"evaluation_pairs":sum(len(row["doc_ids"]) for row in fold_rows),"evaluation_sequences":sequences,"base_original_parameters":BASE_PARAMETERS,"production_original_model_total":ORIGINAL_SYSTEM_PARAMETERS,**contract}
    write_json(args.input_manifest,manifest); print(json.dumps(manifest,indent=2))


def verify(args) -> None:
    manifest=json.loads(args.output_manifest.read_text(encoding="utf-8")); failures=[]
    for name,item in manifest["files"].items():
        path=Path(item["path"]); observed=sha256(path) if path.exists() else None
        if observed!=item["sha256"]: failures.append({"file":name,"expected":item["sha256"],"observed":observed})
    db=sqlite3.connect(f"file:{args.score_db.resolve().as_posix()}?mode=ro&immutable=1",uri=True); integrity=db.execute("PRAGMA integrity_check").fetchone()[0]; progress=db.execute("SELECT COUNT(*),SUM(sequences) FROM progress").fetchone(); count=db.execute("SELECT COUNT(*) FROM scores").fetchone()[0]; db.close()
    status="PASS" if not failures and integrity=="ok" and progress[0]==EXPECTED_QUERIES and count==EXPECTED_PAIRS else "FAIL"
    result={"status":status,"manifest_sha256":sha256(args.output_manifest),"hash_failures":failures,"database":{"integrity":integrity,"queries":progress[0],"sequences":progress[1],"score_rows":count}}
    print(json.dumps(result,indent=2));
    if status!="PASS": raise RuntimeError("verification failed")


def parser():
    root=Path(__file__).resolve().parents[2]; out=root/"results/research_v2_open_rl"; cache=root/"cache/research_v2_open_rl/e5_passage_alignment_top2mean_fold0"; train_dir=out/"e5_passage_alignment_top2mean_fold0_training"
    p=argparse.ArgumentParser(); p.add_argument("stage",choices=["preflight","prepare","smoke","train","score","evaluate","verify"])
    p.add_argument("--root",type=Path,default=root)
    p.add_argument("--folds",type=Path,default=root/"results/research_v2_forensic/V2_FOLDS.json"); p.add_argument("--pool",type=Path,default=root/"results/research_v2_forensic/V2_CANDIDATE_POOL.jsonl")
    p.add_argument("--groups",type=Path,default=root/"results/research_v2_forensic/V2_BOUNDARY_GROUPS.jsonl"); p.add_argument("--groups-manifest",type=Path,default=root/"results/research_v2_forensic/V2_BOUNDARY_GROUPS_MANIFEST.json")
    p.add_argument("--model",type=Path,default=root/"cache/research_v2_e5_confirmation/bundle-v1/vietlegal-e5"); p.add_argument("--query-adapter",type=Path,default=root/"results/research_v2_e5_transfer/research_v2_e5_transfer_fold0/training/epoch-2.pt"); p.add_argument("--query-core",type=Path,default=root/"src/research_v2_e5_transfer/e5_transfer_runner.py")
    p.add_argument("--contexts",type=Path,default=root/"cache/research_v2_forensic/kaggle_input/research-v2-jina-boundary-v4/V2_CONTEXTS.jsonl"); p.add_argument("--renderer",type=Path,default=root/"benchmark_jina_reranker_holdouts.py")
    p.add_argument("--anchor",type=Path,default=root/"results/research_v2_post_e5/V2_ADAPTED_E5_LAL_EQUAL_RRF32_PREDICTIONS.jsonl"); p.add_argument("--e5-predictions",type=Path,default=root/"results/research_v2_e5_confirmation/fold0_runner_parity/E5_CONFIRMATION_FOLD_0_PREDICTIONS.jsonl"); p.add_argument("--jina-predictions",type=Path,default=root/"results/research_v2_forensic/V2_ZERO_SHOT_LEXICAL_PREDICTIONS.jsonl"); p.add_argument("--sources-db",type=Path,default=root.parent/"LegalIR/cache/exp112_task_adaptive_retrieval/sources.sqlite")
    p.add_argument("--preregistration",type=Path,default=out/"E5_PASSAGE_ALIGNMENT_TOP2MEAN_FOLD0_PREREGISTRATION.json"); p.add_argument("--input-manifest",type=Path,default=out/"E5_PASSAGE_ALIGNMENT_TOP2MEAN_FOLD0_INPUT_MANIFEST.json"); p.add_argument("--query-vectors",type=Path,default=cache/"adapted_query_vectors.f32.npy"); p.add_argument("--query-ids",type=Path,default=cache/"query_ids.json"); p.add_argument("--query-cache-manifest",type=Path,default=out/"E5_PASSAGE_ALIGNMENT_TOP2MEAN_FOLD0_QUERY_CACHE_MANIFEST.json"); p.add_argument("--smoke-report",type=Path,default=out/"E5_PASSAGE_ALIGNMENT_TOP2MEAN_FOLD0_SMOKE.json")
    p.add_argument("--training-dir",type=Path,default=train_dir); p.add_argument("--training-success",type=Path,default=train_dir/"_SUCCESS.json"); p.add_argument("--resume",type=Path,default=train_dir/"resume.pt"); p.add_argument("--checkpoint",type=Path,default=train_dir/"epoch-2.pt")
    p.add_argument("--score-db",type=Path,default=cache/"scores.sqlite"); p.add_argument("--report",type=Path,default=out/"E5_PASSAGE_ALIGNMENT_TOP2MEAN_FOLD0_REPORT.json"); p.add_argument("--predictions",type=Path,default=out/"E5_PASSAGE_ALIGNMENT_TOP2MEAN_FOLD0_PREDICTIONS.jsonl"); p.add_argument("--prediction-lock",type=Path,default=out/"E5_PASSAGE_ALIGNMENT_TOP2MEAN_FOLD0_PREDICTION_LOCK.json"); p.add_argument("--output-manifest",type=Path,default=out/"E5_PASSAGE_ALIGNMENT_TOP2MEAN_FOLD0_OUTPUT_MANIFEST.json")
    return p


def main():
    args=parser().parse_args(); args.input_manifest.parent.mkdir(parents=True,exist_ok=True); args.score_db.parent.mkdir(parents=True,exist_ok=True)
    if args.stage=="preflight": preflight(args)
    elif args.stage=="prepare": prepare_query_cache(args)
    elif args.stage=="smoke": smoke(args)
    elif args.stage=="train": train(args)
    elif args.stage=="score": score(args)
    elif args.stage=="evaluate": evaluate(args)
    elif args.stage=="verify": verify(args)


if __name__ == "__main__": main()
