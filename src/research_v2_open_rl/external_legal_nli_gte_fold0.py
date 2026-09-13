"""Sealed external-legal-NLI GTE Fold-0 pilot for DSC2026 Research V2.

Stages are deliberately linear: train -> parity -> score -> evaluate.  Training
uses only a removal-only decontaminated public dataset.  Parity and scoring do
not read labels.  Evaluation is fail-closed and refuses to overwrite a report.
"""

from __future__ import annotations

import argparse
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


EXPECTED = {
    "folds": "94ad5c6d5e582ced5eec8d2c3c15f938454c17e713614391091e72abea9aba19",
    "pool": "96a44e66549cc211e1f9d0fabb84fc825db3f21f32d5b349eeca3b1c0413e277",
    "training_data": "10c6b6c55fda966f3599ea6d640871a0fdb12538b53eb53b37171b2787f9a3a3",
    "decontamination_audit": "60361ead7f61d75b083b9462d705382bd86ab66dabd752243e55170635318f2a",
    "preregistration": "55a5cba648142df6c803ee88db3c4c88e06a1b61288434677313b6815ed81a91",
    "base_model": "10ebaa49322dd7e01a13a91c49810939e3f91f231aceaa47fdf0cab3083954f6",
    "anchor": "1854494964f2258243bc00896c76b11d56a3c02752a23af0ee04bac8e260de4d",
    "e5_predictions": "801d25d56e1f8334eea546b8acedbe3035f90a29b2d441a24bece2821b6722f8",
    "jina_predictions": "74587e93df5cfaf486a334c8694ed1df3b7c9c36fcac5ed79e4840be478433b2",
    "sources_db": "ef763bf8c5e3da91fb6447f8a03ab321fdaca5362bfde0213057a15311824f1e",
    "contexts": "55c77371edd3b4f28e3e8ca548447e27424e57ce219e3d6da7e9f51238a22291",
}
SEED = 20260913
PARAMETER_TOTAL = 3_703_755_521
PARAMETER_LIMIT = 4_000_000_000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8", newline="\n",
    )
    os.replace(temporary, path)


def require_hash(name: str, path: Path) -> str:
    observed = sha256(path)
    if observed != EXPECTED[name]:
        raise RuntimeError(f"{name} hash mismatch: {observed} != {EXPECTED[name]}")
    return observed


def validate_contract(args: argparse.Namespace, include_eval_inputs: bool = False) -> dict:
    paths = {
        "folds": args.folds,
        "pool": args.pool,
        "training_data": args.training_data,
        "decontamination_audit": args.decontamination_audit,
        "preregistration": args.preregistration,
        "base_model": args.model / "model.safetensors",
        "contexts": args.contexts_jsonl,
    }
    if include_eval_inputs:
        paths.update({
            "anchor": args.anchor,
            "e5_predictions": args.e5_predictions,
            "jina_predictions": args.jina_predictions,
            "sources_db": args.sources_db,
        })
    hashes = {name: require_hash(name, path) for name, path in paths.items()}
    prereg = json.loads(args.preregistration.read_text(encoding="utf-8"))
    audit = json.loads(args.decontamination_audit.read_text(encoding="utf-8"))
    checks = {
        "preregistered_before_training": prereg.get("status") == "SEALED_BEFORE_TRAINING_OR_V2_METRIC",
        "no_augmentation": prereg["external_training"].get("augmentation") is False,
        "rows_added_zero": audit.get("result", {}).get("rows_added") == 0,
        "rows_modified_zero": audit.get("result", {}).get("rows_modified") == 0,
        "removal_only_status": audit.get("status") == "PASS_REMOVAL_ONLY_NO_AUGMENTATION",
        "training_rows": audit.get("output", {}).get("rows") == 31930,
        "parameter_budget": PARAMETER_TOTAL < PARAMETER_LIMIT and prereg["parameter_budget"].get("pass") is True,
        "adaptive_k_disabled": prereg["inference"].get("adaptive_k") is False,
    }
    if not all(checks.values()):
        raise RuntimeError(f"sealed contract failure: {checks}")
    return {"hashes": hashes, "checks": checks}


def set_seed() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def patch_position_ids(model) -> None:
    embeddings = model.new.embeddings
    length = int(model.config.max_position_embeddings)
    embeddings.register_buffer(
        "position_ids", torch.arange(length, dtype=torch.long).expand((1, -1)), persistent=False
    )


def load_base(args: argparse.Namespace, dtype: torch.dtype):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=True, trust_remote_code=True, use_fast=True
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, local_files_only=True, trust_remote_code=True, dtype=dtype
    )
    patch_position_ids(model)
    return model, tokenizer


def encode_pairs(tokenizer, pairs: list[list[str]], max_length: int = 512,
                 fixed_padding: bool = False) -> dict[str, torch.Tensor]:
    batch = tokenizer(
        pairs, padding="max_length" if fixed_padding else True, truncation=True,
        max_length=max_length, return_tensors="pt"
    )
    batch.pop("token_type_ids", None)
    return batch


def training_fingerprint(args: argparse.Namespace) -> str:
    return hashlib.sha256(canonical_json({
        "preregistration": EXPECTED["preregistration"],
        "training_data": EXPECTED["training_data"],
        "base_model": EXPECTED["base_model"],
        "seed": SEED, "epochs": 1, "batch_groups": 32, "max_length": 512,
        "learning_rate": 1e-4, "lora": [8, 16, 0.05, ["qkv_proj", "o_proj"]],
    }).encode()).hexdigest()


def save_checkpoint(path: Path, model, optimizer, scaler, next_batch: int, order_hash: str, fingerprint: str) -> None:
    state = {
        "fingerprint": fingerprint, "next_batch": next_batch, "order_sha256": order_hash,
        "trainable": {name: parameter.detach().cpu() for name, parameter in model.named_parameters() if parameter.requires_grad},
        "optimizer": optimizer.state_dict(),
        "grad_scaler": scaler.state_dict(),
    }
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def train(args: argparse.Namespace) -> None:
    contract = validate_contract(args)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    success = args.run_dir / "_SUCCESS.json"
    if success.exists():
        print(success.read_text(encoding="utf-8"), flush=True)
        return
    rows = list(read_jsonl(args.training_data))
    if len(rows) != 31930 or any(set(row) != {"query", "positive", "hard_neg"} for row in rows):
        raise RuntimeError("external training row contract failure")
    order = list(range(len(rows)))
    random.Random(SEED).shuffle(order)
    order_hash = hashlib.sha256(("\n".join(map(str, order)) + "\n").encode()).hexdigest()
    set_seed()
    base, tokenizer = load_base(args, torch.float16)
    base.gradient_checkpointing_enable()
    base.enable_input_require_grads()
    from peft import LoraConfig, TaskType, get_peft_model
    model = get_peft_model(base, LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.05, bias="none",
        target_modules=["qkv_proj", "o_proj"], task_type=TaskType.SEQ_CLS,
    )).to("cuda")
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.float()
    model.train()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if trainable != 443137:
        raise RuntimeError(f"trainable parameter mismatch: {trainable}")
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    checkpoint = args.run_dir / "resume_state.pt"
    start_batch = 0
    fingerprint = training_fingerprint(args)
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if saved["fingerprint"] != fingerprint or saved["order_sha256"] != order_hash:
            raise RuntimeError("resume checkpoint contract mismatch")
        named = dict(model.named_parameters())
        if set(saved["trainable"]) != {n for n, p in named.items() if p.requires_grad}:
            raise RuntimeError("resume trainable-name mismatch")
        for name, value in saved["trainable"].items():
            named[name].data.copy_(value.to(named[name].device))
        optimizer.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved["grad_scaler"])
        start_batch = int(saved["next_batch"])
        print(f"resume_batch={start_batch}", flush=True)
    batch_groups = 32
    batches = math.ceil(len(order) / batch_groups)
    losses: list[float] = []
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for batch_index in range(start_batch, batches):
        indices = order[batch_index * batch_groups:(batch_index + 1) * batch_groups]
        pairs: list[list[str]] = []
        for index in indices:
            row = rows[index]
            pairs.extend([[row["query"], row["positive"]], [row["query"], row["hard_neg"]]])
        encoded = {key: value.to("cuda") for key, value in encode_pairs(tokenizer, pairs).items()}
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(**encoded, return_dict=True).logits.view(-1).float()
            positive, negative = logits[0::2], logits[1::2]
            loss = F.softplus(-(positive - negative)).mean()
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at batch {batch_index}")
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        losses.append(float(loss.detach().cpu()))
        completed = batch_index + 1
        if completed % 50 == 0 or completed == batches:
            elapsed = time.perf_counter() - started
            rate = (completed - start_batch) / elapsed
            eta = (batches - completed) / rate / 60 if rate else math.inf
            print(f"train={completed}/{batches} loss50={np.mean(losses[-50:]):.6f} eta_min={eta:.1f}", flush=True)
        if completed % 100 == 0 and completed < batches:
            save_checkpoint(checkpoint, model, optimizer, scaler, completed, order_hash, fingerprint)
    elapsed = time.perf_counter() - started
    adapter_dir = args.run_dir / "adapter"
    model.save_pretrained(adapter_dir, safe_serialization=True)
    tokenizer.save_pretrained(adapter_dir)
    adapter_files = {p.name: sha256(p) for p in sorted(adapter_dir.iterdir()) if p.is_file()}
    report = {
        "schema_version": "dsc2026.research_v2.external_legal_nli_gte_training.v1",
        "status": "COMPLETE", "rows": len(rows), "epochs": 1, "batches": batches,
        "start_batch_this_process": start_batch, "seed": SEED, "order_sha256": order_hash,
        "fingerprint": fingerprint, "batch_groups": batch_groups, "max_length": 512,
        "optimizer": "AdamW", "learning_rate": 1e-4,
        "numerical_compatibility": "FP16 backbone/autocast, FP32 trainable adapter parameters, dynamic GradScaler",
        "loss": "softplus(-(positive_logit-hard_negative_logit))",
        "loss_first_observed": losses[0] if losses else None,
        "loss_last_50_mean": float(np.mean(losses[-50:])) if losses else None,
        "loss_process_mean": float(np.mean(losses)) if losses else None,
        "seconds_this_process": elapsed,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "trainable_parameters": trainable, "base_parameters": 305959681,
        "deployed_original_parameter_total": PARAMETER_TOTAL,
        "parameter_limit": PARAMETER_LIMIT, "parameter_budget_pass": PARAMETER_TOTAL < PARAMETER_LIMIT,
        "augmentation": False, "rows_added": 0, "rows_modified": 0,
        "adapter_files": adapter_files, "contract": contract,
    }
    report_path = args.run_dir / "TRAINING_REPORT.json"
    write_json(report_path, report)
    write_json(success, {"status": "COMPLETE", "report_sha256": sha256(report_path), "adapter_files": adapter_files})
    print(json.dumps({"status": "COMPLETE", "loss_last_50_mean": report["loss_last_50_mean"],
                      "seconds": elapsed, "peak_mib": report["peak_allocated_mib"]}, indent=2), flush=True)


def numerical_smoke(args: argparse.Namespace) -> None:
    """Three external-only optimizer steps; reads no V2 labels or metric."""
    validate_contract(args)
    rows = list(read_jsonl(args.training_data))[:96]
    set_seed()
    base, tokenizer = load_base(args, torch.float16)
    base.gradient_checkpointing_enable()
    base.enable_input_require_grads()
    from peft import LoraConfig, TaskType, get_peft_model
    model = get_peft_model(base, LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.05, bias="none",
        target_modules=["qkv_proj", "o_proj"], task_type=TaskType.SEQ_CLS,
    )).to("cuda").train()
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.float()
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    losses = []
    for batch_index in range(3):
        group = rows[batch_index * 32:(batch_index + 1) * 32]
        pairs = [[text, doc] for row in group for text, doc in (
            (row["query"], row["positive"]), (row["query"], row["hard_neg"])
        )]
        encoded = {key: value.to("cuda") for key, value in encode_pairs(tokenizer, pairs).items()}
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(**encoded, return_dict=True).logits.view(-1).float()
            loss = F.softplus(-(logits[0::2] - logits[1::2])).mean()
        if not torch.isfinite(loss):
            raise RuntimeError(f"smoke non-finite loss at step {batch_index}")
        scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        if not all(torch.isfinite(parameter).all() for parameter in model.parameters() if parameter.requires_grad):
            raise RuntimeError(f"smoke non-finite parameter at step {batch_index}")
        losses.append(float(loss.detach().cpu()))
    report = {
        "schema_version": "dsc2026.research_v2.external_legal_nli_gte_numerical_smoke.v1",
        "status": "PASS", "v2_labels_read": False, "steps": 3, "groups": 96,
        "losses": losses, "grad_scaler": True, "final_scale": scaler.get_scale(),
        "scientific_contract_changed": False,
    }
    write_json(args.runtime_smoke_report, report)
    print(json.dumps(report, indent=2), flush=True)


def fold0_rows(args: argparse.Namespace) -> list[dict]:
    folds = json.loads(args.folds.read_text(encoding="utf-8"))
    fold0 = set(map(str, folds["folds"]["fold_0"]))
    rows = [row for row in read_jsonl(args.pool) if str(row["qid"]) in fold0]
    if len(rows) != 1398 or any(str(row["fold"]) != "fold_0" for row in rows):
        raise RuntimeError(f"Fold0 mismatch: {len(rows)}")
    return rows


def build_inference_inputs(args: argparse.Namespace, rows: list[dict]):
    sys.path.insert(0, str(args.root))
    from benchmark_jina_reranker_holdouts import top_passages
    needed = {str(doc) for row in rows for doc in row["doc_ids"]}
    documents = {}
    for item in read_jsonl(args.contexts_jsonl):
        doc = str(item["doc_id"])
        if doc in needed:
            documents[doc] = str(item["passage"])
    if set(documents) != needed:
        raise RuntimeError(f"materialized context coverage mismatch: {len(documents)}/{len(needed)}")
    for row in rows:
        query = str(row["query"])
        docs = list(map(str, row["doc_ids"]))
        passages = []
        for doc in docs:
            selected = list(top_passages(query, documents[doc], count=1))
            if len(selected) != 1:
                raise RuntimeError(f"evidence count mismatch {row['qid']} {doc}: {len(selected)}")
            passages.append(str(selected[0]))
        yield row, docs, [[query, passage] for passage in passages]


def load_expert(args: argparse.Namespace, dtype: torch.dtype = torch.float32):
    from peft import PeftModel
    base, tokenizer = load_base(args, dtype)
    model = PeftModel.from_pretrained(base, args.run_dir / "adapter", is_trainable=False).eval().to("cuda")
    return model, tokenizer


@torch.inference_mode()
def score_pairs(model, tokenizer, pairs: list[list[str]], batch_size: int) -> list[float]:
    values: list[float] = []
    for start in range(0, len(pairs), batch_size):
        encoded = {key: value.to("cuda") for key, value in encode_pairs(
            tokenizer, pairs[start:start + batch_size], fixed_padding=True
        ).items()}
        dtype = next(model.parameters()).dtype
        with torch.autocast("cuda", dtype=torch.float16, enabled=dtype == torch.float16):
            logits = model(**encoded, return_dict=True).logits.view(-1).float()
        if not torch.isfinite(logits).all():
            raise RuntimeError("non-finite inference logit")
        values.extend(float(x) for x in logits.cpu())
    return values


def parity(args: argparse.Namespace) -> None:
    validate_contract(args)
    if not (args.run_dir / "_SUCCESS.json").exists():
        raise RuntimeError("training incomplete")
    rows = sorted(fold0_rows(args), key=lambda row: hashlib.sha256(str(row["qid"]).encode()).hexdigest())[:2]
    rendered = list(build_inference_inputs(args, rows))
    pairs = [pair for _, _, group in rendered for pair in group]
    model, tokenizer = load_expert(args, torch.float32)
    reference = score_pairs(model, tokenizer, pairs, 1)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    candidate = score_pairs(model, tokenizer, pairs, 32)
    elapsed = time.perf_counter() - started
    errors = np.abs(np.asarray(reference) - np.asarray(candidate))
    cursor = 0
    top5_equal = True
    for _, docs, group in rendered:
        n = len(group)
        rank1 = sorted(docs, key=lambda doc: (-reference[cursor + docs.index(doc)], doc))
        rank32 = sorted(docs, key=lambda doc: (-candidate[cursor + docs.index(doc)], doc))
        top5_equal &= rank1[:5] == rank32[:5]
        cursor += n
    status = "PASS" if top5_equal and float(errors.max()) <= 1e-4 else "FAIL"
    report = {
        "schema_version": "dsc2026.research_v2.external_legal_nli_gte_inference_parity.v1",
        "status": status, "labels_read": False, "queries": len(rows), "sequences": len(pairs),
        "reference_batch": 1, "authorized_batch": 32, "dtype": "float32",
        "max_abs_error": float(errors.max()), "mean_abs_error": float(errors.mean()),
        "top5_identical": top5_equal, "seconds_batch32": elapsed,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
    }
    write_json(args.parity_report, report)
    print(json.dumps(report, indent=2), flush=True)
    if status != "PASS":
        raise RuntimeError("inference batch parity failed")


def open_db(path: Path) -> sqlite3.Connection:
    db = sqlite3.connect(path)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("CREATE TABLE IF NOT EXISTS scores(qid TEXT,doc_id TEXT,score REAL,PRIMARY KEY(qid,doc_id))")
    db.execute("CREATE TABLE IF NOT EXISTS progress(qid TEXT PRIMARY KEY,seconds REAL,sequences INTEGER,peak_mib REAL)")
    db.execute("CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT)")
    db.commit()
    return db


def score(args: argparse.Namespace) -> None:
    validate_contract(args)
    parity_report = json.loads(args.parity_report.read_text(encoding="utf-8"))
    if parity_report.get("status") != "PASS" or parity_report.get("authorized_batch") != 32:
        raise RuntimeError("inference parity gate closed")
    rows = fold0_rows(args)
    db = open_db(args.score_db)
    fingerprint = training_fingerprint(args)
    stored = dict(db.execute("SELECT key,value FROM metadata"))
    expected_meta = {"fingerprint": fingerprint, "pool_sha256": EXPECTED["pool"], "folds_sha256": EXPECTED["folds"]}
    if stored and stored != expected_meta:
        raise RuntimeError(f"score cache metadata mismatch: {stored}")
    if not stored:
        with db:
            db.executemany("INSERT INTO metadata VALUES(?,?)", expected_meta.items())
    complete = {str(row[0]) for row in db.execute("SELECT qid FROM progress")}
    model, tokenizer = load_expert(args, torch.float32)
    process_started = time.perf_counter()
    newly_done = 0
    for row, docs, pairs in build_inference_inputs(args, rows):
        qid = str(row["qid"])
        if qid in complete:
            continue
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        values = score_pairs(model, tokenizer, pairs, 32)
        elapsed = time.perf_counter() - started
        with db:
            db.executemany("INSERT INTO scores VALUES(?,?,?)", [(qid, doc, value) for doc, value in zip(docs, values)])
            db.execute("INSERT INTO progress VALUES(?,?,?,?)", (qid, elapsed, len(values), torch.cuda.max_memory_allocated() / 2**20))
        newly_done += 1
        if newly_done % 10 == 0:
            done = len(complete) + newly_done
            qps = newly_done / (time.perf_counter() - process_started)
            print(f"score={done}/1398 qps={qps:.3f} eta_min={(1398-done)/qps/60:.1f}", flush=True)
    stats = db.execute("SELECT COUNT(*),COUNT(DISTINCT qid),SUM(sequences),SUM(seconds),MAX(peak_mib) FROM progress").fetchone()
    score_rows = db.execute("SELECT COUNT(*) FROM scores").fetchone()[0]
    integrity = db.execute("PRAGMA integrity_check").fetchone()[0]
    db.close()
    print(json.dumps({"progress": stats, "score_rows": score_rows, "integrity": integrity}, indent=2), flush=True)


def recall(docs, gold) -> float:
    return len(set(docs) & set(gold)) / len(gold)


def load_clean_sets(args: argparse.Namespace, pool: dict[str, set[str]]) -> dict[str, set[str]]:
    e5 = {str(row["qid"]): (
              set(map(str, row["ft_order"][:5])), set(map(str, row["base_order"][:5]))
          ) for row in read_jsonl(args.e5_predictions) if str(row["qid"]) in pool}
    jina_v2 = {str(row["qid"]): set(map(str, row["top5"]))
               for row in read_jsonl(args.jina_predictions) if str(row["qid"]) in pool}
    db = sqlite3.connect(f"file:{args.sources_db.resolve().as_posix()}?mode=ro&immutable=1", uri=True)
    native = {}
    for qid, source, payload in db.execute("SELECT q,source,payload FROM sources WHERE source IN ('lal','jina')"):
        qid = str(qid)
        if qid in pool:
            native[(qid, str(source))] = [str(item["doc_id"]) for item in json.loads(payload)
                                          if str(item["doc_id"]) in pool[qid]][:5]
    db.close()
    clean = {}
    for qid in pool:
        adapted_e5, frozen_e5 = e5[qid]
        components = [adapted_e5, frozen_e5, jina_v2[qid],
                      set(native[(qid, "lal")]), set(native[(qid, "jina")])]
        if any(len(component) != 5 or not component <= pool[qid] for component in components):
            raise RuntimeError(f"clean expert contract failure: {qid}")
        clean[qid] = set().union(*components)
    return clean


def evaluate(args: argparse.Namespace) -> None:
    if args.report.exists():
        raise RuntimeError(f"refusing to re-evaluate sealed pilot: {args.report}")
    contract = validate_contract(args, include_eval_inputs=True)
    rows = fold0_rows(args)
    pool = {str(row["qid"]): set(map(str, row["doc_ids"])) for row in rows}
    anchor = {str(row["qid"]): row for row in read_jsonl(args.anchor) if str(row["qid"]) in pool}
    if set(anchor) != set(pool):
        raise RuntimeError("anchor qid mismatch")
    clean = load_clean_sets(args, pool)
    db = sqlite3.connect(f"file:{args.score_db.resolve().as_posix()}?mode=ro&immutable=1", uri=True)
    integrity = db.execute("PRAGMA integrity_check").fetchone()[0]
    progress = db.execute("SELECT COUNT(*),COUNT(DISTINCT qid),SUM(sequences),SUM(seconds),MAX(peak_mib) FROM progress").fetchone()
    scores = {(str(q), str(d)): float(s) for q, d, s in db.execute("SELECT qid,doc_id,score FROM scores")}
    metadata = dict(db.execute("SELECT key,value FROM metadata"))
    db.close()
    expected_pairs = {(qid, doc) for qid, docs in pool.items() for doc in docs}
    integrity_ok = integrity == "ok" and progress[0] == progress[1] == 1398 and progress[2] == len(scores) == len(expected_pairs) and set(scores) == expected_pairs
    metadata_ok = metadata.get("fingerprint") == training_fingerprint(args)
    if not integrity_ok or not metadata_ok:
        raise RuntimeError(f"score cache incomplete/inconsistent: integrity={integrity_ok} metadata={metadata_ok}")
    expert_r = []; expert_p = []; anchor_r = []; clean_r = []; clean_plus = []; candidate = []
    single = [[], []]; multi = [[], []]; depth = {k: [] for k in (5, 10, 20, 50)}
    wins = losses = churn = crossings_in = crossings_out = 0
    predictions = []; gold_buckets = Counter()
    for row in rows:
        qid = str(row["qid"]); docs = list(map(str, row["doc_ids"])); gold = set(map(str, anchor[qid]["gold"]))
        ordered = sorted(docs, key=lambda doc: (-scores[(qid, doc)], doc)); top5 = ordered[:5]
        base = list(map(str, anchor[qid]["fused_top5"])); er = recall(top5, gold); ar = recall(base, gold)
        expert_r.append(er); expert_p.append(len(set(top5) & gold) / 5); anchor_r.append(ar)
        clean_r.append(recall(clean[qid], gold)); clean_plus.append(recall(clean[qid] | set(top5), gold)); candidate.append(recall(pool[qid], gold))
        target = single if len(gold) == 1 else multi; target[0].append(er); target[1].append(ar)
        wins += er > ar; losses += er < ar; churn += top5 != base
        crossings_in += len((set(top5) - set(base)) & gold); crossings_out += len((set(base) - set(top5)) & gold)
        for k in depth: depth[k].append(recall(ordered[:k], gold))
        for doc in gold:
            rank = ordered.index(doc) + 1 if doc in pool[qid] else None
            bucket = "missing" if rank is None else "1-5" if rank <= 5 else "6-10" if rank <= 10 else "11-20" if rank <= 20 else "21-50" if rank <= 50 else "51+"
            gold_buckets[bucket] += 1
        predictions.append({"qid": qid, "top5": top5, "ranking": ordered, "scores": [scores[(qid, doc)] for doc in ordered]})
    metrics = {
        "recall_at_5": float(np.mean(expert_r)), "precision_at_5": float(np.mean(expert_p)),
        "current_anchor_recall_at_5": float(np.mean(anchor_r)),
        "existing_clean_experts_union": float(np.mean(clean_r)),
        "clean_experts_plus_gte_union": float(np.mean(clean_plus)),
        "clean_union_delta": float(np.mean(clean_plus) - np.mean(clean_r)),
        "candidate_ceiling": float(np.mean(candidate)),
        "single_gold": {"queries": len(single[0]), "expert_recall_at_5": float(np.mean(single[0])), "anchor_recall_at_5": float(np.mean(single[1]))},
        "multi_gold": {"queries": len(multi[0]), "expert_recall_at_5": float(np.mean(multi[0])), "anchor_recall_at_5": float(np.mean(multi[1]))},
        "per_fold": {"fold_0": {"queries": len(rows), "expert_recall_at_5": float(np.mean(expert_r)), "anchor_recall_at_5": float(np.mean(anchor_r)), "delta": float(np.mean(expert_r) - np.mean(anchor_r))}},
        "wins_losses_ties_vs_anchor": {"wins": wins, "losses": losses, "ties": len(rows) - wins - losses},
        "gold_crossings": {"into_top5": crossings_in, "out_of_top5": crossings_out},
        "top5_churn_queries": churn, "recall_depth": {str(k): float(np.mean(value)) for k, value in depth.items()},
        "gold_rank_buckets": dict(gold_buckets),
    }
    pass_standalone = metrics["recall_at_5"] >= 0.90 and metrics["clean_union_delta"] >= 0.006
    pass_orthogonal = metrics["recall_at_5"] >= 0.84 and metrics["clean_union_delta"] >= 0.005
    kill = metrics["recall_at_5"] < 0.80 or metrics["clean_union_delta"] < 0.003 or not all(contract["checks"].values()) or not integrity_ok
    verdict = "KILL" if kill else "PASS_STANDALONE" if pass_standalone else "PASS_ORTHOGONAL" if pass_orthogonal else "INCONCLUSIVE_NO_TUNING"
    with args.predictions.open("x", encoding="utf-8", newline="\n") as stream:
        for row in predictions:
            stream.write(canonical_json(row) + "\n")
    report = {
        "schema_version": "dsc2026.research_v2.external_legal_nli_gte_fold0_report.v1",
        "status": "COMPLETE", "verdict": verdict, "metrics": metrics,
        "gate": {"pass_standalone": pass_standalone, "pass_orthogonal": pass_orthogonal,
                 "kill_recall_lt_0_80": metrics["recall_at_5"] < 0.80,
                 "kill_clean_delta_lt_0_003": metrics["clean_union_delta"] < 0.003,
                 "integrity": integrity_ok, "contract": all(contract["checks"].values())},
        "runtime": {"seconds": progress[3], "sequences": progress[2], "peak_mib": progress[4]},
        "hashes": {**contract["hashes"], "score_db": sha256(args.score_db),
                   "training_report": sha256(args.run_dir / "TRAINING_REPORT.json"),
                   "predictions": sha256(args.predictions)},
        "fixed_interface": {"evidence_count": 1, "max_length": 512, "aggregation": "identity", "ranking": "score_desc_doc_id_string_asc_top5", "adaptive_k": False},
        "anti_rescue": "No training, external-subset, evidence, model, score, fusion, threshold, or adaptive-K grid.",
    }
    write_json(args.report, report)
    top5_rows = [canonical_json({"qid": row["qid"], "top5": row["top5"]}) for row in predictions]
    lock = {"schema_version": "dsc2026.research_v2.external_legal_nli_gte_prediction_lock.v1",
            "status": "LOCKED", "verdict": verdict, "queries": len(predictions),
            "predictions_sha256": sha256(args.predictions),
            "canonical_qid_top5_sha256": hashlib.sha256(("\n".join(top5_rows) + "\n").encode()).hexdigest()}
    write_json(args.prediction_lock, lock)
    output_files = [args.report, args.predictions, args.prediction_lock, args.parity_report,
                    args.run_dir / "TRAINING_REPORT.json", args.run_dir / "_SUCCESS.json"]
    write_json(args.output_manifest, {
        "schema_version": "dsc2026.research_v2.external_legal_nli_gte_output_manifest.v1",
        "status": "COMPLETE", "verdict": verdict,
        "files": {path.name: {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size} for path in output_files},
        "reproduce": f'"{sys.executable}" "{Path(__file__).resolve()}" all',
    })
    print(json.dumps({"verdict": verdict, "metrics": metrics}, ensure_ascii=False, indent=2), flush=True)


def parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[2]
    workspace = root.parent
    output = root / "results/research_v2_open_rl"
    cache = root / "cache/research_v2_open_rl/external_legal_nli_gte_fold0"
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["smoke", "train", "parity", "score", "evaluate", "all"])
    p.add_argument("--root", type=Path, default=root)
    p.add_argument("--model", type=Path, default=Path(r"C:\Users\nguye\.cache\huggingface\hub\models--Alibaba-NLP--gte-multilingual-reranker-base\snapshots\8215cf04918ba6f7b6a62bb44238ce2953d8831c"))
    p.add_argument("--folds", type=Path, default=root / "results/research_v2_forensic/V2_FOLDS.json")
    p.add_argument("--pool", type=Path, default=root / "results/research_v2_forensic/V2_CANDIDATE_POOL.jsonl")
    p.add_argument("--training-data", type=Path, default=root / "cache/research_v2_open_rl/external_audits/vinli_zalo/law_vi_decontaminated.jsonl")
    p.add_argument("--decontamination-audit", type=Path, default=output / "EXTERNAL_LEGAL_NLI_DECONTAMINATION_AUDIT.json")
    p.add_argument("--preregistration", type=Path, default=output / "EXTERNAL_LEGAL_NLI_GTE_FOLD0_PREREGISTRATION.json")
    p.add_argument("--contexts-jsonl", type=Path, default=root / "cache/research_v2_forensic/kaggle_input/research-v2-jina-boundary-v4/V2_CONTEXTS.jsonl")
    p.add_argument("--anchor", type=Path, default=root / "results/research_v2_post_e5/V2_ADAPTED_E5_LAL_EQUAL_RRF32_PREDICTIONS.jsonl")
    p.add_argument("--e5-predictions", type=Path, default=root / "results/research_v2_e5_confirmation/fold0_runner_parity/E5_CONFIRMATION_FOLD_0_PREDICTIONS.jsonl")
    p.add_argument("--jina-predictions", type=Path, default=root / "results/research_v2_forensic/V2_ZERO_SHOT_LEXICAL_PREDICTIONS.jsonl")
    p.add_argument("--sources-db", type=Path, default=workspace / "LegalIR/cache/exp112_task_adaptive_retrieval/sources.sqlite")
    p.add_argument("--run-dir", type=Path, default=cache)
    p.add_argument("--score-db", type=Path, default=cache / "scores.sqlite")
    p.add_argument("--parity-report", type=Path, default=output / "EXTERNAL_LEGAL_NLI_GTE_FOLD0_INFERENCE_PARITY_V3.json")
    p.add_argument("--runtime-smoke-report", type=Path, default=output / "EXTERNAL_LEGAL_NLI_GTE_NUMERICAL_SMOKE.json")
    p.add_argument("--report", type=Path, default=output / "EXTERNAL_LEGAL_NLI_GTE_FOLD0_REPORT.json")
    p.add_argument("--predictions", type=Path, default=output / "EXTERNAL_LEGAL_NLI_GTE_FOLD0_PREDICTIONS.jsonl")
    p.add_argument("--prediction-lock", type=Path, default=output / "EXTERNAL_LEGAL_NLI_GTE_FOLD0_PREDICTION_LOCK.json")
    p.add_argument("--output-manifest", type=Path, default=output / "EXTERNAL_LEGAL_NLI_GTE_FOLD0_OUTPUT_MANIFEST.json")
    return p


def main() -> None:
    args = parser().parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    if args.stage == "smoke": numerical_smoke(args)
    if args.stage in {"train", "all"}: train(args)
    if args.stage in {"parity", "all"}: parity(args)
    if args.stage in {"score", "all"}: score(args)
    if args.stage in {"evaluate", "all"}: evaluate(args)


if __name__ == "__main__":
    main()
