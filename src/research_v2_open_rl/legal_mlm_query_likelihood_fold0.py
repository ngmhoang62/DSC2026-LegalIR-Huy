"""Frozen legal-MLM document-to-query likelihood for Research V2 Fold 0.

The sealed interface masks every query token, conditions on one of the locked
lexical passages, and ranks parents by the maximum mean query-token log
likelihood.  Scoring is resumable and label-free.  Evaluation is one-shot.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
import platform
import sqlite3
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch


EXPECTED = {
    "folds": "94ad5c6d5e582ced5eec8d2c3c15f938454c17e713614391091e72abea9aba19",
    "pool": "96a44e66549cc211e1f9d0fabb84fc825db3f21f32d5b349eeca3b1c0413e277",
    "contexts": "55c77371edd3b4f28e3e8ca548447e27424e57ce219e3d6da7e9f51238a22291",
    "anchor": "1854494964f2258243bc00896c76b11d56a3c02752a23af0ee04bac8e260de4d",
    "e5_predictions": "801d25d56e1f8334eea546b8acedbe3035f90a29b2d441a24bece2821b6722f8",
    "jina_predictions": "74587e93df5cfaf486a334c8694ed1df3b7c9c36fcac5ed79e4840be478433b2",
    "sources_db": "ef763bf8c5e3da91fb6447f8a03ab321fdaca5362bfde0213057a15311824f1e",
    "selector": "1f09377b9a329f68b0efcf254b6a493790e5e8b1b903ed03b035796c1c82917a",
    "preregistration": "fb6cc44f6a275481e969bf109c0e740580bb4ffa180e32dfcbbc57fc76eae9fb",
    "model": "0787d99187a15bfae5d0104eeed6aad0cb296d92be8956681312ceb7365ada1f",
    "uts_dictionary": "ec8c62b3881c6682c18959988e1fe08800cc835777778845bded470ca02745d9",
    "uts_features": "cf93263c7c7973bf50122b0be76a394443ff61be00c1368713dee1b7d5b8a435",
    "uts_model": "c3a64c64349f1d73304f89babacaa8f8cd6078b1ec1a1b2c104e9e411ac1d2c3",
}
MODEL_PARAMETERS = 135_063_809
EXISTING_PARAMETERS = 3_397_795_840
UNDERTHESEA_PACKAGE_BYTE_UPPER_BOUND = 25_642_737
DEPLOYED_PARAMETER_UPPER_BOUND = 3_558_502_386
PARAMETER_LIMIT = 4_000_000_000
EXPECTED_QUERIES = 1398
EXPECTED_PAIRS = 73128
EXPECTED_SEQUENCES = 146249
MAX_LENGTH = 256


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


def underthesea_paths() -> dict[str, Path]:
    import underthesea
    root = Path(inspect.getfile(underthesea)).parent
    model = root / "pipeline/word_tokenize/models/ws_crf_vlsp2013_20230727"
    return {
        "uts_dictionary": model / "dictionary.bin",
        "uts_features": model / "features.bin",
        "uts_model": model / "models.bin",
    }


def contract_paths(args: argparse.Namespace, include_eval: bool = False) -> dict[str, Path]:
    paths = {
        "folds": args.folds, "pool": args.pool, "contexts": args.contexts,
        "selector": args.selector, "preregistration": args.preregistration,
        "model": args.model / "model.safetensors", **underthesea_paths(),
    }
    if include_eval:
        paths.update({
            "anchor": args.anchor, "e5_predictions": args.e5_predictions,
            "jina_predictions": args.jina_predictions, "sources_db": args.sources_db,
        })
    return paths


def validate_contract(args: argparse.Namespace, include_eval: bool = False) -> dict:
    hashes = {name: require_hash(name, path) for name, path in contract_paths(args, include_eval).items()}
    prereg = json.loads(args.preregistration.read_text(encoding="utf-8"))
    checks = {
        "sealed_before_inference": prereg.get("status") == "SEALED_BEFORE_INFERENCE_OR_V2_METRIC",
        "frozen_model": prereg["model"].get("parameters") == MODEL_PARAMETERS,
        "no_augmentation": prereg["preprocessing"].get("augmentation") is False,
        "rows_added_zero": prereg["preprocessing"].get("rows_added") == 0,
        "rows_modified_zero": prereg["preprocessing"].get("rows_modified") == 0,
        "fixed_length": prereg["inference"].get("max_length") == MAX_LENGTH,
        "fixed_masking": prereg["inference"].get("masking") == "all_query_tokens_simultaneously",
        "fixed_parent_aggregation": prereg["inference"].get("parent_aggregation") == "max_over_evidence",
        "adaptive_k_disabled": prereg["inference"].get("adaptive_k") is False,
        "parameter_budget": DEPLOYED_PARAMETER_UPPER_BOUND < PARAMETER_LIMIT and prereg["parameter_budget"].get("pass") is True,
        "underthesea_version": importlib.metadata.version("underthesea") == "9.5.0",
    }
    if not all(checks.values()):
        raise RuntimeError(f"sealed contract failure: {checks}")
    return {"hashes": hashes, "checks": checks}


def set_determinism() -> None:
    torch.manual_seed(20260913)
    torch.cuda.manual_seed_all(20260913)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def fold0_rows(args: argparse.Namespace) -> list[dict]:
    folds = json.loads(args.folds.read_text(encoding="utf-8"))
    fold0 = set(map(str, folds["folds"]["fold_0"]))
    rows = [row for row in read_jsonl(args.pool) if str(row["qid"]) in fold0]
    if len(rows) != EXPECTED_QUERIES or any(str(row["fold"]) != "fold_0" for row in rows):
        raise RuntimeError(f"Fold-0 pool mismatch: {len(rows)}")
    if sum(len(row["doc_ids"]) for row in rows) != EXPECTED_PAIRS:
        raise RuntimeError("Fold-0 candidate-pair mismatch")
    return rows


def load_documents(args: argparse.Namespace, rows: list[dict]) -> dict[str, str]:
    needed = {str(doc) for row in rows for doc in row["doc_ids"]}
    documents = {}
    for item in read_jsonl(args.contexts):
        doc = str(item["doc_id"])
        if doc in needed:
            documents[doc] = str(item["passage"])
    if set(documents) != needed:
        raise RuntimeError(f"context coverage mismatch: {len(documents)}/{len(needed)}")
    return documents


def evidence_for_row(args: argparse.Namespace, row: dict, documents: dict[str, str]):
    sys.path.insert(0, str(args.root))
    from benchmark_jina_reranker_holdouts import top_passages
    query = str(row["query"])
    owners: list[tuple[str, int]] = []
    passages: list[str] = []
    for doc in map(str, row["doc_ids"]):
        selected = list(top_passages(query, documents[doc], count=2))
        if not selected or len(selected) > 2:
            raise RuntimeError(f"evidence count mismatch: {row['qid']} {doc} {len(selected)}")
        for passage_index, passage in enumerate(selected):
            owners.append((doc, passage_index))
            passages.append(str(passage))
    return query, owners, passages


def load_model(args: argparse.Namespace, dtype: torch.dtype):
    from transformers import AutoModelForMaskedLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, use_fast=False)
    model = AutoModelForMaskedLM.from_pretrained(
        args.model, local_files_only=True, dtype=dtype, attn_implementation="eager"
    )
    count = sum(parameter.numel() for parameter in model.parameters())
    if count != MODEL_PARAMETERS:
        raise RuntimeError(f"model parameter mismatch: {count}")
    if tokenizer.mask_token_id != 64000 or model.config.max_position_embeddings != 258:
        raise RuntimeError("model/tokenizer structural contract mismatch")
    model.eval().to("cuda")
    return model, tokenizer


def encode_instance(tokenizer, query: str, evidence: str) -> tuple[list[int], list[int], list[int]]:
    from underthesea import word_tokenize
    segmented_query = word_tokenize(query, format="text")
    segmented_evidence = word_tokenize(evidence, format="text")
    query_ids = tokenizer.encode(segmented_query, add_special_tokens=False)
    if not 1 <= len(query_ids) <= 37:
        raise RuntimeError(f"query token contract failure: {len(query_ids)}")
    evidence_ids = tokenizer.encode(segmented_evidence, add_special_tokens=False)
    available = MAX_LENGTH - tokenizer.num_special_tokens_to_add(pair=True) - len(query_ids)
    if available <= 0:
        raise RuntimeError("no evidence capacity")
    masked = [int(tokenizer.mask_token_id)] * len(query_ids)
    input_ids = tokenizer.build_inputs_with_special_tokens(evidence_ids[:available], masked)
    if len(input_ids) > MAX_LENGTH:
        raise RuntimeError(f"encoded length overflow: {len(input_ids)}")
    mask_positions = [index for index, token in enumerate(input_ids) if token == tokenizer.mask_token_id]
    if len(mask_positions) < len(query_ids):
        raise RuntimeError("mask positions missing")
    mask_positions = mask_positions[-len(query_ids):]
    return input_ids, mask_positions, query_ids


@torch.inference_mode()
def score_sequences(model, tokenizer, query: str, passages: list[str], batch_size: int) -> list[float]:
    encoded = [encode_instance(tokenizer, query, passage) for passage in passages]
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    output: list[float] = []
    for start in range(0, len(encoded), batch_size):
        group = encoded[start:start + batch_size]
        width = max(len(item[0]) for item in group)
        input_ids = torch.full((len(group), width), int(tokenizer.pad_token_id), dtype=torch.long)
        attention_mask = torch.zeros((len(group), width), dtype=torch.long)
        for index, (tokens, _, _) in enumerate(group):
            input_ids[index, :len(tokens)] = torch.tensor(tokens)
            attention_mask[index, :len(tokens)] = 1
        input_ids = input_ids.to(device); attention_mask = attention_mask.to(device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=dtype == torch.float16):
            hidden = model.roberta(input_ids=input_ids, attention_mask=attention_mask, return_dict=True).last_hidden_state
            selected = torch.cat([
                hidden[index, torch.tensor(positions, device=device)]
                for index, (_, positions, _) in enumerate(group)
            ], dim=0)
            logits = model.lm_head(selected)
        log_probs = torch.log_softmax(logits.float(), dim=-1)
        cursor = 0
        for _, positions, labels in group:
            length = len(positions)
            targets = torch.tensor(labels, dtype=torch.long, device=device)
            score = log_probs[cursor:cursor + length].gather(1, targets[:, None]).mean()
            if not torch.isfinite(score):
                raise RuntimeError("non-finite query likelihood")
            output.append(float(score.cpu()))
            cursor += length
        del hidden, selected, logits, log_probs
    return output


def parent_ranking(owners: list[tuple[str, int]], values: list[float]):
    parent: dict[str, float] = {}
    for (doc, _), value in zip(owners, values):
        parent[doc] = max(parent.get(doc, -math.inf), value)
    return sorted(parent, key=lambda doc: (-parent[doc], doc)), parent


def deterministic_rows(rows: list[dict], count: int) -> list[dict]:
    return sorted(rows, key=lambda row: hashlib.sha256(str(row["qid"]).encode()).hexdigest())[:count]


def preflight(args: argparse.Namespace) -> None:
    contract = validate_contract(args)
    rows = fold0_rows(args)
    documents = load_documents(args, rows)
    model_files = {
        path.name: {"sha256": sha256(path), "bytes": path.stat().st_size}
        for path in sorted(args.model.iterdir()) if path.is_file()
    }
    report = {
        "schema_version": "dsc2026.research_v2.legal_mlm_query_likelihood_input_manifest.v1",
        "status": "PASS_INPUT_ONLY_NO_LABEL_METRIC_READ",
        "queries": len(rows), "candidate_pairs": sum(len(row["doc_ids"]) for row in rows),
        "unique_candidate_parents": len(documents), "expected_sequences": EXPECTED_SEQUENCES,
        "model_repo": "NghiemAbe/Vi-Legal-PhoBert",
        "model_revision": "1ad5c52f96918aa62245dfe68e5eb467233a1a90",
        "model_parameters": MODEL_PARAMETERS,
        "deployed_parameter_upper_bound": DEPLOYED_PARAMETER_UPPER_BOUND,
        "parameter_limit": PARAMETER_LIMIT,
        "model_files": model_files, "contract": contract,
        "runtime": {"python": platform.python_version(), "torch": torch.__version__,
                    "transformers": importlib.metadata.version("transformers"),
                    "underthesea": importlib.metadata.version("underthesea")},
        "labels_read": False, "metric_read": False, "augmentation": False,
    }
    if args.input_manifest.exists():
        existing = json.loads(args.input_manifest.read_text(encoding="utf-8"))
        if canonical_json(existing) != canonical_json(report):
            raise RuntimeError("existing input manifest differs")
    else:
        write_json(args.input_manifest, report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


def parity(args: argparse.Namespace) -> None:
    validate_contract(args)
    if not args.input_manifest.exists():
        raise RuntimeError("preflight manifest missing")
    rows = deterministic_rows(fold0_rows(args), 2)
    documents = load_documents(args, rows)
    rendered = []
    for row in rows:
        query, owners, passages = evidence_for_row(args, row, documents)
        rendered.append((row, query, owners, passages))
    total = sum(len(item[3]) for item in rendered)
    set_determinism(); torch.cuda.reset_peak_memory_stats(); started = time.perf_counter()
    fp32_model, tokenizer = load_model(args, torch.float32)
    fp32 = []
    for _, query, _, passages in rendered:
        fp32.extend(score_sequences(fp32_model, tokenizer, query, passages, 8))
    fp32_seconds = time.perf_counter() - started
    del fp32_model; gc.collect(); torch.cuda.empty_cache()
    set_determinism(); torch.cuda.reset_peak_memory_stats(); started = time.perf_counter()
    fp16_model, tokenizer = load_model(args, torch.float16)
    fp16 = []
    for _, query, _, passages in rendered:
        fp16.extend(score_sequences(fp16_model, tokenizer, query, passages, 32))
    fp16_seconds = time.perf_counter() - started
    fp16_peak = torch.cuda.max_memory_allocated() / 2**20
    errors = np.abs(np.asarray(fp32) - np.asarray(fp16))
    cursor = 0; top5_equal = True
    for _, _, owners, passages in rendered:
        count = len(passages)
        rank32, _ = parent_ranking(owners, fp32[cursor:cursor + count])
        rank16, _ = parent_ranking(owners, fp16[cursor:cursor + count])
        top5_equal &= rank32[:5] == rank16[:5]
        cursor += count
    fp16_estimate = EXPECTED_SEQUENCES * fp16_seconds / total
    fp32_estimate = EXPECTED_SEQUENCES * fp32_seconds / total
    fp16_safe = top5_equal and float(errors.max()) <= 0.02 and fp16_estimate <= 14400
    fp32_safe = fp32_estimate <= 14400
    status = "PASS_FP16" if fp16_safe else "PASS_FP32" if fp32_safe else "BLOCKED_LOCAL_RUNTIME"
    report = {
        "schema_version": "dsc2026.research_v2.legal_mlm_query_likelihood_parity.v1",
        "status": status, "authorized_dtype": "float16" if fp16_safe else "float32" if fp32_safe else None,
        "authorized_batch": 32 if fp16_safe else 8 if fp32_safe else None,
        "queries": len(rows), "sequences": total, "score_max_abs_error": float(errors.max()),
        "score_mean_abs_error": float(errors.mean()), "parent_top5_identical": top5_equal,
        "fp32_seconds_including_preprocessing_and_load": fp32_seconds,
        "fp16_seconds_including_preprocessing_and_load": fp16_seconds,
        "fp32_estimated_full_seconds_conservative": fp32_estimate,
        "fp16_estimated_full_seconds_conservative": fp16_estimate,
        "fp16_peak_allocated_mib": fp16_peak, "labels_read": False, "metric_read": False,
    }
    if args.parity_report.exists():
        raise RuntimeError(f"refusing to overwrite parity report: {args.parity_report}")
    write_json(args.parity_report, report)
    print(json.dumps(report, indent=2), flush=True)


def fingerprint(args: argparse.Namespace) -> str:
    return hashlib.sha256(canonical_json({
        "preregistration": EXPECTED["preregistration"], "model": EXPECTED["model"],
        "folds": EXPECTED["folds"], "pool": EXPECTED["pool"], "contexts": EXPECTED["contexts"],
        "selector": EXPECTED["selector"], "underthesea": [EXPECTED["uts_dictionary"], EXPECTED["uts_features"], EXPECTED["uts_model"]],
        "max_length": MAX_LENGTH, "masking": "all_query_tokens_simultaneously",
        "score": "mean_fp32_log_probability", "evidence_count": 2, "aggregation": "parent_max",
    }).encode()).hexdigest()


def open_db(path: Path) -> sqlite3.Connection:
    db = sqlite3.connect(path)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("CREATE TABLE IF NOT EXISTS scores(qid TEXT,doc_id TEXT,passage_index INTEGER,score REAL,PRIMARY KEY(qid,doc_id,passage_index))")
    db.execute("CREATE TABLE IF NOT EXISTS progress(qid TEXT PRIMARY KEY,seconds REAL,sequences INTEGER,peak_mib REAL)")
    db.execute("CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT)")
    db.commit()
    return db


def score(args: argparse.Namespace) -> None:
    validate_contract(args)
    parity_report = json.loads(args.parity_report.read_text(encoding="utf-8"))
    if parity_report.get("status") not in {"PASS_FP16", "PASS_FP32"}:
        raise RuntimeError("numerical/runtime gate closed")
    dtype = torch.float16 if parity_report["authorized_dtype"] == "float16" else torch.float32
    batch_size = int(parity_report["authorized_batch"])
    rows = fold0_rows(args); documents = load_documents(args, rows)
    args.score_db.parent.mkdir(parents=True, exist_ok=True)
    db = open_db(args.score_db)
    expected_meta = {"fingerprint": fingerprint(args), "folds_sha256": EXPECTED["folds"], "pool_sha256": EXPECTED["pool"]}
    stored = dict(db.execute("SELECT key,value FROM metadata"))
    if stored and stored != expected_meta:
        raise RuntimeError(f"score cache metadata mismatch: {stored}")
    if not stored:
        with db:
            db.executemany("INSERT INTO metadata VALUES(?,?)", expected_meta.items())
    complete = {str(row[0]) for row in db.execute("SELECT qid FROM progress")}
    set_determinism(); model, tokenizer = load_model(args, dtype)
    process_started = time.perf_counter(); newly_done = 0
    for row in rows:
        qid = str(row["qid"])
        if qid in complete:
            continue
        query, owners, passages = evidence_for_row(args, row, documents)
        torch.cuda.reset_peak_memory_stats(); started = time.perf_counter()
        values = score_sequences(model, tokenizer, query, passages, batch_size)
        elapsed = time.perf_counter() - started
        if len(values) != len(owners):
            raise RuntimeError("score cardinality mismatch")
        with db:
            db.executemany("INSERT INTO scores VALUES(?,?,?,?)", [
                (qid, doc, passage_index, value)
                for (doc, passage_index), value in zip(owners, values)
            ])
            db.execute("INSERT INTO progress VALUES(?,?,?,?)", (
                qid, elapsed, len(values), torch.cuda.max_memory_allocated() / 2**20,
            ))
        newly_done += 1
        if newly_done % 10 == 0:
            done = len(complete) + newly_done
            qps = newly_done / (time.perf_counter() - process_started)
            print(f"score={done}/{EXPECTED_QUERIES} qps={qps:.3f} eta_min={(EXPECTED_QUERIES-done)/qps/60:.1f}", flush=True)
    stats = db.execute("SELECT COUNT(*),COUNT(DISTINCT qid),SUM(sequences),SUM(seconds),MAX(peak_mib) FROM progress").fetchone()
    score_rows = db.execute("SELECT COUNT(*) FROM scores").fetchone()[0]
    integrity = db.execute("PRAGMA integrity_check").fetchone()[0]
    db.execute("PRAGMA wal_checkpoint(TRUNCATE)"); db.close()
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
            native[(qid, str(source))] = [str(item["doc_id"]) for item in json.loads(payload) if str(item["doc_id"]) in pool[qid]][:5]
    db.close(); clean = {}
    for qid in pool:
        adapted_e5, frozen_e5 = e5[qid]
        components = [adapted_e5, frozen_e5, jina_v2[qid], set(native[(qid, "lal")]), set(native[(qid, "jina")])]
        if any(len(component) != 5 or not component <= pool[qid] for component in components):
            raise RuntimeError(f"clean expert contract failure: {qid}")
        clean[qid] = set().union(*components)
    return clean


def evaluate(args: argparse.Namespace) -> None:
    if args.report.exists():
        raise RuntimeError(f"refusing to re-evaluate sealed pilot: {args.report}")
    contract = validate_contract(args, include_eval=True)
    rows = fold0_rows(args); pool = {str(row["qid"]): set(map(str, row["doc_ids"])) for row in rows}
    anchor = {str(row["qid"]): row for row in read_jsonl(args.anchor) if str(row["qid"]) in pool}
    if set(anchor) != set(pool):
        raise RuntimeError("anchor qid mismatch")
    clean = load_clean_sets(args, pool)
    db = sqlite3.connect(f"file:{args.score_db.resolve().as_posix()}?mode=ro&immutable=1", uri=True)
    integrity = db.execute("PRAGMA integrity_check").fetchone()[0]
    progress = db.execute("SELECT COUNT(*),COUNT(DISTINCT qid),SUM(sequences),SUM(seconds),MAX(peak_mib) FROM progress").fetchone()
    score_rows = [(str(q), str(d), int(p), float(s)) for q, d, p, s in db.execute("SELECT qid,doc_id,passage_index,score FROM scores")]
    metadata = dict(db.execute("SELECT key,value FROM metadata")); db.close()
    parent_scores: dict[tuple[str, str], float] = {}
    for qid, doc, _, value in score_rows:
        parent_scores[(qid, doc)] = max(parent_scores.get((qid, doc), -math.inf), value)
    expected_pairs = {(qid, doc) for qid, docs in pool.items() for doc in docs}
    integrity_ok = (
        integrity == "ok" and progress[0] == progress[1] == EXPECTED_QUERIES
        and progress[2] == len(score_rows) == EXPECTED_SEQUENCES
        and len(parent_scores) == EXPECTED_PAIRS and set(parent_scores) == expected_pairs
        and metadata.get("fingerprint") == fingerprint(args)
    )
    if not integrity_ok:
        raise RuntimeError(f"score cache incomplete/inconsistent: {integrity} {progress} {len(score_rows)} {len(parent_scores)}")
    expert_r = []; expert_p = []; anchor_r = []; clean_r = []; clean_plus = []; candidate = []
    single = [[], []]; multi = [[], []]; depth = {k: [] for k in (5, 10, 20, 50)}
    wins = losses = churn = crossings_in = crossings_out = 0
    predictions = []; gold_buckets = Counter()
    for row in rows:
        qid = str(row["qid"]); docs = list(map(str, row["doc_ids"])); gold = set(map(str, anchor[qid]["gold"]))
        ordered = sorted(docs, key=lambda doc: (-parent_scores[(qid, doc)], doc)); top5 = ordered[:5]
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
        predictions.append({"qid": qid, "top5": top5, "ranking": ordered, "scores": [parent_scores[(qid, doc)] for doc in ordered]})
    metrics = {
        "recall_at_5": float(np.mean(expert_r)), "precision_at_5": float(np.mean(expert_p)),
        "current_anchor_recall_at_5": float(np.mean(anchor_r)),
        "existing_clean_experts_union": float(np.mean(clean_r)),
        "clean_experts_plus_legal_mlm_union": float(np.mean(clean_plus)),
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
    pass_standalone = metrics["recall_at_5"] >= 0.80 and metrics["clean_union_delta"] >= 0.004
    pass_orthogonal = metrics["recall_at_5"] >= 0.50 and metrics["clean_union_delta"] >= 0.005
    kill = metrics["recall_at_5"] < 0.45 or metrics["clean_union_delta"] < 0.0025 or not all(contract["checks"].values()) or not integrity_ok
    verdict = "KILL" if kill else "PASS_STANDALONE" if pass_standalone else "PASS_ORTHOGONAL" if pass_orthogonal else "INCONCLUSIVE_NO_TUNING"
    with args.predictions.open("x", encoding="utf-8", newline="\n") as stream:
        for prediction in predictions:
            stream.write(canonical_json(prediction) + "\n")
    parity = json.loads(args.parity_report.read_text(encoding="utf-8"))
    report = {
        "schema_version": "dsc2026.research_v2.legal_mlm_query_likelihood_fold0_report.v1",
        "status": "COMPLETE", "verdict": verdict, "metrics": metrics,
        "gate": {"pass_standalone": pass_standalone, "pass_orthogonal": pass_orthogonal,
                 "kill_recall_lt_0_45": metrics["recall_at_5"] < 0.45,
                 "kill_clean_delta_lt_0_0025": metrics["clean_union_delta"] < 0.0025,
                 "integrity": integrity_ok, "contract": all(contract["checks"].values())},
        "runtime": {"seconds": progress[3], "sequences": progress[2], "peak_mib": progress[4],
                    "dtype": parity["authorized_dtype"], "batch": parity["authorized_batch"]},
        "hashes": {**contract["hashes"], "score_db": sha256(args.score_db), "predictions": sha256(args.predictions)},
        "fixed_interface": {"evidence_count": 2, "max_length": MAX_LENGTH,
                            "masking": "all_query_tokens_simultaneously",
                            "score": "mean_fp32_log_probability_of_original_query_tokens",
                            "aggregation": "parent_max", "ranking": "score_desc_doc_id_string_asc_top5", "adaptive_k": False},
        "anti_rescue": "No model, mask-rate, query transformation, length, evidence, aggregation, score-sign, fusion, threshold, swap, routing, or adaptive-K grid.",
    }
    write_json(args.report, report)
    top5_rows = [canonical_json({"qid": row["qid"], "top5": row["top5"]}) for row in predictions]
    lock = {"schema_version": "dsc2026.research_v2.legal_mlm_query_likelihood_prediction_lock.v1",
            "status": "LOCKED", "verdict": verdict, "queries": len(predictions),
            "predictions_sha256": sha256(args.predictions),
            "canonical_qid_top5_sha256": hashlib.sha256(("\n".join(top5_rows) + "\n").encode()).hexdigest()}
    write_json(args.prediction_lock, lock)
    output_files = [args.input_manifest, args.parity_report, args.score_db, args.report, args.predictions, args.prediction_lock, args.preregistration, args.provenance_audit]
    manifest = {
        "schema_version": "dsc2026.research_v2.legal_mlm_query_likelihood_output_manifest.v1",
        "status": "COMPLETE", "verdict": verdict,
        "files": {path.name: {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size} for path in output_files},
        "reproduce": f'"{sys.executable}" "{Path(__file__).resolve()}" verify',
    }
    write_json(args.output_manifest, manifest)
    print(json.dumps({"verdict": verdict, "metrics": metrics}, ensure_ascii=False, indent=2), flush=True)


def verify(args: argparse.Namespace) -> None:
    manifest = json.loads(args.output_manifest.read_text(encoding="utf-8"))
    failures = []
    for name, item in manifest["files"].items():
        path = Path(item["path"])
        observed = sha256(path) if path.exists() else None
        if observed != item["sha256"]:
            failures.append({"file": name, "expected": item["sha256"], "observed": observed})
    db = sqlite3.connect(f"file:{args.score_db.resolve().as_posix()}?mode=ro&immutable=1", uri=True)
    integrity = db.execute("PRAGMA integrity_check").fetchone()[0]
    progress = db.execute("SELECT COUNT(*),SUM(sequences) FROM progress").fetchone()
    score_rows = db.execute("SELECT COUNT(*) FROM scores").fetchone()[0]; db.close()
    status = "PASS" if not failures and integrity == "ok" and progress == (EXPECTED_QUERIES, EXPECTED_SEQUENCES) and score_rows == EXPECTED_SEQUENCES else "FAIL"
    result = {"status": status, "manifest_sha256": sha256(args.output_manifest), "hash_failures": failures,
              "database": {"integrity": integrity, "queries": progress[0], "sequences": progress[1], "score_rows": score_rows}}
    print(json.dumps(result, indent=2), flush=True)
    if status != "PASS":
        raise RuntimeError("verification failed")


def parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[2]; output = root / "results/research_v2_open_rl"
    cache = root / "cache/research_v2_open_rl/legal_mlm_query_likelihood_fold0"
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["preflight", "parity", "score", "evaluate", "verify", "all"])
    p.add_argument("--root", type=Path, default=root)
    p.add_argument("--model", type=Path, default=root / "cache/research_v2_open_rl/models/vi-legal-phobert")
    p.add_argument("--folds", type=Path, default=root / "results/research_v2_forensic/V2_FOLDS.json")
    p.add_argument("--pool", type=Path, default=root / "results/research_v2_forensic/V2_CANDIDATE_POOL.jsonl")
    p.add_argument("--contexts", type=Path, default=root / "cache/research_v2_forensic/kaggle_input/research-v2-jina-boundary-v4/V2_CONTEXTS.jsonl")
    p.add_argument("--selector", type=Path, default=root / "benchmark_jina_reranker_holdouts.py")
    p.add_argument("--preregistration", type=Path, default=output / "LEGAL_MLM_QUERY_LIKELIHOOD_FOLD0_PREREGISTRATION.json")
    p.add_argument("--provenance-audit", type=Path, default=output / "LEGAL_MLM_QUERY_LIKELIHOOD_PROVENANCE_AUDIT.md")
    p.add_argument("--anchor", type=Path, default=root / "results/research_v2_post_e5/V2_ADAPTED_E5_LAL_EQUAL_RRF32_PREDICTIONS.jsonl")
    p.add_argument("--e5-predictions", type=Path, default=root / "results/research_v2_e5_confirmation/fold0_runner_parity/E5_CONFIRMATION_FOLD_0_PREDICTIONS.jsonl")
    p.add_argument("--jina-predictions", type=Path, default=root / "results/research_v2_forensic/V2_ZERO_SHOT_LEXICAL_PREDICTIONS.jsonl")
    p.add_argument("--sources-db", type=Path, default=root.parent / "LegalIR/cache/exp112_task_adaptive_retrieval/sources.sqlite")
    p.add_argument("--score-db", type=Path, default=cache / "scores.sqlite")
    p.add_argument("--input-manifest", type=Path, default=output / "LEGAL_MLM_QUERY_LIKELIHOOD_FOLD0_INPUT_MANIFEST.json")
    p.add_argument("--parity-report", type=Path, default=output / "LEGAL_MLM_QUERY_LIKELIHOOD_FOLD0_PARITY.json")
    p.add_argument("--report", type=Path, default=output / "LEGAL_MLM_QUERY_LIKELIHOOD_FOLD0_REPORT.json")
    p.add_argument("--predictions", type=Path, default=output / "LEGAL_MLM_QUERY_LIKELIHOOD_FOLD0_PREDICTIONS.jsonl")
    p.add_argument("--prediction-lock", type=Path, default=output / "LEGAL_MLM_QUERY_LIKELIHOOD_FOLD0_PREDICTION_LOCK.json")
    p.add_argument("--output-manifest", type=Path, default=output / "LEGAL_MLM_QUERY_LIKELIHOOD_FOLD0_OUTPUT_MANIFEST.json")
    return p


def main() -> None:
    args = parser().parse_args(); args.score_db.parent.mkdir(parents=True, exist_ok=True); args.report.parent.mkdir(parents=True, exist_ok=True)
    if args.stage in {"preflight", "all"}: preflight(args)
    if args.stage in {"parity", "all"}: parity(args)
    if args.stage in {"score", "all"}: score(args)
    if args.stage in {"evaluate", "all"}: evaluate(args)
    if args.stage == "verify": verify(args)


if __name__ == "__main__":
    main()
