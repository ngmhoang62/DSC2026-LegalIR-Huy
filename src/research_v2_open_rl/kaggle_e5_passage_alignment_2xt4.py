"""Fail-closed 2xT4 transport for the sealed E5 passage-alignment experiment.

This runner changes execution only. The scientific data, objective, optimizer,
schedule, epochs, seed, ranking, and PASS/KILL equations are inherited from the
sealed Fold-0 preregistration and checked from the Kaggle input bundle.
"""

from __future__ import annotations

import argparse
import contextlib
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
import torch.distributed as dist
import torch.nn.functional as F


SEED = 113
EXPECTED_ALL_GROUPS = 6991
EXPECTED_TRAIN_GROUPS = 5586
EXPECTED_EVAL_QUERIES = 1398
EXPECTED_EVAL_PAIRS = 73128
EXPECTED_EVAL_SEQUENCES = 146249
BASE_PARAMETERS = 559890432
LAL_PARAMETERS = 596049920
ORIGINAL_SYSTEM_PARAMETERS = BASE_PARAMETERS + LAL_PARAMETERS
ORIGINAL_PREREG_SHA = "10e30921bce67a6276355be5fc43cb65c0d98a84d477341a83618111c8b7c192"
RUNTIME_PREREG_SHA = "38df8087e5701b38c0b1a974928b3a51df638fe0d09e72acad3033bd82349364"
FOLDS_SHA = "94ad5c6d5e582ced5eec8d2c3c15f938454c17e713614391091e72abea9aba19"
POOL_SHA = "96a44e66549cc211e1f9d0fabb84fc825db3f21f32d5b349eeca3b1c0413e277"
GROUPS_SHA = "681ca1340dac9e6b498cbde30fa8ac7fee98840664ae51a126491d9173009b2f"
GROUPS_MANIFEST_SHA = "4a6270b9f3e0f9f7f260443107b9868d3fcf634de6a8c0d93e32a20d95a59644"
QUERY_ADAPTER_SHA = "ef4c293ba78917522c81fa119a7406c3c936330c0985b53e5a0188cf91e36cc6"
MODEL_SHA = "afa0f907c7e1d8290854b8c295cd7d77521591b4c2f2a27c261258de92333ced"
QUERY_VECTORS_SHA = "6be3967d4b072c96f63f41f6221f54c0f9e742d0732cc27a29da1b14975a73ac"
QUERY_IDS_SHA = "6219072a8711ef27e533a21de5fd4eea09b02a101f22a644225ae22eeadfe3d4"
ANCHOR_SHA = "1854494964f2258243bc00896c76b11d56a3c02752a23af0ee04bac8e260de4d"
TEMPERATURE = 0.05
DRIFT_WEIGHT = 0.05
EPOCHS = 2
GLOBAL_BATCH = 16
LOCAL_BATCH = 8
GROUPS_PER_FORWARD = 2
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


def atomic_torch_save(value, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def mean(values) -> float:
    return float(np.mean(values))


def recall(docs, gold) -> float:
    return len(set(docs) & set(gold)) / len(gold)


def seed_all(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def paths(args) -> dict[str, Path]:
    inp, out = args.input_root.resolve(), args.output_root.resolve()
    return {
        "input": inp,
        "output": out,
        "bundle_manifest": inp / "BUNDLE_MANIFEST.json",
        "original_prereg": inp / "preregistration" / "E5_PASSAGE_ALIGNMENT_TOP2MEAN_FOLD0_PREREGISTRATION.json",
        "runtime_prereg": inp / "preregistration" / "E5_PASSAGE_ALIGNMENT_2XT4_RUNTIME_PREREGISTRATION.json",
        "folds": inp / "data" / "V2_FOLDS.json",
        "pool": inp / "data" / "V2_CANDIDATE_POOL.jsonl",
        "groups": inp / "data" / "V2_BOUNDARY_GROUPS.jsonl",
        "groups_manifest": inp / "data" / "V2_BOUNDARY_GROUPS_MANIFEST.json",
        "eval_rendered": inp / "data" / "FOLD0_EVAL_RENDERED_TOP2.jsonl",
        "clean_union": inp / "data" / "FOLD0_CLEAN_EXPERT_UNION.jsonl",
        "anchor": inp / "data" / "V2_ADAPTED_E5_LAL_EQUAL_RRF32_PREDICTIONS.jsonl",
        "query_vectors": inp / "cache" / "adapted_query_vectors.f32.npy",
        "query_ids": inp / "cache" / "query_ids.json",
        "query_cache_manifest": inp / "cache" / "QUERY_CACHE_MANIFEST.json",
        "query_adapter": inp / "model" / "fold0_query_adapter_epoch2.pt",
        "model": inp / "model" / "vietlegal-e5",
        "preflight": out / "INPUT_PREFLIGHT.json",
        "parity": out / "RUNTIME_PARITY_AND_COST_GATE.json",
        "training_dir": out / "training",
        "resume": out / "training" / "resume.pt",
        "training_success": out / "training" / "_SUCCESS.json",
        "checkpoint": out / "training" / "epoch-2.pt",
        "score_dir": out / "score",
        "merged_db": out / "score" / "scores.sqlite",
        "merge_report": out / "SCORE_MERGE_REPORT.json",
        "report": out / "E5_PASSAGE_ALIGNMENT_TOP2MEAN_FOLD0_REPORT.json",
        "predictions": out / "E5_PASSAGE_ALIGNMENT_TOP2MEAN_FOLD0_PREDICTIONS.jsonl",
        "prediction_lock": out / "E5_PASSAGE_ALIGNMENT_TOP2MEAN_FOLD0_PREDICTION_LOCK.json",
        "output_manifest": out / "OUTPUT_MANIFEST.json",
    }


def verify_bundle(p: dict[str, Path]) -> dict:
    manifest = json.loads(p["bundle_manifest"].read_text(encoding="utf-8"))
    if manifest.get("status") != "SEALED_KAGGLE_INPUT_BUNDLE":
        raise RuntimeError("bundle is not sealed")
    failures = []
    for relative, expected in manifest["files"].items():
        candidate = p["input"] / relative
        observed = sha256(candidate) if candidate.is_file() else None
        if observed != expected["sha256"] or (candidate.is_file() and candidate.stat().st_size != expected["bytes"]):
            failures.append({"path": relative, "expected": expected, "observed_sha256": observed})
    if failures:
        raise RuntimeError(f"bundle hash failure: {failures[:3]}")
    return manifest


def load_groups(p: dict[str, Path]) -> list[dict]:
    rows = list(records(p["groups"]))
    if len(rows) != EXPECTED_ALL_GROUPS or len({str(row["qid"]) for row in rows}) != EXPECTED_ALL_GROUPS:
        raise RuntimeError("boundary group cardinality mismatch")
    for row in rows:
        if not row["positives"] or len(row["negatives"]) != 4:
            raise RuntimeError(f"boundary group structure mismatch: {row['qid']}")
        for parent in row["positives"] + row["negatives"]:
            if len(parent["passages"]) not in (1, 2):
                raise RuntimeError(f"passage count mismatch: {row['qid']} {parent['doc_id']}")
        positive = {str(x["doc_id"]) for x in row["positives"]}
        negative = {str(x["doc_id"]) for x in row["negatives"]}
        if positive & negative:
            raise RuntimeError(f"sibling gold negative: {row['qid']}")
    return rows


def training_rows(p: dict[str, Path], rows: list[dict]) -> list[dict]:
    manifest = json.loads(p["groups_manifest"].read_text(encoding="utf-8"))
    excluded = set(map(str, manifest["held_fold_duplicate_exclusions"]["fold_0"]))
    selected = [row for row in rows if row["fold"] != "fold_0" and str(row["qid"]) not in excluded]
    if len(selected) != EXPECTED_TRAIN_GROUPS or any(row["fold"] == "fold_0" for row in selected):
        raise RuntimeError("strict Fold-0 training isolation failure")
    return selected


def load_query_cache(p: dict[str, Path]) -> tuple[np.ndarray, dict[str, int]]:
    ids = list(map(str, json.loads(p["query_ids"].read_text(encoding="utf-8"))))
    vectors = np.load(p["query_vectors"], mmap_mode="r")
    if vectors.shape != (EXPECTED_ALL_GROUPS, 1024) or len(ids) != len(set(ids)) != EXPECTED_ALL_GROUPS:
        raise RuntimeError("query cache shape or IDs invalid")
    return vectors, {qid: index for index, qid in enumerate(ids)}


def load_eval_rows(p: dict[str, Path]) -> list[dict]:
    rows = list(records(p["eval_rendered"]))
    qids = {str(row["qid"]) for row in rows}
    pairs = sum(len(row["doc_ids"]) for row in rows)
    sequences = sum(len(row["passages"]) for row in rows)
    if len(rows) != EXPECTED_EVAL_QUERIES or len(qids) != EXPECTED_EVAL_QUERIES:
        raise RuntimeError("rendered evaluation qid cardinality")
    if pairs != EXPECTED_EVAL_PAIRS or sequences != EXPECTED_EVAL_SEQUENCES:
        raise RuntimeError("rendered evaluation pair/sequence cardinality")
    for row in rows:
        docs = set(map(str, row["doc_ids"]))
        owners = [str(item["doc_id"]) for item in row["passages"]]
        counts = Counter(owners)
        if set(owners) != docs or any(value not in (1, 2) for value in counts.values()):
            raise RuntimeError(f"rendered ownership mismatch: {row['qid']}")
    return rows


def preflight(args) -> None:
    p = paths(args)
    p["output"].mkdir(parents=True, exist_ok=True)
    manifest = verify_bundle(p)
    observed = {
        "original_prereg": sha256(p["original_prereg"]),
        "runtime_prereg": sha256(p["runtime_prereg"]),
        "folds": sha256(p["folds"]),
        "pool": sha256(p["pool"]),
        "groups": sha256(p["groups"]),
        "groups_manifest": sha256(p["groups_manifest"]),
        "query_adapter": sha256(p["query_adapter"]),
        "model_safetensors": sha256(p["model"] / "model.safetensors"),
        "query_vectors": sha256(p["query_vectors"]),
        "query_ids": sha256(p["query_ids"]),
        "anchor": sha256(p["anchor"]),
    }
    expected = {
        "original_prereg": ORIGINAL_PREREG_SHA, "runtime_prereg": RUNTIME_PREREG_SHA,
        "folds": FOLDS_SHA, "pool": POOL_SHA, "groups": GROUPS_SHA,
        "groups_manifest": GROUPS_MANIFEST_SHA, "query_adapter": QUERY_ADAPTER_SHA,
        "model_safetensors": MODEL_SHA, "query_vectors": QUERY_VECTORS_SHA,
        "query_ids": QUERY_IDS_SHA, "anchor": ANCHOR_SHA,
    }
    checks = {key: observed[key] == value for key, value in expected.items()}
    original = json.loads(p["original_prereg"].read_text(encoding="utf-8"))
    runtime = json.loads(p["runtime_prereg"].read_text(encoding="utf-8"))
    checks["original_sealed"] = original["status"] == "SEALED_AFTER_INPUT_CARDINALITY_ONLY_BEFORE_TRAINING_OR_V2_METRIC"
    checks["runtime_sealed"] = runtime["status"] == "SEALED_BEFORE_KAGGLE_TRAINING_OR_HELD_FOLD_METRIC"
    checks["no_augmentation"] = original["training"]["augmentation"] is False
    checks["parameter_limit"] = ORIGINAL_SYSTEM_PARAMETERS < 4_000_000_000
    groups = load_groups(p)
    train = training_rows(p, groups)
    evaluation = load_eval_rows(p)
    vectors, lookup = load_query_cache(p)
    checks["query_coverage"] = all(str(row["qid"]) in lookup for row in groups)
    clean = {str(row["qid"]): list(map(str, row["top5_union"])) for row in records(p["clean_union"])}
    anchor = {str(row["qid"]): row for row in records(p["anchor"])}
    eval_qids = {str(row["qid"]) for row in evaluation}
    checks["evaluation_support"] = set(clean) == eval_qids and eval_qids <= set(anchor)
    checks["two_gpus"] = torch.cuda.device_count() == 2
    if not all(checks.values()):
        raise RuntimeError(f"preflight failure: {checks}")
    report = {
        "schema_version": "dsc2026.research_v2.e5_passage_alignment_2xt4_input_preflight.v1",
        "status": "PASS_INPUT_AND_TWO_GPU_NO_HELD_METRIC",
        "bundle_manifest_sha256": sha256(p["bundle_manifest"]),
        "bundle_files": len(manifest["files"]),
        "all_groups": len(groups), "training_groups": len(train),
        "evaluation_queries": len(evaluation), "evaluation_pairs": EXPECTED_EVAL_PAIRS,
        "evaluation_sequences": EXPECTED_EVAL_SEQUENCES,
        "query_vectors_shape": list(vectors.shape),
        "original_model_parameters": ORIGINAL_SYSTEM_PARAMETERS,
        "gpus": [torch.cuda.get_device_name(index) for index in range(2)],
        "observed_hashes": observed, "checks": checks,
    }
    write_json(p["preflight"], report)
    print(json.dumps(report, indent=2), flush=True)


class PassageEncoder(torch.nn.Module):
    def __init__(self, model_path: Path, device: torch.device, dtype: torch.dtype,
                 checkpoint: Path | None = None):
        super().__init__()
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModel, AutoTokenizer
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
        base = AutoModel.from_pretrained(str(model_path), local_files_only=True,
                                         dtype=dtype, attn_implementation="eager")
        count = sum(value.numel() for value in base.parameters())
        if count != BASE_PARAMETERS:
            raise RuntimeError(f"base parameter mismatch: {count}")
        self.model = get_peft_model(
            base,
            LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                       target_modules=["query", "value"], bias="none"),
        )
        self.model.config.use_cache = False
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.model.enable_input_require_grads()
        if checkpoint is not None:
            state = torch.load(checkpoint, map_location="cpu", weights_only=False)
            self.load_state_dict(state["adapter"], strict=False)
        self.to(device)

    def tokenize(self, texts: list[str]):
        batch = self.tokenizer(["passage: " + text for text in texts], padding=True,
                               truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
        return {key: value.to(self.device, non_blocking=True) for key, value in batch.items()}

    def forward(self, batch) -> torch.Tensor:
        hidden = self.model(**batch).last_hidden_state.float()
        mask = batch["attention_mask"].unsqueeze(-1)
        return F.normalize((hidden * mask).sum(1) / mask.sum(1).clamp_min(1), dim=-1)

    @contextlib.contextmanager
    def adapter_disabled(self):
        with self.model.disable_adapter():
            yield

    def adapter_state(self) -> dict[str, torch.Tensor]:
        return {name: value.detach().cpu() for name, value in self.named_parameters() if value.requires_grad}

    def adapter_gradient(self) -> torch.Tensor:
        chunks = []
        for name, value in sorted(self.named_parameters()):
            if value.requires_grad:
                if value.grad is None:
                    raise RuntimeError(f"missing adapter gradient: {name}")
                chunks.append(value.grad.detach().float().cpu().reshape(-1))
        return torch.cat(chunks)


def group_losses(forward_model, module: PassageEncoder, rows: list[dict], queries: torch.Tensor) -> list[torch.Tensor]:
    texts, sequence_groups, parent_layout = [], [], []
    for group_index, row in enumerate(rows):
        parents = row["positives"] + row["negatives"]
        layout = []
        for parent in parents:
            start = len(texts)
            for text in parent["passages"]:
                texts.append(str(text))
                sequence_groups.append(group_index)
            layout.append((start, len(texts)))
        parent_layout.append((layout, len(row["positives"])))
    batch = module.tokenize(texts)
    with torch.no_grad(), module.adapter_disabled():
        frozen = module(batch).detach()
    adapted = forward_model(batch)
    group_index = torch.as_tensor(sequence_groups, device=module.device, dtype=torch.long)
    raw_scores = (adapted * queries[group_index]).sum(1)
    losses = []
    for layout, positive_count in parent_layout:
        scores = torch.stack([raw_scores[start:end].mean() for start, end in layout])
        positives = scores[:positive_count] / TEMPERATURE
        negatives = scores[positive_count:] / TEMPERATURE
        contrastive = (torch.logaddexp(positives, torch.logsumexp(negatives, dim=0)) - positives).mean()
        first, last = layout[0][0], layout[-1][1]
        drift = (1 - F.cosine_similarity(adapted[first:last], frozen[first:last], dim=1)).mean()
        losses.append(contrastive + DRIFT_WEIGHT * drift)
    return losses


def gradient_probe(p: dict[str, Path], rows: list[dict], vectors, lookup, dtype: torch.dtype,
                   batched: bool, training: bool = False) -> dict:
    seed_all()
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(device)
    model = PassageEncoder(p["model"], device, dtype=dtype)
    model.train(training); model.zero_grad(set_to_none=True)
    started = time.perf_counter(); losses = []
    if batched:
        q = torch.stack([torch.as_tensor(np.array(vectors[lookup[str(row["qid"])]], copy=True), device=device) for row in rows])
        items = group_losses(model, model, rows, q)
        (torch.stack(items).mean()).backward()
        losses = [float(item.detach()) for item in items]
    else:
        for row in rows:
            q = torch.as_tensor(np.array(vectors[lookup[str(row["qid"])]], copy=True), device=device)[None]
            item = group_losses(model, model, [row], q)[0]
            (item / len(rows)).backward(); losses.append(float(item.detach()))
    torch.cuda.synchronize(device)
    gradient = model.adapter_gradient()
    result = {
        "training_mode": training,
        "losses": losses,
        "gradient": gradient,
        "gradient_norm": float(torch.linalg.vector_norm(gradient)),
        "gradient_sha256": hashlib.sha256(gradient.numpy().tobytes()).hexdigest(),
        "seconds": time.perf_counter() - started,
        "peak_mib": torch.cuda.max_memory_allocated(device) / 2**20,
    }
    del model
    torch.cuda.empty_cache()
    return result


def cosine_and_relative(first: torch.Tensor, second: torch.Tensor) -> tuple[float, float]:
    cosine = float(F.cosine_similarity(first, second, dim=0))
    relative = float(torch.linalg.vector_norm(first - second) / torch.linalg.vector_norm(first).clamp_min(1e-12))
    return cosine, relative


@torch.inference_mode()
def encode_texts(model: PassageEncoder, texts: list[str], batch_size: int) -> np.ndarray:
    output = []
    for start in range(0, len(texts), batch_size):
        output.append(model(model.tokenize(texts[start:start + batch_size])).cpu().numpy())
    return np.concatenate(output) if output else np.empty((0, 1024), dtype=np.float32)


def parity(args) -> None:
    p = paths(args)
    if json.loads(p["preflight"].read_text(encoding="utf-8"))["status"] != "PASS_INPUT_AND_TWO_GPU_NO_HELD_METRIC":
        raise RuntimeError("preflight not passed")
    if p["parity"].exists():
        existing = json.loads(p["parity"].read_text(encoding="utf-8"))
        if existing["status"] != "PASS_RUNTIME_AND_COST_NO_HELD_METRIC":
            raise RuntimeError("existing parity report is not PASS")
        print(json.dumps(existing, indent=2)); return
    train = training_rows(p, load_groups(p))[:2]
    vectors, lookup = load_query_cache(p)
    fp32_scalar = gradient_probe(p, train, vectors, lookup, torch.float32, False)
    fp16_scalar = gradient_probe(p, train, vectors, lookup, torch.float16, False)
    fp16_batch_eval = gradient_probe(p, train, vectors, lookup, torch.float16, True)
    fp16_train_a = gradient_probe(p, train, vectors, lookup, torch.float16, True, training=True)
    fp16_train_b = gradient_probe(p, train, vectors, lookup, torch.float16, True, training=True)
    batch_cosine, batch_relative = cosine_and_relative(fp16_scalar["gradient"], fp16_batch_eval["gradient"])
    precision_cosine, precision_relative = cosine_and_relative(fp32_scalar["gradient"], fp16_scalar["gradient"])
    batch_loss_error = max(abs(a - b) for a, b in zip(fp16_scalar["losses"], fp16_batch_eval["losses"]))
    precision_loss_error = max(abs(a - b) for a, b in zip(fp32_scalar["losses"], fp16_scalar["losses"]))
    replay_exact = fp16_train_a["gradient_sha256"] == fp16_train_b["gradient_sha256"] and fp16_train_a["losses"] == fp16_train_b["losses"]
    # Two groups per call, two ranks in parallel, two epochs. Add 35% compute
    # margin and 900 seconds for DDP/checkpoints/scoring/merge.
    projected_training = fp16_train_a["seconds"] * (EXPECTED_TRAIN_GROUPS * EPOCHS / GROUPS_PER_FORWARD / 2) * 1.35
    sample_eval = load_eval_rows(p)[:8]
    device = torch.device("cuda:0"); seed_all(); model = PassageEncoder(p["model"], device, torch.float16); model.eval()
    score_started = time.perf_counter()
    for row in sample_eval:
        encode_texts(model, [str(item["text"]) for item in row["passages"]], 32)
    torch.cuda.synchronize(device)
    score_sample_seconds = time.perf_counter() - score_started
    score_peak = torch.cuda.max_memory_allocated(device) / 2**20
    projected_score = score_sample_seconds / len(sample_eval) * EXPECTED_EVAL_QUERIES / 2 * 1.35
    projected_total = projected_training + projected_score + 900
    peak = max(fp32_scalar["peak_mib"], fp16_scalar["peak_mib"], fp16_batch_eval["peak_mib"], fp16_train_a["peak_mib"], fp16_train_b["peak_mib"], score_peak)
    checks = {
        "batch_loss": batch_loss_error <= 0.001,
        "batch_gradient_cosine": batch_cosine >= 0.999,
        "batch_gradient_relative_l2": batch_relative <= 0.02,
        "precision_loss": precision_loss_error <= 0.01,
        "precision_gradient_cosine": precision_cosine >= 0.99,
        "optimized_replay_exact": replay_exact,
        "nonzero_gradient": fp16_train_a["gradient_norm"] > 0,
        "peak_memory": peak < 15000,
        "projected_cost": projected_total < 10800,
    }
    status = "PASS_RUNTIME_AND_COST_NO_HELD_METRIC" if all(checks.values()) else "FAIL_RUNTIME_GATE_NO_TRAINING_NO_HELD_METRIC"
    def public_probe(value):
        return {key: item for key, item in value.items() if key != "gradient"}
    report = {
        "schema_version": "dsc2026.research_v2.e5_passage_alignment_2xt4_runtime_gate.v1",
        "status": status, "groups": 2, "held_fold_metric_read": False,
        "fp32_scalar": public_probe(fp32_scalar), "fp16_scalar": public_probe(fp16_scalar),
        "fp16_batched_eval": public_probe(fp16_batch_eval),
        "fp16_training_replay_a": public_probe(fp16_train_a), "fp16_training_replay_b": public_probe(fp16_train_b),
        "comparisons": {
            "batched_vs_scalar_loss_max_abs": batch_loss_error,
            "batched_vs_scalar_gradient_cosine": batch_cosine,
            "batched_vs_scalar_gradient_relative_l2": batch_relative,
            "fp16_vs_fp32_loss_max_abs": precision_loss_error,
            "fp16_vs_fp32_gradient_cosine": precision_cosine,
            "fp16_vs_fp32_gradient_relative_l2_diagnostic": precision_relative,
            "optimized_replay_exact": replay_exact,
        },
        "cost": {"projected_training_seconds": projected_training,
                 "score_sample_queries": len(sample_eval), "score_sample_seconds": score_sample_seconds,
                 "projected_scoring_seconds": projected_score,
                 "fixed_overhead_seconds": 900, "projected_total_seconds": projected_total,
                 "limit_seconds": 10800, "peak_mib": peak},
        "checks": checks,
    }
    write_json(p["parity"], report)
    print(json.dumps(report, indent=2), flush=True)
    del model, fp32_scalar, fp16_scalar, fp16_batch_eval, fp16_train_a, fp16_train_b
    torch.cuda.empty_cache()
    if not all(checks.values()):
        raise RuntimeError("sealed 2xT4 runtime gate failed")


def capture_rng(rank: int, updates: int) -> dict:
    return {"rank": rank, "updates": updates, "python": random.getstate(),
            "numpy": np.random.get_state(), "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state(rank)}


def restore_rng(value: dict, rank: int, updates: int) -> None:
    if int(value["rank"]) != rank or int(value["updates"]) != updates:
        raise RuntimeError("rank RNG resume mismatch")
    random.setstate(value["python"]); np.random.set_state(value["numpy"])
    torch.set_rng_state(value["torch_cpu"]); torch.cuda.set_rng_state(value["torch_cuda"], rank)


def save_distributed_state(p, ddp, optimizer, scheduler, epoch, position, updates, contract_hash, rank) -> None:
    rng_path = p["training_dir"] / f"rng-rank{rank}.pt"
    atomic_torch_save(capture_rng(rank, updates), rng_path)
    dist.barrier()
    if rank == 0:
        state = {"adapter": ddp.module.adapter_state(), "optimizer": optimizer.state_dict(),
                 "scheduler": scheduler.state_dict(), "epoch": epoch, "position": position,
                 "updates": updates, "contract_hash": contract_hash}
        atomic_torch_save(state, p["resume"])
    dist.barrier()


def train(args) -> None:
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    rank = dist.get_rank(); world = dist.get_world_size()
    if world != 2:
        raise RuntimeError(f"sealed runtime requires exactly two DDP ranks, got {world}")
    device = torch.device(f"cuda:{rank}"); torch.cuda.set_device(device); seed_all()
    p = paths(args)
    preflight_report = json.loads(p["preflight"].read_text(encoding="utf-8"))
    parity_report = json.loads(p["parity"].read_text(encoding="utf-8"))
    if preflight_report["status"] != "PASS_INPUT_AND_TWO_GPU_NO_HELD_METRIC" or parity_report["status"] != "PASS_RUNTIME_AND_COST_NO_HELD_METRIC":
        raise RuntimeError("preflight/parity gate not passed")
    rows = training_rows(p, load_groups(p)); vectors, lookup = load_query_cache(p)
    qids = [str(row["qid"]) for row in rows]
    contract = {"runtime_preregistration": RUNTIME_PREREG_SHA, "groups": GROUPS_SHA,
                "qids": qids, "epochs": EPOCHS, "global_batch": GLOBAL_BATCH,
                "local_batch": LOCAL_BATCH, "groups_per_forward": GROUPS_PER_FORWARD,
                "seed": SEED, "world_size": world}
    contract_hash = digest(contract)
    if p["training_success"].exists():
        done = json.loads(p["training_success"].read_text(encoding="utf-8"))
        if done["contract_hash"] != contract_hash:
            raise RuntimeError("completed training contract mismatch")
        if rank == 0: print(json.dumps(done, indent=2))
        dist.barrier(); dist.destroy_process_group(); return
    from torch.nn.parallel import DistributedDataParallel
    from transformers import get_cosine_schedule_with_warmup
    model = PassageEncoder(p["model"], device, torch.float16)
    optimizer = torch.optim.AdamW([value for value in model.parameters() if value.requires_grad], lr=LR, weight_decay=WEIGHT_DECAY)
    steps_epoch = math.ceil(len(rows) / GLOBAL_BATCH)
    scheduler = get_cosine_schedule_with_warmup(optimizer, max(1, int(0.1 * steps_epoch * EPOCHS)), steps_epoch * EPOCHS)
    start_epoch = start_position = updates = 0
    resume_state = None
    if p["resume"].exists():
        resume_state = torch.load(p["resume"], map_location="cpu", weights_only=False)
        if resume_state["contract_hash"] != contract_hash:
            raise RuntimeError("resume contract mismatch")
        model.load_state_dict(resume_state["adapter"], strict=False)
        optimizer.load_state_dict(resume_state["optimizer"]); scheduler.load_state_dict(resume_state["scheduler"])
        start_epoch, start_position, updates = int(resume_state["epoch"]), int(resume_state["position"]), int(resume_state["updates"])
    ddp = DistributedDataParallel(model, device_ids=[rank], output_device=rank,
                                  broadcast_buffers=False, find_unused_parameters=False)
    if resume_state is not None:
        restore_rng(torch.load(p["training_dir"] / f"rng-rank{rank}.pt", map_location="cpu", weights_only=False), rank, updates)
    dist.barrier(); started = time.perf_counter(); torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(start_epoch, EPOCHS):
        order = list(range(len(rows))); random.Random(SEED + epoch).shuffle(order)
        begin = start_position if epoch == start_epoch else 0
        for position in range(begin, len(order), GLOBAL_BATCH):
            chosen = order[position:position + GLOBAL_BATCH]
            local_indices = chosen[rank::world]
            if not local_indices:
                raise RuntimeError("empty local DDP batch")
            ddp.train(); optimizer.zero_grad(set_to_none=True); local_loss_value = 0.0
            microbatches = [local_indices[start:start + GROUPS_PER_FORWARD]
                            for start in range(0, len(local_indices), GROUPS_PER_FORWARD)]
            for micro_index, indices in enumerate(microbatches):
                micro_rows = [rows[index] for index in indices]
                queries = torch.stack([torch.as_tensor(np.array(vectors[lookup[str(row["qid"])]], copy=True), device=device) for row in micro_rows])
                sync = micro_index == len(microbatches) - 1
                context = contextlib.nullcontext() if sync else ddp.no_sync()
                with context:
                    items = group_losses(ddp, ddp.module, micro_rows, queries)
                    loss = torch.stack(items).sum() / len(local_indices)
                    if not torch.isfinite(loss): raise FloatingPointError("nonfinite loss")
                    loss.backward(); local_loss_value += float(loss.detach())
            norm = torch.nn.utils.clip_grad_norm_(ddp.module.parameters(), 1.0, error_if_nonfinite=True)
            if float(norm) == 0: raise RuntimeError("zero passage-adapter gradient")
            optimizer.step(); scheduler.step(); updates += 1
            next_position = position + len(chosen)
            if updates % 10 == 0 or next_position >= len(order):
                save_distributed_state(p, ddp, optimizer, scheduler, epoch, next_position, updates, contract_hash, rank)
                loss_tensor = torch.tensor(local_loss_value, device=device)
                dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM); loss_tensor /= world
                if rank == 0:
                    elapsed = time.perf_counter() - started
                    print(json.dumps({"stage": "train", "epoch": epoch + 1,
                                      "position": next_position, "queries": len(order),
                                      "updates": updates, "loss": float(loss_tensor),
                                      "gradient_norm": float(norm), "elapsed_seconds": elapsed}), flush=True)
        save_distributed_state(p, ddp, optimizer, scheduler, epoch + 1, 0, updates, contract_hash, rank)
        if rank == 0:
            state = torch.load(p["resume"], map_location="cpu", weights_only=False)
            atomic_torch_save(state, p["training_dir"] / f"epoch-{epoch + 1}.pt")
        dist.barrier(); start_position = 0
    peak = torch.tensor(torch.cuda.max_memory_allocated(device) / 2**20, device=device)
    dist.all_reduce(peak, op=dist.ReduceOp.MAX)
    if rank == 0:
        if updates != math.ceil(EXPECTED_TRAIN_GROUPS / GLOBAL_BATCH) * EPOCHS:
            raise RuntimeError(f"update cardinality mismatch: {updates}")
        done = {"schema_version": "dsc2026.research_v2.e5_passage_alignment_2xt4_training.v1",
                "status": "COMPLETE_EPOCH2", "held_fold": "fold_0",
                "training_queries": len(rows), "epochs": EPOCHS, "updates": updates,
                "world_size": world, "global_batch": GLOBAL_BATCH,
                "contract_hash": contract_hash, "runtime_seconds_this_process": time.perf_counter() - started,
                "peak_mib_max_rank": float(peak), "checkpoint_sha256": sha256(p["checkpoint"]),
                "training_qids_sha256": digest(sorted(qids, key=int)),
                "reported_as_oof_scope": "Fold-0 strict OOF only; folds1-4 labels trained"}
        write_json(p["training_success"], done); print(json.dumps(done, indent=2), flush=True)
    dist.barrier(); dist.destroy_process_group()


def shard_db(p, rank: int, fingerprint: str):
    path = p["score_dir"] / f"scores-rank{rank}.sqlite"; path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path); db.execute("PRAGMA journal_mode=WAL"); db.execute("PRAGMA synchronous=FULL")
    db.execute("CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL)")
    db.execute("CREATE TABLE IF NOT EXISTS scores(qid TEXT,doc_id TEXT,score REAL,PRIMARY KEY(qid,doc_id))")
    db.execute("CREATE TABLE IF NOT EXISTS progress(qid TEXT PRIMARY KEY,seconds REAL,sequences INTEGER,peak_mib REAL)")
    existing = dict(db.execute("SELECT key,value FROM metadata"))
    if existing and existing.get("fingerprint") != fingerprint:
        raise RuntimeError("score shard fingerprint mismatch")
    if not existing:
        db.executemany("INSERT INTO metadata VALUES(?,?)", [("fingerprint", fingerprint), ("rank", str(rank))]); db.commit()
    return db, path


def parent_scores(row: dict, embeddings: np.ndarray, query: np.ndarray) -> dict[str, float]:
    raw = embeddings.astype(np.float32) @ query.astype(np.float32)
    grouped: dict[str, list[float]] = {}
    for item, value in zip(row["passages"], raw):
        grouped.setdefault(str(item["doc_id"]), []).append(float(value))
    if any(len(items) not in (1, 2) for items in grouped.values()):
        raise RuntimeError("parent passage count")
    return {doc: float(np.mean(items)) for doc, items in grouped.items()}


def score(args) -> None:
    if not dist.is_initialized(): dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    if world != 2: raise RuntimeError("scoring requires exactly two ranks")
    device = torch.device(f"cuda:{rank}"); torch.cuda.set_device(device); seed_all()
    p = paths(args)
    if json.loads(p["training_success"].read_text(encoding="utf-8"))["status"] != "COMPLETE_EPOCH2":
        raise RuntimeError("training incomplete")
    if json.loads(p["parity"].read_text(encoding="utf-8"))["status"] != "PASS_RUNTIME_AND_COST_NO_HELD_METRIC":
        raise RuntimeError("runtime gate incomplete")
    rows = load_eval_rows(p); vectors, lookup = load_query_cache(p)
    fingerprint = digest([RUNTIME_PREREG_SHA, sha256(p["checkpoint"]), QUERY_VECTORS_SHA,
                          sha256(p["eval_rendered"])])
    db, db_path = shard_db(p, rank, fingerprint)
    complete = {str(row[0]) for row in db.execute("SELECT qid FROM progress")}
    model = PassageEncoder(p["model"], device, torch.float16, checkpoint=p["checkpoint"]); model.eval()
    selected = [(index, row) for index, row in enumerate(rows) if index % world == rank]
    started_process = time.perf_counter(); newly = 0; torch.cuda.reset_peak_memory_stats(device)
    for _, row in selected:
        qid = str(row["qid"])
        if qid in complete: continue
        started = time.perf_counter(); texts = [str(item["text"]) for item in row["passages"]]
        embeddings = encode_texts(model, texts, 32)
        query = np.array(vectors[lookup[qid]], dtype=np.float32, copy=True)
        scores = parent_scores(row, embeddings, query); docs = list(map(str, row["doc_ids"]))
        if set(scores) != set(docs) or any(not math.isfinite(scores[doc]) for doc in docs):
            raise RuntimeError(f"score contract: {qid}")
        elapsed = time.perf_counter() - started
        with db:
            db.executemany("INSERT INTO scores VALUES(?,?,?)", [(qid, doc, scores[doc]) for doc in docs])
            db.execute("INSERT INTO progress VALUES(?,?,?,?)", (qid, elapsed, len(texts), torch.cuda.max_memory_allocated(device) / 2**20))
        newly += 1
        if newly % 10 == 0:
            done = len(complete) + newly; rate = newly / (time.perf_counter() - started_process)
            print(f"rank={rank} score={done}/{len(selected)} qps={rate:.3f} eta_min={(len(selected)-done)/rate/60:.1f}", flush=True)
    stats = db.execute("SELECT COUNT(*),SUM(sequences),SUM(seconds),MAX(peak_mib) FROM progress").fetchone()
    count = db.execute("SELECT COUNT(*) FROM scores").fetchone()[0]
    integrity = db.execute("PRAGMA integrity_check").fetchone()[0]
    db.execute("PRAGMA wal_checkpoint(TRUNCATE)"); db.close()
    print(json.dumps({"rank": rank, "database": str(db_path), "progress": stats,
                      "score_rows": count, "integrity": integrity}, indent=2), flush=True)
    dist.barrier(); dist.destroy_process_group()


def merge(args) -> None:
    p = paths(args); rows = load_eval_rows(p)
    if p["merged_db"].exists():
        report = json.loads(p["merge_report"].read_text(encoding="utf-8"))
        if report["status"] != "COMPLETE_FAIL_CLOSED": raise RuntimeError("existing merge not complete")
        print(json.dumps(report, indent=2)); return
    expected_qids = {str(row["qid"]) for row in rows}
    expected_pairs = {(str(row["qid"]), str(doc)) for row in rows for doc in row["doc_ids"]}
    temporary = p["merged_db"].with_name(f"scores.{os.getpid()}.tmp.sqlite"); temporary.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(temporary); db.execute("PRAGMA journal_mode=DELETE"); db.execute("PRAGMA synchronous=FULL")
    db.execute("CREATE TABLE scores(qid TEXT,doc_id TEXT,score REAL,PRIMARY KEY(qid,doc_id))")
    db.execute("CREATE TABLE progress(qid TEXT PRIMARY KEY,seconds REAL,sequences INTEGER,peak_mib REAL)")
    shard_hashes = {}
    for rank in range(2):
        shard = p["score_dir"] / f"scores-rank{rank}.sqlite"
        source = sqlite3.connect(f"file:{shard.resolve().as_posix()}?mode=ro&immutable=1", uri=True)
        if source.execute("PRAGMA integrity_check").fetchone()[0] != "ok": raise RuntimeError(f"shard {rank} integrity")
        score_rows = list(source.execute("SELECT qid,doc_id,score FROM scores"))
        progress_rows = list(source.execute("SELECT qid,seconds,sequences,peak_mib FROM progress")); source.close()
        db.executemany("INSERT INTO scores VALUES(?,?,?)", score_rows); db.executemany("INSERT INTO progress VALUES(?,?,?,?)", progress_rows)
        shard_hashes[f"rank_{rank}"] = sha256(shard)
    db.commit(); integrity = db.execute("PRAGMA integrity_check").fetchone()[0]
    qids = {str(row[0]) for row in db.execute("SELECT qid FROM progress")}
    pairs = {(str(q), str(d)) for q, d in db.execute("SELECT qid,doc_id FROM scores")}
    progress = db.execute("SELECT COUNT(*),SUM(sequences),SUM(seconds),MAX(peak_mib) FROM progress").fetchone(); db.close()
    checks = {"integrity": integrity == "ok", "qids": qids == expected_qids,
              "pairs": pairs == expected_pairs, "query_count": progress[0] == EXPECTED_EVAL_QUERIES,
              "sequence_count": progress[1] == EXPECTED_EVAL_SEQUENCES,
              "pair_count": len(pairs) == EXPECTED_EVAL_PAIRS}
    if not all(checks.values()): raise RuntimeError(f"merge failure: {checks}")
    os.replace(temporary, p["merged_db"])
    report = {"schema_version": "dsc2026.research_v2.e5_passage_alignment_2xt4_score_merge.v1",
              "status": "COMPLETE_FAIL_CLOSED", "queries": progress[0], "sequences": progress[1],
              "pairs": len(pairs), "summed_gpu_seconds": progress[2], "peak_mib": progress[3],
              "sqlite_integrity": integrity, "checks": checks, "shard_sha256": shard_hashes,
              "merged_sha256": sha256(p["merged_db"])}
    write_json(p["merge_report"], report); print(json.dumps(report, indent=2), flush=True)


def evaluate(args) -> None:
    p = paths(args)
    if p["report"].exists(): raise RuntimeError("refusing a second sealed evaluation")
    if json.loads(p["merge_report"].read_text(encoding="utf-8"))["status"] != "COMPLETE_FAIL_CLOSED":
        raise RuntimeError("score merge incomplete")
    rows = load_eval_rows(p); pool = {str(row["qid"]): set(map(str, row["doc_ids"])) for row in rows}
    anchor = {str(row["qid"]): row for row in records(p["anchor"]) if str(row["qid"]) in pool}
    clean = {str(row["qid"]): set(map(str, row["top5_union"])) for row in records(p["clean_union"])}
    if set(anchor) != set(pool) or set(clean) != set(pool): raise RuntimeError("evaluation support mismatch")
    db = sqlite3.connect(f"file:{p['merged_db'].resolve().as_posix()}?mode=ro&immutable=1", uri=True)
    integrity = db.execute("PRAGMA integrity_check").fetchone()[0]
    scores = {(str(q), str(d)): float(s) for q, d, s in db.execute("SELECT qid,doc_id,score FROM scores")}
    progress = db.execute("SELECT COUNT(*),SUM(sequences),SUM(seconds),MAX(peak_mib) FROM progress").fetchone(); db.close()
    expected = {(qid, doc) for qid, docs in pool.items() for doc in docs}
    integrity_ok = integrity == "ok" and progress[0] == EXPECTED_EVAL_QUERIES and len(scores) == EXPECTED_EVAL_PAIRS and set(scores) == expected
    if not integrity_ok: raise RuntimeError("completed cache audit failed")
    expert=[]; precision=[]; base_r=[]; clean_r=[]; clean_plus=[]; ceiling=[]; single=[[],[]]; multi=[[],[]]
    depth={key:[] for key in (5,10,20,50)}; wins=losses=inside=outside=churn=0; buckets=Counter(); predictions=[]
    for row in rows:
        qid=str(row["qid"]); docs=list(map(str,row["doc_ids"])); gold=set(map(str,anchor[qid]["gold"]))
        order=sorted(docs,key=lambda doc:(-scores[(qid,doc)],doc)); top5=order[:5]; base=list(map(str,anchor[qid]["fused_top5"]))
        er=recall(top5,gold); br=recall(base,gold); expert.append(er); precision.append(len(set(top5)&gold)/5); base_r.append(br)
        clean_r.append(recall(clean[qid],gold)); clean_plus.append(recall(clean[qid]|set(top5),gold)); ceiling.append(recall(pool[qid],gold))
        target=single if len(gold)==1 else multi; target[0].append(er); target[1].append(br)
        wins+=er>br; losses+=er<br; inside+=len((set(top5)-set(base))&gold); outside+=len((set(base)-set(top5))&gold); churn+=top5!=base
        for key in depth: depth[key].append(recall(order[:key],gold))
        for doc in gold:
            rank=order.index(doc)+1 if doc in pool[qid] else None
            bucket="missing" if rank is None else "1-5" if rank<=5 else "6-10" if rank<=10 else "11-20" if rank<=20 else "21-50" if rank<=50 else "51+"
            buckets[bucket]+=1
        predictions.append({"qid":qid,"top5":top5,"ranking":order,"scores":[scores[(qid,doc)] for doc in order]})
    metrics={"recall_at_5":mean(expert),"precision_at_5":mean(precision),"current_anchor_recall_at_5":mean(base_r),
             "clean_expert_union":mean(clean_r),"clean_plus_passage_alignment_union":mean(clean_plus),
             "clean_union_delta":mean(clean_plus)-mean(clean_r),"candidate_ceiling":mean(ceiling),
             "single_gold":{"queries":len(single[0]),"expert":mean(single[0]),"anchor":mean(single[1]),"delta":mean(single[0])-mean(single[1])},
             "multi_gold":{"queries":len(multi[0]),"expert":mean(multi[0]),"anchor":mean(multi[1]),"delta":mean(multi[0])-mean(multi[1])},
             "per_fold":{"fold_0":{"queries":len(rows),"expert":mean(expert),"anchor":mean(base_r),"delta":mean(expert)-mean(base_r)}},
             "wins_losses_ties":{"wins":wins,"losses":losses,"ties":len(rows)-wins-losses},
             "gold_crossings":{"into_top5":inside,"out_of_top5":outside},"top5_churn_queries":churn,
             "recall_depth":{str(key):mean(value) for key,value in depth.items()},"gold_rank_buckets":dict(buckets)}
    directional=wins>losses and inside>outside and metrics["multi_gold"]["delta"]>=-0.005
    pass_s=metrics["recall_at_5"]>=0.90 and metrics["clean_union_delta"]>=0.006 and directional
    pass_o=metrics["recall_at_5"]>=0.86 and metrics["clean_union_delta"]>=0.008 and directional
    kill=metrics["recall_at_5"]<0.84 or metrics["clean_union_delta"]<0.003 or wins<=losses or inside<=outside or metrics["multi_gold"]["delta"] < -0.02 or not integrity_ok
    verdict="KILL" if kill else "PASS_STANDALONE" if pass_s else "PASS_ORTHOGONAL" if pass_o else "INCONCLUSIVE_NO_TUNING"
    with p["predictions"].open("x",encoding="utf-8",newline="\n") as stream:
        for item in predictions: stream.write(json.dumps(item,ensure_ascii=False,sort_keys=True,separators=(",",":"))+"\n")
    report={"schema_version":"dsc2026.research_v2.e5_passage_alignment_fold0_report.v1","status":"COMPLETE","verdict":verdict,
            "metrics":metrics,"gate":{"pass_standalone":pass_s,"pass_orthogonal":pass_o,"kill":kill,"directional":directional,"integrity":integrity_ok},
            "runtime":{"training":json.loads(p["training_success"].read_text(encoding="utf-8")),"scoring_seconds_sum_gpu":progress[2],"scoring_sequences":progress[1],"scoring_peak_mib":progress[3]},
            "hashes":{"original_preregistration":ORIGINAL_PREREG_SHA,"runtime_preregistration":RUNTIME_PREREG_SHA,
                      "checkpoint":sha256(p["checkpoint"]),"query_vectors":sha256(p["query_vectors"]),"score_db":sha256(p["merged_db"]),
                      "rendered_evidence":sha256(p["eval_rendered"]),"clean_union":sha256(p["clean_union"])},
            "fixed_interface":{"query_adapter":"confirmed_fold0_frozen","passage_adapter":"qv_lora_r16_epoch2","evidence":"sealed_pre_rendered_lexical_top2","parent_aggregation":"arithmetic_mean_singleton_top1","ranking":"score_desc_parent_id_string_asc_top5","adaptive_k":False},
            "anti_rescue":"No passage-count, aggregation, LoRA, loss, LR, epoch, evidence, fusion, threshold, routing, rule, model or adaptive-K grid."}
    write_json(p["report"],report)
    canonical=[json.dumps({"qid":item["qid"],"top5":item["top5"]},sort_keys=True,separators=(",",":")) for item in predictions]
    write_json(p["prediction_lock"],{"schema_version":"dsc2026.research_v2.e5_passage_alignment_prediction_lock.v1","status":"LOCKED","verdict":verdict,"queries":len(predictions),"predictions_sha256":sha256(p["predictions"]),"canonical_qid_top5_sha256":hashlib.sha256(("\n".join(canonical)+"\n").encode()).hexdigest()})
    files=[p["preflight"],p["parity"],p["training_success"],p["checkpoint"],p["merge_report"],p["merged_db"],p["report"],p["predictions"],p["prediction_lock"]]
    manifest={"schema_version":"dsc2026.research_v2.e5_passage_alignment_2xt4_output_manifest.v1","status":"COMPLETE","verdict":verdict,
              "files":{item.name:{"path":str(item),"bytes":item.stat().st_size,"sha256":sha256(item)} for item in files}}
    write_json(p["output_manifest"],manifest); print(json.dumps({"verdict":verdict,"metrics":metrics},indent=2),flush=True)


def verify(args) -> None:
    p=paths(args); manifest=json.loads(p["output_manifest"].read_text(encoding="utf-8")); failures=[]
    for name,item in manifest["files"].items():
        path=Path(item["path"]); observed=sha256(path) if path.is_file() else None
        if observed!=item["sha256"]: failures.append({"file":name,"expected":item["sha256"],"observed":observed})
    db=sqlite3.connect(f"file:{p['merged_db'].resolve().as_posix()}?mode=ro&immutable=1",uri=True)
    integrity=db.execute("PRAGMA integrity_check").fetchone()[0]; progress=db.execute("SELECT COUNT(*),SUM(sequences) FROM progress").fetchone(); count=db.execute("SELECT COUNT(*) FROM scores").fetchone()[0]; db.close()
    status="PASS" if not failures and integrity=="ok" and progress==(EXPECTED_EVAL_QUERIES,EXPECTED_EVAL_SEQUENCES) and count==EXPECTED_EVAL_PAIRS else "FAIL"
    result={"status":status,"verdict":manifest["verdict"],"manifest_sha256":sha256(p["output_manifest"]),"hash_failures":failures,
            "database":{"integrity":integrity,"queries":progress[0],"sequences":progress[1],"score_rows":count}}
    print(json.dumps(result,indent=2),flush=True)
    if status!="PASS": raise RuntimeError("verification failed")


def parser():
    parser=argparse.ArgumentParser()
    parser.add_argument("stage",choices=["preflight","parity","train","score","merge","evaluate","verify"])
    parser.add_argument("--input-root",type=Path,required=True)
    parser.add_argument("--output-root",type=Path,required=True)
    return parser


def main() -> None:
    args=parser().parse_args(); args.output_root.mkdir(parents=True,exist_ok=True)
    if args.stage=="preflight": preflight(args)
    elif args.stage=="parity": parity(args)
    elif args.stage=="train": train(args)
    elif args.stage=="score": score(args)
    elif args.stage=="merge": merge(args)
    elif args.stage=="evaluate": evaluate(args)
    elif args.stage=="verify": verify(args)


if __name__=="__main__":
    main()
