"""Strict-V2 native LAL query-adapter transfer and fixed endpoint evaluation.

The learning mechanism is the already-tested LegalIR LAL adapter.  This runner
changes only the fold/population protocol and the preregistered Research V2
endpoint.  It is deliberately fail-closed and contains no hyperparameter grid.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import math
import os
import random
import sqlite3
import time
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from research_v2_e5_transfer import e5_transfer_runner as core
from research_v2_e5_confirmation.e5_confirmation_runner import ConfirmationData


ROOT = Path(__file__).resolve().parents[2]
LEGALIR = ROOT.parent / "LegalIR"
BUNDLE = ROOT / "cache/research_v2_e5_confirmation/bundle-v1"
OUT = ROOT / "results/research_v2_lal_transfer"
CACHE = ROOT / "cache/research_v2_lal_transfer"
PREREG = OUT / "V2_LAL_TASK_ADAPTATION_PREREGISTRATION.json"
LAL_MATRIX = LEGALIR / "cache/exp112_task_adaptive_retrieval/lal.f16.npy"
LAL_MATRIX_RECEIPT = LEGALIR / "cache/exp112_task_adaptive_retrieval/lal.f16.json"
LAL_QUERY_CACHE = LEGALIR / "cache/exp109b_encoder_complementarity/embeddings/vnlegal_lal/queries.npz"
LAL_EMBED_MANIFEST = LEGALIR / "cache/exp109b_encoder_complementarity/embeddings/vnlegal_lal/manifest.json"
LAL_SUCCESS = LEGALIR / "cache/exp109b_encoder_complementarity/embeddings/vnlegal_lal/_SUCCESS.json"
SOURCE_DB = LEGALIR / "cache/exp112_task_adaptive_retrieval/sources.sqlite"
CURRENT_RRF_PREDICTIONS = ROOT / "results/research_v2_post_e5/V2_ADAPTED_E5_LAL_EQUAL_RRF32_PREDICTIONS.jsonl"

MODEL_ID = "darklethelong/vnlegal-lal"
QUERY_PREFIX = "Instruct: Given a Vietnamese legal question, retrieve relevant legal passages that answer the question\nQuery: "
MAX_LENGTH = 2048
LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
EPOCHS = 2
EFFECTIVE_BATCH = 16
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.01
DRIFT_WEIGHT = 0.05
BOUNDARY_WEIGHT = 0.25
GRADIENT_CLIP = 1.0
SEED = 112
RRF_K = 32
NATIVE_DEPTH = 500


def model_snapshot() -> Path:
    base = Path("C:/Users/nguye/.cache/huggingface/hub/models--darklethelong--vnlegal-lal")
    ref = base / "refs/main"
    candidates = [base / "snapshots" / ref.read_text(encoding="utf-8").strip()] if ref.exists() else []
    candidates.extend(sorted((base / "snapshots").glob("*")))
    for candidate in candidates:
        if (candidate / "config.json").is_file() and (candidate / "model.safetensors").is_file():
            return candidate
    raise FileNotFoundError(f"local LAL snapshot unavailable: {base}")


def jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


class LALData(ConfirmationData):
    def __init__(self, held_fold: str):
        super().__init__(BUNDLE, held_fold)
        if core.sha256(LAL_MATRIX) != core.read_json(LAL_MATRIX_RECEIPT)["sha256"]:
            raise RuntimeError("LAL materialized bank hash mismatch")
        manifest = core.read_json(LAL_EMBED_MANIFEST)
        success = core.read_json(LAL_SUCCESS)
        if success.get("status") != "PASS" or success.get("manifest_fingerprint") != manifest.get("manifest_fingerprint"):
            raise RuntimeError("LAL source encoding success/manifest mismatch")
        if int(manifest["chunks"]) != len(self.chunk_ids) or manifest["contract"]["pooling"] != "last_non_padding":
            raise RuntimeError("LAL bank cardinality or pooling mismatch")
        self.lal_manifest = manifest
        self.vectors = np.load(LAL_MATRIX, mmap_mode="r")
        if self.vectors.shape != (343347, 1024) or self.vectors.dtype != np.float16:
            raise RuntimeError(f"unexpected LAL matrix: {self.vectors.shape}/{self.vectors.dtype}")
        with np.load(LAL_QUERY_CACHE, allow_pickle=False) as archive:
            ids = [str(value) for value in archive["query_ids"].tolist()]
            vectors = np.array(archive["vectors"], dtype=np.float32)
            fingerprint = str(archive["manifest_fingerprint"][0])
        if fingerprint != manifest["manifest_fingerprint"] or len(ids) != len(set(ids)):
            raise RuntimeError("LAL query cache provenance mismatch")
        self.lal_query_row = {qid: index for index, qid in enumerate(ids)}
        self.lal_query_vectors = vectors
        if set(self.questions) - set(self.lal_query_row):
            raise RuntimeError("LAL query cache does not cover V2")
        self.fingerprint = core.digest([
            self.fingerprint,
            core.sha256(LAL_MATRIX),
            core.sha256(LAL_QUERY_CACHE),
            manifest["manifest_fingerprint"],
            held_fold,
        ])

    def query_vector(self, qid: str) -> np.ndarray:
        vector = np.array(self.lal_query_vectors[self.lal_query_row[qid]], dtype=np.float32)
        return vector / max(float(np.linalg.norm(vector)), 1e-12)


class LALQueryEncoder(nn.Module):
    def __init__(self, checkpoint_path: Path | None = None, device: str = "cuda"):
        super().__init__()
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModel, AutoTokenizer

        snapshot = model_snapshot()
        self.tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True, use_fast=True)
        base = AutoModel.from_pretrained(str(snapshot), local_files_only=True, dtype=torch.float16)
        self.model = get_peft_model(base, LoraConfig(
            r=LORA_RANK,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            target_modules=["q_proj", "v_proj"],
            bias="none",
        ))
        for parameter in self.model.parameters():
            if parameter.requires_grad:
                parameter.data = parameter.data.float()
        self.model.config.use_cache = False
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.model.enable_input_require_grads()
        self.to(device)
        if checkpoint_path is not None:
            self.load_adapter(checkpoint_path)

    def forward(self, texts: list[str]) -> torch.Tensor:
        batch = self.tokenizer(
            [QUERY_PREFIX + text for text in texts],
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        ).to(next(self.parameters()).device)
        hidden = self.model(**batch).last_hidden_state.float()
        mask = batch["attention_mask"].bool()
        if (~mask.any(dim=1)).any():
            raise RuntimeError("cannot last-token pool all-padding query")
        positions = mask.shape[1] - 1 - torch.flip(mask, dims=[1]).long().argmax(dim=1)
        pooled = hidden[torch.arange(hidden.shape[0], device=hidden.device), positions]
        return F.normalize(pooled, dim=-1)

    def adapter_state(self) -> dict[str, torch.Tensor]:
        return {name: value.detach().cpu() for name, value in self.named_parameters() if value.requires_grad}

    def load_adapter(self, path: Path) -> None:
        value = torch.load(path, map_location="cpu", weights_only=False)
        self.load_state_dict(value["adapter"], strict=False)

    @contextlib.contextmanager
    def adapter_disabled(self):
        with self.model.disable_adapter():
            yield


def native_order(scores: torch.Tensor, data: LALData, depth: int = NATIVE_DEPTH) -> list[str]:
    order = torch.argsort(scores, descending=True, stable=True)[:depth].detach().cpu().tolist()
    return [data.doc_ids[index] for index in order]


def pool_order(native: list[str], pool: list[str]) -> list[str]:
    membership = set(pool)
    result = [doc_id for doc_id in native if doc_id in membership]
    if len(result) != len(set(result)):
        raise RuntimeError("duplicate LAL pool ranking")
    return result


def source_rows(qids: list[str]) -> dict[str, list[dict[str, Any]]]:
    connection = sqlite3.connect(f"file:{SOURCE_DB.as_posix()}?mode=ro", uri=True)
    connection.execute("PRAGMA query_only=ON")
    if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
        raise RuntimeError("source database integrity failure")
    result = {}
    for qid in qids:
        row = connection.execute(
            "SELECT payload FROM sources WHERE q=? AND source='lal'", (qid,)
        ).fetchone()
        if row is None:
            raise RuntimeError(f"missing frozen LAL source {qid}")
        result[qid] = json.loads(row[0])
    connection.close()
    return result


def adapted_e5_rows(held_fold: str) -> dict[str, dict[str, Any]]:
    if held_fold == "fold_0":
        path = ROOT / "results/research_v2_e5_transfer/research_v2_e5_transfer_fold0/score/E5_TRANSFER_FOLD0_PREDICTIONS.jsonl"
    else:
        path = ROOT / f"results/research_v2_e5_confirmation/{held_fold}/score/E5_CONFIRMATION_{held_fold.upper()}_PREDICTIONS.jsonl"
    return {str(row["qid"]): row for row in jsonl(path)}


def rrf(left: list[str], right: list[str], universe: list[str]) -> list[str]:
    lr = {doc_id: index for index, doc_id in enumerate(left, 1)}
    rr = {doc_id: index for index, doc_id in enumerate(right, 1)}
    scores = {
        doc_id: (1.0 / (RRF_K + lr[doc_id]) if doc_id in lr else 0.0)
        + (1.0 / (RRF_K + rr[doc_id]) if doc_id in rr else 0.0)
        for doc_id in universe
    }
    return sorted(universe, key=lambda doc_id: (-scores[doc_id], doc_id))


def training_contract(data: LALData, qids: list[str], microbatch: int) -> dict[str, Any]:
    return {
        "data_fingerprint": data.fingerprint,
        "held_fold": data.held_fold,
        "qids": qids,
        "epochs": EPOCHS,
        "microbatch": microbatch,
        "effective_batch": EFFECTIVE_BATCH,
        "mechanism": "native_lal_query_only_exp112_v1",
        "model_id": MODEL_ID,
        "pooling": "last_non_padding_l2",
        "max_length": MAX_LENGTH,
        "lora": {"r": LORA_RANK, "alpha": LORA_ALPHA, "dropout": LORA_DROPOUT, "targets": ["q_proj", "v_proj"]},
        "loss": {"temperature": 0.05, "drift": DRIFT_WEIGHT, "epoch2_boundary": BOUNDARY_WEIGHT},
        "negative_policy": "exact_exp112_64",
        "optimizer": {"name": "AdamW", "lr": LEARNING_RATE, "weight_decay": WEIGHT_DECAY},
        "seed": SEED,
        "runner_sha256": core.sha256(Path(__file__)),
        "preregistration_sha256": core.sha256(PREREG),
    }


def train(data: LALData, output: Path, microbatch: int = 4, qids: list[str] | None = None,
          stop_after_updates: int | None = None) -> dict[str, Any]:
    from transformers import get_cosine_schedule_with_warmup

    output.mkdir(parents=True, exist_ok=True)
    allowed = set(data.training_qids())
    qids = list(qids or data.training_qids())
    if set(qids) - allowed:
        raise RuntimeError("training qids violate held-fold isolation")
    contract = training_contract(data, qids, microbatch)
    contract_hash = core.digest(contract)
    done = output / "_SUCCESS.json"
    if done.exists():
        result = core.read_json(done)
        if result["contract_hash"] != contract_hash:
            raise RuntimeError("completed training contract mismatch")
        for name, expected in result["checkpoint_sha256"].items():
            if core.sha256(output / name) != expected:
                raise RuntimeError("completed checkpoint hash mismatch")
        return result

    core.seed_all(SEED)
    model = LALQueryEncoder()
    bank = core.ParentBank(data.vectors, data.parent)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    steps_per_epoch = math.ceil(len(qids) / EFFECTIVE_BATCH)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        max(1, int(0.1 * steps_per_epoch * EPOCHS)),
        steps_per_epoch * EPOCHS,
    )
    resume = output / "resume.pt"
    start_epoch = start_position = updates = 0
    if resume.exists():
        receipt = resume.with_suffix(".sha.json")
        if not receipt.exists() or core.read_json(receipt)["sha256"] != core.sha256(resume):
            raise RuntimeError("resume hash mismatch")
        state = torch.load(resume, map_location="cpu", weights_only=False)
        if state["contract_hash"] != contract_hash:
            raise RuntimeError("resume contract mismatch")
        model.load_state_dict(state["adapter"], strict=False)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        core.set_rng(state["rng"])
        start_epoch, start_position, updates = int(state["epoch"]), int(state["position"]), int(state["updates"])

    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    try:
        for epoch in range(start_epoch, EPOCHS):
            order = list(qids)
            random.Random(SEED + epoch).shuffle(order)
            model.train()
            begin = start_position if epoch == start_epoch else 0
            for position in range(begin, len(order), EFFECTIVE_BATCH):
                group = order[position:position + EFFECTIVE_BATCH]
                optimizer.zero_grad(set_to_none=True)
                loss_value = 0.0
                for offset in range(0, len(group), microbatch):
                    ids = group[offset:offset + microbatch]
                    queries = model([data.questions[qid] for qid in ids])
                    parent_scores, indices = bank.mine(queries)
                    ordering = torch.argsort(parent_scores, dim=1, descending=True, stable=True)
                    ranks = torch.argsort(ordering, dim=1, stable=True) + 1
                    losses = []
                    for row, qid in enumerate(ids):
                        current = [data.doc_ids[index] for index in ordering[row].detach().cpu().tolist()]
                        negatives = core.select_negatives(
                            current, data.gold[qid], data.sources[qid], data.doc_ids, qid, epoch,
                        )
                        positives = sorted(data.gold[qid])
                        if set(negatives) & set(positives) or len(negatives) != 64 or len(set(negatives)) != 64:
                            raise RuntimeError("negative safety failure")
                        pi = torch.tensor([data.doc_row[doc_id] for doc_id in positives], device="cuda")
                        ni = torch.tensor([data.doc_row[doc_id] for doc_id in negatives], device="cuda")
                        positive_scores = bank.rescore(queries[row], indices[row, pi])
                        negative_scores = bank.rescore(queries[row], indices[row, ni])
                        loss = core.multi_loss(positive_scores, negative_scores)
                        if epoch > 0:
                            loss = loss + BOUNDARY_WEIGHT * core.boundary_loss(
                                positive_scores, negative_scores, ranks[row, pi], ranks[row, ni]
                            )
                        frozen = torch.as_tensor(data.query_vector(qid), device="cuda")
                        loss = loss + DRIFT_WEIGHT * (
                            1 - F.cosine_similarity(queries[row:row + 1], frozen[None]).mean()
                        )
                        losses.append(loss)
                    batch_loss = torch.stack(losses).sum() / len(group)
                    if not torch.isfinite(batch_loss):
                        raise FloatingPointError("non-finite LAL loss")
                    batch_loss.backward()
                    loss_value += float(batch_loss.detach())
                gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP, error_if_nonfinite=True)
                if float(gradient_norm) == 0.0:
                    raise RuntimeError("LAL adapter received no gradient")
                optimizer.step(); scheduler.step(); updates += 1
                next_position = position + len(group)
                core.save_checkpoint(
                    resume, model, optimizer, scheduler, contract_hash=contract_hash,
                    epoch=epoch, position=next_position, updates=updates,
                )
                if updates % 16 == 0 or next_position == len(order):
                    print(json.dumps({
                        "stage": "train_lal", "held_fold": data.held_fold, "epoch": epoch + 1,
                        "position": next_position, "queries": len(order), "updates": updates,
                        "loss": loss_value, "gradient_norm": float(gradient_norm),
                    }), flush=True)
                if stop_after_updates is not None and updates >= stop_after_updates:
                    return {"status": "INTERRUPTED_FOR_RESUME_TEST", "contract_hash": contract_hash, "updates": updates}
            core.save_checkpoint(
                output / f"epoch-{epoch + 1}.pt", model, optimizer, scheduler,
                contract_hash=contract_hash, epoch=epoch + 1, position=0, updates=updates,
            )
        result = {
            "schema_version": "dsc2026.research_v2.lal_transfer_checkpoint.v1",
            "status": "COMPLETE_EPOCH2",
            "held_fold": data.held_fold,
            "training_queries": len(qids),
            "duplicate_exclusions": sorted(data.duplicate_exclusions, key=int),
            "contract_hash": contract_hash,
            "scientific_contract": contract,
            "epochs": EPOCHS,
            "updates": updates,
            "runtime_seconds_this_process": time.monotonic() - started,
            "peak_allocated_mib": torch.cuda.max_memory_allocated() / (1 << 20),
            "peak_reserved_mib": torch.cuda.max_memory_reserved() / (1 << 20),
            "checkpoint_sha256": {f"epoch-{epoch}.pt": core.sha256(output / f"epoch-{epoch}.pt") for epoch in (1, 2)},
        }
        core.write_json(done, result)
        return result
    finally:
        del model, bank, optimizer, scheduler
        gc.collect(); torch.cuda.empty_cache()


def audit(data: LALData) -> dict[str, Any]:
    training = data.training_qids()
    held = data.fold_qids(data.held_fold)
    v2_chunks = BUNDLE / "chunk_ids.jsonl"
    legalir_chunks = LEGALIR / "cache/e5_final_v1/chunk_ids.jsonl"
    report = {
        "schema_version": "dsc2026.research_v2.lal_transfer_data_audit.v1",
        "status": "PASS",
        "held_fold": data.held_fold,
        "held_queries": len(held),
        "training_queries": len(training),
        "duplicate_exclusions": sorted(data.duplicate_exclusions, key=int),
        "held_training_intersection": sorted(set(held) & set(training), key=int),
        "excluded_training_intersection": sorted(set(training) & data.duplicate_exclusions, key=int),
        "training_qids_sha256": core.digest(sorted(training, key=int)),
        "chunk_ids_sha256": core.sha256(v2_chunks),
        "legalir_chunk_ids_sha256": core.sha256(legalir_chunks),
        "chunk_order_exact": core.sha256(v2_chunks) == core.sha256(legalir_chunks),
        "lal_matrix_sha256": core.sha256(LAL_MATRIX),
        "lal_query_cache_sha256": core.sha256(LAL_QUERY_CACHE),
        "folds_sha256": core.sha256(BUNDLE / "V2_FOLDS.json"),
        "candidate_pool_sha256": core.sha256(BUNDLE / "V2_CANDIDATE_POOL.jsonl"),
        "source_database_sha256": core.sha256(SOURCE_DB),
        "runner_sha256": core.sha256(Path(__file__)),
    }
    if report["held_training_intersection"] or report["excluded_training_intersection"] or not report["chunk_order_exact"]:
        report["status"] = "FAIL"
        raise RuntimeError(f"LAL data audit failed: {report}")
    core.write_json(OUT / data.held_fold / "DATA_AUDIT.json", report)
    return report


def parity(data: LALData, output: Path) -> dict[str, Any]:
    qids = data.fold_qids(data.held_fold)
    expected = source_rows(qids)
    bank = core.ParentBank(data.vectors, data.parent)
    cached_score_errors: list[float] = []
    top50_disagreement_reference_gaps: list[float] = []
    cached_top5_exact = cached_top50_exact = 0
    started = time.monotonic()
    try:
        for index, qid in enumerate(qids, 1):
            query = torch.as_tensor(data.query_vector(qid), device="cuda")
            scores, _ = bank.mine(query)
            actual = native_order(scores[0], data)
            reference = [str(row["doc_id"]) for row in expected[qid]]
            reference_scores = {str(row["doc_id"]): float(row["score"]) for row in expected[qid]}
            cached_top5_exact += actual[:5] == reference[:5]
            cached_top50_exact += actual[:50] == reference[:50]
            if actual[:50] != reference[:50]:
                for actual_doc, reference_doc in zip(actual[:50], reference[:50]):
                    if actual_doc != reference_doc:
                        top50_disagreement_reference_gaps.append(
                            abs(reference_scores[actual_doc] - reference_scores[reference_doc])
                        )
            for doc_id in reference[:50]:
                cached_score_errors.append(abs(float(scores[0, data.doc_row[doc_id]].cpu()) - reference_scores[doc_id]))
            if index % 200 == 0:
                print(json.dumps({"stage": "lal_cached_parity", "completed": index, "total": len(qids)}), flush=True)
    finally:
        del bank
        gc.collect(); torch.cuda.empty_cache()

    sample = [qids[index] for index in np.linspace(0, len(qids) - 1, 32, dtype=int)]
    core.seed_all(SEED)
    model = LALQueryEncoder()
    model.eval()
    cosines: list[float] = []
    max_abs: list[float] = []
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    try:
        for start in range(0, len(sample), 4):
            ids = sample[start:start + 4]
            with torch.no_grad(), model.adapter_disabled():
                values = model([data.questions[qid] for qid in ids]).cpu().numpy()
            for qid, value in zip(ids, values):
                cached = data.query_vector(qid)
                cosines.append(float(value @ cached))
                max_abs.append(float(np.max(np.abs(value - cached))))
    finally:
        del model
        gc.collect(); torch.cuda.empty_cache()

    checks = {
        "cached_top5_exact_all": cached_top5_exact == len(qids),
        "cached_top50_disagreements_within_1e_5": max(top50_disagreement_reference_gaps, default=0.0) <= 1e-5,
        "cached_score_max_abs_lte_1e_5": max(cached_score_errors, default=0.0) <= 1e-5,
        "model_identity_min_cosine_gte_0_9999": min(cosines) >= 0.9999,
        "trainable_parameters_exact": trainable == 2293760,
        "total_parameters_exact": total == 598343680,
    }
    report = {
        "schema_version": "dsc2026.research_v2.lal_transfer_frozen_parity.v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "held_fold": data.held_fold,
        "queries": len(qids),
        "cached_top5_exact": cached_top5_exact,
        "cached_top50_exact": cached_top50_exact,
        "cached_top50_nonexact": len(qids) - cached_top50_exact,
        "cached_top50_disagreement_max_reference_score_gap": max(top50_disagreement_reference_gaps, default=0.0),
        "cached_score_max_abs_error": max(cached_score_errors, default=0.0),
        "cached_score_mean_abs_error": mean(cached_score_errors),
        "model_identity_sample": sample,
        "model_identity_min_cosine": min(cosines),
        "model_identity_mean_cosine": mean(cosines),
        "model_identity_max_abs_vector_error": max(max_abs),
        "trainable_parameters": trainable,
        "total_parameters": total,
        "runtime_seconds": time.monotonic() - started,
        "checks": checks,
    }
    output.mkdir(parents=True, exist_ok=True)
    core.write_json(output / "FROZEN_LAL_PARITY.json", report)
    if report["status"] != "PASS":
        raise RuntimeError(f"frozen LAL parity failed: {report}")
    return report


def smoke(data: LALData, output: Path, microbatch: int = 4) -> dict[str, Any]:
    qids = data.training_qids()[:16]
    continuous = output / "continuous"
    resumed = output / "resumed"
    left_result = train(data, continuous, microbatch=microbatch, qids=qids)
    if not (resumed / "_SUCCESS.json").exists():
        train(data, resumed, microbatch=microbatch, qids=qids, stop_after_updates=1)
    right_result = train(data, resumed, microbatch=microbatch, qids=qids)
    left = torch.load(continuous / "epoch-2.pt", map_location="cpu", weights_only=False)["adapter"]
    right = torch.load(resumed / "epoch-2.pt", map_location="cpu", weights_only=False)["adapter"]
    exact = left.keys() == right.keys() and all(torch.equal(left[key], right[key]) for key in left)
    errors = [float((left[key] - right[key]).abs().max()) for key in left]
    report = {
        "schema_version": "dsc2026.research_v2.lal_transfer_smoke.v1",
        "status": "PASS" if exact else "FAIL",
        "queries": qids,
        "forward_backward_complete": True,
        "resume_tensor_exact": exact,
        "max_tensor_abs_error": max(errors, default=0.0),
        "continuous_peak_reserved_mib": left_result["peak_reserved_mib"],
        "resumed_peak_reserved_mib": right_result["peak_reserved_mib"],
        "continuous_epoch2_sha256": core.sha256(continuous / "epoch-2.pt"),
        "resumed_epoch2_sha256": core.sha256(resumed / "epoch-2.pt"),
    }
    output.mkdir(parents=True, exist_ok=True)
    core.write_json(output / "TRAINING_SMOKE_RESUME_PARITY.json", report)
    if not exact:
        raise RuntimeError(f"smoke/resume parity failed: {report}")
    return report


def slice_metrics(rows: list[dict[str, Any]], base_key: str, new_key: str) -> dict[str, Any]:
    return {
        "queries": len(rows),
        "base": mean(row[base_key] for row in rows),
        "new": mean(row[new_key] for row in rows),
        "delta": mean(row[new_key] - row[base_key] for row in rows),
        "wins": sum(row[new_key] > row[base_key] for row in rows),
        "losses": sum(row[new_key] < row[base_key] for row in rows),
    }


def score(data: LALData, checkpoint: Path, output: Path) -> dict[str, Any]:
    qids = data.fold_qids(data.held_fold)
    frozen_sources = source_rows(qids)
    e5 = adapted_e5_rows(data.held_fold)
    current = {str(row["qid"]): row for row in jsonl(CURRENT_RRF_PREDICTIONS)}
    model = LALQueryEncoder(checkpoint)
    model.eval()
    bank = core.ParentBank(data.vectors, data.parent)
    rows: list[dict[str, Any]] = []
    lal_moves: Counter[str] = Counter()
    system_moves: Counter[str] = Counter()
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    try:
        for start in range(0, len(qids), 4):
            ids = qids[start:start + 4]
            with torch.no_grad():
                query_vectors = model([data.questions[qid] for qid in ids])
                parent_scores, _ = bank.mine(query_vectors)
            for row_index, qid in enumerate(ids):
                pool = data.pool[qid]
                pool_set = set(pool)
                frozen_native = [str(row["doc_id"]) for row in frozen_sources[qid]]
                adapted_native = native_order(parent_scores[row_index], data)
                frozen_lal = pool_order(frozen_native, pool)
                adapted_lal = pool_order(adapted_native, pool)
                e5_order = [str(value) for value in e5[qid]["ft_order"]]
                if set(e5_order) != pool_set:
                    raise RuntimeError(f"adapted E5 pool mismatch: {qid}")
                base_system = rrf(e5_order, frozen_lal, pool)
                new_system = rrf(e5_order, adapted_lal, pool)
                locked = current[qid]
                if locked["fold"] != data.held_fold or base_system[:5] != list(map(str, locked["fused_top5"])):
                    raise RuntimeError(f"current RRF endpoint parity failed: {qid}")
                gold = data.gold[qid]
                frozen_hits = len(gold & set(frozen_lal[:5]))
                adapted_hits = len(gold & set(adapted_lal[:5]))
                base_hits = len(gold & set(base_system[:5]))
                new_hits = len(gold & set(new_system[:5]))
                for doc_id in gold:
                    fb = frozen_lal.index(doc_id) + 1 if doc_id in frozen_lal else None
                    ab = adapted_lal.index(doc_id) + 1 if doc_id in adapted_lal else None
                    sb = base_system.index(doc_id) + 1 if doc_id in pool_set else None
                    sn = new_system.index(doc_id) + 1 if doc_id in pool_set else None
                    lal_moves[f"{core.rank_bucket(fb)}->{core.rank_bucket(ab)}"] += 1
                    system_moves[f"{core.rank_bucket(sb)}->{core.rank_bucket(sn)}"] += 1
                rows.append({
                    "qid": qid,
                    "fold": data.held_fold,
                    "gold": sorted(gold),
                    "frozen_lal_order": frozen_lal,
                    "adapted_lal_order": adapted_lal,
                    "adapted_e5_order": e5_order,
                    "base_system_top5": base_system[:5],
                    "new_system_top5": new_system[:5],
                    "frozen_lal_recall_at_5": frozen_hits / len(gold),
                    "adapted_lal_recall_at_5": adapted_hits / len(gold),
                    "base_system_recall_at_5": base_hits / len(gold),
                    "new_system_recall_at_5": new_hits / len(gold),
                    "frozen_lal_hits": frozen_hits,
                    "adapted_lal_hits": adapted_hits,
                    "base_system_hits": base_hits,
                    "new_system_hits": new_hits,
                    "adapted_query_cosine_to_frozen": float(query_vectors[row_index].detach().cpu().numpy() @ data.query_vector(qid)),
                })
            print(json.dumps({"stage": "score_lal", "held_fold": data.held_fold, "completed": min(start + len(ids), len(qids)), "total": len(qids)}), flush=True)
    finally:
        del model, bank
        gc.collect(); torch.cuda.empty_cache()

    output.mkdir(parents=True, exist_ok=True)
    predictions = output / f"LAL_TRANSFER_{data.held_fold.upper()}_PREDICTIONS.jsonl"
    with predictions.open("w", encoding="utf-8", newline="\n") as sink:
        for row in rows:
            sink.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")

    single = [row for row in rows if len(row["gold"]) == 1]
    multi = [row for row in rows if len(row["gold"]) > 1]
    expert = slice_metrics(rows, "frozen_lal_recall_at_5", "adapted_lal_recall_at_5")
    expert["single_gold"] = slice_metrics(single, "frozen_lal_recall_at_5", "adapted_lal_recall_at_5")
    expert["multi_gold"] = slice_metrics(multi, "frozen_lal_recall_at_5", "adapted_lal_recall_at_5")
    expert["changed_top5_sets"] = sum(set(row["frozen_lal_order"][:5]) != set(row["adapted_lal_order"][:5]) for row in rows)
    expert["gold_crossings_into_top5"] = sum(value for key, value in lal_moves.items() if not key.startswith("1-5->") and key.endswith("->1-5"))
    expert["gold_crossings_out_of_top5"] = sum(value for key, value in lal_moves.items() if key.startswith("1-5->") and not key.endswith("->1-5"))
    expert["rank_movements"] = dict(sorted(lal_moves.items()))
    system = slice_metrics(rows, "base_system_recall_at_5", "new_system_recall_at_5")
    system["single_gold"] = slice_metrics(single, "base_system_recall_at_5", "new_system_recall_at_5")
    system["multi_gold"] = slice_metrics(multi, "base_system_recall_at_5", "new_system_recall_at_5")
    system["changed_top5_sets"] = sum(set(row["base_system_top5"]) != set(row["new_system_top5"]) for row in rows)
    system["gold_crossings_into_top5"] = sum(value for key, value in system_moves.items() if not key.startswith("1-5->") and key.endswith("->1-5"))
    system["gold_crossings_out_of_top5"] = sum(value for key, value in system_moves.items() if key.startswith("1-5->") and not key.endswith("->1-5"))
    system["rank_movements"] = dict(sorted(system_moves.items()))

    parity_file = OUT / data.held_fold / "preflight/FROZEN_LAL_PARITY.json"
    smoke_file = OUT / data.held_fold / "preflight/TRAINING_SMOKE_RESUME_PARITY.json"
    data_file = OUT / data.held_fold / "DATA_AUDIT.json"
    integrity = all(core.read_json(path)["status"] == "PASS" for path in (parity_file, smoke_file, data_file))
    expert_checks = {
        "delta_gte_0_005": expert["delta"] >= 0.005,
        "wins_exceed_losses": expert["wins"] > expert["losses"],
        "crossings_in_exceed_out": expert["gold_crossings_into_top5"] > expert["gold_crossings_out_of_top5"],
        "multi_gold_delta_gte_minus_0_005": expert["multi_gold"]["delta"] >= -0.005,
        "integrity_pass": integrity,
    }
    system_checks = {
        "delta_gte_0_003": system["delta"] >= 0.003,
        "multi_gold_delta_gte_minus_0_005": system["multi_gold"]["delta"] >= -0.005,
        "current_endpoint_exact_parity": True,
    }
    both = all(expert_checks.values()) and all(system_checks.values())
    if both:
        verdict = "PASS_BOTH_GATES"
    elif all(expert_checks.values()):
        verdict = "EXPERT_IMPROVES_BUT_COMPLEMENTARITY_COLLAPSES"
    else:
        verdict = "KILL_FOLD0_NO_CONFIRMATION"
    report = {
        "schema_version": "dsc2026.research_v2.lal_transfer_fold.v1",
        "status": verdict,
        "held_fold": data.held_fold,
        "queries": len(rows),
        "expert": expert,
        "system_endpoint": system,
        "expert_gate_checks": expert_checks,
        "system_gate_checks": system_checks,
        "query_drift_cosine_mean": mean(row["adapted_query_cosine_to_frozen"] for row in rows),
        "runtime_seconds": time.monotonic() - started,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / (1 << 20),
        "checkpoint_sha256": core.sha256(checkpoint),
        "predictions_sha256": core.sha256(predictions),
        "folds_sha256": core.sha256(BUNDLE / "V2_FOLDS.json"),
        "candidate_pool_sha256": core.sha256(BUNDLE / "V2_CANDIDATE_POOL.jsonl"),
        "preregistration_sha256": core.sha256(PREREG),
        "runner_sha256": core.sha256(Path(__file__)),
        "anti_rescue": "No epoch/rank/LR/mining/aggregation/seed/RRF weight/K/subset tuning.",
    }
    core.write_json(output / f"LAL_TRANSFER_{data.held_fold.upper()}_REPORT.json", report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("audit", "parity", "smoke", "train", "score"))
    parser.add_argument("--held-fold", choices=[f"fold_{index}" for index in range(5)], default="fold_0")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--microbatch", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if core.read_json(PREREG)["status"] != "PREREGISTERED_BEFORE_V2_METRICS":
        raise RuntimeError("LAL preregistration is not sealed")
    data = LALData(args.held_fold)
    output = args.output or (OUT / args.held_fold)
    output.mkdir(parents=True, exist_ok=True)
    if args.command == "audit":
        result = audit(data)
    elif args.command == "parity":
        result = parity(data, output)
    elif args.command == "smoke":
        result = smoke(data, output, args.microbatch)
    elif args.command == "train":
        result = train(data, output, args.microbatch)
    else:
        if args.checkpoint is None or not args.checkpoint.is_file():
            raise ValueError("score requires --checkpoint")
        result = score(data, args.checkpoint, output)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2), flush=True)


if __name__ == "__main__":
    main()
