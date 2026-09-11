"""Exact-mechanism EXP-112 query-adapter transfer runner.

The learning mechanism intentionally mirrors the sealed EXP-112 source.  The
only scientific change is downstream evaluation on Research V2's immutable
candidate pool.  Commands are fail-closed and Fold-0 only.
"""

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
import time
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


SEED = 112
TEMPERATURE = 0.05
LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
MAX_LENGTH = 512
EPOCHS = 2
EFFECTIVE_BATCH_QUERIES = 16
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.01
GRADIENT_CLIP = 1.0
DRIFT_WEIGHT = 0.05
BOUNDARY_WEIGHT = 0.25
NEGATIVE_COUNT = 64


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def records(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def seed_all(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def set_rng(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def multi_loss(positives: torch.Tensor, negatives: torch.Tensor, temperature: float = TEMPERATURE) -> torch.Tensor:
    if not len(positives) or not len(negatives):
        raise ValueError("Both positive and negative scores required")
    p, n = positives / temperature, negatives / temperature
    return (torch.logaddexp(p, torch.logsumexp(n, dim=0)) - p).mean()


def boundary_loss(
    positives: torch.Tensor,
    negatives: torch.Tensor,
    positive_ranks: torch.Tensor,
    negative_ranks: torch.Tensor,
    temperature: float = TEMPERATURE,
) -> torch.Tensor:
    mask = (positive_ranks[:, None] <= 5) != (negative_ranks[None, :] <= 5)
    terms = F.softplus((negatives[None, :] - positives[:, None]) / temperature) * mask
    return (terms.sum(1) / mask.sum(1).clamp_min(1)).mean()


def select_negatives_with_trace(
    current: list[str],
    gold: set[str],
    sources: dict[str, list[str]],
    universe: list[str],
    qid: str,
    epoch: int,
    count: int = NEGATIVE_COUNT,
) -> tuple[list[str], list[str]]:
    """Byte-for-behaviour EXP-112 selection plus non-causal trace labels."""
    rng = random.Random(int(digest([qid, epoch, SEED])[:16], 16))
    blocked, selected, trace = set(gold), [], []

    def add(values: Iterable[str], number: int, label: str) -> None:
        found = 0
        for raw_doc_id in values:
            doc_id = str(raw_doc_id)
            if doc_id not in blocked:
                selected.append(doc_id)
                trace.append(label)
                blocked.add(doc_id)
                found += 1
                if found >= number:
                    break

    if count == 6:
        add(current, 2, "current_hardest")
        middle = list(current[8:32]); rng.shuffle(middle); add(middle, 2, "rotated_middle")
        add(sources.get("bm25", []) + sources.get("lal", []), 1, "sparse_lal")
        deep = list(current[32:]) or list(universe); rng.shuffle(deep); add(deep, 1, "random_deep")
    else:
        add(current, 16, "current_hardest")
        middle = list(current[16:100]); rng.shuffle(middle); add(middle, 16, "adapted_17_100_rotated")
        disagreement: list[str] = []
        base = set(current[:16])
        for items in zip(sources.get("bm25", []), sources.get("lal", [])):
            disagreement.extend(doc_id for doc_id in items if doc_id not in base)
        rng.shuffle(disagreement); add(disagreement, 16, "sparse_lal_disagreement")
        add(sources.get("jina", []) or sources.get("e5", []), 8, "jina_or_e5_confuser")
        randoms = list(universe); rng.shuffle(randoms); add(randoms, 8, "random")
    if len(selected) < count:
        add(current, count - len(selected), "backfill_current")
    if len(selected) < count:
        rest = list(universe); rng.shuffle(rest); add(rest, count - len(selected), "backfill_universe")
    if len(selected) != count:
        raise ValueError("Not enough unique negatives")
    return selected, trace


def select_negatives(*args: Any, **kwargs: Any) -> list[str]:
    return select_negatives_with_trace(*args, **kwargs)[0]


class TransferData:
    def __init__(self, bundle: Path):
        self.bundle = bundle.resolve()
        manifest = read_json(self.bundle / "E5_TRANSFER_INPUT_MANIFEST.json")
        if manifest["status"] not in {"STAGED_FOR_LOCAL_PARITY", "SEALED_FOLD0_ONLY"}:
            raise RuntimeError("bundle has an unsupported staging status")
        for relative, expected in manifest["files_sha256"].items():
            path = self.bundle / relative
            if not path.is_file() or sha256(path) != expected:
                raise RuntimeError(f"bundle hash mismatch: {relative}")
        self.manifest = manifest
        query_rows = list(records(self.bundle / "V2_TRANSFER_QUERIES.jsonl"))
        self.questions = {str(row["qid"]): str(row["question"]) for row in query_rows}
        self.gold = {str(row["qid"]): {str(value) for value in row["gold"]} for row in query_rows}
        self.fold_for = {str(row["qid"]): str(row["fold"]) for row in query_rows}
        if len(self.questions) != 6991 or any(not values for values in self.gold.values()):
            raise RuntimeError("V2 query population mismatch")
        self.pool = {str(row["qid"]): [str(value) for value in row["doc_ids"]]
                     for row in records(self.bundle / "V2_CANDIDATE_POOL.jsonl")}
        if set(self.pool) != set(self.questions):
            raise RuntimeError("candidate pool/query population mismatch")
        self.duplicate_exclusions = set(manifest["fold0_duplicate_exclusions"])
        self.chunk_ids, self.chunk_parents = [], []
        for row in records(self.bundle / "chunk_ids.jsonl"):
            self.chunk_ids.append(str(row["chunk_id"]))
            self.chunk_parents.append(str(row["doc_id"]))
        self.doc_ids = sorted(set(self.chunk_parents))
        if len(self.doc_ids) != 8507 or len(self.chunk_ids) != 343347:
            raise RuntimeError("frozen chunk bank cardinality mismatch")
        self.doc_row = {doc_id: index for index, doc_id in enumerate(self.doc_ids)}
        self.parent = np.array([self.doc_row[doc_id] for doc_id in self.chunk_parents], dtype=np.int64)
        position_lists: list[list[int]] = [[] for _ in self.doc_ids]
        for index, parent in enumerate(self.parent):
            position_lists[int(parent)].append(index)
        self.positions = [np.array(values, dtype=np.int64) for values in position_lists]
        self.vectors = np.load(self.bundle / "embeddings.f16.npy", mmap_mode="r")
        if self.vectors.shape != (343347, 1024):
            raise RuntimeError(f"unexpected chunk matrix shape: {self.vectors.shape}")
        query_ids = [str(value) for value in read_json(self.bundle / "train_query_ids.json")]
        self.query_vectors = np.load(self.bundle / "train_queries.f32.npy", mmap_mode="r")
        self.query_row = {qid: index for index, qid in enumerate(query_ids)}
        if len(self.query_row) != len(query_ids) or set(self.questions) - set(self.query_row):
            raise RuntimeError("frozen query vector cache does not cover the evaluable V2 population exactly once")
        self.sources = {
            str(row["qid"]): {key: [str(value) for value in row[key]]
                              for key in ("e5", "lal", "bm25", "jina")}
            for row in records(self.bundle / "V2_EXP112_MINER_SOURCES.jsonl")
        }
        self.frozen_reference = {
            str(row["qid"]): row for row in records(self.bundle / "V2_FOLD0_FROZEN_REFERENCE.jsonl")
        }
        expected_fold0 = {qid for qid, fold in self.fold_for.items() if fold == "fold_0"}
        if set(self.frozen_reference) != expected_fold0:
            raise RuntimeError("Fold-0 frozen reference mismatch")
        self.fingerprint = digest([
            manifest["v2_folds_sha256"], manifest["candidate_pool_sha256"],
            manifest["files_sha256"]["embeddings.f16.npy"],
            manifest["files_sha256"]["V2_EXP112_MINER_SOURCES.jsonl"],
        ])

    def fold_qids(self, fold: str) -> list[str]:
        return [qid for qid, assigned in self.fold_for.items() if assigned == fold]

    def training_qids(self) -> list[str]:
        qids = [qid for qid, fold in self.fold_for.items() if fold != "fold_0"]
        qids = [qid for qid in qids if qid not in self.duplicate_exclusions]
        if set(qids) & set(self.fold_qids("fold_0")) or set(qids) & self.duplicate_exclusions:
            raise RuntimeError("Fold-0 or duplicate-linked qid leaked into training")
        if len(qids) != 5586:
            raise RuntimeError(f"unexpected Fold-0 training population: {len(qids)}")
        return qids

    def query_vector(self, qid: str) -> np.ndarray:
        vector = np.array(self.query_vectors[self.query_row[qid]], dtype=np.float32)
        return vector / max(float(np.linalg.norm(vector)), 1e-12)


class ParentBank:
    """Exact EXP-112 full-corpus top2_mean bank for training/mining."""
    def __init__(self, vectors: np.ndarray, parent_indices: np.ndarray, device: str = "cuda"):
        self.vectors = torch.empty(tuple(vectors.shape), dtype=torch.float32, device=device)
        for start in range(0, len(self.vectors), 8192):
            block = self.vectors[start:start + 8192]
            block.copy_(torch.from_numpy(np.array(vectors[start:start + 8192], dtype=np.float32, copy=True)))
            block.copy_(F.normalize(block, dim=1))
        self.parent = torch.as_tensor(parent_indices, dtype=torch.long, device=device)
        self.count = int(self.parent.max()) + 1
        self.counts = torch.bincount(self.parent, minlength=self.count)
        self.chunk_ids = torch.arange(len(self.parent), device=device)

    @torch.no_grad()
    def mine(self, query: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scores = query.detach().float() @ self.vectors.T
        if scores.ndim == 1:
            scores = scores[None]
        batch = len(scores)
        ids = self.parent.expand(batch, -1)
        first = torch.full((batch, self.count), -torch.inf, device=scores.device)
        first.scatter_reduce_(1, ids, scores, reduce="amax", include_self=True)
        sentinel = len(self.parent)
        arg1 = torch.full((batch, self.count), sentinel, device=scores.device, dtype=torch.long)
        eligible = torch.where(scores == first.gather(1, ids), self.chunk_ids, sentinel)
        arg1.scatter_reduce_(1, ids, eligible, reduce="amin", include_self=True)
        rest = scores.masked_fill(self.chunk_ids[None] == arg1.gather(1, ids), -torch.inf)
        second = torch.full_like(first, -torch.inf)
        second.scatter_reduce_(1, ids, rest, reduce="amax", include_self=True)
        arg2 = torch.full_like(arg1, sentinel)
        eligible2 = torch.where(rest == second.gather(1, ids), self.chunk_ids, sentinel)
        arg2.scatter_reduce_(1, ids, eligible2, reduce="amin", include_self=True)
        singleton = self.counts == 1
        arg2[:, singleton] = arg1[:, singleton]
        result = torch.where(singleton, first, 0.5 * (first + second))
        return result, torch.stack((arg1, arg2), dim=-1)

    def rescore(self, query: torch.Tensor, selected_indices: torch.Tensor) -> torch.Tensor:
        return (self.vectors[selected_indices] * query.float()).sum(-1).mean(-1)

    @torch.no_grad()
    def score_pool(self, query: torch.Tensor, doc_ids: list[str], data: TransferData) -> list[float]:
        values: list[float] = []
        for doc_id in doc_ids:
            indices = torch.as_tensor(data.positions[data.doc_row[doc_id]], dtype=torch.long, device=query.device)
            scores = self.vectors[indices] @ query.float()
            count = min(2, len(scores))
            values.append(float(torch.topk(scores, k=count, sorted=False).values.mean().cpu()))
        return values


class QueryEncoder(nn.Module):
    def __init__(self, model_path: Path, device: str = "cuda", checkpoint_path: Path | None = None):
        super().__init__()
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
        base = AutoModel.from_pretrained(str(model_path), local_files_only=True, torch_dtype=torch.float32)
        self.model = get_peft_model(base, LoraConfig(
            r=LORA_RANK,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            target_modules=["query", "value"],
            bias="none",
        ))
        self.model.config.use_cache = False
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.model.enable_input_require_grads()
        self.to(device)
        if checkpoint_path is not None:
            self.load_adapter(checkpoint_path)

    def forward(self, texts: list[str]) -> torch.Tensor:
        batch = self.tokenizer(
            ["query: " + text for text in texts], padding=True, truncation=True,
            max_length=MAX_LENGTH, return_tensors="pt",
        ).to(next(self.parameters()).device)
        hidden = self.model(**batch).last_hidden_state.float()
        mask = batch["attention_mask"].unsqueeze(-1)
        return F.normalize((hidden * mask).sum(1) / mask.sum(1).clamp_min(1), dim=-1)

    def adapter_state(self) -> dict[str, torch.Tensor]:
        return {name: value.detach().cpu() for name, value in self.named_parameters() if value.requires_grad}

    def load_adapter(self, path: Path) -> None:
        value = torch.load(path, map_location="cpu", weights_only=False)
        self.load_state_dict(value["adapter"], strict=False)

    @contextlib.contextmanager
    def adapter_disabled(self):
        with self.model.disable_adapter():
            yield


def save_checkpoint(
    path: Path,
    model: QueryEncoder,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    **extra: Any,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save({
        "adapter": model.adapter_state(), "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(), "rng": rng_state(), **extra,
    }, temporary)
    os.replace(temporary, path)
    write_json(path.with_suffix(".sha.json"), {"sha256": sha256(path)})


def train_adapter(
    data: TransferData,
    output: Path,
    *,
    microbatch: int = 4,
    qids: list[str] | None = None,
    stop_after_updates: int | None = None,
) -> dict[str, Any]:
    from transformers import get_cosine_schedule_with_warmup

    output.mkdir(parents=True, exist_ok=True)
    qids = list(qids or data.training_qids())
    if set(qids) - set(data.training_qids()):
        raise RuntimeError("training qids violate Fold-0 isolation")
    contract = {
        "data": data.fingerprint,
        "qids": qids,
        "epochs": EPOCHS,
        "microbatch": microbatch,
        "mechanism": "exact_exp112_query_only_v1",
        "runner_sha256": sha256(Path(__file__)),
    }
    contract_hash = digest(contract)
    done = output / "_SUCCESS.json"
    if done.exists():
        result = read_json(done)
        if result["contract_hash"] != contract_hash:
            raise RuntimeError("completed training contract mismatch")
        return result

    seed_all()
    model = QueryEncoder(data.bundle / "vietlegal-e5", checkpoint_path=None)
    bank = ParentBank(data.vectors, data.parent)
    optimizer = torch.optim.AdamW(
        [value for value in model.parameters() if value.requires_grad],
        lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
    )
    steps_per_epoch = math.ceil(len(qids) / EFFECTIVE_BATCH_QUERIES)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        max(1, int(0.1 * steps_per_epoch * EPOCHS)),
        steps_per_epoch * EPOCHS,
    )
    resume_path = output / "resume.pt"
    start_epoch, start_position, updates = 0, 0, 0
    if resume_path.exists():
        receipt = resume_path.with_suffix(".sha.json")
        if not receipt.exists() or read_json(receipt)["sha256"] != sha256(resume_path):
            raise RuntimeError("resume checkpoint hash mismatch")
        state = torch.load(resume_path, map_location="cpu", weights_only=False)
        if state["contract_hash"] != contract_hash:
            raise RuntimeError("resume training contract mismatch")
        model.load_state_dict(state["adapter"], strict=False)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        set_rng(state["rng"])
        start_epoch = int(state["epoch"])
        start_position = int(state["position"])
        updates = int(state["updates"])

    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    try:
        for epoch in range(start_epoch, EPOCHS):
            order = list(qids)
            random.Random(SEED + epoch).shuffle(order)
            model.train()
            begin = start_position if epoch == start_epoch else 0
            for position in range(begin, len(order), EFFECTIVE_BATCH_QUERIES):
                group = order[position:position + EFFECTIVE_BATCH_QUERIES]
                optimizer.zero_grad(set_to_none=True)
                for start in range(0, len(group), microbatch):
                    ids = group[start:start + microbatch]
                    query_vectors = model([data.questions[qid] for qid in ids])
                    scores, indices = bank.mine(query_vectors)
                    ordering = torch.argsort(scores, dim=1, descending=True, stable=True)
                    ranks = torch.argsort(ordering, dim=1, stable=True) + 1
                    local_losses = []
                    for batch_index, qid in enumerate(ids):
                        current = [data.doc_ids[index] for index in ordering[batch_index].cpu().tolist()]
                        negatives = select_negatives(
                            current, data.gold[qid], data.sources[qid], data.doc_ids, qid, epoch,
                        )
                        positives = sorted(data.gold[qid])
                        positive_rows = torch.tensor([data.doc_row[doc_id] for doc_id in positives], device="cuda")
                        negative_rows = torch.tensor([data.doc_row[doc_id] for doc_id in negatives], device="cuda")
                        positive_scores = bank.rescore(query_vectors[batch_index], indices[batch_index, positive_rows])
                        negative_scores = bank.rescore(query_vectors[batch_index], indices[batch_index, negative_rows])
                        loss = multi_loss(positive_scores, negative_scores)
                        if epoch > 0:
                            loss = loss + BOUNDARY_WEIGHT * boundary_loss(
                                positive_scores, negative_scores,
                                ranks[batch_index, positive_rows], ranks[batch_index, negative_rows],
                            )
                        frozen = torch.as_tensor(data.query_vector(qid), device="cuda")
                        loss = loss + DRIFT_WEIGHT * (
                            1 - F.cosine_similarity(query_vectors[batch_index:batch_index + 1], frozen[None]).mean()
                        )
                        local_losses.append(loss)
                    batch_loss = torch.stack(local_losses).sum() / len(group)
                    if not torch.isfinite(batch_loss):
                        raise FloatingPointError("non-finite query loss")
                    batch_loss.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP, error_if_nonfinite=True)
                if float(gradient_norm) == 0.0:
                    raise RuntimeError("query adapter received no gradients")
                optimizer.step(); scheduler.step(); updates += 1
                next_position = position + len(group)
                save_checkpoint(
                    resume_path, model, optimizer, scheduler, contract_hash=contract_hash,
                    epoch=epoch, position=next_position, updates=updates,
                )
                print(json.dumps({
                    "stage": "train", "epoch": epoch + 1, "position": next_position,
                    "queries": len(order), "updates": updates, "loss": float(batch_loss.detach()),
                    "gradient_norm": float(gradient_norm),
                }), flush=True)
                if stop_after_updates is not None and updates >= stop_after_updates:
                    return {
                        "status": "INTERRUPTED_FOR_RESUME_TEST", "contract_hash": contract_hash,
                        "updates": updates, "epoch": epoch, "position": next_position,
                    }
            save_checkpoint(
                output / f"epoch-{epoch + 1}.pt", model, optimizer, scheduler,
                contract_hash=contract_hash, epoch=epoch + 1, position=0, updates=updates,
            )
        result = {
            "schema_version": "dsc2026.research_v2.exp112_e5_transfer_checkpoint.v1",
            "status": "COMPLETE_EPOCH2",
            "contract_hash": contract_hash,
            "held_fold": "fold_0",
            "training_queries": len(qids),
            "duplicate_exclusions": sorted(data.duplicate_exclusions),
            "epochs": EPOCHS,
            "updates": updates,
            "runtime_seconds_this_process": time.monotonic() - started,
            "peak_allocated_mib": torch.cuda.max_memory_allocated() / (1 << 20),
            "peak_reserved_mib": torch.cuda.max_memory_reserved() / (1 << 20),
            "checkpoint_sha256": {
                f"epoch-{epoch}.pt": sha256(output / f"epoch-{epoch}.pt") for epoch in (1, 2)
            },
            "scientific_contract": contract,
        }
        write_json(done, result)
        return result
    finally:
        del model, bank, optimizer, scheduler
        gc.collect(); torch.cuda.empty_cache()


def ranking(doc_ids: list[str], scores: list[float]) -> list[str]:
    return [doc_ids[index] for index in sorted(
        range(len(doc_ids)), key=lambda index: (-float(scores[index]), str(doc_ids[index]))
    )]


def run_frozen_parity(data: TransferData, output: Path, batch_size: int = 4) -> dict[str, Any]:
    qids = data.fold_qids("fold_0")
    seed_all()
    model = QueryEncoder(data.bundle / "vietlegal-e5")
    model.eval()
    bank = ParentBank(data.vectors, data.parent)
    query_errors, score_errors = [], []
    rank_exact = top5_exact = 0
    rank_disagreement_reference_gaps: list[float] = []
    prediction_rows = []
    started = time.monotonic()
    try:
        for start in range(0, len(qids), batch_size):
            ids = qids[start:start + batch_size]
            with torch.no_grad(), model.adapter_disabled():
                vectors = model([data.questions[qid] for qid in ids])
            for qid, vector in zip(ids, vectors):
                cached = data.query_vector(qid)
                query_errors.extend(np.abs(vector.detach().cpu().numpy() - cached).tolist())
                docs = data.pool[qid]
                scores = bank.score_pool(vector, docs, data)
                reference = data.frozen_reference[qid]
                if docs != [str(value) for value in reference["doc_ids"]]:
                    raise RuntimeError(f"pool/reference doc ordering mismatch: {qid}")
                reference_scores = [float(value) for value in reference["scores"]]
                score_errors.extend(abs(a - b) for a, b in zip(scores, reference_scores))
                order = ranking(docs, scores)
                reference_order = ranking(docs, reference_scores)
                rank_exact += order == reference_order
                top5_exact += order[:5] == reference_order[:5]
                if order != reference_order:
                    reference_by_doc = dict(zip(docs, reference_scores))
                    for actual_doc, expected_doc in zip(order, reference_order):
                        if actual_doc != expected_doc:
                            rank_disagreement_reference_gaps.append(
                                abs(reference_by_doc[actual_doc] - reference_by_doc[expected_doc])
                            )
                prediction_rows.append({"qid": qid, "order": order, "scores": [scores[docs.index(d)] for d in order]})
            print(json.dumps({"stage": "frozen_parity", "completed": min(start + len(ids), len(qids)), "total": len(qids)}), flush=True)
    finally:
        del model, bank
        gc.collect(); torch.cuda.empty_cache()
    prediction_path = output / "FROZEN_RUNNER_PREDICTIONS_FOLD0.jsonl"
    output.mkdir(parents=True, exist_ok=True)
    with prediction_path.open("w", encoding="utf-8", newline="\n") as sink:
        for row in prediction_rows:
            sink.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
    numerical_rank_tolerance = 1e-5
    rank_disagreements_within_tolerance = max(rank_disagreement_reference_gaps, default=0.0) <= numerical_rank_tolerance
    report = {
        "schema_version": "dsc2026.research_v2.e5_frozen_parity.v1",
        "status": "PASS" if max(score_errors) <= 1e-5 and rank_disagreements_within_tolerance and top5_exact == len(qids) else "FAIL",
        "queries": len(qids),
        "query_embedding_max_abs_error": max(query_errors),
        "query_embedding_mean_abs_error": mean(query_errors),
        "score_max_abs_error": max(score_errors),
        "score_mean_abs_error": mean(score_errors),
        "full_pool_rank_exact_queries": rank_exact,
        "full_pool_rank_agreement": rank_exact / len(qids),
        "full_pool_rank_nonexact_queries": len(qids) - rank_exact,
        "rank_disagreement_max_reference_score_gap": max(rank_disagreement_reference_gaps, default=0.0),
        "rank_disagreements_within_numerical_tolerance": rank_disagreements_within_tolerance,
        "numerical_rank_tolerance": numerical_rank_tolerance,
        "top5_exact_queries": top5_exact,
        "top5_agreement": top5_exact / len(qids),
        "runtime_seconds": time.monotonic() - started,
        "predictions_sha256": sha256(prediction_path),
        "reference_sha256": sha256(data.bundle / "V2_FOLD0_FROZEN_REFERENCE.jsonl"),
        "candidate_pool_sha256": data.manifest["candidate_pool_sha256"],
    }
    write_json(output / "FROZEN_BASELINE_PARITY.json", report)
    if report["status"] != "PASS":
        raise RuntimeError(f"frozen baseline parity failed: {report}")
    return report


def run_mining_parity(snapshot_zip: Path, output: Path) -> dict[str, Any]:
    import sys
    sys.path.insert(0, str(snapshot_zip))
    try:
        from exp112.learning import select_negatives as reference_select  # type: ignore
    finally:
        sys.path.pop(0)
    universe = [f"d{i:04d}" for i in range(300)]
    current = universe[:140]
    sources = {
        "e5": universe[10:210],
        "lal": universe[80:280],
        "bm25": universe[60:260],
        "jina": universe[30:230],
    }
    fixtures = []
    for qid, epoch, gold in (("fixture-main", 0, {"d0001", "d0060"}), ("fixture-rotation", 1, {"d0002", "d0030", "d0090"})):
        actual, trace = select_negatives_with_trace(current, gold, sources, universe, qid, epoch)
        expected = reference_select(current, gold, sources, universe, qid, epoch)
        composition = {label: trace.count(label) for label in sorted(set(trace))}
        fixtures.append({
            "qid": qid, "epoch": epoch, "gold": sorted(gold),
            "exact_reference_match": actual == expected,
            "unique": len(actual) == len(set(actual)) == 64,
            "gold_excluded": not bool(set(actual) & gold),
            "composition": composition,
            "selection_sha256": digest(actual),
        })
    replay_a, trace_a = select_negatives_with_trace(current, {"d0001"}, sources, universe, "fixture-replay", 1)
    replay_b, trace_b = select_negatives_with_trace(current, {"d0001"}, sources, universe, "fixture-replay", 1)
    expected_composition = {
        "current_hardest": 16,
        "adapted_17_100_rotated": 16,
        "sparse_lal_disagreement": 16,
        "jina_or_e5_confuser": 8,
        "random": 8,
    }
    composition_pass = all(row["composition"] == expected_composition for row in fixtures)
    status = "PASS" if all(
        row["exact_reference_match"] and row["unique"] and row["gold_excluded"] for row in fixtures
    ) and composition_pass and replay_a == replay_b and trace_a == trace_b else "FAIL"
    report = {
        "schema_version": "dsc2026.research_v2.exp112_mining_parity.v1",
        "status": status,
        "reference_snapshot_sha256": sha256(snapshot_zip),
        "expected_composition": expected_composition,
        "fixtures": fixtures,
        "deterministic_replay": replay_a == replay_b and trace_a == trace_b,
        "replay_sha256": digest(replay_a),
    }
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "MINING_POLICY_PARITY.json", report)
    if status != "PASS":
        raise RuntimeError(f"mining parity failed: {report}")
    return report


def tensor_state(path: Path) -> dict[str, torch.Tensor]:
    return torch.load(path, map_location="cpu", weights_only=False)["adapter"]


def run_smoke_resume(data: TransferData, output: Path, microbatch: int = 4) -> dict[str, Any]:
    qids = data.training_qids()[:16]
    continuous = output / "continuous"
    resumed = output / "resumed"
    continuous_result = train_adapter(data, continuous, microbatch=microbatch, qids=qids)
    if not (resumed / "_SUCCESS.json").exists():
        train_adapter(data, resumed, microbatch=microbatch, qids=qids, stop_after_updates=1)
    resumed_result = train_adapter(data, resumed, microbatch=microbatch, qids=qids)
    left = tensor_state(continuous / "epoch-2.pt")
    right = tensor_state(resumed / "epoch-2.pt")
    if left.keys() != right.keys():
        raise RuntimeError("resume adapter key mismatch")
    errors = [float((left[key] - right[key]).abs().max()) for key in left]
    exact = all(torch.equal(left[key], right[key]) for key in left)
    report = {
        "schema_version": "dsc2026.research_v2.exp112_e5_smoke_resume.v1",
        "status": "PASS" if exact else "FAIL",
        "queries": qids,
        "updates": continuous_result["updates"],
        "forward_backward_complete": True,
        "resume_tensor_exact": exact,
        "max_tensor_abs_error": max(errors, default=0.0),
        "continuous_epoch2_sha256": sha256(continuous / "epoch-2.pt"),
        "resumed_epoch2_sha256": sha256(resumed / "epoch-2.pt"),
        "continuous_peak_allocated_mib": continuous_result["peak_allocated_mib"],
        "resumed_peak_allocated_mib": resumed_result["peak_allocated_mib"],
    }
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "TRAINING_SMOKE_RESUME_PARITY.json", report)
    if not exact:
        raise RuntimeError(f"resume tensor parity failed: {report}")
    return report


def query_recall(top5: list[str], gold: set[str]) -> float:
    return len(set(top5) & gold) / len(gold)


def rank_bucket(rank: int | None) -> str:
    if rank is None: return "missing"
    if rank <= 5: return "1-5"
    if rank <= 10: return "6-10"
    if rank <= 20: return "11-20"
    if rank <= 50: return "21-50"
    return "51+"


def score_and_evaluate(data: TransferData, checkpoint_path: Path, output: Path) -> dict[str, Any]:
    qids = data.fold_qids("fold_0")
    model = QueryEncoder(data.bundle / "vietlegal-e5", checkpoint_path=checkpoint_path)
    model.eval()
    bank = ParentBank(data.vectors, data.parent)
    rows, transitions = [], {}
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    try:
        for index, qid in enumerate(qids, 1):
            with torch.no_grad():
                adapted_vector = model([data.questions[qid]])[0]
                with model.adapter_disabled():
                    frozen_vector = model([data.questions[qid]])[0]
            docs = data.pool[qid]
            adapted_scores = bank.score_pool(adapted_vector, docs, data)
            frozen_scores = bank.score_pool(frozen_vector, docs, data)
            adapted_order = ranking(docs, adapted_scores)
            frozen_order = ranking(docs, frozen_scores)
            gold = data.gold[qid]
            adapted_ranks = {doc_id: adapted_order.index(doc_id) + 1 if doc_id in adapted_order else None for doc_id in gold}
            frozen_ranks = {doc_id: frozen_order.index(doc_id) + 1 if doc_id in frozen_order else None for doc_id in gold}
            for doc_id in gold:
                key = f"{rank_bucket(frozen_ranks[doc_id])}->{rank_bucket(adapted_ranks[doc_id])}"
                transitions[key] = transitions.get(key, 0) + 1
            rows.append({
                "qid": qid, "gold": sorted(gold),
                "base_order": frozen_order, "base_scores": [frozen_scores[docs.index(doc_id)] for doc_id in frozen_order],
                "ft_order": adapted_order, "ft_scores": [adapted_scores[docs.index(doc_id)] for doc_id in adapted_order],
                "base_recall_at_5": query_recall(frozen_order[:5], gold),
                "ft_recall_at_5": query_recall(adapted_order[:5], gold),
                "frozen_query_cosine": float(adapted_vector.detach().cpu().numpy() @ data.query_vector(qid)),
            })
            if index % 25 == 0 or index == len(qids):
                print(json.dumps({"stage": "score", "completed": index, "total": len(qids)}), flush=True)
    finally:
        del model, bank
        gc.collect(); torch.cuda.empty_cache()

    prediction_path = output / "E5_TRANSFER_FOLD0_PREDICTIONS.jsonl"
    output.mkdir(parents=True, exist_ok=True)
    with prediction_path.open("w", encoding="utf-8", newline="\n") as sink:
        for row in rows:
            sink.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
    base = mean(row["base_recall_at_5"] for row in rows)
    adapted = mean(row["ft_recall_at_5"] for row in rows)
    wins = sum(row["ft_recall_at_5"] > row["base_recall_at_5"] for row in rows)
    losses = sum(row["ft_recall_at_5"] < row["base_recall_at_5"] for row in rows)
    single = [row for row in rows if len(row["gold"]) == 1]
    multi = [row for row in rows if len(row["gold"]) > 1]
    multi_delta = mean(row["ft_recall_at_5"] - row["base_recall_at_5"] for row in multi)
    into = sum(value for key, value in transitions.items() if not key.startswith("1-5->") and key.endswith("->1-5"))
    out = sum(value for key, value in transitions.items() if key.startswith("1-5->") and not key.endswith("->1-5"))
    delta = adapted - base
    integrity = read_json(data.bundle / "PRE_GPU_PARITY_GATE.json")
    checks = {
        "delta_recall_at_5_gte_0_005": delta >= 0.005,
        "wins_exceed_losses": wins > losses,
        "gold_crossings_into_exceed_out": into > out,
        "multi_gold_delta_gte_minus_0_005": multi_delta >= -0.005,
        "pre_gpu_parity_pass": integrity["status"] == "PASS_ALL_THREE_GATES",
    }
    if delta < 0.002 or delta < 0 or into <= out or multi_delta < -0.005 or not checks["pre_gpu_parity_pass"]:
        decision = "KILL_STOP_WITHOUT_GRID"
    elif all(checks.values()):
        decision = "PASS_FOLD0_STOP_FOR_USER_REVIEW"
    else:
        decision = "INCONCLUSIVE_STOP_FOR_USER_REVIEW"
    report = {
        "schema_version": "dsc2026.research_v2.exp112_e5_transfer_fold0_pilot.v1",
        "status": decision,
        "full_five_fold_launched": False,
        "queries": len(rows),
        "base_recall_at_5": base,
        "ft_recall_at_5": adapted,
        "delta_recall_at_5": delta,
        "wins": wins, "losses": losses, "ties": len(rows) - wins - losses,
        "changed_top5_sets": sum(set(row["base_order"][:5]) != set(row["ft_order"][:5]) for row in rows),
        "single_gold": {
            "queries": len(single),
            "base": mean(row["base_recall_at_5"] for row in single),
            "ft": mean(row["ft_recall_at_5"] for row in single),
        },
        "multi_gold": {
            "queries": len(multi),
            "base": mean(row["base_recall_at_5"] for row in multi),
            "ft": mean(row["ft_recall_at_5"] for row in multi),
            "delta": multi_delta,
        },
        "gold_crossings_into_top5": into,
        "gold_crossings_out_of_top5": out,
        "gold_rank_bucket_movements": dict(sorted(transitions.items())),
        "query_drift_cosine_mean": mean(row["frozen_query_cosine"] for row in rows),
        "gate_checks": checks,
        "runtime_seconds": time.monotonic() - started,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / (1 << 20),
        "checkpoint_sha256": sha256(checkpoint_path),
        "predictions_sha256": sha256(prediction_path),
        "candidate_pool_sha256": data.manifest["candidate_pool_sha256"],
        "folds_sha256": data.manifest["v2_folds_sha256"],
        "scientific_contract": "exact_exp112_query_only_v1_to_immutable_v2_pool_direct_top5",
    }
    write_json(output / "E5_TRANSFER_FOLD0_PILOT_REPORT.json", report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("frozen-parity", "smoke-resume", "train-fold0", "score-fold0"):
        command = subparsers.add_parser(name)
        command.add_argument("--bundle", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--microbatch", type=int, default=4)
        if name == "score-fold0":
            command.add_argument("--checkpoint", type=Path, required=True)
    mining = subparsers.add_parser("mining-parity")
    mining.add_argument("--snapshot", type=Path, required=True)
    mining.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "mining-parity":
        result = run_mining_parity(args.snapshot, args.output)
    else:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU required for exact EXP-112 runner")
        data = TransferData(args.bundle)
        if args.command in {"train-fold0", "score-fold0"} and data.manifest["status"] != "SEALED_FOLD0_ONLY":
            raise RuntimeError("training/scoring requires a finalized SEALED_FOLD0_ONLY bundle")
        if args.command == "frozen-parity":
            result = run_frozen_parity(data, args.output, batch_size=args.microbatch)
        elif args.command == "smoke-resume":
            result = run_smoke_resume(data, args.output, microbatch=args.microbatch)
        elif args.command == "train-fold0":
            result = train_adapter(data, args.output, microbatch=args.microbatch)
        elif args.command == "score-fold0":
            result = score_and_evaluate(data, args.checkpoint, args.output)
        else:
            raise AssertionError(args.command)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
