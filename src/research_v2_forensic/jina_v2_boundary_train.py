"""Strict-fold pairwise LoRA training and direct Top-5 scoring for Jina-v2.

This file is copied verbatim into the Kaggle bundle.  It deliberately contains
no fusion or hand-written promotion rule.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import sqlite3
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


TOKEN_RE = re.compile(r"\w+", re.UNICODE)
SPACE_RE = re.compile(r"\S+", re.UNICODE)
STOPWORDS = {
    "bị", "các", "có", "của", "cho", "được", "để", "đến", "đối", "gì",
    "hay", "khi", "không", "là", "làm", "một", "nào", "những", "như",
    "phải", "ra", "sẽ", "theo", "thì", "thế", "trong", "trên", "từ",
    "và", "về", "với", "việc", "bao", "nhiêu", "người", "quy", "định",
}


def tokens(text: str) -> list[str]:
    return TOKEN_RE.findall((text or "").lower())


def top_passages(question: str, text: str, count: int = 2,
                 window: int = 220, overlap: int = 70) -> list[str]:
    """Byte-for-logic copy of Huy's locked lexical evidence selector."""
    words = SPACE_RE.findall(text or "")
    if len(words) <= window + 80:
        return [" ".join(words)]
    query_tokens = tokens(question)
    content = {t for t in query_tokens if len(t) >= 3 and t not in STOPWORDS}
    numbers = {t for t in query_tokens if any(c.isdigit() for c in t)}
    bigrams = {" ".join(query_tokens[i:i + 2]) for i in range(len(query_tokens) - 1)}
    header = " ".join(words[:70]); scored = []; step = window - overlap
    for start in range(0, len(words), step):
        end = min(start + window, len(words)); part = " ".join(words[start:end])
        normalized = tokens(part); token_set = set(normalized); norm_text = " ".join(normalized)
        coverage = sum(1.0 + .20 * min(normalized.count(t), 3) for t in content if t in token_set)
        numeric = 3.0 * sum(t in token_set for t in numbers)
        phrase = 1.8 * sum(p in norm_text for p in bigrams)
        density = (coverage + numeric + phrase) / math.sqrt(max(len(normalized), 1))
        scored.append((density, coverage + numeric + phrase, -start, part))
        if end == len(words): break
    scored.sort(reverse=True); passages = []
    for _, _, neg_start, part in scored:
        candidate = part if -neg_start < 70 else header + "\n[ĐOẠN PHÙ HỢP]\n" + part
        if candidate not in passages: passages.append(candidate)
        if len(passages) >= count: break
    return passages


def patch_transformers_v5() -> None:
    import transformers.models.xlm_roberta.modeling_xlm_roberta as module
    if hasattr(module, "create_position_ids_from_input_ids"):
        return
    def helper(input_ids, padding_idx, past_key_values_length=0):
        mask = input_ids.ne(padding_idx).int()
        positions = (torch.cumsum(mask, dim=1) + past_key_values_length) * mask
        return positions.long() + padding_idx
    module.create_position_ids_from_input_ids = helper


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def directory_hashes(path: Path) -> dict[str, str]:
    return {str(p.relative_to(path)).replace("\\", "/"): sha256(p)
            for p in sorted(path.rglob("*")) if p.is_file()}


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)


def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def base_model(path: Path):
    patch_transformers_v5()
    import transformers
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True, fix_mistral_regex=True)
    dtype_kwarg = ({"dtype": torch.float16} if int(transformers.__version__.split(".")[0]) >= 5
                   else {"torch_dtype": torch.float16})
    model = AutoModelForSequenceClassification.from_pretrained(
        path, trust_remote_code=True, **dtype_kwarg)
    model.config.use_cache = False
    return model, tok


def lora_model(path: Path, adapter: Path | None, rank: int):
    from peft import LoraConfig, TaskType, get_peft_model
    model, tok = base_model(path)
    config = LoraConfig(
        task_type=TaskType.SEQ_CLS, r=rank, lora_alpha=2 * rank,
        lora_dropout=0.05, target_modules=["Wqkv", "out_proj"],
        modules_to_save=["classifier"], bias="none")
    model = get_peft_model(model, config)
    # Cast before loading a checkpoint.  Loading FP32 adapter tensors into the
    # freshly created FP16 modules first would round away the small updates and
    # make resume diverge from an uninterrupted run.
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.float()
    if adapter is not None:
        # Recreate exactly the original topology, then load its state.  Calling
        # PeftModel.from_pretrained here lets SEQ_CLS auto-discovery append
        # classifier/score wrappers a second time, which breaks exact resume.
        from peft.utils.save_and_load import load_peft_weights, set_peft_model_state_dict
        adapter_state = load_peft_weights(str(adapter), device="cpu")
        classifier_prefix = "base_model.model.classifier."
        serialized_classifier = {
            key: value for key, value in adapter_state.items()
            if key.startswith(classifier_prefix)
        }
        set_peft_model_state_dict(model, adapter_state, adapter_name="default")
        # PEFT 0.20 does not restore SEQ_CLS ``modules_to_save`` values when
        # the custom remote-code classifier is wrapped after auto-discovery.
        # Map the four serialized classifier tensors to the active wrapper
        # explicitly and fail closed if the expected contract changes.
        classifier_state = {
            key[len(classifier_prefix):]: value
            for key, value in serialized_classifier.items()
        }
        if len(classifier_state) != 4:
            raise RuntimeError(f"expected four classifier tensors, got {len(classifier_state)}")
        active_classifier = model.base_model.model.classifier.modules_to_save["default"]
        incompatible = active_classifier.load_state_dict(classifier_state, strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(f"classifier adapter load mismatch: {incompatible}")
    patch_tuple_returning_lora(model)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    return model.to("cuda"), tok


def patch_tuple_returning_lora(model) -> None:
    """Make PEFT LoRA compatible with Jina-v2's ``LinearResidual``.

    Jina-v2 configures each attention ``Wqkv`` projection to return
    ``(projected, residual)``.  PEFT's stock Linear wrapper assumes that every
    wrapped ``nn.Linear`` returns a tensor and accesses ``result.dtype``.  The
    adapter mathematics is unchanged here: apply the LoRA delta to the
    projected tensor and pass the residual through byte-for-byte.
    """
    from types import MethodType

    def tuple_lora_forward(self, x, *args, **kwargs):
        # Mixed-adapter inference is deliberately outside this experiment.
        if kwargs.get("adapter_names") is not None:
            raise RuntimeError("mixed-adapter batches are unsupported for LinearResidual")
        kwargs.pop("adapter_names", None)
        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            return self.base_layer(x, *args, **kwargs)
        if self.merged:
            return self.base_layer(x, *args, **kwargs)

        projected, residual = self.base_layer(x, *args, **kwargs)
        result_dtype = projected.dtype
        for active_adapter in self.active_adapters:
            if active_adapter not in self.lora_A:
                continue
            lora_A = self.lora_A[active_adapter]
            lora_B = self.lora_B[active_adapter]
            dropout = self.lora_dropout[active_adapter]
            scaling = self.scaling[active_adapter]
            adapter_x = self._cast_input_dtype(x, lora_A.weight.dtype)
            projected = projected + lora_B(lora_A(dropout(adapter_x))) * scaling
        return projected.to(result_dtype), residual

    patched = 0
    for module in model.modules():
        base = getattr(module, "base_layer", None)
        if base is not None and base.__class__.__name__ == "LinearResidual":
            module.forward = MethodType(tuple_lora_forward, module)
            patched += 1
    if patched == 0:
        raise RuntimeError("expected Jina-v2 LinearResidual LoRA modules were not found")


def parent_pair_loss(model, tok, batch: list[tuple[str, list[str], list[str]]], max_length: int):
    texts, spans = [], []
    for query, positives, negatives in batch:
        p0 = len(texts); texts.extend((query, p) for p in positives); p1 = len(texts)
        n0 = len(texts); texts.extend((query, p) for p in negatives); n1 = len(texts)
        spans.append((p0, p1, n0, n1))
    encoded = tok(texts, padding=True, truncation=True, max_length=max_length,
                  return_tensors="pt")
    encoded = {k: v.to("cuda", non_blocking=True) for k, v in encoded.items()}
    logits = model(**encoded).logits.float().view(-1)
    losses = []
    for p0, p1, n0, n1 in spans:
        pos = logits[p0:p1].max()
        neg = logits[n0:n1].max()
        losses.append(F.softplus(-(pos - neg)))
    return torch.stack(losses).mean(), logits


def build_pairs(groups: list[dict]) -> list[tuple[str, str, list[str], list[str]]]:
    pairs = []
    for group in groups:
        for positive in group["positives"]:
            for negative in group["negatives"]:
                pairs.append((str(group["qid"]), str(group["query"]),
                              list(positive["passages"]), list(negative["passages"])))
    return pairs


@torch.no_grad()
def validation_accuracy(model, tok, pairs, max_length: int, limit: int = 400) -> float:
    model.eval(); correct = total = 0
    for _, query, positives, negatives in pairs[:limit]:
        with torch.cuda.amp.autocast(dtype=torch.float16):
            _, logits = parent_pair_loss(model, tok, [(query, positives, negatives)], max_length)
        split = len(positives)
        correct += int(logits[:split].max() > logits[split:].max()); total += 1
    model.train()
    return correct / max(total, 1)


def save_checkpoint(model, optimizer, scaler, output: Path, state: dict) -> None:
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output / "adapter")
    torch.save(optimizer.state_dict(), output / "optimizer.pt")
    torch.save(scaler.state_dict(), output / "scaler.pt")
    torch.save({"python": random.getstate(), "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()},
               output / "rng_state.pt")
    (output / "trainer_state.json").write_text(json.dumps(state, indent=2), encoding="utf-8")


def train(args) -> None:
    seed_all(args.seed)
    if args.eval_steps % args.accumulation:
        raise ValueError("eval_steps must align with an optimizer boundary for exact resume")
    manifest = json.loads(args.groups_manifest.read_text(encoding="utf-8"))
    held = args.fold
    forbidden = set(manifest["held_fold_duplicate_exclusions"].get(held, []))
    groups = [row for row in read_jsonl(args.groups)
              if row["fold"] != held and str(row["qid"]) not in forbidden]
    train_groups = [g for g in groups if int(hashlib.sha256(str(g["qid"]).encode()).hexdigest(), 16) % 10]
    valid_groups = [g for g in groups if not int(hashlib.sha256(str(g["qid"]).encode()).hexdigest(), 16) % 10]
    if args.max_train_groups:
        train_groups = sorted(train_groups, key=lambda g: hashlib.sha256(str(g["qid"]).encode()).hexdigest())[:args.max_train_groups]
    train_pairs, valid_pairs = build_pairs(train_groups), build_pairs(valid_groups)
    random.Random(args.seed).shuffle(train_pairs)

    torch.cuda.reset_peak_memory_stats()
    resume_adapter = args.resume / "adapter" if args.resume else None
    model, tok = lora_model(args.model, resume_adapter, args.lora_rank)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr)
    start_step = 0
    best_acc, bad_checks = -1.0, 0
    if args.resume:
        optimizer.load_state_dict(torch.load(args.resume / "optimizer.pt", map_location="cpu"))
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to("cuda")
        resume_state = json.loads((args.resume / "trainer_state.json").read_text())
        start_step = resume_state["step"]
        best_acc = float(resume_state.get("best_validation_pair_accuracy", -1.0))
        bad_checks = int(resume_state.get("bad_checks", 0))
    scaler = torch.cuda.amp.GradScaler()
    if args.resume:
        scaler.load_state_dict(torch.load(args.resume / "scaler.pt", map_location="cpu"))
        rng = torch.load(args.resume / "rng_state.pt", map_location="cpu", weights_only=False)
        random.setstate(rng["python"]); np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch"]); torch.cuda.set_rng_state_all(rng["cuda"])
    model.train()
    started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    total_steps = min(args.max_steps or math.ceil(len(train_pairs) / args.microbatch),
                      math.ceil(len(train_pairs) / args.microbatch))
    for step in range(start_step, total_steps):
        lo = step * args.microbatch
        items = train_pairs[lo:lo + args.microbatch]
        batch = [(q, p, n) for _, q, p, n in items]
        # Make stochastic adapter dropout a pure function of the microstep.
        # This removes serialization/runtime side effects across process resume
        # while retaining the preregistered dropout probability.
        torch.manual_seed(args.seed + step)
        torch.cuda.manual_seed_all(args.seed + step)
        with torch.cuda.amp.autocast(dtype=torch.float16):
            loss, _ = parent_pair_loss(model, tok, batch, args.max_length)
            loss = loss / args.accumulation
        scaler.scale(loss).backward()
        if (step + 1) % args.accumulation == 0 or step + 1 == total_steps:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
        if (step + 1) % args.eval_steps == 0 or step + 1 == total_steps:
            acc = validation_accuracy(model, tok, valid_pairs, args.max_length)
            improved = acc > best_acc + 1e-5
            if improved:
                best_acc, bad_checks = acc, 0
            else:
                bad_checks += 1
            state = {"step": step + 1, "validation_pair_accuracy": acc,
                     "best_validation_pair_accuracy": best_acc,
                     "bad_checks": bad_checks,
                     "train_pairs": len(train_pairs), "valid_pairs": len(valid_pairs)}
            checkpoint = args.output / f"checkpoint-{step + 1:06d}"
            print(json.dumps({**state, "loss": float(loss) * args.accumulation}), flush=True)
            # Save the optional best snapshot first.  The resumable checkpoint
            # must be last because adapter serialization advances CUDA RNG.
            if improved:
                save_checkpoint(model, optimizer, scaler, args.output / "best", state)
            save_checkpoint(model, optimizer, scaler, checkpoint, state)
            if bad_checks >= args.patience:
                break
    runtime = time.perf_counter() - started
    final = {
        "schema_version": "dsc2026.research_v2.jina_boundary_checkpoint.v1",
        "status": "COMPLETE", "held_fold": held,
        "base_model": str(args.model), "renderer": manifest["renderer"],
        "train_groups": len(train_groups), "valid_groups": len(valid_groups),
        "forbidden_duplicate_qids": sorted(forbidden),
        "pairwise_parent_max_loss": True, "max_length": args.max_length,
        "microbatch_parent_pairs": args.microbatch, "gradient_accumulation": args.accumulation,
        "lora": {"rank": args.lora_rank, "targets": ["Wqkv", "out_proj"], "classifier_trainable": True},
        "runtime_seconds": runtime,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "groups_sha256": sha256(args.groups), "groups_manifest_sha256": sha256(args.groups_manifest),
        "best_adapter_files_sha256": directory_hashes(args.output / "best" / "adapter"),
    }
    (args.output / "TRAINING_MANIFEST.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    print(json.dumps(final, indent=2))


@torch.no_grad()
def score(args) -> None:
    seed_all(args.seed)
    torch.cuda.reset_peak_memory_stats()
    model, tok = lora_model(args.model, args.adapter, args.lora_rank)
    model.eval(); started = time.perf_counter()
    base = sqlite3.connect(f"file:{args.base_score_db.resolve().as_posix()}?mode=ro", uri=True)
    base_scores = {(str(q), str(d)): float(s) for q, d, s in
                   base.execute("SELECT qid,doc_id,score FROM scores WHERE arm=?", (args.renderer,))}
    base.close()
    group_gold = {str(g["qid"]): {str(p["doc_id"]) for p in g["positives"]}
                  for g in read_jsonl(args.groups)}
    rows, deltas, base_precisions, ft_precisions = [], [], [], []
    single, multi, boundary_base, boundary_ft = [], [], [], []
    contexts = {str(row["doc_id"]): str(row["passage"]) for row in read_jsonl(args.contexts_pack)}
    if args.renderer != "lexical":
        raise RuntimeError("This locked executable bundle supports the lexical Phase-2 winner only")
    for row in read_jsonl(args.pool):
        if row["fold"] != args.fold:
            continue
        qid, query = str(row["qid"]), str(row["query"])
        parents, passages, owners = [], [], []
        for parent in row["doc_ids"]:
            doc = str(parent); parents.append(doc)
            for passage in top_passages(query, contexts[doc], count=2):
                owners.append(doc); passages.append(str(passage))
        parent_scores = {d: -1e9 for d in parents}
        for start in range(0, len(passages), args.score_batch):
            pairs = [(query, p) for p in passages[start:start + args.score_batch]]
            encoded = tok(pairs, padding=True, truncation=True, max_length=args.max_length,
                          return_tensors="pt")
            encoded = {k: v.to("cuda") for k, v in encoded.items()}
            with torch.cuda.amp.autocast(dtype=torch.float16):
                logits = model(**encoded).logits.float().view(-1).cpu().tolist()
            for doc, value in zip(owners[start:start + args.score_batch], logits):
                parent_scores[doc] = max(parent_scores[doc], float(value))
        ft_order = sorted(parents, key=lambda d: (-parent_scores[d], d))
        base_order = sorted(parents, key=lambda d: (-base_scores[(qid, d)], d))
        gold = group_gold[qid]
        base_recall = len(set(base_order[:5]) & gold) / len(gold)
        ft_recall = len(set(ft_order[:5]) & gold) / len(gold)
        base_precisions.append(len(set(base_order[:5]) & gold) / 5.0)
        ft_precisions.append(len(set(ft_order[:5]) & gold) / 5.0)
        delta = ft_recall - base_recall; deltas.append(delta)
        (single if len(gold) == 1 else multi).append((base_recall, ft_recall))
        # Fixed evaluation universe from the frozen base CE; do not let the
        # trained model choose which pairs count as its own boundary test.
        band = base_order[3:20]
        for pos in [d for d in band if d in gold]:
            for neg in [d for d in band if d not in gold]:
                boundary_base.append(base_scores[(qid, pos)] > base_scores[(qid, neg)])
                boundary_ft.append(parent_scores[pos] > parent_scores[neg])
        rows.append({"qid": qid, "base_top5": base_order[:5], "ft_top5": ft_order[:5],
                     "gold": sorted(gold), "base_recall": base_recall, "ft_recall": ft_recall})
        if len(rows) % 100 == 0:
            print(f"scored={len(rows)}", flush=True)
    output = args.output / f"{args.fold}_PREDICTIONS.jsonl"
    args.output.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as sink:
        for row in rows:
            sink.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
    def slice_metrics(values):
        return {"queries": len(values), "base_recall": float(np.mean([x[0] for x in values])),
                "ft_recall": float(np.mean([x[1] for x in values]))} if values else {"queries": 0}
    report = {
        "schema_version": "dsc2026.research_v2.jina_boundary_fold_metric.v1",
        "status": "COMPLETE", "fold": args.fold, "queries": len(rows),
        "base_recall_at_5": float(np.mean([r["base_recall"] for r in rows])),
        "ft_recall_at_5": float(np.mean([r["ft_recall"] for r in rows])),
        "base_precision_at_5": float(np.mean(base_precisions)),
        "ft_precision_at_5": float(np.mean(ft_precisions)),
        "delta_recall_at_5": float(np.mean(deltas)),
        "wins": sum(x > 0 for x in deltas), "losses": sum(x < 0 for x in deltas),
        "ties": sum(x == 0 for x in deltas),
        "changed_top5_sets": sum(set(r["base_top5"]) != set(r["ft_top5"]) for r in rows),
        "single_gold": slice_metrics(single), "multi_gold": slice_metrics(multi),
        "boundary_pair_accuracy": {"base": float(np.mean(boundary_base)),
                                    "ft": float(np.mean(boundary_ft)), "pairs": len(boundary_ft)},
        "runtime_seconds": time.perf_counter() - started,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "predictions_sha256": sha256(output),
        "adapter_files_sha256": directory_hashes(args.adapter),
    }
    (args.output / f"{args.fold}_METRICS.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def parser():
    p = argparse.ArgumentParser(); sub = p.add_subparsers(dest="stage", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--model", type=Path, required=True)
    common.add_argument("--groups", type=Path, required=True)
    common.add_argument("--fold", choices=[f"fold_{i}" for i in range(5)], required=True)
    common.add_argument("--max-length", type=int, default=512)
    common.add_argument("--lora-rank", type=int, default=8)
    common.add_argument("--seed", type=int, default=20260909)
    t = sub.add_parser("train", parents=[common])
    t.add_argument("--groups-manifest", type=Path, required=True)
    t.add_argument("--output", type=Path, required=True)
    t.add_argument("--microbatch", type=int, default=2)
    t.add_argument("--accumulation", type=int, default=8)
    t.add_argument("--lr", type=float, default=2e-5)
    t.add_argument("--max-steps", type=int, default=None)
    t.add_argument("--max-train-groups", type=int, default=None)
    t.add_argument("--eval-steps", type=int, default=200)
    t.add_argument("--patience", type=int, default=2)
    t.add_argument("--resume", type=Path, default=None)
    s = sub.add_parser("score", parents=[common])
    s.add_argument("--adapter", type=Path, required=True)
    s.add_argument("--pool", type=Path, required=True)
    s.add_argument("--contexts-pack", type=Path, required=True)
    s.add_argument("--base-score-db", type=Path, required=True)
    s.add_argument("--renderer", choices=["lexical", "structural"], required=True)
    s.add_argument("--score-batch", type=int, default=16)
    s.add_argument("--output", type=Path, required=True)
    return p


if __name__ == "__main__":
    args = parser().parse_args()
    train(args) if args.stage == "train" else score(args)
