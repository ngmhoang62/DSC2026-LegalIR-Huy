"""Lightweight BURST-specific pairwise fine-tuning for Jina reranker.

Only the last two transformer blocks and classification head are updated.  The
evaluation query blocks are excluded, and negatives are taken from the current
BURST multistage ranking (hard negatives rather than random documents).
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from benchmark_jina_reranker_holdouts import load_documents, top_passages
from tune_burst_kernel_posterior import load_queries


class TripletDataset(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


def trainable_state(model):
    return {name: value.detach().cpu() for name, value in model.state_dict().items()
            if any(name == n for n, p in model.named_parameters() if p.requires_grad)}


def hard_negative_rankings(root, qids):
    """RRF over all four raw lexical retrievers; available for all 1,150 ids."""
    saved = pickle.loads(
        (root / "results/burst_large_ltr/retrieval_train1000_tune50_val100.pkl")
        .read_bytes()
    )["cache"]
    output = {}
    for qid in qids:
        scores = {}
        for branch in saved[qid]:
            for rank, item in enumerate(branch[:100]):
                doc = str(item[0])
                scores[doc] = scores.get(doc, 0.0) + 1.0 / (20 + rank)
        output[qid] = sorted(scores, key=lambda doc: (-scores[doc], doc))
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Number of positive/negative triplets per step")
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--last-layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--head-lr", type=float, default=8e-5)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    data = root / "DSC2026-LegalIR-main/v4_run/public_test_dataset"
    output = root / "results/jina_reranker"
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "burst_pairwise_state.pt"
    manifest_path = output / "burst_pairwise_training.json"

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True

    queries = load_queries(root)
    qids = list(queries)
    # Clean training pool: none of validation A (750:850), B (1250:1350),
    # or the untouched final test C (1350:1450) is included.
    train_ids = qids[:750] + qids[850:1150]
    print(f"Building hard-negative rankings for {len(train_ids)} queries", flush=True)
    rankings = hard_negative_rankings(root, train_ids)
    documents = load_documents(data)

    rows = []
    missing_docs = 0
    for qid in train_ids:
        question, gold = queries[qid]
        negatives = [doc for doc in rankings[qid][:20] if doc not in gold][:4]
        if not negatives:
            continue
        for pi, positive in enumerate(sorted(gold)):
            if positive not in documents:
                missing_docs += 1
                continue
            positive_text = top_passages(question, documents[positive], count=1)[0]
            # Two strong but diverse negatives per relevant document.
            chosen = [negatives[pi % len(negatives)],
                      negatives[(pi + 2) % len(negatives)]]
            for negative in dict.fromkeys(chosen):
                negative_text = top_passages(question, documents[negative], count=1)[0]
                rows.append((question, positive_text, negative_text, qid,
                             positive, negative))
    random.shuffle(rows)
    print(f"Pairwise triplets: {len(rows)}; missing positive docs: {missing_docs}",
          flush=True)

    model_path = root / "models/jina-reranker-v2-base-multilingual"
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, fix_mistral_regex=True
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, trust_remote_code=True, dtype=torch.bfloat16
    )
    for parameter in model.parameters():
        parameter.requires_grad = False
    layers = model.roberta.encoder.layers
    for layer in layers[-args.last_layers:]:
        for parameter in layer.parameters():
            parameter.requires_grad = True
    for parameter in model.classifier.parameters():
        parameter.requires_grad = True

    tail_params, head_params = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (head_params if name.startswith("classifier.") else tail_params).append(parameter)
    optimizer = AdamW([
        {"params": tail_params, "lr": args.lr, "weight_decay": .01},
        {"params": head_params, "lr": args.head_lr, "weight_decay": .01},
    ])
    total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_trainable:,}", flush=True)

    def collate(batch):
        pairs = []
        for question, positive, negative, *_ in batch:
            pairs.extend(((question, positive), (question, negative)))
        encoded = tokenizer(pairs, padding=True, truncation=True,
                            max_length=args.max_length, return_tensors="pt")
        return encoded

    loader = DataLoader(TripletDataset(rows), batch_size=args.batch_size,
                        shuffle=True, collate_fn=collate, num_workers=0,
                        pin_memory=True)
    total_steps = max(1, len(loader) * args.epochs)
    warmup = max(10, int(.08 * total_steps))

    model.train().to("cuda")
    started = time.perf_counter()
    running = 0.0
    global_step = 0
    for epoch in range(args.epochs):
        for batch in loader:
            global_step += 1
            batch = {key: value.to("cuda", non_blocking=True)
                     for key, value in batch.items()}
            logits = model(**batch, return_dict=True).logits.float().view(-1)
            positive, negative = logits[0::2], logits[1::2]
            # Logistic pairwise loss. A small margin makes boundary mistakes count.
            loss = F.softplus(-(positive - negative - .15)).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(tail_params + head_params, 1.0)
            if global_step <= warmup:
                scale = global_step / warmup
            else:
                progress = (global_step - warmup) / max(total_steps - warmup, 1)
                scale = .1 + .9 * .5 * (1.0 + math.cos(math.pi * progress))
            for group, base_lr in zip(optimizer.param_groups,
                                      (args.lr, args.head_lr)):
                group["lr"] = base_lr * scale
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            running += float(loss)
            if global_step % 50 == 0 or global_step == total_steps:
                elapsed = time.perf_counter() - started
                print(f"epoch {epoch+1}/{args.epochs} step {global_step}/{total_steps} "
                      f"loss={running/min(global_step,50):.4f} "
                      f"elapsed={elapsed:.1f}s gpu={torch.cuda.max_memory_allocated()/2**30:.2f}GB",
                      flush=True)
                running = 0.0

    state = {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()
             if name.startswith("classifier.") or any(
                 name.startswith(f"roberta.encoder.layers.{i}.")
                 for i in range(len(layers)-args.last_layers, len(layers))
             )}
    torch.save({"state_dict": state, "args": vars(args),
                "train_ids": train_ids, "triplets": len(rows)}, checkpoint)
    manifest = {
        "base_model": str(model_path), "checkpoint": str(checkpoint),
        "train_queries": len(train_ids), "triplets": len(rows),
        "trainable_parameters": total_trainable, "seconds": time.perf_counter()-started,
        "settings": vars(args),
        "excluded_ranges": ["750:850", "1250:1350", "1350:1450"],
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                             encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
