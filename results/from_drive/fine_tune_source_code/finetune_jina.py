"""Fine-tune the Jina cross-encoder -- the `jina` channel of the BURST stack.

Architecture kept identical to the shipped runner
(run_burst_expanded_fusion_submission.rerank):

    AutoModelForSequenceClassification("jina-reranker-v2-base-multilingual",
                                       trust_remote_code=True)
    pairs = (question, passage) for 2 density-selected windows per document,
    max_length 512, document score = max over its windows

What this script recomputes, and what it reuses
-----------------------------------------------
Reused from results/ (never recomputed, no GPU spent):
    layer 1  BM25 multi-branch retrieval, multistage top-20, dense expansion,
             corpus dense rank cap=32  -> the candidate pool is byte-identical
             every epoch, so epochs are comparable to each other and to the
             shipped baseline
    layer 2  the `dense` and `vnlegal_lal` channels stay on their cached scores
Recomputed each epoch:
    layer 2  the `jina` channel only -- this model's own output
    layer 3  the 6-channel LTR fusion (CPU, ~2 s)
    layer 4  dynamic threshold alpha=0.15

Training data is every labelled query whose retrieval is already cached
(train.json indices 0-1149 and 1250-1749, ~1,650 queries) -- this now
INCLUDES the 600-query LOBO block, so the per-epoch evaluate() below is
scoring on queries the model was also trained on and its numbers are
optimistic, not a generalization estimate.  Hard negatives are the top of
each query's cached lexical pool minus the gold documents, so the model
trains against the exact distractors the deployed retrieval produces.

Loss defaults to pairwise RankNet, matching the objective the shipped
checkpoint is named for (results/jina_reranker/burst_pairwise_state.pt).

Outputs, under --work/jina/ (default /kaggle/working/jina/):
    best_state.pt            {"state_dict": ...}, drop-in for the runner's
                             torch.load(...)["state_dict"] + strict=False load
    best_predictions.json    {qid: {"answer": [...]}} after the dynamic threshold
    best_ranking.json        top-20 fused ranking per query
    best_channel_scores.pkl  {qid: {doc: score}} -- feeds straight back into the
                             pipeline as a jina score cache
    history.json             every epoch's metrics, including the cached baseline

Usage
    python finetune_jina.py --epochs 3
    python finetune_jina.py --epochs 3 --eval-before-training   # adds an epoch-0
                                                                # reference run
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import burst_common as bc
import torch_common as tc

TAG = "jina"
DEFAULT_MODEL = "models/jina-reranker-v2-base-multilingual"
HF_REPO = "jinaai/jina-reranker-v2-base-multilingual"
SHIPPED_CHECKPOINT = "results/jina_reranker/burst_pairwise_state.pt"


def load_model(root, model_path, init_checkpoint, gradient_checkpointing):
    source = tc.resolve_model_source(root, model_path, HF_REPO)

    # This is the only one of the three models loaded through trust_remote_code,
    # so it is the only one that can break on a transformers upgrade.
    tc.patch_hub_code_compat()

    try:
        tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=True,
                                                  fix_mistral_regex=True)
    except TypeError:
        tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=True)

    # fp32 weights + autocast: training the bf16/fp16 weights the runner loads
    # for inference directly would not converge.  config.json asks for bfloat16,
    # so fp32 has to be requested explicitly -- see tc.load_fp32.
    try:
        model = tc.load_fp32(AutoModelForSequenceClassification, source,
                             trust_remote_code=True, use_flash_attn=False)
    except TypeError:
        model = tc.load_fp32(AutoModelForSequenceClassification, source,
                             trust_remote_code=True)

    if init_checkpoint and init_checkpoint != "none":
        checkpoint = Path(init_checkpoint)
        if init_checkpoint == "auto":
            checkpoint = Path(root) / SHIPPED_CHECKPOINT
        if checkpoint.exists():
            saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
            missing, unexpected = model.load_state_dict(saved["state_dict"],
                                                        strict=False)
            print(f"Resuming from {checkpoint} "
                  f"(missing={len(missing)} unexpected={len(unexpected)})", flush=True)
        elif init_checkpoint != "auto":
            raise FileNotFoundError(checkpoint)
        else:
            print(f"No shipped checkpoint at {checkpoint}; starting from base weights",
                  flush=True)

    tc.enable_gradient_checkpointing(model, gradient_checkpointing)
    model._tokenizer = tokenizer
    return model, tokenizer


def train_epoch(model, tokenizer, sampler, args, optimizer, scheduler, scaler,
                device, amp_dtype, epoch):
    tc.guard_parameters(model, epoch, "before")
    model.train()
    order = sampler.epoch(epoch)
    total, finite, skipped, processed, steps, bad_steps = 0.0, 0, 0, 0, 0, 0
    optimizer.zero_grad(set_to_none=True)
    starts = range(0, len(order), args.batch_size)
    bar = bc.progress(starts, desc=f"epoch {epoch} train", total=len(starts),
                      unit="batch")
    for start in bar:
        indices = order[start:start + args.batch_size]
        questions, passages, sizes = [], [], []
        for index in indices:
            question, texts, size = sampler.group(index)
            questions.extend([question] * size)
            passages.extend(texts)
            sizes.append(size)
        with tc.autocast(device, amp_dtype):
            logits = tc.cross_encoder_logits(model, tokenizer, questions, passages,
                                             args.max_length, device)
        # Loss outside the autocast block so the ranking objective runs in fp32.
        losses, offset = [], 0
        for size in sizes:
            group = logits[offset:offset + size]
            losses.append(tc.pairwise_loss(group, size - 1)
                          if args.loss == "pairwise"
                          else tc.listwise_loss(group, size - 1, args.temperature))
            offset += size
        loss = torch.stack(losses).mean() / args.accum
        processed += 1
        if torch.isfinite(loss):
            scaler.scale(loss).backward()
            total += float(loss.detach()) * args.accum
            finite += 1
        else:
            # Never backward a NaN: it would poison every weight at the next
            # step.  Grads already accumulated in this window stay valid.
            skipped += 1
        if processed % args.accum == 0:
            bad_steps += not tc.step_if_finite(model, optimizer, scaler,
                                               args.max_grad_norm)
            scheduler.step()
            steps += 1
        bar.set_postfix(loss=f"{total/max(finite,1):.4f}", step=steps,
                        skipped=skipped, bad=bad_steps,
                        lr=f"{scheduler.get_last_lr()[0]:.1e}")
    if processed % args.accum:
        bad_steps += not tc.step_if_finite(model, optimizer, scaler,
                                           args.max_grad_norm)
    if bad_steps:
        print(f"  {bad_steps}/{steps} optimizer steps had non-finite gradients "
              f"and were not applied.", flush=True)
    tc.report_skipped(skipped, processed)
    tc.guard_parameters(model, epoch, "after")
    return total / max(finite, 1)


def evaluate(model, tokenizer, bundle, args, device, amp_dtype):
    questions = {q: bundle.queries[q][0] for q in bundle.all_ids}
    scores = tc.score_cross_encoder(
        model, tokenizer, questions, bundle.extended, bundle.documents,
        bundle.all_ids, device, amp_dtype, max_length=args.max_length,
        batch_size=args.eval_batch_size, label=TAG)
    views = {TAG: bc.rank_by(bundle.extended, scores)}
    metrics, ranked, predictions = bc.lobo_evaluate(bundle, {TAG: scores}, views)
    metrics["channel_only"] = bc.channel_metrics(bundle, scores)
    return metrics, ranked, predictions, scores


def main():
    ap = bc.common_args(__doc__.splitlines()[0])
    ap.add_argument("--model-path", default=DEFAULT_MODEL)
    ap.add_argument("--init-checkpoint", default="auto",
                    help="'auto' resumes from the shipped burst_pairwise state if "
                         "present, 'none' starts from base weights, or a path")
    ap.add_argument("--loss", choices=("pairwise", "listwise"), default="pairwise")
    ap.add_argument("--temperature", type=float, default=1.0,
                    help="listwise loss only")
    ap.add_argument("--passages-per-doc", type=int, default=1,
                    help="windows per document during TRAINING; evaluation always "
                         "uses the pipeline's 2")
    ap.set_defaults(lr=1e-5, batch_size=4, accum=4, negatives=7, eval_batch_size=16)
    args = ap.parse_args()

    device, amp_dtype, n_gpus = tc.setup(args.seed)
    if args.max_gpus:
        n_gpus = min(n_gpus, args.max_gpus)
    tc.scale_for_gpus(args, n_gpus)
    args.work.mkdir(parents=True, exist_ok=True)

    print("\n=== stage 1-2 cache: rebuilding the 600-query holdout ===", flush=True)
    documents = bc.DocumentStore(Path(args.root) / bc.DATA_SUBDIR, preload=True)
    bundle = bc.build_eval_bundle(args.root, documents)

    print("\n=== cached baseline (all six channels from results/) ===", flush=True)
    baseline, _, _ = bc.lobo_evaluate(bundle)
    print(f"recall={baseline['recall']:.4f} precision={baseline['precision']:.4f} "
          f"f2={baseline['f2']:.4f} Recall@5={baseline['Recall@5']:.4f} "
          f"answers/query={baseline['mean_answers']:.3f}", flush=True)
    recorder = bc.RunRecorder(args.work, TAG, baseline, resume=args.resume)

    print("\n=== training pool from the cached retrieval ===", flush=True)
    examples = bc.build_train_pool(args.root, negatives=args.negative_depth,
                                   limit=args.train_queries, seed=args.seed,
                                   documents=documents)

    print("\n=== model ===", flush=True)
    model, tokenizer = load_model(args.root, args.model_path, args.init_checkpoint,
                                  args.gradient_checkpointing)
    if args.resume:
        recorder.load_best_state(model)
    model.to(device)

    probe_texts = [documents[d] for d in examples[0].negatives[:4]]
    amp_dtype = tc.choose_precision(model, tokenizer, probe_texts, device,
                                    amp_dtype, args.max_length, kind="cross",
                                    requested=args.precision)
    tc.check_pooling_discriminates(model, tokenizer, probe_texts, device,
                                   amp_dtype, args.max_length, kind="cross",
                                   label=TAG)
    tc.freeze_lower_layers(model, args.train_top_layers)
    tc.report_memory_budget(model, n_gpus, device)
    model = tc.wrap_parallel(model, n_gpus)

    if args.eval_before_training:
        print("\n=== epoch 0: this model before any fine-tuning ===", flush=True)
        metrics, ranked, predictions, scores = evaluate(model, tokenizer, bundle,
                                                        args, device, amp_dtype)
        recorder.consider(0, metrics, tc.unwrap(model).state_dict(), ranked, predictions,
                          extra={"note": "pre-finetune, same eval path"},
                          always_write_ranking=args.keep_every_epoch)

    sampler = tc.GroupSampler(examples, documents, args.negatives,
                              args.passages_per_doc, seed=args.seed)
    groups_per_epoch = (len(examples) + args.batch_size - 1) // args.batch_size
    total_steps = max(1, args.epochs * groups_per_epoch // args.accum)
    optimizer = tc.make_optimizer(tc.unwrap(model), args.lr, args.weight_decay)
    scheduler = tc.make_scheduler(optimizer, total_steps, args.warmup_ratio)
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=(amp_dtype == torch.float16
                                                       and device == "cuda"))
    except TypeError:                                               # older torch
        scaler = torch.cuda.amp.GradScaler(enabled=(amp_dtype == torch.float16
                                                    and device == "cuda"))
    print(f"{len(examples)} training queries, {groups_per_epoch} groups/epoch, "
          f"{total_steps} optimizer steps over {args.epochs} epochs", flush=True)

    for epoch in range(1, args.epochs + 1):
        if recorder.already_done(epoch):
            print(f"\n=== epoch {epoch}/{args.epochs}: already recorded, skipping ===",
                  flush=True)
            continue
        print(f"\n=== epoch {epoch}/{args.epochs}: training ===", flush=True)
        loss = train_epoch(model, tokenizer, sampler, args, optimizer, scheduler,
                           scaler, device, amp_dtype, epoch)
        print(f"epoch {epoch} mean loss = {loss:.4f}", flush=True)

        print(f"=== epoch {epoch}: rescoring the `{TAG}` channel "
              f"({len(bundle.all_ids)} queries) ===", flush=True)
        metrics, ranked, predictions, scores = evaluate(model, tokenizer, bundle,
                                                        args, device, amp_dtype)
        improved = recorder.consider(epoch, metrics, tc.unwrap(model).state_dict(), ranked,
                                     predictions, extra={"train_loss": loss},
                                     always_write_ranking=args.keep_every_epoch)
        if improved:
            recorder.save_channel_scores(scores)
        recorder.package()

    print("\n=== summary ===", flush=True)
    print(json.dumps({"baseline": {k: baseline[k] for k in
                                   ("recall", "precision", "f2", "Recall@5")},
                      "best_epoch": recorder.best_epoch,
                      "output": str(recorder.dir)}, indent=2), flush=True)
    if recorder.best_epoch is None:
        print("No epoch was scored -- nothing to save.", flush=True)
    elif recorder.best_key <= recorder.baseline_key:
        print(f"Best epoch is {recorder.best_epoch}; its weights are saved, but it "
              f"is still below the cached baseline, so the shipped scores remain "
              f"the ones to submit.\nRemember this project's noise floor: a Recall "
              f"difference under 0.008 is not evidence of anything.", flush=True)


if __name__ == "__main__":
    main()
