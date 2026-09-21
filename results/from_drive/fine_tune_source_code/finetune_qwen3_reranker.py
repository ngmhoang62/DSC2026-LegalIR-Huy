"""Fine-tune Qwen/Qwen3-Reranker-4B as a CANDIDATE for the `jina` channel.

Architecture: generative yes/no cross-encoder, not a sequence-classification
head like jina's own model. Qwen3-Reranker (and the Prism-Qwen3.5 family
reused via finetune_prism_reranker.py) reads relevance off the next-token
distribution after a chat prompt -- P(yes) vs P(no) -- exactly as documented
on both model cards (score = sigmoid(logit(yes) - logit(no))). See
torch_common.causal_yesno_logits/score_causal_yesno for the shared
implementation.

Why this substitutes for `jina` and not `dense`/`aiteamvn`
------------------------------------------------------------
The `dense` channel (AITeamVN) also feeds two OTHER candidate-generation
tiers -- dense expansion union-50 and the corpus dense index -- both of which
need a real embedding vector per document to do full-corpus similarity. A
generative yes/no reranker can only score (query, candidate) pairs already
retrieved; it cannot produce those vectors. So this model (and Prism) can
only ever stand in for a LATE reranking channel, i.e. `jina`'s role, never
`dense`'s.

This script's output tag is "qwen3_reranker" (its own results/ subfolder,
kept separate from finetune_jina.py's own "jina" run), but it substitutes
into the LTR fusion under the CHANNEL key "jina" -- score_overrides={"jina":
...} -- exactly the substitution mechanism finetune_jina.py itself uses.
That makes the two directly comparable: whichever model gives a higher fused
`recall` in its own history.json is the better candidate for shipping as the
`jina` channel.

Why this does NOT reuse tc.choose_precision / tc.load_fp32 / GradScaler
--------------------------------------------------------------------------
Those exist for the OTHER two scripts, which full-fine-tune a small (<600M)
model loaded in fp32 with autocast picking bf16/fp16 per batch. A 4B/2B
causal LM does not fit VRAM that way. Instead: base weights frozen in fp16,
LoRA adapters (peft) upcast to fp32 -- the same recipe already used and
documented in vietlegal-tune-kaggle-hnsw.ipynb. No autocast is needed because
the dtype split is fixed at load time, and peft casts the LoRA branch's input
to fp32 before the adapter matmul, so activations after any LoRA-touched
layer are already fp32 -- no GradScaler either, same as that notebook.

Reused from results/ (never recomputed, no GPU spent):
    layer 1  BM25 multi-branch retrieval, multistage top-20, dense expansion,
             corpus dense rank cap=32  -> the candidate pool is byte-identical
             every epoch
    layer 2  the `dense` and `vnlegal_lal` channels stay on their cached
             scores; only the `jina` slot is replaced by this model's output
Recomputed each epoch:
    layer 2  the (substituted) `jina` slot only
    layer 3  the 6-channel LTR fusion (CPU, ~2 s)
    layer 4  dynamic threshold alpha=0.15

Outputs, under --work/qwen3_reranker/ (default /kaggle/working/qwen3_reranker/):
    best_state.pt            {"state_dict": <LoRA params only>} -- a few MB,
                             NOT the 4B base (see lora_state_dict() below)
    best_predictions.json    {qid: {"answer": [...]}} after the dynamic threshold
    best_ranking.json        top-20 fused ranking per query
    best_channel_scores.pkl  {qid: {doc: score}} -- feeds back as a jina-slot
                             score cache
    history.json             every epoch's metrics, including the cached
                             baseline (the same 0.9511/0.6144 finetune_jina.py
                             compares against)

Usage
    python finetune_qwen3_reranker.py --epochs 3
    python finetune_qwen3_reranker.py --epochs 3 --eval-before-training
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import burst_common as bc
import torch_common as tc

TAG = "qwen3_reranker"                         # output subfolder; NOT the LTR channel key
SUBSTITUTES_CHANNEL = "jina"                   # LTR channel key this model's scores replace
DEFAULT_MODEL = "models/qwen3-reranker-4b"
HF_REPO = "Qwen/Qwen3-Reranker-4B"
SYSTEM_PROMPT = ('Judge whether the Document meets the requirements based on the '
                 'Query and the Instruct provided. Note that the answer can only '
                 'be "yes" or "no".')
DEFAULT_INSTRUCTION = ('Given a Vietnamese legal question, determine whether the '
                       'Document contains the answer to the Query')


def load_model(root, model_path, gradient_checkpointing, lora_r, lora_alpha,
               lora_dropout, lora_target_modules):
    source = tc.resolve_model_source(root, model_path, HF_REPO)
    tokenizer = AutoTokenizer.from_pretrained(source, padding_side="left")
    base_model = AutoModelForCausalLM.from_pretrained(source, dtype=torch.float16)

    from peft import LoraConfig, TaskType, get_peft_model

    # Kaggle/Colab sometimes ship a torchao older than peft's minimum; peft probes
    # is_torchao_available() for every Linear layer it wraps regardless of whether
    # quantization is in use, and raises ImportError instead of returning False on
    # a too-old version. Neutralise that probe -- this model runs plain fp16/fp32,
    # never torchao. No-op if torchao is absent or peft no longer imports it.
    try:
        import peft.tuners.lora.torchao as _lora_torchao
        _lora_torchao.is_torchao_available = lambda: False
    except ImportError:
        pass

    lora_config = LoraConfig(
        r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
        target_modules=lora_target_modules, bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(base_model, lora_config)
    # Base frozen at fp16; LoRA (A/B) upcast to fp32 so the optimizer's master
    # weights are full precision -- standard mixed-precision LoRA. peft casts
    # the adapter's input to lora_A's dtype before the matmul, so mixing dtypes
    # here is safe and needs no autocast wrapper at train time.
    trainable = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            param.data = param.data.float()
            trainable += param.numel()
    total = sum(p.numel() for p in model.parameters())
    print(f"LoRA: {trainable/1e6:.1f}M / {total/1e6:.0f}M trainable "
          f"({trainable/max(total,1):.2%})", flush=True)

    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()   # required: base is frozen, so the
                                              # checkpointing backward chain would
                                              # otherwise break at the input embedding
        print("Gradient checkpointing enabled", flush=True)

    model._tokenizer = tokenizer
    return model, tokenizer


def lora_state_dict(model):
    """Only the LoRA A/B tensors -- a few MB, not the frozen 4B/2B base.

    RunRecorder._save_state writes whatever dict it is handed and
    load_best_state() restores with strict=False, so handing it this filtered
    dict on both ends is a correct, minimal checkpoint: the base weights come
    fresh from HuggingFace every run, and only the adapter deltas need to
    persist.
    """
    return {k: v for k, v in tc.unwrap(model).state_dict().items() if "lora_" in k}


@torch.no_grad()
def sanity_check(model, tokenizer, documents, probe_docs, device, max_length,
                 instruction, system_prompt, label):
    """Cheap replacement for tc.choose_precision/check_pooling_discriminates.

    Those two assume a model loaded via tc.load_fp32 with autocast deciding
    precision per batch -- not this model's fixed fp16-base/fp32-LoRA setup.
    What still matters here is the same two questions they answer: does the
    forward pass produce finite numbers, and does the model actually tell
    different passages apart (as opposed to reading a position whose
    representation ignores the passage, which would make training push the
    embedding table toward infinity instead of learning anything)?
    """
    model.eval()
    texts = [documents[d] for d in probe_docs]
    question = "văn bản pháp luật về thuế thu nhập"
    values = tc.causal_yesno_logits(model, tokenizer, [question] * len(texts),
                                    texts, max_length, device, instruction,
                                    system_prompt=system_prompt)
    values = values.float().cpu().numpy()
    finite = bool((values == values).all() and (abs(values) < float("inf")).all())
    spread = float(values.std())
    distinct = len(set(values.round(6)))
    print(f"  sanity check `{label}`: finite={finite} distinct={distinct}/"
          f"{len(texts)} spread={spread:.4f}", flush=True)
    if not finite:
        raise RuntimeError(
            f"`{label}` produced non-finite yes/no logits on the very first probe "
            f"batch -- fp16 is overflowing for this model. Nothing downstream can "
            f"be trusted; check the base model loaded correctly.")
    if distinct <= 2 or spread < 1e-3:
        raise RuntimeError(
            f"`{label}` gives near-identical yes/no scores for different passages "
            f"({distinct} distinct values, spread {spread:.2e}). This channel "
            f"cannot rank -- training it will not learn anything useful.")


def train_epoch(model, tokenizer, sampler, args, optimizer, scheduler, device, epoch):
    tc.guard_parameters(model, epoch, "before")
    model.train()
    order = sampler.epoch(epoch)
    total, finite, skipped, processed, steps, bad_steps = 0.0, 0, 0, 0, 0, 0
    optimizer.zero_grad(set_to_none=True)
    starts = range(0, len(order), args.batch_size)
    bar = bc.progress(starts, desc=f"epoch {epoch} train", total=len(starts), unit="batch")
    for start in bar:
        indices = order[start:start + args.batch_size]
        questions, passages, sizes = [], [], []
        for index in indices:
            question, texts, size = sampler.group(index)
            questions.extend([question] * size)
            passages.extend(texts)
            sizes.append(size)
        logits = tc.causal_yesno_logits(model, tokenizer, questions, passages,
                                        args.max_length, device, args.instruction,
                                        system_prompt=args.system_prompt)
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
            loss.backward()
            total += float(loss.detach()) * args.accum
            finite += 1
        else:
            skipped += 1
        if processed % args.accum == 0:
            norm = torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], args.max_grad_norm)
            ok = bool(torch.isfinite(norm))
            if ok:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            bad_steps += not ok
            scheduler.step()
            steps += 1
        bar.set_postfix(loss=f"{total/max(finite,1):.4f}", step=steps,
                        skipped=skipped, bad=bad_steps,
                        lr=f"{scheduler.get_last_lr()[0]:.1e}")
    if processed % args.accum:
        norm = torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], args.max_grad_norm)
        if torch.isfinite(norm):
            optimizer.step()
        else:
            bad_steps += 1
        optimizer.zero_grad(set_to_none=True)
    if bad_steps:
        print(f"  {bad_steps}/{steps} optimizer steps had non-finite gradients "
              f"and were not applied.", flush=True)
    tc.report_skipped(skipped, processed)
    tc.guard_parameters(model, epoch, "after")
    return total / max(finite, 1)


def evaluate(model, tokenizer, bundle, args, device, tag):
    questions = {q: bundle.queries[q][0] for q in bundle.all_ids}
    scores = tc.score_causal_yesno(
        model, tokenizer, questions, bundle.extended, bundle.documents,
        bundle.all_ids, device, args.instruction, system_prompt=args.system_prompt,
        max_length=args.max_length, batch_size=args.eval_batch_size, label=tag)
    views = {SUBSTITUTES_CHANNEL: bc.rank_by(bundle.extended, scores)}
    metrics, ranked, predictions = bc.lobo_evaluate(
        bundle, {SUBSTITUTES_CHANNEL: scores}, views)
    metrics["channel_only"] = bc.channel_metrics(bundle, scores)
    return metrics, ranked, predictions, scores


def add_reranker_args(ap, default_model):
    ap.add_argument("--model-path", default=default_model)
    ap.add_argument("--loss", choices=("pairwise", "listwise"), default="pairwise")
    ap.add_argument("--temperature", type=float, default=1.0, help="listwise loss only")
    ap.add_argument("--passages-per-doc", type=int, default=1,
                    help="windows per document during TRAINING; evaluation always "
                         "uses the pipeline's 2")
    ap.add_argument("--instruction", default=DEFAULT_INSTRUCTION,
                    help="the <Instruct>: line in the yes/no prompt")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--lora-target-modules", default="all-linear")
    ap.set_defaults(lr=2e-4, batch_size=2, accum=8, negatives=3, eval_batch_size=8,
                    max_length=1024)
    return ap


def run(tag, default_model, hf_repo, system_prompt, load_model_fn):
    ap = bc.common_args(__doc__.splitlines()[0])
    add_reranker_args(ap, default_model)
    args = ap.parse_args()
    args.system_prompt = system_prompt

    device, _, n_gpus = tc.setup(args.seed)
    if args.max_gpus:
        n_gpus = min(n_gpus, args.max_gpus)
    args.work.mkdir(parents=True, exist_ok=True)

    print("\n=== stage 1-2 cache: rebuilding the 600-query holdout ===", flush=True)
    documents = bc.DocumentStore(Path(args.root) / bc.DATA_SUBDIR, preload=True)
    bundle = bc.build_eval_bundle(args.root, documents)

    print("\n=== cached baseline (all six channels from results/) ===", flush=True)
    baseline, _, _ = bc.lobo_evaluate(bundle)
    print(f"recall={baseline['recall']:.4f} precision={baseline['precision']:.4f} "
          f"f2={baseline['f2']:.4f} Recall@5={baseline['Recall@5']:.4f} "
          f"answers/query={baseline['mean_answers']:.3f}", flush=True)
    recorder = bc.RunRecorder(args.work, tag, baseline, resume=args.resume)

    print("\n=== training pool from the cached retrieval ===", flush=True)
    examples = bc.build_train_pool(args.root, negatives=args.negative_depth,
                                   limit=args.train_queries, seed=args.seed,
                                   documents=documents)

    print(f"\n=== model: {hf_repo} (LoRA r={args.lora_r}) ===", flush=True)
    model, tokenizer = load_model_fn(args.root, args.model_path,
                                     args.gradient_checkpointing, args.lora_r,
                                     args.lora_alpha, args.lora_dropout,
                                     args.lora_target_modules)
    model.to(device)
    if args.resume:
        loaded = recorder.load_best_state(model)
        if not loaded:
            print(f"[{tag}] no prior best_state.pt -- starting LoRA from scratch",
                  flush=True)

    probe_docs = examples[0].negatives[:4]
    sanity_check(model, tokenizer, documents, probe_docs, device, args.max_length,
                args.instruction, args.system_prompt, tag)

    if device == "cuda":
        free, total = torch.cuda.mem_get_info()
        print(f"  GPU memory after load: {(total-free)/2**30:.2f} / {total/2**30:.2f} "
              f"GB used", flush=True)

    model = tc.wrap_parallel(model, n_gpus)

    if args.eval_before_training:
        print("\n=== epoch 0: this model before any fine-tuning ===", flush=True)
        metrics, ranked, predictions, scores = evaluate(model, tokenizer, bundle,
                                                        args, device, tag)
        recorder.consider(0, metrics, lora_state_dict(model), ranked, predictions,
                          extra={"note": "pre-finetune, same eval path"},
                          always_write_ranking=args.keep_every_epoch)

    sampler = tc.GroupSampler(examples, documents, args.negatives,
                              args.passages_per_doc, seed=args.seed)
    groups_per_epoch = (len(examples) + args.batch_size - 1) // args.batch_size
    total_steps = max(1, args.epochs * groups_per_epoch // args.accum)
    optimizer = tc.make_optimizer(tc.unwrap(model), args.lr, args.weight_decay)
    scheduler = tc.make_scheduler(optimizer, total_steps, args.warmup_ratio)
    print(f"{len(examples)} training queries, {groups_per_epoch} groups/epoch, "
          f"{total_steps} optimizer steps over {args.epochs} epochs", flush=True)

    for epoch in range(1, args.epochs + 1):
        if recorder.already_done(epoch):
            print(f"\n=== epoch {epoch}/{args.epochs}: already recorded, skipping ===",
                  flush=True)
            continue
        print(f"\n=== epoch {epoch}/{args.epochs}: training ===", flush=True)
        loss = train_epoch(model, tokenizer, sampler, args, optimizer, scheduler,
                           device, epoch)
        print(f"epoch {epoch} mean loss = {loss:.4f}", flush=True)

        print(f"=== epoch {epoch}: rescoring the `{SUBSTITUTES_CHANNEL}` slot with "
              f"`{tag}` ({len(bundle.all_ids)} queries) ===", flush=True)
        metrics, ranked, predictions, scores = evaluate(model, tokenizer, bundle,
                                                        args, device, tag)
        improved = recorder.consider(epoch, metrics, lora_state_dict(model), ranked,
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


def main():
    run(TAG, DEFAULT_MODEL, HF_REPO, SYSTEM_PROMPT, load_model)


if __name__ == "__main__":
    main()
