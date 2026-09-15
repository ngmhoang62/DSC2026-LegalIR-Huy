"""Torch-side helpers shared by the three reranker fine-tuning scripts.

Scoring here reproduces the inference path of the shipped pipeline exactly:

  cross-encoder (Jina)   pairs = (question, passage), score = model logit,
                         document score = max over its passages
  bi-encoder (AITeamVN,  CLS token of last_hidden_state, L2-normalised, score =
  vnlegal-lal)           passage vector . question vector, max over passages

`PASSAGES_PER_DOC = 2` windows per document from `burst_common.top_passages`,
`max_length = 512`.  Training may use one window per document for speed
(--passages-per-doc), evaluation always uses two so the measured number is the
architecture the submission actually runs.
"""

from __future__ import annotations

import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import burst_common as bc


# --------------------------------------------------------------------------
# Where the weights come from
# --------------------------------------------------------------------------

def resolve_model_source(root, model_path, hf_repo):
    """A local weights directory if it really holds weights, else HuggingFace.

    The reproduction package ships models/ almost empty on purpose -- it holds
    a placeholder so the submission runner skips a download it does not need
    when the score caches are complete.  Fine-tuning does need the real
    weights, so an empty or placeholder-only directory has to fall through to
    the hub rather than be passed to from_pretrained as a bogus path.
    """
    path = Path(model_path)
    if not path.is_absolute():
        path = Path(root) / path
    if path.is_dir() and (any(path.glob("*.safetensors")) or any(path.glob("*.bin"))):
        print(f"Loading weights from {path}", flush=True)
        return str(path)
    reason = "no weights in" if path.exists() else "no directory"
    print(f"{reason} {path}; loading {hf_repo} from HuggingFace "
          f"(needs Kaggle internet enabled)", flush=True)
    return hf_repo


def enable_gradient_checkpointing(model, wanted):
    if not wanted or not hasattr(model, "gradient_checkpointing_enable"):
        return
    try:
        model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled", flush=True)
    except Exception as error:                                      # noqa: BLE001
        print(f"Gradient checkpointing unavailable: {error}", flush=True)


# --------------------------------------------------------------------------
# Setup
# --------------------------------------------------------------------------

def setup(seed=2026):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_gpus = torch.cuda.device_count() if device == "cuda" else 0
    native_bf16 = False
    if device == "cuda":
        for i in range(n_gpus):
            properties = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {properties.name} "
                  f"({properties.total_memory / 2**30:.0f} GB, "
                  f"compute {properties.major}.{properties.minor})", flush=True)
        # is_bf16_supported() answers True on Turing too, where bf16 is emulated
        # rather than run on tensor cores -- correct but slow.  Native bf16
        # starts at Ampere (8.0); a T4 is 7.5.
        native_bf16 = (torch.cuda.is_bf16_supported()
                       and torch.cuda.get_device_properties(0).major >= 8)
        if torch.cuda.is_bf16_supported() and not native_bf16:
            print("  bf16 is emulated on this GPU (pre-Ampere), so fp16 is the "
                  "faster default; the precision probe below is the safety net",
                  flush=True)
    else:
        print("No GPU", flush=True)
    amp_dtype = torch.bfloat16 if native_bf16 else torch.float16
    print(f"autocast dtype: {amp_dtype}", flush=True)
    return device, amp_dtype, n_gpus


def wrap_parallel(model, n_gpus):
    """Split every batch across all visible GPUs with nn.DataParallel.

    DataParallel rather than DDP because these runs are driven cell by cell in
    a notebook, where DDP's process group is more trouble than it is worth.  Its
    usual drawback -- gathering outputs onto GPU 0 -- barely applies here: a
    cross-encoder returns one logit per sequence and a bi-encoder one 1024-dim
    vector, so what crosses back is kilobytes while the activations stay put.
    """
    if n_gpus > 1:
        print(f"nn.DataParallel across {n_gpus} GPUs", flush=True)
        return torch.nn.DataParallel(model)
    return model


def unwrap(model):
    """The real module, whether or not DataParallel is in the way."""
    return getattr(model, "module", model)


def scale_for_gpus(args, n_gpus):
    """Give each GPU a slice of the same step instead of enlarging the step.

    batch_size grows by n_gpus and accum shrinks by the same factor, so the
    effective optimizer batch -- and therefore the training maths -- is
    unchanged from the single-GPU run; only the wall clock moves.  Evaluation
    batches just grow, since there is no optimizer step to keep equivalent.
    """
    if n_gpus <= 1:
        return args
    args.eval_batch_size *= n_gpus
    if args.accum % n_gpus == 0:
        args.batch_size *= n_gpus
        args.accum //= n_gpus
        print(f"Scaled for {n_gpus} GPUs: batch_size={args.batch_size} "
              f"accum={args.accum} (effective batch unchanged), "
              f"eval_batch_size={args.eval_batch_size}", flush=True)
    else:
        print(f"accum={args.accum} is not divisible by {n_gpus}; leaving "
              f"batch_size/accum alone, eval_batch_size={args.eval_batch_size}",
              flush=True)
    return args


def autocast(device, dtype, enabled=True):
    return torch.autocast(device_type="cuda" if device == "cuda" else "cpu",
                          dtype=dtype or torch.float16,
                          enabled=(device == "cuda" and enabled and dtype is not None))


@torch.inference_mode()
def probe_precision(model, tokenizer, texts, device, amp_dtype, max_length,
                    kind="bi", question="kiểm tra độ chính xác số học"):
    """Run one batch and report the largest activation each precision produces.

    Worth the ten seconds.  A T4 has no bf16, so autocast falls back to fp16,
    whose exponent range stops at 65504 -- and a model trained in bf16 (as
    LLM-style embedders are) can carry activations far past that.  They become
    inf, F.normalize turns inf/inf into NaN, and the loss is NaN from the first
    step with nothing in the log to say why.  Better to find out here.
    """
    if device != "cuda":
        return True, 0.0
    model.eval()
    sample = texts[:4] if len(texts) >= 4 else texts
    with autocast(device, amp_dtype):
        if kind == "cross":
            out = cross_encoder_logits(model, tokenizer, [question] * len(sample),
                                       sample, max_length, device)
        else:
            out = encode_cls_grad(model, tokenizer, sample, max_length, device)
    finite = bool(torch.isfinite(out).all())
    peak = float(out.abs()[torch.isfinite(out)].max()) if torch.isfinite(out).any() else float("inf")
    print(f"  {amp_dtype} probe: finite={finite} max|out|={peak:.3g}", flush=True)
    return finite, peak


def choose_precision(model, tokenizer, texts, device, amp_dtype, max_length,
                     kind="bi", requested="auto"):
    """Settle on a precision before burning an epoch on NaNs.

    'auto' probes the requested autocast dtype and drops to fp32 if the forward
    pass is not finite.  fp32 costs roughly 2x the time and memory, which is a
    far better trade than an epoch of NaN.
    """
    if requested == "fp32":
        print("Precision: fp32 (autocast off, requested)", flush=True)
        return None
    if requested in ("fp16", "bf16"):
        forced = torch.float16 if requested == "fp16" else torch.bfloat16
        print(f"Precision: {forced} (requested, not probed)", flush=True)
        return forced
    finite, _ = probe_precision(model, tokenizer, texts, device, amp_dtype,
                                max_length, kind)
    if finite:
        print(f"Precision: {amp_dtype} autocast", flush=True)
        return amp_dtype
    print(f"Precision: {amp_dtype} overflowed on this model -- falling back to "
          f"fp32.\n  Slower and heavier; if it runs out of memory, lower "
          f"--batch-size or --negatives.", flush=True)
    return None


@torch.inference_mode()
def check_pooling_discriminates(model, tokenizer, texts, device, amp_dtype,
                                max_length, kind="bi", label="model",
                                question="văn bản pháp luật về thuế thu nhập"):
    """Refuse to train a model whose pooling cannot tell two passages apart.

    A causal decoder masks position 0 to attend only to itself, so its
    position-0 ("CLS") hidden state is a function of the first token alone --
    the same vector for every passage that starts the same way.  Such a channel
    cannot rank, and training it only teaches the optimizer to drive that one
    token embedding to infinity, which is what `embed_tokens.weight` going
    non-finite looks like from the outside.

    Ten seconds here saves an epoch of NaN and a wrong conclusion.
    """
    if device != "cuda" or len(texts) < 3:
        return True
    model.eval()
    with autocast(device, amp_dtype):
        if kind == "cross":
            values = cross_encoder_logits(model, tokenizer,
                                          [question] * len(texts), texts,
                                          max_length, device).float().cpu().numpy()
            spread = float(np.std(values))
            distinct = len(set(np.round(values, 6)))
        else:
            vectors = encode_cls_grad(model, tokenizer, texts, max_length,
                                      device).float().cpu().numpy()
            similarity = vectors @ vectors.T
            off = similarity[~np.eye(len(texts), dtype=bool)]
            spread = float(1.0 - off.mean())
            distinct = len(set(np.round(off, 6)))
    print(f"  pooling check: {distinct} distinct values over {len(texts)} "
          f"passages, spread={spread:.5f}", flush=True)
    if distinct > 2 and spread > 1e-3:
        return True
    raise RuntimeError(
        f"`{label}` gives near-identical outputs for different passages "
        f"({distinct} distinct values, spread {spread:.2e}). Its pooling reads a "
        f"position whose representation does not depend on the passage -- the "
        f"signature of first-token pooling on a causal decoder.\n"
        f"  This channel cannot rank, and training it will drive the token "
        f"embedding to infinity rather than learn. Fine-tune a model whose "
        f"pooling actually sees the text, or change the pooling -- but if you "
        f"change it, A/B the result on the real leaderboard, not on CV.")


def find_layers(model):
    """The ModuleList of transformer blocks, whatever this architecture calls it.

    Encoder models expose it as encoder.layer, decoder-style embedders as
    model.layers; picking the longest ModuleList finds either.
    """
    best = None
    for name, child in unwrap(model).named_modules():
        if isinstance(child, torch.nn.ModuleList) and len(child) >= 4:
            if best is None or len(child) > len(best[1]):
                best = (name, child)
    return best


def freeze_lower_layers(model, keep_top):
    """Train only the top `keep_top` blocks, plus the head; freeze the rest.

    With 1,050 training queries against a ~570M-parameter model, full
    fine-tuning is wildly over-parameterised, and every frozen tensor is one
    fewer place for a run to diverge.  It also cuts optimizer memory enough
    that fp32 fits comfortably.
    """
    base = unwrap(model)
    total = sum(p.numel() for p in base.parameters())
    if keep_top <= 0:
        return total, total
    found = find_layers(base)
    if found is None:
        print("  could not locate transformer blocks; training everything",
              flush=True)
        return total, total
    prefix, layers = found
    first_kept = max(0, len(layers) - keep_top)
    kept = tuple(f"{prefix}.{i}." for i in range(first_kept, len(layers)))
    frozen_embeddings = {id(p) for module in base.modules()
                         if isinstance(module, torch.nn.Embedding)
                         for p in module.parameters()}
    trainable = 0
    for name, param in base.named_parameters():
        if name.startswith(prefix + "."):
            param.requires_grad = name.startswith(kept)
        else:
            param.requires_grad = id(param) not in frozen_embeddings
        if param.requires_grad:
            trainable += param.numel()
    print(f"  training blocks {first_kept}-{len(layers)-1} of {prefix} "
          f"+ head: {trainable/1e6:.0f}M / {total/1e6:.0f}M parameters "
          f"({trainable/total:.0%})", flush=True)
    return trainable, total


def report_memory_budget(model, n_gpus, device="cuda"):
    """Show where GPU 0's memory goes before the first step, not after an OOM.

    GPU 0 carries everything DataParallel does not shard: the weights, the
    gradients, both AdamW moments, and the flat buffer reduce_add_coalesced
    needs to sum the other GPU's gradients.  Two T4s give 2x15 GB of compute
    but only one of them holds the optimizer, so "we have 30 GB" is the wrong
    way to read the budget.
    """
    if device != "cuda":
        return True
    base = unwrap(model)
    total = sum(p.numel() for p in base.parameters())
    trainable = sum(p.numel() for p in base.parameters() if p.requires_grad)
    gigabytes = lambda count: count * 4 / 2**30

    weights = gigabytes(total)
    grads = gigabytes(trainable)
    adam = gigabytes(trainable) * 2
    reduce_buffer = gigabytes(trainable) if n_gpus > 1 else 0.0
    fixed = weights + grads + adam + reduce_buffer
    capacity = torch.cuda.get_device_properties(0).total_memory / 2**30
    headroom = capacity - fixed

    print(f"  trainable {trainable/1e6:.0f}M / {total/1e6:.0f}M "
          f"({trainable/max(total,1):.0%})", flush=True)
    print(f"  GPU 0 budget: weights {weights:.2f} + grads {grads:.2f} + "
          f"AdamW {adam:.2f}" + (f" + DP reduce {reduce_buffer:.2f}"
                                 if reduce_buffer else "") +
          f" = {fixed:.2f} GB of {capacity:.2f} GB "
          f"({headroom:.2f} GB left for activations)", flush=True)

    advice = (
        "  Fixes, most effective first:\n"
        "    TRAIN_TOP_LAYERS = 8   freeze the lower blocks (biggest win: grads "
        "and AdamW shrink with the number of TRAINED parameters)\n"
        "    MAX_GPUS = 1           drop DataParallel, which keeps a full extra "
        "gradient buffer on GPU 0\n"
        "    BATCH_SIZE / 2, ACCUM x 2\n"
        "  Note BATCH_SIZE and MAX_LENGTH only shrink activations -- they do "
        "nothing for the fixed cost above, which is why lowering them alone "
        "does not stop this OOM.")
    if headroom < 2.0:
        raise RuntimeError(
            f"Only {headroom:.2f} GB would be left for activations on GPU 0 -- "
            f"this will run out of memory a few batches in, once AdamW "
            f"allocates its state.\n{advice}")
    if headroom < 5.0:
        print(f"  WARNING: only {headroom:.2f} GB left for activations; CUDA OOM "
              f"is likely.\n{advice}", flush=True)
        return False
    return True


def step_if_finite(model, optimizer, scaler, max_grad_norm):
    """Apply the optimizer step only when the gradients are finite.

    GradScaler already refuses to step on inf/NaN gradients -- but only while
    it is enabled, and it is disabled in fp32, which is exactly the mode the
    precision probe selects for a model fp16 cannot hold.  A finite loss can
    still back-propagate a NaN (a degenerate batch, a diverging step), and one
    of those turns every weight NaN for the rest of the run: the next epoch
    then reports 100% skipped batches with nothing to say why.  So the check
    lives here, where it covers both precisions.

    clip_grad_norm_ returns the pre-clip total norm, so the test costs nothing
    extra.
    """
    scaler.unscale_(optimizer)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    finite = bool(torch.isfinite(norm))
    if finite:
        scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    return finite


def first_bad_parameter(model):
    """Name of the first weight that has gone inf/NaN, or None if all are fine."""
    for name, param in unwrap(model).named_parameters():
        if not torch.isfinite(param).all():
            return name
    return None


def guard_parameters(model, epoch, when):
    """Fail loudly the moment the weights are unusable, not an epoch later."""
    bad = first_bad_parameter(model)
    if bad is None:
        return
    if when == "before":
        raise RuntimeError(
            f"Weights are already non-finite before epoch {epoch} began "
            f"(first bad tensor: {bad}). They were destroyed during an earlier "
            f"epoch, so nothing this epoch computes can mean anything.\n"
            f"  Recover the last checkpoint that beat the baseline with "
            f"recorder.load_best_state(model), or reload the model from "
            f"scratch, then re-run with a lower LR (try LR/4).")
    raise RuntimeError(
        f"Weights went non-finite during epoch {epoch} (first bad tensor: "
        f"{bad}). The run diverged.\n"
        f"  Reload the model, then lower the learning rate (try LR/4) before "
        f"trying again; raising --accum or lowering --temperature also helps.")


def report_skipped(skipped, processed):
    """Say plainly how much of the epoch was thrown away, and refuse a dead one."""
    if not skipped:
        return
    share = skipped / max(processed, 1)
    print(f"  WARNING: {skipped}/{processed} batches ({share:.0%}) had a "
          f"non-finite loss and were skipped.", flush=True)
    if share >= 1.0:
        raise RuntimeError(
            "Every batch produced a NaN loss -- this epoch trained on nothing.\n"
            "  If this is the FIRST epoch: fp16 overflow on a model trained in "
            "bf16, which a T4 cannot run natively. Re-run with PRECISION='fp32'.\n"
            "  If an earlier epoch ran fine: the weights were destroyed by that "
            "epoch. Reload the model and lower the learning rate (try LR/4).")
    if share > .2:
        print("  More than a fifth of the epoch was skipped -- treat this "
              "checkpoint as unreliable.", flush=True)


def make_optimizer(model, lr, weight_decay):
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        (no_decay if param.ndim == 1 or name.endswith(".bias") else decay).append(param)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": weight_decay},
         {"params": no_decay, "weight_decay": 0.0}], lr=lr)


def make_scheduler(optimizer, total_steps, warmup_ratio):
    warmup = max(1, int(total_steps * warmup_ratio))

    def factor(step):
        if step < warmup:
            return step / warmup
        remaining = max(1, total_steps - warmup)
        return max(0.0, (total_steps - step) / remaining)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


# --------------------------------------------------------------------------
# Passage plumbing
# --------------------------------------------------------------------------

def passages_for(question, documents, docs, count):
    """Flat (owner, passage) lists for one query, in the pipeline's own order."""
    owners, passages = [], []
    for doc in docs:
        for passage in bc.top_passages(question, documents[doc], count=count):
            owners.append(doc)
            passages.append(passage)
    return owners, passages


def reduce_max(owners, values, docs, floor=-1e9):
    """Document score = max over its passages -- the pipeline's pooling rule.

    Returns the non-finite count alongside, because silence here is dangerous:
    `max(-1e9, nan)` is -1e9 in Python (nan > -1e9 is False), so a model that
    has gone NaN leaves every document tied at the floor.  rank_by then orders
    them by document id, and the channel scores a perfectly plausible-looking
    Recall@5 that measures nothing at all.  That is how a dead checkpoint can
    read as an improvement over the cached one.
    """
    out = {d: floor for d in docs}
    bad = 0
    for doc, value in zip(owners, values):
        number = float(value)
        if number != number or number in (float("inf"), float("-inf")):
            bad += 1
            continue
        out[doc] = max(out[doc], number)
    return out, bad


def report_bad_scores(bad, total, label):
    """Refuse to hand back scores a broken model produced."""
    if not bad:
        return
    share = bad / max(total, 1)
    print(f"  WARNING: {bad:,}/{total:,} ({share:.1%}) of `{label}` passage "
          f"scores were non-finite.", flush=True)
    if share > .5:
        raise RuntimeError(
            f"`{label}` produced non-finite scores for {share:.0%} of passages -- "
            f"these weights are broken, and the metrics computed from them are "
            f"meaningless (every document ties at the floor and gets ordered by "
            f"document id).\n"
            f"  Reload the model and re-run training; if it keeps happening, "
            f"lower the learning rate.")


# --------------------------------------------------------------------------
# Cross-encoder (Jina) scoring
# --------------------------------------------------------------------------

def cross_encoder_logits(model, tokenizer, questions, passages, max_length,
                         device="cuda"):
    """Differentiable equivalent of the model's own compute_score().

    Inputs go to `device` (GPU 0 under DataParallel, which scatters from there)
    rather than to model.device, which a DataParallel wrapper does not have.
    """
    batch = tokenizer(questions, passages, padding=True, truncation=True,
                      max_length=max_length, return_tensors="pt")
    batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
    return model(**batch, return_dict=True).logits.view(-1).float()


@torch.inference_mode()
def score_cross_encoder(model, tokenizer, questions, pool, documents, ids,
                        device, amp_dtype, max_length=bc.MAX_LENGTH,
                        batch_size=16, passages_per_doc=bc.PASSAGES_PER_DOC,
                        label="jina"):
    """Score every (query, candidate) pair the way the submission runner does."""
    model.eval()
    scores = {}
    started = time.perf_counter()
    bar = bc.progress(ids, desc=f"score {label}", unit="q")
    seen_passages, bad_total = 0, 0
    for q in bar:
        question = questions[q]
        owners, passages = passages_for(question, documents, pool[q], passages_per_doc)
        values = []
        with autocast(device, amp_dtype):
            for start in range(0, len(passages), batch_size):
                chunk = passages[start:start + batch_size]
                values.extend(cross_encoder_logits(
                    model, tokenizer, [question] * len(chunk), chunk,
                    max_length, device).cpu().tolist())
        scores[q], bad = reduce_max(owners, values, pool[q])
        bad_total += bad
        seen_passages += len(passages)
        bar.set_postfix(passages=seen_passages, bad=bad_total,
                        rate=f"{seen_passages/max(time.perf_counter()-started, 1e-9):.0f}/s")
    print(f"  {label}: {seen_passages:,} đoạn văn trong "
          f"{(time.perf_counter()-started)/60:.1f} phút", flush=True)
    report_bad_scores(bad_total, seen_passages, label)
    return scores


# --------------------------------------------------------------------------
# Bi-encoder (AITeamVN, vnlegal-lal) scoring -- CLS pooling, L2 normalised
# --------------------------------------------------------------------------

def encode_cls_grad(model, tokenizer, texts, max_length, device="cuda"):
    """benchmark_aiteamvn_holdouts.encode_cls, kept differentiable.

    CLS pooling is deliberate for vnlegal-lal: last-token pooling scored better
    on CV (F2 +0.0127) but lost real leaderboard recall and was reverted.
    """
    batch = tokenizer(texts, max_length=max_length, padding=True, truncation=True,
                      return_tensors="pt")
    batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
    cls = model(**batch).last_hidden_state[:, 0]
    return F.normalize(cls.float(), p=2, dim=1)


@torch.inference_mode()
def encode_cls(model, tokenizer, texts, batch_size, max_length, device, amp_dtype):
    vectors = []
    with autocast(device, amp_dtype):
        for start in range(0, len(texts), batch_size):
            vectors.append(encode_cls_grad(
                model, tokenizer, texts[start:start + batch_size], max_length,
                device).float().cpu().numpy())
    return np.vstack(vectors) if vectors else np.zeros((0, 1), dtype=np.float32)


@torch.inference_mode()
def score_bi_encoder(model, tokenizer, questions, pool, documents, ids,
                     device, amp_dtype, max_length=bc.MAX_LENGTH,
                     batch_size=32, passages_per_doc=bc.PASSAGES_PER_DOC,
                     label="dense"):
    model.eval()
    scores = {}
    started = time.perf_counter()
    bar = bc.progress(ids, desc=f"score {label}", unit="q")
    seen_passages, bad_total = 0, 0
    for q in bar:
        question = questions[q]
        owners, passages = passages_for(question, documents, pool[q], passages_per_doc)
        qvec = encode_cls(model, tokenizer, [question], 1, max_length,
                          device, amp_dtype)[0]
        pvec = encode_cls(model, tokenizer, passages, batch_size, max_length,
                          device, amp_dtype)
        scores[q], bad = reduce_max(owners, pvec @ qvec, pool[q])
        bad_total += bad
        seen_passages += len(passages)
        bar.set_postfix(passages=seen_passages, bad=bad_total,
                        rate=f"{seen_passages/max(time.perf_counter()-started, 1e-9):.0f}/s")
    print(f"  {label}: {seen_passages:,} đoạn văn trong "
          f"{(time.perf_counter()-started)/60:.1f} phút", flush=True)
    report_bad_scores(bad_total, seen_passages, label)
    return scores


# --------------------------------------------------------------------------
# Training batches
# --------------------------------------------------------------------------

class GroupSampler:
    """One training group = a query, one of its gold docs, and hard negatives.

    Negatives are drawn without replacement from the cached lexical pool, so
    across epochs the model sees a different slice of the same distractor set
    the deployed retrieval actually produces.
    """

    def __init__(self, examples, documents, negatives, passages_per_doc, seed=2026):
        self.examples = examples
        self.documents = documents
        self.negatives = negatives
        self.passages_per_doc = passages_per_doc
        self.rng = random.Random(seed)

    def epoch(self, epoch):
        order = list(range(len(self.examples)))
        random.Random(self.rng.randint(0, 2**31) + epoch).shuffle(order)
        return order

    def group(self, index):
        example = self.examples[index]
        positive = self.rng.choice(example.positives)
        pool = list(example.negatives)
        self.rng.shuffle(pool)
        negatives = pool[:self.negatives]
        docs = [positive] + negatives
        texts = []
        for doc in docs:
            windows = bc.top_passages(example.question, self.documents[doc],
                                      count=self.passages_per_doc)
            texts.append(windows[0] if len(windows) == 1
                         else windows[self.rng.randrange(len(windows))])
        return example.question, texts, len(docs)


def pairwise_loss(scores, n_negatives):
    """RankNet logistic loss over (positive, negative) pairs within a group.

    Matches the objective the shipped checkpoint is named for
    (results/jina_reranker/burst_pairwise_state.pt).
    """
    positive = scores[0]
    negatives = scores[1:1 + n_negatives]
    return F.softplus(negatives - positive).mean()


def listwise_loss(scores, n_negatives, temperature=1.0):
    """Softmax cross-entropy with the positive at index 0."""
    logits = scores[:1 + n_negatives].unsqueeze(0) / temperature
    target = torch.zeros(1, dtype=torch.long, device=scores.device)
    return F.cross_entropy(logits, target)


def infonce_loss(qvec, dvec, group_sizes, temperature=.05, in_batch=True):
    """InfoNCE for the bi-encoders: positive at the head of each group.

    in_batch=True also treats the other groups' documents as negatives, which
    is free (they are already encoded) and sharpens the embedding space beyond
    what the mined hard negatives alone reach.
    """
    losses = []
    offset = 0
    similarity = qvec @ dvec.T / temperature          # (groups, all documents)
    for i, size in enumerate(group_sizes):
        if in_batch:
            logits = similarity[i]
            target = offset
        else:
            logits = similarity[i, offset:offset + size]
            target = 0
        losses.append(F.cross_entropy(
            logits.unsqueeze(0),
            torch.tensor([target], device=logits.device, dtype=torch.long)))
        offset += size
    return torch.stack(losses).mean()
