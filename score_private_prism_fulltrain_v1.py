#!/usr/bin/env python
"""
SCORE PRIVATE D1 CANDIDATES WITH FULL-TRAIN PRISM-2B LORA
=========================================================

Self-contained inference runner. No friend's inference source code required.

Base:
  infgrad/Prism-Qwen3.5-Reranker-2B

Checkpoint:
  PEFT LoRA-only `best_state.pt` with {"state_dict": ...}

Scoring:
  raw relevance logit = logit("yes") - logit("no")
  at the first assistant token, using Prism's official fixed prompt.

Why raw logits?
  The supplied held-out `best_channel_scores.pkl` is in raw-logit scale
  (~6..29), and D1 ltr_features performs per-query z-normalisation anyway.

Safety:
  1. verifies v14 workload candidate stats against PRIVATE_D1_FAST_V14_REPORT
     when the report contains those stats;
  2. infers LoRA rank/target_modules directly from checkpoint;
  3. auto-calibrates LoRA alpha and passage packaging against the supplied
     held-out Prism score cache BEFORE private inference;
  4. blocks if calibration rank correlation is too low;
  5. resumable atomic private score cache;
  6. never touches old D1 caches.

Default passage contract candidates:
  - top1:     score best lexical passage only
  - top2_max: score two historical top_passages independently; doc=max(score)
The calibration chooses whichever reproduces held-out Prism scores better.

Private output:
  results/manual/huy_private_prism_v1/prism_private_scores.pkl
"""

from __future__ import annotations

import argparse
import gc
import inspect
import json
import math
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch


BASE_MODEL_DEFAULT = "infgrad/Prism-Qwen3.5-Reranker-2B"

SYSTEM_PROMPT = (
    "Judge whether the Document meets the requirements based on "
    "the Query and the Instruct provided. "
)
INSTRUCTION = (
    'Judge if the document is relevant to the query. Reply "yes" or "no".\n'
    'On "yes", also emit:\n'
    "<contribution>One sentence covering every core point the document "
    "contributes to the query, without elaboration.</contribution>\n"
    "<evidence>Self-contained rewrite of the query-relevant content. Rules:\n"
    "- Faithful: rephrase only; add or infer nothing.\n"
    "- Self-contained: evidence alone must fully answer the query.\n"
    "- Concise: drop query-irrelevant background.\n"
    "- Verbatim (no translation): proper nouns, terms, abbreviations, "
    "numbers, dates, code, URLs.\n"
    "- Output language: multilingual doc → query's language; else doc's language."
    "</evidence>"
)
PROMPT_TEMPLATE = (
    "<|im_start|>system\n{system}<|im_end|>\n"
    "<|im_start|>user\n"
    "<Instruct>: {instruction}\n"
    "<Query>: {query}\n"
    "<Document>: {doc}<|im_end|>\n"
    "<|im_start|>assistant\n<think>\n\n</think>\n\n"
)


def atomic_pickle(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_bytes(pickle.dumps(obj, protocol=5))
    os.replace(tmp, path)


def load_pickle(path: Path):
    return pickle.loads(path.read_bytes())


def build_prompt(query: str, doc: str) -> str:
    return PROMPT_TEMPLATE.format(
        system=SYSTEM_PROMPT,
        instruction=INSTRUCTION,
        query=query,
        doc=doc,
    )


def ranks(x):
    # Stable average-free ranks are enough here because exact score ties are rare.
    order = np.argsort(np.asarray(x), kind="mergesort")
    r = np.empty(len(order), dtype=np.float64)
    r[order] = np.arange(len(order), dtype=np.float64)
    return r


def spearman(a, b):
    if len(a) < 2:
        return 1.0
    ra, rb = ranks(a), ranks(b)
    if np.std(ra) == 0 or np.std(rb) == 0:
        return 0.0
    return float(np.corrcoef(ra, rb)[0, 1])


def pearson(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def infer_lora_contract(state_dict):
    a_keys = [k for k in state_dict if ".lora_A." in k]
    b_keys = [k for k in state_dict if ".lora_B." in k]
    if not a_keys or len(a_keys) != len(b_keys):
        raise RuntimeError(
            f"Not a recognized LoRA checkpoint: A={len(a_keys)} B={len(b_keys)}"
        )
    ranks_seen = {
        int(state_dict[k].shape[0])
        for k in a_keys
    }
    if len(ranks_seen) != 1:
        raise RuntimeError(f"Mixed LoRA ranks not supported: {ranks_seen}")
    rank = next(iter(ranks_seen))
    targets = sorted({
        k.split(".lora_A.")[0].split(".")[-1]
        for k in a_keys
    })
    return rank, targets


def set_lora_alpha_runtime(model, alpha: float, rank: int):
    changed = 0
    for module in model.modules():
        scaling = getattr(module, "scaling", None)
        if isinstance(scaling, dict) and "default" in scaling:
            scaling["default"] = float(alpha) / float(rank)
            changed += 1
    if changed == 0:
        raise RuntimeError("Could not find active PEFT LoRA scaling modules")
    return changed


def load_model(base_model: str, checkpoint: Path, device: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, TaskType, get_peft_model

    obj = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = obj.get("state_dict", obj)
    rank, targets = infer_lora_contract(state)

    print(f"  LoRA rank={rank}")
    print(f"  target_modules={targets}")

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    use_cuda = str(device).startswith("cuda")
    if use_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")

    if use_cuda and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    elif use_cuda:
        dtype = torch.float16
    else:
        dtype = torch.float32

    kwargs = dict(
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )
    try:
        base = AutoModelForCausalLM.from_pretrained(
            base_model, dtype=dtype, **kwargs
        )
    except TypeError:
        base = AutoModelForCausalLM.from_pretrained(
            base_model, torch_dtype=dtype, **kwargs
        )

    # Alpha is temporarily rank (scaling=1). Calibration changes it later.
    cfg = LoraConfig(
        r=rank,
        lora_alpha=rank,
        target_modules=targets,
        lora_dropout=0.0,  # irrelevant in eval; adapter dropout is disabled
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(base, cfg)

    model_keys = set(model.state_dict())
    missing_ckpt = [k for k in state if k not in model_keys]
    if missing_ckpt:
        raise RuntimeError(
            "PEFT reconstruction does not match checkpoint. "
            f"Missing {len(missing_ckpt)} keys, sample={missing_ckpt[:5]}"
        )

    incompat = model.load_state_dict(state, strict=False)
    if incompat.unexpected_keys:
        raise RuntimeError(
            f"Unexpected checkpoint keys: {incompat.unexpected_keys[:10]}"
        )

    model.eval().to(device)
    print(
        f"  base={base_model} dtype={dtype} device={device} "
        f"adapter_tensors={len(state)}"
    )
    return model, tokenizer, rank, targets, dtype


@torch.inference_mode()
def raw_prism_scores(model, tokenizer, prompts, *, batch_size, max_length, device):
    """
    Compute only the final prompt-position hidden state and LM-head yes/no logits.
    Avoids generation and avoids materialising vocabulary logits for every token.
    """
    base = model.get_base_model()
    yes_ids = tokenizer.encode("yes", add_special_tokens=False)
    no_ids = tokenizer.encode("no", add_special_tokens=False)
    if len(yes_ids) != 1 or len(no_ids) != 1:
        raise RuntimeError(
            f"Expected single-token yes/no ids, got yes={yes_ids}, no={no_ids}"
        )
    yes_id, no_id = yes_ids[0], no_ids[0]

    result = []
    st = 0
    current_bs = max(1, int(batch_size))

    while st < len(prompts):
        bs = min(current_bs, len(prompts) - st)
        chunk = prompts[st:st + bs]
        try:
            enc = tokenizer(
                chunk,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
                add_special_tokens=False,
            )
            enc = {k: v.to(device) for k, v in enc.items()}

            # Qwen causal LM: `.model` returns final normalized hidden states.
            out = base.model(
                input_ids=enc["input_ids"],
                attention_mask=enc.get("attention_mask"),
                use_cache=False,
                return_dict=True,
            )
            last_hidden = out.last_hidden_state[:, -1, :]
            logits = base.lm_head(last_hidden).float()
            diff = logits[:, yes_id] - logits[:, no_id]
            result.extend(diff.detach().cpu().tolist())
            st += bs

        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if current_bs <= 1:
                raise
            current_bs = max(1, current_bs // 2)
            print(
                f"  CUDA OOM -> reducing Prism batch_size to {current_bs}",
                flush=True,
            )

    return result, current_bs


def load_documents(root: Path):
    from run_burst_expanded_fusion_submission import DocumentStore
    data = root / "DSC2026-LegalIR-main/v4_run/public_test_dataset"
    return DocumentStore(sorted((data / "selected-contexts").glob("context_*.json")))


def passage_texts(query, doc_text, mode, top_passages):
    if mode == "top1":
        return top_passages(query, doc_text, count=1)
    if mode == "top2_max":
        return top_passages(query, doc_text, count=2)
    raise ValueError(mode)


def score_doc_rows(model, tokenizer, query, docs, document_store, top_passages,
                   *, mode, batch_size, max_length, device):
    prompts, owners = [], []
    for doc in docs:
        parts = passage_texts(
            query, document_store[doc], mode, top_passages
        )
        for p in parts:
            prompts.append(build_prompt(query, p))
            owners.append(doc)

    vals, actual_bs = raw_prism_scores(
        model, tokenizer, prompts,
        batch_size=batch_size,
        max_length=max_length,
        device=device,
    )
    scores = {d: -1e30 for d in docs}
    for d, s in zip(owners, vals):
        scores[d] = max(scores[d], float(s))
    return scores, actual_bs


def choose_calibration_sample(raw_scores, queries, n_queries=10, docs_per_query=5):
    qids = sorted(set(raw_scores) & set(queries))
    if not qids:
        raise RuntimeError("No overlap between heldout Prism cache and CAL queries")
    if n_queries < len(qids):
        idx = np.linspace(0, len(qids) - 1, n_queries, dtype=int)
        qids = [qids[int(i)] for i in idx]

    sample = {}
    for q in qids:
        row = raw_scores[q]
        docs = sorted(row, key=lambda d: row[d])
        if not docs:
            continue
        if len(docs) <= docs_per_query:
            pick = docs
        else:
            pos = np.linspace(0, len(docs) - 1, docs_per_query, dtype=int)
            pick = [docs[int(i)] for i in pos]
        sample[q] = pick
    return sample


def calibrate(root, model, tokenizer, rank, heldout_scores,
              *, alpha_candidates, modes, batch_size, max_length, device,
              sample_queries):
    from benchmark_jina_reranker_holdouts import top_passages
    from src.gemini.huy_vnlegal_rank_ablation_v1.evaluate_ablation_cal import (
        load_cal_inputs,
    )

    (
        queries, _blocks, all_ids, _extended, _views, _channels, _gold,
        _vn, _type, _cite,
    ) = load_cal_inputs()
    qtext = {q: str(queries[q][0]) for q in all_ids}
    docs = load_documents(root)

    sample = choose_calibration_sample(
        heldout_scores, qtext, n_queries=sample_queries
    )
    print(
        f"  calibration sample: {len(sample)} queries, "
        f"{sum(len(x) for x in sample.values())} docs"
    )

    trials = []
    actual_bs = batch_size

    for alpha in alpha_candidates:
        changed = set_lora_alpha_runtime(model, alpha, rank)
        print(f"  alpha={alpha:g} active_lora_modules={changed}")

        for mode in modes:
            refs, preds = [], []
            q_spearman = []

            for q, qdocs in sample.items():
                scored, actual_bs = score_doc_rows(
                    model, tokenizer, qtext[q], qdocs, docs, top_passages,
                    mode=mode, batch_size=actual_bs,
                    max_length=max_length, device=device,
                )
                rv = [float(heldout_scores[q][d]) for d in qdocs]
                pv = [float(scored[d]) for d in qdocs]
                refs.extend(rv)
                preds.extend(pv)
                q_spearman.append(spearman(rv, pv))

            sp = spearman(refs, preds)
            pr = pearson(refs, preds)
            mae = float(np.mean(np.abs(np.asarray(refs) - np.asarray(preds))))
            rmse = float(np.sqrt(np.mean(
                (np.asarray(refs) - np.asarray(preds)) ** 2
            )))
            qsp = float(np.mean(q_spearman))

            row = {
                "alpha": float(alpha),
                "mode": mode,
                "spearman_global": sp,
                "pearson_global": pr,
                "mean_query_spearman": qsp,
                "mae_raw_logit": mae,
                "rmse_raw_logit": rmse,
            }
            trials.append(row)
            print(
                f"    {mode:9s} "
                f"rho={sp:.4f} q-rho={qsp:.4f} "
                f"pearson={pr:.4f} MAE={mae:.3f}"
            )

    # Ranking is primary because private D1 consumes rank + query-normalized score.
    # Raw MAE breaks close ties to recover the actual alpha/scaling contract.
    trials.sort(
        key=lambda x: (
            x["mean_query_spearman"],
            x["spearman_global"],
            -x["mae_raw_logit"],
        ),
        reverse=True,
    )
    best = trials[0]
    return best, trials, actual_bs


def audit_workload_vs_v14(root: Path, workload_meta: dict):
    report_path = (
        root
        / "results/manual/huy_private_d1_rel_l0_exact_v1/"
        "PRIVATE_D1_FAST_V14_REPORT.json"
    )
    result = {
        "report_present": report_path.is_file(),
        "status": "NO_REPORT",
    }
    if not report_path.is_file():
        return result

    report = json.loads(report_path.read_text(encoding="utf-8"))
    d1 = report.get("d1", {})
    observed = workload_meta.get("candidate_stats", {})
    result.update({
        "status": "CHECKED",
        "v14_min": d1.get("candidate_pool_min"),
        "v14_mean": d1.get("candidate_pool_mean"),
        "v14_max": d1.get("candidate_pool_max"),
        "workload_min": observed.get("min"),
        "workload_mean": observed.get("mean"),
        "workload_max": observed.get("max"),
    })

    for key in ("min", "mean", "max"):
        a = d1.get(f"candidate_pool_{key}")
        b = observed.get(key)
        if a is not None and b is not None:
            tol = 1e-9 if key != "mean" else 1e-6
            if abs(float(a) - float(b)) > tol:
                result["status"] = "BLOCKED_MISMATCH"
                raise RuntimeError(
                    f"Prism workload != actual v14 candidate pool for {key}: "
                    f"v14={a} workload={b}. Refusing expensive Prism inference."
                )
    result["status"] = "PASS"
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--heldout-scores", type=Path, required=True)
    ap.add_argument(
        "--workload",
        type=Path,
        default=None,
    )
    ap.add_argument(
        "--base-model",
        default=BASE_MODEL_DEFAULT,
    )
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument(
        "--alpha-candidates",
        default="16,32,64",
        help="Comma-separated LoRA alpha values to probe.",
    )
    ap.add_argument(
        "--modes",
        default="top1,top2_max",
        help="Comma-separated passage packaging modes.",
    )
    ap.add_argument("--calibration-queries", type=int, default=10)
    ap.add_argument("--min-calibration-rho", type=float, default=.90)
    ap.add_argument("--save-every", type=int, default=10)
    ap.add_argument("--calibrate-only", action="store_true")
    args = ap.parse_args()

    root = args.repo_root.resolve()
    sys.path.insert(0, str(root))

    checkpoint = args.checkpoint.resolve()
    heldout_path = args.heldout_scores.resolve()
    workload = (
        args.workload.resolve()
        if args.workload
        else (
            root
            / "results/manual/huy_private_prism_v1/"
            "PRISM_PRIVATE_WORKLOAD.jsonl"
        ).resolve()
    )
    workload_meta_path = workload.with_name("PRISM_PRIVATE_WORKLOAD_META.json")

    for p in (checkpoint, heldout_path, workload):
        if not p.is_file():
            raise FileNotFoundError(p)

    print("[1/5] Loading Prism-2B base + LoRA checkpoint...")
    model, tokenizer, rank, targets, dtype = load_model(
        args.base_model, checkpoint, args.device
    )

    print("[2/5] Auditing exact v14 workload contract...")
    meta = (
        json.loads(workload_meta_path.read_text(encoding="utf-8"))
        if workload_meta_path.is_file()
        else {}
    )
    workload_audit = audit_workload_vs_v14(root, meta)
    print("  workload audit:", workload_audit["status"])

    print("[3/5] Auto-calibrating LoRA alpha + passage packaging...")
    heldout_scores = load_pickle(heldout_path)
    if isinstance(heldout_scores, dict) and isinstance(
        heldout_scores.get("scores"), dict
    ):
        heldout_scores = heldout_scores["scores"]
    heldout_scores = {
        str(q): {str(d): float(s) for d, s in row.items()}
        for q, row in heldout_scores.items()
    }

    alpha_candidates = [
        float(x.strip())
        for x in args.alpha_candidates.split(",")
        if x.strip()
    ]
    modes = [x.strip() for x in args.modes.split(",") if x.strip()]

    best, trials, actual_bs = calibrate(
        root, model, tokenizer, rank, heldout_scores,
        alpha_candidates=alpha_candidates,
        modes=modes,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=args.device,
        sample_queries=args.calibration_queries,
    )

    print("  BEST CALIBRATION:", best)
    if best["mean_query_spearman"] < args.min_calibration_rho:
        raise RuntimeError(
            "Prism inference contract reproduction is too weak: "
            f"mean query rho={best['mean_query_spearman']:.4f} "
            f"< {args.min_calibration_rho:.4f}. "
            "Do not run private inference."
        )

    set_lora_alpha_runtime(model, best["alpha"], rank)

    out = root / "results/manual/huy_private_prism_v1"
    out.mkdir(parents=True, exist_ok=True)
    calib_path = out / "PRISM_INFERENCE_CALIBRATION.json"
    calib_path.write_text(
        json.dumps(
            {
                "base_model": args.base_model,
                "checkpoint": str(checkpoint),
                "lora_rank": rank,
                "target_modules": targets,
                "dtype": str(dtype),
                "max_length": args.max_length,
                "best": best,
                "trials": trials,
                "workload_audit": workload_audit,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )

    if args.calibrate_only:
        print("[4/5] --calibrate-only set; stopping before private inference.")
        print("Calibration:", calib_path)
        return

    print("[4/5] Loading private workload + resume cache...")
    rows = []
    with workload.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    if len(rows) != 2080:
        raise RuntimeError(f"Expected 2080 workload rows, got {len(rows)}")

    score_path = out / "prism_private_scores.pkl"
    saved = load_pickle(score_path) if score_path.is_file() else {}
    saved = {
        str(q): {str(d): float(s) for d, s in row.items()}
        for q, row in saved.items()
    }

    remaining = []
    for row in rows:
        q = str(row["qid"])
        docs = [str(d) for d in row["candidate_doc_ids"]]
        if q in saved and all(d in saved[q] for d in docs):
            continue
        remaining.append(row)

    print(
        f"  Prism cache complete={len(rows)-len(remaining)}/{len(rows)} "
        f"remaining={len(remaining)}"
    )

    from benchmark_jina_reranker_holdouts import top_passages
    documents = load_documents(root)

    started = time.perf_counter()
    done = 0
    for row in remaining:
        q = str(row["qid"])
        question = str(row["question"])
        docs = [str(d) for d in row["candidate_doc_ids"]]

        existing = dict(saved.get(q, {}))
        need = [d for d in docs if d not in existing]
        if need:
            new_scores, actual_bs = score_doc_rows(
                model, tokenizer, question, need, documents, top_passages,
                mode=best["mode"],
                batch_size=actual_bs,
                max_length=args.max_length,
                device=args.device,
            )
            existing.update(new_scores)

        if any(d not in existing for d in docs):
            raise RuntimeError(f"Incomplete Prism scoring q={q}")
        saved[q] = {d: float(existing[d]) for d in docs}
        done += 1

        if done % args.save_every == 0 or done == len(remaining):
            atomic_pickle(score_path, saved)
            elapsed = time.perf_counter() - started
            rate = elapsed / max(done, 1)
            left = len(remaining) - done
            print(
                f"  Prism {done}/{len(remaining)} new queries | "
                f"total={len(saved)}/2080 | "
                f"{rate:.2f}s/q | eta={rate*left/60:.1f}m | "
                f"batch={actual_bs}",
                flush=True,
            )

    print("[5/5] Final coverage + report...")
    missing = []
    total_pairs = 0
    for row in rows:
        q = str(row["qid"])
        docs = [str(d) for d in row["candidate_doc_ids"]]
        total_pairs += len(docs)
        if q not in saved:
            missing.append((q, None))
            continue
        for d in docs:
            if d not in saved[q]:
                missing.append((q, d))
                if len(missing) >= 20:
                    break

    if missing:
        raise RuntimeError(f"Final Prism cache incomplete: {missing[:20]}")

    report = {
        "schema": "manual.private_prism_scoring_v1",
        "status": "COMPLETE",
        "queries": len(rows),
        "pairs": total_pairs,
        "base_model": args.base_model,
        "checkpoint": str(checkpoint),
        "lora_rank": rank,
        "target_modules": targets,
        "selected_alpha": best["alpha"],
        "selected_mode": best["mode"],
        "max_length": args.max_length,
        "final_batch_size": actual_bs,
        "calibration": best,
        "score_cache": str(score_path),
        "score_min": float(min(
            s for row in saved.values() for s in row.values()
        )),
        "score_max": float(max(
            s for row in saved.values() for s in row.values()
        )),
    }
    report_path = out / "PRISM_PRIVATE_SCORING_REPORT.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("=" * 105)
    print("PRISM PRIVATE SCORES COMPLETE")
    print("scores:", score_path)
    print("queries:", len(rows), "pairs:", total_pairs)
    print("alpha:", best["alpha"], "mode:", best["mode"])
    print("calibration q-rho:", best["mean_query_spearman"])
    print("report:", report_path)
    print("=" * 105)


if __name__ == "__main__":
    main()
