"""Benchmark training throughput and CAL inference batch size to project total runtime."""

from __future__ import annotations

import gc
import json
import os
import sys
import time
from pathlib import Path
import torch

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

ROOT = Path("D:/Study/DSC2026/sota")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src/huy_fasttrack") not in sys.path:
    sys.path.insert(0, str(ROOT / "src/huy_fasttrack"))

import run_huy_5fold_fasttrack as core
from run_burst_expanded_fusion_submission import DocumentStore
from tune_corpus_cap32_fusion import build_training_cap
import src.gemini.huy_d1_jina_ft_continuation_v1.common as common
from src.gemini.huy_d1_jina_ft_continuation_v1.audit_train_data import sample_query_negatives

OUT_PATH = ROOT / "results/gemini/huy_d1_jina_ft_continuation_v1/RUNTIME_PROJECTION.json"


def benchmark_training(train_mb: int = 8, n_benchmark: int = 64) -> dict:
    common.seed_everything(2026)
    print(f"Benchmarking training on {n_benchmark} query groups with pair_mb={train_mb}...", flush=True)

    # Load data
    folds, pools, questions, v2_golds, _, _, _, _ = core.load_inputs()
    with open(ROOT / "results/gemini/huy_d1_jina_ft_continuation_v1/JINA_TRAIN_DATA_AUDIT.json", "r") as f:
        train_audit = json.load(f)
    
    contexts = common.load_contexts()
    h_local = common.load_h_local()
    v2_doc_ids = set(contexts.keys())

    # Build model
    model, tok = common.build_lora_jina_model(r=16, lora_alpha=32, lora_dropout=0.05)
    model.to("cuda")
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=5e-5,
        weight_decay=0.01,
    )

    # Select first n_benchmark queries
    eligible_qids = [str(q) for q in sorted(list(pools.keys()), key=int)]
    from src.gemini.huy_d1_lal_case_memory_v1.audit_data_isolation import load_duplicate_graph
    _, _, all_ids, _, _, _ = build_training_cap(ROOT, 32, "results/corpus_index/holdout_extended_scores_cap32.pkl", depth=20)
    cal_qids = set(str(q) for q in all_ids)
    _, _, _, dup_map = load_duplicate_graph()
    cal_dups = {str(dup) for q in cal_qids for dup in dup_map.get(q, set())}
    forbidden = cal_qids | cal_dups
    sample_qids = [q for q in eligible_qids if q not in forbidden][:n_benchmark]

    # Warmup
    optimizer.zero_grad(set_to_none=True)
    total_pairs = 0
    t0 = time.perf_counter()

    for idx, q in enumerate(sample_qids):
        pool = [str(d) for d in pools[q]]
        gold = {str(d) for d in v2_golds.get(q, set())}
        positives = [d for d in pool if d in gold]
        if not positives:
            continue
        negs = sample_query_negatives(q, pool, gold, h_local, v2_doc_ids, seed=2026)

        q_text = questions[q]
        all_docs = positives + negs
        # 1 passage per doc for training
        passages = [common.top_passages(q_text, contexts[d], count=1)[0] for d in all_docs]
        pairs = [(q_text, p) for p in passages]
        total_pairs += len(pairs)

        # Microbatch forward/backward
        logits = []
        for i in range(0, len(pairs), train_mb):
            chunk = pairs[i : i + train_mb]
            batch = tok(
                [c[0] for c in chunk],
                [c[1] for c in chunk],
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True,
            ).to("cuda")
            with torch.amp.autocast("cuda", dtype=torch.float16):
                out = model(**batch)
                logits.append(out.logits.view(-1))

        all_logits = torch.cat(logits)
        pos_l = all_logits[: len(positives)]
        neg_l = all_logits[len(positives) :]
        loss = torch.nn.functional.softplus(-(pos_l.unsqueeze(1) - neg_l.unsqueeze(0))).mean()
        (loss / 16.0).backward()

        if (idx + 1) % 16 == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

    torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    groups_per_sec = len(sample_qids) / dt
    pairs_per_sec = total_pairs / dt
    sec_per_opt_step = dt / (len(sample_qids) / 16.0)

    total_training_queries = train_audit["training_population_count"]
    projected_train_sec = total_training_queries / groups_per_sec

    print(
        f"Training benchmark: {groups_per_sec:.2f} groups/s, {pairs_per_sec:.2f} pairs/s, "
        f"{sec_per_opt_step:.2f} s/step. Projected {total_training_queries} queries: {projected_train_sec/60:.1f} min",
        flush=True,
    )

    del model, optimizer
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "benchmark_queries": len(sample_qids),
        "benchmark_pairs": total_pairs,
        "elapsed_sec": dt,
        "groups_per_sec": groups_per_sec,
        "pairs_per_sec": pairs_per_sec,
        "seconds_per_optimizer_step": sec_per_opt_step,
        "projected_train_seconds": projected_train_sec,
        "projected_train_minutes": projected_train_sec / 60.0,
    }


def benchmark_inference(ladder: list = [4, 8, 16, 32, 64], n_cal: int = 32) -> dict:
    common.seed_everything(2026)
    print(f"Benchmarking inference batch size on {n_cal} CAL queries across {ladder}...", flush=True)

    docs = DocumentStore(
        sorted(
            (
                ROOT
                / "DSC2026-LegalIR-main/v4_run/public_test_dataset/selected-contexts"
            ).glob("context_*.json")
        )
    )
    queries, blocks, all_ids, extended, local, _ = build_training_cap(
        ROOT, 32, "results/corpus_index/holdout_extended_scores_cap32.pkl", depth=20
    )

    model, tok = common.load_jina_base_with_shipped_weights()
    model._tokenizer = tok
    model.eval().to("cuda")

    cal_sample = [str(q) for q in all_ids[:n_cal]]
    all_pairs = []
    for q in cal_sample:
        text = queries[q][0]
        for d in extended[q]:
            passages = common.top_passages(text, docs[d], count=2)
            for p in passages:
                all_pairs.append((text, p))

    ladder_results = {}
    chosen_infer_bs = None

    for bs in ladder:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        t0 = time.perf_counter()
        try:
            raw = model.compute_score(all_pairs, batch_size=bs, max_length=512)
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0

            peak_res = torch.cuda.max_memory_reserved() / (1024**2)
            free_mem, _ = torch.cuda.mem_get_info()
            free_mib = free_mem / (1024**2)

            is_safe = (peak_res <= 5000.0) and (free_mib >= 600.0) and (peak_res <= 5300.0)
            pairs_per_sec = len(all_pairs) / dt
            ladder_results[bs] = {
                "batch_size": bs,
                "peak_reserved_mib": peak_res,
                "free_mib": free_mib,
                "pairs_per_sec": pairs_per_sec,
                "time_sec": dt,
                "safe": is_safe,
            }
            print(
                f"  infer_bs={bs}: peak_res={peak_res:.1f}MiB, free={free_mib:.1f}MiB, "
                f"pairs/s={pairs_per_sec:.1f}, safe={is_safe}",
                flush=True,
            )
            if is_safe:
                chosen_infer_bs = bs
        except Exception as e:
            print(f"  infer_bs={bs} FAILED: {e}", flush=True)
            ladder_results[bs] = {"batch_size": bs, "safe": False, "error": str(e)}

    # Total CAL pairs estimate
    total_cal_pairs = sum(len(extended[str(q)]) * 2 for q in all_ids)
    pairs_per_sec = ladder_results[chosen_infer_bs]["pairs_per_sec"]
    projected_cal_sec = total_cal_pairs / pairs_per_sec

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "chosen_inference_batch_size": chosen_infer_bs,
        "total_cal600_pairs_estimate": total_cal_pairs,
        "pairs_per_sec": pairs_per_sec,
        "projected_cal_inference_seconds": projected_cal_sec,
        "projected_cal_inference_minutes": projected_cal_sec / 60.0,
        "ladder_results": ladder_results,
    }


def run_benchmark() -> dict:
    with open(ROOT / "results/gemini/huy_d1_jina_ft_continuation_v1/VRAM_CALIBRATION.json", "r") as f:
        calib = json.load(f)
    train_mb = calib.get("chosen_pair_microbatch", 8)

    train_stats = benchmark_training(train_mb=train_mb, n_benchmark=64)
    infer_stats = benchmark_inference(ladder=[4, 8, 16, 32, 64], n_cal=32)

    fusion_sec = 3.0  # CPU LTR LOBO takes ~2-3 seconds
    total_projected_sec = (
        train_stats["projected_train_seconds"]
        + infer_stats["projected_cal_inference_seconds"]
        + fusion_sec
    )
    total_hours = total_projected_sec / 3600.0

    recommendation = (
        "CONTINUE_LOCAL"
        if total_hours <= 3.0
        else "LOCAL_TOO_SLOW_KAGGLE_RECOMMENDED"
    )

    report = {
        "experiment_id": "HUY_D1_JINA_FT_CONTINUATION_V1",
        "chosen_train_pair_microbatch": train_mb,
        "chosen_inference_batch_size": infer_stats["chosen_inference_batch_size"],
        "training_projection": train_stats,
        "cal_inference_projection": infer_stats,
        "fusion_lobo_seconds_estimate": fusion_sec,
        "total_projected_seconds": total_projected_sec,
        "total_projected_minutes": total_projected_sec / 60.0,
        "total_projected_hours": total_hours,
        "time_limit_hours": 3.0,
        "verdict": recommendation,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(
        f"\nWrote {OUT_PATH}: Total projected runtime = {total_hours:.2f} hours. Verdict: {recommendation}",
        flush=True,
    )
    return report


if __name__ == "__main__":
    run_benchmark()
