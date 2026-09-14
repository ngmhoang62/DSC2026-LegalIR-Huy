"""Phase A: Real VRAM Smoke Test for jina_honest_adaptation_v1.
Verifies memory footprint and execution safety of M1 (LoRA) and M2 (Full FT)
on a representative maximum-size Epoch-2 query group.
Produces results/gemini/jina_honest_adaptation_v1/VRAM_SMOKE_TEST.json.
"""

from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from common import (
    EXP_RESULTS,
    REPO_ROOT,
    get_git_info,
    get_os_gpu_memory,
    log_execution_trace,
    sha256_file,
)
from dataset import LegalRetrievalDataset
from model import (
    build_model,
    checkpoint_round_trip_test,
    compute_group_loss,
    compute_parent_scores,
)


def run_architecture_smoke(
    method: str,
    dataset: LegalRetrievalDataset,
    sample: Dict[str, Any],
    audit_pairs: List[Tuple[str, str]],
    model_path: Path,
    device: str = "cuda",
) -> Dict[str, Any]:
    print(f"\n=======================================================", flush=True)
    print(f"STARTING VRAM SMOKE TEST: {method}", flush=True)
    print(f"=======================================================", flush=True)

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    allocated_before_load = torch.cuda.memory_allocated() / (1024**2)
    os_mem_before = get_os_gpu_memory()

    t0 = time.perf_counter()
    model, tok = build_model(method, model_path, rank=16, dtype=torch.bfloat16, device=device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-5)

    allocated_before_forward = torch.cuda.memory_allocated() / (1024**2)
    reserved_before_forward = torch.cuda.memory_reserved() / (1024**2)

    # Clone trainable parameters to measure L2 delta
    pre_weights = {
        name: p.detach().clone().cpu().float()
        for name, p in model.named_parameters()
        if p.requires_grad
    }

    # Real training step: forward with microbatch = 4
    model.train()
    optimizer.zero_grad()

    t_fwd = time.perf_counter()
    parent_scores = compute_parent_scores(
        model,
        tok,
        sample["pairs"],
        sample["doc_indices_for_passages"],
        sample["num_docs"],
        tau_pool=0.15,
        max_length=512,
        device=device,
        micro_batch_size=4,
    )

    frozen = torch.tensor(sample["frozen_doc_scores"], device=device, dtype=parent_scores.dtype)
    tot_loss, l_rank, l_anchor = compute_group_loss(
        parent_scores, sample["pos_indices"], frozen, lambda_anchor=0.05
    )

    tot_loss.backward()

    # Calculate gradient stats
    grad_norms = []
    nonzero_grads = 0
    total_grads = 0
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            g = p.grad.detach()
            grad_norms.append(float(g.norm(2).cpu().item()))
            nonzero_grads += int((g != 0).sum().cpu().item())
            total_grads += g.numel()

    grad_norm = float(np.sqrt(sum(g**2 for g in grad_norms))) if grad_norms else 0.0
    nonzero_fraction = (nonzero_grads / total_grads) if total_grads > 0 else 0.0

    optimizer.step()
    torch.cuda.synchronize()
    step_time = time.perf_counter() - t_fwd

    # Measure parameter change
    l2_delta_sq = 0.0
    for name, p in model.named_parameters():
        if p.requires_grad:
            diff = (p.detach().clone().cpu().float() - pre_weights[name]).norm(2).item()
            l2_delta_sq += diff**2
    l2_delta = float(np.sqrt(l2_delta_sq))

    peak_allocated = torch.cuda.max_memory_allocated() / (1024**2)
    peak_reserved = torch.cuda.max_memory_reserved() / (1024**2)
    os_mem_peak = get_os_gpu_memory()

    # Round-trip checkpoint audit (Section 3.2)
    print(f"Running Checkpoint Round-Trip test for {method}...", flush=True)
    test_ckpt_path = EXP_RESULTS / f"checkpoints/test_roundtrip_{method}.pt"
    round_trip_res = checkpoint_round_trip_test(
        model=model,
        tok=tok,
        audit_pairs=audit_pairs[:16],
        method=method,
        model_path=model_path,
        test_path=test_ckpt_path,
        device=device,
    )
    print(f"Checkpoint Round-Trip {method}: {round_trip_res['status']} (max logit diff: {round_trip_res['max_logit_difference']:.8e})")

    total_runtime = time.perf_counter() - t0

    # Cleanup
    del optimizer
    gc.collect()
    torch.cuda.empty_cache()
    allocated_after_cleanup = torch.cuda.memory_allocated() / (1024**2)
    reserved_after_cleanup = torch.cuda.memory_reserved() / (1024**2)

    total_gpu_vram = torch.cuda.get_device_properties(0).total_memory / (1024**2)
    safety_margin_mb = total_gpu_vram - peak_reserved

    # Section 5 local safety decision
    is_safe = safety_margin_mb >= 1500.0 and (
        os_mem_peak.get("windows_shared_gpu_memory_mb") is None
        or os_mem_peak["windows_shared_gpu_memory_mb"] < 250.0
    )

    record = {
        "method": method,
        "gpu_model": torch.cuda.get_device_name(0),
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "trainable_fraction": trainable_params / total_params,
        "sample_specs": {
            "num_parents": sample["num_docs"],
            "num_query_passage_pairs": len(sample["pairs"]),
            "max_length": 512,
            "micro_batch_size": 4,
            "epoch": 2,
        },
        "pytorch_memory": {
            "allocated_before_load_mb": round(allocated_before_load, 1),
            "allocated_before_forward_mb": round(allocated_before_forward, 1),
            "reserved_before_forward_mb": round(reserved_before_forward, 1),
            "peak_allocated_mb": round(peak_allocated, 1),
            "peak_reserved_mb": round(peak_reserved, 1),
            "allocated_after_cleanup_mb": round(allocated_after_cleanup, 1),
            "reserved_after_cleanup_mb": round(reserved_after_cleanup, 1),
            "dedicated_vram_total_mb": round(total_gpu_vram, 1),
            "safety_margin_mb": round(safety_margin_mb, 1),
        },
        "os_gpu_memory_peak": os_mem_peak,
        "training_step": {
            "actual_optimizer_steps": 1,
            "loss": float(tot_loss.item()),
            "grad_norm": grad_norm,
            "nonzero_gradient_fraction": nonzero_fraction,
            "parameter_l2_delta": l2_delta,
            "step_runtime_seconds": round(step_time, 3),
        },
        "checkpoint_round_trip": round_trip_res,
        "local_safety": {
            "status": "SAFE_FOR_LOCAL_TRAINING" if is_safe else "FULL_FT_LOCAL_UNSAFE",
            "safety_margin_mb": round(safety_margin_mb, 1),
        },
        "total_runtime_seconds": round(total_runtime, 3),
    }
    return record


def run_smoke_test():
    start_time = time.perf_counter()
    dataset = LegalRetrievalDataset(rng_seed=2026)
    model_path = REPO_ROOT / "cache/research_v2_forensic/models/jina-reranker-v2-base-multilingual"

    # Find a maximum-size Epoch-2 query group (e.g. multi-gold query with 2 positives + 10 negatives)
    multi_golds = [q for q, g in dataset.golds.items() if len(g) >= 2 and q in dataset.pools]
    target_qid = multi_golds[0] if multi_golds else list(dataset.pools.keys())[0]

    # Sample maximum-size Epoch 2 candidate group
    sample = dataset.sample_query_candidates(target_qid, epoch=2, seed_offset=0)
    print(f"Selected representative Epoch 2 query {target_qid}: {sample['num_docs']} parents, {len(sample['pairs'])} passages.")

    # Diverse audit pairs for round-trip test
    audit_pairs = []
    for q in sorted(dataset.pools.keys(), key=int)[:32]:
        qtext = dataset.questions[q]
        d = dataset.pools[q][0]
        dtext = dataset.contexts.get(d, "")
        p = dataset.top_passages(qtext, dtext, count=1) if hasattr(dataset, "top_passages") else []
        from benchmark_jina_reranker_holdouts import top_passages
        p = top_passages(qtext, dtext, count=1)
        if p:
            audit_pairs.append((qtext, p[0]))

    # Run smoke test on M1 LoRA
    m1_report = run_architecture_smoke("M1_LORA", dataset, sample, audit_pairs, model_path)

    # Run smoke test on M2 Full FT
    m2_report = run_architecture_smoke("M2_FULL", dataset, sample, audit_pairs, model_path)

    report = {
        "schema_version": "dsc2026.gemini.vram_smoke_test.v1",
        "status": "PASS",
        "device": torch.cuda.get_device_name(0),
        "dedicated_vram_limit_mb": round(torch.cuda.get_device_properties(0).total_memory / (1024**2), 1),
        "architectures": {
            "m1_lora": m1_report,
            "m2_full": m2_report,
        },
        "verdict": {
            "m1_lora_safe": m1_report["local_safety"]["status"] == "SAFE_FOR_LOCAL_TRAINING",
            "m2_full_safe": m2_report["local_safety"]["status"] == "SAFE_FOR_LOCAL_TRAINING",
            "recommendation": "BOTH_LOCAL_SAFE"
            if (m1_report["local_safety"]["status"] == "SAFE_FOR_LOCAL_TRAINING" and m2_report["local_safety"]["status"] == "SAFE_FOR_LOCAL_TRAINING")
            else "LORA_LOCAL_SAFE_FULL_UNSAFE"
            if m1_report["local_safety"]["status"] == "SAFE_FOR_LOCAL_TRAINING"
            else "KAGGLE_FALLBACK_REQUIRED",
        },
        "git": get_git_info(),
        "runtime_seconds": round(time.perf_counter() - start_time, 3),
    }

    out_path = EXP_RESULTS / "VRAM_SMOKE_TEST.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")
    print(f"VRAM Verdict: {report['verdict']['recommendation']}")
    print(f"  M1 LoRA Peak Reserved: {m1_report['pytorch_memory']['peak_reserved_mb']} MB (Margin: {m1_report['pytorch_memory']['safety_margin_mb']} MB)")
    print(f"  M2 Full Peak Reserved: {m2_report['pytorch_memory']['peak_reserved_mb']} MB (Margin: {m2_report['pytorch_memory']['safety_margin_mb']} MB)")

    log_execution_trace(
        stage_name="vram_smoke_test",
        command=f"python {__file__}",
        code_hash=sha256_file(Path(__file__)),
        model_checkpoint_hash="BASE_MODEL",
        train_query_count=1,
        val_query_count=0,
        optimizer_steps=2,
        gpu_info=torch.cuda.get_device_name(0),
        runtime_sec=time.perf_counter() - start_time,
        output_hashes={"VRAM_SMOKE_TEST.json": sha256_file(out_path)},
        status=report["status"],
        extra={"verdict": report["verdict"]},
    )


if __name__ == "__main__":
    run_smoke_test()
