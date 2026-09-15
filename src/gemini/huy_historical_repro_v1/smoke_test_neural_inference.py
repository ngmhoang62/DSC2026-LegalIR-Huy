"""Smoke test neural inference on 5 CV queries for aiteamvn_ft, jina_ft, and title_embed."""

from __future__ import annotations

import gc
import json
import pickle
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[3]
SMOKE_OUTPUT = REPO_ROOT / "results/gemini/huy_historical_repro_v1/INFERENCE_SMOKE_TEST.json"

# Import helper functions from historical repo
sys_path = str(REPO_ROOT)
import sys
if sys_path not in sys.path:
    sys.path.insert(0, sys_path)

from benchmark_aiteamvn_holdouts import encode_cls
from benchmark_jina_reranker_holdouts import top_passages
from run_burst_expanded_fusion_submission import DocumentStore
from tune_corpus_cap32_fusion import build_training_cap


def patch_transformers_v5() -> None:
    import transformers.models.xlm_roberta.modeling_xlm_roberta as module
    if hasattr(module, "create_position_ids_from_input_ids"):
        return
    def helper(input_ids, padding_idx, past_key_values_length=0):
        mask = input_ids.ne(padding_idx).int()
        positions = (torch.cumsum(mask, dim=1) + past_key_values_length) * mask
        return positions.long() + padding_idx
    module.create_position_ids_from_input_ids = helper


def smoke_test():
    patch_transformers_v5()
    report = {}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Smoke test on {device} ({torch.cuda.get_device_name(0)})")
    initial_vram = torch.cuda.memory_allocated() / 1024**2
    print(f"Initial PyTorch VRAM allocated: {initial_vram:.1f} MB")

    docs = DocumentStore(sorted(
        (REPO_ROOT / "DSC2026-LegalIR-main/v4_run/public_test_dataset/selected-contexts")
        .glob("context_*.json")))
    queries, blocks, all_ids, extended, local, _ = build_training_cap(
        REPO_ROOT, 32, "results/corpus_index/holdout_extended_scores_cap32.pkl", depth=20)

    test_qids = all_ids[:5]
    print(f"Testing 5 queries: {test_qids}")

    # 1. Smoke test: title_embed
    print("\n--- Testing title_embed ---")
    t0 = time.perf_counter()
    model_path = REPO_ROOT / "models/AITeamVN_Vietnamese_Embedding"
    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path, dtype=torch.float16).eval().to(device)
    vram_title = torch.cuda.max_memory_allocated() / 1024**2

    q_vecs = encode_cls(model, tok, [queries[q][0] for q in test_qids], 5, 128)
    del model, tok
    torch.cuda.empty_cache()
    gc.collect()
    t_title = time.perf_counter() - t0
    report["title_embed"] = {
        "status": "PASS",
        "time_sec": t_title,
        "peak_vram_mb": vram_title,
        "queries_tested": len(test_qids)
    }
    print(f"title_embed smoke pass: {t_title:.2f}s, peak VRAM: {vram_title:.1f} MB")

    # 2. Smoke test: aiteamvn_ft (bi-encoder)
    print("\n--- Testing aiteamvn_ft bi-encoder ---")
    t0 = time.perf_counter()
    ft_bi_path = REPO_ROOT / "models/from_drive/AITeamVN_Vietnamese_Embedding"
    tok = AutoTokenizer.from_pretrained(ft_bi_path)
    model = AutoModel.from_pretrained(ft_bi_path, dtype=torch.float16, low_cpu_mem_usage=True).eval().to(device)
    vram_bi = torch.cuda.max_memory_allocated() / 1024**2

    hist_aiteam = pickle.loads((REPO_ROOT / "results/from_drive/aiteamvn_ft_cv.pkl").read_bytes())
    max_diff_bi = 0.0
    for q in test_qids:
        text = queries[q][0]
        qvec = encode_cls(model, tok, [text], 1, 512)[0]
        owners, passages = [], []
        for d in extended[q]:
            for p in top_passages(text, docs[d], count=2):
                owners.append(d)
                passages.append(p)
        ds = {}
        if passages:
            pvec = encode_cls(model, tok, passages, 16, 512)
            for d, s in zip(owners, pvec @ qvec):
                ds[d] = max(ds.get(d, -1e9), float(s))
        for d in ds:
            if d in hist_aiteam[q]:
                diff = abs(ds[d] - hist_aiteam[q][d])
                max_diff_bi = max(max_diff_bi, diff)

    del model, tok
    torch.cuda.empty_cache()
    gc.collect()
    t_bi = time.perf_counter() - t0
    report["aiteamvn_ft"] = {
        "status": "PASS",
        "time_sec": t_bi,
        "peak_vram_mb": vram_bi,
        "max_abs_diff_vs_historical": max_diff_bi,
        "queries_tested": len(test_qids)
    }
    print(f"aiteamvn_ft smoke pass: {t_bi:.2f}s, peak VRAM: {vram_bi:.1f} MB, max diff vs hist: {max_diff_bi:.2e}")

    # 3. Smoke test: jina_ft (cross-encoder)
    print("\n--- Testing jina_ft cross-encoder ---")
    t0 = time.perf_counter()
    jina_base_path = REPO_ROOT / "models/jina-reranker-v2-base-multilingual"
    jina_weights = REPO_ROOT / "models/from_drive/jina_finetuned/model.safetensors"
    tok = AutoTokenizer.from_pretrained(jina_base_path, trust_remote_code=True, fix_mistral_regex=True)
    model = AutoModelForSequenceClassification.from_pretrained(jina_base_path, trust_remote_code=True, dtype=torch.float16)
    state = load_file(jina_weights)
    missing, unexpected = model.load_state_dict({k: v.to(torch.float16) for k, v in state.items()}, strict=False)
    assert not unexpected, f"Unexpected keys: {unexpected[:5]}"
    model._tokenizer = tok
    model.eval().to(device)
    vram_jina = torch.cuda.max_memory_allocated() / 1024**2

    hist_jina = pickle.loads((REPO_ROOT / "results/from_drive/jina_ft_cv.pkl").read_bytes())
    max_diff_jina = 0.0
    for q in test_qids:
        text = queries[q][0]
        owners, passages = [], []
        for d in extended[q]:
            for p in top_passages(text, docs[d], count=2):
                owners.append(d)
                passages.append(p)
        ds = {}
        if passages:
            raw = model.compute_score([(text, p) for p in passages], batch_size=16, max_length=512)
            if isinstance(raw, float):
                raw = [raw]
            for d, s in zip(owners, raw):
                ds[d] = max(ds.get(d, -1e9), float(s))
        for d in ds:
            if d in hist_jina[q]:
                diff = abs(ds[d] - hist_jina[q][d])
                max_diff_jina = max(max_diff_jina, diff)

    del model, tok
    torch.cuda.empty_cache()
    gc.collect()
    t_jina = time.perf_counter() - t0
    report["jina_ft"] = {
        "status": "PASS",
        "time_sec": t_jina,
        "peak_vram_mb": vram_jina,
        "max_abs_diff_vs_historical": max_diff_jina,
        "queries_tested": len(test_qids)
    }
    print(f"jina_ft smoke pass: {t_jina:.2f}s, peak VRAM: {vram_jina:.1f} MB, max diff vs hist: {max_diff_jina:.2e}")

    SMOKE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    SMOKE_OUTPUT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved smoke test report to {SMOKE_OUTPUT}")


if __name__ == "__main__":
    smoke_test()
