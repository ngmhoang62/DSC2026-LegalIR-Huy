"""Score exact Huy Public candidate pool with frozen and full-data task-adapted VietLegal-E5.

Deployment Inference Protocol:
- Full-data adapter (epoch-2.pt from results/research_v2_open_rl/v2_anchor_submission_candidate/full_data_adapter/epoch-2.pt)
  trained on all 6,991 evaluable training queries.
- Frozen query encoder with adapter disabled for counterfactual delta reference.
- Scored against exact Huy candidate pool.
- Documents absent from the 8,507 V2 parent bank have score=np.nan (neutral missing).
- Deterministic tie-breaking: score DESC, then doc_id ASC.

Produces results/gemini/huy_e5_adaptation_delta_v1/PUBLIC_VIETLEGAL_E5_SCORES.pkl.
"""

from __future__ import annotations

import json
import math
import os
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = ROOT / "results/gemini/huy_e5_adaptation_delta_v1"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

BUNDLE_DIR = ROOT / "cache/research_v2_e5_confirmation/bundle-v1"
MODEL_DIR = BUNDLE_DIR / "vietlegal-e5"
CHUNKS_PATH = BUNDLE_DIR / "embeddings.f16.npy"
CHUNK_IDS_PATH = BUNDLE_DIR / "chunk_ids.jsonl"
FULL_DATA_ADAPTER_PATH = ROOT / "results/research_v2_open_rl/v2_anchor_submission_candidate/full_data_adapter/epoch-2.pt"

DATA_DIR = ROOT / "DSC2026-LegalIR-main/v4_run/public_test_dataset"
PUBLIC_JSON = DATA_DIR / "public-official.json"

MAX_LENGTH = 512
LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05


class E5QueryEncoder(torch.nn.Module):
    def __init__(self, model_path: Path, device: str = "cuda"):
        super().__init__()
        self.device = device
        print(f"Loading tokenizer & base model from {model_path}...", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
        base = AutoModel.from_pretrained(
            str(model_path),
            local_files_only=True,
            torch_dtype=torch.float32,
        )
        self.model = get_peft_model(
            base,
            LoraConfig(
                r=LORA_RANK,
                lora_alpha=LORA_ALPHA,
                lora_dropout=LORA_DROPOUT,
                target_modules=["query", "value"],
                bias="none",
            ),
        )
        self.model.config.use_cache = False
        self.to(device)
        self.eval()

    @torch.no_grad()
    def encode_queries(self, texts: List[str], batch_size: int = 16) -> torch.Tensor:
        all_vecs = []
        for i in range(0, len(texts), batch_size):
            batch_texts = ["query: " + t for t in texts[i : i + batch_size]]
            batch = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            ).to(self.device)
            hidden = self.model(**batch).last_hidden_state.float()
            mask = batch["attention_mask"].unsqueeze(-1)
            vecs = F.normalize((hidden * mask).sum(1) / mask.sum(1).clamp_min(1), dim=-1)
            all_vecs.append(vecs)
        return torch.cat(all_vecs, dim=0)

    def load_adapter_weights(self, checkpoint_path: Path):
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(ckpt["adapter"], strict=False)


def load_chunk_bank(device: str = "cuda"):
    print("Loading chunk metadata...", flush=True)
    chunk_parents = []
    with CHUNK_IDS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            chunk_parents.append(json.loads(line)["doc_id"])

    doc_ids = sorted(set(chunk_parents))
    doc_row = {d: i for i, d in enumerate(doc_ids)}
    parent_indices = np.array([doc_row[d] for d in chunk_parents], dtype=np.int64)

    positions = [[] for _ in doc_ids]
    for idx, p in enumerate(parent_indices):
        positions[p].append(idx)
    positions_tensor = [torch.tensor(p, dtype=torch.long, device=device) for p in positions]

    print("Loading chunk embeddings into GPU memory...", flush=True)
    vectors_np = np.load(CHUNKS_PATH)
    vectors = torch.from_numpy(vectors_np).to(device).float()
    vectors = F.normalize(vectors, dim=-1)
    print(f"Chunk bank loaded: {vectors.shape[0]} chunks, {len(doc_ids)} parent docs.", flush=True)

    return doc_row, positions_tensor, vectors


def sort_candidates(doc_scores: Dict[str, float]) -> List[str]:
    """Deterministic tie-break: score DESC, doc_id ASC. NaN scores at end."""
    return sorted(
        doc_scores.keys(),
        key=lambda d: (
            1 if (math.isnan(doc_scores[d]) or np.isnan(doc_scores[d])) else 0,
            -doc_scores[d] if not (math.isnan(doc_scores[d]) or np.isnan(doc_scores[d])) else 0.0,
            d,
        ),
    )


def load_exact_huy_public_candidates() -> Tuple[Dict[str, str], Dict[str, List[str]]]:
    """Reconstruct exact candidate pools for all 1,000 public queries."""
    import sys
    sys.path.insert(0, str(ROOT))
    from benchmark_dense_expansion_holdouts import raw_union
    from run_burst_expanded_fusion_submission import (
        CORPUS_CAP,
        CORPUS_DEPTH,
        DocumentStore,
        EXPANSION_CONFIG,
        RERANK_CONFIG,
        corpus_dense,
        dense_expansion,
        load_public_retrieval,
    )
    from run_burst_multistage_submission import load_metadata
    from tune_burst_multistage_posterior import weighted_rrf

    class DummyArgs:
        data_dir = DATA_DIR
        db = ROOT / "benchmarks/legalir_full_fts.sqlite"
        cache_dir = ROOT / "results/burst_expanded_fusion"

    paths, doc_ids, train, public = load_metadata(DATA_DIR)
    public_ids = list(public)
    retrieval = load_public_retrieval(ROOT, DummyArgs, doc_ids, train, public, public_ids)

    base = pickle.loads((ROOT / "results/burst_gpu_threeview/cpu_top20.pkl").read_bytes())["rankings"]
    documents = DocumentStore(paths)

    raw = {q: raw_union(retrieval[q], EXPANSION_CONFIG["depth"]) for q in public_ids}
    retrieval.clear()
    expansion_scores = dense_expansion(ROOT, DummyArgs.cache_dir, public, public_ids, raw, documents, "cuda")

    dense_rank = {q: sorted(raw[q], key=lambda d: (-expansion_scores[q][d], d)) for q in public_ids}
    expanded = weighted_rrf([raw, dense_rank], RERANK_CONFIG["expansion_weights"], RERANK_CONFIG["expansion_rrf_k"])
    corpus_rank, _ = corpus_dense(ROOT, DummyArgs.cache_dir, public, public_ids, "cuda", cap=CORPUS_CAP, depth=CORPUS_DEPTH)

    candidates = {
        q: list(dict.fromkeys(
            list(base[q]) + expanded[q][:RERANK_CONFIG["expanded_depth"]] + corpus_rank[q][:CORPUS_DEPTH]
        ))
        for q in public_ids
    }
    return public, candidates


def main():
    started_all = time.perf_counter()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== Scoring Public Queries with VietLegal-E5 on {device} ===", flush=True)

    # 1. Load Public Queries & Candidates
    print("Loading exact Huy public candidates...", flush=True)
    public_dict, candidates = load_exact_huy_public_candidates()
    public_ids = list(public_dict.keys())
    print(f"Loaded {len(public_ids)} public queries.", flush=True)

    # 2. Load Chunk Bank
    doc_row, positions_tensor, chunk_vectors = load_chunk_bank(device=device)

    # 3. Load Model
    encoder = E5QueryEncoder(MODEL_DIR, device=device)
    query_texts = [public_dict[q] for q in public_ids]
    qid_to_idx = {q: i for i, q in enumerate(public_ids)}

    # 4. Score Frozen VietLegal-E5
    print("Encoding frozen query representations...", flush=True)
    with encoder.model.disable_adapter():
        frozen_qvecs = encoder.encode_queries(query_texts, batch_size=16)

    print("Scoring frozen VietLegal-E5 on public candidates...", flush=True)
    frozen_scores: Dict[str, Dict[str, float]] = {}
    frozen_orders: Dict[str, List[str]] = {}
    missing_docs_set: Set[str] = set()
    missing_slots_frozen = 0
    total_candidate_rows = 0

    for qid in public_ids:
        q_idx = qid_to_idx[qid]
        q_vec = frozen_qvecs[q_idx]
        cands = candidates[qid]
        total_candidate_rows += len(cands)
        q_scores: Dict[str, float] = {}

        for doc_id in cands:
            if doc_id not in doc_row:
                q_scores[doc_id] = float("nan")
                missing_docs_set.add(doc_id)
                missing_slots_frozen += 1
                continue
            pos = positions_tensor[doc_row[doc_id]]
            sims = chunk_vectors[pos] @ q_vec
            count = min(2, len(sims))
            val = float(torch.topk(sims, k=count, sorted=False).values.mean().item())
            q_scores[doc_id] = val

        frozen_scores[qid] = q_scores
        frozen_orders[qid] = sort_candidates(q_scores)

    # 5. Score Full-Data Adapted VietLegal-E5
    print(f"Loading full-data adapter weights from {FULL_DATA_ADAPTER_PATH.name}...", flush=True)
    encoder.load_adapter_weights(FULL_DATA_ADAPTER_PATH)

    print("Encoding full-data adapted query representations...", flush=True)
    with torch.no_grad():
        adapted_qvecs = encoder.encode_queries(query_texts, batch_size=16)

    print("Scoring adapted VietLegal-E5 on public candidates...", flush=True)
    adapted_scores: Dict[str, Dict[str, float]] = {}
    adapted_orders: Dict[str, List[str]] = {}
    missing_slots_adapted = 0

    for qid in public_ids:
        q_idx = qid_to_idx[qid]
        q_vec = adapted_qvecs[q_idx]
        cands = candidates[qid]
        q_scores: Dict[str, float] = {}

        for doc_id in cands:
            if doc_id not in doc_row:
                q_scores[doc_id] = float("nan")
                missing_slots_adapted += 1
                continue
            pos = positions_tensor[doc_row[doc_id]]
            sims = chunk_vectors[pos] @ q_vec
            count = min(2, len(sims))
            val = float(torch.topk(sims, k=count, sorted=False).values.mean().item())
            q_scores[doc_id] = val

        adapted_scores[qid] = q_scores
        adapted_orders[qid] = sort_candidates(q_scores)

    elapsed = time.perf_counter() - started_all
    print(f"Scoring complete in {elapsed:.2f}s! Total rows: {total_candidate_rows}")
    print(f"Missing docs in bank: {len(missing_docs_set)} ({sorted(missing_docs_set)})")
    print(f"Missing slots: frozen={missing_slots_frozen}, adapted={missing_slots_adapted}")

    # Distribution summary
    all_ad_scores = [v for s in adapted_scores.values() for v in s.values() if not (math.isnan(v) or np.isnan(v))]
    all_fr_scores = [v for s in frozen_scores.values() for v in s.values() if not (math.isnan(v) or np.isnan(v))]

    ad_mean, ad_std = float(np.mean(all_ad_scores)), float(np.std(all_ad_scores))
    fr_mean, fr_std = float(np.mean(all_fr_scores)), float(np.std(all_fr_scores))

    print(f"Adapted score distribution: mean={ad_mean:.4f}, std={ad_std:.4f}, min={min(all_ad_scores):.4f}, max={max(all_ad_scores):.4f}")
    print(f"Frozen score distribution:  mean={fr_mean:.4f}, std={fr_std:.4f}, min={min(all_fr_scores):.4f}, max={max(all_fr_scores):.4f}")

    # Save
    payload = {
        "schema_version": "dsc2026.gemini.huy_e5_adaptation_delta_v1.public_scores.v1",
        "frozen_scores": frozen_scores,
        "adapted_scores": adapted_scores,
        "frozen_orders": frozen_orders,
        "adapted_orders": adapted_orders,
        "metadata": {
            "query_count": len(public_ids),
            "candidate_rows": total_candidate_rows,
            "missing_docs_in_bank": sorted(missing_docs_set),
            "missing_slots": missing_slots_frozen,
            "adapted_distribution": {"mean": ad_mean, "std": ad_std, "min": float(min(all_ad_scores)), "max": float(max(all_ad_scores))},
            "frozen_distribution": {"mean": fr_mean, "std": fr_std, "min": float(min(all_fr_scores)), "max": float(max(all_fr_scores))},
            "runtime_seconds": elapsed,
        },
    }

    out_pkl = RESULTS_DIR / "PUBLIC_VIETLEGAL_E5_SCORES.pkl"
    with out_pkl.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved: {out_pkl} ({out_pkl.stat().st_size / (1024*1024):.2f} MB)", flush=True)


if __name__ == "__main__":
    main()
