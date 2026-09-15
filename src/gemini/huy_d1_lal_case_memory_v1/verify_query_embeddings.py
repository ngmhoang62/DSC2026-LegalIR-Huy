"""Verify query embedding parity between fresh encoding and frozen cached embeddings."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = ROOT / "results" / "gemini" / "huy_d1_lal_case_memory_v1"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

LAL_QUERIES = ROOT.parent / "LegalIR" / "cache" / "exp109b_encoder_complementarity" / "embeddings" / "vnlegal_lal" / "queries.npz"
MODEL_PATH = ROOT / "models" / "vnlegal-lal"
TRAIN_PATH = ROOT.parent / "LegalIR" / "public_test_dataset" / "train.json"

QUERY_PREFIX = "Instruct: Given a Vietnamese legal question, retrieve relevant legal passages that answer the question\nQuery: "


def normalize(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    return array / np.maximum(norms, 1e-12)


class FrozenLALQueryEncoder:
    def __init__(self, model_path: Path, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True, use_fast=True)
        self.model = AutoModel.from_pretrained(str(model_path), local_files_only=True)
        self.model.to(self.device)
        self.model.eval()

    def encode(self, texts: Sequence[str], batch_size: int = 16) -> np.ndarray:
        prepared = [QUERY_PREFIX + str(t) for t in texts]
        vectors = []
        with torch.inference_mode():
            for i in range(0, len(prepared), batch_size):
                batch = prepared[i : i + batch_size]
                tokens = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=2048,
                    return_tensors="pt",
                )
                tokens = {k: v.to(self.device) for k, v in tokens.items()}
                hidden = self.model(**tokens).last_hidden_state
                mask = tokens["attention_mask"].bool()
                reversed_mask = torch.flip(mask, dims=[1])
                from_right = reversed_mask.long().argmax(dim=1)
                positions = mask.shape[1] - 1 - from_right
                pooled = hidden[torch.arange(hidden.shape[0], device=hidden.device), positions]
                normed = F.normalize(pooled.float(), p=2.0, dim=-1, eps=1e-12).cpu().numpy()
                vectors.append(normed)
        return np.vstack(vectors) if vectors else np.empty((0, 1024), dtype=np.float32)


def run_embedding_parity(sample_size: int = 64, seed: int = 2026) -> Dict[str, Any]:
    with np.load(LAL_QUERIES, allow_pickle=False) as z:
        cached_qids = list(map(str, z["query_ids"].tolist()))
        cached_vectors = normalize(z["vectors"].astype(np.float32))

    qid_to_vec = {qid: cached_vectors[i] for i, qid in enumerate(cached_qids)}

    with open(TRAIN_PATH, "r", encoding="utf-8") as f:
        train_data = json.load(f)

    rng = random.Random(seed)
    common_qids = sorted(set(cached_qids) & set(train_data.keys()), key=int)
    sample_qids = rng.sample(common_qids, sample_size)

    encoder = FrozenLALQueryEncoder(MODEL_PATH)
    questions = [train_data[qid]["question"] for qid in sample_qids]
    fresh_vectors = encoder.encode(questions)

    cosines = []
    l2s = []
    top20_agreements = []

    for i, qid in enumerate(sample_qids):
        cached_v = qid_to_vec[qid]
        fresh_v = fresh_vectors[i]
        cos = float(np.dot(cached_v, fresh_v))
        l2 = float(np.linalg.norm(cached_v - fresh_v))
        cosines.append(cos)
        l2s.append(l2)

        sim_cached = cached_vectors @ cached_v
        sim_fresh = cached_vectors @ fresh_v
        top20_cached = set(np.argpartition(-sim_cached, 20)[:20])
        top20_fresh = set(np.argpartition(-sim_fresh, 20)[:20])
        jaccard = len(top20_cached & top20_fresh) / len(top20_cached | top20_fresh)
        top20_agreements.append(jaccard)

    min_cos = float(min(cosines))
    mean_cos = float(np.mean(cosines))
    max_l2 = float(max(l2s))
    mean_top20 = float(np.mean(top20_agreements))
    min_top20 = float(min(top20_agreements))

    mean_pass = mean_cos >= 0.99999
    min_pass = min_cos >= 0.9999
    overall_pass = mean_pass and min_pass

    parity_report = {
        "schema_version": "dsc2026.gemini.huy_d1_lal_case_memory_v1.query_embedding_parity.v1",
        "status": "PASS" if overall_pass else "BLOCKED_EMBEDDING_PARITY",
        "encoder_model_path": str(MODEL_PATH),
        "cached_queries_npz": str(LAL_QUERIES),
        "sample_size": sample_size,
        "seed": seed,
        "mean_cosine": mean_cos,
        "min_cosine": min_cos,
        "max_l2": max_l2,
        "mean_top20_agreement": mean_top20,
        "min_top20_agreement": min_top20,
        "thresholds": {
            "mean_cosine_min": 0.99999,
            "min_cosine_min": 0.9999,
        },
        "gates": {
            "mean_cosine_passed": mean_pass,
            "min_cosine_passed": min_pass,
        },
        "all_parity_passed": overall_pass,
    }

    out_path = RESULTS_DIR / "LAL_QUERY_EMBEDDING_PARITY.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(parity_report, f, indent=2)
    print(f"Wrote {out_path} (status: {parity_report['status']})")
    return parity_report


if __name__ == "__main__":
    run_embedding_parity()
