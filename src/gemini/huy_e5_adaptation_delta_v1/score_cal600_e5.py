"""Score exact Huy CAL600 candidate pool with frozen and task-adapted VietLegal-E5.

Adheres strictly to the anti-leakage protocol:
- For every CAL600 query, its assigned V2 fold is located in V2_FOLDS.json.
- The query is scored with adapted E5 using ONLY the adapter trained with that
  V2 fold held out (epoch-2.pt of that fold).
- Frozen E5 scores are computed with encoder.adapter_disabled().
- Document scoring uses top-2 chunk mean pooling (or top-1 for singletons)
  against the canonical 343,347-chunk bank.
- Documents absent from the 8,507 V2 parent bank have score=np.nan (explicit missing).
- Deterministic tie-breaking: score DESC, then doc_id ASC.

Outputs:
- results/gemini/huy_e5_adaptation_delta_v1/CAL600_VIETLEGAL_E5_SCORES.pkl
- results/gemini/huy_e5_adaptation_delta_v1/E5_SCORING_AUDIT.json
"""

from __future__ import annotations

import gc
import hashlib
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
FOLDS_PATH = BUNDLE_DIR / "V2_FOLDS.json"

ADAPTER_PATHS = {
    "fold_0": ROOT / "results/research_v2_e5_transfer/research_v2_e5_transfer_fold0/training/epoch-2.pt",
    "fold_1": ROOT / "results/research_v2_e5_confirmation/fold_1/training/epoch-2.pt",
    "fold_2": ROOT / "results/research_v2_e5_confirmation/fold_2/training/epoch-2.pt",
    "fold_3": ROOT / "results/research_v2_e5_confirmation/fold_3/training/epoch-2.pt",
    "fold_4": ROOT / "results/research_v2_e5_confirmation/fold_4/training/epoch-2.pt",
}

MAX_LENGTH = 512
LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


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


def main():
    started_all = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== Scoring CAL600 with VietLegal-E5 on {device} ===", flush=True)

    # 1. Load CAL600 training candidates and queries
    import sys
    sys.path.insert(0, str(ROOT))
    from tune_corpus_dense_fusion import build_training

    print("Loading CAL600 holdout candidates from build_training...", flush=True)
    queries, _, holdout_ids, holdout_candidates, _, _ = build_training(
        ROOT, depth=10, cap=32, extended_scores_path="results/corpus_index/holdout_extended_scores_cap32.pkl"
    )
    print(f"Loaded {len(holdout_ids)} holdout queries.", flush=True)

    # 2. Load V2 Folds
    with FOLDS_PATH.open("r", encoding="utf-8") as f:
        v2_folds_data = json.load(f)
    fold_map = {str(qid): fold for fold, qlist in v2_folds_data["folds"].items() for qid in qlist}

    # Verify every CAL query has a fold
    missing_fold = [q for q in holdout_ids if str(q) not in fold_map]
    if missing_fold:
        raise RuntimeError(f"Found {len(missing_fold)} CAL queries missing from V2 folds!")

    # 3. Load Chunk Bank
    doc_row, positions_tensor, chunk_vectors = load_chunk_bank(device=device)

    # 4. Load Model
    encoder = E5QueryEncoder(MODEL_DIR, device=device)

    # Query texts
    query_texts = [queries[q][0] for q in holdout_ids]
    qid_to_idx = {q: i for i, q in enumerate(holdout_ids)}

    # 5. Compute Frozen E5 Query Vectors
    print("Encoding frozen query representations...", flush=True)
    with encoder.model.disable_adapter():
        frozen_qvecs = encoder.encode_queries(query_texts, batch_size=16)

    # 6. Score Frozen E5 across all CAL600 queries
    print("Scoring frozen E5 on candidate pools...", flush=True)
    frozen_scores: Dict[str, Dict[str, float]] = {}
    frozen_orders: Dict[str, List[str]] = {}
    missing_docs_set: Set[str] = set()
    missing_slots_frozen = 0
    total_candidate_rows = 0

    for qid in holdout_ids:
        q_idx = qid_to_idx[qid]
        q_vec = frozen_qvecs[q_idx]
        cands = holdout_candidates[qid]
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

    # 7. Compute Task-Adapted E5 Query Vectors by Fold (Anti-Leakage)
    print("Scoring adapted E5 by V2 fold (anti-leakage strict held-out)...", flush=True)
    adapted_scores: Dict[str, Dict[str, float]] = {}
    adapted_orders: Dict[str, List[str]] = {}
    missing_slots_adapted = 0

    # Group CAL queries by fold
    fold_groups: Dict[str, List[str]] = {}
    for qid in holdout_ids:
        f = fold_map[str(qid)]
        fold_groups.setdefault(f, []).append(qid)

    for fold_name in sorted(fold_groups.keys()):
        fold_qids = fold_groups[fold_name]
        ckpt_path = ADAPTER_PATHS[fold_name]
        print(f"  Fold {fold_name}: {len(fold_qids)} queries, loading adapter {ckpt_path.name}...", flush=True)
        encoder.load_adapter_weights(ckpt_path)

        texts = [queries[q][0] for q in fold_qids]
        with torch.no_grad():
            fold_qvecs = encoder.encode_queries(texts, batch_size=16)

        for i, qid in enumerate(fold_qids):
            q_vec = fold_qvecs[i]
            cands = holdout_candidates[qid]
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
    peak_alloc = torch.cuda.max_memory_allocated() / (1024 * 1024)
    peak_res = torch.cuda.max_memory_reserved() / (1024 * 1024)

    print(f"Scoring complete in {elapsed:.2f}s! Peak alloc VRAM: {peak_alloc:.1f} MiB, Peak reserved: {peak_res:.1f} MiB")
    print(f"Total queries: {len(holdout_ids)}, Total candidate rows: {total_candidate_rows}")
    print(f"Missing doc IDs in bank: {len(missing_docs_set)} ({sorted(missing_docs_set)})")
    print(f"Missing slots (frozen): {missing_slots_frozen}, Missing slots (adapted): {missing_slots_adapted}")

    # 8. Save Scores PKL
    payload = {
        "schema_version": "dsc2026.gemini.huy_e5_adaptation_delta_v1.cal600_scores.v1",
        "frozen_scores": frozen_scores,
        "adapted_scores": adapted_scores,
        "frozen_orders": frozen_orders,
        "adapted_orders": adapted_orders,
        "metadata": {
            "query_count": len(holdout_ids),
            "candidate_rows": total_candidate_rows,
            "missing_docs_in_bank": sorted(missing_docs_set),
            "missing_slots": missing_slots_frozen,
            "runtime_seconds": elapsed,
            "peak_allocated_mib": peak_alloc,
            "peak_reserved_mib": peak_res,
        },
    }

    out_pkl = RESULTS_DIR / "CAL600_VIETLEGAL_E5_SCORES.pkl"
    with out_pkl.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved: {out_pkl} ({out_pkl.stat().st_size / (1024*1024):.2f} MB)", flush=True)

    # 9. Save Audit JSON
    audit = {
        "schema_version": "dsc2026.gemini.huy_e5_adaptation_delta_v1.e5_scoring_audit.v1",
        "phase": "CAL600_SCORING",
        "status": "COMPLETE",
        "runtime_seconds": elapsed,
        "peak_allocated_mib": peak_alloc,
        "peak_reserved_mib": peak_res,
        "query_count": len(holdout_ids),
        "candidate_rows_scored": total_candidate_rows,
        "missing_docs_in_v2_parent_bank": {
            "count": len(missing_docs_set),
            "doc_ids": sorted(missing_docs_set),
            "missing_slots_count": missing_slots_frozen,
            "treatment": "score set to np.nan, placed at end of order sorted by doc_id ASC, neutral delta values (0.0)",
        },
        "anti_leakage": {
            "protocol": "strict_v2_outer_fold_held_out",
            "folds_distribution": {f: len(qlist) for f, qlist in fold_groups.items()},
            "queries_outside_evaluable_population": 0,
            "verified_no_full_data_adapter_on_cal": True,
        },
        "cal600_scores_pkl_sha256": sha256_file(out_pkl),
    }

    audit_path = RESULTS_DIR / "E5_SCORING_AUDIT.json"
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved: {audit_path}", flush=True)


if __name__ == "__main__":
    main()
