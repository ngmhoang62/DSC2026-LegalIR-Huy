"""Frozen AITeamVN-FT full-corpus Top-50 source loader and exact parity verification.

Module for HUY_D1_AITEAM50_SOFT_ADMISSION_V1.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from src.gemini.huy_d1_aiteam50_soft_admission_v1.common import (
    AITEAM_REPORT_PATH,
    AITEAM_TOP50_PATH,
    EXPECTED_AITEAM_REPORT_SHA256,
    EXPECTED_AITEAM_TOP50_SHA256,
    RESULTS_DIR,
    ROOT,
    sha256_file,
)

CACHE_DIR = ROOT / "cache/sol_high_rl/aiteam_ft_full_corpus"
META_PATH = CACHE_DIR / "chunks_cap32.json"
VECTOR_PATH = CACHE_DIR / "chunks_cap32.f16"
QUERY_VECTOR_PATH = CACHE_DIR / "cal600_query_vectors_max512.f32.npy"
CHUNK_BATCH = 16384


def load_and_verify_aiteam_full_corpus(
    all_ids: List[str],
) -> Tuple[Dict[str, List[str]], Dict[str, Dict[str, float]], Dict[str, Any]]:
    """Loads frozen AITeam Top-50, computes full-corpus max-chunk cosine on GPU,
    verifies 600/600 Top-50 exact ordered parity, and writes AITEAM50_SOURCE_PARITY.json.
    """
    print("=== COMPUTING AITEAMVN-FT FULL-CORPUS MAX-CHUNK COSINE ON GPU ===", flush=True)
    t0 = time.perf_counter()

    # 1. Verify frozen input hashes
    top50_sha = sha256_file(AITEAM_TOP50_PATH)
    if top50_sha != EXPECTED_AITEAM_TOP50_SHA256:
        raise RuntimeError(
            f"AITEAM_TOP50_PATH SHA256 mismatch! Got {top50_sha}, expected {EXPECTED_AITEAM_TOP50_SHA256}"
        )
    report_sha = sha256_file(AITEAM_REPORT_PATH)
    if report_sha != EXPECTED_AITEAM_REPORT_SHA256:
        raise RuntimeError(
            f"AITEAM_REPORT_PATH SHA256 mismatch! Got {report_sha}, expected {EXPECTED_AITEAM_REPORT_SHA256}"
        )

    frozen_top50: Dict[str, List[str]] = json.loads(
        AITEAM_TOP50_PATH.read_text(encoding="utf-8")
    )
    if len(frozen_top50) != 600:
        raise ValueError(f"Expected 600 queries in AITeam Top-50, got {len(frozen_top50)}")

    # 2. Load index metadata and vectors
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    docs: List[str] = meta["documents"]
    counts: List[int] = meta["counts"]
    parent_index = np.repeat(
        np.arange(len(docs), dtype=np.int32), np.asarray(counts, dtype=np.int32)
    )
    total_chunks = len(parent_index)

    query_vectors = np.load(QUERY_VECTOR_PATH)
    vectors = np.memmap(
        VECTOR_PATH, mode="r", dtype=np.float16, shape=(total_chunks, 1024)
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    qgpu = torch.from_numpy(query_vectors).to(device=device, dtype=torch.float16)
    parent_tensor = torch.from_numpy(parent_index).to(device=device, dtype=torch.int64)

    doc_scores = torch.full(
        (len(all_ids), len(docs)), -torch.inf, device=device, dtype=torch.float32
    )

    for begin in range(0, total_chunks, CHUNK_BATCH):
        end = min(begin + CHUNK_BATCH, total_chunks)
        block = torch.from_numpy(np.asarray(vectors[begin:end])).to(
            device=device, dtype=torch.float16
        )
        sims = qgpu @ block.T
        sub_parent = parent_tensor[begin:end]
        doc_scores.scatter_reduce_(
            dim=1,
            index=sub_parent.unsqueeze(0).expand(len(all_ids), -1),
            src=sims.float(),
            reduce="amax",
            include_self=True,
        )

    # 3. Extract Top-60 for exact parity comparison with doc_id tie-breaking
    top_scores, top_indices = torch.topk(doc_scores, k=60, dim=1)
    top_scores_np = top_scores.cpu().numpy()
    top_indices_np = top_indices.cpu().numpy()
    doc_scores_cpu = doc_scores.cpu().numpy()

    # Map qid -> doc -> score
    full_doc_scores: Dict[str, Dict[str, float]] = {}
    doc_to_idx = {d: i for i, d in enumerate(docs)}

    matches = 0
    mismatches = []

    for qi, qid in enumerate(all_ids):
        # Sort top 60 with tie-break
        items = [(top_scores_np[qi, j], docs[top_indices_np[qi, j]]) for j in range(60)]
        items.sort(key=lambda x: (-x[0], x[1]))
        rebuilt_top50 = [x[1] for x in items[:50]]

        target_top50 = frozen_top50[qid]
        if rebuilt_top50 == target_top50:
            matches += 1
        else:
            mismatches.append({
                "qid": qid,
                "rebuilt_top5": rebuilt_top50[:5],
                "frozen_top5": target_top50[:5],
            })

    elapsed = time.perf_counter() - t0
    print(
        f"AITeam full-corpus GPU scoring finished in {elapsed:.2f}s. Matches: {matches} / {len(all_ids)}",
        flush=True,
    )

    parity_pass = (matches == len(all_ids))
    parity_doc = {
        "schema_version": "dsc2026.gemini.huy_d1_aiteam50_soft_admission_v1.aiteam50_parity.v1",
        "experiment_id": "HUY_D1_AITEAM50_SOFT_ADMISSION_V1",
        "top50_path": str(AITEAM_TOP50_PATH.relative_to(ROOT)).replace("\\", "/"),
        "top50_sha256": top50_sha,
        "report_sha256": report_sha,
        "query_count": len(all_ids),
        "exact_matches": matches,
        "parity_pass": parity_pass,
        "status": "PASS" if parity_pass else "BLOCKED_AITEAM_SCORE_PARITY",
        "elapsed_seconds": round(elapsed, 2),
        "mismatches": mismatches[:5],
    }

    out_path = RESULTS_DIR / "AITEAM50_SOURCE_PARITY.json"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(parity_doc, indent=2, ensure_ascii=False), encoding="utf-8")

    if not parity_pass:
        raise RuntimeError(f"BLOCKED_AITEAM_SCORE_PARITY: only {matches}/600 matched!")

    # Build per-query lookup for any requested document in S(q)
    # To keep memory bounded, we store doc_scores_cpu and doc_to_idx accessor
    class AITeamScoreAccessor:
        def __init__(self, scores_matrix: np.ndarray, doc_map: Dict[str, int], id_list: List[str]):
            self.matrix = scores_matrix
            self.doc_map = doc_map
            self.qid_to_row = {q: i for i, q in enumerate(id_list)}

        def get_score(self, qid: str, doc_id: str) -> float:
            row = self.qid_to_row[qid]
            col = self.doc_map.get(doc_id)
            if col is None:
                return -1.0
            return float(self.matrix[row, col])

    accessor = AITeamScoreAccessor(doc_scores_cpu, doc_to_idx, all_ids)
    return frozen_top50, accessor, parity_doc
