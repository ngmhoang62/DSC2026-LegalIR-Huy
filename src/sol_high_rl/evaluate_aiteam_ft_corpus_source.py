"""Evaluate the sealed AITeamVN-FT cap-32 index as a label-free CAL600 source."""

from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from full_corpus_title_retrieval import encode, load_model
from tune_corpus_cap32_fusion import build_training_cap


CACHE = ROOT / "cache/sol_high_rl/aiteam_ft_full_corpus"
META_PATH = CACHE / "chunks_cap32.json"
VECTOR_PATH = CACHE / "chunks_cap32.f16"
QUERY_VECTOR_PATH = CACHE / "cal600_query_vectors_max512.f32.npy"
RANKING_PATH = ROOT / "results/sol_high_rl/AITEAM_FT_FULL_CORPUS_TOP50.json"
REPORT_PATH = ROOT / "results/sol_high_rl/AITEAM_FT_FULL_CORPUS_REPORT.json"
TOP_CHUNKS = 2048
CHUNK_BATCH = 8192


def recall(queries, rankings, ids, depth):
    return float(np.mean([len(set(rankings[q][:depth]) & queries[q][1]) / len(queries[q][1]) for q in ids]))


def candidate_oracle(queries, candidates, ids):
    return float(np.mean([len(set(candidates[q]) & queries[q][1]) / len(queries[q][1]) for q in ids]))


def validate_index(meta):
    context_ids = {
        p.stem[len("context_") :]
        for p in (ROOT / "DSC2026-LegalIR-main/v4_run/public_test_dataset/selected-contexts").glob("context_*.json")
    }
    docs = meta["documents"]
    counts = meta["counts"]
    expected_bytes = sum(counts) * 1024 * 2
    vectors = np.memmap(VECTOR_PATH, mode="r", dtype=np.float16, shape=(sum(counts), 1024))
    sample_indices = np.linspace(0, len(vectors) - 1, 257).round().astype(int)
    norms = np.linalg.norm(np.asarray(vectors[sample_indices], dtype=np.float32), axis=1)
    return {
        "documents": len(docs),
        "unique_documents": len(set(docs)),
        "context_documents": len(context_ids),
        "missing_parent_mappings": sorted(context_ids - set(docs)),
        "extra_parent_mappings": sorted(set(docs) - context_ids),
        "duplicate_parent_mappings": len(docs) - len(set(docs)),
        "chunk_count": sum(counts),
        "min_chunks_per_parent": min(counts),
        "max_chunks_per_parent": max(counts),
        "expected_bytes": expected_bytes,
        "actual_bytes": VECTOR_PATH.stat().st_size,
        "sample_nonfinite_values": int(np.sum(~np.isfinite(np.asarray(vectors[sample_indices])))),
        "sample_norm_min": float(norms.min()),
        "sample_norm_max": float(norms.max()),
        "passed": (
            len(docs) == len(context_ids) == len(set(docs)) == 8532
            and set(docs) == context_ids
            and expected_bytes == VECTOR_PATH.stat().st_size
            and int(np.sum(~np.isfinite(np.asarray(vectors[sample_indices])))) == 0
        ),
    }


def main():
    started = time.perf_counter()
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    integrity = validate_index(meta)
    if not integrity["passed"]:
        raise RuntimeError(f"Index integrity failed: {integrity}")
    queries, old_blocks, all_ids, current, _, _ = build_training_cap(
        ROOT, 32, "results/corpus_index/holdout_extended_scores_cap32.pkl", depth=20
    )
    split = json.loads((ROOT / "results/sol_high_rl/CAL600_STRATIFIED_5FOLD_SEED42.json").read_text(encoding="utf-8"))
    folds = split["folds"]
    if QUERY_VECTOR_PATH.exists():
        query_vectors = np.load(QUERY_VECTOR_PATH)
    else:
        model, tokenizer = load_model()
        query_vectors = encode(model, tokenizer, [queries[q][0] for q in all_ids], batch_size=64, max_length=512)
        np.save(QUERY_VECTOR_PATH, query_vectors.astype(np.float32))
        del model
        gc.collect()
        torch.cuda.empty_cache()

    docs = meta["documents"]
    parent_index = np.repeat(np.arange(len(docs), dtype=np.int32), np.asarray(meta["counts"], dtype=np.int32))
    total_chunks = len(parent_index)
    vectors = np.memmap(VECTOR_PATH, mode="r", dtype=np.float16, shape=(total_chunks, 1024))
    qgpu = torch.from_numpy(query_vectors).to(device="cuda", dtype=torch.float16)
    best_scores = torch.full((len(all_ids), TOP_CHUNKS), -torch.inf, device="cuda", dtype=torch.float16)
    best_indices = torch.full((len(all_ids), TOP_CHUNKS), -1, device="cuda", dtype=torch.int64)

    missing_rows = json.loads((ROOT / "results/sol_high_rl/OUT_OF_POOL_GOLD.json").read_text(encoding="utf-8"))
    missing_qids = sorted({row["qid"] for row in missing_rows}, key=all_ids.index)
    miss_query_rows = [all_ids.index(q) for q in missing_qids]
    miss_parent_scores = np.full((len(missing_qids), len(docs)), -np.inf, dtype=np.float32)

    for begin in range(0, total_chunks, CHUNK_BATCH):
        end = min(begin + CHUNK_BATCH, total_chunks)
        block = torch.from_numpy(np.asarray(vectors[begin:end])).to("cuda")
        score = qgpu @ block.T
        k = min(TOP_CHUNKS, end - begin)
        local_scores, local_indices = torch.topk(score, k=k, dim=1)
        local_indices += begin
        merged_scores = torch.cat([best_scores, local_scores], dim=1)
        merged_indices = torch.cat([best_indices, local_indices], dim=1)
        best_scores, positions = torch.topk(merged_scores, k=TOP_CHUNKS, dim=1)
        best_indices = torch.gather(merged_indices, 1, positions)

        miss_scores = score[miss_query_rows].float().cpu().numpy()
        block_parents = parent_index[begin:end]
        for row_index in range(len(missing_qids)):
            np.maximum.at(miss_parent_scores[row_index], block_parents, miss_scores[row_index])
        del block, score, local_scores, local_indices, merged_scores, merged_indices, positions
        if (begin // CHUNK_BATCH + 1) % 5 == 0 or end == total_chunks:
            print(f"retrieved chunks {end}/{total_chunks}", flush=True)

    chunk_scores = best_scores.float().cpu().numpy()
    chunk_indices = best_indices.cpu().numpy()
    rankings = {}
    for qi, qid in enumerate(all_ids):
        parent_best = {}
        for score, chunk_index in zip(chunk_scores[qi], chunk_indices[qi]):
            parent = int(parent_index[int(chunk_index)])
            parent_best[parent] = max(parent_best.get(parent, -np.inf), float(score))
        ordered = sorted(parent_best, key=lambda p: (-parent_best[p], docs[p]))
        if len(ordered) < 50:
            raise RuntimeError(f"Top-chunk guarantee failed for {qid}: {len(ordered)} parents")
        rankings[qid] = [docs[p] for p in ordered[:50]]
    RANKING_PATH.write_text(json.dumps(rankings, ensure_ascii=False, indent=2), encoding="utf-8")

    exact_missing_ranks = []
    missing_index = {q: i for i, q in enumerate(missing_qids)}
    doc_pos = {d: i for i, d in enumerate(docs)}
    for row in missing_rows:
        scores = miss_parent_scores[missing_index[row["qid"]]]
        target = doc_pos[row["doc_id"]]
        order = np.lexsort((np.asarray(docs), -scores))
        rank = int(np.where(order == target)[0][0]) + 1
        exact_missing_ranks.append({**row, "aiteam_ft_parent_rank": rank, "max_chunk_cosine": float(scores[target])})

    current_oracle = candidate_oracle(queries, current, all_ids)
    report = {
        "status": "COMPLETE",
        "contract_parity": "PASS_CONTRACT",
        "index_integrity": integrity,
        "source_contract": {"checkpoint": "AITeamVN-FT", "pooling": "CLS+L2", "prefixes": "none", "max_length": 512, "window_words": 220, "step_words": 150, "cap": 32, "parent_aggregation": "max chunk cosine"},
        "current_candidate_oracle": current_oracle,
        "depths": {},
        "exact_out_of_pool_gold_ranks": exact_missing_ranks,
        "runtime_seconds": time.perf_counter() - started,
    }
    missing_pairs = {(row["qid"], row["doc_id"]) for row in missing_rows}
    for depth in (5, 10, 20, 50):
        source = {q: rankings[q][:depth] for q in all_ids}
        union = {q: list(dict.fromkeys(current[q] + source[q])) for q in all_ids}
        rescued = {(q, d) for q, d in missing_pairs if d in source[q]}
        novel = [sum(d not in current[q] for d in source[q]) for q in all_ids]
        report["depths"][str(depth)] = {
            "standalone_recall": recall(queries, rankings, all_ids, depth),
            "union_oracle": candidate_oracle(queries, union, all_ids),
            "union_oracle_delta": candidate_oracle(queries, union, all_ids) - current_oracle,
            "unique_out_of_pool_gold_rescues": len(rescued),
            "rescued": [{"qid": q, "doc_id": d} for q, d in sorted(rescued)],
            "mean_novel_documents": float(np.mean(novel)),
            "median_novel_documents": float(np.median(novel)),
            "folds": {
                name: {"standalone_recall": recall(queries, rankings, ids, depth), "union_oracle": candidate_oracle(queries, union, ids), "rescues": sum(q in set(ids) for q, _ in rescued)}
                for name, ids in folds.items()
            },
            "old_block_stress": {
                name: {"standalone_recall": recall(queries, rankings, ids, depth), "union_oracle": candidate_oracle(queries, union, ids), "rescues": sum(q in set(ids) for q, _ in rescued)}
                for name, ids in old_blocks.items()
            },
        }
    gate = report["depths"]["20"]
    rescue_blocks = sum(x["rescues"] > 0 for x in gate["old_block_stress"].values())
    report["candidate_gate"] = {
        "criteria": "top20 union delta >=0.003, >=4 occurrence rescues, >=2 old blocks",
        "rescue_blocks": rescue_blocks,
        "passed": gate["union_oracle_delta"] >= 0.003 and gate["unique_out_of_pool_gold_rescues"] >= 4 and rescue_blocks >= 2,
    }
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    compact = dict(report)
    compact["exact_out_of_pool_gold_ranks"] = exact_missing_ranks
    print(json.dumps(compact, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
