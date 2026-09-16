"""Build and seal teacher scores cache for training pairs with full provenance and doc-parsing cache."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import numpy as np
import torch

from .audit_split import run_split_audit
from .common import (
    EXPECTED_SHIPPED_SHA256,
    RES_DIR,
    ROOT,
    SRC_DIR,
    V2_CANDIDATE_POOL_JSONL,
    V2_CONTEXTS_JSONL,
    V2_QUERIES_JSONL,
    WEIGHTS_JINA_FT,
    compute_qid_list_fingerprint,
    get_git_status,
    load_jina_base_with_shipped_weights,
    load_v2_contexts,
    load_v2_inputs,
    seed_everything,
    sha256_file,
)
from .legal_section_parser import LegalSection, parse_document_into_sections, preselect_legal_sections

TEACHER_PAIRS_FILE = RES_DIR / "TEACHER_TRAINING_PAIRS.jsonl"
TEACHER_SEAL_FILE = RES_DIR / "TEACHER_SCORE_CACHE_SEAL.json"


def compute_raw_batch_logits(
    model, tok, sentence_pairs: List[Tuple[str, str]], batch_size: int = 64, max_length: int = 512
) -> List[float]:
    """Compute raw pre-sigmoid logits efficiently in batches."""
    all_logits: List[float] = []
    for start_idx in range(0, len(sentence_pairs), batch_size):
        batch = sentence_pairs[start_idx : start_idx + batch_size]
        inputs = tok(
            batch,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=max_length,
        ).to(model.device)
        with torch.no_grad():
            logits = model(**inputs, return_dict=True).logits.view(-1).float()
        all_logits.extend(logits.cpu().numpy().tolist())
    return all_logits


def build_teacher_cache(
    force_fresh: bool = False,
    query_chunk_size: int = 50,
    batch_size: int = 64,
) -> Dict[str, Any]:
    print("=== STAGE: BUILD & SEAL TEACHER TRAINING PAIRS ===", flush=True)
    RES_DIR.mkdir(parents=True, exist_ok=True)
    seed_everything(2026)
    git_info = get_git_status()

    # 1. Load exact split audit and assert split parity
    split_audit_path = RES_DIR / "SPLIT_AUDIT.json"
    if not split_audit_path.exists():
        run_split_audit()
    split_audit = json.loads(split_audit_path.read_text(encoding="utf-8"))

    sealed_train_qids = split_audit["training_partition"]["train_qids_with_in_pool_gold"]
    expected_fp = split_audit["training_partition"]["train_qids_with_in_pool_gold_fingerprint"]
    actual_fp = compute_qid_list_fingerprint(sealed_train_qids)

    if actual_fp != expected_fp:
        raise RuntimeError(
            f"BLOCKED_SPLIT_PARITY: train_qids_with_in_pool_gold fingerprint mismatch!\n"
            f"Expected: {expected_fp}\nActual:   {actual_fp}"
        )

    print(f"[TEACHER_CACHE] Loaded {len(sealed_train_qids)} sealed training queries (FP: {actual_fp[:12]}...).", flush=True)

    # 2. Check if valid sealed cache already exists
    if not force_fresh and TEACHER_SEAL_FILE.exists() and TEACHER_PAIRS_FILE.exists():
        try:
            seal_data = json.loads(TEACHER_SEAL_FILE.read_text(encoding="utf-8"))
            pairs_sha = sha256_file(TEACHER_PAIRS_FILE)
            if (
                seal_data.get("pairs_sha256") == pairs_sha
                and seal_data.get("git_commit") == git_info["head_commit"]
                and seal_data.get("train_qids_fingerprint") == expected_fp
                and seal_data.get("weights_sha256") == EXPECTED_SHIPPED_SHA256
                and seal_data.get("status") == "PASS"
            ):
                print(
                    f"[TEACHER_CACHE] Valid sealed cache matches current Git commit and split fingerprint: "
                    f"{TEACHER_PAIRS_FILE} ({seal_data.get('total_training_pairs')} pairs). Reusing.",
                    flush=True,
                )
                return seal_data
        except Exception as e:
            print(f"[TEACHER_CACHE] Could not reuse existing seal: {e}. Rebuilding.", flush=True)

    # 3. Load V2 inputs and contexts
    folds, pools, questions, v2_golds = load_v2_inputs()
    contexts = load_v2_contexts()

    # Pre-parse and cache document sections in memory: unique doc parsed at most ONCE
    doc_sections_cache: Dict[str, List[LegalSection]] = {}

    def get_cached_sections(doc_id: str) -> List[LegalSection]:
        if doc_id not in doc_sections_cache:
            raw_text = contexts.get(doc_id, "")
            doc_sections_cache[doc_id] = parse_document_into_sections(
                doc_id, raw_text, max_chunk_words=220, overlap_words=60
            )
        return doc_sections_cache[doc_id]

    # 4. Load Shipped Teacher Model
    print("[TEACHER_CACHE] Loading frozen shipped Jina model...", flush=True)
    model, tok = load_jina_base_with_shipped_weights()
    model.eval().to("cuda")

    # 5. Process queries in chunks
    if TEACHER_PAIRS_FILE.exists():
        TEACHER_PAIRS_FILE.unlink()

    total_pairs = 0
    total_attempted_pairs = 0
    unavailable_section_docs_count = 0
    t0_start = time.time()
    out_f = open(TEACHER_PAIRS_FILE, "w", encoding="utf-8")

    try:
        for chunk_idx in range(0, len(sealed_train_qids), query_chunk_size):
            chunk_qids = sealed_train_qids[chunk_idx : chunk_idx + query_chunk_size]
            pair_list: List[Tuple[str, str]] = []
            pair_meta: List[Tuple[str, str, int]] = []  # (qid, doc_id, section_idx)
            query_doc_secs: Dict[Tuple[str, str], List[str]] = {}

            # Preselect sections for all candidate documents in chunk
            for qid in chunk_qids:
                q_text = questions[qid]
                cand_docs = [str(d) for d in pools[qid]]
                for doc_id in cand_docs:
                    secs = get_cached_sections(doc_id)
                    chosen_secs = preselect_legal_sections(q_text, secs, count=2)
                    if not chosen_secs:
                        unavailable_section_docs_count += 1
                        continue
                    sec_texts = [s.text for s in chosen_secs]
                    query_doc_secs[(qid, doc_id)] = sec_texts
                    for s_idx, sec in enumerate(chosen_secs):
                        pair_list.append((q_text, sec.text))
                        pair_meta.append((qid, doc_id, s_idx))

            # Batch scoring through GPU
            if pair_list:
                raw_logits = compute_raw_batch_logits(
                    model, tok, pair_list, batch_size=batch_size, max_length=512
                )
            else:
                raw_logits = []

            # Aggregate MAX doc logit: (qid, doc_id) -> float
            doc_logits: Dict[Tuple[str, str], float] = {}
            for (qid, doc_id, s_idx), logit_val in zip(pair_meta, raw_logits):
                doc_logits[(qid, doc_id)] = max(doc_logits.get((qid, doc_id), -1e9), float(logit_val))

            # Construct training pairs per query
            for qid in chunk_qids:
                q_text = questions[qid]
                cand_docs = [str(d) for d in pools[qid]]
                gold_docs = set(str(d) for d in v2_golds.get(qid, set()))

                # Only consider documents with usable sections
                pos_docs = [d for d in cand_docs if d in gold_docs and (qid, d) in query_doc_secs]
                non_gold_docs = [d for d in cand_docs if d not in gold_docs and (qid, d) in query_doc_secs]

                if not pos_docs or not non_gold_docs:
                    continue

                # Sort non-gold docs by teacher score descending
                non_gold_docs.sort(key=lambda d: doc_logits.get((qid, d), -1e9), reverse=True)
                top3_hard_negs = non_gold_docs[:3]

                # Weight normalization: total weight per query = 1.0
                pair_weight = 1.0 / (len(pos_docs) * len(top3_hard_negs))

                for p_id in pos_docs:
                    p_logit = doc_logits.get((qid, p_id), 0.0)
                    p_secs = query_doc_secs[qid, p_id]
                    for n_id in top3_hard_negs:
                        n_logit = doc_logits.get((qid, n_id), 0.0)
                        n_secs = query_doc_secs[qid, n_id]
                        total_attempted_pairs += 1

                        # Strict assertion: never fabricate empty or query-query pairs
                        assert len(p_secs) > 0 and len(n_secs) > 0, "Empty sections detected in pair!"

                        rec = {
                            "qid": qid,
                            "query_text": q_text,
                            "pos_doc_id": p_id,
                            "pos_sections": p_secs,
                            "teacher_pos_logit": p_logit,
                            "neg_doc_id": n_id,
                            "neg_sections": n_secs,
                            "teacher_neg_logit": n_logit,
                            "pair_weight": pair_weight,
                        }
                        out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        total_pairs += 1

            if (chunk_idx + query_chunk_size) % 500 == 0 or (chunk_idx + query_chunk_size) >= len(sealed_train_qids):
                elapsed = time.time() - t0_start
                done_cnt = min(chunk_idx + query_chunk_size, len(sealed_train_qids))
                print(
                    f"[TEACHER_CACHE] Processed {done_cnt}/{len(sealed_train_qids)} queries "
                    f"({total_pairs} pairs, elapsed: {elapsed:.1f}s, unique parsed docs cached: {len(doc_sections_cache)})",
                    flush=True,
                )
    finally:
        out_f.close()

    # 6. Seal the cache with complete provenance
    pairs_sha256 = sha256_file(TEACHER_PAIRS_FILE)
    seal = {
        "schema_version": "dsc2026.gemini.huy_d1_jina_passage_adaptation_pilot_v2.teacher_cache_seal.v2",
        "experiment_id": "HUY_D1_JINA_PASSAGE_ADAPTATION_PILOT_V2",
        "status": "PASS",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_info["head_commit"],
        "build_teacher_cache_sha256": sha256_file(SRC_DIR / "build_teacher_cache.py"),
        "common_sha256": sha256_file(SRC_DIR / "common.py"),
        "legal_section_parser_sha256": sha256_file(SRC_DIR / "legal_section_parser.py"),
        "weights_sha256": EXPECTED_SHIPPED_SHA256,
        "train_qids_fingerprint": expected_fp,
        "v2_queries_fingerprint": sha256_file(V2_QUERIES_JSONL),
        "v2_candidate_pool_fingerprint": sha256_file(V2_CANDIDATE_POOL_JSONL),
        "v2_corpus_fingerprint": sha256_file(V2_CONTEXTS_JSONL),
        "section_parser_config": {
            "max_chunk_words": 220,
            "overlap_words": 60,
            "count_sections": 2,
            "aggregation": "MAX",
        },
        "hard_negative_config": {
            "top_k_negatives": 3,
            "strategy": "frozen_teacher_highest_non_gold",
        },
        "teacher_score_semantics": "raw_pre_sigmoid_logits_max_over_top2_sections",
        "max_length": 512,
        "coverage_statistics": {
            "total_training_queries": len(sealed_train_qids),
            "total_training_pairs": total_pairs,
            "total_attempted_pairs": total_attempted_pairs,
            "unavailable_section_docs_count": unavailable_section_docs_count,
            "usable_pair_coverage_fraction": total_pairs / max(1, total_attempted_pairs),
        },
        "pairs_file": str(TEACHER_PAIRS_FILE.relative_to(ROOT)),
        "pairs_sha256": pairs_sha256,
    }

    TEACHER_SEAL_FILE.write_text(json.dumps(seal, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"[TEACHER_CACHE] Cache sealed successfully -> {TEACHER_SEAL_FILE} "
        f"({total_pairs} pairs, SHA256: {pairs_sha256[:12]}...)",
        flush=True,
    )
    return seal


if __name__ == "__main__":
    build_teacher_cache()
