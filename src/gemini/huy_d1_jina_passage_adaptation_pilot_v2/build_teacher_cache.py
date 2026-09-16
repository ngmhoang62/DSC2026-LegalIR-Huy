"""Build and seal teacher scores cache for training pairs."""

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
    WEIGHTS_JINA_FT,
    get_git_status,
    load_jina_base_with_shipped_weights,
    load_v2_contexts,
    load_v2_inputs,
    seed_everything,
    sha256_file,
)
from .legal_section_parser import parse_document_into_sections, preselect_legal_sections

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

    # 1. Check if sealed cache already exists and is valid
    if not force_fresh and TEACHER_SEAL_FILE.exists() and TEACHER_PAIRS_FILE.exists():
        try:
            seal_data = json.loads(TEACHER_SEAL_FILE.read_text(encoding="utf-8"))
            pairs_sha = sha256_file(TEACHER_PAIRS_FILE)
            if (
                seal_data.get("pairs_sha256") == pairs_sha
                and seal_data.get("weights_sha256") == EXPECTED_SHIPPED_SHA256
                and seal_data.get("status") == "PASS"
            ):
                print(
                    f"[TEACHER_CACHE] Valid sealed cache found: {TEACHER_PAIRS_FILE} "
                    f"({seal_data.get('total_training_pairs')} pairs). Reusing.",
                    flush=True,
                )
                return seal_data
        except Exception as e:
            print(f"[TEACHER_CACHE] Could not verify existing seal: {e}. Rebuilding.", flush=True)

    # 2. Load audited split and inputs
    split_audit = run_split_audit()
    train_queries_with_gold = set(split_audit["training_partition"]["sample_train_qids"])
    # Full list of valid training queries with in-pool gold
    folds, pools, questions, v2_golds = load_v2_inputs()
    contexts = load_v2_contexts()

    # Determine exact training population from audit rules
    cal_raw_qids = split_audit["canonical_v2_total_queries"]
    # Re-extract the exact 5,043 train qids with in-pool gold
    held_qids = set(split_audit["held_partition"]["sample_held_qids"])  # sample
    from tune_corpus_cap32_fusion import build_training_cap
    _, _, all_cal_ids, _, _, _ = build_training_cap(
        ROOT, 32, "results/corpus_index/holdout_extended_scores_cap32.pkl", depth=20
    )
    cal_qids = set(str(q) for q in all_cal_ids)

    train_raw = []
    for f in ["fold_1", "fold_2", "fold_3", "fold_4"]:
        train_raw.extend([str(q) for q in folds[f]])
    train_non_cal = [q for q in train_raw if q not in cal_qids]

    from .common import normalize_text
    cal_norm_texts = {normalize_text(questions[q]) for q in cal_qids if q in questions}
    held_raw_qids = [str(q) for q in folds["fold_0"] if str(q) not in cal_qids]
    held_norm_texts = {normalize_text(questions[q]) for q in held_raw_qids}

    valid_train_qids = []
    for q in train_non_cal:
        q_norm = normalize_text(questions[q])
        if q_norm not in cal_norm_texts and q_norm not in held_norm_texts:
            cand_set = set(str(d) for d in pools[q])
            gold_set = set(str(d) for d in v2_golds.get(q, set()))
            if len(cand_set & gold_set) > 0:
                valid_train_qids.append(q)

    print(f"[TEACHER_CACHE] Identified {len(valid_train_qids)} training queries with in-pool gold.", flush=True)

    # 3. Load Shipped Teacher Model
    print("[TEACHER_CACHE] Loading frozen shipped Jina model...", flush=True)
    model, tok = load_jina_base_with_shipped_weights()
    model.eval().to("cuda")

    # 4. Generate pairs by processing queries in chunks
    if TEACHER_PAIRS_FILE.exists():
        TEACHER_PAIRS_FILE.unlink()

    total_pairs = 0
    t0_start = time.time()
    out_f = open(TEACHER_PAIRS_FILE, "w", encoding="utf-8")

    try:
        for chunk_idx in range(0, len(valid_train_qids), query_chunk_size):
            chunk_qids = valid_train_qids[chunk_idx : chunk_idx + query_chunk_size]
            pair_list: List[Tuple[str, str]] = []
            pair_meta: List[Tuple[str, str, int]] = []  # (qid, doc_id, section_idx)
            query_doc_secs: Dict[Tuple[str, str], List[str]] = {}

            # Prepare section pairs for all candidate documents in chunk
            for qid in chunk_qids:
                q_text = questions[qid]
                cand_docs = [str(d) for d in pools[qid]]
                for doc_id in cand_docs:
                    raw_text = contexts.get(doc_id, "")
                    secs = parse_document_into_sections(doc_id, raw_text, max_chunk_words=220, overlap_words=60)
                    chosen_secs = preselect_legal_sections(q_text, secs, count=2)
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

                pos_docs = [d for d in cand_docs if d in gold_docs]
                non_gold_docs = [d for d in cand_docs if d not in gold_docs]

                if not pos_docs or not non_gold_docs:
                    continue

                # Sort non-gold docs by teacher score descending
                non_gold_docs.sort(key=lambda d: doc_logits.get((qid, d), -1e9), reverse=True)
                top3_hard_negs = non_gold_docs[:3]

                # Weight normalization: total weight per query = 1.0
                pair_weight = 1.0 / (len(pos_docs) * len(top3_hard_negs))

                for p_id in pos_docs:
                    p_logit = doc_logits.get((qid, p_id), 0.0)
                    p_secs = query_doc_secs.get((qid, p_id), [])
                    for n_id in top3_hard_negs:
                        n_logit = doc_logits.get((qid, n_id), 0.0)
                        n_secs = query_doc_secs.get((qid, n_id), [])

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

            if (chunk_idx + query_chunk_size) % 500 == 0 or (chunk_idx + query_chunk_size) >= len(valid_train_qids):
                elapsed = time.time() - t0_start
                done_cnt = min(chunk_idx + query_chunk_size, len(valid_train_qids))
                print(
                    f"[TEACHER_CACHE] Processed {done_cnt}/{len(valid_train_qids)} queries "
                    f"({total_pairs} pairs, elapsed: {elapsed:.1f}s)",
                    flush=True,
                )
    finally:
        out_f.close()

    # 5. Seal the cache
    pairs_sha256 = sha256_file(TEACHER_PAIRS_FILE)
    seal = {
        "schema_version": "dsc2026.gemini.huy_d1_jina_passage_adaptation_pilot_v2.teacher_cache_seal.v1",
        "experiment_id": "HUY_D1_JINA_PASSAGE_ADAPTATION_PILOT_V2",
        "status": "PASS",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_info["head_commit"],
        "weights_sha256": EXPECTED_SHIPPED_SHA256,
        "section_parser_sha256": sha256_file(SRC_DIR / "legal_section_parser.py"),
        "total_training_queries": len(valid_train_qids),
        "total_training_pairs": total_pairs,
        "pairs_file": str(TEACHER_PAIRS_FILE.relative_to(ROOT)),
        "pairs_sha256": pairs_sha256,
    }

    TEACHER_SEAL_FILE.write_text(json.dumps(seal, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[TEACHER_CACHE] Cache sealed successfully -> {TEACHER_SEAL_FILE} (SHA256: {pairs_sha256[:12]}...)", flush=True)
    return seal


if __name__ == "__main__":
    build_teacher_cache()
