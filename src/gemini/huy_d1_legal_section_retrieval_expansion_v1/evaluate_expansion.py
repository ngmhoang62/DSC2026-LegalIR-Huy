"""Evaluation of candidate expansion on CAL600 and Strict-V2 benchmarks.

Anti-contamination rule:
Additions file SHA256 is strictly verified against its SEAL before gold labels are opened.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Set

from .common import (
    EXPECTED_CAL_BASE_CANDIDATE_RECALL,
    EXPECTED_V2_BASE_CANDIDATE_RECALL,
    RES_DIR,
    ROOT,
    SRC_DIR,
    compute_candidate_fingerprint,
    compute_corpus_fingerprint,
    compute_query_fingerprint,
    get_git_status,
    load_cal_corpus,
    load_cal_generation_inputs,
    load_cal_gold_labels,
    load_v2_generation_inputs,
    load_v2_gold_labels,
    sha256_file,
)


def verify_seal(
    dataset_name: str,
    additions_path: Path,
    seal_path: Path,
    db_path: Path,
    prov_path: Path,
    expected_query_fp: str,
    expected_cand_fp: str,
    expected_corpus_fp: str,
) -> Dict[str, Any]:
    """Strictly verify the full provenance contract before gold labels are opened.
    Raises RuntimeError('BLOCKED_ADDITIONS_PROVENANCE: ...') if any check fails.
    """
    if not additions_path.exists():
        raise RuntimeError(f"BLOCKED_ADDITIONS_PROVENANCE: Additions file missing: {additions_path}")
    if not seal_path.exists():
        raise RuntimeError(f"BLOCKED_ADDITIONS_PROVENANCE: Seal file missing: {seal_path}")
    if not db_path.exists():
        raise RuntimeError(f"BLOCKED_ADDITIONS_PROVENANCE: Index DB file missing: {db_path}")
    if not prov_path.exists():
        raise RuntimeError(f"BLOCKED_ADDITIONS_PROVENANCE: Index provenance manifest missing: {prov_path}")

    try:
        seal = json.loads(seal_path.read_text(encoding="utf-8"))
    except Exception as e:
        raise RuntimeError(f"BLOCKED_ADDITIONS_PROVENANCE: Corrupted seal JSON: {e}")

    if seal.get("dataset") != dataset_name:
        raise RuntimeError(
            f"BLOCKED_ADDITIONS_PROVENANCE: Seal dataset mismatch! Expected {dataset_name}, got {seal.get('dataset')}"
        )

    actual_add_sha = sha256_file(additions_path)
    if seal.get("generated_additions_artifact_sha256") != actual_add_sha:
        raise RuntimeError(
            f"BLOCKED_ADDITIONS_PROVENANCE: Additions file {additions_path.name} SHA256 mismatch!\n"
            f"Expected from seal: {seal.get('generated_additions_artifact_sha256')}\n"
            f"Actual file sha256: {actual_add_sha}"
        )

    git_info = get_git_status()
    if seal.get("git_commit_sha") != git_info["head_commit"]:
        raise RuntimeError(
            f"BLOCKED_ADDITIONS_PROVENANCE: Git commit SHA mismatch!\n"
            f"Expected from seal: {seal.get('git_commit_sha')}\n"
            f"Current HEAD commit: {git_info['head_commit']}"
        )

    src_hashes = {
        "candidate_generator": (SRC_DIR / "candidate_generator.py", seal.get("candidate_generator_source_sha256")),
        "section_retriever": (SRC_DIR / "section_retriever.py", seal.get("section_retriever_source_sha256")),
        "legal_section_parser": (SRC_DIR / "legal_section_parser.py", seal.get("legal_section_parser_source_sha256")),
    }
    for name, (src_file, expected_h) in src_hashes.items():
        actual_h = sha256_file(src_file)
        if actual_h != expected_h:
            raise RuntimeError(
                f"BLOCKED_ADDITIONS_PROVENANCE: Source hash mismatch for {name} ({src_file.name})!\n"
                f"Expected from seal: {expected_h}\n"
                f"Actual file sha256: {actual_h}"
            )

    if seal.get("query_fingerprint") != expected_query_fp:
        raise RuntimeError(
            f"BLOCKED_ADDITIONS_PROVENANCE: Query fingerprint mismatch on {dataset_name}!\n"
            f"Expected from seal: {seal.get('query_fingerprint')}\n"
            f"Actual input fp:    {expected_query_fp}"
        )
    if seal.get("baseline_candidate_pool_fingerprint") != expected_cand_fp:
        raise RuntimeError(
            f"BLOCKED_ADDITIONS_PROVENANCE: Candidate pool fingerprint mismatch on {dataset_name}!\n"
            f"Expected from seal: {seal.get('baseline_candidate_pool_fingerprint')}\n"
            f"Actual input fp:    {expected_cand_fp}"
        )
    if seal.get("corpus_fingerprint") != expected_corpus_fp:
        raise RuntimeError(
            f"BLOCKED_ADDITIONS_PROVENANCE: Corpus fingerprint mismatch on {dataset_name}!\n"
            f"Expected from seal: {seal.get('corpus_fingerprint')}\n"
            f"Actual corpus fp:   {expected_corpus_fp}"
        )

    actual_db_sha = sha256_file(db_path)
    if seal.get("section_index_sha256") != actual_db_sha:
        raise RuntimeError(
            f"BLOCKED_ADDITIONS_PROVENANCE: Index DB SHA256 mismatch on {dataset_name}!\n"
            f"Expected from seal: {seal.get('section_index_sha256')}\n"
            f"Actual file sha256: {actual_db_sha}"
        )
    actual_prov_sha = sha256_file(prov_path)
    if seal.get("section_index_provenance_sha256") != actual_prov_sha:
        raise RuntimeError(
            f"BLOCKED_ADDITIONS_PROVENANCE: Index provenance SHA256 mismatch on {dataset_name}!\n"
            f"Expected from seal: {seal.get('section_index_provenance_sha256')}\n"
            f"Actual file sha256: {actual_prov_sha}"
        )

    if seal.get("cap") != 8 or seal.get("section_hit_depth") != 128:
        raise RuntimeError(
            f"BLOCKED_ADDITIONS_PROVENANCE: Budget mismatch on {dataset_name}: "
            f"cap={seal.get('cap')}, depth={seal.get('section_hit_depth')}"
        )

    print(f"[EVAL] Pre-gold full provenance contract VERIFIED for {dataset_name} ({additions_path.name})", flush=True)
    return seal


def evaluate_cal_expansion() -> Dict[str, Any]:
    print("[EVAL] Evaluating CAL600 expansion...", flush=True)
    additions_file = RES_DIR / "CAL_SECTION_RETRIEVAL_ADDITIONS.jsonl"
    seal_file = RES_DIR / "CAL_SECTION_RETRIEVAL_ADDITIONS_SEAL.json"
    db_file = RES_DIR / "indexes/cal_sections.db"
    prov_file = RES_DIR / "CAL_SECTION_INDEX_PROVENANCE.json"

    # 1. Load generation inputs and corpus STRICTLY WITHOUT GOLD LABELS
    corpus = load_cal_corpus()
    query_texts, blocks, all_ids, extended = load_cal_generation_inputs()
    cand_fp = compute_candidate_fingerprint(all_ids, extended)
    query_fp = compute_query_fingerprint(all_ids, query_texts)
    corpus_fp = compute_corpus_fingerprint(corpus)

    # 2. Hard-gate verification of seal before gold labels are opened
    verify_seal(
        dataset_name="CAL600",
        additions_path=additions_file,
        seal_path=seal_file,
        db_path=db_file,
        prov_path=prov_file,
        expected_query_fp=query_fp,
        expected_cand_fp=cand_fp,
        expected_corpus_fp=corpus_fp,
    )

    # 3. Gold labels opened ONLY AFTER full provenance contract verified
    gold_labels = load_cal_gold_labels(all_ids)

    # Load additions
    additions_by_qid: Dict[str, List[Dict[str, Any]]] = {}
    with open(additions_file, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                additions_by_qid[row["qid"]] = row.get("additions", [])

    base_recalls = []
    exp_recalls = []
    recovered_cases = []
    lost_cases = []

    single_gold_base = []
    single_gold_exp = []
    multi_gold_base = []
    multi_gold_exp = []

    addition_counts = []
    base_pool_sizes = []
    exp_pool_sizes = []
    total_recovered_gold_docs = 0

    for qid in all_ids:
        g = gold_labels[qid]
        base_cands = set(extended[qid])
        adds = additions_by_qid.get(qid, [])
        add_cands = set(c["doc_id"] for c in adds)
        exp_cands = base_cands | add_cands

        b_rec = len(g & base_cands) / len(g) if g else 1.0
        e_rec = len(g & exp_cands) / len(g) if g else 1.0

        base_recalls.append(b_rec)
        exp_recalls.append(e_rec)

        addition_counts.append(len(add_cands))
        base_pool_sizes.append(len(base_cands))
        exp_pool_sizes.append(len(exp_cands))

        if len(g) == 1:
            single_gold_base.append(b_rec)
            single_gold_exp.append(e_rec)
        else:
            multi_gold_base.append(b_rec)
            multi_gold_exp.append(e_rec)

        delta = e_rec - b_rec
        if delta > 1e-9:
            # Query recovered gold doc(s)
            rec_gold = (g & add_cands) - base_cands
            total_recovered_gold_docs += len(rec_gold)
            rec_info = []
            for d in rec_gold:
                cand_meta = next((c for c in adds if c["doc_id"] == d), None)
                rec_info.append(cand_meta or {"doc_id": d})

            recovered_cases.append({
                "qid": qid,
                "query_text": query_texts[qid],
                "gold_labels": sorted(list(g)),
                "baseline_gold_found": sorted(list(g & base_cands)),
                "recovered_gold_docs": sorted(list(rec_gold)),
                "recovered_doc_details": rec_info,
                "baseline_recall": b_rec,
                "expanded_recall": e_rec,
                "delta": delta,
                "baseline_pool_size": len(base_cands),
                "additions_count": len(adds),
            })
        elif delta < -1e-9:
            lost_cases.append({"qid": qid, "baseline_recall": b_rec, "expanded_recall": e_rec})

    macro_base = sum(base_recalls) / len(base_recalls)
    macro_exp = sum(exp_recalls) / len(exp_recalls)
    delta_cal = macro_exp - macro_base

    # Block breakdowns
    block_results = {}
    for block_name, qids in blocks.items():
        b_b = [
            len(gold_labels[q] & set(extended[q])) / len(gold_labels[q]) if gold_labels[q] else 1.0
            for q in qids
        ]
        b_e = [
            len(gold_labels[q] & (set(extended[q]) | set(c["doc_id"] for c in additions_by_qid.get(q, []))))
            / len(gold_labels[q]) if gold_labels[q] else 1.0
            for q in qids
        ]
        m_b = sum(b_b) / len(b_b) if b_b else 0.0
        m_e = sum(b_e) / len(b_e) if b_e else 0.0
        block_results[block_name] = {
            "query_count": len(qids),
            "baseline_recall": m_b,
            "expanded_recall": m_e,
            "delta": m_e - m_b,
        }

    git_info = get_git_status()

    cal_results = {
        "benchmark": "CAL600",
        "baseline_candidate_recall": macro_base,
        "expanded_candidate_recall": macro_exp,
        "delta_recall": delta_cal,
        "query_count": len(all_ids),
        "recovered_queries_count": len(recovered_cases),
        "lost_queries_count": len(lost_cases),
        "total_recovered_gold_docs": total_recovered_gold_docs,
        "single_gold": {
            "query_count": len(single_gold_base),
            "baseline_recall": sum(single_gold_base) / len(single_gold_base),
            "expanded_recall": sum(single_gold_exp) / len(single_gold_exp),
            "delta": (sum(single_gold_exp) - sum(single_gold_base)) / len(single_gold_base),
        },
        "multi_gold": {
            "query_count": len(multi_gold_base),
            "baseline_recall": sum(multi_gold_base) / len(multi_gold_base),
            "expanded_recall": sum(multi_gold_exp) / len(multi_gold_exp),
            "delta": (sum(multi_gold_exp) - sum(multi_gold_base)) / len(multi_gold_base),
        },
        "block_breakdown": block_results,
        "git_commit": git_info["head_commit"],
    }

    out_res = RES_DIR / "CAL_SECTION_RETRIEVAL_RESULTS.json"
    out_res.write_text(json.dumps(cal_results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[EVAL] CAL results saved -> {out_res}", flush=True)

    # Forensics
    out_forensic = RES_DIR / "CAL_RECOVERED_CASES_FORENSIC.json"
    out_forensic.write_text(json.dumps(recovered_cases, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[EVAL] CAL forensic saved -> {out_forensic} ({len(recovered_cases)} recovered cases)", flush=True)

    # Noise audit
    total_adds = sum(addition_counts)
    noise_audit = {
        "benchmark": "CAL600",
        "total_queries": len(all_ids),
        "total_additions": total_adds,
        "mean_additions_per_query": statistics.mean(addition_counts),
        "median_additions_per_query": statistics.median(addition_counts),
        "min_additions_per_query": min(addition_counts),
        "max_additions_per_query": max(addition_counts),
        "baseline_pool_mean": statistics.mean(base_pool_sizes),
        "expanded_pool_mean": statistics.mean(exp_pool_sizes),
        "pool_expansion_percent": (statistics.mean(exp_pool_sizes) - statistics.mean(base_pool_sizes)) / statistics.mean(base_pool_sizes) * 100,
        "total_gold_recovered": total_recovered_gold_docs,
        "addition_precision": total_recovered_gold_docs / max(total_adds, 1),
    }
    out_noise = RES_DIR / "CAL_EXPANSION_NOISE_AUDIT.json"
    out_noise.write_text(json.dumps(noise_audit, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[EVAL] CAL noise audit saved -> {out_noise}", flush=True)

    return cal_results


def evaluate_v2_expansion() -> Dict[str, Any]:
    print("[EVAL] Evaluating Strict-V2 expansion...", flush=True)
    additions_file = RES_DIR / "V2_SECTION_RETRIEVAL_ADDITIONS.jsonl"
    seal_file = RES_DIR / "V2_SECTION_RETRIEVAL_ADDITIONS_SEAL.json"
    db_file = RES_DIR / "indexes/v2_sections.db"
    prov_file = RES_DIR / "V2_SECTION_INDEX_PROVENANCE.json"

    # 1. Load V2 generation inputs and corpus STRICTLY WITHOUT GOLD LABELS
    corpus, queries, v2_qids, candidate_pools = load_v2_generation_inputs()
    v2_cand_fp = compute_candidate_fingerprint(v2_qids, candidate_pools)
    v2_query_fp = compute_query_fingerprint(v2_qids, queries)
    v2_corpus_fp = compute_corpus_fingerprint(corpus)

    # 2. Hard-gate verification of seal before gold labels are opened
    verify_seal(
        dataset_name="Strict-V2",
        additions_path=additions_file,
        seal_path=seal_file,
        db_path=db_file,
        prov_path=prov_file,
        expected_query_fp=v2_query_fp,
        expected_cand_fp=v2_cand_fp,
        expected_corpus_fp=v2_corpus_fp,
    )

    # 3. Gold labels opened ONLY AFTER full provenance contract verified
    gold_labels = load_v2_gold_labels(v2_qids)

    additions_by_qid: Dict[str, List[Dict[str, Any]]] = {}
    with open(additions_file, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                additions_by_qid[row["qid"]] = row.get("additions", [])

    base_recalls = []
    exp_recalls = []
    recovered_cases = []
    lost_cases = []
    addition_counts = []
    total_recovered_gold = 0

    for qid in v2_qids:
        g = gold_labels.get(qid, set())
        b_cands = set(candidate_pools[qid])
        adds = additions_by_qid.get(qid, [])
        a_cands = set(c["doc_id"] for c in adds)
        e_cands = b_cands | a_cands

        b_rec = len(g & b_cands) / len(g) if g else 1.0
        e_rec = len(g & e_cands) / len(g) if g else 1.0

        base_recalls.append(b_rec)
        exp_recalls.append(e_rec)
        addition_counts.append(len(a_cands))

        delta = e_rec - b_rec
        if delta > 1e-9:
            rec_gold = (g & a_cands) - b_cands
            total_recovered_gold += len(rec_gold)
            rec_info = []
            for d in sorted(rec_gold):
                cand_meta = next((c for c in adds if c["doc_id"] == d), None)
                rec_info.append(cand_meta or {"doc_id": d})

            recovered_cases.append({
                "qid": qid,
                "query_text": queries[qid],
                "gold_labels": sorted(list(g)),
                "baseline_gold_found": sorted(list(g & b_cands)),
                "recovered_gold_docs": sorted(list(rec_gold)),
                "recovered_doc_details": rec_info,
                "baseline_recall": b_rec,
                "expanded_recall": e_rec,
                "delta": delta,
                "baseline_pool_size": len(b_cands),
                "additions_count": len(adds),
            })
        elif delta < -1e-9:
            lost_cases.append({"qid": qid, "baseline_recall": b_rec, "expanded_recall": e_rec})

    macro_base = sum(base_recalls) / len(base_recalls)
    macro_exp = sum(exp_recalls) / len(exp_recalls)
    delta_v2 = macro_exp - macro_base

    git_info = get_git_status()
    total_adds = sum(addition_counts)

    v2_results = {
        "benchmark": "Strict-V2",
        "baseline_candidate_recall": macro_base,
        "expanded_candidate_recall": macro_exp,
        "delta_recall": delta_v2,
        "query_count": len(v2_qids),
        "recovered_queries_count": len(recovered_cases),
        "lost_queries_count": len(lost_cases),
        "total_recovered_gold_docs": total_recovered_gold,
        "total_additions": total_adds,
        "mean_additions_per_query": statistics.mean(addition_counts) if addition_counts else 0.0,
        "addition_precision": total_recovered_gold / max(total_adds, 1),
        "git_commit": git_info["head_commit"],
    }

    out_res = RES_DIR / "V2_SECTION_RETRIEVAL_RESULTS.json"
    out_res.write_text(json.dumps(v2_results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[EVAL] V2 results saved -> {out_res}", flush=True)

    # Save V2 forensic artifact
    out_forensic = RES_DIR / "V2_RECOVERED_CASES_FORENSIC.json"
    out_forensic.write_text(json.dumps(recovered_cases, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[EVAL] V2 forensic saved -> {out_forensic} ({len(recovered_cases)} recovered cases)", flush=True)

    return v2_results


def run_all_evaluations() -> Dict[str, Any]:
    cal_res = evaluate_cal_expansion()
    v2_res = evaluate_v2_expansion()
    print(
        f"[EVAL] SUMMARY:\n"
        f"  CAL600: Base={cal_res['baseline_candidate_recall']:.6f}, "
        f"Exp={cal_res['expanded_candidate_recall']:.6f}, "
        f"Delta={cal_res['delta_recall']:+.6f} ({cal_res['recovered_queries_count']} recovered)\n"
        f"  Strict-V2: Base={v2_res['baseline_candidate_recall']:.6f}, "
        f"Exp={v2_res['expanded_candidate_recall']:.6f}, "
        f"Delta={v2_res['delta_recall']:+.6f} ({v2_res['recovered_queries_count']} recovered)"
    )
    return {"cal": cal_res, "v2": v2_res}


if __name__ == "__main__":
    run_all_evaluations()
