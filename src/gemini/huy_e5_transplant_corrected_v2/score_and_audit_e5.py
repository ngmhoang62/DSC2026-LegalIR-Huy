import sys
import os
import json
import time
import pickle
import math
from pathlib import Path
from typing import Dict, List, Set, Any, Tuple
import numpy as np
import torch
from torch.nn import functional as F
from scipy.stats import spearmanr

# Ensure stdout is utf-8
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

REPO_ROOT = Path("d:/Study/DSC2026/sota").resolve()
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from src.research_v2_e5_transfer import e5_transfer_runner as core
from src.research_v2_e5_confirmation.e5_confirmation_runner import ConfirmationData
from tune_corpus_cap32_fusion import build_training_cap
from run_burst_multistage_submission import load_metadata
from run_burst_expanded_fusion_submission import (
    EXPANSION_CONFIG, RERANK_CONFIG, dense_expansion, corpus_dense, rerank, weighted_rrf, raw_union,
    load_public_retrieval, DocumentStore, CORPUS_CAP, CORPUS_DEPTH
)

RESULTS_DIR = REPO_ROOT / "results" / "gemini" / "huy_e5_transplant_corrected_v2"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

BUNDLE_DIR = REPO_ROOT / "cache" / "research_v2_e5_confirmation" / "bundle-v1"
MODEL_DIR = BUNDLE_DIR / "vietlegal-e5"

ADAPTER_PATHS = {
    "fold_0": REPO_ROOT / "results/research_v2_e5_transfer/research_v2_e5_transfer_fold0/training/epoch-2.pt",
    "fold_1": REPO_ROOT / "results/research_v2_e5_confirmation/fold_1/training/epoch-2.pt",
    "fold_2": REPO_ROOT / "results/research_v2_e5_confirmation/fold_2/training/epoch-2.pt",
    "fold_3": REPO_ROOT / "results/research_v2_e5_confirmation/fold_3/training/epoch-2.pt",
    "fold_4": REPO_ROOT / "results/research_v2_e5_confirmation/fold_4/training/epoch-2.pt",
    "full_data": REPO_ROOT / "results/research_v2_open_rl/v2_anchor_submission_candidate/full_data_adapter/epoch-2.pt",
}

def rank_candidates(candidates: List[str], scores: Dict[str, float]) -> List[str]:
    # Sort descending by score, tie-break by doc_id ASC. Missing scores (-inf or nan) at the end.
    def sort_key(d: str):
        val = scores.get(d, float("-inf"))
        if math.isnan(val):
            val = float("-inf")
        return (-val, str(d))
    return sorted(candidates, key=sort_key)

def compute_recall_at_k(order: List[str], gold: Set[str], k: int) -> float:
    hits = len(gold & set(order[:k]))
    return hits / len(gold) if gold else 0.0

def main():
    print("=== SCORING CORRECTED ADAPTED E5 ON HUY CANDIDATE POOLS ===", flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}", flush=True)

    # 1. Load V2 data bundle (chunk bank)
    print("Loading V2 frozen chunk bank...", flush=True)
    v2_data = ConfirmationData(BUNDLE_DIR, held_fold="fold_0")
    bank = core.ParentBank(v2_data.vectors, v2_data.parent, device=device)
    v2_doc_set = set(v2_data.doc_ids)
    print(f"Chunk bank loaded: {len(v2_data.chunk_ids)} chunks, {len(v2_data.doc_ids)} unique parent docs.", flush=True)

    # Load V2 fold mapping
    with open(BUNDLE_DIR / "V2_FOLDS.json", "r", encoding="utf-8") as f:
        folds_obj = json.load(f)
        v2_folds_dict = folds_obj.get("folds", folds_obj)
    v2_query_to_fold = {}
    for fold_name, qids in v2_folds_dict.items():
        for qid in qids:
            v2_query_to_fold[str(qid)] = fold_name

    # 2. CAL600 candidate pool & queries
    print("Loading CAL600 queries and candidate pool...", flush=True)
    queries_cal, blocks_cal, all_ids_cal, extended_cal, local_cal, base_scores_cal = build_training_cap(
        REPO_ROOT, 32, "results/corpus_index/holdout_extended_scores_cap32.pkl", depth=20
    )
    gold_cal = {q: queries_cal[q][1] for q in all_ids_cal}

    # Group CAL queries by V2 fold
    cal_fold_groups: Dict[str, List[str]] = {f"fold_{i}": [] for i in range(5)}
    unmapped_cal = []
    for qid in all_ids_cal:
        f_name = v2_query_to_fold.get(str(qid))
        if f_name in cal_fold_groups:
            cal_fold_groups[f_name].append(str(qid))
        else:
            unmapped_cal.append(str(qid))

    print("CAL600 queries assigned to strict-V2 held folds:")
    for f_name, qids in cal_fold_groups.items():
        print(f"  {f_name}: {len(qids)} queries")
    if unmapped_cal:
        raise RuntimeError(f"Unmapped CAL queries found: {unmapped_cal}")

    # Track missing doc stats
    all_cal_cands = set(d for q in all_ids_cal for d in extended_cal[q])
    missing_cal_docs = sorted(list(all_cal_cands - v2_doc_set))
    missing_cal_slots = sum(1 for q in all_ids_cal for d in extended_cal[q] if d not in v2_doc_set)
    print(f"CAL missing docs in V2 parent bank: {len(missing_cal_docs)} docs ({missing_cal_slots} candidate slots)", flush=True)

    # 3. Score CAL600 using outer fold adapters
    print("\nScoring CAL600 with held-fold adapters...", flush=True)
    torch.cuda.reset_peak_memory_stats()
    cal_start = time.monotonic()

    cal_adapted_scores: Dict[str, Dict[str, float]] = {}
    cal_frozen_scores: Dict[str, Dict[str, float]] = {}
    cal_adapted_orders: Dict[str, List[str]] = {}
    cal_frozen_orders: Dict[str, List[str]] = {}

    for fold_name, qids in cal_fold_groups.items():
        ckpt_path = ADAPTER_PATHS[fold_name]
        print(f"  Loading {fold_name} model from {ckpt_path.name}...", flush=True)
        model = core.QueryEncoder(MODEL_DIR, device=device, checkpoint_path=ckpt_path)
        model.eval()

        for qid in qids:
            q_text = queries_cal[qid][0]
            cands = extended_cal[qid]

            with torch.no_grad():
                ad_vec = model([q_text])[0]
                with model.adapter_disabled():
                    fr_vec = model([q_text])[0]

            # Score candidates
            # For docs present in V2 bank, score with ParentBank.score_pool
            # For missing docs, set to np.nan
            ad_s_map = {}
            fr_s_map = {}
            present_cands = [d for d in cands if d in v2_doc_set]
            
            if present_cands:
                ad_vals = bank.score_pool(ad_vec, present_cands, v2_data)
                fr_vals = bank.score_pool(fr_vec, present_cands, v2_data)
                for d, val in zip(present_cands, ad_vals):
                    ad_s_map[d] = float(val)
                for d, val in zip(present_cands, fr_vals):
                    fr_s_map[d] = float(val)

            for d in cands:
                if d not in v2_doc_set:
                    ad_s_map[d] = float("nan")
                    fr_s_map[d] = float("nan")

            cal_adapted_scores[qid] = ad_s_map
            cal_frozen_scores[qid] = fr_s_map
            cal_adapted_orders[qid] = rank_candidates(cands, ad_s_map)
            cal_frozen_orders[qid] = rank_candidates(cands, fr_s_map)

        del model
        torch.cuda.empty_cache()

    cal_runtime = time.monotonic() - cal_start
    cal_peak_vram = torch.cuda.max_memory_allocated() / (1024 * 1024)
    print(f"CAL600 scoring completed in {cal_runtime:.2f}s, peak VRAM: {cal_peak_vram:.1f} MiB", flush=True)

    # Save CAL scores
    cal_save_path = RESULTS_DIR / "CAL600_CORRECTED_VIETLEGAL_E5_SCORES.pkl"
    with open(cal_save_path, "wb") as f:
        pickle.dump({
            "adapted_scores": cal_adapted_scores,
            "frozen_scores": cal_frozen_scores,
            "adapted_orders": cal_adapted_orders,
            "frozen_orders": cal_frozen_orders,
        }, f, protocol=5)
    print(f"Saved CAL scores to {cal_save_path}", flush=True)

    # 4. Public Test candidate pool scoring using full-data adapter
    print("\nLoading Public test metadata and candidates...", flush=True)
    paths_meta, doc_ids_meta, train_meta, public_meta = load_metadata(REPO_ROOT / "DSC2026-LegalIR-main/v4_run/public_test_dataset")
    public_ids = list(public_meta)
    docs_store = DocumentStore(paths_meta)

    base_pub = pickle.loads((REPO_ROOT / "results/burst_gpu_threeview/cpu_top20.pkl").read_bytes())["rankings"]
    class DummyArgs:
        db = REPO_ROOT / "benchmarks/legalir_full_fts.sqlite"
        workers = 4
        cache_dir = REPO_ROOT / "results/burst_expanded_fusion"

    pub_retrieval = load_public_retrieval(REPO_ROOT, DummyArgs, doc_ids_meta, train_meta, public_meta, public_ids)
    raw_pub = {q: raw_union(pub_retrieval[q], EXPANSION_CONFIG["depth"]) for q in public_ids}
    pub_retrieval.clear()

    expansion_scores_pub = dense_expansion(REPO_ROOT, DummyArgs.cache_dir, public_meta, public_ids, raw_pub, docs_store, "cuda")
    dense_rank_pub = {q: sorted(raw_pub[q], key=lambda d: (-expansion_scores_pub[q][d], d)) for q in public_ids}
    expanded_pub = weighted_rrf([raw_pub, dense_rank_pub], RERANK_CONFIG["expansion_weights"], RERANK_CONFIG["expansion_rrf_k"])
    corpus_rank_pub, corpus_score_pub = corpus_dense(REPO_ROOT, DummyArgs.cache_dir, public_meta, public_ids, "cuda", cap=CORPUS_CAP, depth=CORPUS_DEPTH)

    public_candidates = {
        q: list(dict.fromkeys(
            list(base_pub[q]) + expanded_pub[q][:RERANK_CONFIG["expanded_depth"]] + corpus_rank_pub[q][:CORPUS_DEPTH]
        ))
        for q in public_ids
    }

    all_pub_cands = set(d for q in public_ids for d in public_candidates[q])
    missing_pub_docs = sorted(list(all_pub_cands - v2_doc_set))
    missing_pub_slots = sum(1 for q in public_ids for d in public_candidates[q] if d not in v2_doc_set)
    print(f"Public test queries: {len(public_ids)}, total candidate pairs: {sum(len(v) for v in public_candidates.values())}")
    print(f"Public missing docs in V2 bank: {len(missing_pub_docs)} docs ({missing_pub_slots} candidate slots)", flush=True)

    print("\nScoring Public test candidates with full-data deployment adapter...", flush=True)
    torch.cuda.reset_peak_memory_stats()
    pub_start = time.monotonic()

    full_ckpt_path = ADAPTER_PATHS["full_data"]
    print(f"Loading full-data deployment model from {full_ckpt_path.name}...", flush=True)
    model_pub = core.QueryEncoder(MODEL_DIR, device=device, checkpoint_path=full_ckpt_path)
    model_pub.eval()

    public_adapted_scores: Dict[str, Dict[str, float]] = {}
    public_frozen_scores: Dict[str, Dict[str, float]] = {}
    public_adapted_orders: Dict[str, List[str]] = {}
    public_frozen_orders: Dict[str, List[str]] = {}

    for i, qid in enumerate(public_ids, 1):
        q_text = public_meta[qid]
        cands = public_candidates[qid]

        with torch.no_grad():
            ad_vec = model_pub([q_text])[0]
            with model_pub.adapter_disabled():
                fr_vec = model_pub([q_text])[0]

        ad_s_map = {}
        fr_s_map = {}
        present_cands = [d for d in cands if d in v2_doc_set]

        if present_cands:
            ad_vals = bank.score_pool(ad_vec, present_cands, v2_data)
            fr_vals = bank.score_pool(fr_vec, present_cands, v2_data)
            for d, val in zip(present_cands, ad_vals):
                ad_s_map[d] = float(val)
            for d, val in zip(present_cands, fr_vals):
                fr_s_map[d] = float(val)

        for d in cands:
            if d not in v2_doc_set:
                ad_s_map[d] = float("nan")
                fr_s_map[d] = float("nan")

        public_adapted_scores[qid] = ad_s_map
        public_frozen_scores[qid] = fr_s_map
        public_adapted_orders[qid] = rank_candidates(cands, ad_s_map)
        public_frozen_orders[qid] = rank_candidates(cands, fr_s_map)

        if i % 200 == 0:
            print(f"  Scored {i}/{len(public_ids)} queries", flush=True)

    del model_pub
    torch.cuda.empty_cache()

    pub_runtime = time.monotonic() - pub_start
    pub_peak_vram = torch.cuda.max_memory_allocated() / (1024 * 1024)
    print(f"Public scoring completed in {pub_runtime:.2f}s, peak VRAM: {pub_peak_vram:.1f} MiB", flush=True)

    # Save Public scores
    pub_save_path = RESULTS_DIR / "PUBLIC_CORRECTED_VIETLEGAL_E5_SCORES.pkl"
    with open(pub_save_path, "wb") as f:
        pickle.dump({
            "adapted_scores": public_adapted_scores,
            "frozen_scores": public_frozen_scores,
            "adapted_orders": public_adapted_orders,
            "frozen_orders": public_frozen_orders,
            "public_candidates": public_candidates,
        }, f, protocol=5)
    print(f"Saved Public scores to {pub_save_path}", flush=True)

    # 5. Compute Audit Statistics on CAL (Frozen vs Adapted)
    print("\nComputing scoring audit statistics on CAL600...", flush=True)
    abs_score_diffs = []
    spearman_corrs = []
    top5_diff_queries = 0
    full_order_diff_queries = 0

    for qid in all_ids_cal:
        cands = extended_cal[qid]
        ad_map = cal_adapted_scores[qid]
        fr_map = cal_frozen_scores[qid]

        common_docs = [d for d in cands if not math.isnan(ad_map.get(d, float("nan")))]
        if common_docs:
            for d in common_docs:
                abs_score_diffs.append(abs(ad_map[d] - fr_map[d]))
            ad_s = [ad_map[d] for d in common_docs]
            fr_s = [fr_map[d] for d in common_docs]
            if len(common_docs) > 1 and np.std(ad_s) > 1e-9 and np.std(fr_s) > 1e-9:
                corr, _ = spearmanr(ad_s, fr_s)
                if not math.isnan(corr):
                    spearman_corrs.append(corr)

        ad_top5 = set(cal_adapted_orders[qid][:5])
        fr_top5 = set(cal_frozen_orders[qid][:5])
        if ad_top5 != fr_top5:
            top5_diff_queries += 1
        if cal_adapted_orders[qid] != cal_frozen_orders[qid]:
            full_order_diff_queries += 1

    scoring_audit = {
        "schema_version": "dsc2026.gemini.huy_e5_transplant_corrected_v2.corrected_e5_scoring_audit.v1",
        "cal600": {
            "query_count": len(all_ids_cal),
            "candidate_rows_scored": sum(len(v) for v in extended_cal.values()),
            "runtime_seconds": cal_runtime,
            "peak_vram_mib": cal_peak_vram,
            "missing_docs_in_v2_bank": {
                "doc_count": len(missing_cal_docs),
                "slot_count": missing_cal_slots,
                "missing_doc_ids": missing_cal_docs,
                "handling": "np.nan score, placed at end of order tie-broken by doc_id ASC"
            },
            "frozen_vs_adapted_comparison": {
                "mean_absolute_score_difference": float(np.mean(abs_score_diffs)),
                "max_absolute_score_difference": float(np.max(abs_score_diffs)),
                "mean_spearman_rank_correlation": float(np.mean(spearman_corrs)),
                "queries_with_top5_difference": top5_diff_queries,
                "queries_with_top5_difference_pct": top5_diff_queries / len(all_ids_cal) * 100,
                "queries_with_full_order_difference": full_order_diff_queries,
                "queries_with_full_order_difference_pct": full_order_diff_queries / len(all_ids_cal) * 100,
            }
        },
        "public": {
            "query_count": len(public_ids),
            "candidate_rows_scored": sum(len(v) for v in public_candidates.values()),
            "runtime_seconds": pub_runtime,
            "peak_vram_mib": pub_peak_vram,
            "missing_docs_in_v2_bank": {
                "doc_count": len(missing_pub_docs),
                "slot_count": missing_pub_slots,
            }
        }
    }

    scoring_audit_path = RESULTS_DIR / "CORRECTED_E5_SCORING_AUDIT.json"
    with open(scoring_audit_path, "w", encoding="utf-8") as f:
        json.dump(scoring_audit, f, indent=2, ensure_ascii=False)
    print(f"Wrote scoring audit to {scoring_audit_path}", flush=True)

    # 6. Standalone Diagnostic (Section 8)
    print("\n--- STANDALONE DIAGNOSTIC ON CAL600 ---", flush=True)
    # A. Historical generic e5
    hist_e5_scores = base_scores_cal["e5"]
    hist_e5_orders = {q: rank_candidates(extended_cal[q], hist_e5_scores[q]) for q in all_ids_cal}

    models_eval = {
        "historical_generic_e5": hist_e5_orders,
        "frozen_vietlegal_e5": cal_frozen_orders,
        "corrected_adapted_vietlegal_e5": cal_adapted_orders,
    }

    eval_stats = {}
    ks = [1, 5, 8, 10]

    for m_name, orders in models_eval.items():
        m_res = {}
        for k in ks:
            rec = float(np.mean([compute_recall_at_k(orders[q], gold_cal[q], k) for q in all_ids_cal]))
            m_res[f"recall_at_{k}"] = rec

        # Single vs multi gold
        single_rec = float(np.mean([compute_recall_at_k(orders[q], gold_cal[q], 5) for q in all_ids_cal if len(gold_cal[q]) == 1]))
        multi_rec = float(np.mean([compute_recall_at_k(orders[q], gold_cal[q], 5) for q in all_ids_cal if len(gold_cal[q]) > 1]))
        m_res["single_gold_recall_at_5"] = single_rec
        m_res["multi_gold_recall_at_5"] = multi_rec
        eval_stats[m_name] = m_res

    # Head-to-head comparisons at Top-5
    # Adapted vs Frozen
    ad_vs_fr_wins = sum(1 for q in all_ids_cal if compute_recall_at_k(cal_adapted_orders[q], gold_cal[q], 5) > compute_recall_at_k(cal_frozen_orders[q], gold_cal[q], 5))
    ad_vs_fr_losses = sum(1 for q in all_ids_cal if compute_recall_at_k(cal_adapted_orders[q], gold_cal[q], 5) < compute_recall_at_k(cal_frozen_orders[q], gold_cal[q], 5))
    ad_vs_fr_ties = sum(1 for q in all_ids_cal if compute_recall_at_k(cal_adapted_orders[q], gold_cal[q], 5) == compute_recall_at_k(cal_frozen_orders[q], gold_cal[q], 5))

    # Adapted vs Historical generic E5
    ad_vs_hist_wins = sum(1 for q in all_ids_cal if compute_recall_at_k(cal_adapted_orders[q], gold_cal[q], 5) > compute_recall_at_k(hist_e5_orders[q], gold_cal[q], 5))
    ad_vs_hist_losses = sum(1 for q in all_ids_cal if compute_recall_at_k(cal_adapted_orders[q], gold_cal[q], 5) < compute_recall_at_k(hist_e5_orders[q], gold_cal[q], 5))
    ad_vs_hist_ties = sum(1 for q in all_ids_cal if compute_recall_at_k(cal_adapted_orders[q], gold_cal[q], 5) == compute_recall_at_k(hist_e5_orders[q], gold_cal[q], 5))

    print(f"Historical Generic E5 Standalone R@5: {eval_stats['historical_generic_e5']['recall_at_5']:.4f}")
    print(f"Frozen VietLegal-E5 Standalone R@5:   {eval_stats['frozen_vietlegal_e5']['recall_at_5']:.4f}")
    print(f"Adapted VietLegal-E5 Standalone R@5:  {eval_stats['corrected_adapted_vietlegal_e5']['recall_at_5']:.4f}")
    print(f"Adapted vs Frozen Wins/Losses/Ties: {ad_vs_fr_wins} / {ad_vs_fr_losses} / {ad_vs_fr_ties}")
    print(f"Adapted vs Historical E5 Wins/Losses/Ties: {ad_vs_hist_wins} / {ad_vs_hist_losses} / {ad_vs_hist_ties}")

    standalone_report = {
        "schema_version": "dsc2026.gemini.huy_e5_transplant_corrected_v2.e5_standalone_report.v1",
        "standalone_metrics": eval_stats,
        "head_to_head_top5": {
            "adapted_vs_frozen": {
                "wins": ad_vs_fr_wins,
                "losses": ad_vs_fr_losses,
                "ties": ad_vs_fr_ties,
                "net_gain": ad_vs_fr_wins - ad_vs_fr_losses,
                "delta_recall_at_5": eval_stats["corrected_adapted_vietlegal_e5"]["recall_at_5"] - eval_stats["frozen_vietlegal_e5"]["recall_at_5"]
            },
            "adapted_vs_historical_generic_e5": {
                "wins": ad_vs_hist_wins,
                "losses": ad_vs_hist_losses,
                "ties": ad_vs_hist_ties,
                "net_gain": ad_vs_hist_wins - ad_vs_hist_losses,
                "delta_recall_at_5": eval_stats["corrected_adapted_vietlegal_e5"]["recall_at_5"] - eval_stats["historical_generic_e5"]["recall_at_5"]
            }
        }
    }

    standalone_path = RESULTS_DIR / "E5_STANDALONE_REPORT.json"
    with open(standalone_path, "w", encoding="utf-8") as f:
        json.dump(standalone_report, f, indent=2, ensure_ascii=False)
    print(f"Wrote standalone diagnostic report to {standalone_path}", flush=True)

if __name__ == "__main__":
    main()
