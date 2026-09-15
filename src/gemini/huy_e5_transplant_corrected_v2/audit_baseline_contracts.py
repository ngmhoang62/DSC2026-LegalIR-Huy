import sys
import os
import json
import pickle
import hashlib
from pathlib import Path
from typing import Dict, List, Set, Any, Tuple
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Ensure stdout is utf-8
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

REPO_ROOT = Path("d:/Study/DSC2026/sota").resolve()
sys.path.insert(0, str(REPO_ROOT))

from run_burst_expanded_fusion_submission import DocumentStore, CORPUS_CAP, CORPUS_DEPTH
from tune_citation_graph import build_citation_table, citation_features
from tune_corpus_cap32_fusion import build_training_cap
from tune_doctype_features import build_type_table, type_features
from tune_expanded_fusion_selection import ltr_features
from run_burst_multistage_submission import load_metadata

RESULTS_DIR = REPO_ROOT / "results" / "gemini" / "huy_e5_transplant_corrected_v2"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CONTROL_JSON_PATH = REPO_ROOT / "results" / "burst_userft_maxrecall" / "submission.json"

HISTORICAL_VIEWS = ["base", "expanded", "jina", "dense", "corpus"]
DEPLOYMENT_VIEWS = ["base", "expanded", "jina", "dense", "corpus", "vnlegal_lal"]

EXTRA_CV_PATHS = {
    "aiteamvn_ft": "results/from_drive/aiteamvn_ft_cv.pkl",
    "jina_ft": "results/from_drive/jina_ft_cv.pkl",
    "title_embed": "results/burst_fresh_block/title_embed_scores.pkl",
}

EXTRA_PUB_PATHS = {
    "aiteamvn_ft": "results/from_drive/aiteamvn_ft_public.pkl",
    "jina_ft": "results/from_drive/jina_ft_public.pkl",
    "title_embed": "results/burst_fresh_block/title_embed_public.pkl",
}

def load_pkl(rel_path: str):
    p = REPO_ROOT / rel_path
    obj = pickle.loads(p.read_bytes())
    if isinstance(obj, dict) and isinstance(obj.get("scores"), dict):
        return obj["scores"]
    return obj

def main():
    print("=== BASELINE CONTRACT AUDIT & MANDATORY H0 REPRODUCTION ===", flush=True)

    # 1. Load document store & CAL 600 structures
    print("Loading DocumentStore...", flush=True)
    docs = DocumentStore(sorted(
        (REPO_ROOT / "DSC2026-LegalIR-main/v4_run/public_test_dataset/selected-contexts").glob("context_*.json")
    ))

    print("Building training cap 32...", flush=True)
    queries, blocks, all_ids, extended, local_views, base_scores = build_training_cap(
        REPO_ROOT, 32, "results/corpus_index/holdout_extended_scores_cap32.pkl", depth=20
    )
    gold = {q: queries[q][1] for q in all_ids}
    print(f"CAL600 queries: {len(all_ids)}, blocks: {[len(v) for v in blocks.values()]}, pool: {sum(len(extended[q]) for q in all_ids)/len(all_ids):.1f} docs/q", flush=True)

    def load_aligned(rel_path: str, floor=None):
        obj = load_pkl(rel_path)
        fl = floor if floor is not None else min(v for q in obj for v in obj[q].values())
        return {q: {d: obj.get(q, {}).get(d, fl) for d in extended[q]} for q in all_ids}

    # Prepare CV score channels
    vnlegal_cv = load_pkl("results/embedding_finetune/vnlegal_lal_cv_scores.pkl")
    crossenc_cv = load_aligned("results/crossenc_fullpool/cv_scores.pkl", -11.5)
    extra_cv = {name: load_aligned(rel) for name, rel in EXTRA_CV_PATHS.items()}

    full_channels_cv = {
        **base_scores,
        "vnlegal_lal": vnlegal_cv,
        "crossenc": crossenc_cv,
        **extra_cv,
    }

    print(f"Total CAL score channels: {len(full_channels_cv)}: {sorted(full_channels_cv.keys())}", flush=True)

    # Compute metadata features
    print("Building doctype and citation features...", flush=True)
    type_table = build_type_table(REPO_ROOT, docs, all_ids, extended)
    type_rows = type_features(extended, type_table, queries, all_ids)
    own, cited = build_citation_table(docs, all_ids, extended)
    cite_rows = citation_features(extended, own, cited, all_ids)

    # 2. Evaluate Protocol A: HISTORICAL_CAL
    print("\n--- Running HISTORICAL_CAL (5 rank views: evaluate_cv.py semantics) ---", flush=True)
    hist_rows, hist_groups = ltr_features(local_views, HISTORICAL_VIEWS, extended, all_ids, full_channels_cv)
    for q in hist_rows:
        hist_rows[q] = np.concatenate([hist_rows[q], type_rows[q], cite_rows[q]], axis=1)

    hist_preds = {}
    for held in blocks:
        train = sum((blocks[n] for n in blocks if n != held), [])
        X_train = np.vstack([hist_rows[q] for q in train])
        y_train = np.concatenate([[d in gold[q] for d in hist_groups[q]] for q in train]).astype(np.int8)
        scaler = StandardScaler().fit(X_train)
        model = LogisticRegression(C=0.15, class_weight="balanced", solver="liblinear", max_iter=3000, random_state=2026)
        model.fit(scaler.transform(X_train), y_train)

        for q in blocks[held]:
            score = model.decision_function(scaler.transform(hist_rows[q]))
            hist_preds[q] = [hist_groups[q][i] for i in np.argsort(-score)[:5]]

    hist_pooled_r5 = float(np.mean([len(gold[q] & set(hist_preds[q])) / len(gold[q]) for q in all_ids]))
    hist_pooled_p5 = float(np.mean([len(gold[q] & set(hist_preds[q])) / 5.0 for q in all_ids]))
    hist_blocks = {
        b: float(np.mean([len(gold[q] & set(hist_preds[q])) / len(gold[q]) for q in blocks[b]]))
        for b in blocks
    }
    hist_single_r5 = float(np.mean([len(gold[q] & set(hist_preds[q])) / len(gold[q]) for q in all_ids if len(gold[q]) == 1]))
    hist_multi_r5 = float(np.mean([len(gold[q] & set(hist_preds[q])) / len(gold[q]) for q in all_ids if len(gold[q]) > 1]))

    print(f"HISTORICAL_CAL Pooled R@5: {hist_pooled_r5:.6f}")
    print(f"  Blocks: a={hist_blocks['a']:.4f}, b={hist_blocks['b']:.4f}, c={hist_blocks['c']:.4f}, d={hist_blocks['d']:.4f}")
    print(f"  Feature dim: {hist_rows[all_ids[0]].shape[1]}")

    # 3. Evaluate Protocol B: DEPLOYMENT_CAL
    print("\n--- Running DEPLOYMENT_CAL (6 rank views: run_vnlegal_extra_channel_submission.py semantics) ---", flush=True)
    dep_local_views = dict(local_views)
    dep_local_views["vnlegal_lal"] = {
        q: sorted(extended[q], key=lambda d: (-vnlegal_cv.get(q, {}).get(d, -1e9), d))
        for q in all_ids
    }

    dep_rows, dep_groups = ltr_features(dep_local_views, DEPLOYMENT_VIEWS, extended, all_ids, full_channels_cv)
    for q in dep_rows:
        dep_rows[q] = np.concatenate([dep_rows[q], type_rows[q], cite_rows[q]], axis=1)

    dep_preds = {}
    for held in blocks:
        train = sum((blocks[n] for n in blocks if n != held), [])
        X_train = np.vstack([dep_rows[q] for q in train])
        y_train = np.concatenate([[d in gold[q] for d in dep_groups[q]] for q in train]).astype(np.int8)
        scaler = StandardScaler().fit(X_train)
        model = LogisticRegression(C=0.15, class_weight="balanced", solver="liblinear", max_iter=3000, random_state=2026)
        model.fit(scaler.transform(X_train), y_train)

        for q in blocks[held]:
            score = model.decision_function(scaler.transform(dep_rows[q]))
            dep_preds[q] = [dep_groups[q][i] for i in np.argsort(-score)[:5]]

    dep_pooled_r5 = float(np.mean([len(gold[q] & set(dep_preds[q])) / len(gold[q]) for q in all_ids]))
    dep_pooled_p5 = float(np.mean([len(gold[q] & set(dep_preds[q])) / 5.0 for q in all_ids]))
    dep_blocks = {
        b: float(np.mean([len(gold[q] & set(dep_preds[q])) / len(gold[q]) for q in blocks[b]]))
        for b in blocks
    }
    dep_single_r5 = float(np.mean([len(gold[q] & set(dep_preds[q])) / len(gold[q]) for q in all_ids if len(gold[q]) == 1]))
    dep_multi_r5 = float(np.mean([len(gold[q] & set(dep_preds[q])) / len(gold[q]) for q in all_ids if len(gold[q]) > 1]))

    print(f"DEPLOYMENT_CAL Pooled R@5: {dep_pooled_r5:.6f}")
    print(f"  Blocks: a={dep_blocks['a']:.4f}, b={dep_blocks['b']:.4f}, c={dep_blocks['c']:.4f}, d={dep_blocks['d']:.4f}")
    print(f"  Feature dim: {dep_rows[all_ids[0]].shape[1]}")

    # 4. Mandatory Public H0 Reproduction
    print("\n--- Rebuilding H0 Public Predictions from Production Features + LTR ---", flush=True)
    from run_burst_expanded_fusion_submission import (
        EXPANSION_CONFIG, RERANK_CONFIG, dense_expansion, corpus_dense, rerank, weighted_rrf, raw_union,
        load_public_retrieval
    )

    class DummyArgs:
        db = REPO_ROOT / "benchmarks/legalir_full_fts.sqlite"
        workers = 4
        cache_dir = REPO_ROOT / "results/burst_expanded_fusion"
        output_dir = REPO_ROOT / "results/burst_userft_maxrecall"

    paths_meta, doc_ids_meta, train_meta, public_meta = load_metadata(REPO_ROOT / "DSC2026-LegalIR-main/v4_run/public_test_dataset")
    public_ids = list(public_meta)
    valid_docs = set(doc_ids_meta)

    print("Loading public candidate pool and retrieval views...", flush=True)
    base_pub = load_pkl("results/burst_gpu_threeview/cpu_top20.pkl")["rankings"]
    pub_retrieval = load_public_retrieval(REPO_ROOT, DummyArgs, doc_ids_meta, train_meta, public_meta, public_ids)
    raw_pub = {q: raw_union(pub_retrieval[q], EXPANSION_CONFIG["depth"]) for q in public_ids}
    pub_retrieval.clear()

    expansion_scores_pub = dense_expansion(REPO_ROOT, DummyArgs.cache_dir, public_meta, public_ids, raw_pub, docs, "cuda")
    dense_rank_pub = {q: sorted(raw_pub[q], key=lambda d: (-expansion_scores_pub[q][d], d)) for q in public_ids}
    expanded_pub = weighted_rrf([raw_pub, dense_rank_pub], RERANK_CONFIG["expansion_weights"], RERANK_CONFIG["expansion_rrf_k"])
    corpus_rank_pub, corpus_score_pub = corpus_dense(REPO_ROOT, DummyArgs.cache_dir, public_meta, public_ids, "cuda", cap=CORPUS_CAP, depth=CORPUS_DEPTH)

    public_candidates = {
        q: list(dict.fromkeys(
            list(base_pub[q]) + expanded_pub[q][:RERANK_CONFIG["expanded_depth"]] + corpus_rank_pub[q][:CORPUS_DEPTH]
        ))
        for q in public_ids
    }
    corpus_score_pub = {q: {d: corpus_score_pub[q][d] for d in public_candidates[q] if d in corpus_score_pub[q]} for q in public_ids}
    expansion_scores_pub = {q: {d: expansion_scores_pub[q][d] for d in public_candidates[q] if d in expansion_scores_pub[q]} for q in public_ids}
    expanded_pub = {q: expanded_pub[q][:RERANK_CONFIG["expanded_depth"]] for q in public_ids}

    rerank_scores_pub = rerank(REPO_ROOT, DummyArgs.cache_dir, public_meta, public_ids, public_candidates, docs, "cuda")
    vnlegal_pub = load_pkl("results/burst_userft_maxrecall/vnlegal_scores.pkl")

    view_rank_pub = {
        "base": {q: list(base_pub[q]) for q in public_ids},
        "expanded": expanded_pub,
        "jina": {q: sorted(public_candidates[q], key=lambda d: (-rerank_scores_pub["jina"][q][d], d)) for q in public_ids},
        "dense": {q: sorted(public_candidates[q], key=lambda d: (-rerank_scores_pub["dense"][q][d], d)) for q in public_ids},
        "corpus": {q: sorted((d for d in public_candidates[q] if d in corpus_score_pub[q]), key=lambda d: (-corpus_score_pub[q][d], d)) for q in public_ids},
        "vnlegal_lal": {q: sorted(public_candidates[q], key=lambda d: (-vnlegal_pub.get(q, {}).get(d, -1e9), d)) for q in public_ids},
    }

    three_view_pub = load_pkl("results/burst_gpu_threeview/gpu_scores.checkpoint.pkl")
    crossenc_pub = load_pkl("results/crossenc_fullpool/public_scores.pkl")
    floor_ce_pub = min(min(v.values()) for v in crossenc_cv.values() if v)

    public_scores_h0 = {
        "jina": rerank_scores_pub["jina"],
        "dense": rerank_scores_pub["dense"],
        "expansion": expansion_scores_pub,
        "e5": {q: three_view_pub[q]["e5"] for q in public_ids},
        "corpus": {q: {d: corpus_score_pub[q].get(d, -1.0) for d in public_candidates[q]} for q in public_ids},
        "vnlegal_lal": vnlegal_pub,
        "crossenc": {q: {d: crossenc_pub.get(q, {}).get(d, floor_ce_pub) for d in public_candidates[q]} for q in public_ids},
    }

    for name, pub_rel in EXTRA_PUB_PATHS.items():
        raw_p = load_pkl(pub_rel)
        fl_p = min(v for q in raw_p for v in raw_p[q].values())
        public_scores_h0[name] = {q: {d: raw_p.get(q, {}).get(d, fl_p) for d in public_candidates[q]} for q in public_ids}

    # Public metadata features
    print("Building public doctype and citation features...", flush=True)
    public_queries_meta = {q: (public_meta[q], set()) for q in public_ids}
    public_types = build_type_table(REPO_ROOT, docs, public_ids, public_candidates)
    public_t_rows = type_features(public_candidates, public_types, public_queries_meta, public_ids)
    public_own, public_cited = build_citation_table(docs, public_ids, public_candidates)
    public_c_rows = citation_features(public_candidates, public_own, public_cited, public_ids)

    # Public H0 LTR feature rows
    print("Generating public H0 feature rows...", flush=True)
    public_rows_base, public_groups = ltr_features(view_rank_pub, DEPLOYMENT_VIEWS, public_candidates, public_ids, public_scores_h0)
    public_rows_h0 = {q: np.concatenate([public_rows_base[q], public_t_rows[q], public_c_rows[q]], axis=1) for q in public_ids}

    # Fit LTR on all 600 CAL queries using DEPLOYMENT_CAL features
    print("Fitting production H0 LTR model on all 600 CAL queries...", flush=True)
    X_full_cal = np.vstack([dep_rows[q] for q in all_ids])
    y_full_cal = np.concatenate([[d in gold[q] for d in dep_groups[q]] for q in all_ids]).astype(np.int8)
    full_scaler = StandardScaler().fit(X_full_cal)
    full_ltr = LogisticRegression(C=0.15, class_weight="balanced", solver="liblinear", max_iter=3000, random_state=2026)
    full_ltr.fit(full_scaler.transform(X_full_cal), y_full_cal)

    # Generate H0 public predictions
    rebuilt_preds = {}
    for q in public_ids:
        proba = full_ltr.predict_proba(full_scaler.transform(public_rows_h0[q]))[:, 1]
        order = np.argsort(-proba)
        fused = [public_groups[q][i] for i in order]

        final = [d for d in fused if d in valid_docs][:5]
        for pool in (public_candidates[q], doc_ids_meta):
            if len(final) >= 5:
                break
            for doc in pool:
                if len(final) >= 5:
                    break
                if doc not in final and doc in valid_docs:
                    final.append(doc)
        final = final[:5]
        rebuilt_preds[q] = {"answer": final}

    # Save H0_REBUILT.json
    rebuilt_json_path = RESULTS_DIR / "H0_REBUILT.json"
    with open(rebuilt_json_path, "w", encoding="utf-8") as f:
        json.dump(rebuilt_preds, f, indent=2, ensure_ascii=False)
    print(f"Wrote {rebuilt_json_path}", flush=True)

    # Compare H0_REBUILT.json against CONTROL_JSON_PATH
    with open(CONTROL_JSON_PATH, "r", encoding="utf-8") as f:
        control_preds = json.load(f)

    ordered_matches = 0
    set_matches = 0
    changed_queries = 0
    jaccards = []

    for q in public_ids:
        ans_rebuilt = rebuilt_preds[q]["answer"]
        ans_control = control_preds[q]["answer"]
        
        if ans_rebuilt == ans_control:
            ordered_matches += 1
        if set(ans_rebuilt) == set(ans_control):
            set_matches += 1
        else:
            changed_queries += 1

        jacc = len(set(ans_rebuilt) & set(ans_control)) / len(set(ans_rebuilt) | set(ans_control))
        jaccards.append(jacc)

    print(f"\n--- H0 REPRODUCTION AUDIT REPORT ---")
    print(f"Ordered Top-5 Exact Matches: {ordered_matches}/1000 ({ordered_matches/10:.1f}%)")
    print(f"Top-5 Set Exact Matches: {set_matches}/1000 ({set_matches/10:.1f}%)")
    print(f"Changed Queries vs Control: {changed_queries}")
    print(f"Mean Top-5 Jaccard: {np.mean(jaccards):.6f}")

    h0_gate_passed = (set_matches == 1000)
    print(f"Mandatory Gate (Top-5 SET match == 100%): {'PASS' if h0_gate_passed else 'FAIL'}")

    # Write BASELINE_CONTRACT_AUDIT.json
    audit_report = {
        "schema_version": "dsc2026.gemini.huy_e5_transplant_corrected_v2.baseline_contract_audit.v1",
        "h0_reproduction_gate_passed": h0_gate_passed,
        "h0_reproduction": {
            "control_json_path": str(CONTROL_JSON_PATH),
            "rebuilt_json_path": str(rebuilt_json_path),
            "total_queries": 1000,
            "ordered_matches": ordered_matches,
            "set_matches": set_matches,
            "changed_queries": changed_queries,
            "mean_top5_jaccard": float(np.mean(jaccards)),
            "ordered_match_pct": ordered_matches / 10.0,
            "set_match_pct": set_matches / 10.0,
        },
        "protocols": {
            "HISTORICAL_CAL": {
                "source_script": "evaluate_cv.py",
                "rank_views": HISTORICAL_VIEWS,
                "rank_views_count": len(HISTORICAL_VIEWS),
                "score_channels": sorted(list(full_channels_cv.keys())),
                "score_channels_count": len(full_channels_cv),
                "feature_count": int(hist_rows[all_ids[0]].shape[1]),
                "feature_breakdown": {
                    "rank_view_features": len(HISTORICAL_VIEWS) * 2 + 2, # 12
                    "score_channel_features": len(full_channels_cv) * 2, # 20
                    "doctype_features": 15,
                    "citation_features": 5,
                    "total": 12 + 20 + 15 + 5 # 52
                },
                "metrics": {
                    "pooled_recall_at_5": hist_pooled_r5,
                    "pooled_precision_at_5": hist_pooled_p5,
                    "single_gold_recall_at_5": hist_single_r5,
                    "multi_gold_recall_at_5": hist_multi_r5,
                    "blocks": hist_blocks
                }
            },
            "DEPLOYMENT_CAL": {
                "source_script": "run_vnlegal_extra_channel_submission.py",
                "rank_views": DEPLOYMENT_VIEWS,
                "rank_views_count": len(DEPLOYMENT_VIEWS),
                "score_channels": sorted(list(full_channels_cv.keys())),
                "score_channels_count": len(full_channels_cv),
                "feature_count": int(dep_rows[all_ids[0]].shape[1]),
                "feature_breakdown": {
                    "rank_view_features": len(DEPLOYMENT_VIEWS) * 2 + 2, # 14
                    "score_channel_features": len(full_channels_cv) * 2, # 20
                    "doctype_features": 15,
                    "citation_features": 5,
                    "total": 14 + 20 + 15 + 5 # 54 (or 50 if dropping certain unused views)
                },
                "metrics": {
                    "pooled_recall_at_5": dep_pooled_r5,
                    "pooled_precision_at_5": dep_pooled_p5,
                    "single_gold_recall_at_5": dep_single_r5,
                    "multi_gold_recall_at_5": dep_multi_r5,
                    "blocks": dep_blocks
                }
            }
        },
        "contract_difference_explanation": (
            "HISTORICAL_CAL (evaluate_cv.py) treats vnlegal_lal strictly as a score channel with 5 rank views "
            "(['base', 'expanded', 'jina', 'dense', 'corpus']), yielding ~0.9561 recall. "
            "DEPLOYMENT_CAL (run_vnlegal_extra_channel_submission.py) additionally appends vnlegal_lal as the 6th "
            "rank view (view_rank['vnlegal_lal']), yielding ~0.950556 recall under LOBO cross-validation. "
            "Both protocols are valid: HISTORICAL_CAL measures the original research claim, while DEPLOYMENT_CAL "
            "measures the exact feature configuration serialized into public submissions."
        )
    }

    out_audit = RESULTS_DIR / "BASELINE_CONTRACT_AUDIT.json"
    with open(out_audit, "w", encoding="utf-8") as f:
        json.dump(audit_report, f, indent=2, ensure_ascii=False)
    print(f"\nWrote baseline contract audit to {out_audit}")

    if not h0_gate_passed:
        raise RuntimeError(f"H0 REPRODUCTION FAILED: Set match was {set_matches}/1000. STOPPING EXPERIMENT.")

if __name__ == "__main__":
    main()
