"""Train final models on all CAL600 and materialize Public test packages for H0, Q1, Q2."""

from __future__ import annotations

import hashlib
import json
import pickle
import sys
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = ROOT / "results" / "gemini" / "huy_query_balanced_pairwise_ltr_v1"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT))
from run_burst_expanded_fusion_submission import (
    CORPUS_CAP,
    CORPUS_DEPTH,
    EXPANSION_CONFIG,
    RERANK_CONFIG,
    DocumentStore,
    corpus_dense,
    dense_expansion,
    load_public_retrieval,
    raw_union,
    rerank,
    weighted_rrf,
)
from run_burst_multistage_submission import load_metadata
from tune_citation_graph import build_citation_table, citation_features
from tune_corpus_cap32_fusion import build_training_cap
from tune_doctype_features import build_type_table, type_features
from tune_expanded_fusion_selection import ltr_features

from pairwise_ranker import QueryBalancedPairwiseRanker

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


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def md5_file(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def load_pkl(rel_path: str):
    p = ROOT / rel_path
    obj = pickle.loads(p.read_bytes())
    if isinstance(obj, dict) and isinstance(obj.get("scores"), dict):
        return obj["scores"]
    return obj


def validate_submission_zip(zip_path: Path, json_path: Path, valid_docs: Set[str]):
    assert zip_path.exists(), f"Zip file missing: {zip_path}"
    with zipfile.ZipFile(zip_path, "r") as zf:
        namelist = zf.namelist()
        assert namelist == ["submission.json"], f"Zip contains unexpected files: {namelist}"
        extracted_bytes = zf.read("submission.json")
        source_bytes = json_path.read_bytes()
        assert extracted_bytes == source_bytes, "Extracted bytes do not equal source JSON bytes"
        payload = json.loads(extracted_bytes.decode("utf-8"))
        assert len(payload) == 1000, f"Expected 1000 queries, got {len(payload)}"
        for q, val in payload.items():
            docs = val["answer"] if isinstance(val, dict) else val
            assert len(docs) == 5, f"Query {q} does not have 5 docs"
            assert len(set(docs)) == 5, f"Query {q} contains duplicate docs"
            for d in docs:
                assert d in valid_docs, f"Doc {d} in query {q} not in valid_docs"
    return True


def audit_public_churn(
    cand_preds: Dict[str, Dict[str, List[str]]],
    ref_preds: Dict[str, Dict[str, List[str]]],
) -> Dict[str, Any]:
    qids = sorted(cand_preds.keys())
    changed_top5_sets = 0
    changed_ordered_outputs = 0
    jaccards = []
    entering_docs = 0
    leaving_docs = 0
    rank5_boundary_changes = 0

    for q in qids:
        c_list = cand_preds[q]["answer"]
        r_list = ref_preds[q]["answer"]
        c_set = set(c_list)
        r_set = set(r_list)

        if c_set != r_set:
            changed_top5_sets += 1
            entering_docs += len(c_set - r_set)
            leaving_docs += len(r_set - c_set)

        if c_list != r_list:
            changed_ordered_outputs += 1

        if c_list[4] != r_list[4]:
            rank5_boundary_changes += 1

        jacc = len(c_set & r_set) / len(c_set | r_set) if (c_set | r_set) else 1.0
        jaccards.append(jacc)

    return {
        "total_queries": len(qids),
        "changed_top5_sets": changed_top5_sets,
        "changed_top5_sets_pct": (changed_top5_sets / len(qids)) * 100.0,
        "changed_ordered_outputs": changed_ordered_outputs,
        "changed_ordered_outputs_pct": (changed_ordered_outputs / len(qids)) * 100.0,
        "mean_top5_jaccard": float(np.mean(jaccards)),
        "entering_docs_count": entering_docs,
        "leaving_docs_count": leaving_docs,
        "rank5_boundary_changes": rank5_boundary_changes,
    }


def main() -> Dict[str, Any]:
    started_all = time.perf_counter()
    print("=== Step 4: Materialize Public Candidate Packages ===", flush=True)

    # 1. Load document store and CAL structures
    print("Loading DocumentStore and CAL inputs...", flush=True)
    docs = DocumentStore(
        sorted(
            (
                ROOT
                / "DSC2026-LegalIR-main/v4_run/public_test_dataset/selected-contexts"
            ).glob("context_*.json")
        )
    )
    queries, blocks, all_ids, extended, local_views, base_scores = (
        build_training_cap(
            ROOT,
            32,
            "results/corpus_index/holdout_extended_scores_cap32.pkl",
            depth=20,
        )
    )
    gold = {q: queries[q][1] for q in all_ids}

    def load_aligned(rel_path: str, floor=None):
        obj = load_pkl(rel_path)
        fl = floor if floor is not None else min(v for q in obj for v in obj[q].values())
        return {q: {d: obj.get(q, {}).get(d, fl) for d in extended[q]} for q in all_ids}

    vnlegal_cv = load_pkl("results/embedding_finetune/vnlegal_lal_cv_scores.pkl")
    crossenc_cv = load_aligned("results/crossenc_fullpool/cv_scores.pkl", -11.5)
    extra_cv = {name: load_aligned(rel) for name, rel in EXTRA_CV_PATHS.items()}
    full_channels_cv = {
        **base_scores,
        "vnlegal_lal": vnlegal_cv,
        "crossenc": crossenc_cv,
        **extra_cv,
    }

    type_table = build_type_table(ROOT, docs, all_ids, extended)
    type_rows = type_features(extended, type_table, queries, all_ids)
    own, cited = build_citation_table(docs, all_ids, extended)
    cite_rows = citation_features(extended, own, cited, all_ids)

    # 2. Load Public Candidates and Views
    class DummyArgs:
        db = ROOT / "benchmarks/legalir_full_fts.sqlite"
        workers = 4
        cache_dir = ROOT / "results/burst_expanded_fusion"
        output_dir = ROOT / "results/burst_userft_maxrecall"

    paths_meta, doc_ids_meta, train_meta, public_meta = load_metadata(
        ROOT / "DSC2026-LegalIR-main/v4_run/public_test_dataset"
    )
    public_ids = list(public_meta)
    valid_docs = set(doc_ids_meta)

    print("Loading public candidate pool and retrieval views...", flush=True)
    base_pub = load_pkl("results/burst_gpu_threeview/cpu_top20.pkl")["rankings"]
    pub_retrieval = load_public_retrieval(
        ROOT, DummyArgs, doc_ids_meta, train_meta, public_meta, public_ids
    )
    raw_pub = {
        q: raw_union(pub_retrieval[q], EXPANSION_CONFIG["depth"]) for q in public_ids
    }
    pub_retrieval.clear()

    expansion_scores_pub = dense_expansion(
        ROOT, DummyArgs.cache_dir, public_meta, public_ids, raw_pub, docs, "cuda"
    )
    dense_rank_pub = {
        q: sorted(raw_pub[q], key=lambda d: (-expansion_scores_pub[q][d], d))
        for q in public_ids
    }
    expanded_pub = weighted_rrf(
        [raw_pub, dense_rank_pub],
        RERANK_CONFIG["expansion_weights"],
        RERANK_CONFIG["expansion_rrf_k"],
    )
    corpus_rank_pub, corpus_score_pub = corpus_dense(
        ROOT,
        DummyArgs.cache_dir,
        public_meta,
        public_ids,
        "cuda",
        cap=CORPUS_CAP,
        depth=CORPUS_DEPTH,
    )

    public_candidates = {
        q: list(
            dict.fromkeys(
                list(base_pub[q])
                + expanded_pub[q][: RERANK_CONFIG["expanded_depth"]]
                + corpus_rank_pub[q][:CORPUS_DEPTH]
            )
        )
        for q in public_ids
    }
    corpus_score_pub = {
        q: {
            d: corpus_score_pub[q][d]
            for d in public_candidates[q]
            if d in corpus_score_pub[q]
        }
        for q in public_ids
    }
    expansion_scores_pub = {
        q: {
            d: expansion_scores_pub[q][d]
            for d in public_candidates[q]
            if d in expansion_scores_pub[q]
        }
        for q in public_ids
    }
    expanded_pub = {
        q: expanded_pub[q][: RERANK_CONFIG["expanded_depth"]] for q in public_ids
    }

    rerank_scores_pub = rerank(
        ROOT,
        DummyArgs.cache_dir,
        public_meta,
        public_ids,
        public_candidates,
        docs,
        "cuda",
    )
    vnlegal_pub = load_pkl("results/burst_userft_maxrecall/vnlegal_scores.pkl")

    view_rank_pub = {
        "base": {q: list(base_pub[q]) for q in public_ids},
        "expanded": expanded_pub,
        "jina": {
            q: sorted(
                public_candidates[q],
                key=lambda d: (-rerank_scores_pub["jina"][q][d], d),
            )
            for q in public_ids
        },
        "dense": {
            q: sorted(
                public_candidates[q],
                key=lambda d: (-rerank_scores_pub["dense"][q][d], d),
            )
            for q in public_ids
        },
        "corpus": {
            q: sorted(
                (d for d in public_candidates[q] if d in corpus_score_pub[q]),
                key=lambda d: (-corpus_score_pub[q][d], d),
            )
            for q in public_ids
        },
        "vnlegal_lal": {
            q: sorted(
                public_candidates[q],
                key=lambda d: (-vnlegal_pub.get(q, {}).get(d, -1e9), d),
            )
            for q in public_ids
        },
    }

    three_view_pub = load_pkl("results/burst_gpu_threeview/gpu_scores.checkpoint.pkl")
    crossenc_pub = load_pkl("results/crossenc_fullpool/public_scores.pkl")
    floor_ce_pub = min(min(v.values()) for v in crossenc_cv.values() if v)

    public_scores_h0 = {
        "jina": rerank_scores_pub["jina"],
        "dense": rerank_scores_pub["dense"],
        "expansion": expansion_scores_pub,
        "e5": {q: three_view_pub[q]["e5"] for q in public_ids},
        "corpus": {
            q: {d: corpus_score_pub[q].get(d, -1.0) for d in public_candidates[q]}
            for q in public_ids
        },
        "vnlegal_lal": vnlegal_pub,
        "crossenc": {
            q: {
                d: crossenc_pub.get(q, {}).get(d, floor_ce_pub)
                for d in public_candidates[q]
            }
            for q in public_ids
        },
    }

    for name, pub_rel in EXTRA_PUB_PATHS.items():
        raw_p = load_pkl(pub_rel)
        fl_p = min(v for q in raw_p for v in raw_p[q].values())
        public_scores_h0[name] = {
            q: {d: raw_p.get(q, {}).get(d, fl_p) for d in public_candidates[q]}
            for q in public_ids
        }

    # Public metadata features
    print("Building public doctype and citation features...", flush=True)
    public_queries_meta = {q: (public_meta[q], set()) for q in public_ids}
    public_types = build_type_table(ROOT, docs, public_ids, public_candidates)
    public_t_rows = type_features(
        public_candidates, public_types, public_queries_meta, public_ids
    )
    public_own, public_cited = build_citation_table(
        docs, public_ids, public_candidates
    )
    public_c_rows = citation_features(
        public_candidates, public_own, public_cited, public_ids
    )

    # 3. Load Profile BM25 rankings cache
    profile_cache_path = (
        ROOT
        / "results"
        / "gemini"
        / "huy_fulltrain_profile_port_v1"
        / "PROFILE_BM25_RANKINGS.pkl"
    )
    profile_cache = pickle.loads(profile_cache_path.read_bytes())
    cal_oof_rankings = profile_cache["cal_oof_rankings"]
    public_rankings = profile_cache["public_rankings"]

    # --- ARM 0: CONTROL_H0 (50D Pointwise) ---
    print("\n--- Materializing CONTROL_H0 ---", flush=True)
    dep_local_views = dict(local_views)
    dep_local_views["vnlegal_lal"] = {
        q: sorted(extended[q], key=lambda d: (-vnlegal_cv.get(q, {}).get(d, -1e9), d))
        for q in all_ids
    }
    dep_rows_base, dep_groups = ltr_features(
        dep_local_views, DEPLOYMENT_VIEWS, extended, all_ids, full_channels_cv
    )
    dep_rows_h0 = {
        q: np.concatenate([dep_rows_base[q], type_rows[q], cite_rows[q]], axis=1)
        for q in all_ids
    }

    X_h0 = np.vstack([dep_rows_h0[q] for q in all_ids])
    y_by_qid_h0 = {
        q: np.asarray([d in gold[q] for d in dep_groups[q]], dtype=np.int8)
        for q in all_ids
    }
    y_h0 = np.concatenate([y_by_qid_h0[q] for q in all_ids])

    scaler_h0 = StandardScaler().fit(X_h0)
    ltr_h0 = LogisticRegression(
        C=0.15,
        class_weight="balanced",
        solver="liblinear",
        max_iter=3000,
        random_state=2026,
    )
    ltr_h0.fit(scaler_h0.transform(X_h0), y_h0)

    public_rows_base_h0, _ = ltr_features(
        view_rank_pub, DEPLOYMENT_VIEWS, public_candidates, public_ids, public_scores_h0
    )
    public_rows_h0 = {
        q: np.concatenate(
            [public_rows_base_h0[q], public_t_rows[q], public_c_rows[q]], axis=1
        )
        for q in public_ids
    }

    preds_h0 = {}
    for q in public_ids:
        proba = ltr_h0.predict_proba(scaler_h0.transform(public_rows_h0[q]))[:, 1]
        order = np.argsort(-proba)
        fused = [public_candidates[q][i] for i in order]
        preds_h0[q] = {"answer": [d for d in fused if d in valid_docs][:5]}

    control_json = RESULTS_DIR / "CONTROL_H0.json"
    control_zip = RESULTS_DIR / "CONTROL_H0.zip"
    control_json.write_text(json.dumps(preds_h0, ensure_ascii=False, indent=2), encoding="utf-8")
    with zipfile.ZipFile(control_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("submission.json", control_json.read_bytes())
    validate_submission_zip(control_zip, control_json, valid_docs)

    prod_sub = json.loads((ROOT / "results/burst_userft_maxrecall/submission.json").read_text(encoding="utf-8"))
    h0_set_matches = sum(set(preds_h0[q]["answer"]) == set(prod_sub[q]["answer"]) for q in public_ids)
    h0_ord_matches = sum(preds_h0[q]["answer"] == prod_sub[q]["answer"] for q in public_ids)
    print(f"CONTROL_H0 parity vs burst_userft_maxrecall: Set={h0_set_matches}/1000, Ordered={h0_ord_matches}/1000")
    assert h0_set_matches == 1000 and h0_ord_matches == 1000, "CONTROL_H0 does not match burst_userft_maxrecall"

    # --- ARM 1: CANDIDATE_Q1_PAIRWISE (50D Pairwise) ---
    print("\n--- Materializing CANDIDATE_Q1_PAIRWISE ---", flush=True)
    ranker_q1 = QueryBalancedPairwiseRanker(C=0.15, max_iter=3000, random_state=2026)
    ranker_q1.fit(dep_rows_h0, y_by_qid_h0, all_ids)

    preds_q1 = {}
    for q in public_ids:
        utility = ranker_q1.predict_utility(public_rows_h0[q])
        order = np.argsort(-utility)
        fused = [public_candidates[q][i] for i in order]
        preds_q1[q] = {"answer": [d for d in fused if d in valid_docs][:5]}

    q1_json = RESULTS_DIR / "CANDIDATE_Q1_PAIRWISE.json"
    q1_zip = RESULTS_DIR / "CANDIDATE_Q1_PAIRWISE.zip"
    q1_json.write_text(json.dumps(preds_q1, ensure_ascii=False, indent=2), encoding="utf-8")
    with zipfile.ZipFile(q1_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("submission.json", q1_json.read_bytes())
    validate_submission_zip(q1_zip, q1_json, valid_docs)
    print(f"Validated CANDIDATE_Q1_PAIRWISE.zip (SHA256: {sha256_file(q1_zip)})")

    # --- ARM 2: CANDIDATE_Q2_PAIRWISE_PROFILE (52D Pairwise + Profile) ---
    print("\n--- Materializing CANDIDATE_Q2_PAIRWISE_PROFILE ---", flush=True)
    p2_views_cal = dict(dep_local_views)
    p2_views_cal["fulltrain_huy_profile"] = cal_oof_rankings
    p2_names = list(DEPLOYMENT_VIEWS) + ["fulltrain_huy_profile"]

    p2_rows_cal_base, _ = ltr_features(
        p2_views_cal, p2_names, extended, all_ids, full_channels_cv
    )
    p2_rows_cal = {
        q: np.concatenate([p2_rows_cal_base[q], type_rows[q], cite_rows[q]], axis=1)
        for q in all_ids
    }

    ranker_q2 = QueryBalancedPairwiseRanker(C=0.15, max_iter=3000, random_state=2026)
    ranker_q2.fit(p2_rows_cal, y_by_qid_h0, all_ids)

    p2_views_pub = dict(view_rank_pub)
    p2_views_pub["fulltrain_huy_profile"] = public_rankings

    public_rows_base_p2, _ = ltr_features(
        p2_views_pub, p2_names, public_candidates, public_ids, public_scores_h0
    )
    public_rows_p2 = {
        q: np.concatenate(
            [public_rows_base_p2[q], public_t_rows[q], public_c_rows[q]], axis=1
        )
        for q in public_ids
    }

    preds_q2 = {}
    for q in public_ids:
        utility = ranker_q2.predict_utility(public_rows_p2[q])
        order = np.argsort(-utility)
        fused = [public_candidates[q][i] for i in order]
        preds_q2[q] = {"answer": [d for d in fused if d in valid_docs][:5]}

    q2_json = RESULTS_DIR / "CANDIDATE_Q2_PAIRWISE_PROFILE.json"
    q2_zip = RESULTS_DIR / "CANDIDATE_Q2_PAIRWISE_PROFILE.zip"
    q2_json.write_text(json.dumps(preds_q2, ensure_ascii=False, indent=2), encoding="utf-8")
    with zipfile.ZipFile(q2_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("submission.json", q2_json.read_bytes())
    validate_submission_zip(q2_zip, q2_json, valid_docs)
    print(f"Validated CANDIDATE_Q2_PAIRWISE_PROFILE.zip (SHA256: {sha256_file(q2_zip)})")

    # Audit Public Churn
    churn_q1_vs_h0 = audit_public_churn(preds_q1, preds_h0)
    churn_q2_vs_h0 = audit_public_churn(preds_q2, preds_h0)
    churn_q2_vs_q1 = audit_public_churn(preds_q2, preds_q1)

    # Check promotion gates from Dual CAL and V2 Shadow
    dual_cal_rep = json.loads((RESULTS_DIR / "HUY_PAIRWISE_DUAL_CAL_REPORT.json").read_text(encoding="utf-8"))
    v2_shadow_rep = json.loads((RESULTS_DIR / "V2_PAIRWISE_SHADOW_REPORT.json").read_text(encoding="utf-8"))

    v2_passed = v2_shadow_rep["generalization_gates"]["all_v2_gates_passed"]

    def evaluate_gates(arm_key: str) -> Dict[str, Any]:
        arm_data_hist = dual_cal_rep["historical_cal"][arm_key]
        arm_data_dep = dual_cal_rep["deployment_cal"][arm_key]

        hist_r5 = arm_data_hist["metrics"]["pooled_recall_at_5"]
        q0_hist_r5 = dual_cal_rep["historical_cal"]["q0_pointwise"]["metrics"]["pooled_recall_at_5"]
        dep_r5 = arm_data_dep["metrics"]["pooled_recall_at_5"]
        q0_dep_r5 = dual_cal_rep["deployment_cal"]["q0_pointwise"]["metrics"]["pooled_recall_at_5"]

        gate_a = hist_r5 > q0_hist_r5
        gate_b = dep_r5 >= (q0_dep_r5 - 1e-9)
        block_d_regress = arm_data_hist["delta_vs_q0"]["block_deltas"].get("d", 0.0) < -1e-9
        gate_c = not block_d_regress
        gate_d = arm_data_hist["comparison_vs_q0"]["wins"] > arm_data_hist["comparison_vs_q0"]["losses"]
        gate_e = arm_data_hist["delta_vs_q0"]["delta_multi_gold"] >= -0.003
        gate_f = True  # Parity passed

        cal_passed = gate_a and gate_b and gate_c and gate_d and gate_e and gate_f
        all_passed = cal_passed and v2_passed

        return {
            "gate_a_hist_cal_gain": gate_a,
            "gate_b_dep_cal_non_regressing": gate_b,
            "gate_c_block_d_non_regressing": gate_c,
            "gate_d_hist_wins_gt_losses": gate_d,
            "gate_e_multi_gold_delta_ge_neg_0_003": gate_e,
            "gate_f_parity_passed": gate_f,
            "gate_g_v2_overall_non_regressing": v2_shadow_rep["generalization_gates"]["gate_g_overall_non_regressing"],
            "gate_h_v2_at_least_3_folds_non_regressing": v2_shadow_rep["generalization_gates"]["gate_h_at_least_3_folds_non_regressing"],
            "gate_i_v2_no_fold_regresses_more_than_0_0015": v2_shadow_rep["generalization_gates"]["gate_i_no_fold_regresses_more_than_0_0015"],
            "all_gates_passed": all_passed,
            "historical_cal_r5": hist_r5,
        }

    q1_gates = evaluate_gates("q1_pairwise")
    q2_gates = evaluate_gates("q2_pairwise_plus_profile")

    promoted_arm = None
    promoted_zip = None
    if q1_gates["all_gates_passed"] and q2_gates["all_gates_passed"]:
        if q2_gates["historical_cal_r5"] > q1_gates["historical_cal_r5"]:
            promoted_arm = "Q2_PAIRWISE_PLUS_PROFILE"
            promoted_src_zip = q2_zip
            promoted_src_json = q2_json
        else:
            promoted_arm = "Q1_PAIRWISE"
            promoted_src_zip = q1_zip
            promoted_src_json = q1_json
    elif q1_gates["all_gates_passed"]:
        promoted_arm = "Q1_PAIRWISE"
        promoted_src_zip = q1_zip
        promoted_src_json = q1_json
    elif q2_gates["all_gates_passed"]:
        promoted_arm = "Q2_PAIRWISE_PLUS_PROFILE"
        promoted_src_zip = q2_zip
        promoted_src_json = q2_json

    if promoted_arm:
        promoted_zip = RESULTS_DIR / "PROMOTED.zip"
        promoted_zip.write_bytes(promoted_src_zip.read_bytes())
        validate_submission_zip(promoted_zip, promoted_src_json, valid_docs)
        print(f"PROMOTED.zip created from {promoted_arm} (SHA256: {sha256_file(promoted_zip)})")
    else:
        print("NO arm passed all promotion gates; PROMOTED.zip not created.")

    public_audit_report = {
        "schema_version": "dsc2026.gemini.huy_query_balanced_pairwise_ltr_v1.public_pairwise_audit.v1",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "packages": {
            "control_h0": {
                "json_path": str(control_json.relative_to(ROOT)).replace("\\", "/"),
                "zip_path": str(control_zip.relative_to(ROOT)).replace("\\", "/"),
                "sha256": sha256_file(control_zip),
                "md5": md5_file(control_zip),
                "exact_parity_with_burst_userft_maxrecall": h0_set_matches == 1000 and h0_ord_matches == 1000,
            },
            "candidate_q1_pairwise": {
                "json_path": str(q1_json.relative_to(ROOT)).replace("\\", "/"),
                "zip_path": str(q1_zip.relative_to(ROOT)).replace("\\", "/"),
                "sha256": sha256_file(q1_zip),
                "md5": md5_file(q1_zip),
                "churn_vs_h0": churn_q1_vs_h0,
            },
            "candidate_q2_pairwise_profile": {
                "json_path": str(q2_json.relative_to(ROOT)).replace("\\", "/"),
                "zip_path": str(q2_zip.relative_to(ROOT)).replace("\\", "/"),
                "sha256": sha256_file(q2_zip),
                "md5": md5_file(q2_zip),
                "churn_vs_h0": churn_q2_vs_h0,
                "churn_vs_q1": churn_q2_vs_q1,
            },
            "promoted": {
                "promoted_arm": promoted_arm,
                "zip_path": str(promoted_zip.relative_to(ROOT)).replace("\\", "/") if promoted_zip else None,
                "sha256": sha256_file(promoted_zip) if promoted_zip else None,
            },
        },
        "promotion_gates_evaluation": {
            "q1_pairwise": q1_gates,
            "q2_pairwise_plus_profile": q2_gates,
        },
    }

    out_file = RESULTS_DIR / "PUBLIC_PAIRWISE_AUDIT.json"
    out_file.write_text(json.dumps(public_audit_report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved: {out_file}", flush=True)
    return public_audit_report


if __name__ == "__main__":
    main()
