"""Fit final LTR models on all CAL600 queries and produce public submission packages.

Produces:
- results/gemini/huy_e5_adaptation_delta_v1/submission.json & submission.zip
  (materialized from H0 control per decision rules since no experimental arm beat H0 without regression)
- results/gemini/huy_e5_adaptation_delta_v1/best_experimental_candidate.json & best_experimental_candidate.zip
  (from Arm H3, for forensic comparison)
- results/gemini/huy_e5_adaptation_delta_v1/PUBLIC_SUBMISSION_AUDIT.json
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import pickle
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = ROOT / "results/gemini/huy_e5_adaptation_delta_v1"
CONTROL_DIR = RESULTS_DIR / "control"
DATA_DIR = ROOT / "DSC2026-LegalIR-main/v4_run/public_test_dataset"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def md5_file(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def compute_standardized_scores(raw_scores: Dict[str, float], docs: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    values = np.asarray([raw_scores.get(d, np.nan) for d in docs], dtype=np.float64)
    present = values[~np.isnan(values)]
    if present.size:
        mean = float(present.mean())
        std = float(present.std()) or 1.0
        top = float(present.max())
    else:
        mean, std, top = 0.0, 1.0, 0.0
    filled = np.where(np.isnan(values), mean - 2 * std, values)
    z = (filled - mean) / std
    gap = (filled - top) / std
    return z, gap


def build_h0_and_h3_matrices():
    """Build full CAL600 training matrices and Public test matrices for H0 and H3."""
    import sys
    sys.path.insert(0, str(ROOT))
    from benchmark_dense_expansion_holdouts import raw_union
    from run_burst_expanded_fusion_submission import (
        CORPUS_CAP,
        CORPUS_DEPTH,
        DocumentStore,
        EXPANSION_CONFIG,
        RERANK_CONFIG,
        VIEWS,
        corpus_dense,
        dense_expansion,
        load_public_retrieval,
        rerank,
    )
    from run_burst_multistage_submission import load_metadata
    from tune_burst_multistage_posterior import weighted_rrf
    from tune_citation_graph import build_citation_table, citation_features
    from tune_corpus_dense_fusion import build_training
    from tune_doctype_features import build_type_table, type_features
    from tune_expanded_fusion_selection import ltr_features

    class DummyArgs:
        data_dir = DATA_DIR
        db = ROOT / "benchmarks/legalir_full_fts.sqlite"
        cache_dir = ROOT / "results/burst_expanded_fusion"
        output_dir = ROOT / "results/burst_userft_maxrecall"

    paths, doc_ids, train, public = load_metadata(DATA_DIR)
    public_ids = list(public)
    valid_docs = set(doc_ids)
    documents = DocumentStore(paths)

    # --- CAL600 Training Setup ---
    print("Building CAL600 training data...", flush=True)
    queries, blocks, holdout_ids, holdout_candidates, holdout_views, training_scores = build_training(
        ROOT, depth=10, cap=32, extended_scores_path="results/corpus_index/holdout_extended_scores_cap32.pkl"
    )

    holdout_vnlegal = pickle.loads((ROOT / "results/embedding_finetune/vnlegal_lal_cv_scores.pkl").read_bytes())
    names_h0 = ["base", "expanded", "jina", "dense", "corpus", "vnlegal_lal"]
    training_scores = dict(training_scores)
    training_scores["vnlegal_lal"] = holdout_vnlegal
    holdout_views = dict(holdout_views)
    holdout_views["vnlegal_lal"] = {
        q: sorted(holdout_candidates[q], key=lambda d: (-holdout_vnlegal.get(q, {}).get(d, -1e9), d))
        for q in holdout_ids
    }

    cv_ce = pickle.loads((ROOT / "results/crossenc_fullpool/cv_scores.pkl").read_bytes())["scores"]
    floor_ce = min(min(v.values()) for v in cv_ce.values() if v)
    training_scores["crossenc"] = {
        q: {d: cv_ce.get(q, {}).get(d, floor_ce) for d in holdout_candidates[q]}
        for q in holdout_ids
    }

    extra_specs = [
        ("aiteamvn_ft", "results/from_drive/aiteamvn_ft_cv.pkl", "results/from_drive/aiteamvn_ft_public.pkl"),
        ("jina_ft", "results/from_drive/jina_ft_cv.pkl", "results/from_drive/jina_ft_public.pkl"),
        ("title_embed", "results/burst_fresh_block/title_embed_scores.pkl", "results/burst_fresh_block/title_embed_public.pkl"),
    ]
    pub_extra = {}
    for name, cv_p, pub_p in extra_specs:
        raw_cv = pickle.loads((ROOT / cv_p).read_bytes())
        if isinstance(raw_cv, dict) and isinstance(raw_cv.get("scores"), dict):
            raw_cv = raw_cv["scores"]
        fl = min(v for q in raw_cv for v in raw_cv[q].values())
        training_scores[name] = {q: {d: raw_cv.get(q, {}).get(d, fl) for d in holdout_candidates[q]} for q in holdout_ids}

        raw_pub = pickle.loads((ROOT / pub_p).read_bytes())
        if isinstance(raw_pub, dict) and isinstance(raw_pub.get("scores"), dict):
            raw_pub = raw_pub["scores"]
        pub_extra[name] = raw_pub

    # Load CAL E5 scores
    cal_e5 = pickle.loads((RESULTS_DIR / "CAL600_VIETLEGAL_E5_SCORES.pkl").read_bytes())
    cal_frozen_scores = cal_e5["frozen_scores"]
    cal_adapted_scores = cal_e5["adapted_scores"]
    cal_frozen_orders = cal_e5["frozen_orders"]
    cal_adapted_orders = cal_e5["adapted_orders"]

    # Doctype & Citation for CAL
    holdout_types = build_type_table(ROOT, documents, holdout_ids, holdout_candidates)
    t_rows_cal = type_features(holdout_candidates, holdout_types, queries, holdout_ids)
    holdout_own, holdout_cited = build_citation_table(documents, holdout_ids, holdout_candidates)
    c_rows_cal = citation_features(holdout_candidates, holdout_own, holdout_cited, holdout_ids)

    # CAL H0 rows
    rows_cal_base_h0, groups_cal = ltr_features(holdout_views, names_h0, holdout_candidates, holdout_ids, training_scores)
    rows_cal_h0 = {q: np.concatenate([rows_cal_base_h0[q], t_rows_cal[q], c_rows_cal[q]], axis=1) for q in holdout_ids}

    # CAL H3 rows
    names_h3 = names_h0 + ["adapted_e5"]
    views_cal_h3 = dict(holdout_views)
    views_cal_h3["adapted_e5"] = cal_adapted_orders
    scores_cal_h3 = dict(training_scores)
    scores_cal_h3["adapted_e5"] = cal_adapted_scores

    rows_cal_base_h3, _ = ltr_features(views_cal_h3, names_h3, holdout_candidates, holdout_ids, scores_cal_h3)
    rows_cal_h2 = {q: np.concatenate([rows_cal_base_h3[q], t_rows_cal[q], c_rows_cal[q]], axis=1) for q in holdout_ids}

    rows_cal_h3 = {}
    for q in holdout_ids:
        docs = holdout_candidates[q]
        n_cands = len(docs)
        ad_z, ad_gap = compute_standardized_scores(cal_adapted_scores[q], docs)
        fr_z, fr_gap = compute_standardized_scores(cal_frozen_scores[q], docs)
        r_ad = {d: i + 1 for i, d in enumerate(cal_adapted_orders[q])}
        r_fr = {d: i + 1 for i, d in enumerate(cal_frozen_orders[q])}

        d_rows = []
        for i, d in enumerate(docs):
            has_ad = d in cal_adapted_scores[q] and not (math.isnan(cal_adapted_scores[q][d]) or np.isnan(cal_adapted_scores[q][d]))
            has_fr = d in cal_frozen_scores[q] and not (math.isnan(cal_frozen_scores[q][d]) or np.isnan(cal_frozen_scores[q][d]))
            if has_ad and has_fr:
                d_z = float(ad_z[i] - fr_z[i])
                d_gap = float(ad_gap[i] - fr_gap[i])
                ra = r_ad.get(d, 60)
                rf = r_fr.get(d, 60)
                rg = float(rf - ra) / float(n_cands)
                rrg = 1.0 / (10.0 + ra) - 1.0 / (10.0 + rf)
                prom = 1.0 if (ra <= 5 and rf > 5) else 0.0
                dem = 1.0 if (rf <= 5 and ra > 5) else 0.0
            else:
                d_z = d_gap = rg = rrg = prom = dem = 0.0
            d_rows.append([d_z, d_gap, rg, rrg, prom, dem])
        rows_cal_h3[q] = np.concatenate([rows_cal_h2[q], np.asarray(d_rows, dtype=np.float32)], axis=1)

    # --- Public Test Setup ---
    print("Building Public test data...", flush=True)
    retrieval = load_public_retrieval(ROOT, DummyArgs, doc_ids, train, public, public_ids)
    base = pickle.loads((ROOT / "results/burst_gpu_threeview/cpu_top20.pkl").read_bytes())["rankings"]
    raw = {q: raw_union(retrieval[q], EXPANSION_CONFIG["depth"]) for q in public_ids}
    retrieval.clear()
    expansion_scores = dense_expansion(ROOT, DummyArgs.cache_dir, public, public_ids, raw, documents, "cuda")

    dense_rank = {q: sorted(raw[q], key=lambda d: (-expansion_scores[q][d], d)) for q in public_ids}
    expanded = weighted_rrf([raw, dense_rank], RERANK_CONFIG["expansion_weights"], RERANK_CONFIG["expansion_rrf_k"])
    corpus_rank, corpus_score = corpus_dense(ROOT, DummyArgs.cache_dir, public, public_ids, "cuda", cap=CORPUS_CAP, depth=CORPUS_DEPTH)

    public_candidates = {
        q: list(dict.fromkeys(
            list(base[q]) + expanded[q][:RERANK_CONFIG["expanded_depth"]] + corpus_rank[q][:CORPUS_DEPTH]
        ))
        for q in public_ids
    }
    corpus_score = {q: {d: corpus_score[q][d] for d in public_candidates[q] if d in corpus_score[q]} for q in public_ids}
    expansion_scores = {q: {d: expansion_scores[q][d] for d in public_candidates[q] if d in expansion_scores[q]} for q in public_ids}
    expanded = {q: expanded[q][:RERANK_CONFIG["expanded_depth"]] for q in public_ids}

    scores = rerank(ROOT, DummyArgs.cache_dir, public, public_ids, public_candidates, documents, "cuda")
    vnlegal_public_scores = pickle.loads((ROOT / "results/burst_userft_maxrecall/vnlegal_scores.pkl").read_bytes())

    view_rank_pub = {
        "base": {q: list(base[q]) for q in public_ids},
        "expanded": expanded,
        "jina": {q: sorted(public_candidates[q], key=lambda d: (-scores["jina"][q][d], d)) for q in public_ids},
        "dense": {q: sorted(public_candidates[q], key=lambda d: (-scores["dense"][q][d], d)) for q in public_ids},
        "corpus": {q: sorted((d for d in public_candidates[q] if d in corpus_score[q]), key=lambda d: (-corpus_score[q][d], d)) for q in public_ids},
        "vnlegal_lal": {q: sorted(public_candidates[q], key=lambda d: (-vnlegal_public_scores.get(q, {}).get(d, -1e9), d)) for q in public_ids},
    }

    three_view = pickle.loads((ROOT / "results/burst_gpu_threeview/gpu_scores.checkpoint.pkl").read_bytes())["scores"]
    public_ce = pickle.loads((ROOT / "results/crossenc_fullpool/public_scores.pkl").read_bytes())["scores"]

    public_scores_h0 = {
        "jina": scores["jina"],
        "dense": scores["dense"],
        "expansion": expansion_scores,
        "e5": {q: three_view[q]["e5"] for q in public_ids},
        "corpus": {q: {d: corpus_score[q].get(d, -1.0) for d in public_candidates[q]} for q in public_ids},
        "vnlegal_lal": vnlegal_public_scores,
        "crossenc": {q: {d: public_ce.get(q, {}).get(d, floor_ce) for d in public_candidates[q]} for q in public_ids},
    }
    for name in pub_extra:
        raw_p = pub_extra[name]
        fl_p = min(v for q in raw_p for v in raw_p[q].values())
        public_scores_h0[name] = {q: {d: raw_p.get(q, {}).get(d, fl_p) for d in public_candidates[q]} for q in public_ids}

    # Public doctype & citation
    public_queries = {q: (public[q], set()) for q in public_ids}
    public_types = build_type_table(ROOT, documents, public_ids, public_candidates)
    public_t_rows = type_features(public_candidates, public_types, public_queries, public_ids)
    public_own, public_cited = build_citation_table(documents, public_ids, public_candidates)
    public_c_rows = citation_features(public_candidates, public_own, public_cited, public_ids)

    # Public H0 rows
    public_rows_base_h0, public_groups = ltr_features(view_rank_pub, names_h0, public_candidates, public_ids, public_scores_h0)
    public_rows_h0 = {q: np.concatenate([public_rows_base_h0[q], public_t_rows[q], public_c_rows[q]], axis=1) for q in public_ids}

    # Public H3 rows
    pub_e5 = pickle.loads((RESULTS_DIR / "PUBLIC_VIETLEGAL_E5_SCORES.pkl").read_bytes())
    pub_frozen_scores = pub_e5["frozen_scores"]
    pub_adapted_scores = pub_e5["adapted_scores"]
    pub_frozen_orders = pub_e5["frozen_orders"]
    pub_adapted_orders = pub_e5["adapted_orders"]

    view_rank_pub_h3 = dict(view_rank_pub)
    view_rank_pub_h3["adapted_e5"] = pub_adapted_orders
    public_scores_h3 = dict(public_scores_h0)
    public_scores_h3["adapted_e5"] = pub_adapted_scores

    public_rows_base_h3, _ = ltr_features(view_rank_pub_h3, names_h3, public_candidates, public_ids, public_scores_h3)
    public_rows_h2 = {q: np.concatenate([public_rows_base_h3[q], public_t_rows[q], public_c_rows[q]], axis=1) for q in public_ids}

    public_rows_h3 = {}
    for q in public_ids:
        docs = public_candidates[q]
        n_cands = len(docs)
        ad_z, ad_gap = compute_standardized_scores(pub_adapted_scores[q], docs)
        fr_z, fr_gap = compute_standardized_scores(pub_frozen_scores[q], docs)
        r_ad = {d: i + 1 for i, d in enumerate(pub_adapted_orders[q])}
        r_fr = {d: i + 1 for i, d in enumerate(pub_frozen_orders[q])}

        d_rows = []
        for i, d in enumerate(docs):
            has_ad = d in pub_adapted_scores[q] and not (math.isnan(pub_adapted_scores[q][d]) or np.isnan(pub_adapted_scores[q][d]))
            has_fr = d in pub_frozen_scores[q] and not (math.isnan(pub_frozen_scores[q][d]) or np.isnan(pub_frozen_scores[q][d]))
            if has_ad and has_fr:
                d_z = float(ad_z[i] - fr_z[i])
                d_gap = float(ad_gap[i] - fr_gap[i])
                ra = r_ad.get(d, 60)
                rf = r_fr.get(d, 60)
                rg = float(rf - ra) / float(n_cands)
                rrg = 1.0 / (10.0 + ra) - 1.0 / (10.0 + rf)
                prom = 1.0 if (ra <= 5 and rf > 5) else 0.0
                dem = 1.0 if (rf <= 5 and ra > 5) else 0.0
            else:
                d_z = d_gap = rg = rrg = prom = dem = 0.0
            d_rows.append([d_z, d_gap, rg, rrg, prom, dem])
        public_rows_h3[q] = np.concatenate([public_rows_h2[q], np.asarray(d_rows, dtype=np.float32)], axis=1)

    return (
        holdout_ids,
        queries,
        groups_cal,
        rows_cal_h0,
        rows_cal_h3,
        public_ids,
        public_groups,
        public_candidates,
        public_rows_h0,
        public_rows_h3,
        valid_docs,
        doc_ids,
    )


def fit_and_predict(
    rows_train: Dict[str, np.ndarray],
    train_ids: List[str],
    train_groups: Dict[str, List[str]],
    queries: Dict[str, Tuple[str, Set[str]]],
    rows_test: Dict[str, np.ndarray],
    test_ids: List[str],
    test_groups: Dict[str, List[str]],
    test_candidates: Dict[str, List[str]],
    valid_docs: Set[str],
    all_doc_ids: List[str],
    ltr_c: float = 0.15,
) -> Dict[str, Dict[str, List[str]]]:
    x_train = np.vstack([rows_train[q] for q in train_ids])
    y_train = np.concatenate([[d in queries[q][1] for d in train_groups[q]] for q in train_ids]).astype(np.int8)

    scaler = StandardScaler().fit(x_train)
    model = LogisticRegression(
        C=ltr_c,
        class_weight="balanced",
        solver="liblinear",
        max_iter=3000,
        random_state=2026,
    )
    model.fit(scaler.transform(x_train), y_train)

    predictions = {}
    for q in test_ids:
        proba = model.predict_proba(scaler.transform(rows_test[q]))[:, 1]
        order = np.argsort(-proba)
        fused = [test_groups[q][i] for i in order]

        final = [d for d in fused if d in valid_docs][:5]
        for pool in (test_candidates[q], all_doc_ids):
            for doc in pool:
                if len(final) >= 5:
                    break
                if doc not in final and doc in valid_docs:
                    final.append(doc)
        final = final[:5]
        predictions[q] = {"answer": final}

    return predictions


def verify_submission_structure(sub: Dict[str, Dict[str, List[str]]], valid_docs: Set[str], expected_qids: List[str]) -> bool:
    if len(sub) != len(expected_qids):
        raise ValueError(f"Expected {len(expected_qids)} queries, got {len(sub)}")
    for q in expected_qids:
        if q not in sub:
            raise ValueError(f"Missing query {q}")
        ans = sub[q]["answer"]
        if len(ans) != 5:
            raise ValueError(f"Query {q} has {len(ans)} answers instead of 5")
        if len(set(ans)) != 5:
            raise ValueError(f"Query {q} has duplicate answers: {ans}")
        for d in ans:
            if d not in valid_docs:
                raise ValueError(f"Query {q} has invalid doc_id {d}")
    return True


def make_zip(json_path: Path, zip_path: Path):
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as arc:
        arc.write(json_path, arcname="submission.json")


def main():
    started_all = time.perf_counter()
    print("=== Step 5 & 6: Fitting Public LTR and Packaging Submissions ===", flush=True)

    (
        holdout_ids,
        queries,
        groups_cal,
        rows_cal_h0,
        rows_cal_h3,
        public_ids,
        public_groups,
        public_candidates,
        public_rows_h0,
        public_rows_h3,
        valid_docs,
        doc_ids,
    ) = build_h0_and_h3_matrices()

    print(f"H0 feature matrix: CAL={rows_cal_h0[holdout_ids[0]].shape[1]} dims, Public={public_rows_h0[public_ids[0]].shape[1]} dims")
    print(f"H3 feature matrix: CAL={rows_cal_h3[holdout_ids[0]].shape[1]} dims, Public={public_rows_h3[public_ids[0]].shape[1]} dims")

    # 1. Fit H0 (Current Production Baseline)
    print("\nFitting H0 model on all 600 CAL queries...", flush=True)
    preds_h0 = fit_and_predict(
        rows_cal_h0, holdout_ids, groups_cal, queries,
        public_rows_h0, public_ids, public_groups, public_candidates,
        valid_docs, doc_ids, ltr_c=0.15,
    )
    verify_submission_structure(preds_h0, valid_docs, public_ids)

    # 2. Fit H3 (Best Experimental Candidate)
    print("Fitting H3 model on all 600 CAL queries...", flush=True)
    preds_h3 = fit_and_predict(
        rows_cal_h3, holdout_ids, groups_cal, queries,
        public_rows_h3, public_ids, public_groups, public_candidates,
        valid_docs, doc_ids, ltr_c=0.15,
    )
    verify_submission_structure(preds_h3, valid_docs, public_ids)

    # 3. Decision check per rules:
    # Arms H1, H2, H3 all regressed vs H0 on CAL pooled and had block regressions.
    # Therefore decision is KILL / NO_PROMOTION.
    # The official submission.zip must be materialized directly from H0 control per Section 17.
    # The H3 predictions are packaged as best_experimental_candidate.zip for forensic analysis.
    import shutil
    sub_json = RESULTS_DIR / "submission.json"
    sub_zip = RESULTS_DIR / "submission.zip"
    shutil.copy2(CONTROL_DIR / "submission.json", sub_json)
    shutil.copy2(CONTROL_DIR / "submission.zip", sub_zip)
    print(f"\nMaterialized official submission from H0 Control: {sub_zip}")

    exp_json = RESULTS_DIR / "best_experimental_candidate.json"
    exp_zip = RESULTS_DIR / "best_experimental_candidate.zip"
    exp_json.write_text(json.dumps(preds_h3, ensure_ascii=False, indent=2), encoding="utf-8")
    make_zip(exp_json, exp_zip)
    print(f"Saved best experimental candidate (H3): {exp_zip}")

    # 4. Compare H3 vs H0 on Public
    control_preds = json.loads((CONTROL_DIR / "submission.json").read_text(encoding="utf-8"))
    changed_queries = 0
    jaccards = []
    slot_changes = 0

    for q in public_ids:
        top_h0 = control_preds[q]["answer"]
        top_h3 = preds_h3[q]["answer"]

        if top_h0 != top_h3:
            changed_queries += 1

        s0, s3 = set(top_h0), set(top_h3)
        jaccards.append(len(s0 & s3) / len(s0 | s3))

        for d0, d3 in zip(top_h0, top_h3):
            if d0 != d3:
                slot_changes += 1

    mean_jaccard = float(np.mean(jaccards))

    print(f"\nPublic Comparison (H3 vs H0):")
    print(f"  Changed queries: {changed_queries} / 1000")
    print(f"  Mean Top-5 Jaccard similarity: {mean_jaccard:.4f}")
    print(f"  Total slot changes: {slot_changes} / 5000 ({slot_changes / 50:.1f}%)")

    # 5. Check bit-for-bit extraction of ZIP
    def verify_zip_extract(zip_p: Path, json_p: Path):
        with zipfile.ZipFile(zip_p, "r") as z:
            names = z.namelist()
            if names != ["submission.json"]:
                raise ValueError(f"ZIP namelist mismatch: {names}")
            extracted_bytes = z.read("submission.json")
            original_bytes = json_p.read_bytes()
            if extracted_bytes != original_bytes:
                raise ValueError("Extracted JSON does not match original byte-for-byte!")

    verify_zip_extract(sub_zip, sub_json)
    verify_zip_extract(exp_zip, exp_json)
    print("ZIP extraction verified byte-for-byte!")

    # Compare with Control Snapshot in control/
    control_json = CONTROL_DIR / "submission.json"
    control_match = False
    if control_json.exists():
        c_hash = sha256_file(control_json)
        sub_hash = sha256_file(sub_json)
        control_match = (c_hash == sub_hash)
        print(f"Control snapshot match: {control_match} (SHA256={sub_hash[:16]}...)")

    # 6. Write PUBLIC_SUBMISSION_AUDIT.json
    audit = {
        "schema_version": "dsc2026.gemini.huy_e5_adaptation_delta_v1.public_submission_audit.v1",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "decision": "KILL_NO_PROMOTION",
        "rationale": "CAL600 LOBO evaluation showed no pooled recall gain for H1, H2, or H3 vs H0, and all experimental arms regressed blocks a and d. Per protocol Section 17, H0 control is materialized as the official submission.",
        "promoted_submission": {
            "source_arm": "H0_BASELINE",
            "json_path": str(sub_json.resolve()),
            "zip_path": str(sub_zip.resolve()),
            "json_sha256": sha256_file(sub_json),
            "json_md5": md5_file(sub_json),
            "zip_sha256": sha256_file(sub_zip),
            "zip_md5": md5_file(sub_zip),
            "matches_preflight_control_byte_exact": control_match,
            "queries": len(preds_h0),
            "answers_per_query": 5,
            "all_answers_unique": True,
            "all_answers_valid_canonical": True,
        },
        "best_experimental_candidate": {
            "source_arm": "H3_ADAPTATION_DELTA",
            "json_path": str(exp_json.resolve()),
            "zip_path": str(exp_zip.resolve()),
            "json_sha256": sha256_file(exp_json),
            "json_md5": md5_file(exp_json),
            "zip_sha256": sha256_file(exp_zip),
            "zip_md5": md5_file(exp_zip),
            "queries": len(preds_h3),
            "answers_per_query": 5,
            "all_answers_unique": True,
            "all_answers_valid_canonical": True,
            "status": "NOT_PROMOTED_FORENSIC_ONLY",
        },
        "comparison_h3_vs_h0_public": {
            "changed_queries": changed_queries,
            "total_queries": len(public_ids),
            "mean_top5_jaccard": mean_jaccard,
            "total_slot_changes": slot_changes,
            "slot_change_percentage": slot_changes / 50.0,
        },
    }

    audit_path = RESULTS_DIR / "PUBLIC_SUBMISSION_AUDIT.json"
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved: {audit_path}", flush=True)


if __name__ == "__main__":
    main()
