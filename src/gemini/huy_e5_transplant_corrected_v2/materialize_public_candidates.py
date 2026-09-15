import sys
import os
import json
import time
import pickle
import zipfile
import hashlib
from datetime import datetime, timezone
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

from run_burst_expanded_fusion_submission import (
    EXPANSION_CONFIG, RERANK_CONFIG, dense_expansion, corpus_dense, rerank, weighted_rrf, raw_union,
    load_public_retrieval, DocumentStore, CORPUS_CAP, CORPUS_DEPTH
)
from run_burst_multistage_submission import load_metadata
from tune_citation_graph import build_citation_table, citation_features
from tune_corpus_cap32_fusion import build_training_cap
from tune_doctype_features import build_type_table, type_features
from tune_expanded_fusion_selection import ltr_features

RESULTS_DIR = REPO_ROOT / "results" / "gemini" / "huy_e5_transplant_corrected_v2"
CONTROL_JSON_PATH = REPO_ROOT / "results" / "burst_userft_maxrecall" / "submission.json"

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

def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()

def compute_md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()

def load_pkl(rel_path: str):
    p = REPO_ROOT / rel_path
    obj = pickle.loads(p.read_bytes())
    if isinstance(obj, dict) and isinstance(obj.get("scores"), dict):
        return obj["scores"]
    return obj

def package_and_verify_zip(json_path: Path, zip_path: Path, valid_docs: Set[str], public_ids: List[str]) -> Dict[str, Any]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Verification checks
    assert len(data) == 1000, f"Expected 1000 queries, got {len(data)}"
    assert set(data.keys()) == set(public_ids), "Query ID set mismatch"
    for q, item in data.items():
        ans = item["answer"]
        assert len(ans) == 5, f"Query {q} does not have 5 answers"
        assert len(set(ans)) == 5, f"Query {q} has duplicates"
        assert all(d in valid_docs for d in ans), f"Query {q} contains invalid docs"

    # Create zip containing exactly submission.json
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.write(json_path, arcname="submission.json")

    # Verify extracted JSON matches byte-for-byte
    with zipfile.ZipFile(zip_path, "r") as z:
        names = z.namelist()
        assert names == ["submission.json"], f"Unexpected zip namelist: {names}"
        extracted_bytes = z.read("submission.json")
        source_bytes = json_path.read_bytes()
        assert extracted_bytes == source_bytes, "Extracted bytes do not match source bytes"

    return {
        "json_path": str(json_path),
        "zip_path": str(zip_path),
        "json_sha256": compute_sha256(json_path),
        "json_md5": compute_md5(json_path),
        "zip_sha256": compute_sha256(zip_path),
        "zip_md5": compute_md5(zip_path),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "query_count": len(data),
        "all_checks_passed": True,
    }

def main():
    print("=== PUBLIC MATERIALIZATION & CHURN AUDIT ===", flush=True)

    # 1. Load document store and metadata
    paths_meta, doc_ids_meta, train_meta, public_meta = load_metadata(REPO_ROOT / "DSC2026-LegalIR-main/v4_run/public_test_dataset")
    public_ids = list(public_meta)
    valid_docs = set(doc_ids_meta)
    docs = DocumentStore(paths_meta)

    # 2. Load CAL600 data
    queries_cal, blocks_cal, all_ids_cal, extended_cal, local_cal, base_scores_cal = build_training_cap(
        REPO_ROOT, 32, "results/corpus_index/holdout_extended_scores_cap32.pkl", depth=20
    )
    gold_cal = {q: queries_cal[q][1] for q in all_ids_cal}

    def load_aligned_cv(rel_path: str, floor=None):
        obj = load_pkl(rel_path)
        fl = floor if floor is not None else min(v for q in obj for v in obj[q].values())
        return {q: {d: obj.get(q, {}).get(d, fl) for d in extended_cal[q]} for q in all_ids_cal}

    vnlegal_cv = load_pkl("results/embedding_finetune/vnlegal_lal_cv_scores.pkl")
    crossenc_cv = load_aligned_cv("results/crossenc_fullpool/cv_scores.pkl", -11.5)
    extra_cv = {name: load_aligned_cv(rel) for name, rel in EXTRA_CV_PATHS.items()}

    # CAL score channels
    channels_cal_e0 = {
        **base_scores_cal,
        "vnlegal_lal": vnlegal_cv,
        "crossenc": crossenc_cv,
        **extra_cv,
    }

    # Load corrected CAL and Public E5 scores
    cal_e5 = pickle.loads((RESULTS_DIR / "CAL600_CORRECTED_VIETLEGAL_E5_SCORES.pkl").read_bytes())
    pub_e5 = pickle.loads((RESULTS_DIR / "PUBLIC_CORRECTED_VIETLEGAL_E5_SCORES.pkl").read_bytes())

    cal_adapted_e5_scores = cal_e5["adapted_scores"]
    cal_adapted_e5_orders = cal_e5["adapted_orders"]

    pub_adapted_e5_scores = pub_e5["adapted_scores"]
    pub_adapted_e5_orders = pub_e5["adapted_orders"]

    # CAL Metadata features
    type_table_cal = build_type_table(REPO_ROOT, docs, all_ids_cal, extended_cal)
    type_rows_cal = type_features(extended_cal, type_table_cal, queries_cal, all_ids_cal)
    own_cal, cited_cal = build_citation_table(docs, all_ids_cal, extended_cal)
    cite_rows_cal = citation_features(extended_cal, own_cal, cited_cal, all_ids_cal)
    meta_cal = {q: np.concatenate([type_rows_cal[q], cite_rows_cal[q]], axis=1) for q in all_ids_cal}

    # CAL Views (Deployment contract has 6 views)
    views_cal_dep = dict(local_cal)
    views_cal_dep["vnlegal_lal"] = {
        q: sorted(extended_cal[q], key=lambda d: (-vnlegal_cv.get(q, {}).get(d, -1e9), d))
        for q in all_ids_cal
    }
    view_names_dep = ["base", "expanded", "jina", "dense", "corpus", "vnlegal_lal"]

    views_cal_e3 = dict(views_cal_dep)
    views_cal_e3["adapted_vietlegal_e5"] = cal_adapted_e5_orders
    view_names_e3 = view_names_dep + ["adapted_vietlegal_e5"]

    # 3. Load Public Candidates and Views
    base_pub = load_pkl("results/burst_gpu_threeview/cpu_top20.pkl")["rankings"]
    class DummyArgs:
        db = REPO_ROOT / "benchmarks/legalir_full_fts.sqlite"
        workers = 4
        cache_dir = REPO_ROOT / "results/burst_expanded_fusion"

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
    view_rank_pub_e3 = dict(view_rank_pub)
    view_rank_pub_e3["adapted_vietlegal_e5"] = pub_adapted_e5_orders

    three_view_pub = load_pkl("results/burst_gpu_threeview/gpu_scores.checkpoint.pkl")
    crossenc_pub = load_pkl("results/crossenc_fullpool/public_scores.pkl")
    floor_ce_pub = min(min(v.values()) for v in crossenc_cv.values() if v)

    public_scores_e0 = {
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
        public_scores_e0[name] = {q: {d: raw_p.get(q, {}).get(d, fl_p) for d in public_candidates[q]} for q in public_ids}

    # Public Metadata features
    public_queries_meta = {q: (public_meta[q], set()) for q in public_ids}
    public_types = build_type_table(REPO_ROOT, docs, public_ids, public_candidates)
    public_t_rows = type_features(public_candidates, public_types, public_queries_meta, public_ids)
    public_own, public_cited = build_citation_table(docs, public_ids, public_candidates)
    public_c_rows = citation_features(public_candidates, public_own, public_cited, public_ids)
    meta_pub = {q: np.concatenate([public_t_rows[q], public_c_rows[q]], axis=1) for q in public_ids}

    # Channels setup for arms
    # E0: Baseline channels
    channels_pub_e0 = public_scores_e0

    # E1: REPLACE e5 with adapted_vietlegal_e5
    channels_cal_e1 = {k: v for k, v in channels_cal_e0.items() if k != "e5"}
    channels_cal_e1["adapted_vietlegal_e5"] = cal_adapted_e5_scores
    channels_pub_e1 = {k: v for k, v in channels_pub_e0.items() if k != "e5"}
    channels_pub_e1["adapted_vietlegal_e5"] = pub_adapted_e5_scores

    # E2: AUGMENT SCORE ONLY
    channels_cal_e2 = dict(channels_cal_e0)
    channels_cal_e2["adapted_vietlegal_e5"] = cal_adapted_e5_scores
    channels_pub_e2 = dict(channels_pub_e0)
    channels_pub_e2["adapted_vietlegal_e5"] = pub_adapted_e5_scores

    # E3: AUGMENT SCORE PLUS RANK
    channels_cal_e3 = dict(channels_cal_e0)
    channels_cal_e3["adapted_vietlegal_e5"] = cal_adapted_e5_scores
    channels_pub_e3 = dict(channels_pub_e0)
    channels_pub_e3["adapted_vietlegal_e5"] = pub_adapted_e5_scores

    arms = [
        ("E0", "CONTROL_H0", views_cal_dep, view_names_dep, channels_cal_e0, view_rank_pub, view_names_dep, channels_pub_e0),
        ("E1", "CANDIDATE_E1_REPLACE_E5", views_cal_dep, view_names_dep, channels_cal_e1, view_rank_pub, view_names_dep, channels_pub_e1),
        ("E2", "CANDIDATE_E2_AUGMENT_SCORE", views_cal_dep, view_names_dep, channels_cal_e2, view_rank_pub, view_names_dep, channels_pub_e2),
        ("E3", "CANDIDATE_E3_AUGMENT_SCORE_RANK", views_cal_e3, view_names_e3, channels_cal_e3, view_rank_pub_e3, view_names_e3, channels_pub_e3),
    ]

    predictions_by_arm = {}

    for arm_id, pkg_name, v_cal, vn_cal, ch_cal, v_pub, vn_pub, ch_pub in arms:
        print(f"\nProcessing {arm_id} ({pkg_name})...", flush=True)
        # Compute training rows
        r_cal, g_cal = ltr_features(v_cal, vn_cal, extended_cal, all_ids_cal, ch_cal)
        X_cal = np.vstack([np.concatenate([r_cal[q], meta_cal[q]], axis=1) for q in all_ids_cal])
        y_cal = np.concatenate([[d in gold_cal[q] for d in g_cal[q]] for q in all_ids_cal]).astype(np.int8)

        # Fit model
        scaler = StandardScaler().fit(X_cal)
        model = LogisticRegression(C=0.15, class_weight="balanced", solver="liblinear", max_iter=3000, random_state=2026)
        model.fit(scaler.transform(X_cal), y_cal)

        # Compute public rows
        r_pub, g_pub = ltr_features(v_pub, vn_pub, public_candidates, public_ids, ch_pub)
        X_pub = {q: np.concatenate([r_pub[q], meta_pub[q]], axis=1) for q in public_ids}

        # Predict
        arm_preds = {}
        for q in public_ids:
            proba = model.predict_proba(scaler.transform(X_pub[q]))[:, 1]
            order = np.argsort(-proba)
            fused = [g_pub[q][i] for i in order]

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
            arm_preds[q] = {"answer": final}

        predictions_by_arm[arm_id] = arm_preds

        # Save standalone JSON and package ZIP
        json_file = RESULTS_DIR / f"{pkg_name}.json"
        zip_file = RESULTS_DIR / f"{pkg_name}.zip"

        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(arm_preds, f, indent=2, ensure_ascii=False)

        audit_pkg = package_and_verify_zip(json_file, zip_file, valid_docs, public_ids)
        print(f"  Packaged {zip_file.name} (SHA256: {audit_pkg['zip_sha256']})")

    # 4. Verify E0 matches control byte-exact
    e0_json = RESULTS_DIR / "CONTROL_H0.json"
    control_bytes = CONTROL_JSON_PATH.read_bytes()
    e0_bytes = e0_json.read_bytes()
    e0_matches_control = (control_bytes == e0_bytes)
    print(f"\nCONTROL_H0 byte-exact matches burst_userft_maxrecall/submission.json: {e0_matches_control}")

    # 5. Public Churn Audit (Section 14)
    print("\nComputing public candidate churn audit vs H0...", flush=True)
    h0_preds = predictions_by_arm["E0"]
    churn_audit = {}

    for arm_id, pkg_name, _, _, _, _, _, _ in arms[1:]:
        cand_preds = predictions_by_arm[arm_id]

        changed_set_queries = 0
        changed_order_queries = 0
        jaccards = []
        docs_entering = 0
        docs_leaving = 0
        rank5_boundary_changes = 0

        for q in public_ids:
            ans_h0 = h0_preds[q]["answer"]
            ans_c = cand_preds[q]["answer"]

            set_h0 = set(ans_h0)
            set_c = set(ans_c)

            if ans_h0 != ans_c:
                changed_order_queries += 1
            if set_h0 != set_c:
                changed_set_queries += 1

            jacc = len(set_h0 & set_c) / len(set_h0 | set_c)
            jaccards.append(jacc)

            docs_entering += len(set_c - set_h0)
            docs_leaving += len(set_h0 - set_c)

            if ans_h0[4] != ans_c[4]:
                rank5_boundary_changes += 1

        churn_audit[arm_id] = {
            "package_name": pkg_name,
            "total_queries": 1000,
            "changed_top5_set_queries": changed_set_queries,
            "changed_top5_set_pct": changed_set_queries / 10.0,
            "changed_order_queries": changed_order_queries,
            "changed_order_pct": changed_order_queries / 10.0,
            "mean_top5_jaccard": float(np.mean(jaccards)),
            "docs_entering_top5": docs_entering,
            "docs_leaving_top5": docs_leaving,
            "rank5_boundary_changes": rank5_boundary_changes,
        }
        print(f"{arm_id} ({pkg_name}):")
        print(f"  Changed Top-5 Set: {changed_set_queries}/1000 ({changed_set_queries/10:.1f}%)")
        print(f"  Changed Order:    {changed_order_queries}/1000 ({changed_order_queries/10:.1f}%)")
        print(f"  Mean Jaccard:     {np.mean(jaccards):.4f}")
        print(f"  Entering/Leaving: +{docs_entering} / -{docs_leaving}")

    # Compile packages summary
    packages_summary = {}
    for arm_id, pkg_name, _, _, _, _, _, _ in arms:
        zf = RESULTS_DIR / f"{pkg_name}.zip"
        jf = RESULTS_DIR / f"{pkg_name}.json"
        packages_summary[pkg_name] = {
            "arm": arm_id,
            "json_path": str(jf),
            "zip_path": str(zf),
            "json_sha256": compute_sha256(jf),
            "json_md5": compute_md5(jf),
            "zip_sha256": compute_sha256(zf),
            "zip_md5": compute_md5(zf),
        }

    public_audit_report = {
        "schema_version": "dsc2026.gemini.huy_e5_transplant_corrected_v2.public_candidate_audit.v1",
        "h0_control_matches_burst_userft_maxrecall": e0_matches_control,
        "packages": packages_summary,
        "churn_vs_h0": churn_audit,
    }

    out_audit = RESULTS_DIR / "PUBLIC_CANDIDATE_AUDIT.json"
    with open(out_audit, "w", encoding="utf-8") as f:
        json.dump(public_audit_report, f, indent=2, ensure_ascii=False)
    print(f"\nWrote public candidate audit to {out_audit}", flush=True)

if __name__ == "__main__":
    main()
