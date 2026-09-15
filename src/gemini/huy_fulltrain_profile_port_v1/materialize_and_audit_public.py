"""Train final LTR on CAL600 and materialize Public test packages for P0 and P1."""

from __future__ import annotations

import hashlib
import json
import pickle
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[3]
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


def validate_submission_zip(zip_path: Path, json_path: Path, valid_docs: set[str]):
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


def compute_profile_shift(
    cal_ids: list[str],
    extended: dict[str, list[str]],
    cal_oof_rankings: dict[str, list[str]],
    cal_lexical_support: dict[str, int],
    public_ids: list[str],
    public_candidates: dict[str, list[str]],
    public_rankings: dict[str, list[str]],
    public_lexical_support: dict[str, int],
):
    def get_stats(ids, candidate_dict, ranking_dict):
        total_slots = 0
        r_le5 = 0
        r_le10 = 0
        r_le20 = 0
        r_fallback = 0
        rr_vals = []
        norm_r_vals = []

        for q in ids:
            cand_docs = candidate_dict[q]
            ranked_list = ranking_dict[q]
            rank_map = {d: i + 1 for i, d in enumerate(ranked_list)}
            for d in cand_docs:
                r = rank_map.get(d, 60)
                total_slots += 1
                if r <= 5:
                    r_le5 += 1
                if r <= 10:
                    r_le10 += 1
                if r <= 20:
                    r_le20 += 1
                if r == 60:
                    r_fallback += 1
                rr_vals.append(1.0 / (10.0 + r))
                norm_r_vals.append(r / 60.0)

        return {
            "total_candidate_slots": total_slots,
            "pct_rank_le_5": (r_le5 / total_slots) * 100.0,
            "pct_rank_le_10": (r_le10 / total_slots) * 100.0,
            "pct_rank_le_20": (r_le20 / total_slots) * 100.0,
            "pct_fallback_60": (r_fallback / total_slots) * 100.0,
            "mean_reciprocal_rank": float(np.mean(rr_vals)),
            "mean_normalized_rank": float(np.mean(norm_r_vals)),
        }

    cal_stats = get_stats(cal_ids, extended, cal_oof_rankings)
    pub_stats = get_stats(public_ids, public_candidates, public_rankings)

    cal_supp = [cal_lexical_support[q] for q in cal_ids]
    pub_supp = [public_lexical_support[q] for q in public_ids]

    cal_stats["lexical_support"] = {
        "mean": float(np.mean(cal_supp)),
        "median": float(np.median(cal_supp)),
        "q25": float(np.percentile(cal_supp, 25)),
        "q75": float(np.percentile(cal_supp, 75)),
    }
    pub_stats["lexical_support"] = {
        "mean": float(np.mean(pub_supp)),
        "median": float(np.median(pub_supp)),
        "q25": float(np.percentile(pub_supp, 25)),
        "q75": float(np.percentile(pub_supp, 75)),
    }

    return {
        "schema_version": "dsc2026.gemini.huy_fulltrain_profile_port_v1.train_deploy_profile_shift.v1",
        "cal_training_oof": cal_stats,
        "public_deployment_full": pub_stats,
        "relative_shift": {
            "delta_mean_reciprocal_rank": pub_stats["mean_reciprocal_rank"] - cal_stats["mean_reciprocal_rank"],
            "delta_pct_fallback": pub_stats["pct_fallback_60"] - cal_stats["pct_fallback_60"],
            "delta_mean_lexical_support": pub_stats["lexical_support"]["mean"] - cal_stats["lexical_support"]["mean"],
        }
    }


def main():
    started = time.perf_counter()
    out_dir = ROOT / "results" / "gemini" / "huy_fulltrain_profile_port_v1"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load document store and CAL structures
    print("Loading DocumentStore and CAL inputs...", flush=True)
    docs = DocumentStore(sorted((ROOT / "DSC2026-LegalIR-main/v4_run/public_test_dataset/selected-contexts").glob("context_*.json")))
    queries, blocks, all_ids, extended, local_views, base_scores = build_training_cap(
        ROOT, 32, "results/corpus_index/holdout_extended_scores_cap32.pkl", depth=20
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

    paths_meta, doc_ids_meta, train_meta, public_meta = load_metadata(ROOT / "DSC2026-LegalIR-main/v4_run/public_test_dataset")
    public_ids = list(public_meta)
    valid_docs = set(doc_ids_meta)

    print("Loading public candidate pool and retrieval views...", flush=True)
    base_pub = load_pkl("results/burst_gpu_threeview/cpu_top20.pkl")["rankings"]
    pub_retrieval = load_public_retrieval(ROOT, DummyArgs, doc_ids_meta, train_meta, public_meta, public_ids)
    raw_pub = {q: raw_union(pub_retrieval[q], EXPANSION_CONFIG["depth"]) for q in public_ids}
    pub_retrieval.clear()

    expansion_scores_pub = dense_expansion(ROOT, DummyArgs.cache_dir, public_meta, public_ids, raw_pub, docs, "cuda")
    dense_rank_pub = {q: sorted(raw_pub[q], key=lambda d: (-expansion_scores_pub[q][d], d)) for q in public_ids}
    expanded_pub = weighted_rrf([raw_pub, dense_rank_pub], RERANK_CONFIG["expansion_weights"], RERANK_CONFIG["expansion_rrf_k"])
    corpus_rank_pub, corpus_score_pub = corpus_dense(ROOT, DummyArgs.cache_dir, public_meta, public_ids, "cuda", cap=CORPUS_CAP, depth=CORPUS_DEPTH)

    public_candidates = {
        q: list(dict.fromkeys(
            list(base_pub[q]) + expanded_pub[q][:RERANK_CONFIG["expanded_depth"]] + corpus_rank_pub[q][:CORPUS_DEPTH]
        ))
        for q in public_ids
    }
    corpus_score_pub = {q: {d: corpus_score_pub[q][d] for d in public_candidates[q] if d in corpus_score_pub[q]} for q in public_ids}
    expansion_scores_pub = {q: {d: expansion_scores_pub[q][d] for d in public_candidates[q] if d in expansion_scores_pub[q]} for q in public_ids}
    expanded_pub = {q: expanded_pub[q][:RERANK_CONFIG["expanded_depth"]] for q in public_ids}

    rerank_scores_pub = rerank(ROOT, DummyArgs.cache_dir, public_meta, public_ids, public_candidates, docs, "cuda")
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
    public_types = build_type_table(ROOT, docs, public_ids, public_candidates)
    public_t_rows = type_features(public_candidates, public_types, public_queries_meta, public_ids)
    public_own, public_cited = build_citation_table(docs, public_ids, public_candidates)
    public_c_rows = citation_features(public_candidates, public_own, public_cited, public_ids)

    # 3. Load Profile BM25 rankings cache
    cache_path = out_dir / "PROFILE_BM25_RANKINGS.pkl"
    profile_cache = pickle.loads(cache_path.read_bytes())
    cal_oof_rankings = profile_cache["cal_oof_rankings"]
    cal_lexical_support = profile_cache["cal_lexical_support"]
    public_rankings = profile_cache["public_rankings"]
    public_lexical_support = profile_cache["public_lexical_support"]

    # 4. Train / Deploy Shift Analysis (Section 18)
    shift_report = compute_profile_shift(
        all_ids,
        extended,
        cal_oof_rankings,
        cal_lexical_support,
        public_ids,
        public_candidates,
        public_rankings,
        public_lexical_support,
    )
    shift_file = out_dir / "TRAIN_DEPLOY_PROFILE_SHIFT.json"
    with shift_file.open("w", encoding="utf-8") as f:
        json.dump(shift_report, f, indent=2)
    print(f"Wrote {shift_file}")

    # 5. Fit Production H0 Model & Generate CONTROL_H0 (Section 8 & 19)
    print("Generating H0 public predictions...", flush=True)
    dep_local_views = dict(local_views)
    dep_local_views["vnlegal_lal"] = {
        q: sorted(extended[q], key=lambda d: (-vnlegal_cv.get(q, {}).get(d, -1e9), d))
        for q in all_ids
    }
    dep_rows_base, dep_groups = ltr_features(dep_local_views, DEPLOYMENT_VIEWS, extended, all_ids, full_channels_cv)
    dep_rows_h0 = {q: np.concatenate([dep_rows_base[q], type_rows[q], cite_rows[q]], axis=1) for q in all_ids}

    X_h0 = np.vstack([dep_rows_h0[q] for q in all_ids])
    y_h0 = np.concatenate([[d in gold[q] for d in dep_groups[q]] for q in all_ids]).astype(np.int8)
    scaler_h0 = StandardScaler().fit(X_h0)
    ltr_h0 = LogisticRegression(C=0.15, class_weight="balanced", solver="liblinear", max_iter=3000, random_state=2026)
    ltr_h0.fit(scaler_h0.transform(X_h0), y_h0)

    public_rows_base_h0, _ = ltr_features(view_rank_pub, DEPLOYMENT_VIEWS, public_candidates, public_ids, public_scores_h0)
    public_rows_h0 = {q: np.concatenate([public_rows_base_h0[q], public_t_rows[q], public_c_rows[q]], axis=1) for q in public_ids}

    preds_h0 = {}
    for q in public_ids:
        proba = ltr_h0.predict_proba(scaler_h0.transform(public_rows_h0[q]))[:, 1]
        order = np.argsort(-proba)
        fused = [public_candidates[q][i] for i in order]
        preds_h0[q] = {"answer": [d for d in fused if d in valid_docs][:5]}

    # Save CONTROL_H0.json & CONTROL_H0.zip
    control_json_path = out_dir / "CONTROL_H0.json"
    control_zip_path = out_dir / "CONTROL_H0.zip"
    with control_json_path.open("w", encoding="utf-8") as f:
        json.dump(preds_h0, f, ensure_ascii=False, indent=2)
    with zipfile.ZipFile(control_zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("submission.json", control_json_path.read_bytes())
    validate_submission_zip(control_zip_path, control_json_path, valid_docs)

    # Verify parity with production burst_userft_maxrecall/submission.json
    prod_sub_path = ROOT / "results/burst_userft_maxrecall/submission.json"
    prod_sub = json.loads(prod_sub_path.read_text(encoding="utf-8"))
    h0_set_matches = sum(set(preds_h0[q]["answer"]) == set(prod_sub[q]["answer"]) for q in public_ids)
    h0_ord_matches = sum(preds_h0[q]["answer"] == prod_sub[q]["answer"] for q in public_ids)
    print(f"CONTROL_H0 parity vs burst_userft_maxrecall: Set={h0_set_matches}/1000, Ordered={h0_ord_matches}/1000")

    # 6. Fit P1 LTR Model & Generate CANDIDATE_PROFILE_P1
    print("Fitting P1 LTR model on CAL600 using OOF profile rankings...", flush=True)
    p1_views_cal = dict(dep_local_views)
    p1_views_cal["fulltrain_huy_profile"] = cal_oof_rankings
    p1_names = list(DEPLOYMENT_VIEWS) + ["fulltrain_huy_profile"]

    p1_rows_cal_base, _ = ltr_features(p1_views_cal, p1_names, extended, all_ids, full_channels_cv)
    p1_rows_cal = {q: np.concatenate([p1_rows_cal_base[q], type_rows[q], cite_rows[q]], axis=1) for q in all_ids}

    X_p1 = np.vstack([p1_rows_cal[q] for q in all_ids])
    y_p1 = np.concatenate([[d in gold[q] for d in dep_groups[q]] for q in all_ids]).astype(np.int8)
    scaler_p1 = StandardScaler().fit(X_p1)
    ltr_p1 = LogisticRegression(C=0.15, class_weight="balanced", solver="liblinear", max_iter=3000, random_state=2026)
    ltr_p1.fit(scaler_p1.transform(X_p1), y_p1)

    print("Generating P1 public predictions using full 6991 profile rankings...", flush=True)
    p1_views_pub = dict(view_rank_pub)
    p1_views_pub["fulltrain_huy_profile"] = public_rankings

    public_rows_base_p1, _ = ltr_features(p1_views_pub, p1_names, public_candidates, public_ids, public_scores_h0)
    public_rows_p1 = {q: np.concatenate([public_rows_base_p1[q], public_t_rows[q], public_c_rows[q]], axis=1) for q in public_ids}

    preds_p1 = {}
    for q in public_ids:
        proba = ltr_p1.predict_proba(scaler_p1.transform(public_rows_p1[q]))[:, 1]
        order = np.argsort(-proba)
        fused = [public_candidates[q][i] for i in order]
        preds_p1[q] = {"answer": [d for d in fused if d in valid_docs][:5]}

    # Save CANDIDATE_PROFILE_P1.json & CANDIDATE_PROFILE_P1.zip
    cand_json_path = out_dir / "CANDIDATE_PROFILE_P1.json"
    cand_zip_path = out_dir / "CANDIDATE_PROFILE_P1.zip"
    with cand_json_path.open("w", encoding="utf-8") as f:
        json.dump(preds_p1, f, ensure_ascii=False, indent=2)
    with zipfile.ZipFile(cand_zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("submission.json", cand_json_path.read_bytes())
    validate_submission_zip(cand_zip_path, cand_json_path, valid_docs)

    # 7. Public Churn vs H0
    changed_set_queries = sum(set(preds_p1[q]["answer"]) != set(preds_h0[q]["answer"]) for q in public_ids)
    changed_order_queries = sum(preds_p1[q]["answer"] != preds_h0[q]["answer"] for q in public_ids)
    jaccards = [len(set(preds_p1[q]["answer"]) & set(preds_h0[q]["answer"])) / len(set(preds_p1[q]["answer"]) | set(preds_h0[q]["answer"])) for q in public_ids]
    entering_docs = sum(len(set(preds_p1[q]["answer"]) - set(preds_h0[q]["answer"])) for q in public_ids)
    leaving_docs = sum(len(set(preds_h0[q]["answer"]) - set(preds_p1[q]["answer"])) for q in public_ids)
    rank5_boundary_changes = sum(preds_p1[q]["answer"][4] != preds_h0[q]["answer"][4] for q in public_ids)

    churn_audit = {
        "total_queries": 1000,
        "changed_top5_set_queries": changed_set_queries,
        "changed_top5_set_pct": (changed_set_queries / 1000.0) * 100.0,
        "changed_order_queries": changed_order_queries,
        "changed_order_pct": (changed_order_queries / 1000.0) * 100.0,
        "mean_top5_jaccard": float(np.mean(jaccards)),
        "docs_entering_top5": entering_docs,
        "docs_leaving_top5": leaving_docs,
        "rank5_boundary_changes": rank5_boundary_changes,
    }

    # Load dual cal report to check promotion verdict
    dual_cal_path = out_dir / "PROFILE_DUAL_CAL_REPORT.json"
    dual_cal_rep = json.loads(dual_cal_path.read_text(encoding="utf-8"))
    verdict = dual_cal_rep["final_verdict"]

    promoted_zip_created = False
    if verdict == "PROMOTE_PROFILE":
        promoted_zip_path = out_dir / "PROMOTED.zip"
        with zipfile.ZipFile(promoted_zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("submission.json", cand_json_path.read_bytes())
        validate_submission_zip(promoted_zip_path, cand_json_path, valid_docs)
        promoted_zip_created = True

    packages_audit = {
        "schema_version": "dsc2026.gemini.huy_fulltrain_profile_port_v1.public_candidate_audit.v1",
        "final_verdict": verdict,
        "promoted_zip_created": promoted_zip_created,
        "packages": {
            "CONTROL_H0": {
                "json_path": str(control_json_path),
                "zip_path": str(control_zip_path),
                "json_sha256": sha256_file(control_json_path),
                "json_md5": md5_file(control_json_path),
                "zip_sha256": sha256_file(control_zip_path),
                "zip_md5": md5_file(control_zip_path),
                "byte_exact_match_burst_userft_maxrecall": (h0_ord_matches == 1000),
            },
            "CANDIDATE_PROFILE_P1": {
                "json_path": str(cand_json_path),
                "zip_path": str(cand_zip_path),
                "json_sha256": sha256_file(cand_json_path),
                "json_md5": md5_file(cand_json_path),
                "zip_sha256": sha256_file(cand_zip_path),
                "zip_md5": md5_file(cand_zip_path),
            },
        },
        "churn_p1_vs_h0": churn_audit,
    }

    pub_audit_file = out_dir / "PUBLIC_CANDIDATE_AUDIT.json"
    with pub_audit_file.open("w", encoding="utf-8") as f:
        json.dump(packages_audit, f, indent=2)

    print(f"\nPUBLIC CANDIDATE AUDIT COMPLETE:")
    print(f"  CONTROL_H0 ZIP SHA256: {packages_audit['packages']['CONTROL_H0']['zip_sha256']}")
    print(f"  CANDIDATE_PROFILE_P1 ZIP SHA256: {packages_audit['packages']['CANDIDATE_PROFILE_P1']['zip_sha256']}")
    print(f"  P1 vs H0 Churn: {changed_set_queries}/1000 ({churn_audit['changed_top5_set_pct']}%), Mean Jaccard: {churn_audit['mean_top5_jaccard']:.4f}")
    print(f"Wrote {pub_audit_file} in {time.perf_counter() - started:.2f}s")


if __name__ == "__main__":
    main()
