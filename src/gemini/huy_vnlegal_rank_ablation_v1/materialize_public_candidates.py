"""Train final models on all CAL600 and materialize Public test packages for D0 and D1."""

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
RESULTS_DIR = ROOT / "results" / "gemini" / "huy_vnlegal_rank_ablation_v1"
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

D0_VIEWS = ["base", "expanded", "jina", "dense", "corpus", "vnlegal_lal"]
D1_VIEWS = ["base", "expanded", "jina", "dense", "corpus"]

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


def validate_submission_zip(zip_path: Path, json_path: Path, valid_docs: Set[str]) -> bool:
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
    d0_preds: Dict[str, Dict[str, List[str]]],
    d1_preds: Dict[str, Dict[str, List[str]]],
    d0_scores_all: Dict[str, np.ndarray],
    d1_scores_all: Dict[str, np.ndarray],
    public_candidates: Dict[str, List[str]],
) -> Dict[str, Any]:
    qids = sorted(d0_preds.keys())
    changed_top5_sets = 0
    changed_ordered_outputs = 0
    jaccards = []
    entering_docs = 0
    leaving_docs = 0
    rank5_boundary_changes = 0

    margin_diffs = []

    for q in qids:
        c0 = d0_preds[q]["answer"]
        c1 = d1_preds[q]["answer"]
        s0 = set(c0)
        s1 = set(c1)

        if s0 != s1:
            changed_top5_sets += 1
            entering_docs += len(s1 - s0)
            leaving_docs += len(s0 - s1)

        if c0 != c1:
            changed_ordered_outputs += 1

        if c0[4] != c1[4]:
            rank5_boundary_changes += 1

        jacc = len(s0 & s1) / len(s0 | s1) if (s0 | s1) else 1.0
        jaccards.append(jacc)

        # Margin at rank 5 / rank 6 boundary
        # Sort candidate scores descending
        s_arr0 = np.sort(d0_scores_all[q])[::-1]
        s_arr1 = np.sort(d1_scores_all[q])[::-1]
        if len(s_arr0) >= 6 and len(s_arr1) >= 6:
            margin0 = float(s_arr0[4] - s_arr0[5])
            margin1 = float(s_arr1[4] - s_arr1[5])
            margin_diffs.append(margin1 - margin0)

    margin_stats = {
        "mean_margin_diff": float(np.mean(margin_diffs)) if margin_diffs else 0.0,
        "median_margin_diff": float(np.median(margin_diffs)) if margin_diffs else 0.0,
        "std_margin_diff": float(np.std(margin_diffs)) if margin_diffs else 0.0,
        "min_margin_diff": float(np.min(margin_diffs)) if margin_diffs else 0.0,
        "max_margin_diff": float(np.max(margin_diffs)) if margin_diffs else 0.0,
    }

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
        "rank5_rank6_margin_distribution": margin_stats,
    }


def main() -> Dict[str, Any]:
    started = time.perf_counter()
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

    public_scores_base = {
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
        public_scores_base[name] = {
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

    # --- 3. FIT D0: Current Production (50D) & Parity Check ---
    print("\n--- Training D0 LTR Model on CAL600 & Scoring Public ---", flush=True)
    d0_views_cal = dict(local_views)
    d0_views_cal["vnlegal_lal"] = {
        q: sorted(extended[q], key=lambda d: (-vnlegal_cv.get(q, {}).get(d, -1e9), d))
        for q in all_ids
    }
    d0_rows_cal_base, d0_groups = ltr_features(
        d0_views_cal, D0_VIEWS, extended, all_ids, full_channels_cv
    )
    d0_rows_cal = {
        q: np.concatenate([d0_rows_cal_base[q], type_rows[q], cite_rows[q]], axis=1)
        for q in all_ids
    }

    X_d0 = np.vstack([d0_rows_cal[q] for q in all_ids])
    y_d0 = np.concatenate(
        [[d in gold[q] for d in d0_groups[q]] for q in all_ids]
    ).astype(np.int8)

    scaler_d0 = StandardScaler().fit(X_d0)
    ltr_d0 = LogisticRegression(
        C=0.15,
        class_weight="balanced",
        solver="liblinear",
        max_iter=3000,
        random_state=2026,
    )
    ltr_d0.fit(scaler_d0.transform(X_d0), y_d0)

    public_rows_base_d0, _ = ltr_features(
        view_rank_pub, D0_VIEWS, public_candidates, public_ids, public_scores_base
    )
    public_rows_d0 = {
        q: np.concatenate(
            [public_rows_base_d0[q], public_t_rows[q], public_c_rows[q]], axis=1
        )
        for q in public_ids
    }

    preds_d0: Dict[str, Dict[str, List[str]]] = {}
    d0_scores_all: Dict[str, np.ndarray] = {}

    for q in public_ids:
        scores = ltr_d0.decision_function(scaler_d0.transform(public_rows_d0[q]))
        d0_scores_all[q] = scores
        order = np.argsort(-scores)
        fused = [public_candidates[q][i] for i in order]
        preds_d0[q] = {"answer": [d for d in fused if d in valid_docs][:5]}

    control_json = RESULTS_DIR / "CONTROL_D0_CURRENT_PRODUCTION.json"
    control_zip = RESULTS_DIR / "CONTROL_D0_CURRENT_PRODUCTION.zip"
    control_json.write_text(json.dumps(preds_d0, ensure_ascii=False, indent=2), encoding="utf-8")
    with zipfile.ZipFile(control_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("submission.json", control_json.read_bytes())
    validate_submission_zip(control_zip, control_json, valid_docs)

    prod_sub_path = ROOT / "results/burst_userft_maxrecall/submission.json"
    prod_sub = json.loads(prod_sub_path.read_text(encoding="utf-8"))
    d0_set_matches = sum(set(preds_d0[q]["answer"]) == set(prod_sub[q]["answer"]) for q in public_ids)
    d0_ord_matches = sum(preds_d0[q]["answer"] == prod_sub[q]["answer"] for q in public_ids)

    print(f"D0 Public Parity vs burst_userft_maxrecall: Set={d0_set_matches}/1000, Ordered={d0_ord_matches}/1000")
    assert d0_set_matches == 1000 and d0_ord_matches == 1000, "BLOCKED_PUBLIC_PARITY: D0 does not match burst_userft_maxrecall"

    # --- 4. FIT D1: Score-Only vnlegal_lal (48D) ---
    print("\n--- Training D1 LTR Model on CAL600 & Scoring Public ---", flush=True)
    d1_views_cal = dict(local_views)
    d1_rows_cal_base, d1_groups = ltr_features(
        d1_views_cal, D1_VIEWS, extended, all_ids, full_channels_cv
    )
    d1_rows_cal = {
        q: np.concatenate([d1_rows_cal_base[q], type_rows[q], cite_rows[q]], axis=1)
        for q in all_ids
    }

    X_d1 = np.vstack([d1_rows_cal[q] for q in all_ids])
    y_d1 = np.concatenate(
        [[d in gold[q] for d in d1_groups[q]] for q in all_ids]
    ).astype(np.int8)

    scaler_d1 = StandardScaler().fit(X_d1)
    ltr_d1 = LogisticRegression(
        C=0.15,
        class_weight="balanced",
        solver="liblinear",
        max_iter=3000,
        random_state=2026,
    )
    ltr_d1.fit(scaler_d1.transform(X_d1), y_d1)

    public_rows_base_d1, _ = ltr_features(
        view_rank_pub, D1_VIEWS, public_candidates, public_ids, public_scores_base
    )
    public_rows_d1 = {
        q: np.concatenate(
            [public_rows_base_d1[q], public_t_rows[q], public_c_rows[q]], axis=1
        )
        for q in public_ids
    }

    preds_d1: Dict[str, Dict[str, List[str]]] = {}
    d1_scores_all: Dict[str, np.ndarray] = {}

    for q in public_ids:
        scores = ltr_d1.decision_function(scaler_d1.transform(public_rows_d1[q]))
        d1_scores_all[q] = scores
        order = np.argsort(-scores)
        fused = [public_candidates[q][i] for i in order]
        preds_d1[q] = {"answer": [d for d in fused if d in valid_docs][:5]}

    cand_json = RESULTS_DIR / "CANDIDATE_D1_VNLEGAL_SCORE_ONLY.json"
    cand_zip = RESULTS_DIR / "CANDIDATE_D1_VNLEGAL_SCORE_ONLY.zip"
    cand_json.write_text(json.dumps(preds_d1, ensure_ascii=False, indent=2), encoding="utf-8")
    with zipfile.ZipFile(cand_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("submission.json", cand_json.read_bytes())
    validate_submission_zip(cand_zip, cand_json, valid_docs)
    print(f"Validated CANDIDATE_D1_VNLEGAL_SCORE_ONLY.zip (SHA256: {sha256_file(cand_zip)})")

    # --- 5. Public Churn / Risk Audit ---
    churn_audit = audit_public_churn(preds_d0, preds_d1, d0_scores_all, d1_scores_all, public_candidates)

    public_report = {
        "schema_version": "dsc2026.gemini.huy_vnlegal_rank_ablation_v1.public_d1_audit.v1",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "runtime_seconds": time.perf_counter() - started,
        "packages": {
            "control_d0": {
                "name": "CONTROL_D0_CURRENT_PRODUCTION",
                "json_path": str(control_json.relative_to(ROOT)).replace("\\", "/"),
                "zip_path": str(control_zip.relative_to(ROOT)).replace("\\", "/"),
                "sha256_zip": sha256_file(control_zip),
                "sha256_json": sha256_file(control_json),
                "md5_json": md5_file(control_json),
                "exact_parity_with_burst_userft_maxrecall": (d0_set_matches == 1000 and d0_ord_matches == 1000),
            },
            "candidate_d1": {
                "name": "CANDIDATE_D1_VNLEGAL_SCORE_ONLY",
                "json_path": str(cand_json.relative_to(ROOT)).replace("\\", "/"),
                "zip_path": str(cand_zip.relative_to(ROOT)).replace("\\", "/"),
                "sha256_zip": sha256_file(cand_zip),
                "sha256_json": sha256_file(cand_json),
                "md5_json": md5_file(cand_json),
            },
        },
        "churn_analysis_d1_vs_d0": churn_audit,
        "structural_validation_passed": True,
    }

    out_file = RESULTS_DIR / "PUBLIC_D1_AUDIT.json"
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(public_report, f, indent=2)

    print(f"Saved: {out_file}", flush=True)
    return public_report


if __name__ == "__main__":
    main()
