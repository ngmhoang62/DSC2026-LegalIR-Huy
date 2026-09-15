"""Train final models on all CAL600 and materialize Public test packages for S0 and S1."""

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

from .common import (
    D1_VIEWS,
    EXTRA_CV_PATHS,
    EXTRA_PUB_PATHS,
    RESULTS_DIR,
    ROOT,
    S1_VIEWS,
    SPARSE_RANK_VIEWS,
    SPARSE_SCORE_CHANNELS,
    load_aligned,
    load_pkl,
    load_sparse_channels_and_views,
    md5_file,
    sha256_file,
)

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
    s0_preds: Dict[str, Dict[str, List[str]]],
    s1_preds: Dict[str, Dict[str, List[str]]],
    s0_scores_all: Dict[str, np.ndarray],
    s1_scores_all: Dict[str, np.ndarray],
    public_candidates: Dict[str, List[str]],
) -> Dict[str, Any]:
    qids = sorted(s0_preds.keys())
    changed_top5_sets = 0
    changed_ordered_outputs = 0
    jaccards = []
    entering_docs = 0
    leaving_docs = 0
    rank5_boundary_changes = 0

    for q in qids:
        c0 = s0_preds[q]["answer"]
        c1 = s1_preds[q]["answer"]
        set0 = set(c0)
        set1 = set(c1)

        if set0 != set1:
            changed_top5_sets += 1
            entering_docs += len(set1 - set0)
            leaving_docs += len(set0 - set1)

        if c0 != c1:
            changed_ordered_outputs += 1

        jacc = len(set0 & set1) / len(set0 | set1)
        jaccards.append(jacc)

        cand_docs = public_candidates[q]
        s0_sc = s0_scores_all[q]
        s1_sc = s1_scores_all[q]
        doc_to_s0 = {d: s0_sc[i] for i, d in enumerate(cand_docs)}
        doc_to_s1 = {d: s1_sc[i] for i, d in enumerate(cand_docs)}

        s0_sorted = sorted(cand_docs, key=lambda d: -doc_to_s0[d])
        s1_sorted = sorted(cand_docs, key=lambda d: -doc_to_s1[d])

        if len(s0_sorted) >= 6 and len(s1_sorted) >= 6:
            r5_doc_s0 = s0_sorted[4]
            r6_doc_s0 = s0_sorted[5]
            margin_s0 = doc_to_s0[r5_doc_s0] - doc_to_s0[r6_doc_s0]

            r5_doc_s1 = s1_sorted[4]
            r6_doc_s1 = s1_sorted[5]
            margin_s1 = doc_to_s1[r5_doc_s1] - doc_to_s1[r6_doc_s1]

            if (r5_doc_s0 != r5_doc_s1) or (r6_doc_s0 != r6_doc_s1) or (abs(margin_s1 - margin_s0) > 0.05):
                rank5_boundary_changes += 1

    return {
        "total_queries": len(qids),
        "changed_top5_sets": changed_top5_sets,
        "changed_top5_sets_pct": float(changed_top5_sets / len(qids) * 100.0),
        "changed_ordered_outputs": changed_ordered_outputs,
        "changed_ordered_outputs_pct": float(changed_ordered_outputs / len(qids) * 100.0),
        "mean_top5_jaccard": float(np.mean(jaccards)),
        "total_entering_docs": entering_docs,
        "total_leaving_docs": leaving_docs,
        "rank5_boundary_changes": rank5_boundary_changes,
        "rank5_boundary_changes_pct": float(rank5_boundary_changes / len(qids) * 100.0),
    }


def materialize_public_packages() -> Dict[str, Any]:
    t0 = time.perf_counter()
    ctx_dir = ROOT / "DSC2026-LegalIR-main" / "v4_run" / "public_test_dataset" / "selected-contexts"
    docs = DocumentStore(sorted(ctx_dir.glob("context_*.json")))

    queries, blocks, all_ids, extended, local_views, base_scores = (
        build_training_cap(
            ROOT,
            32,
            "results/corpus_index/holdout_extended_scores_cap32.pkl",
            depth=20,
        )
    )
    gold = {q: queries[q][1] for q in all_ids}

    vnlegal_cv = load_pkl("results/embedding_finetune/vnlegal_lal_cv_scores.pkl")
    crossenc_cv = load_aligned("results/crossenc_fullpool/cv_scores.pkl", extended, all_ids, -11.5)
    extra_cv = {name: load_aligned(rel, extended, all_ids) for name, rel in EXTRA_CV_PATHS.items()}

    sparse_scores_cal, sparse_ranks_cal = load_sparse_channels_and_views(extended, all_ids)

    d1_channels_cal = {
        **base_scores,
        "vnlegal_lal": vnlegal_cv,
        "crossenc": crossenc_cv,
        **extra_cv,
    }

    s1_channels_cal = {
        **d1_channels_cal,
        "legalir_bm25": sparse_scores_cal["legalir_bm25"],
        "legalir_trigram": sparse_scores_cal["legalir_trigram"],
    }

    views_map_cal = {
        **local_views,
        "legalir_bm25": sparse_ranks_cal["legalir_bm25"],
        "legalir_trigram": sparse_ranks_cal["legalir_trigram"],
    }

    type_table = build_type_table(ROOT, docs, all_ids, extended)
    type_rows = type_features(extended, type_table, queries, all_ids)
    own, cited = build_citation_table(docs, all_ids, extended)
    cite_rows = citation_features(extended, own, cited, all_ids)

    # 2. Public Retrieval and Candidates
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

    # Load sparse channels & views for public
    sparse_scores_pub, sparse_ranks_pub = load_sparse_channels_and_views(public_candidates, public_ids)

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
        "legalir_bm25": sparse_ranks_pub["legalir_bm25"],
        "legalir_trigram": sparse_ranks_pub["legalir_trigram"],
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

    public_scores_s1 = {
        **public_scores_base,
        "legalir_bm25": sparse_scores_pub["legalir_bm25"],
        "legalir_trigram": sparse_scores_pub["legalir_trigram"],
    }

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

    # --- 3. FIT S0 (D1 5-View) on Full CAL600 & Predict Public ---
    print("Fitting S0 (D1 5-View) on CAL600...", flush=True)
    s0_rows_cal_base, s0_groups = ltr_features(
        local_views, D1_VIEWS, extended, all_ids, d1_channels_cal
    )
    s0_rows_cal = {
        q: np.concatenate([s0_rows_cal_base[q], type_rows[q], cite_rows[q]], axis=1)
        for q in all_ids
    }
    X_s0 = np.vstack([s0_rows_cal[q] for q in all_ids])
    y_s0 = np.concatenate(
        [[d in gold[q] for d in s0_groups[q]] for q in all_ids]
    ).astype(np.int8)

    scaler_s0 = StandardScaler().fit(X_s0)
    ltr_s0 = LogisticRegression(
        C=0.15,
        class_weight="balanced",
        solver="liblinear",
        max_iter=3000,
        random_state=2026,
    )
    ltr_s0.fit(scaler_s0.transform(X_s0), y_s0)

    public_rows_base_s0, _ = ltr_features(
        view_rank_pub, D1_VIEWS, public_candidates, public_ids, public_scores_base
    )
    public_rows_s0 = {
        q: np.concatenate(
            [public_rows_base_s0[q], public_t_rows[q], public_c_rows[q]], axis=1
        )
        for q in public_ids
    }

    preds_s0: Dict[str, Dict[str, List[str]]] = {}
    s0_scores_all: Dict[str, np.ndarray] = {}
    for q in public_ids:
        scores = ltr_s0.decision_function(scaler_s0.transform(public_rows_s0[q]))
        s0_scores_all[q] = scores
        order = np.argsort(-scores)
        fused = [public_candidates[q][i] for i in order]
        preds_s0[q] = {"answer": [d for d in fused if d in valid_docs][:5]}

    control_json = RESULTS_DIR / "CONTROL_D1_5VIEW.json"
    control_zip = RESULTS_DIR / "CONTROL_D1_5VIEW.zip"
    control_json.write_text(json.dumps(preds_s0, ensure_ascii=False, indent=2), encoding="utf-8")
    with zipfile.ZipFile(control_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("submission.json", control_json.read_bytes())
    validate_submission_zip(control_zip, control_json, valid_docs)

    # Parity check against CANDIDATE_D1_VNLEGAL_SCORE_ONLY.json
    prev_d1_file = ROOT / "results/gemini/huy_vnlegal_rank_ablation_v1/CANDIDATE_D1_VNLEGAL_SCORE_ONLY.json"
    if not prev_d1_file.exists():
        raise FileNotFoundError(f"Missing prior D1 candidate file: {prev_d1_file}")
    prev_d1 = json.loads(prev_d1_file.read_text(encoding="utf-8"))

    s0_set_matches = sum(set(preds_s0[q]["answer"]) == set(prev_d1[q]["answer"]) for q in public_ids)
    s0_ord_matches = sum(preds_s0[q]["answer"] == prev_d1[q]["answer"] for q in public_ids)
    print(f"S0 Public Parity vs previous D1: Set={s0_set_matches}/1000, Ordered={s0_ord_matches}/1000")
    if s0_ord_matches != 1000:
        raise RuntimeError(f"BLOCKED_PUBLIC_PARITY: S0 ordered match is {s0_ord_matches}/1000, expected 1000/1000")

    # --- 4. FIT S1 (D1 + LegalIR Sparse) on Full CAL600 & Predict Public ---
    print("Fitting S1 (D1 + LegalIR Sparse) on CAL600...", flush=True)
    s1_rows_cal_base, s1_groups = ltr_features(
        views_map_cal, S1_VIEWS, extended, all_ids, s1_channels_cal
    )
    s1_rows_cal = {
        q: np.concatenate([s1_rows_cal_base[q], type_rows[q], cite_rows[q]], axis=1)
        for q in all_ids
    }
    X_s1 = np.vstack([s1_rows_cal[q] for q in all_ids])
    y_s1 = np.concatenate(
        [[d in gold[q] for d in s1_groups[q]] for q in all_ids]
    ).astype(np.int8)

    scaler_s1 = StandardScaler().fit(X_s1)
    ltr_s1 = LogisticRegression(
        C=0.15,
        class_weight="balanced",
        solver="liblinear",
        max_iter=3000,
        random_state=2026,
    )
    ltr_s1.fit(scaler_s1.transform(X_s1), y_s1)

    public_rows_base_s1, _ = ltr_features(
        view_rank_pub, S1_VIEWS, public_candidates, public_ids, public_scores_s1
    )
    public_rows_s1 = {
        q: np.concatenate(
            [public_rows_base_s1[q], public_t_rows[q], public_c_rows[q]], axis=1
        )
        for q in public_ids
    }

    preds_s1: Dict[str, Dict[str, List[str]]] = {}
    s1_scores_all: Dict[str, np.ndarray] = {}
    for q in public_ids:
        scores = ltr_s1.decision_function(scaler_s1.transform(public_rows_s1[q]))
        s1_scores_all[q] = scores
        order = np.argsort(-scores)
        fused = [public_candidates[q][i] for i in order]
        preds_s1[q] = {"answer": [d for d in fused if d in valid_docs][:5]}

    candidate_json = RESULTS_DIR / "CANDIDATE_S1_D1_LEGALIR_SPARSE.json"
    candidate_zip = RESULTS_DIR / "CANDIDATE_S1_D1_LEGALIR_SPARSE.zip"
    candidate_json.write_text(json.dumps(preds_s1, ensure_ascii=False, indent=2), encoding="utf-8")
    with zipfile.ZipFile(candidate_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("submission.json", candidate_json.read_bytes())
    validate_submission_zip(candidate_zip, candidate_json, valid_docs)

    # Churn analysis
    churn = audit_public_churn(preds_s0, preds_s1, s0_scores_all, s1_scores_all, public_candidates)

    public_audit = {
        "schema_version": "dsc2026.gemini.huy_d1_legalir_sparse_port_v1.public_sparse_audit.v1",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "runtime_seconds": float(time.perf_counter() - t0),
        "s0_control": {
            "json_path": str(control_json.relative_to(ROOT)),
            "zip_path": str(control_zip.relative_to(ROOT)),
            "json_sha256": sha256_file(control_json),
            "zip_sha256": sha256_file(control_zip),
            "json_md5": md5_file(control_json),
            "feature_dim": X_s0.shape[1],
            "public_parity_vs_prev_d1": {
                "set_exact_count": s0_set_matches,
                "ordered_exact_count": s0_ord_matches,
                "status": "PASS_EXACT_1000_OF_1000"
            }
        },
        "s1_candidate": {
            "json_path": str(candidate_json.relative_to(ROOT)),
            "zip_path": str(candidate_zip.relative_to(ROOT)),
            "json_sha256": sha256_file(candidate_json),
            "zip_sha256": sha256_file(candidate_zip),
            "json_md5": md5_file(candidate_json),
            "feature_dim": X_s1.shape[1],
        },
        "public_churn_s1_vs_s0": churn,
        "structural_validation": {
            "control_zip_valid": True,
            "candidate_zip_valid": True,
        }
    }

    # Check promotion to determine PROMOTED.zip
    cal_rep_file = RESULTS_DIR / "SPARSE_D1_CAL_REPORT.json"
    if cal_rep_file.exists():
        cal_rep = json.loads(cal_rep_file.read_text(encoding="utf-8"))
        if cal_rep.get("promotion_gates", {}).get("all_gates_passed", False):
            promoted_zip = RESULTS_DIR / "PROMOTED.zip"
            promoted_zip.write_bytes(candidate_zip.read_bytes())
            public_audit["promoted_package"] = {
                "zip_path": str(promoted_zip.relative_to(ROOT)),
                "zip_sha256": sha256_file(promoted_zip),
            }

    out_path = RESULTS_DIR / "PUBLIC_SPARSE_AUDIT.json"
    out_path.write_text(json.dumps(public_audit, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote public sparse audit to {out_path}")
    return public_audit


if __name__ == "__main__":
    materialize_public_packages()
