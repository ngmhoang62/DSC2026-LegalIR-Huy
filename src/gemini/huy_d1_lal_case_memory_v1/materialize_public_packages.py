"""Train final models on all CAL600 and materialize Public test packages."""

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
RESULTS_DIR = ROOT / "results" / "gemini" / "huy_d1_lal_case_memory_v1"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src" / "huy_fasttrack"))
sys.path.insert(0, str(ROOT.parent / "LegalIR" / "scripts"))

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
import run_huy_5fold_fasttrack as core
from exp_final_memory_ltr_probe import MEMORY_NAMES, memory_features, support_index
from src.gemini.huy_d1_lal_case_memory_v1.audit_data_isolation import load_duplicate_graph
from src.gemini.huy_d1_lal_case_memory_v1.verify_query_embeddings import FrozenLALQueryEncoder

LAL_QUERIES = ROOT.parent / "LegalIR" / "cache" / "exp109b_encoder_complementarity" / "embeddings" / "vnlegal_lal" / "queries.npz"
MODEL_PATH = ROOT / "models" / "vnlegal-lal"
D1_PREVIOUS_CANDIDATE = ROOT / "results" / "gemini" / "huy_vnlegal_rank_ablation_v1" / "CANDIDATE_D1_VNLEGAL_SCORE_ONLY.json"

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


def normalize(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    return array / np.maximum(norms, 1e-12)


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
    qids: List[str],
    m0_cand: Dict[str, List[str]],
    m0_scores: Dict[str, np.ndarray],
    arm_cand: Dict[str, List[str]],
    arm_scores: Dict[str, np.ndarray],
) -> Dict[str, Any]:
    changed_top5_sets = 0
    changed_ordered = 0
    rank5_boundary = 0
    entering_docs = 0
    leaving_docs = 0
    jaccards = []
    margin_diffs = []

    for q in qids:
        c0 = m0_cand[q]
        c1 = arm_cand[q]
        s0 = set(c0)
        s1 = set(c1)

        if s0 != s1:
            changed_top5_sets += 1
            entering_docs += len(s1 - s0)
            leaving_docs += len(s0 - s1)

        if c0 != c1:
            changed_ordered += 1

        if c0[4] != c1[4]:
            rank5_boundary += 1

        jacc = len(s0 & s1) / len(s0 | s1) if (s0 | s1) else 1.0
        jaccards.append(jacc)

        s_arr0 = np.sort(m0_scores[q])[::-1]
        s_arr1 = np.sort(arm_scores[q])[::-1]
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
        "changed_ordered_outputs": changed_ordered,
        "changed_ordered_outputs_pct": (changed_ordered / len(qids)) * 100.0,
        "mean_top5_jaccard": float(np.mean(jaccards)),
        "entering_docs_count": entering_docs,
        "leaving_docs_count": leaving_docs,
        "rank5_boundary_changes": rank5_boundary,
        "rank5_rank6_margin_distribution": margin_stats,
    }


def run_public_materialization() -> Dict[str, Any]:
    started = time.perf_counter()
    print("=== Materialize Public Packages ===", flush=True)

    # 1. Load CAL inputs
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

    def load_aligned_cv(rel_path: str, floor=None):
        obj = load_pkl(rel_path)
        fl = floor if floor is not None else min(v for q in obj for v in obj[q].values())
        return {q: {d: obj.get(q, {}).get(d, fl) for d in extended[q]} for q in all_ids}

    vnlegal_cv = load_pkl("results/embedding_finetune/vnlegal_lal_cv_scores.pkl")
    crossenc_cv = load_aligned_cv("results/crossenc_fullpool/cv_scores.pkl", -11.5)
    extra_cv = {name: load_aligned_cv(rel) for name, rel in EXTRA_CV_PATHS.items()}
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

    base_d1_rows, d1_groups = ltr_features(
        local_views, D1_VIEWS, extended, all_ids, full_channels_cv
    )

    # Load duplicate graph and V2 population
    folds, pools, questions, v2_golds, e5_orders, e5_scores, v2_dup, _ = core.load_inputs()
    _, _, _, dup_map = load_duplicate_graph()
    v2_population = sorted(pools.keys(), key=int)
    v2_set = set(v2_population)

    with np.load(LAL_QUERIES, allow_pickle=False) as z:
        all_npz_qids = list(map(str, z["query_ids"].tolist()))
        all_npz_vecs = normalize(z["vectors"].astype(np.float32))

    qid_to_vec_idx = {qid: i for i, qid in enumerate(all_npz_qids)}

    # CAL600 final training memory features: 4-block OOF: P - X - dup(X)
    cal_mem_rows = {}
    for block_name, block_qids in blocks.items():
        b_set = set(block_qids)
        b_dups = {dup for q in b_set for dup in dup_map.get(q, set())}
        b_support = sorted(v2_set - b_set - b_dups, key=int)
        b_by_doc, b_freq = support_index(v2_golds, b_support)
        b_support_indices = [qid_to_vec_idx[q] for q in b_support]
        b_support_matrix = all_npz_vecs[b_support_indices]

        for q in block_qids:
            q_vec = all_npz_vecs[qid_to_vec_idx[q]]
            sims = b_support_matrix @ q_vec
            cal_mem_rows[q] = memory_features(
                sims, extended[q], b_support, v2_golds, b_by_doc, b_freq
            )

    # Train final models on CAL600
    y_train = np.concatenate([
        np.asarray([doc in gold[q] for doc in extended[q]], dtype=np.int8)
        for q in all_ids
    ])

    x_train_m0 = np.vstack([
        np.concatenate([base_d1_rows[q], type_rows[q], cite_rows[q]], axis=1)
        for q in all_ids
    ])
    x_train_m1 = np.vstack([
        np.concatenate([base_d1_rows[q], type_rows[q], cite_rows[q], cal_mem_rows[q]], axis=1)
        for q in all_ids
    ])
    x_train_m2 = np.vstack([
        np.concatenate([base_d1_rows[q], cite_rows[q], cal_mem_rows[q]], axis=1)
        for q in all_ids
    ])

    scaler_m0 = StandardScaler().fit(x_train_m0)
    model_m0 = LogisticRegression(C=0.15, class_weight="balanced", solver="liblinear", max_iter=3000, random_state=2026).fit(scaler_m0.transform(x_train_m0), y_train)

    scaler_m1 = StandardScaler().fit(x_train_m1)
    model_m1 = LogisticRegression(C=0.15, class_weight="balanced", solver="liblinear", max_iter=3000, random_state=2026).fit(scaler_m1.transform(x_train_m1), y_train)

    scaler_m2 = StandardScaler().fit(x_train_m2)
    model_m2 = LogisticRegression(C=0.15, class_weight="balanced", solver="liblinear", max_iter=3000, random_state=2026).fit(scaler_m2.transform(x_train_m2), y_train)

    print("Trained final CAL600 models for M0, M1, M2.", flush=True)

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

    base_pub_rows, _ = ltr_features(
        view_rank_pub, D1_VIEWS, public_candidates, public_ids, public_scores_base
    )

    # 3. Public Memory Features
    print("Encoding public questions with FrozenLALQueryEncoder...", flush=True)
    encoder = FrozenLALQueryEncoder(MODEL_PATH)
    pub_questions = [public_meta[q] for q in public_ids]
    pub_vectors = encoder.encode(pub_questions)

    # Support pool for public is all 6,991 V2 queries
    pub_support = v2_population
    pub_by_doc, pub_freq = support_index(v2_golds, pub_support)
    pub_support_indices = [qid_to_vec_idx[q] for q in pub_support]
    pub_support_matrix = all_npz_vecs[pub_support_indices]

    print("Computing public memory features...", flush=True)
    pub_sim_matrix = pub_vectors @ pub_support_matrix.T  # (1000, 6991)

    pub_mem_rows = {}
    for i, q in enumerate(public_ids):
        pub_mem_rows[q] = memory_features(
            pub_sim_matrix[i], public_candidates[q], pub_support, v2_golds, pub_by_doc, pub_freq
        )

    # 4. Predict on Public for M0, M1, M2
    predictions_pub: Dict[str, Dict[str, List[str]]] = {}
    scores_pub: Dict[str, Dict[str, np.ndarray]] = {}

    for arm in ["M0_D1_BASELINE", "M1_D1_PLUS_LAL_MEMORY", "M2_D1_PLUS_LAL_MEMORY_NO_DOCTYPE"]:
        predictions_pub[arm] = {}
        scores_pub[arm] = {}

        if arm == "M0_D1_BASELINE":
            scaler = scaler_m0
            model = model_m0
            rows = {
                q: np.concatenate([base_pub_rows[q], public_t_rows[q], public_c_rows[q]], axis=1)
                for q in public_ids
            }
        elif arm == "M1_D1_PLUS_LAL_MEMORY":
            scaler = scaler_m1
            model = model_m1
            rows = {
                q: np.concatenate([base_pub_rows[q], public_t_rows[q], public_c_rows[q], pub_mem_rows[q]], axis=1)
                for q in public_ids
            }
        elif arm == "M2_D1_PLUS_LAL_MEMORY_NO_DOCTYPE":
            scaler = scaler_m2
            model = model_m2
            rows = {
                q: np.concatenate([base_pub_rows[q], public_c_rows[q], pub_mem_rows[q]], axis=1)
                for q in public_ids
            }

        for q in public_ids:
            scores = model.decision_function(scaler.transform(rows[q]))
            scores_pub[arm][q] = scores
            order = np.argsort(-scores)
            fused = [public_candidates[q][i] for i in order]
            predictions_pub[arm][q] = [d for d in fused if d in valid_docs][:5]

    # 5. Public Parity Check vs Previous D1 Candidate
    if not D1_PREVIOUS_CANDIDATE.exists():
        raise FileNotFoundError(f"Missing previous D1 candidate {D1_PREVIOUS_CANDIDATE}")

    with open(D1_PREVIOUS_CANDIDATE, "r", encoding="utf-8") as f:
        prev_d1_data = json.load(f)

    prev_d1_cand = {q: prev_d1_data[q]["answer"] for q in public_ids}

    m0_cand = predictions_pub["M0_D1_BASELINE"]
    exact_matches = sum(1 for q in public_ids if m0_cand[q] == prev_d1_cand[q])
    public_parity_passed = (exact_matches == 1000)

    print(f"Public M0 exact match vs previous D1 candidate: {exact_matches}/1000", flush=True)

    if not public_parity_passed:
        raise RuntimeError(f"Public parity failed! Expected 1000/1000 exact match, got {exact_matches}/1000")

    # 6. Materialize JSON and ZIP packages
    packages_info = {}

    packages_to_build = [
        ("CONTROL_D1_5VIEW", "M0_D1_BASELINE"),
        ("CANDIDATE_M1_D1_LAL_MEMORY", "M1_D1_PLUS_LAL_MEMORY"),
        ("CANDIDATE_M2_D1_LAL_MEMORY_NO_DOCTYPE", "M2_D1_PLUS_LAL_MEMORY_NO_DOCTYPE"),
    ]

    for pkg_name, arm in packages_to_build:
        json_path = RESULTS_DIR / f"{pkg_name}.json"
        zip_path = RESULTS_DIR / f"{pkg_name}.zip"

        payload = {q: {"answer": predictions_pub[arm][q]} for q in public_ids}
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.write(json_path, arcname="submission.json")

        validate_submission_zip(zip_path, json_path, valid_docs)

        packages_info[pkg_name] = {
            "arm": arm,
            "json_path": str(json_path),
            "json_sha256": sha256_file(json_path),
            "zip_path": str(zip_path),
            "zip_sha256": sha256_file(zip_path),
            "zip_md5": md5_file(zip_path),
            "validation_passed": True,
        }
        print(f"Created and validated {zip_path.name}", flush=True)

    # 7. Audit Public Churn vs D1
    m1_churn = audit_public_churn(public_ids, m0_cand, scores_pub["M0_D1_BASELINE"], predictions_pub["M1_D1_PLUS_LAL_MEMORY"], scores_pub["M1_D1_PLUS_LAL_MEMORY"])
    m2_churn = audit_public_churn(public_ids, m0_cand, scores_pub["M0_D1_BASELINE"], predictions_pub["M2_D1_PLUS_LAL_MEMORY_NO_DOCTYPE"], scores_pub["M2_D1_PLUS_LAL_MEMORY_NO_DOCTYPE"])

    public_audit = {
        "schema_version": "dsc2026.gemini.huy_d1_lal_case_memory_v1.public_audit.v1",
        "public_parity": {
            "previous_d1_candidate": str(D1_PREVIOUS_CANDIDATE),
            "exact_matches": exact_matches,
            "total_queries": len(public_ids),
            "parity_passed": public_parity_passed,
            "status": "PASS" if public_parity_passed else "BLOCKED_PUBLIC_PARITY",
        },
        "packages": packages_info,
        "public_churn_vs_d1": {
            "m1_d1_plus_lal_memory": m1_churn,
            "m2_d1_plus_lal_memory_no_doctype": m2_churn,
        },
    }

    out_path = RESULTS_DIR / "PUBLIC_LAL_MEMORY_AUDIT.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(public_audit, f, indent=2)
    print(f"Wrote {out_path}", flush=True)
    return public_audit


if __name__ == "__main__":
    run_public_materialization()
