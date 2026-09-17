"""Stage 3: Verify Public D1 Control Parity against Champion Submission (1000/1000 matches required)."""

from __future__ import annotations

import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

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
from tune_doctype_features import build_type_table, type_features
from tune_expanded_fusion_selection import ltr_features

from .common import (
    D1_CHAMPION_JSON_PATH,
    D1_CHAMPION_ZIP_PATH,
    D1_VIEWS,
    EXTRA_PUB_PATHS,
    RESULTS_DIR,
    ROOT,
    compute_candidate_fingerprint,
    compute_query_fingerprint,
    get_git_status,
    load_cal_data_label_free,
    load_cal_gold_labels,
    load_pkl,
    seed_everything,
    sha256_file,
)


def run_public_d1_control_parity() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    print("=== STAGE 3: PUBLIC D1 CONTROL PARITY AUDIT ===", flush=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    seed_everything(2026)
    git_info = get_git_status()

    # 1. Load DocumentStore and Public Metadata
    print("Loading DocumentStore and Public Test metadata...", flush=True)
    docs_store = DocumentStore(
        sorted(
            (
                ROOT
                / "DSC2026-LegalIR-main/v4_run/public_test_dataset/selected-contexts"
            ).glob("context_*.json")
        )
    )

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

    # 2. Reconstruct Public Candidate Pool and Retrieval Views
    print("Reconstructing public candidate pool and 5 retrieval views...", flush=True)
    base_pub = load_pkl("results/burst_gpu_threeview/cpu_top20.pkl")["rankings"]
    pub_retrieval = load_public_retrieval(
        ROOT, DummyArgs, doc_ids_meta, train_meta, public_meta, public_ids
    )
    raw_pub = {
        q: raw_union(pub_retrieval[q], EXPANSION_CONFIG["depth"]) for q in public_ids
    }
    pub_retrieval.clear()

    expansion_scores_pub = dense_expansion(
        ROOT, DummyArgs.cache_dir, public_meta, public_ids, raw_pub, docs_store, "cuda"
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
        docs_store,
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

    # Reconstruct CAL data and full-CAL-trained D1 model
    print("Training production D1 learner on full CAL600...", flush=True)
    docs, queries, blocks, all_ids, extended, local_views, full_channels_cv, type_rows, cite_rows = load_cal_data_label_free()
    gold, _ = load_cal_gold_labels(all_ids)

    three_view_pub = load_pkl("results/burst_gpu_threeview/gpu_scores.checkpoint.pkl")
    crossenc_pub = load_pkl("results/crossenc_fullpool/public_scores.pkl")
    floor_ce_cv = min(min(v.values()) for v in full_channels_cv["crossenc"].values() if v)

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
                d: crossenc_pub.get(q, {}).get(d, floor_ce_cv)
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
    public_types = build_type_table(ROOT, docs_store, public_ids, public_candidates)
    public_t_rows = type_features(
        public_candidates, public_types, public_queries_meta, public_ids
    )
    public_own, public_cited = build_citation_table(
        docs_store, public_ids, public_candidates
    )
    public_c_rows = citation_features(
        public_candidates, public_own, public_cited, public_ids
    )

    # Train D1 48D on full CAL600
    d1_views_cal = dict(local_views)
    d1_rows_cal_base, d1_groups = ltr_features(
        d1_views_cal, D1_VIEWS, extended, all_ids, full_channels_cv
    )
    d1_rows_cal = {
        q: np.concatenate([d1_rows_cal_base[q], type_rows[q], cite_rows[q]], axis=1)
        for q in all_ids
    }

    d1_dim = d1_rows_cal[all_ids[0]].shape[1]
    assert d1_dim == 48, f"Expected D1 feature dim 48, got {d1_dim}"

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

    # Predict on Public Test
    public_rows_base_d1, _ = ltr_features(
        view_rank_pub, D1_VIEWS, public_candidates, public_ids, public_scores_base
    )
    public_rows_d1 = {
        q: np.concatenate(
            [public_rows_base_d1[q], public_t_rows[q], public_c_rows[q]], axis=1
        )
        for q in public_ids
    }

    preds_control_d1: Dict[str, List[str]] = {}
    scores_control_d1: Dict[str, Dict[str, float]] = {}
    for q in public_ids:
        scores = ltr_d1.decision_function(scaler_d1.transform(public_rows_d1[q]))
        order = np.argsort(-scores)
        fused = [public_candidates[q][i] for i in order]
        preds_control_d1[q] = [d for d in fused if d in valid_docs][:5]
        scores_control_d1[q] = {
            public_candidates[q][i]: float(scores[i]) for i in range(len(scores))
        }

    # 3. Verify against D1 Champion Submission JSON
    if not D1_CHAMPION_JSON_PATH.exists():
        raise FileNotFoundError(f"Missing current D1 champion JSON: {D1_CHAMPION_JSON_PATH}")
    if not D1_CHAMPION_ZIP_PATH.exists():
        raise FileNotFoundError(f"Missing current D1 champion ZIP: {D1_CHAMPION_ZIP_PATH}")

    champion_data = json.loads(D1_CHAMPION_JSON_PATH.read_text(encoding="utf-8"))
    champion_preds = {q: champion_data[q]["answer"] for q in public_ids}

    matches_count = sum(1 for q in public_ids if preds_control_d1[q] == champion_preds[q])
    control_parity_passed = (matches_count == 1000)

    pub_query_fp = compute_query_fingerprint(public_ids, public_meta)
    pub_cand_fp = compute_candidate_fingerprint(public_ids, public_candidates)

    print(f"Public D1 Control Parity: {matches_count}/1000 exact ordered matches.")
    control_parity_report = {
        "schema_version": "dsc2026.gemini.huy_d1_frozen_section_public_v1.public_d1_control_parity.v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_info["head_commit"],
        "status": "PASS" if control_parity_passed else "BLOCKED_PUBLIC_D1_CONTROL_PARITY",
        "total_public_queries": len(public_ids),
        "exact_matches_count": matches_count,
        "control_parity_passed": control_parity_passed,
        "d1_feature_dimension": d1_dim,
        "public_query_fingerprint": pub_query_fp,
        "public_candidate_pool_fingerprint": pub_cand_fp,
        "champion_json_path": str(D1_CHAMPION_JSON_PATH.relative_to(ROOT)).replace("\\", "/"),
        "champion_json_sha256": sha256_file(D1_CHAMPION_JSON_PATH),
        "champion_zip_path": str(D1_CHAMPION_ZIP_PATH.relative_to(ROOT)).replace("\\", "/"),
        "champion_zip_sha256": sha256_file(D1_CHAMPION_ZIP_PATH),
    }

    control_parity_path = RESULTS_DIR / "PUBLIC_D1_CONTROL_PARITY.json"
    control_parity_path.write_text(json.dumps(control_parity_report, indent=2), encoding="utf-8")
    print(f"Wrote {control_parity_path}", flush=True)

    if not control_parity_passed:
        raise RuntimeError(
            f"BLOCKED_PUBLIC_D1_CONTROL_PARITY: D1 control matched {matches_count}/1000 queries vs current D1 submission!"
        )

    public_bundle = {
        "docs_store": docs_store,
        "public_ids": public_ids,
        "public_candidates": public_candidates,
        "view_rank_pub": view_rank_pub,
        "public_scores_base": public_scores_base,
        "public_t_rows": public_t_rows,
        "public_c_rows": public_c_rows,
        "public_meta": public_meta,
        "valid_docs": valid_docs,
        "preds_control_d1": preds_control_d1,
        "scores_control_d1": scores_control_d1,
        "pub_query_fp": pub_query_fp,
        "pub_cand_fp": pub_cand_fp,
    }
    return control_parity_report, public_bundle


if __name__ == "__main__":
    run_public_d1_control_parity()
