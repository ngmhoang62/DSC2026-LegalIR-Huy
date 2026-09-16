"""Stage 6: Public Stage (Conditional) - D1 Control Parity, Public Adapted Scoring, and Submission Packaging."""

from __future__ import annotations

import hashlib
import json
import pickle
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import numpy as np
import torch
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
    ADAPTER_DIR,
    D1_CHAMPION_JSON_PATH,
    D1_CHAMPION_ZIP_PATH,
    D1_VIEWS,
    EXTRA_PUB_PATHS,
    RESULTS_DIR,
    ROOT,
    compute_fingerprint,
    get_git_status,
    load_adapted_jina_model,
    load_cal_data,
    load_pkl,
    seed_everything,
    sha256_file,
)
from .legal_section_parser import parse_document_into_sections, preselect_legal_sections
from .score_cal_adapted_section_ce import CAL_ADAPTED_CACHE_PATH

PUB_ADAPTED_CACHE_PATH = RESULTS_DIR / "adapted_section_ce_public.pkl"
PUB_ADAPTED_MANIFEST_PATH = RESULTS_DIR / "adapted_section_ce_public_manifest.json"


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


def run_public_stage() -> Dict[str, Any]:
    print("=== STAGE 6: PUBLIC CONTROL REPRODUCTION & ADAPTED CANDIDATE MATERIALIZATION ===", flush=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    seed_everything(2026)
    git_info = get_git_status()

    # 1. Check local verdict
    report_file = RESULTS_DIR / "LOCAL_THREE_ARMS_REPORT.json"
    if not report_file.exists():
        raise FileNotFoundError(f"Missing local evaluation report: {report_file}")
    local_report = json.loads(report_file.read_text(encoding="utf-8"))
    local_verdict = local_report.get("verdict")

    permitted_verdicts = [
        "PUBLIC_CANDIDATE_ADAPTED_SECTION_CHANNEL",
        "STRONG_PUBLIC_CANDIDATE_ADAPTED_SECTION_CHANNEL",
    ]
    if local_verdict not in permitted_verdicts:
        print(
            f"Local verdict '{local_verdict}' does not permit public scoring / packaging. Skipping public stage.",
            flush=True,
        )
        return {"status": "SKIPPED_LOCAL_GATE", "verdict": local_verdict}

    print(f"Local verdict '{local_verdict}' PERMITS public stage. Proceeding...", flush=True)

    # 2. Load DocumentStore, CAL data, and Public Test Metadata
    print("Loading DocumentStore, CAL data, and Public Test metadata...", flush=True)
    docs_store = DocumentStore(
        sorted(
            (
                ROOT
                / "DSC2026-LegalIR-main/v4_run/public_test_dataset/selected-contexts"
            ).glob("context_*.json")
        )
    )
    docs, queries, blocks, all_ids, extended, local_views, full_channels_cv, gold, type_rows, cite_rows = load_cal_data()

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

    # 3. Load Public Candidates and Views (exact D1 5-view contract)
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

    # 4. Step 6a: Public D1 Control Parity Check vs Current D1 Submission
    print("
--- Running Step 6a: Public D1 Control Parity Check ---", flush=True)
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

    preds_control_d1: Dict[str, List[str]] = {}
    for q in public_ids:
        scores = ltr_d1.decision_function(scaler_d1.transform(public_rows_d1[q]))
        order = np.argsort(-scores)
        fused = [public_candidates[q][i] for i in order]
        preds_control_d1[q] = [d for d in fused if d in valid_docs][:5]

    # Verify against exact current D1 champion submission JSON
    if not D1_CHAMPION_JSON_PATH.exists():
        raise FileNotFoundError(f"Missing current D1 champion JSON: {D1_CHAMPION_JSON_PATH}")
    champion_data = json.loads(D1_CHAMPION_JSON_PATH.read_text(encoding="utf-8"))
    champion_preds = {q: champion_data[q]["answer"] for q in public_ids}

    matches_count = sum(1 for q in public_ids if preds_control_d1[q] == champion_preds[q])
    control_parity_passed = (matches_count == 1000)

    print(f"Public D1 Control Parity: {matches_count}/1000 exact ordered matches.")
    control_parity_report = {
        "schema_version": "dsc2026.gemini.huy_d1_adapted_section_channel_public_v1.public_control_parity.v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_info["head_commit"],
        "status": "PASS" if control_parity_passed else "BLOCKED_PUBLIC_D1_CONTROL_PARITY",
        "total_public_queries": len(public_ids),
        "exact_matches_count": matches_count,
        "control_parity_passed": control_parity_passed,
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

    # 5. Step 6b: Score adapted Section CE probability on Public Candidates
    print("
--- Running Step 6b: Scoring Adapted Section CE on Public Candidates ---", flush=True)
    total_public_cand_pairs = sum(len(public_candidates[q]) for q in public_ids)
    print(f"Public candidate pool: {total_public_cand_pairs} candidate pairs across 1000 queries.", flush=True)

    pub_q_fingerprint = compute_fingerprint([f"{q}:{public_meta[q]}" for q in public_ids])
    pub_cand_fingerprint = compute_fingerprint([f"{q}:{','.join(sorted(public_candidates[q]))}" for q in public_ids])
    adapter_model_sha = sha256_file(ADAPTER_DIR / "adapter_model.safetensors")
    parser_sha = sha256_file(Path(__file__).parent / "legal_section_parser.py")

    pub_contract = {
        "git_commit": git_info["head_commit"],
        "adapter_model_sha256": adapter_model_sha,
        "parser_sha256": parser_sha,
        "public_query_fingerprint": pub_q_fingerprint,
        "public_candidate_pool_fingerprint": pub_cand_fingerprint,
        "score_semantics": "adapted_probability = sigmoid(adapted_raw_logit)",
        "aggregation": "MAX",
        "count_sections": 2,
    }

    adapted_pub_scores: Dict[str, Dict[str, float]] = {}
    use_existing_pub = False
    if PUB_ADAPTED_CACHE_PATH.exists() and PUB_ADAPTED_MANIFEST_PATH.exists():
        try:
            cached_pub = pickle.loads(PUB_ADAPTED_CACHE_PATH.read_bytes())
            cached_man = json.loads(PUB_ADAPTED_MANIFEST_PATH.read_text(encoding="utf-8"))
            if (
                cached_man.get("adapter_model_sha256") == adapter_model_sha
                and cached_man.get("public_query_fingerprint") == pub_q_fingerprint
                and cached_man.get("public_candidate_pool_fingerprint") == pub_cand_fingerprint
                and len(cached_pub.get("scores", {})) == len(public_ids)
            ):
                adapted_pub_scores = cached_pub["scores"]
                use_existing_pub = True
                print("Reusing valid existing public adapted score cache.", flush=True)
        except Exception as e:
            print(f"Could not load existing pub cache: {e}", flush=True)

    if not use_existing_pub:
        pub_doc_sections_cache: Dict[str, List[Any]] = {}
        pub_pair_records: List[Tuple[str, str, int, str, str]] = []

        for qid in sorted(public_ids):
            q_text = public_meta[qid]
            cands = public_candidates[qid]
            for did in cands:
                if did not in pub_doc_sections_cache:
                    pub_doc_sections_cache[did] = parse_document_into_sections(
                        did, docs_store[did], max_chunk_words=220, overlap_words=60
                    )
                secs = pub_doc_sections_cache[did]
                chosen_secs = preselect_legal_sections(q_text, secs, count=2)
                for s_idx, sec in enumerate(chosen_secs):
                    pub_pair_records.append((qid, did, s_idx, q_text, sec.text))

        print(f"Generated {len(pub_pair_records)} public section pairs to score on GPU.", flush=True)
        adapted_model, base_model, tok = load_adapted_jina_model(device="cuda")
        t0 = time.perf_counter()
        batch_size = 64
        probs_pub_all: List[float] = []

        for i in range(0, len(pub_pair_records), batch_size):
            batch = pub_pair_records[i : i + batch_size]
            text_pairs = [(r[3], r[4]) for r in batch]
            inputs = tok(
                text_pairs, padding=True, truncation=True, max_length=512, return_tensors="pt"
            ).to("cuda")

            with torch.no_grad():
                logits = adapted_model(**inputs).logits.view(-1).float()
                probs = torch.sigmoid(logits)

            probs_pub_all.extend(probs.cpu().numpy().tolist())

            if (i // batch_size + 1) % 100 == 0 or i + batch_size >= len(pub_pair_records):
                elapsed = time.perf_counter() - t0
                pct = min(100.0, (i + len(batch)) / len(pub_pair_records) * 100.0)
                print(f"[PUB_SCORE] [{i + len(batch)}/{len(pub_pair_records)}] ({pct:.1f}%) Elapsed: {elapsed:.1f}s", flush=True)

        del adapted_model, base_model, tok
        torch.cuda.empty_cache()

        adapted_pub_scores = {q: {} for q in public_ids}
        for (qid, did, s_idx, _, _), prob_val in zip(pub_pair_records, probs_pub_all):
            adapted_pub_scores[qid][did] = max(adapted_pub_scores[qid].get(did, -1e9), float(prob_val))

        PUB_ADAPTED_CACHE_PATH.write_bytes(pickle.dumps({"scores": adapted_pub_scores, "contract": pub_contract}))
        pub_manifest = {
            **pub_contract,
            "cache_sha256": sha256_file(PUB_ADAPTED_CACHE_PATH),
            "total_public_queries": len(public_ids),
            "total_candidate_pairs": total_public_cand_pairs,
            "total_section_pairs_scored": len(pub_pair_records),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
        PUB_ADAPTED_MANIFEST_PATH.write_text(json.dumps(pub_manifest, indent=2), encoding="utf-8")
        print(f"Saved {PUB_ADAPTED_CACHE_PATH}", flush=True)

    # 6. Step 6c: Train A2 on all CAL600 and Predict on Public
    print("
--- Running Step 6c: Fitting A2 Model on CAL600 and Predicting Public ---", flush=True)
    # Load CAL adapted cache
    cached_cal_adapted = pickle.loads(CAL_ADAPTED_CACHE_PATH.read_bytes())
    cal_adapted_scores = cached_cal_adapted["scores"]
    fl_cal_ad = min(v for q in cal_adapted_scores for v in cal_adapted_scores[q].values())
    aligned_cal_adapted = {
        q: {d: cal_adapted_scores.get(q, {}).get(d, fl_cal_ad) for d in extended[q]}
        for q in all_ids
    }

    channels_cal_a2 = {
        **full_channels_cv,
        "legal_section_ce": aligned_cal_adapted,
    }
    cal_rows_a2_base, _ = ltr_features(
        d1_views_cal, D1_VIEWS, extended, all_ids, channels_cal_a2
    )
    cal_rows_a2 = {
        q: np.concatenate([cal_rows_a2_base[q], type_rows[q], cite_rows[q]], axis=1)
        for q in all_ids
    }

    X_train_a2 = np.vstack([cal_rows_a2[q] for q in all_ids])
    y_train_a2 = np.concatenate(
        [[d in gold[q] for d in d1_groups[q]] for q in all_ids]
    ).astype(np.int8)

    print(f"Fitting A2 LTR model on CAL600 (Shape: {X_train_a2.shape}, Dim: {X_train_a2.shape[1]})...", flush=True)
    scaler_a2 = StandardScaler().fit(X_train_a2)
    ltr_a2 = LogisticRegression(
        C=0.15,
        class_weight="balanced",
        solver="liblinear",
        max_iter=3000,
        random_state=2026,
    )
    ltr_a2.fit(scaler_a2.transform(X_train_a2), y_train_a2)

    # Public 50D features
    fl_pub_ad = min(v for q in adapted_pub_scores for v in adapted_pub_scores[q].values())
    aligned_pub_adapted = {
        q: {d: adapted_pub_scores.get(q, {}).get(d, fl_pub_ad) for d in public_candidates[q]}
        for q in public_ids
    }
    public_scores_a2 = {
        **public_scores_base,
        "legal_section_ce": aligned_pub_adapted,
    }

    public_rows_base_a2, _ = ltr_features(
        view_rank_pub, D1_VIEWS, public_candidates, public_ids, public_scores_a2
    )
    public_rows_a2 = {
        q: np.concatenate(
            [public_rows_base_a2[q], public_t_rows[q], public_c_rows[q]], axis=1
        )
        for q in public_ids
    }

    preds_a2: Dict[str, Dict[str, List[str]]] = {}
    scores_pub_a2: Dict[str, np.ndarray] = {}

    for q in public_ids:
        sc = ltr_a2.decision_function(scaler_a2.transform(public_rows_a2[q]))
        scores_pub_a2[q] = sc
        order = np.argsort(-sc)
        fused = [public_candidates[q][i] for i in order]
        preds_a2[q] = {"answer": [d for d in fused if d in valid_docs][:5]}

    # 7. Step 6d: Materialize and Validate Submission ZIP
    print("
--- Running Step 6d: Materializing Submission Candidate ZIP ---", flush=True)
    cand_json_path = RESULTS_DIR / "CANDIDATE_D1_ADAPTED_SECTION_CHANNEL.json"
    cand_zip_path = RESULTS_DIR / "CANDIDATE_D1_ADAPTED_SECTION_CHANNEL.zip"

    cand_json_path.write_text(json.dumps(preds_a2, ensure_ascii=False, indent=2), encoding="utf-8")
    with zipfile.ZipFile(cand_zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("submission.json", cand_json_path.read_bytes())

    validate_submission_zip(cand_zip_path, cand_json_path, valid_docs)
    zip_sha = sha256_file(cand_zip_path)
    json_sha = sha256_file(cand_json_path)
    print(f"Validated {cand_zip_path} (SHA256: {zip_sha})", flush=True)

    # 8. Compute Churn vs Current D1 Submission
    set_churn = 0
    ord_churn = 0
    rank5_changes = 0

    for q in public_ids:
        c_ans = champion_preds[q]
        a2_ans = preds_a2[q]["answer"]
        if set(c_ans) != set(a2_ans):
            set_churn += 1
        if c_ans != a2_ans:
            ord_churn += 1
        if c_ans[4] != a2_ans[4]:
            rank5_changes += 1

    print(f"
--- Public Churn vs Current D1 Submission ---")
    print(f"Top-5 Set Churn:     {set_churn} / 1000 ({set_churn / 10.0:.2f}%)")
    print(f"Top-5 Ordered Churn: {ord_churn} / 1000 ({ord_churn / 10.0:.2f}%)")
    print(f"Rank-5 Boundary Changes: {rank5_changes} / 1000 ({rank5_changes / 10.0:.2f}%)")

    submission_manifest = {
        "schema_version": "dsc2026.gemini.huy_d1_adapted_section_channel_public_v1.submission_manifest.v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_info["head_commit"],
        "status": "MATERIALIZED",
        "package_name": "CANDIDATE_D1_ADAPTED_SECTION_CHANNEL",
        "json_path": str(cand_json_path.relative_to(ROOT)).replace("\\", "/"),
        "zip_path": str(cand_zip_path.relative_to(ROOT)).replace("\\", "/"),
        "sha256_zip": zip_sha,
        "sha256_json": json_sha,
        "total_public_queries": len(public_ids),
        "comparison_vs_d1_champion": {
            "champion_zip_path": str(D1_CHAMPION_ZIP_PATH.relative_to(ROOT)).replace("\\", "/"),
            "champion_zip_sha256": sha256_file(D1_CHAMPION_ZIP_PATH),
            "top5_set_churn": set_churn,
            "top5_set_churn_pct": set_churn / 10.0,
            "top5_ordered_churn": ord_churn,
            "top5_ordered_churn_pct": ord_churn / 10.0,
            "rank5_boundary_changes": rank5_changes,
        },
    }

    manifest_path = RESULTS_DIR / "PUBLIC_SUBMISSION_MANIFEST.json"
    manifest_path.write_text(json.dumps(submission_manifest, indent=2), encoding="utf-8")
    print(f"Wrote {manifest_path}", flush=True)

    return submission_manifest


if __name__ == "__main__":
    run_public_stage()
