"""Public evaluation, parity check, and packaging for promoted J1 candidate."""

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
import torch
from peft import PeftModel
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path("D:/Study/DSC2026/sota")
if str(ROOT) not in sys.path:
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
import src.gemini.huy_d1_jina_ft_continuation_v1.common as common

RES_DIR = ROOT / "results/gemini/huy_d1_jina_ft_continuation_v1"
ADAPTER_DIR = RES_DIR / "jina_ft_continued_adapter"
PROD_D1_JSON = ROOT / "results/gemini/huy_vnlegal_rank_ablation_v1/CANDIDATE_D1_VNLEGAL_SCORE_ONLY.json"

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


def run_public_work() -> dict:
    started = time.perf_counter()
    print("=== Running Public Work and Verification ===", flush=True)

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

    # 2. Public metadata and retrieval views
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

    # 3. Train J0 full LTR on CAL600 and check Public Parity
    print("Training J0 full LTR on all CAL600...", flush=True)
    train_rows, train_groups = ltr_features(
        local_views, D1_VIEWS, extended, all_ids, full_channels_cv
    )
    for q in train_rows:
        train_rows[q] = np.concatenate(
            [train_rows[q], type_rows[q], cite_rows[q]], axis=1
        )

    X_train_full = np.vstack([train_rows[q] for q in all_ids])
    y_train_full = np.concatenate(
        [[d in gold[q] for d in train_groups[q]] for q in all_ids]
    ).astype(np.int8)

    scaler_j0 = StandardScaler().fit(X_train_full)
    model_j0 = LogisticRegression(
        C=0.15,
        class_weight="balanced",
        solver="liblinear",
        max_iter=3000,
        random_state=2026,
    )
    model_j0.fit(scaler_j0.transform(X_train_full), y_train_full)

    # Predict Public J0
    print("Generating Public predictions for J0...", flush=True)
    pub_rows, pub_groups = ltr_features(
        view_rank_pub, D1_VIEWS, public_candidates, public_ids, public_scores_base
    )
    for q in pub_rows:
        pub_rows[q] = np.concatenate(
            [pub_rows[q], public_t_rows[q], public_c_rows[q]], axis=1
        )

    j0_public_preds = {}
    for q in public_ids:
        X_p = scaler_j0.transform(pub_rows[q])
        scores_p = model_j0.decision_function(X_p)
        order_p = sorted(
            range(len(scores_p)), key=lambda i: scores_p[i], reverse=True
        )
        j0_public_preds[q] = {"answer": [pub_groups[q][i] for i in order_p[:5]]}

    # Verify against CANDIDATE_D1_VNLEGAL_SCORE_ONLY.json
    print("Checking Section 23 Public D1 Parity...", flush=True)
    with open(PROD_D1_JSON, "r", encoding="utf-8") as f:
        baseline_prod_d1 = json.load(f)

    exact_matches = sum(
        1
        for q in public_ids
        if j0_public_preds[q]["answer"] == baseline_prod_d1[q]["answer"]
    )
    print(f"Public D1 Parity: {exact_matches}/1000 identical ordered Top-5")
    if exact_matches != 1000:
        print("FATAL: BLOCKED_PUBLIC_PARITY", flush=True)
        return {"status": "BLOCKED_PUBLIC_PARITY"}

    # 4. Score Public pool with adapted Jina model
    pub_adapted_pkl = RES_DIR / "jina_ft_continued_public.pkl"
    if not pub_adapted_pkl.exists():
        print(f"Scoring Public candidates with adapted model -> {pub_adapted_pkl}...", flush=True)
        base_model, tok = common.load_jina_base_with_shipped_weights()
        adapted_model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
        common.patch_tuple_returning_lora(adapted_model)
        adapted_model.eval().to("cuda")

        public_adapted_scores = {}
        for i, q in enumerate(public_ids, 1):
            text = public_meta[q]
            cands = public_candidates[q]
            owners, passages = [], []
            for d in cands:
                p_list = common.top_passages(text, docs[d], count=2, window=220, overlap=70)
                for p in p_list:
                    owners.append(d)
                    passages.append(p)

            sentence_pairs = [(text, p) for p in passages]
            raw_scores = []
            with torch.no_grad():
                for b_start in range(0, len(sentence_pairs), 16):
                    b_chunk = sentence_pairs[b_start : b_start + 16]
                    inputs = tok(
                        [c[0] for c in b_chunk],
                        [c[1] for c in b_chunk],
                        padding=True,
                        truncation=True,
                        return_tensors="pt",
                        max_length=512,
                    ).to("cuda")
                    with torch.amp.autocast("cuda", dtype=torch.float16):
                        logits = adapted_model(**inputs).logits.view(-1).float()
                        sig = torch.sigmoid(logits).cpu().numpy().tolist()
                        if isinstance(sig, float):
                            sig = [sig]
                        raw_scores.extend(sig)

            ds = {}
            for d, s in zip(owners, raw_scores):
                ds[d] = max(ds.get(d, -1e9), float(s))
            public_adapted_scores[q] = ds

            if i % 100 == 0 or i == len(public_ids):
                print(f"  Public scored {i}/{len(public_ids)} queries", flush=True)

        with open(pub_adapted_pkl, "wb") as f:
            pickle.dump(public_adapted_scores, f, protocol=5)
        print(f"Saved {pub_adapted_pkl}", flush=True)
    else:
        print(f"Loaded existing {pub_adapted_pkl}", flush=True)
        with open(pub_adapted_pkl, "rb") as f:
            public_adapted_scores = pickle.load(f)

    # 5. Train J1 model on CAL600 with adapted Jina score channel
    print("Training J1 full LTR on all CAL600 with adapted Jina score channel...", flush=True)
    with open(RES_DIR / "jina_ft_continued_cv.pkl", "rb") as f:
        cal_adapted_raw = pickle.load(f)
    fl_cal = min(v for q in cal_adapted_raw for v in cal_adapted_raw[q].values())
    cal_adapted_aligned = {
        q: {d: cal_adapted_raw.get(q, {}).get(d, fl_cal) for d in extended[q]}
        for q in all_ids
    }
    j1_channels_cv = {**full_channels_cv, "jina_ft": cal_adapted_aligned}

    train_rows_j1, _ = ltr_features(
        local_views, D1_VIEWS, extended, all_ids, j1_channels_cv
    )
    for q in train_rows_j1:
        train_rows_j1[q] = np.concatenate(
            [train_rows_j1[q], type_rows[q], cite_rows[q]], axis=1
        )

    X_train_j1 = np.vstack([train_rows_j1[q] for q in all_ids])
    scaler_j1 = StandardScaler().fit(X_train_j1)
    model_j1 = LogisticRegression(
        C=0.15,
        class_weight="balanced",
        solver="liblinear",
        max_iter=3000,
        random_state=2026,
    )
    model_j1.fit(scaler_j1.transform(X_train_j1), y_train_full)

    # 6. Predict Public J1
    fl_pub = min(v for q in public_adapted_scores for v in public_adapted_scores[q].values())
    pub_adapted_aligned = {
        q: {d: public_adapted_scores.get(q, {}).get(d, fl_pub) for d in public_candidates[q]}
        for q in public_ids
    }
    public_scores_j1 = {**public_scores_base, "jina_ft": pub_adapted_aligned}

    pub_rows_j1, pub_groups_j1 = ltr_features(
        view_rank_pub, D1_VIEWS, public_candidates, public_ids, public_scores_j1
    )
    for q in pub_rows_j1:
        pub_rows_j1[q] = np.concatenate(
            [pub_rows_j1[q], public_t_rows[q], public_c_rows[q]], axis=1
        )

    j1_public_preds = {}
    for q in public_ids:
        X_p = scaler_j1.transform(pub_rows_j1[q])
        scores_p = model_j1.decision_function(X_p)
        order_p = sorted(
            range(len(scores_p)), key=lambda i: scores_p[i], reverse=True
        )
        j1_public_preds[q] = {"answer": [pub_groups_j1[q][i] for i in order_p[:5]]}

    # 7. Write candidate JSON files and create submission ZIPs
    ctrl_json = RES_DIR / "CONTROL_D1_5VIEW.json"
    ctrl_zip = RES_DIR / "CONTROL_D1_5VIEW.zip"
    cand_json = RES_DIR / "CANDIDATE_J1_D1_JINA_CONTINUED.json"
    cand_zip = RES_DIR / "CANDIDATE_J1_D1_JINA_CONTINUED.zip"
    prom_zip = RES_DIR / "PROMOTED.zip"

    with open(ctrl_json, "w", encoding="utf-8") as f:
        json.dump(j0_public_preds, f, indent=2)
    with zipfile.ZipFile(ctrl_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("submission.json", ctrl_json.read_bytes())

    with open(cand_json, "w", encoding="utf-8") as f:
        json.dump(j1_public_preds, f, indent=2)
    with zipfile.ZipFile(cand_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("submission.json", cand_json.read_bytes())
    with zipfile.ZipFile(prom_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("submission.json", cand_json.read_bytes())

    # Validate ZIP files
    validate_submission_zip(ctrl_zip, ctrl_json, valid_docs)
    validate_submission_zip(cand_zip, cand_json, valid_docs)
    validate_submission_zip(prom_zip, cand_json, valid_docs)

    # 8. Compute Public Churn
    changed_top5 = 0
    changed_ordering = 0
    jaccards = []
    entering = 0
    leaving = 0
    rank5_changes = 0

    for q in public_ids:
        c0 = j0_public_preds[q]["answer"]
        c1 = j1_public_preds[q]["answer"]
        s0 = set(c0)
        s1 = set(c1)

        if s0 != s1:
            changed_top5 += 1
            entering += len(s1 - s0)
            leaving += len(s0 - s1)
        if c0 != c1:
            changed_ordering += 1
        if c0[4] != c1[4]:
            rank5_changes += 1

        jacc = len(s0 & s1) / len(s0 | s1) if (s0 | s1) else 1.0
        jaccards.append(jacc)

    churn = {
        "total_queries": len(public_ids),
        "changed_top5_sets": changed_top5,
        "changed_top5_sets_pct": (changed_top5 / len(public_ids)) * 100.0,
        "changed_ordered_outputs": changed_ordering,
        "changed_ordered_outputs_pct": (changed_ordering / len(public_ids)) * 100.0,
        "mean_top5_jaccard": float(np.mean(jaccards)),
        "entering_docs_count": entering,
        "leaving_docs_count": leaving,
        "rank5_boundary_changes": rank5_changes,
    }

    pub_audit = {
        "experiment_id": "HUY_D1_JINA_FT_CONTINUATION_V1",
        "control_zip": {
            "path": str(ctrl_zip).replace("\\", "/"),
            "sha256": sha256_file(ctrl_zip),
            "md5": md5_file(ctrl_zip),
            "size_bytes": ctrl_zip.stat().st_size,
        },
        "candidate_zip": {
            "path": str(cand_zip).replace("\\", "/"),
            "sha256": sha256_file(cand_zip),
            "md5": md5_file(cand_zip),
            "size_bytes": cand_zip.stat().st_size,
        },
        "promoted_zip": {
            "path": str(prom_zip).replace("\\", "/"),
            "sha256": sha256_file(prom_zip),
            "md5": md5_file(prom_zip),
            "size_bytes": prom_zip.stat().st_size,
        },
        "public_parity": {
            "exact_matches_against_baseline": exact_matches,
            "total_queries": 1000,
            "status": "PASS",
        },
        "public_churn": churn,
        "runtime_seconds": time.perf_counter() - started,
    }

    with open(RES_DIR / "PUBLIC_JINA_AUDIT.json", "w", encoding="utf-8") as f:
        json.dump(pub_audit, f, indent=2)

    print(f"Wrote {RES_DIR / 'PUBLIC_JINA_AUDIT.json'}", flush=True)
    print(
        f"Public Churn: changed_top5={changed_top5} ({changed_top5/10:.1f}%), "
        f"changed_ordering={changed_ordering} ({changed_ordering/10:.1f}%), "
        f"mean_jaccard={churn['mean_top5_jaccard']:.4f}",
        flush=True,
    )
    return pub_audit


if __name__ == "__main__":
    run_public_work()
