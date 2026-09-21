#!/usr/bin/env python
"""ONE-SHOT private approximate-D1 + frozen REL_L0 materializer.

Historical public cpu_top20 generator could not be recovered exactly.
This final private runner therefore uses the best unlabeled-output-matched
CPU reconstruction found in the final recovery audit:

  Large-LTR: XGB depth=2, lr=.05, trees=140, pairs=10
  Legal-LTR: C=1.0
  Robust blend: alpha=.30, rrf_k=60
  Pair/profile/graph/final stack: documented production contract
  Final stack weights: robust=.40, pair=.30, profile=.15, graph=.15, k=0

The public historical artifact match of this approximation was:
  exact ordered Top5 = 784/1000
  mean Top5 set overlap = 0.9736
  exact ordered Top20 = 16/1000
  mean Top20 set overlap = 0.9574

This is NOT the exact historical D1 base generator. It is intentionally
treated as a new approximate private arm.

After approximate D1 inference, the script applies the frozen REL_L0 policy:
  REL = CE(rank5) - median(CE(rank1..4))
  drop rank5 iff REL < -3.0393552780151367.

Everything runs end-to-end and packages submission ZIPs.
All new caches are isolated under:
  results/manual/huy_private_d1_rel_l0_approx_v1

The script still runs the exact cached-public downstream D1 parity check
(1000/1000) to protect the D1 learner/feature implementation itself.
"""
from __future__ import annotations

import argparse, gc, hashlib, importlib.util, json, os, pickle, sqlite3, sys, threading, time, zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

REL_L0 = -3.0393552780151367
SALT = "dsc2026-endgame-ce-rank5-veto-v1"
EXPECTED_FOLD0_DEV_GOLD_RANK5 = 17
D1_VIEWS = ["base", "expanded", "jina", "dense", "corpus"]
E5_MODEL_ID = "intfloat/multilingual-e5-large"
REL_E5_MODEL_ID = "mainguyen9/vietlegal-e5"
E5_DIM = 1024
EXPECTED_EVIDENCE_DOCS = 8507
EXPECTED_EVIDENCE_CHUNKS = 343347

FINAL_CPU_RECOVERY_REL = "results/manual/huy_private_d1_rel_l0_v1/final_cpu_recovery"
FINAL_CPU_EXPECTED = {
    "large": {"depth": 2, "rate": 0.05, "trees": 140, "pairs": 10},
    "legal": {"C": 1.0},
    "robust_alpha": 0.30,
    "robust_k": 60,
}
FINAL_CPU_PUBLIC_PARITY = {
    "exact_top5_order": 784,
    "mean_top5_overlap": 0.9736,
    "exact_top20_order": 16,
    "mean_top20_overlap": 0.9574,
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def digest_json(x) -> str:
    return hashlib.sha256(json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def load_pickle(path: Path):
    return pickle.loads(path.read_bytes())


def save_pickle(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_bytes(pickle.dumps(obj, protocol=5))
    os.replace(tmp, path)


def unwrap_scores(obj):
    return obj["scores"] if isinstance(obj, dict) and isinstance(obj.get("scores"), dict) else obj


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def zip_exact(json_path: Path, zip_path: Path):
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        zf.write(json_path, arcname="submission.json")
    with zipfile.ZipFile(zip_path, "r") as zf:
        if zf.namelist() != ["submission.json"] or zf.read("submission.json") != json_path.read_bytes():
            raise RuntimeError("ZIP integrity failed")


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(m)
    return m


def tree_sha256(path: Path) -> str:
    h = hashlib.sha256()
    files = sorted((p for p in path.rglob("*") if p.is_file()), key=lambda p: p.relative_to(path).as_posix())
    if not files:
        raise RuntimeError(f"Empty model directory: {path}")
    for p in files:
        h.update(p.relative_to(path).as_posix().encode())
        h.update(bytes.fromhex(sha256(p)))
    return h.hexdigest()


def load_questions(path: Path):
    raw = json.loads(path.read_text(encoding="utf-8"))
    ids = [str(q) for q in raw]
    questions = {str(q): str(v["question"] if isinstance(v, dict) else v) for q, v in raw.items()}
    if not ids:
        raise RuntimeError(f"No questions in {path}")
    return ids, questions


def train_exact_d1(root: Path, documents):
    from tune_corpus_cap32_fusion import build_training_cap
    from tune_expanded_fusion_selection import ltr_features
    from tune_doctype_features import build_type_table, type_features
    from tune_citation_graph import build_citation_table, citation_features

    queries, _, ids, extended, local_views, base_scores = build_training_cap(
        root, 32, "results/corpus_index/holdout_extended_scores_cap32.pkl", depth=20
    )
    gold = {q: set(queries[q][1]) for q in ids}

    def load(rel):
        return unwrap_scores(load_pickle(root / rel))

    def aligned(rel, floor=None):
        obj = load(rel)
        if floor is None:
            floor = min(v for q in obj for v in obj[q].values())
        return {q: {d: obj.get(q, {}).get(d, floor) for d in extended[q]} for q in ids}

    vnlegal = load("results/embedding_finetune/vnlegal_lal_cv_scores.pkl")
    crossenc = aligned("results/crossenc_fullpool/cv_scores.pkl", -11.5)
    aift = aligned("results/from_drive/aiteamvn_ft_cv.pkl")
    jinaft = aligned("results/from_drive/jina_ft_cv.pkl")
    title = aligned("results/burst_fresh_block/title_embed_scores.pkl")
    channels = {**base_scores, "vnlegal_lal": vnlegal, "crossenc": crossenc,
                "aiteamvn_ft": aift, "jina_ft": jinaft, "title_embed": title}

    types = build_type_table(root, documents, ids, extended)
    trows = type_features(extended, types, queries, ids)
    own, cited = build_citation_table(documents, ids, extended)
    crows = citation_features(extended, own, cited, ids)
    rows0, groups = ltr_features(local_views, D1_VIEWS, extended, ids, channels)
    rows = {q: np.concatenate([rows0[q], trows[q], crows[q]], axis=1) for q in ids}
    X = np.vstack([rows[q] for q in ids])
    y = np.concatenate([[d in gold[q] for d in groups[q]] for q in ids]).astype(np.int8)
    if X.shape[1] != 48:
        raise RuntimeError(f"D1 feature drift: {X.shape[1]} != 48")
    scaler = StandardScaler().fit(X)
    model = LogisticRegression(C=.15, class_weight="balanced", solver="liblinear", max_iter=3000, random_state=2026)
    model.fit(scaler.transform(X), y)
    floors = {
        "crossenc": min(min(v.values()) for v in crossenc.values() if v),
        "aiteamvn_ft": min(v for q in aift for v in aift[q].values()),
        "jina_ft": min(v for q in jinaft for v in jinaft[q].values()),
        "title_embed": min(v for q in title for v in title[q].values()),
    }
    print(f"  D1 trained on {len(ids)} CAL queries, rows={len(y)}, dim=48", flush=True)
    return scaler, model, {"cal_queries": len(ids), "rows": len(y), "positives": int(y.sum()), "feature_dim": 48, "floors": floors}


def cached_public_d1_parity(root: Path, documents, scaler, model):
    """Mandatory downstream parity using authoritative historical public caches."""
    from tune_expanded_fusion_selection import ltr_features
    from tune_doctype_features import build_type_table, type_features
    from tune_citation_graph import build_citation_table, citation_features
    from tune_burst_multistage_posterior import weighted_rrf
    from run_burst_expanded_fusion_submission import RERANK_CONFIG

    data = root / "DSC2026-LegalIR-main/v4_run/public_test_dataset"
    ids, questions = load_questions(data / "public-official.json")
    if len(ids) != 1000:
        raise RuntimeError(f"Public q count drift: {len(ids)}")

    base = load_pickle(root / "results/burst_gpu_threeview/cpu_top20.pkl")["rankings"]
    retrieval_obj = None
    for p in [root / "results/burst_multistage/public_retrieval.pkl",
              root / "results/burst_robust_fusion/public_retrieval.pkl",
              root / "results/burst_expanded_fusion/public_retrieval.pkl"]:
        if p.is_file():
            obj = load_pickle(p)
            cache = obj.get("cache", {})
            if all(q in cache for q in ids):
                retrieval_obj = cache
                break
    if retrieval_obj is None:
        raise RuntimeError("Missing complete historical public retrieval cache")

    from benchmark_dense_expansion_holdouts import raw_union
    from run_burst_expanded_fusion_submission import EXPANSION_CONFIG, CORPUS_DEPTH
    raw = {q: raw_union(retrieval_obj[q], EXPANSION_CONFIG["depth"]) for q in ids}
    exp_saved = load_pickle(root / "results/burst_expanded_fusion/expansion_scores.pkl")["scores"]
    dense_rank = {q: sorted(raw[q], key=lambda d: (-exp_saved[q][d], d)) for q in ids}
    expanded_all = weighted_rrf([raw, dense_rank], RERANK_CONFIG["expansion_weights"], RERANK_CONFIG["expansion_rrf_k"])
    expanded = {q: expanded_all[q][:RERANK_CONFIG["expanded_depth"]] for q in ids}
    corpus_obj = load_pickle(root / "results/burst_expanded_fusion/corpus_rank_cap32.pkl")
    corpus_rank, corpus_score = corpus_obj["ranking"], corpus_obj["scores"]
    candidates = {q: list(dict.fromkeys(list(base[q]) + expanded[q] + corpus_rank[q][:CORPUS_DEPTH])) for q in ids}
    corpus_sparse = {q: {d: corpus_score[q][d] for d in candidates[q] if d in corpus_score[q]} for q in ids}
    expansion = {q: {d: exp_saved[q][d] for d in candidates[q] if d in exp_saved[q]} for q in ids}
    rr = load_pickle(root / "results/burst_expanded_fusion/rerank_scores.pkl")

    three = unwrap_scores(load_pickle(root / "results/burst_gpu_threeview/gpu_scores.checkpoint.pkl"))
    vn = unwrap_scores(load_pickle(root / "results/burst_userft_maxrecall/vnlegal_scores.pkl"))
    ce = unwrap_scores(load_pickle(root / "results/crossenc_fullpool/public_scores.pkl"))
    ai = unwrap_scores(load_pickle(root / "results/from_drive/aiteamvn_ft_public.pkl"))
    jf = unwrap_scores(load_pickle(root / "results/from_drive/jina_ft_public.pkl"))
    ti = unwrap_scores(load_pickle(root / "results/burst_fresh_block/title_embed_public.pkl"))
    cvce = unwrap_scores(load_pickle(root / "results/crossenc_fullpool/cv_scores.pkl"))
    cvai = unwrap_scores(load_pickle(root / "results/from_drive/aiteamvn_ft_cv.pkl"))
    cvjf = unwrap_scores(load_pickle(root / "results/from_drive/jina_ft_cv.pkl"))
    cvti = unwrap_scores(load_pickle(root / "results/burst_fresh_block/title_embed_scores.pkl"))
    floors = {
        "crossenc": min(min(v.values()) for v in cvce.values() if v),
        "aiteamvn_ft": min(v for q in cvai for v in cvai[q].values()),
        "jina_ft": min(v for q in cvjf for v in cvjf[q].values()),
        "title_embed": min(v for q in cvti for v in cvti[q].values()),
    }

    views = {
        "base": {q: list(base[q]) for q in ids},
        "expanded": expanded,
        "jina": {q: sorted(candidates[q], key=lambda d: (-rr["jina"][q][d], d)) for q in ids},
        "dense": {q: sorted(candidates[q], key=lambda d: (-rr["dense"][q][d], d)) for q in ids},
        "corpus": {q: sorted((d for d in candidates[q] if d in corpus_sparse[q]), key=lambda d: (-corpus_sparse[q][d], d)) for q in ids},
    }
    scores = {
        "jina": rr["jina"], "dense": rr["dense"], "expansion": expansion,
        "e5": {q: three[q]["e5"] for q in ids},
        "corpus": {q: {d: corpus_sparse[q].get(d, -1.0) for d in candidates[q]} for q in ids},
        "vnlegal_lal": vn,
        "crossenc": {q: {d: ce.get(q, {}).get(d, floors["crossenc"]) for d in candidates[q]} for q in ids},
        "aiteamvn_ft": {q: {d: ai.get(q, {}).get(d, floors["aiteamvn_ft"]) for d in candidates[q]} for q in ids},
        "jina_ft": {q: {d: jf.get(q, {}).get(d, floors["jina_ft"]) for d in candidates[q]} for q in ids},
        "title_embed": {q: {d: ti.get(q, {}).get(d, floors["title_embed"]) for d in candidates[q]} for q in ids},
    }
    qmeta = {q: (questions[q], set()) for q in ids}
    types = build_type_table(root, documents, ids, candidates)
    trows = type_features(candidates, types, qmeta, ids)
    own, cited = build_citation_table(documents, ids, candidates)
    crows = citation_features(candidates, own, cited, ids)
    rows0, groups = ltr_features(views, D1_VIEWS, candidates, ids, scores)
    pred = {}
    for q in ids:
        X = np.concatenate([rows0[q], trows[q], crows[q]], axis=1)
        s = model.decision_function(scaler.transform(X))
        order = np.argsort(-s)
        pred[q] = [groups[q][i] for i in order[:5]]
    champion_raw = json.loads((root / "results/gemini/huy_vnlegal_rank_ablation_v1/CANDIDATE_D1_VNLEGAL_SCORE_ONLY.json").read_text(encoding="utf-8"))
    champion = {str(q): [str(d) for d in row["answer"]] for q, row in champion_raw.items()}
    matches = sum(pred[q] == champion[q] for q in ids)
    if matches != 1000:
        bad = [q for q in ids if pred[q] != champion[q]][:5]
        raise RuntimeError(f"BLOCKED cached-public D1 parity {matches}/1000, sample={bad}")
    print("  Cached-public D1 parity: 1000/1000 PASS", flush=True)
    return {"exact_matches": 1000, "champion_sha256": sha256(root / "results/gemini/huy_vnlegal_rank_ablation_v1/CANDIDATE_D1_VNLEGAL_SCORE_ONLY.json")}


def resolve_or_build_fts_db(
    *,
    root: Path,
    data_dir: Path,
    cache_dir: Path,
    explicit_db: Path | None,
) -> tuple[Path, dict]:
    """
    Resolve the exact BURST SQLite FTS5 database, or rebuild it from the
    canonical selected-contexts corpus using the repository's own builder.

    Resolution order
    ----------------
    1. --db supplied by user
    2. repo/benchmarks/legalir_full_fts.sqlite
    3. sibling LegalIR/benchmarks/legalir_full_fts.sqlite
    4. parent/benchmarks/legalir_full_fts.sqlite
    5. isolated private cache DB (built if absent)

    Rebuild contract is the historical benchmark default:
      chunk_size=500, overlap=100
    """
    from benchmark_burst_v4_full_sqlite import (
        build_database,
        load_dataset,
    )

    candidates = []
    if explicit_db is not None:
        candidates.append(explicit_db.expanduser().resolve())

    candidates.extend([
        (root / "benchmarks/legalir_full_fts.sqlite").resolve(),
        (root.parent / "LegalIR/benchmarks/legalir_full_fts.sqlite").resolve(),
        (root.parent / "benchmarks/legalir_full_fts.sqlite").resolve(),
    ])

    seen = set()
    for p in candidates:
        ps = str(p).lower()
        if ps in seen:
            continue
        seen.add(ps)
        if p.is_file():
            # Validate that this is actually the expected corpus/index contract.
            try:
                con = sqlite3.connect(
                    f"file:{p.as_posix()}?mode=ro",
                    uri=True,
                )
                meta = dict(
                    con.execute(
                        "SELECT key,value FROM metadata"
                    ).fetchall()
                )
                con.close()
                if (
                    int(meta.get("chunk_size", -1)) == 500
                    and int(meta.get("overlap", -1)) == 100
                    and int(meta.get("documents", -1)) > 0
                ):
                    print(
                        f"  FTS DB: reusing {p} "
                        f"(docs={meta.get('documents')} "
                        f"chunks={meta.get('chunks')} "
                        f"chunk=500 overlap=100)",
                        flush=True,
                    )
                    return p, {
                        "mode": "REUSED_EXISTING",
                        "path": str(p),
                        "documents": int(meta["documents"]),
                        "chunks": int(meta.get("chunks", -1)),
                        "chunk_size": 500,
                        "overlap": 100,
                    }
                else:
                    print(
                        f"  FTS candidate ignored due contract mismatch: {p} "
                        f"meta={meta}",
                        flush=True,
                    )
            except Exception as exc:
                print(
                    f"  FTS candidate ignored (not valid expected DB): "
                    f"{p} :: {exc}",
                    flush=True,
                )

    # Nothing reusable: construct an isolated DB for the private run.
    build_path = (cache_dir / "legalir_full_fts.sqlite").resolve()
    build_path.parent.mkdir(parents=True, exist_ok=True)

    print(
        "  No reusable legalir_full_fts.sqlite found.\n"
        f"  Building isolated FTS5 index -> {build_path}",
        flush=True,
    )
    docs, _ = load_dataset(data_dir)
    if not docs:
        raise RuntimeError(
            f"Cannot build FTS DB: no documents loaded from {data_dir}"
        )

    con, chunks = build_database(
        build_path,
        docs,
        chunk_size=500,
        overlap=100,
    )
    meta = dict(
        con.execute("SELECT key,value FROM metadata").fetchall()
    )
    con.close()

    if (
        int(meta.get("documents", -1)) != len(docs)
        or int(meta.get("chunk_size", -1)) != 500
        or int(meta.get("overlap", -1)) != 100
    ):
        raise RuntimeError(
            f"Fresh FTS DB failed contract validation: {meta}"
        )

    print(
        f"  FTS DB built: docs={len(docs)} chunks={chunks}",
        flush=True,
    )
    return build_path, {
        "mode": "BUILT_PRIVATE_ISOLATED",
        "path": str(build_path),
        "documents": len(docs),
        "chunks": int(chunks),
        "chunk_size": 500,
        "overlap": 100,
    }


def target_retrieval(root: Path, db_path: Path, train, doc_ids, questions, ids, cache_path: Path, workers: int):
    from tune_burst_memory import build_query_memory
    from tune_burst_score_ltr import retrieve
    fp = digest_json([(q, questions[q]) for q in ids])
    cache = {}
    if cache_path.is_file():
        obj = load_pickle(cache_path)
        if obj.get("qids") == ids and obj.get("question_fingerprint") == fp:
            cache = obj.get("cache", {})
    missing = [q for q in ids if q not in cache]
    print(f"  retrieval cache {len(cache)}/{len(ids)} missing={len(missing)}", flush=True)
    if not missing:
        return cache
    db_path = Path(db_path).resolve()
    if not db_path.is_file():
        raise FileNotFoundError(
            f"Resolved FTS database does not exist: {db_path}"
        )
    print(f"  using FTS database: {db_path}", flush=True)
    conn = sqlite3.connect(str(db_path))
    build_query_memory(conn, train)
    conn.close()
    local = threading.local()
    def one(q):
        if not hasattr(local, "conn"):
            local.conn = sqlite3.connect(str(db_path))
        return q, retrieve(local.conn, doc_ids, None, questions[q])
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, (q, result) in enumerate(pool.map(one, missing), 1):
            cache[q] = result
            if i % 25 == 0 or i == len(missing):
                save_pickle(cache_path, {"qids": ids, "question_fingerprint": fp, "cache": cache})
                print(f"    retrieved {i}/{len(missing)} ({time.perf_counter()-started:.1f}s)", flush=True)
    return cache



def resolve_historical_artifact(
    *,
    root: Path,
    rel_path: str,
    validator=None,
) -> tuple[Path, dict]:
    """Resolve historical artifact from current repo or sibling clones."""
    candidates = [
        (root / rel_path).resolve(),
        (root.parent / "LegalIR" / rel_path).resolve(),
        (root.parent / "DSC2026-LegalIR-Huy" / rel_path).resolve(),
    ]
    seen, checked = set(), []
    for p in candidates:
        k = str(p).lower()
        if k in seen:
            continue
        seen.add(k)
        checked.append(p)
        if not p.is_file():
            continue
        obj = load_pickle(p)
        if validator is not None:
            validator(obj)
        print(f"  historical artifact: {rel_path} -> {p}", flush=True)
        return p, obj
    raise FileNotFoundError(
        "Required historical artifact not found in any supported clone:\n"
        + "\n".join(f"  - {p}" for p in checked)
    )


def _validate_large_ltr(obj):
    if not isinstance(obj, dict) or "model" not in obj:
        raise RuntimeError("large_ltr artifact missing model")
    if obj.get("version") not in (None, "large-ltr-v1"):
        raise RuntimeError(f"unexpected large_ltr version={obj.get('version')}")


def _validate_legal_model(obj):
    if not isinstance(obj, dict) or "model" not in obj:
        raise RuntimeError("legal_features artifact missing model")
    if obj.get("feature_version") not in (None, "legal-v1"):
        raise RuntimeError(
            f"unexpected legal feature version={obj.get('feature_version')}"
        )


def _validate_pair_model(obj):
    if not isinstance(obj, dict) or "model" not in obj or "scaler" not in obj:
        raise RuntimeError("pairwise artifact missing model/scaler")
    if obj.get("feature_version") not in (None, "empirical-pairwise-v1"):
        raise RuntimeError(
            f"unexpected pairwise version={obj.get('feature_version')}"
        )


def load_final_recovered_cpu(root: Path):
    """Load and validate the final output-matched CPU reconstruction."""
    rec_dir = root / FINAL_CPU_RECOVERY_REL
    config_path = rec_dir / "BEST_RECOVERED_CONFIG.json"
    large_path = rec_dir / "BEST_RECOVERED_LARGE_MODEL.pkl"
    legal_path = rec_dir / "BEST_RECOVERED_LEGAL_MODEL.pkl"
    pair_path = root / "results/burst_empirical_pairwise/model.pkl"

    for p in (config_path, large_path, legal_path, pair_path):
        if not p.is_file():
            raise FileNotFoundError(
                f"Missing final CPU recovery artifact: {p}\n"
                "Run final_rebuild_cpu_generator_v1.py successfully first."
            )

    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    large_cfg = cfg.get("large_config") or {}
    legal_cfg = cfg.get("legal_config") or {}
    alpha = float(cfg.get("robust_alpha"))
    robust_k = int(cfg.get("robust_k"))

    if large_cfg != FINAL_CPU_EXPECTED["large"]:
        raise RuntimeError(
            f"Recovered Large config drift: {large_cfg} "
            f"!= {FINAL_CPU_EXPECTED['large']}"
        )
    if legal_cfg != FINAL_CPU_EXPECTED["legal"]:
        raise RuntimeError(
            f"Recovered Legal config drift: {legal_cfg} "
            f"!= {FINAL_CPU_EXPECTED['legal']}"
        )
    if abs(alpha - FINAL_CPU_EXPECTED["robust_alpha"]) > 1e-12:
        raise RuntimeError(
            f"Recovered robust alpha drift: {alpha} "
            f"!= {FINAL_CPU_EXPECTED['robust_alpha']}"
        )
    if robust_k != FINAL_CPU_EXPECTED["robust_k"]:
        raise RuntimeError(
            f"Recovered robust k drift: {robust_k} "
            f"!= {FINAL_CPU_EXPECTED['robust_k']}"
        )

    large_model = load_pickle(large_path)
    legal_model = load_pickle(legal_path)
    pair_saved = load_pickle(pair_path)
    _validate_pair_model(pair_saved)

    provenance = {
        "mode": "FINAL_OUTPUT_MATCHED_APPROXIMATION",
        "config_json": str(config_path),
        "config_json_sha256": sha256(config_path),
        "large_model": str(large_path),
        "large_model_sha256": sha256(large_path),
        "legal_model": str(legal_path),
        "legal_model_sha256": sha256(legal_path),
        "pair_model": str(pair_path),
        "pair_model_sha256": sha256(pair_path),
        "large_config": large_cfg,
        "legal_config": legal_cfg,
        "robust_alpha": alpha,
        "robust_k": robust_k,
        "historical_public_output_match": FINAL_CPU_PUBLIC_PARITY,
        "exact_historical_generator_recovered": False,
    }
    return large_model, legal_model, pair_saved, alpha, robust_k, provenance


def target_base_top20(
    root: Path,
    corpus_paths,
    train,
    doc_ids,
    retrieval,
    questions,
    ids,
    cache_path: Path,
):
    from benchmark_burst_v4_full_sqlite import tokens
    from tune_burst_empirical_bayes_ltr import label_frequency
    from tune_burst_empirical_pairwise import features_for, rank as pairwise_rank
    from tune_burst_graph_posterior import build_graph, graph_rerank
    from tune_burst_legal_features import enhanced_features
    from tune_burst_supervised_profile_bm25 import build_profiles, profile_rank
    from tune_burst_multistage_posterior import weighted_rrf
    from tune_burst_pairwise import blend_rankings
    from tune_burst_score_ltr import score_features

    (
        large_model,
        legal_model,
        pair_saved,
        robust_alpha,
        robust_k,
        artifact_provenance,
    ) = load_final_recovered_cpu(root)

    fp = digest_json([(q, questions[q]) for q in ids])
    cfg_fp = digest_json({
        "large": artifact_provenance["large_config"],
        "legal": artifact_provenance["legal_config"],
        "robust_alpha": robust_alpha,
        "robust_k": robust_k,
        "large_sha": artifact_provenance["large_model_sha256"],
        "legal_sha": artifact_provenance["legal_model_sha256"],
        "pair_sha": artifact_provenance["pair_model_sha256"],
        "final_weights": [.40, .30, .15, .15],
        "final_k": 0,
    })

    rankings = {}
    if cache_path.is_file():
        obj = load_pickle(cache_path)
        if (
            obj.get("qids") == ids
            and obj.get("question_fingerprint") == fp
            and obj.get("cpu_config_fingerprint") == cfg_fp
        ):
            rankings = obj.get("rankings", {})

    missing = [q for q in ids if q not in rankings]
    print(
        f"  approximate-base cache {len(rankings)}/{len(ids)} "
        f"missing={len(missing)}",
        flush=True,
    )
    print(
        "  CPU approximation: "
        f"Large={artifact_provenance['large_config']} | "
        f"Legal={artifact_provenance['legal_config']} | "
        f"robust alpha={robust_alpha:.2f}, k={robust_k}",
        flush=True,
    )

    if not missing:
        return rankings, artifact_provenance

    train_ids = list(train)
    profile_model = build_profiles(train, train_ids)
    graph_frequency, graph_adjacency, _ = build_graph(train, train_ids)
    frequency = label_frequency(train, set())

    print("  normalizing corpus for Legal-LTR...", flush=True)
    normalized = {}
    for i, path in enumerate(corpus_paths, 1):
        row = json.loads(path.read_text(encoding="utf-8"))
        did = str(row.get("id", path.stem[len("context_"):]))
        normalized[did] = (
            " " + " ".join(tokens(row.get("passage") or "")) + " "
        )
        if i % 1500 == 0 or i == len(corpus_paths):
            print(f"    normalized {i}/{len(corpus_paths)}", flush=True)

    target_queries = {q: (questions[q], set()) for q in ids}
    pair_features = features_for(
        retrieval,
        target_queries,
        ids,
        frequency,
    )

    started = time.perf_counter()
    for i, q in enumerate(missing, 1):
        lists = retrieval[q]

        # Large-LTR branch.
        c1, x1 = score_features(lists)
        large = [
            c1[j]
            for j in np.argsort(-large_model.predict(x1))[:100]
        ]

        # Legal-LTR branch: production private inference uses lexical_depth=20.
        c2, x2 = enhanced_features(
            lists,
            questions[q],
            normalized,
            lexical_depth=20,
        )
        legal = [
            c2[j]
            for j in np.argsort(
                -legal_model.decision_function(x2)
            )[:100]
        ]

        # Final recovered robust contract.
        robust = blend_rankings(
            {q: large},
            {q: legal[:10]},
            robust_alpha,
            robust_k,
        )[q]

        # Current best/selected empirical-pairwise reconstruction.
        pair = pairwise_rank(
            pair_saved["model"],
            pair_saved["scaler"],
            {q: pair_features[q]},
            [q],
        )[q]

        # Deterministic posterior branches.
        profile = profile_rank(
            questions[q],
            profile_model,
            2,
            1.2,
            .75,
            .3,
        )
        # IMPORTANT: graph is recomputed from THIS recovered robust ranking.
        graph = graph_rerank(
            robust,
            graph_frequency,
            graph_adjacency,
            3,
            .5,
            "conditional",
            3,
            .4,
            0,
        )

        final = weighted_rrf(
            [
                {q: robust},
                {q: pair},
                {q: profile},
                {q: graph},
            ],
            (.40, .30, .15, .15),
            0,
        )[q]

        top20 = list(dict.fromkeys(final))[:20]
        for d in doc_ids:
            if len(top20) >= 20:
                break
            if d not in top20:
                top20.append(d)
        rankings[q] = top20

        if i % 25 == 0 or i == len(missing):
            save_pickle(
                cache_path,
                {
                    "qids": ids,
                    "question_fingerprint": fp,
                    "cpu_config_fingerprint": cfg_fp,
                    "cpu_provenance": artifact_provenance,
                    "rankings": rankings,
                },
            )
            print(
                f"    approximate-base {i}/{len(missing)} "
                f"({time.perf_counter()-started:.1f}s)",
                flush=True,
            )

    return rankings, artifact_provenance


def build_target_candidates(root: Path, cache_dir: Path, questions, ids, retrieval, base, documents, device: str):
    from benchmark_dense_expansion_holdouts import raw_union
    from tune_burst_multistage_posterior import weighted_rrf
    from run_burst_expanded_fusion_submission import CORPUS_CAP, CORPUS_DEPTH, EXPANSION_CONFIG, RERANK_CONFIG, dense_expansion, corpus_dense, rerank
    raw = {q: raw_union(retrieval[q], EXPANSION_CONFIG["depth"]) for q in ids}
    expansion_scores = dense_expansion(root, cache_dir, questions, ids, raw, documents, device)
    dense_rank = {q: sorted(raw[q], key=lambda d: (-expansion_scores[q][d], d)) for q in ids}
    expanded_all = weighted_rrf([raw, dense_rank], RERANK_CONFIG["expansion_weights"], RERANK_CONFIG["expansion_rrf_k"])
    corpus_rank, corpus_score = corpus_dense(root, cache_dir, questions, ids, device, cap=CORPUS_CAP, depth=CORPUS_DEPTH)
    expanded = {q: expanded_all[q][:RERANK_CONFIG["expanded_depth"]] for q in ids}
    candidates = {q: list(dict.fromkeys(list(base[q]) + expanded[q] + corpus_rank[q][:CORPUS_DEPTH])) for q in ids}
    corpus_sparse = {q: {d: corpus_score[q][d] for d in candidates[q] if d in corpus_score[q]} for q in ids}
    expansion_scores = {q: {d: expansion_scores[q][d] for d in candidates[q] if d in expansion_scores[q]} for q in ids}
    rr = rerank(root, cache_dir, questions, ids, candidates, documents, device)
    return candidates, expanded, corpus_sparse, expansion_scores, rr


def patch_transformers_v5():
    import transformers.models.xlm_roberta.modeling_xlm_roberta as module
    if hasattr(module, "create_position_ids_from_input_ids"):
        return
    def helper(input_ids, padding_idx, past_key_values_length=0):
        mask = input_ids.ne(padding_idx).int()
        positions = (torch.cumsum(mask, dim=1) + past_key_values_length) * mask
        return positions.long() + padding_idx
    module.create_position_ids_from_input_ids = helper


@torch.inference_mode()
def e5_mean_encode(model, tok, texts, batch_size, max_length=512):
    import torch.nn.functional as F
    vecs = []
    device = next(model.parameters()).device
    for st in range(0, len(texts), batch_size):
        batch = tok(texts[st:st+batch_size], max_length=max_length, truncation=True, padding=True, return_tensors="pt")
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        hidden = model(**batch).last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1e-9)
        vecs.append(F.normalize(pooled.float(), p=2, dim=1).cpu().numpy())
    return np.vstack(vecs)


def score_e5_channel(root: Path, cache_path: Path, questions, ids, candidates, documents, model_name_or_path: str):
    from transformers import AutoModel, AutoTokenizer
    from benchmark_jina_reranker_holdouts import top_passages
    saved = load_pickle(cache_path) if cache_path.is_file() else {}
    remaining = [q for q in ids if any(d not in saved.get(q, {}) for d in candidates[q])]
    print(f"  e5 cache {len(ids)-len(remaining)}/{len(ids)}", flush=True)
    if not remaining:
        return saved
    source = str(root / model_name_or_path) if (root / model_name_or_path).exists() else model_name_or_path
    tok = AutoTokenizer.from_pretrained(source)
    model = AutoModel.from_pretrained(source, torch_dtype=torch.float16, low_cpu_mem_usage=True).eval().to("cuda")
    started = time.perf_counter()
    for i, q in enumerate(remaining, 1):
        owners, passages = [], []
        for d in candidates[q]:
            for p in top_passages(questions[q], documents[d], count=2):
                owners.append(d); passages.append(p)
        qv = e5_mean_encode(model, tok, ["query: " + questions[q]], 1)[0]
        pv = e5_mean_encode(model, tok, ["passage: " + p for p in passages], 16)
        ds = {d: -1.0 for d in candidates[q]}
        for d, s in zip(owners, pv @ qv):
            ds[d] = max(ds[d], float(s))
        saved[q] = ds
        if i % 25 == 0 or i == len(remaining):
            save_pickle(cache_path, saved)
            print(f"    e5 {i}/{len(remaining)} {(time.perf_counter()-started)/i:.2f}s/q", flush=True)
    del model, tok; gc.collect(); torch.cuda.empty_cache()
    return saved


def score_vnlegal_channel(root: Path, cache_dir: Path, questions, ids, candidates, documents, device):
    from run_vnlegal_extra_channel_submission import ensure_vnlegal_model, score_vnlegal
    ensure_vnlegal_model(root)
    return score_vnlegal(root, cache_dir, questions, ids, candidates, documents, device)


def score_bi_channel(root: Path, cache_path: Path, questions, ids, candidates, documents, model_path: Path):
    from transformers import AutoModel, AutoTokenizer
    from benchmark_aiteamvn_holdouts import encode_cls
    from benchmark_jina_reranker_holdouts import top_passages
    saved = load_pickle(cache_path) if cache_path.is_file() else {}
    remaining = [q for q in ids if any(d not in saved.get(q, {}) for d in candidates[q])]
    print(f"  bi({model_path.name}) cache {len(ids)-len(remaining)}/{len(ids)}", flush=True)
    if not remaining:
        return saved
    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path, dtype=torch.float16, low_cpu_mem_usage=True).eval().to("cuda")
    started = time.perf_counter()
    for i, q in enumerate(remaining, 1):
        owners, passages = [], []
        for d in candidates[q]:
            for p in top_passages(questions[q], documents[d], count=2):
                owners.append(d); passages.append(p)
        qv = encode_cls(model, tok, [questions[q]], 1, 512)[0]
        pv = encode_cls(model, tok, passages, 32, 512)
        ds = {d: -1e9 for d in candidates[q]}
        for d, s in zip(owners, pv @ qv):
            ds[d] = max(ds[d], float(s))
        saved[q] = ds
        if i % 25 == 0 or i == len(remaining):
            save_pickle(cache_path, saved)
            print(f"    bi {i}/{len(remaining)} {(time.perf_counter()-started)/i:.2f}s/q", flush=True)
    del model, tok; gc.collect(); torch.cuda.empty_cache()
    return saved


def score_jina_ft_channel(root: Path, cache_path: Path, questions, ids, candidates, documents, batch_size=16):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from safetensors.torch import load_file
    from benchmark_jina_reranker_holdouts import top_passages

    patch_transformers_v5()
    saved = load_pickle(cache_path) if cache_path.is_file() else {}
    remaining = [q for q in ids if any(d not in saved.get(q, {}) for d in candidates[q])]
    print(f"  jina-ft cache {len(ids)-len(remaining)}/{len(ids)}", flush=True)
    if not remaining:
        return saved

    base = root / "models/jina-reranker-v2-base-multilingual"
    weights = root / "models/from_drive/jina_finetuned/model.safetensors"
    if not base.exists() or not weights.is_file():
        raise FileNotFoundError(f"Missing Jina-FT model: base={base.exists()} weights={weights.exists()}")

    tok = AutoTokenizer.from_pretrained(base, trust_remote_code=True, fix_mistral_regex=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        base, trust_remote_code=True, dtype=torch.float16
    )
    state = {k: v.to(torch.float16) for k, v in load_file(weights).items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise RuntimeError(f"Jina-FT unexpected keys: {unexpected[:5]}")
    model._tokenizer = tok
    model = model.eval().to("cuda")

    started = time.perf_counter()
    for i, q in enumerate(remaining, 1):
        owners, passages = [], []
        for d in candidates[q]:
            for p in top_passages(questions[q], documents[d], count=2):
                owners.append(d); passages.append(p)
        vals = model.compute_score(
            [(questions[q], p) for p in passages],
            batch_size=batch_size,
            max_length=512,
        )
        if isinstance(vals, float):
            vals = [vals]
        ds = {d: -1e9 for d in candidates[q]}
        for d, s in zip(owners, vals):
            ds[d] = max(ds[d], float(s))
        saved[q] = ds
        if i % 25 == 0 or i == len(remaining):
            save_pickle(cache_path, saved)
            print(f"    jina-ft {i}/{len(remaining)} {(time.perf_counter()-started)/i:.2f}s/q", flush=True)

    del model, tok
    gc.collect(); torch.cuda.empty_cache()
    return saved


def score_title_channel(root: Path, cache_path: Path, questions, ids, candidates, documents):
    from transformers import AutoModel, AutoTokenizer
    from benchmark_aiteamvn_holdouts import encode_cls
    from tune_title_features import title_table

    if cache_path.is_file():
        saved = load_pickle(cache_path)
        if set(saved) == set(ids) and all(set(saved[q]) >= set(candidates[q]) for q in ids):
            print("  title_embed cache complete", flush=True)
            return saved

    titles = title_table(documents, ids, candidates)
    model_path = root / "models/AITeamVN_Vietnamese_Embedding"
    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path, dtype=torch.float16).eval().to("cuda")
    dids = list(titles)
    print(f"  title_embed documents={len(dids)}", flush=True)
    tvec = encode_cls(model, tok, [titles[d] or "khong co tieu de" for d in dids], 64, 128)
    tmap = dict(zip(dids, tvec))
    qvec = encode_cls(model, tok, [questions[q] for q in ids], 32, 128)
    qmap = dict(zip(ids, qvec))
    out = {q: {d: float(qmap[q] @ tmap[d]) for d in candidates[q]} for q in ids}
    save_pickle(cache_path, out)
    del model, tok
    gc.collect(); torch.cuda.empty_cache()
    return out


def score_crossenc_channel(root: Path, cache_path: Path, questions, ids, candidates, documents, batch_size=16):
    """Reconstruct the documented AITeamVN/Vietnamese_Reranker score-only channel."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from benchmark_jina_reranker_holdouts import top_passages

    saved = load_pickle(cache_path) if cache_path.is_file() else {}
    remaining = [q for q in ids if any(d not in saved.get(q, {}) for d in candidates[q])]
    print(f"  crossenc cache {len(ids)-len(remaining)}/{len(ids)}", flush=True)
    if not remaining:
        return saved

    path = root / "models/AITeamVN_Vietnamese_Reranker"
    if not path.exists():
        raise FileNotFoundError(path)
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        path, trust_remote_code=True, dtype=torch.float16
    ).eval().to("cuda")

    @torch.inference_mode()
    def score_pairs(pairs):
        if hasattr(model, "compute_score"):
            vals = model.compute_score(pairs, batch_size=batch_size, max_length=512)
            if isinstance(vals, float):
                vals = [vals]
            return [float(x) for x in vals]
        result = []
        for st in range(0, len(pairs), batch_size):
            chunk = pairs[st:st+batch_size]
            enc = tok(
                [a for a, _ in chunk], [b for _, b in chunk],
                max_length=512, truncation=True, padding=True, return_tensors="pt"
            )
            enc = {k: v.to("cuda") for k, v in enc.items()}
            logits = model(**enc).logits
            if logits.ndim == 2 and logits.shape[1] == 1:
                vals = logits[:, 0]
            elif logits.ndim == 2 and logits.shape[1] >= 2:
                vals = logits[:, -1]
            else:
                vals = logits.reshape(-1)
            result.extend(vals.float().cpu().tolist())
        return result

    started = time.perf_counter()
    for i, q in enumerate(remaining, 1):
        owners, passages = [], []
        for d in candidates[q]:
            for p in top_passages(questions[q], documents[d], count=2):
                owners.append(d); passages.append(p)
        vals = score_pairs([(questions[q], p) for p in passages])
        ds = {d: -1e9 for d in candidates[q]}
        for d, s in zip(owners, vals):
            ds[d] = max(ds[d], float(s))
        saved[q] = ds
        if i % 25 == 0 or i == len(remaining):
            save_pickle(cache_path, saved)
            print(f"    crossenc {i}/{len(remaining)} {(time.perf_counter()-started)/i:.2f}s/q", flush=True)

    del model, tok
    gc.collect(); torch.cuda.empty_cache()
    return saved


def infer_target_d1(
    root: Path, documents, ids, questions, candidates, base, expanded,
    corpus_sparse, expansion_scores, rerank_scores, e5_scores, vnlegal_scores,
    crossenc_scores, aiteam_ft_scores, jina_ft_scores, title_scores,
    scaler, model, floors,
):
    from tune_expanded_fusion_selection import ltr_features
    from tune_doctype_features import build_type_table, type_features
    from tune_citation_graph import build_citation_table, citation_features

    view_rank = {
        "base": {q: list(base[q]) for q in ids},
        "expanded": {q: list(expanded[q]) for q in ids},
        "jina": {q: sorted(candidates[q], key=lambda d: (-rerank_scores["jina"][q][d], d)) for q in ids},
        "dense": {q: sorted(candidates[q], key=lambda d: (-rerank_scores["dense"][q][d], d)) for q in ids},
        "corpus": {q: sorted((d for d in candidates[q] if d in corpus_sparse[q]),
                              key=lambda d: (-corpus_sparse[q][d], d)) for q in ids},
    }
    scores = {
        "jina": rerank_scores["jina"],
        "dense": rerank_scores["dense"],
        "expansion": expansion_scores,
        "e5": e5_scores,
        "corpus": {q: {d: corpus_sparse[q].get(d, -1.0) for d in candidates[q]} for q in ids},
        "vnlegal_lal": vnlegal_scores,
        "crossenc": {q: {d: crossenc_scores.get(q, {}).get(d, floors["crossenc"]) for d in candidates[q]} for q in ids},
        "aiteamvn_ft": {q: {d: aiteam_ft_scores.get(q, {}).get(d, floors["aiteamvn_ft"]) for d in candidates[q]} for q in ids},
        "jina_ft": {q: {d: jina_ft_scores.get(q, {}).get(d, floors["jina_ft"]) for d in candidates[q]} for q in ids},
        "title_embed": {q: {d: title_scores.get(q, {}).get(d, floors["title_embed"]) for d in candidates[q]} for q in ids},
    }

    qmeta = {q: (questions[q], set()) for q in ids}
    types = build_type_table(root, documents, ids, candidates)
    trows = type_features(candidates, types, qmeta, ids)
    own, cited = build_citation_table(documents, ids, candidates)
    crows = citation_features(candidates, own, cited, ids)
    rows0, groups = ltr_features(view_rank, D1_VIEWS, candidates, ids, scores)

    top5, decision = {}, {}
    for q in ids:
        X = np.concatenate([rows0[q], trows[q], crows[q]], axis=1)
        if X.shape[1] != 48:
            raise RuntimeError(f"Target D1 feature dim drift q={q}: {X.shape[1]}")
        s = model.decision_function(scaler.transform(X))
        order = np.argsort(-s)
        ranking = [groups[q][i] for i in order]
        top5[q] = ranking[:5]
        decision[q] = {groups[q][i]: float(s[i]) for i in range(len(groups[q]))}
    return top5, decision


def robust_scale(values):
    x = np.asarray(values, dtype=np.float64)
    med = np.median(x)
    mad = np.median(np.abs(x - med)) * 1.4826
    q75, q25 = np.percentile(x, [75, 25])
    iqr = (q75 - q25) / 1.349
    std = np.std(x)
    return max(float(mad), float(iqr), float(std), 1e-6)


def split_fold0(qids):
    dev, confirm = [], []
    for q in qids:
        h = int(hashlib.sha256(f"{SALT}|{q}".encode()).hexdigest(), 16)
        (dev if h % 2 == 0 else confirm).append(q)
    return dev, confirm


def derive_rel_l0(root: Path, sibling: Path):
    base_script = root.parent / "run_noncal_trainable_ce_boundary_v3_fixed.py"
    if not base_script.is_file():
        raise FileNotFoundError(base_script)
    m = load_module(base_script, "cebase_private_rel_l0")
    cal_ids, _ = m.get_cal_ids_label_free(root)
    world = m.load_noncal_world(root, sibling, set(cal_ids))
    fold0 = [q for q in world["noncal"] if world["folds"][q] == "fold_0"]
    dev, confirm = split_fold0(fold0)
    score_dir = root / "results/manual/huy_noncal_trainable_ce_boundary_v1/oof/fold_0/scores"
    vals = []
    for q in dev:
        top5 = world["base"][q][:5]
        if top5[4] not in world["gold"][q]:
            continue
        p = score_dir / f"{q}.json"
        if not p.is_file():
            raise FileNotFoundError(p)
        obj = json.loads(p.read_text(encoding="utf-8"))
        scores = {str(d): float(s) for d, s in obj["scores"].items()}
        rel = scores[top5[4]] - float(np.median([scores[d] for d in top5[:4]]))
        vals.append(float(rel))
    if len(vals) != EXPECTED_FOLD0_DEV_GOLD_RANK5:
        raise RuntimeError(f"REL_L0 provenance population drift: {len(vals)} != {EXPECTED_FOLD0_DEV_GOLD_RANK5}")
    observed = float(min(vals))
    if abs(observed - REL_L0) > 1e-9:
        raise RuntimeError(f"REL_L0 threshold drift: {observed} != {REL_L0}")
    return {
        "threshold": observed, "fold0_n": len(fold0), "dev_n": len(dev),
        "confirm_n": len(confirm), "gold_rank5_dev_n": len(vals),
        "robust_scale": robust_scale(vals), "base_script": str(base_script),
        "base_script_sha256": sha256(base_script),
    }


def encode_rel_queries(questions, ids, model_dir: Path, output_dir: Path, batch_size: int, device: str):
    from transformers import AutoModel, AutoTokenizer

    output_dir.mkdir(parents=True, exist_ok=True)
    ids_path = output_dir / "private_query_ids.json"
    matrix_path = output_dir / "private_queries.f32.npy"
    manifest_path = output_dir / "PRIVATE_QUERY_VECTOR_MANIFEST.json"
    contract = {
        "model_id": REL_E5_MODEL_ID,
        "model_dir_sha256": tree_sha256(model_dir),
        "query_prefix": "query: ",
        "pooling": "attention_mask_mean",
        "normalize": "l2",
        "dimension": E5_DIM,
        "dtype": "float32",
        "question_fingerprint": digest_json([(q, questions[q]) for q in ids]),
    }
    fp = digest_json(contract)
    if matrix_path.is_file() and ids_path.is_file() and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        cached_ids = [str(x) for x in json.loads(ids_path.read_text(encoding="utf-8"))]
        arr = np.load(matrix_path, mmap_mode="r")
        if (manifest.get("contract_fingerprint") == fp and cached_ids == ids
                and arr.shape == (len(ids), E5_DIM) and arr.dtype == np.float32):
            print("  reusing REL_L0 private query-vector cache", flush=True)
            return matrix_path, ids_path, manifest_path

    tok = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True, use_fast=True)
    model = AutoModel.from_pretrained(str(model_dir), local_files_only=True).to(device).eval()
    matrix = np.lib.format.open_memmap(
        matrix_path, mode="w+", dtype=np.float32, shape=(len(ids), E5_DIM)
    )
    started = time.perf_counter()
    for st in range(0, len(ids), batch_size):
        bids = ids[st:st+batch_size]
        enc = tok(
            ["query: " + questions[q] for q in bids],
            add_special_tokens=True, truncation=False, padding=True, return_tensors="pt",
        )
        lengths = enc["attention_mask"].sum(1)
        if int(lengths.max()) > 512:
            raise RuntimeError(f"Private REL-E5 query exceeds 512 tokens: {bids}")
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.inference_mode():
            hidden = model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1e-9)
            vec = torch.nn.functional.normalize(pooled, p=2, dim=1).float().cpu().numpy().astype(np.float32)
        matrix[st:st+len(bids)] = vec
        done = st + len(bids)
        if done % 160 == 0 or done == len(ids):
            print(f"    REL qvec {done}/{len(ids)} qps={done/max(time.perf_counter()-started,1e-9):.2f}", flush=True)
    matrix.flush()
    del matrix, model, tok
    gc.collect(); torch.cuda.empty_cache()
    ids_path.write_text(json.dumps(ids, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    dump(manifest_path, {
        **contract, "contract_fingerprint": fp,
        "matrix_sha256": sha256(matrix_path), "ids_sha256": sha256(ids_path),
    })
    return matrix_path, ids_path, manifest_path


class PrivateRenderData:
    def __init__(self, bundle: Path, questions, qvec_path: Path, qids_path: Path, evidence_db: Path):
        self.questions = dict(questions)
        con = sqlite3.connect(
            f"file:{evidence_db.resolve().as_posix()}?mode=ro&immutable=1", uri=True
        )
        row = con.execute("SELECT signature FROM complete WHERE kind='inventory'").fetchone()
        con.close()
        if row is None:
            raise RuntimeError("Evidence inventory signature missing")
        self.fingerprint = str(row[0])

        chunk_rows = list(read_jsonl(bundle / "chunk_ids.jsonl"))
        self.chunk_ids = [str(r["chunk_id"]) for r in chunk_rows]
        parents = [str(r["doc_id"]) for r in chunk_rows]
        if len(self.chunk_ids) != EXPECTED_EVIDENCE_CHUNKS:
            raise RuntimeError(f"Evidence chunk drift: {len(self.chunk_ids)}")
        self.doc_ids = sorted(set(parents))
        if len(self.doc_ids) != EXPECTED_EVIDENCE_DOCS:
            raise RuntimeError(f"Evidence parent drift: {len(self.doc_ids)}")
        self.doc_row = {d: i for i, d in enumerate(self.doc_ids)}
        positions = [[] for _ in self.doc_ids]
        for i, d in enumerate(parents):
            positions[self.doc_row[d]].append(i)
        self.positions = [np.asarray(v, dtype=np.int64) for v in positions]
        self._matrix = np.load(bundle / "embeddings.f16.npy", mmap_mode="r")
        if self._matrix.shape != (EXPECTED_EVIDENCE_CHUNKS, E5_DIM):
            raise RuntimeError(f"Evidence matrix drift: {self._matrix.shape}")
        qids = [str(x) for x in json.loads(qids_path.read_text(encoding="utf-8"))]
        self._qvec = np.load(qvec_path, mmap_mode="r")
        self._qrow = {q: i for i, q in enumerate(qids)}
        if set(self.questions) != set(self._qrow):
            raise RuntimeError("Private REL qvec population mismatch")
        if self._qvec.shape != (len(qids), E5_DIM):
            raise RuntimeError(f"Private REL qvec shape drift: {self._qvec.shape}")

    def matrix(self, source: str):
        if source != "e5":
            raise ValueError(source)
        return self._matrix

    def query_vector(self, qid: str, source: str):
        if source != "e5":
            raise ValueError(source)
        v = np.asarray(self._qvec[self._qrow[qid]], dtype=np.float32)
        return v / max(float(np.linalg.norm(v)), 1e-12)


@torch.inference_mode()
def score_rel_top5(sibling: Path, render, ids, d1_top5, checkpoint: Path,
                   output_dir: Path, pair_microbatch: int, qvec_path: Path):
    sys.path[:0] = [str(sibling), str(sibling / "src")]
    from exp_final.cross_encoder import CrossEncoder
    from exp_final.evidence import Evidence

    model = CrossEncoder(checkpoint)
    model.eval()
    evidence = Evidence(render, model.tokenizer)
    score_dir = output_dir / "ce_top5"
    score_dir.mkdir(parents=True, exist_ok=True)
    ckpt_sha, qvec_sha = sha256(checkpoint), sha256(qvec_path)
    rows, no_evidence = {}, {}
    started = time.perf_counter()
    try:
        for i, q in enumerate(ids, 1):
            top5 = d1_top5[q]
            missing = [
                d for d in top5
                if evidence.db.execute("SELECT 1 FROM chunks WHERE doc=? LIMIT 1", (d,)).fetchone() is None
            ]
            if missing:
                no_evidence[q] = missing
                continue
            sig = digest_json(["private-exact-d1-ce-top5-v1", ckpt_sha, qvec_sha, q, top5])
            cp = score_dir / f"{q}.json"
            if cp.is_file():
                obj = json.loads(cp.read_text(encoding="utf-8"))
                if obj.get("signature") != sig:
                    raise RuntimeError(f"Private REL CE cache mismatch q={q}")
                scores = {str(d): float(s) for d, s in obj["scores"].items()}
            else:
                vals = []
                for st in range(0, 5, pair_microbatch):
                    docs = top5[st:st+pair_microbatch]
                    vals.extend(model([evidence.package(q, d) for d in docs]).detach().cpu().tolist())
                scores = {d: float(s) for d, s in zip(top5, vals)}
                dump(cp, {"signature": sig, "top5": top5, "scores": scores})
            rel = scores[top5[4]] - float(np.median([scores[d] for d in top5[:4]]))
            rows[q] = {"top5": top5, "scores": scores, "rel_top4_med": float(rel)}
            if i % 50 == 0 or i == len(ids):
                print(
                    f"    REL CE {i}/{len(ids)} scored={len(rows)} no_evidence={len(no_evidence)} "
                    f"qps={i/max(time.perf_counter()-started,1e-9):.3f}", flush=True
                )
    finally:
        evidence.db.close()
        del evidence, model
        gc.collect(); torch.cuda.empty_cache()
    return rows, no_evidence


def validate_d1(submission, ids, valid_docs):
    if set(submission) != set(ids):
        raise RuntimeError("D1 qid population mismatch")
    for q in ids:
        ans = [str(d) for d in submission[q]["answer"]]
        if len(ans) != 5 or len(set(ans)) != 5 or any(d not in valid_docs for d in ans):
            raise RuntimeError(f"Invalid D1 output q={q}: {ans}")


def validate_rel(submission, ids, d1, actions, valid_docs):
    hist = defaultdict(int)
    if set(submission) != set(ids):
        raise RuntimeError("REL qid population mismatch")
    for q in ids:
        ans = [str(d) for d in submission[q]["answer"]]
        if len(ans) not in (4, 5) or len(set(ans)) != len(ans) or any(d not in valid_docs for d in ans):
            raise RuntimeError(f"Invalid REL output q={q}: {ans}")
        if not set(ans) <= set(d1[q]):
            raise RuntimeError(f"REL introduced non-D1 doc q={q}")
        if q in actions:
            if ans != d1[q][:4]:
                raise RuntimeError(f"REL action not exact rank5 drop q={q}")
        elif ans != d1[q]:
            raise RuntimeError(f"REL abstention unexpectedly changed q={q}")
        hist[len(ans)] += 1
    return {str(k): v for k, v in sorted(hist.items())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=Path, required=True)
    ap.add_argument("--private-file", default="private-official.json")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument(
        "--db",
        type=Path,
        default=None,
        help=(
            "Optional existing legalir_full_fts.sqlite. If omitted, the script "
            "auto-detects known locations and otherwise builds an isolated "
            "private FTS DB with chunk_size=500, overlap=100."
        ),
    )
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--pair-microbatch", type=int, default=4)
    ap.add_argument("--e5-batch-size", type=int, default=16)
    ap.add_argument("--historical-e5-model", default=E5_MODEL_ID)
    ap.add_argument("--skip-cached-public-d1-parity", action="store_true")
    args = ap.parse_args()

    root = args.repo_root.resolve()
    sibling = root.parent / "LegalIR"
    data_dir = root / "DSC2026-LegalIR-main/v4_run/public_test_dataset"
    private_path = data_dir / args.private_file
    if not private_path.is_file():
        raise FileNotFoundError(private_path)
    sys.path.insert(0, str(root))
    from run_burst_expanded_fusion_submission import DocumentStore

    out = root / "results/manual/huy_private_d1_rel_l0_approx_v1"
    cache = out / "cache"
    out.mkdir(parents=True, exist_ok=True); cache.mkdir(parents=True, exist_ok=True)

    corpus_paths = sorted((data_dir / "selected-contexts").glob("context_*.json"))
    doc_ids = [p.stem[len("context_"):] for p in corpus_paths]
    valid_docs = set(doc_ids)
    if not corpus_paths:
        raise RuntimeError("No selected-contexts corpus")
    documents = DocumentStore(corpus_paths)

    raw_train = json.loads((data_dir / "train.json").read_text(encoding="utf-8"))
    train = {
        str(q): (str(v["question"]), {str(d) for d in v["answer"]})
        for q, v in raw_train.items() if v.get("answer")
    }

    print("[1/12] Training exact production D1 on CAL600...", flush=True)
    scaler, d1_model, d1_meta = train_exact_d1(root, documents)

    if args.skip_cached_public_d1_parity:
        print("[2/12] WARNING: cached-public D1 parity SKIPPED", flush=True)
        public_parity = {"status": "SKIPPED_BY_USER"}
    else:
        print("[2/12] Mandatory cached-public D1 parity...", flush=True)
        public_parity = cached_public_d1_parity(root, documents, scaler, d1_model)

    print("[3/12] Loading private-official queries...", flush=True)
    ids, questions = load_questions(private_path)
    print(f"  private queries={len(ids)} sha256={sha256(private_path)}", flush=True)

    print("[4/12] Private sparse retrieval + FINAL approximate CPU base Top20...", flush=True)

    # Reuse the already completed 2,080-query retrieval cache from the earlier
    # exact-generator attempt, but only after strict qid/question fingerprint checks.
    retrieval_cache = cache / "private_retrieval.pkl"
    legacy_root = root / "results/manual/huy_private_d1_rel_l0_v1"
    legacy_retrieval = legacy_root / "cache/private_retrieval.pkl"
    private_fp = digest_json([(q, questions[q]) for q in ids])
    if not retrieval_cache.is_file() and legacy_retrieval.is_file():
        old_obj = load_pickle(legacy_retrieval)
        if (
            old_obj.get("qids") == ids
            and old_obj.get("question_fingerprint") == private_fp
            and len(old_obj.get("cache", {})) == len(ids)
        ):
            save_pickle(retrieval_cache, old_obj)
            print(
                f"  reused completed private retrieval cache -> {retrieval_cache}",
                flush=True,
            )

    # Reuse the already built exact 8532-doc FTS DB if available.
    legacy_db = legacy_root / "cache/legalir_full_fts.sqlite"
    explicit_db = args.db
    if explicit_db is None and legacy_db.is_file():
        explicit_db = legacy_db
        print(f"  reusing prior private FTS DB -> {legacy_db}", flush=True)

    fts_db, fts_meta = resolve_or_build_fts_db(
        root=root,
        data_dir=data_dir,
        cache_dir=cache,
        explicit_db=explicit_db,
    )
    retrieval = target_retrieval(
        root, fts_db, train, doc_ids,
        questions, ids, retrieval_cache, args.workers
    )
    base, cpu_base_artifacts = target_base_top20(
        root, corpus_paths, train, doc_ids, retrieval, questions, ids,
        cache / "private_base_top20_finalapprox.pkl"
    )

    print("[5/12] Building FINAL private expanded/corpus candidates + rerank views...", flush=True)
    cand_cache = cache / "candidate_generation"; cand_cache.mkdir(parents=True, exist_ok=True)
    candidates, expanded, corpus_sparse, expansion_scores, rerank_scores = build_target_candidates(
        root, cand_cache, questions, ids, retrieval, base, documents, args.device
    )
    sizes = [len(candidates[q]) for q in ids]
    print(f"  private candidates min={min(sizes)} mean={np.mean(sizes):.2f} max={max(sizes)}", flush=True)

    print("[6/12] Fresh private D1 score channels...", flush=True)
    print("  [6a] historical E5", flush=True)
    e5_scores = score_e5_channel(
        root, cache / "d1_e5_scores.pkl", questions, ids, candidates,
        documents, args.historical_e5_model
    )
    print("  [6b] vnlegal-lal", flush=True)
    vn_dir = cache / "vnlegal"; vn_dir.mkdir(parents=True, exist_ok=True)
    vnlegal_scores = score_vnlegal_channel(root, vn_dir, questions, ids, candidates, documents, args.device)
    print("  [6c] AITeamVN crossencoder", flush=True)
    crossenc_scores = score_crossenc_channel(
        root, cache / "crossenc_scores.pkl", questions, ids, candidates, documents
    )
    print("  [6d] AITeamVN fine-tuned bi-encoder", flush=True)
    aiteam_ft_scores = score_bi_channel(
        root, cache / "aiteamvn_ft_scores.pkl", questions, ids, candidates,
        documents, root / "models/from_drive/AITeamVN_Vietnamese_Embedding"
    )
    print("  [6e] Jina fine-tuned", flush=True)
    jina_ft_scores = score_jina_ft_channel(
        root, cache / "jina_ft_scores.pkl", questions, ids, candidates, documents
    )
    print("  [6f] title embedding", flush=True)
    title_scores = score_title_channel(
        root, cache / "title_embed_scores.pkl", questions, ids, candidates, documents
    )

    print("[7/12] Inferring private D1 Top5...", flush=True)
    d1_top5, d1_scores = infer_target_d1(
        root, documents, ids, questions, candidates, base, expanded,
        corpus_sparse, expansion_scores, rerank_scores, e5_scores, vnlegal_scores,
        crossenc_scores, aiteam_ft_scores, jina_ft_scores, title_scores,
        scaler, d1_model, d1_meta["floors"]
    )
    d1_submission = {q: {"answer": list(d1_top5[q])} for q in ids}
    validate_d1(d1_submission, ids, valid_docs)
    d1_json, d1_zip = out / "D1_PRIVATE_APPROX.json", out / "D1_PRIVATE_APPROX.zip"
    dump(d1_json, d1_submission); zip_exact(d1_json, d1_zip)
    save_pickle(cache / "d1_private_scores.pkl", {"top5": d1_top5, "decision_scores": d1_scores})
    print(f"  D1 private ZIP={d1_zip}", flush=True)

    print("[8/12] Re-deriving frozen REL_L0 from Fold0 DEV...", flush=True)
    rel_contract = derive_rel_l0(root, sibling)
    print(f"  REL_L0={rel_contract['threshold']:+.16f} gold-r5-dev={rel_contract['gold_rank5_dev_n']}", flush=True)

    print("[9/12] Encoding private REL_L0 VietLegal-E5 query vectors...", flush=True)
    bundle = root / "cache/research_v2_e5_confirmation/bundle-v1"
    rel_model_dir = bundle / "vietlegal-e5"
    checkpoint = root / "results/manual/huy_noncal_trainable_ce_boundary_v1/oof/fold_0/training/model.pt"
    evidence_db = sibling / "cache/exp_final_retrieval/evidence.sqlite"
    for p in (bundle / "embeddings.f16.npy", bundle / "chunk_ids.jsonl", rel_model_dir, checkpoint, evidence_db):
        if not p.exists():
            raise FileNotFoundError(p)
    qvec_path, qids_path, qvec_manifest = encode_rel_queries(
        questions, ids, rel_model_dir, cache / "rel_query_vectors",
        args.e5_batch_size, args.device
    )

    print("[10/12] Scoring private D1 Top5 with frozen Fold0 BGE CE...", flush=True)
    render = PrivateRenderData(bundle, questions, qvec_path, qids_path, evidence_db)
    rel_rows, no_evidence = score_rel_top5(
        sibling, render, ids, d1_top5, checkpoint, cache / "rel_l0",
        args.pair_microbatch, qvec_path
    )
    print(f"  REL scorable={len(rel_rows)} no-evidence abstain={len(no_evidence)}", flush=True)

    print("[11/12] Freezing REL_L0 actions + packaging submission...", flush=True)
    actions, rel_submission = {}, {}
    for q in ids:
        if q in rel_rows and rel_rows[q]["rel_top4_med"] < REL_L0:
            rel_submission[q] = {"answer": list(d1_top5[q][:4])}
            actions[q] = {
                "qid": q, "removed_rank5": d1_top5[q][4],
                "rel_top4_med": rel_rows[q]["rel_top4_med"], "threshold": REL_L0,
                "original_top5": d1_top5[q], "new_answer": d1_top5[q][:4],
            }
        else:
            rel_submission[q] = {"answer": list(d1_top5[q])}
    k_hist = validate_rel(rel_submission, ids, d1_top5, actions, valid_docs)
    rel_dir = out / "REL_L0"; rel_dir.mkdir(parents=True, exist_ok=True)
    action_path = rel_dir / "PRIVATE_ACTIONS_LABEL_FREE.json"
    sub_path = rel_dir / "submission.json"
    zip_path = rel_dir / "submission_REL_L0_PRIVATE_APPROX.zip"
    dump(action_path, {
        "schema": "manual.private_d1_rel_l0_approx_v1.actions",
        "status": "SEALED_LABEL_FREE_PRIVATE_APPROX_D1", "threshold": REL_L0,
        "threshold_contract": rel_contract, "private_questions_sha256": sha256(private_path),
        "private_qvec_sha256": sha256(qvec_path), "d1_private_sha256": sha256(d1_json),
        "checkpoint_sha256": sha256(checkpoint), "actions_count": len(actions),
        "no_evidence_abstentions": no_evidence, "actions": actions,
    })
    dump(sub_path, rel_submission); zip_exact(sub_path, zip_path)

    print("[12/12] Writing private report...", flush=True)
    mean_k = float(np.mean([len(rel_submission[q]["answer"]) for q in ids]))
    report = {
        "schema": "manual.private_d1_rel_l0_approx_v1.report",
        "status": "READY_FOR_PRIVATE_SUBMISSION_APPROX_D1",
        "target": {"file": str(private_path), "sha256": sha256(private_path), "queries": len(ids)},
        "public_downstream_parity": public_parity,
        "fts_database": fts_meta,
        "cpu_base_reconstruction": cpu_base_artifacts,
        "historical_generator_caveat": {
            "cpu_top20_original_generator_fully_committed": False,
            "cpu_top20_exact_recovery_attempt_abandoned": True,
            "cpu_top20_private_mode": "BEST_UNLABELED_OUTPUT_MATCHED_APPROXIMATION",
            "cpu_top20_public_recovery_parity": FINAL_CPU_PUBLIC_PARITY,
            "cpu_top20_recovered_config": cpu_base_artifacts,
            "e5_public_original_generator_fully_committed": False,
            "crossenc_public_original_generator_fully_committed": False,
            "private_reconstruction": (
                "fresh inference from documented model/procedure provenance; "
                "all final-run candidate/neural caches isolated from earlier attempts"
            ),
        },
        "d1": {
            "training": d1_meta, "queries": len(ids),
            "candidate_pool_min": int(min(sizes)), "candidate_pool_mean": float(np.mean(sizes)),
            "candidate_pool_max": int(max(sizes)), "submission_json": str(d1_json),
            "submission_json_sha256": sha256(d1_json), "submission_zip": str(d1_zip),
            "submission_zip_sha256": sha256(d1_zip),
        },
        "rel_l0": {
            "threshold": REL_L0, "threshold_contract": rel_contract,
            "ce_scorable": len(rel_rows), "no_evidence_abstentions": len(no_evidence),
            "actions": len(actions), "action_rate": len(actions)/len(ids),
            "k_histogram": k_hist, "mean_k": mean_k,
            "submission_json": str(sub_path), "submission_json_sha256": sha256(sub_path),
            "submission_zip": str(zip_path), "submission_zip_sha256": sha256(zip_path),
            "actions_json": str(action_path), "actions_json_sha256": sha256(action_path),
        },
        "scientific_contract": {
            "private_labels_used": False,
            "d1_trained_on_cal600_labels": True,
            "rel_threshold_uses_cal_labels": False,
            "rel_policy": "drop D1 rank5 iff REL_TOP4_MED < frozen REL_L0",
            "ranks_1_to_4_immutable": True, "introduced_docs": False,
            "rel_query_encoder": REL_E5_MODEL_ID, "rel_query_prefix": "query: ",
            "rel_query_pooling": "attention-mask mean + L2",
            "rel_evidence_renderer": "exp_final.evidence.Evidence.package",
            "rel_chunk_bank": "frozen EXP-021 343347x1024",
        },
        "provenance": {
            "private_file": str(private_path),
            "fold0_checkpoint": str(checkpoint), "fold0_checkpoint_sha256": sha256(checkpoint),
            "rel_query_vector_manifest": str(qvec_manifest),
            "rel_query_vector_sha256": sha256(qvec_path),
            "evidence_chunk_bank_sha256": sha256(bundle / "embeddings.f16.npy"),
            "evidence_chunk_ids_sha256": sha256(bundle / "chunk_ids.jsonl"),
        },
    }
    report_path = out / "PRIVATE_REL_L0_REPORT.json"
    dump(report_path, report)

    print("=" * 112)
    print("PRIVATE APPROX-D1 + REL_L0 SUBMISSION READY")
    print(
        f"queries={len(ids)} | D1 K=5 | REL scorable={len(rel_rows)} | "
        f"REL actions={len(actions)} ({100*len(actions)/len(ids):.2f}%) | meanK={mean_k:.4f}"
    )
    print(f"D1 APPROX ZIP: {d1_zip}")
    print(f"REL_L0 ZIP:    {zip_path}")
    print(f"Report:        {report_path}")
    print(
        "NOTE: CPU base is the final approximate recovery, NOT the exact "
        "historical cpu_top20 generator."
    )
    print("=" * 112)


if __name__ == "__main__":
    main()
