"""FAST private D1-only materializer (v14).

Derived from v13, but intentionally prioritizes throughput over tiny batch-dependent numerical drift.
REL_L0 / Fold-0 BGE CE / all post-D1 stages are intentionally removed.

Original v13 notes:

This runner is used only after restoring the original historical artifacts:
  results/burst_large_ltr/best_model.pkl
  results/burst_legal_features/validation_model.pkl
  results/burst_empirical_pairwise/model.pkl
  results/jina_reranker/burst_pairwise_state.pt

CPU base contract is taken verbatim from run_burst_gpu_submission.py:
  robust = RRF(Large-LTR, Legal-LTR[:10], alpha=.275, k=40)
  profile = profile_rank(..., 2, 1.2, .75, .3)
  graph = graph_rerank(robust, 3, .5, "conditional", 3, .4, 0)
  final = weighted_rrf([robust,pair,profile,graph], [.40,.30,.15,.15], k=0)
  top20 = final[:20]

The strict v13 workflow already verified exact PUBLIC cpu_top20 parity.
v14-fast deliberately does NOT rerun that expensive 1,000-query/8,532-document
public reconstruction on every launch. Private retrieval/base caches remain guarded
by qid/question/artifact fingerprints and are resumed in place.

D1 historical E5 score-channel contract is also taken from
run_burst_gpu_submission.py:
  model = models/multilingual-e5-small
  max_length = 384
  query prefix = "query: "
  passage prefix = "passage: "
  2 top passages / base document
  max-pool passage score per document
  IMPORTANT: only base CPU-top20 docs are scored; other D1 candidates are
  intentionally missing and ltr_features fills them at mean-2*std, matching
  historical public behavior.

Stage-5 neural rerank scores from the previous approximate run are reusable:
rerank() explicitly scores only candidate documents that are not already cached.
"""
from __future__ import annotations

import argparse, gc, hashlib, importlib.util, json, math, os, pickle, sqlite3, subprocess, sys, threading, time, zipfile, warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

D1_VIEWS = ["base", "expanded", "jina", "dense", "corpus"]
E5_MODEL_ID = "models/multilingual-e5-small"



# Historical artifacts were restored from the original public run and were
# already parity-validated in the strict v13 workflow. Suppress only the
# corresponding persistence-version warnings to keep the terminal clean.
try:
    from sklearn.exceptions import InconsistentVersionWarning
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
except Exception:
    pass

warnings.filterwarnings(
    "ignore",
    message=r".*If you are loading a serialized model .*older version of XGBoost.*",
    category=UserWarning,
)

HIST_CPU_CONFIG = {
    "version": "BURST-MultistagePosterior-v1",
    "robust_alpha": 0.275,
    "robust_k": 40,
    "stack_weights": (0.40, 0.30, 0.15, 0.15),
    "stack_k": 0,
}
HIST_GPU_E5 = {
    "model": "models/multilingual-e5-small",
    "candidate_depth": 20,
    "passages_per_doc": 2,
    "max_length": 384,
    "query_prefix": "query: ",
    "passage_prefix": "passage: ",
}

HIST_E5_HF_ID = "intfloat/multilingual-e5-small"
# Pin a pre-existing model revision; later repository commits only add
# evaluation/export artifacts and do not alter the base weights/config.
HIST_E5_HF_REVISION = "fd1525a9fd15316a2d503bf26ab031a61d056e98"
HIST_E5_SAFETENSORS_SHA256 = (
    "1a55775f53449dac10a2bcbc312469fac40b96d53198c407081a831f81c98477"
)

HIST_GPU_SCORES_SHA256 = (
    "9dfb7df958f0ad154a0ad0aa17ed178549e415b14a359c6f7de7062dbfce5c4d"
)

HIST_BENCHMARK_E5_GIT_BLOB_SHA1 = (
    "1f38c8426b0abbcf62e64a8470c6610fcac0e08a"
)


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
    # Historical artifacts were serialized under older sklearn/XGBoost runtimes.
    # We suppress warnings only during unpickle. Strict v13 already established
    # public parity; v14-fast avoids rerunning that expensive audit.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
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


def load_historical_cpu_artifacts(root: Path):
    """Load the restored historical learned artifacts used by cpu_top20."""
    paths = {
        "large": root / "results/burst_large_ltr/best_model.pkl",
        "legal": root / "results/burst_legal_features/validation_model.pkl",
        "pair": root / "results/burst_empirical_pairwise/model.pkl",
        "jina_state": root / "results/jina_reranker/burst_pairwise_state.pt",
    }
    for name, p in paths.items():
        if not p.is_file():
            raise FileNotFoundError(f"Missing restored historical artifact {name}: {p}")

    large_saved = load_pickle(paths["large"])
    legal_saved = load_pickle(paths["legal"])
    pair_saved = load_pickle(paths["pair"])
    if "model" not in large_saved:
        raise RuntimeError("Historical Large-LTR artifact has no model")
    if "model" not in legal_saved:
        raise RuntimeError("Historical Legal-LTR artifact has no model")
    _validate_pair_model(pair_saved)

    provenance = {
        "mode": "RESTORED_HISTORICAL_ARTIFACTS",
        "cpu_config": {
            "version": HIST_CPU_CONFIG["version"],
            "robust_alpha": HIST_CPU_CONFIG["robust_alpha"],
            "robust_k": HIST_CPU_CONFIG["robust_k"],
            "stack_weights": list(HIST_CPU_CONFIG["stack_weights"]),
            "stack_k": HIST_CPU_CONFIG["stack_k"],
        },
        "large_path": str(paths["large"]),
        "large_sha256": sha256(paths["large"]),
        "large_metadata": {
            k: v for k, v in large_saved.items()
            if k in ("kind", "params", "blend_alpha", "blend_k", "version")
        },
        "legal_path": str(paths["legal"]),
        "legal_sha256": sha256(paths["legal"]),
        "legal_metadata": {
            k: v for k, v in legal_saved.items()
            if k in ("C", "train", "feature_version")
        },
        "pair_path": str(paths["pair"]),
        "pair_sha256": sha256(paths["pair"]),
        "pair_feature_version": pair_saved.get("feature_version"),
        "jina_state_path": str(paths["jina_state"]),
        "jina_state_sha256": sha256(paths["jina_state"]),
    }
    return (
        large_saved["model"],
        legal_saved["model"],
        pair_saved,
        provenance,
    )


def _cpu_multistage_rankings(
    root: Path,
    corpus_paths,
    train,
    doc_ids,
    retrieval,
    questions,
    ids,
    cache_path: Path | None = None,
):
    """Exact CPU block from historical run_burst_gpu_submission.py."""
    from benchmark_burst_v4_full_sqlite import tokens
    from tune_burst_empirical_bayes_ltr import label_frequency
    from tune_burst_empirical_pairwise import features_for, rank as pairwise_rank
    from tune_burst_graph_posterior import build_graph, graph_rerank
    from tune_burst_legal_features import enhanced_features
    from tune_burst_multistage_posterior import weighted_rrf
    from tune_burst_pairwise import blend_rankings
    from tune_burst_score_ltr import score_features
    from tune_burst_supervised_profile_bm25 import build_profiles, profile_rank

    large_model, legal_model, pair_saved, provenance = (
        load_historical_cpu_artifacts(root)
    )

    fp = digest_json([(q, questions[q]) for q in ids])
    artifact_fp = digest_json({
        "large": provenance["large_sha256"],
        "legal": provenance["legal_sha256"],
        "pair": provenance["pair_sha256"],
        "config": provenance["cpu_config"],
    })

    rankings = {}
    if cache_path is not None and cache_path.is_file():
        obj = load_pickle(cache_path)
        if (
            obj.get("qids") == ids
            and obj.get("question_fingerprint") == fp
            and obj.get("artifact_fingerprint") == artifact_fp
        ):
            rankings = obj.get("rankings", {})

    missing = [q for q in ids if q not in rankings]
    if cache_path is not None:
        print(
            f"  historical CPU cache {len(rankings)}/{len(ids)} "
            f"missing={len(missing)}",
            flush=True,
        )
    if not missing:
        return rankings, provenance

    train_ids = list(train)
    profile_model = build_profiles(train, train_ids)
    graph_frequency, graph_adjacency, _ = build_graph(train, train_ids)
    frequency = label_frequency(train, set())

    normalized = {}
    print("  normalizing corpus for historical Legal-LTR...", flush=True)
    for i, path in enumerate(corpus_paths, 1):
        row = json.loads(path.read_text(encoding="utf-8"))
        did = str(row.get("id", path.stem[len("context_"):]))
        normalized[did] = " " + " ".join(tokens(row.get("passage") or "")) + " "
        if i % 1500 == 0 or i == len(corpus_paths):
            print(f"    normalized {i}/{len(corpus_paths)}", flush=True)

    qmeta = {q: (questions[q], set()) for q in ids}
    pair_features = features_for(retrieval, qmeta, ids, frequency)

    started = time.perf_counter()
    for i, q in enumerate(missing, 1):
        lists = retrieval[q]

        candidates, x = score_features(lists)
        large = [
            candidates[j]
            for j in np.argsort(-large_model.predict(x))[:100]
        ]

        candidates, x = enhanced_features(
            lists, questions[q], normalized, lexical_depth=20
        )
        legal = [
            candidates[j]
            for j in np.argsort(-legal_model.decision_function(x))[:100]
        ]

        robust = blend_rankings(
            {q: large}, {q: legal[:10]},
            HIST_CPU_CONFIG["robust_alpha"],
            HIST_CPU_CONFIG["robust_k"],
        )[q]

        pair = pairwise_rank(
            pair_saved["model"], pair_saved["scaler"],
            {q: pair_features[q]}, [q]
        )[q]

        profile = profile_rank(
            questions[q], profile_model, 2, 1.2, .75, .3
        )

        graph = graph_rerank(
            robust, graph_frequency, graph_adjacency,
            3, .5, "conditional", 3, .4, 0
        )

        final = weighted_rrf(
            [{q: robust}, {q: pair}, {q: profile}, {q: graph}],
            HIST_CPU_CONFIG["stack_weights"],
            HIST_CPU_CONFIG["stack_k"],
        )[q]
        rankings[q] = final[:20]

        if cache_path is not None and (i % 25 == 0 or i == len(missing)):
            save_pickle(
                cache_path,
                {
                    "qids": ids,
                    "question_fingerprint": fp,
                    "artifact_fingerprint": artifact_fp,
                    "artifact_provenance": provenance,
                    "rankings": rankings,
                },
            )
            print(
                f"    historical CPU ranked {i}/{len(missing)} "
                f"({time.perf_counter()-started:.1f}s)",
                flush=True,
            )

    return rankings, provenance


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
    return _cpu_multistage_rankings(
        root, corpus_paths, train, doc_ids,
        retrieval, questions, ids, cache_path
    )


def _validate_corpus_dense_index(directory: Path, cap: int = 32) -> dict:
    """Validate the exact base-AITeamVN corpus dense index contract."""
    meta_path = directory / f"chunks_cap{cap}.json"
    vec_path = directory / f"chunks_cap{cap}.f16"
    if not meta_path.is_file() or not vec_path.is_file():
        raise FileNotFoundError(
            f"Missing corpus index pair under {directory}: "
            f"{meta_path.name}, {vec_path.name}"
        )

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if int(meta.get("cap", -1)) != cap:
        raise RuntimeError(f"Corpus index cap mismatch: {meta.get('cap')} != {cap}")
    if int(meta.get("window", -1)) != 220:
        raise RuntimeError(
            f"Corpus index window mismatch: {meta.get('window')} != 220"
        )
    if int(meta.get("step", -1)) != 150:
        raise RuntimeError(
            f"Corpus index step mismatch: {meta.get('step')} != 150"
        )

    documents = meta.get("documents") or []
    counts = meta.get("counts") or []
    if len(documents) != 8532 or len(counts) != 8532:
        raise RuntimeError(
            f"Corpus index document-count mismatch: "
            f"documents={len(documents)} counts={len(counts)} expected=8532"
        )
    chunks = int(sum(int(x) for x in counts))
    expected_bytes = chunks * 1024 * 2
    actual_bytes = vec_path.stat().st_size
    if actual_bytes != expected_bytes:
        raise RuntimeError(
            f"Corpus index vector-size mismatch: "
            f"{actual_bytes} bytes != expected {expected_bytes} "
            f"({chunks} x 1024 x fp16)"
        )

    return {
        "directory": str(directory.resolve()),
        "meta_path": str(meta_path.resolve()),
        "vector_path": str(vec_path.resolve()),
        "documents": len(documents),
        "chunks": chunks,
        "cap": cap,
        "window": 220,
        "step": 150,
        "vector_bytes": actual_bytes,
    }


def ensure_corpus_dense_index(root: Path, cap: int = 32) -> dict:
    """
    Resolve exact historical base-encoder corpus index.

    Search:
      1) current repo
      2) sibling ../LegalIR
      3) sibling ../DSC2026-LegalIR-Huy

    If absent, invoke original build_corpus_dense_index.py.
    """
    candidates = [
        (root / "results/corpus_index").resolve(),
        (root.parent / "LegalIR/results/corpus_index").resolve(),
        (root.parent / "DSC2026-LegalIR-Huy/results/corpus_index").resolve(),
    ]

    for directory in candidates:
        try:
            info = _validate_corpus_dense_index(directory, cap)
            print(
                f"  corpus dense index resolved -> {directory} "
                f"(docs={info['documents']}, chunks={info['chunks']})",
                flush=True,
            )
            return info
        except (FileNotFoundError, RuntimeError) as exc:
            print(
                f"  corpus index candidate unavailable/incompatible: "
                f"{directory} ({exc})",
                flush=True,
            )

    builder = root / "build_corpus_dense_index.py"
    if not builder.is_file():
        raise FileNotFoundError(
            f"No reusable cap{cap} corpus index found and builder is missing: "
            f"{builder}"
        )

    model_dir = root / "models/AITeamVN_Vietnamese_Embedding"
    if not model_dir.is_dir():
        raise FileNotFoundError(
            f"Cannot build corpus index: base dense model missing: {model_dir}"
        )

    output = root / "results/corpus_index"
    output.mkdir(parents=True, exist_ok=True)

    print(
        f"  No reusable cap{cap} corpus dense index found.\n"
        f"  Building/resuming exact historical index with "
        f"{builder.name} -> {output}",
        flush=True,
    )
    cmd = [
        sys.executable,
        str(builder),
        "--cap", str(cap),
        "--batch-size", "32",
        "--max-length", "512",
        "--output", str(output),
        "--model-path", "models/AITeamVN_Vietnamese_Embedding",
        "--name", "chunks",
    ]
    subprocess.run(cmd, cwd=str(root), check=True)

    info = _validate_corpus_dense_index(output, cap)
    print(
        f"  corpus dense index build validated: "
        f"docs={info['documents']} chunks={info['chunks']}",
        flush=True,
    )
    return info


def install_corpus_index_override(root: Path, index_info: dict):
    """
    Let corpus_dense() use a sibling index without copying the large fp16 bank.
    """
    directory = Path(index_info["directory"]).resolve()
    canonical = (root / "results/corpus_index").resolve()
    if directory == canonical:
        return

    import benchmark_corpus_dense_recall as corpus_mod

    def load_index_override(_root, cap):
        meta_path = directory / f"chunks_cap{cap}.json"
        vec_path = directory / f"chunks_cap{cap}.f16"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        vectors = np.fromfile(
            vec_path, dtype=np.float16
        ).reshape(-1, 1024)
        counts = np.asarray(meta["counts"])
        if vectors.shape[0] != counts.sum():
            raise RuntimeError(
                f"Index mismatch in override: "
                f"{vectors.shape[0]} vs {counts.sum()}"
            )
        owners = np.repeat(np.arange(len(counts)), counts)
        return meta["documents"], vectors, owners

    corpus_mod.load_index = load_index_override
    print(
        f"  installed zero-copy corpus-index override -> {directory}",
        flush=True,
    )



def install_jina_xlm_roberta_compat_shim():
    """
    Compatibility shim for older Jina reranker remote code.

    jina-reranker-v2-base-multilingual's cached custom embedding.py imports:
        transformers.models.xlm_roberta.modeling_xlm_roberta
            .create_position_ids_from_input_ids

    Newer Transformers releases no longer expose that helper there.
    Restore the historical helper before AutoModelForSequenceClassification
    imports Jina's dynamic module.
    """
    import torch
    import transformers.models.xlm_roberta.modeling_xlm_roberta as xlm_mod

    if hasattr(xlm_mod, "create_position_ids_from_input_ids"):
        print(
            "  Jina/XLM-R compatibility helper already present.",
            flush=True,
        )
        return

    def create_position_ids_from_input_ids(
        input_ids,
        padding_idx,
        past_key_values_length=0,
    ):
        mask = input_ids.ne(padding_idx).int()
        incremental_indices = (
            torch.cumsum(mask, dim=1).type_as(mask)
            + past_key_values_length
        ) * mask
        return incremental_indices.long() + padding_idx

    xlm_mod.create_position_ids_from_input_ids = (
        create_position_ids_from_input_ids
    )
    print(
        "  Installed Jina/XLM-R Transformers compatibility shim "
        "(create_position_ids_from_input_ids).",
        flush=True,
    )


def rerank_incremental_scalar_safe(
    root: Path,
    output: Path,
    public,
    public_ids,
    candidates,
    documents,
    device,
):
    """
    Exact run_burst_expanded_fusion_submission.rerank contract with one
    shape-only compatibility fix for incremental cache reuse.

    Jina's compute_score returns a scalar float when exactly one pair is
    provided. The historical function assumes an iterable because fresh runs
    score many passages/query. Reusing a nearly-complete cache can leave one
    unseen passage, so normalize scalar -> length-1 list before zip().

    Scores themselves are unchanged.
    """
    from concurrent.futures import ThreadPoolExecutor
    from transformers import (
        AutoModel,
        AutoModelForSequenceClassification,
        AutoTokenizer,
    )

    from benchmark_aiteamvn_holdouts import encode_cls
    from benchmark_jina_reranker_holdouts import top_passages
    from run_burst_expanded_fusion_submission import (
        RERANK_CONFIG,
        prefetch,
    )

    path = output / "rerank_scores.pkl"
    saved = {"config": RERANK_CONFIG, "jina": {}, "dense": {}}
    if path.exists():
        loaded = pickle.loads(path.read_bytes())
        if loaded.get("config") == RERANK_CONFIG:
            saved = loaded

    remaining = [
        q for q in public_ids
        if any(
            d not in saved["jina"].get(q, {})
            or d not in saved["dense"].get(q, {})
            for d in candidates[q]
        )
    ]
    print(
        f"Rerank cache {len(public_ids)-len(remaining)}/{len(public_ids)}",
        flush=True,
    )
    if not remaining:
        return saved

    jina_path = root / "models/jina-reranker-v2-base-multilingual"
    jina_tok = AutoTokenizer.from_pretrained(
        jina_path,
        trust_remote_code=True,
        fix_mistral_regex=True,
    )
    jina = AutoModelForSequenceClassification.from_pretrained(
        jina_path,
        trust_remote_code=True,
        dtype=torch.bfloat16,
    )
    jina.load_state_dict(
        torch.load(
            root / "results/jina_reranker/burst_pairwise_state.pt",
            map_location="cpu",
            weights_only=True,
        )["state_dict"],
        strict=False,
    )
    jina._tokenizer = jina_tok
    jina.eval().to(device)

    dense_path = root / "models/AITeamVN_Vietnamese_Embedding"
    dense_tok = AutoTokenizer.from_pretrained(dense_path)
    dense = AutoModel.from_pretrained(
        dense_path,
        dtype=torch.float16,
    ).eval().to(device)

    print(
        f"Rerankers ready on {torch.cuda.get_device_name(0)}",
        flush=True,
    )

    def prepare(q):
        question = public[q]
        known = saved["jina"].get(q, {})
        owners, passages = [], []
        for doc in candidates[q]:
            if (
                doc in known
                and doc in saved["dense"].get(q, {})
            ):
                continue
            for passage in top_passages(
                question,
                documents[doc],
                count=RERANK_CONFIG["passages_per_doc"],
            ):
                owners.append(doc)
                passages.append(passage)
        return q, owners, passages

    started = time.perf_counter()
    done = 0
    scalar_fixes = 0

    with ThreadPoolExecutor(max_workers=2) as pool:
        for q, owners, passages in prefetch(
            pool, remaining, prepare
        ):
            question = public[q]
            js = dict(saved["jina"].get(q, {}))
            ds = dict(saved["dense"].get(q, {}))

            if passages:
                jraw = jina.compute_score(
                    [(question, p) for p in passages],
                    batch_size=16,
                    max_length=RERANK_CONFIG["max_length"],
                )

                # The ONLY behavioral patch relative to historical rerank().
                if torch.is_tensor(jraw):
                    jraw = (
                        jraw.detach()
                        .float()
                        .cpu()
                        .reshape(-1)
                        .tolist()
                    )
                elif isinstance(
                    jraw,
                    (float, int, np.floating, np.integer),
                ):
                    jraw = [float(jraw)]
                    scalar_fixes += 1
                else:
                    jraw = np.asarray(jraw).reshape(-1).tolist()

                if len(jraw) != len(owners):
                    raise RuntimeError(
                        f"Jina output-length mismatch qid={q}: "
                        f"scores={len(jraw)} owners={len(owners)} "
                        f"passages={len(passages)}"
                    )

                qvec = encode_cls(
                    dense,
                    dense_tok,
                    [question],
                    1,
                    RERANK_CONFIG["max_length"],
                )[0]
                pvec = encode_cls(
                    dense,
                    dense_tok,
                    passages,
                    32,
                    RERANK_CONFIG["max_length"],
                )
                dense_raw = pvec @ qvec

                if len(dense_raw) != len(owners):
                    raise RuntimeError(
                        f"Dense output-length mismatch qid={q}: "
                        f"scores={len(dense_raw)} owners={len(owners)}"
                    )

                for doc, j, d in zip(
                    owners, jraw, dense_raw
                ):
                    js[doc] = max(
                        js.get(doc, -1e9),
                        float(j),
                    )
                    ds[doc] = max(
                        ds.get(doc, -1e9),
                        float(d),
                    )

            # Hard-check that every exact candidate now has both scores.
            missing_j = [
                d for d in candidates[q] if d not in js
            ]
            missing_d = [
                d for d in candidates[q] if d not in ds
            ]
            if missing_j or missing_d:
                raise RuntimeError(
                    f"Incremental rerank incomplete qid={q}: "
                    f"missing_jina={len(missing_j)} "
                    f"missing_dense={len(missing_d)}"
                )

            saved["jina"][q] = {
                d: js[d] for d in candidates[q]
            }
            saved["dense"][q] = {
                d: ds[d] for d in candidates[q]
            }
            done += 1

            if done % 20 == 0:
                path.write_bytes(
                    pickle.dumps(saved, protocol=5)
                )
                rate = (
                    time.perf_counter() - started
                ) / done
                left = len(remaining) - done
                print(
                    f"Rerank exact-delta {done}/{len(remaining)} "
                    f"({rate:.2f}s/query, "
                    f"eta {rate*left/60:.1f}m, "
                    f"scalar_fixes={scalar_fixes})",
                    flush=True,
                )

    path.write_bytes(pickle.dumps(saved, protocol=5))
    print(
        f"Rerank exact-delta complete: "
        f"{done} queries, scalar_fixes={scalar_fixes}",
        flush=True,
    )

    del jina, dense, jina_tok, dense_tok
    gc.collect()
    torch.cuda.empty_cache()
    return saved


def build_target_candidates(root: Path, cache_dir: Path, questions, ids, retrieval, base, documents, device: str):
    from benchmark_dense_expansion_holdouts import raw_union
    from tune_burst_multistage_posterior import weighted_rrf
    from run_burst_expanded_fusion_submission import CORPUS_CAP, CORPUS_DEPTH, EXPANSION_CONFIG, RERANK_CONFIG, dense_expansion, corpus_dense
    raw = {q: raw_union(retrieval[q], EXPANSION_CONFIG["depth"]) for q in ids}
    expansion_scores = dense_expansion(root, cache_dir, questions, ids, raw, documents, device)
    dense_rank = {q: sorted(raw[q], key=lambda d: (-expansion_scores[q][d], d)) for q in ids}
    expanded_all = weighted_rrf([raw, dense_rank], RERANK_CONFIG["expansion_weights"], RERANK_CONFIG["expansion_rrf_k"])
    corpus_index_info = ensure_corpus_dense_index(
        root, cap=CORPUS_CAP
    )
    install_corpus_index_override(root, corpus_index_info)
    corpus_rank, corpus_score = corpus_dense(
        root, cache_dir, questions, ids, device,
        cap=CORPUS_CAP, depth=CORPUS_DEPTH
    )
    expanded = {q: expanded_all[q][:RERANK_CONFIG["expanded_depth"]] for q in ids}
    candidates = {q: list(dict.fromkeys(list(base[q]) + expanded[q] + corpus_rank[q][:CORPUS_DEPTH])) for q in ids}
    corpus_sparse = {q: {d: corpus_score[q][d] for d in candidates[q] if d in corpus_score[q]} for q in ids}
    expansion_scores = {q: {d: expansion_scores[q][d] for d in candidates[q] if d in expansion_scores[q]} for q in ids}
    install_jina_xlm_roberta_compat_shim()
    rr = rerank_incremental_scalar_safe(
        root, cache_dir, questions, ids, candidates, documents, device
    )
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


class HistoricalE5RawPassageStore:
    """
    Exact document-text contract used by run_burst_gpu_submission.py.

    Historical code used:
        text = row.get("passage") or ""
        documents[doc] = text

    IMPORTANT: unlike the newer DocumentStore, this MUST NOT replace empty
    passages with a title reconstructed from the URL slug.
    """

    def __init__(self, modern_store):
        paths = getattr(modern_store, "paths", None)
        if not isinstance(paths, dict) or not paths:
            raise RuntimeError(
                "Historical E5 raw-passage store requires DocumentStore.paths"
            )
        self.paths = paths
        self.cache = {}
        self._empty_ids = None

    def __getitem__(self, doc):
        doc = str(doc)
        if doc in self.cache:
            return self.cache[doc]
        path = self.paths.get(doc)
        if path is None:
            return ""
        row = json.loads(Path(path).read_text(encoding="utf-8"))
        text = row.get("passage") or ""
        self.cache[doc] = text
        return text

    def empty_doc_ids(self):
        if self._empty_ids is None:
            empty = set()
            for doc, path in self.paths.items():
                row = json.loads(Path(path).read_text(encoding="utf-8"))
                if not (row.get("passage") or ""):
                    empty.add(str(doc))
            self._empty_ids = empty
        return self._empty_ids


def audit_historical_gpu_score_artifact(root: Path):
    """
    Hard-audit the exact uploaded historical gpu_scores.checkpoint.pkl.

    Facts established from the user's uploaded artifact:
      SHA256 = 9dfb7d...
      1,000 queries
      exactly 20 Jina + 20 E5 doc scores per query
      every historical E5 scalar is exactly representable as float32
      only 6/20,000 are exactly representable as float16

    Therefore the historical `pvectors @ qvector` score output was float32,
    and FP16-output probe variants are not valid reconstructions.
    """
    path = root / "results/burst_gpu_threeview/gpu_scores.checkpoint.pkl"
    if not path.is_file():
        raise FileNotFoundError(f"Historical GPU score artifact missing: {path}")

    got_sha = sha256(path)
    if got_sha != HIST_GPU_SCORES_SHA256:
        raise RuntimeError(
            "Historical gpu_scores.checkpoint.pkl SHA256 mismatch: "
            f"{got_sha} != {HIST_GPU_SCORES_SHA256}"
        )

    obj = load_pickle(path)
    expected_config = {
        "version": "BURST-ThreeViewGPU-v1",
        "candidate_depth": 20,
        "passages_per_doc": 2,
        "max_length": 384,
        "weights_base_jina_e5": (0.50, 0.20, 0.30),
        "rrf_k": 20,
        "top_k": 5,
    }
    if obj.get("config") != expected_config:
        raise RuntimeError(
            f"Historical GPU config mismatch: {obj.get('config')}"
        )

    scores = obj.get("scores")
    if not isinstance(scores, dict) or len(scores) != 1000:
        raise RuntimeError(
            f"Historical GPU score query count != 1000: "
            f"{0 if not isinstance(scores, dict) else len(scores)}"
        )

    e5_values = []
    for q, row in scores.items():
        if set(row) != {"jina", "e5"}:
            raise RuntimeError(f"Historical GPU row schema mismatch qid={q}")
        if len(row["jina"]) != 20 or len(row["e5"]) != 20:
            raise RuntimeError(
                f"Historical GPU depth mismatch qid={q}: "
                f"jina={len(row['jina'])} e5={len(row['e5'])}"
            )
        e5_values.extend(float(v) for v in row["e5"].values())

    arr = np.asarray(e5_values, dtype=np.float64)
    f32_roundtrip = arr.astype(np.float32).astype(np.float64)
    f16_roundtrip = arr.astype(np.float16).astype(np.float64)
    exact_f32 = int(np.count_nonzero(f32_roundtrip == arr))
    exact_f16 = int(np.count_nonzero(f16_roundtrip == arr))

    if exact_f32 != len(arr):
        raise RuntimeError(
            f"Historical E5 scores are not pure float32 scalars: "
            f"{exact_f32}/{len(arr)} exact"
        )

    print(
        "  HISTORICAL GPU SCORE ARTIFACT: "
        f"SHA256={got_sha[:16]}... | queries=1000 | "
        f"E5 scores={len(arr)} | exact-f32={exact_f32}/{len(arr)} | "
        f"exact-f16={exact_f16}/{len(arr)}",
        flush=True,
    )
    print(
        "  E5 numeric contract: FLOAT32 score output confirmed; "
        "discarding FP16-output hypotheses.",
        flush=True,
    )

    return {
        "sha256": got_sha,
        "queries": 1000,
        "e5_scores": len(arr),
        "exact_float32": exact_f32,
        "exact_float16": exact_f16,
    }



def ensure_historical_e5_small(root: Path) -> Path:
    """
    Resolve the historical multilingual-e5-small model.

    Search local repo + common sibling repos first. If unavailable, download a
    pinned Hugging Face revision to the historical local path, then verify the
    exact model weight SHA256.
    """
    local = root / "models/multilingual-e5-small"
    sibling_candidates = [
        local,
        root.parent / "LegalIR/models/multilingual-e5-small",
        root.parent / "DSC2026-LegalIR-Huy/models/multilingual-e5-small",
    ]

    def valid_model_dir(p: Path) -> bool:
        return (
            p.is_dir()
            and (p / "config.json").is_file()
            and (
                (p / "model.safetensors").is_file()
                or (p / "pytorch_model.bin").is_file()
            )
        )

    for p in sibling_candidates:
        p = p.resolve()
        if valid_model_dir(p):
            safe = p / "model.safetensors"
            if safe.is_file():
                got = sha256(safe)
                if got != HIST_E5_SAFETENSORS_SHA256:
                    raise RuntimeError(
                        "Local multilingual-e5-small weight hash mismatch: "
                        f"{got} != {HIST_E5_SAFETENSORS_SHA256} at {p}"
                    )
                print(
                    f"  historical E5-small resolved -> {p} "
                    f"(SHA256 {got[:16]}...)",
                    flush=True,
                )
            else:
                print(
                    f"  historical E5-small resolved -> {p} "
                    "(legacy pytorch_model.bin; no safetensors hash gate)",
                    flush=True,
                )
            return p

    print(
        f"  Historical E5-small is not present locally.\n"
        f"  Downloading pinned {HIST_E5_HF_ID}@"
        f"{HIST_E5_HF_REVISION[:8]} -> {local}",
        flush=True,
    )

    from huggingface_hub import snapshot_download

    local.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=HIST_E5_HF_ID,
        revision=HIST_E5_HF_REVISION,
        local_dir=str(local),
        allow_patterns=[
            "config.json",
            "model.safetensors",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "sentencepiece.bpe.model",
        ],
    )

    if not valid_model_dir(local):
        raise RuntimeError(
            f"E5-small download completed but model directory is incomplete: "
            f"{local}"
        )

    safe = local / "model.safetensors"
    if safe.is_file():
        got = sha256(safe)
        if got != HIST_E5_SAFETENSORS_SHA256:
            raise RuntimeError(
                "Downloaded multilingual-e5-small weight hash mismatch: "
                f"{got} != {HIST_E5_SAFETENSORS_SHA256}"
            )
        print(
            f"  E5-small weight SHA256 verified: {got[:16]}...",
            flush=True,
        )

    return local


def load_exact_historical_e5_helper(root: Path):
    """
    Load the exact benchmark_e5_holdouts.encode restored in commit
    9dd83cdbc9926a2e920476867f7d9e8c66c93f6c.

    Hard-gate the file by its Git blob SHA so we never silently use a modified
    helper with the same function name.
    """
    source = root / "benchmark_e5_holdouts.py"
    if not source.is_file():
        raise FileNotFoundError(
            f"Missing restored historical helper: {source}"
        )

    try:
        proc = subprocess.run(
            ["git", "hash-object", str(source)],
            cwd=str(root),
            check=True,
            capture_output=True,
            text=True,
        )
        blob_sha = proc.stdout.strip()
    except Exception as exc:
        raise RuntimeError(
            "Could not verify benchmark_e5_holdouts.py Git blob SHA: "
            f"{exc}"
        )

    if blob_sha != HIST_BENCHMARK_E5_GIT_BLOB_SHA1:
        raise RuntimeError(
            "benchmark_e5_holdouts.py does not match the restored historical "
            "source from commit 9dd83cdb: "
            f"{blob_sha} != {HIST_BENCHMARK_E5_GIT_BLOB_SHA1}"
        )

    from benchmark_e5_holdouts import encode, average_pool

    # Semantic source-contract assertions. These intentionally mirror the
    # historical file and guard against importing a shadowed module.
    import inspect
    encode_src = inspect.getsource(encode)
    pool_src = inspect.getsource(average_pool)

    required_encode = [
        'max_length=384',
        'padding=True',
        'truncation=True',
        'return_tensors="pt"',
        'model(**batch).last_hidden_state',
        'F.normalize(average_pool(hidden, batch["attention_mask"]), p=2, dim=1)',
        'vectors.float().cpu().numpy()',
        'np.vstack(output)',
    ]
    required_pool = [
        'masked_fill(~attention_mask[..., None].bool(), 0.0)',
        'masked.sum(dim=1) / attention_mask.sum(dim=1)[..., None]',
    ]
    missing = [
        frag for frag in required_encode if frag not in encode_src
    ] + [
        frag for frag in required_pool if frag not in pool_src
    ]
    if missing:
        raise RuntimeError(
            "Historical E5 helper semantic contract mismatch; missing source "
            f"fragments: {missing}"
        )

    print(
        "  benchmark_e5_holdouts.py EXACT source gate PASS "
        f"(git blob {blob_sha[:12]}...)",
        flush=True,
    )
    print(
        "  E5 encode contract: masked mean-pool in model dtype -> "
        "L2 normalize -> float32 NumPy",
        flush=True,
    )
    return encode


def validate_exact_historical_e5_public_scores(
    root: Path,
    model,
    tokenizer,
    encode_impl,
    raw_documents,
):
    """
    Validate the exact restored helper/model/text contract against historical
    gpu_scores.checkpoint.pkl before private inference.

    Probe:
      - 32 queries spread across all 1,000 public rows;
      - every public query whose CPU top20 contains an empty-passage document.

    No labels are used.
    """
    from benchmark_jina_reranker_holdouts import top_passages

    hist = load_pickle(
        root / "results/burst_gpu_threeview/gpu_scores.checkpoint.pkl"
    )["scores"]
    cpu = load_pickle(
        root / "results/burst_gpu_threeview/cpu_top20.pkl"
    )["rankings"]
    public_ids, public_questions = load_questions(
        root
        / "DSC2026-LegalIR-main/v4_run/public_test_dataset/public-official.json"
    )

    if len(public_ids) != 1000:
        raise RuntimeError(
            f"Unexpected public query count: {len(public_ids)}"
        )

    spread_idx = np.linspace(0, len(public_ids) - 1, 32, dtype=int)
    spread = [public_ids[i] for i in spread_idx]

    empty_docs = raw_documents.empty_doc_ids()
    empty_qids = [
        q for q in public_ids
        if any(d in empty_docs for d in cpu[q][:20])
    ]

    sample_ids = []
    seen = set()
    for q in spread + empty_qids:
        if q not in seen:
            sample_ids.append(q)
            seen.add(q)

    print(
        f"  exact E5 public parity sample: spread=32 | "
        f"empty-doc queries={len(empty_qids)} | "
        f"total={len(sample_ids)}",
        flush=True,
    )

    diffs = []
    exact_order = exact_top5 = exact_top10 = 0
    worst = {
        "abs_diff": -1.0,
        "qid": None,
        "doc": None,
        "calc": None,
        "hist": None,
    }

    for qi, q in enumerate(sample_ids, 1):
        question = public_questions[q]
        base_docs = cpu[q][:20]
        owners, passages = [], []

        for d in base_docs:
            for p in top_passages(
                question,
                raw_documents[d],
                count=HIST_GPU_E5["passages_per_doc"],
            ):
                owners.append(d)
                passages.append(HIST_GPU_E5["passage_prefix"] + p)

        qv = encode_impl(
            model,
            tokenizer,
            [HIST_GPU_E5["query_prefix"] + question],
            1,
            HIST_GPU_E5["max_length"],
        )[0]
        pv = encode_impl(
            model,
            tokenizer,
            passages,
            32,
            HIST_GPU_E5["max_length"],
        )

        # Exact historical NumPy float32 matmul contract.
        if qv.dtype != np.float32 or pv.dtype != np.float32:
            raise RuntimeError(
                f"Historical encode dtype mismatch qid={q}: "
                f"q={qv.dtype} p={pv.dtype}"
            )

        calc = {d: -1.0 for d in base_docs}
        for d, score in zip(owners, pv @ qv):
            calc[d] = max(calc[d], float(score))

        old = hist[q]["e5"]
        if set(old) != set(base_docs):
            raise RuntimeError(
                f"Historical E5 doc-set mismatch qid={q}: "
                f"hist={len(old)} base={len(base_docs)}"
            )

        a = np.asarray([calc[d] for d in base_docs], dtype=np.float64)
        b = np.asarray([float(old[d]) for d in base_docs], dtype=np.float64)
        delta = a - b
        diffs.extend(delta.tolist())

        j = int(np.argmax(np.abs(delta)))
        if abs(delta[j]) > worst["abs_diff"]:
            d = base_docs[j]
            worst = {
                "abs_diff": float(abs(delta[j])),
                "qid": q,
                "doc": d,
                "calc": float(a[j]),
                "hist": float(b[j]),
            }

        cr = sorted(base_docs, key=lambda d: (-calc[d], d))
        hr = sorted(base_docs, key=lambda d: (-float(old[d]), d))
        exact_order += int(cr == hr)
        exact_top5 += int(cr[:5] == hr[:5])
        exact_top10 += int(cr[:10] == hr[:10])

        if qi % 16 == 0 or qi == len(sample_ids):
            print(
                f"    E5 public parity {qi}/{len(sample_ids)}",
                flush=True,
            )

    diffs = np.asarray(diffs, dtype=np.float64)
    metrics = {
        "queries": len(sample_ids),
        "docs": int(diffs.size),
        "exact_order": exact_order,
        "exact_top5": exact_top5,
        "exact_top10": exact_top10,
        "mean_abs_diff": float(np.mean(np.abs(diffs))),
        "max_abs_diff": float(np.max(np.abs(diffs))),
        "rmse": float(np.sqrt(np.mean(diffs ** 2))),
        "worst": worst,
    }

    print(
        "  EXACT HISTORICAL E5 PUBLIC PARITY: "
        f"top5={exact_top5}/{len(sample_ids)} | "
        f"top10={exact_top10}/{len(sample_ids)} | "
        f"full={exact_order}/{len(sample_ids)} | "
        f"MAE={metrics['mean_abs_diff']:.9f} | "
        f"MAX={metrics['max_abs_diff']:.9f}",
        flush=True,
    )
    print(
        f"  worst E5 residual: qid={worst['qid']} doc={worst['doc']} "
        f"{worst['calc']:.8f}->{worst['hist']:.8f}",
        flush=True,
    )

    n = len(sample_ids)

    # Historical source/model/text contracts are already hard-gated above:
    #   - exact benchmark_e5_holdouts.py Git blob
    #   - exact multilingual-e5-small weight SHA256
    #   - exact historical raw-passage semantics
    #   - exact historical gpu_scores artifact
    #
    # The remaining differences are therefore allowed only as bounded
    # floating-point/runtime drift. D1 depends on E5 score geometry, so we
    # require perfect E5 Top-5 ordering on the parity sample and very small
    # raw-score residuals. We intentionally do NOT require bit-identical
    # full Top-20 ordering because that is unnecessarily sensitive to tiny
    # FP16/CUDA kernel differences on near-tied documents.
    top10_floor = math.floor(0.98 * n)
    full_order_floor = math.floor(0.90 * n)

    parity_pass = (
        exact_top5 == n
        and exact_top10 >= top10_floor
        and exact_order >= full_order_floor
        and metrics["mean_abs_diff"] <= 0.00020
        and metrics["max_abs_diff"] <= 0.00150
    )

    if not parity_pass:
        raise RuntimeError(
            "EXACT RESTORED E5 HELPER FAILED BOUNDED HISTORICAL PARITY. "
            "Do not run private E5. "
            f"required: top5={n}/{n}, "
            f"top10>={top10_floor}/{n}, "
            f"full>={full_order_floor}/{n}, "
            "MAE<=0.00020, MAX<=0.00150; "
            f"observed={metrics}"
        )

    metrics["parity_policy"] = {
        "status": "PASS_BOUNDED_NUMERIC_DRIFT",
        "required_top5": n,
        "required_top10": top10_floor,
        "required_full_order": full_order_floor,
        "max_mean_abs_diff": 0.00020,
        "max_abs_diff": 0.00150,
        "reason": (
            "Exact historical source/model/text/artifact contracts verified; "
            "remaining difference accepted only as bounded FP16/CUDA runtime drift."
        ),
    }
    print(
        "  E5 HISTORICAL PARITY ACCEPTED: exact Top-5 preserved on all "
        f"{n} probe queries; residual numerical drift is within bounds.",
        flush=True,
    )

    return metrics



def score_historical_e5_channel(
    root: Path,
    cache_path: Path,
    questions,
    ids,
    base,
    documents,
):
    """
    Exact three-view E5 score contract from run_burst_gpu_submission.py.

    Only CPU-base top20 docs are scored. Expanded/corpus-only D1 candidates
    intentionally remain missing for this channel, exactly like historical D1.
    """
    from transformers import AutoModel, AutoTokenizer
    from benchmark_jina_reranker_holdouts import top_passages

    model_path = ensure_historical_e5_small(root)
    raw_documents = HistoricalE5RawPassageStore(documents)

    saved = load_pickle(cache_path) if cache_path.is_file() else {}
    remaining = [
        q for q in ids
        if any(d not in saved.get(q, {}) for d in base[q])
    ]
    print(
        f"  historical E5-small cache "
        f"{len(ids)-len(remaining)}/{len(ids)}",
        flush=True,
    )
    if not remaining:
        return saved

    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(
        model_path, dtype=torch.float16
    ).eval().to("cuda")

    encode_impl = load_exact_historical_e5_helper(root)
    e5_gate = validate_exact_historical_e5_public_scores(
        root,
        model,
        tok,
        encode_impl,
        raw_documents,
    )
    encode_name = "benchmark_e5_holdouts.encode@9dd83cdb"

    started = time.perf_counter()
    for i, q in enumerate(remaining, 1):
        question = questions[q]
        owners, passages = [], []
        for d in base[q]:
            for p in top_passages(
                question,
                raw_documents[d],
                count=HIST_GPU_E5["passages_per_doc"],
            ):
                owners.append(d)
                passages.append(HIST_GPU_E5["passage_prefix"] + p)

        qv = encode_impl(
            model,
            tok,
            [HIST_GPU_E5["query_prefix"] + question],
            1,
            HIST_GPU_E5["max_length"],
        )[0]
        pv = encode_impl(
            model,
            tok,
            passages,
            32,
            HIST_GPU_E5["max_length"],
        )

        ds = {d: -1.0 for d in base[q]}
        for d, score in zip(owners, pv @ qv):
            ds[d] = max(ds[d], float(score))
        saved[q] = ds

        if i % 25 == 0 or i == len(remaining):
            save_pickle(cache_path, saved)
            rate = (time.perf_counter() - started) / i
            print(
                f"    historical e5-small {i}/{len(remaining)} "
                f"({rate:.2f}s/q)",
                flush=True,
            )

    del model, tok
    gc.collect()
    torch.cuda.empty_cache()
    return saved


def score_vnlegal_channel(
    root: Path,
    cache_dir: Path,
    questions,
    ids,
    candidates,
    documents,
    device,
):
    """
    STRICT historical vnlegal-lal scorer with scheduling-only optimization.

    NUMERICAL / LOGIC CONTRACT IS FROZEN:
      - same darklethelong/vnlegal-lal model
      - same CLS pooling via benchmark_aiteamvn_holdouts.encode_cls
      - query batch = 1
      - top_passages(..., count=2), unchanged implementation
      - candidate order unchanged
      - passage order unchanged
      - passage batch = 64 (historical source contract)
      - max_length = 512
      - same `pvec @ qvec`
      - same max passage score per document
      - same query processing order

    ONLY optimization:
      while GPU handles query q, ONE background CPU worker prepares the exact
      owners/passages lists for query q+1.

    One worker is intentional:
      - CPU prepare order stays serial and deterministic;
      - DocumentStore is never touched by two CPU workers concurrently;
      - no cross-query neural batching or reordered floating-point operations.

    Uses an isolated cache because earlier experimental runs may have mixed
    passage batch sizes (64 and 96), whose fp16 numerical equivalence cannot be
    guaranteed bit-for-bit.
    """
    from transformers import AutoModel, AutoTokenizer
    from benchmark_aiteamvn_holdouts import encode_cls
    from benchmark_jina_reranker_holdouts import top_passages
    from run_vnlegal_extra_channel_submission import ensure_vnlegal_model

    HIST_PASSAGE_BATCH = 64

    ensure_vnlegal_model(root)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Intentionally separate from old/mixed vnlegal_scores.pkl.
    path = cache_dir / "vnlegal_scores_exact64_prefetch.pkl"

    saved = load_pickle(path) if path.is_file() else {}
    remaining = [
        q for q in ids
        if any(d not in saved.get(q, {}) for d in candidates[q])
    ]
    print(
        f"vnlegal-lal STRICT64 cache "
        f"{len(ids)-len(remaining)}/{len(ids)} "
        f"-> {path.name}",
        flush=True,
    )
    if not remaining:
        return saved

    model_path = root / "models/vnlegal-lal"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(
        model_path,
        dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).eval().to(device)

    print(
        f"vnlegal-lal ready on {torch.cuda.get_device_name(0)} "
        f"| STRICT historical passage batch={HIST_PASSAGE_BATCH} "
        "| CPU lookahead=1",
        flush=True,
    )

    def prepare(q):
        # EXACT same CPU logic as the historical/synchronous scorer.
        question = questions[q]
        owners, passages = [], []
        for d in candidates[q]:
            for p in top_passages(question, documents[d], count=2):
                owners.append(d)
                passages.append(p)
        return q, question, owners, passages

    started = time.perf_counter()
    cpu_wait_total = 0.0
    neural_total = 0.0
    prepared_passages = 0

    # Safety proof on the first two queries:
    # prepare once in the worker, once synchronously, and require byte-for-byte
    # identical owner/passages order before enabling overlap. This costs only
    # two extra CPU preparations and catches any unexpected DocumentStore/
    # threading behavior.
    verify_n = min(2, len(remaining))

    with ThreadPoolExecutor(max_workers=1) as pool:
        # Start first CPU preparation.
        future = pool.submit(prepare, remaining[0])

        for i, expected_q in enumerate(remaining, 1):
            wait_started = time.perf_counter()
            q, question, owners, passages = future.result()
            cpu_wait_total += time.perf_counter() - wait_started

            if q != expected_q:
                raise RuntimeError(
                    f"vnlegal prefetch order drift: expected {expected_q}, got {q}"
                )

            # For the first two rows, prove that threaded preparation returns
            # exactly the same sequence as synchronous historical preparation.
            if i <= verify_n:
                rq, rquestion, rowners, rpassages = prepare(q)
                if (
                    rq != q
                    or rquestion != question
                    or rowners != owners
                    or rpassages != passages
                ):
                    raise RuntimeError(
                        f"vnlegal CPU prefetch parity failed qid={q}"
                    )
                print(
                    f"  CPU prefetch parity qid={q}: "
                    f"owners/passages EXACT ({len(passages)} passages)",
                    flush=True,
                )

            # Submit q+1 BEFORE GPU work for q. One worker means CPU preparation
            # remains serial; it simply overlaps with CUDA work below.
            if i < len(remaining):
                future = pool.submit(prepare, remaining[i])

            neural_started = time.perf_counter()

            # Historical query encoding: batch=1, max_length=512.
            qvec = encode_cls(
                model,
                tokenizer,
                [question],
                1,
                512,
            )[0]

            ds = dict(saved.get(q, {}))

            if passages:
                # Historical passage encoding: batch=64, max_length=512.
                pvec = encode_cls(
                    model,
                    tokenizer,
                    passages,
                    HIST_PASSAGE_BATCH,
                    512,
                )
                for d, s in zip(owners, pvec @ qvec):
                    ds[d] = max(ds.get(d, -1e9), float(s))

            neural_total += time.perf_counter() - neural_started
            prepared_passages += len(passages)
            saved[q] = ds

            if i % 25 == 0 or i == len(remaining):
                save_pickle(path, saved)
                elapsed = time.perf_counter() - started
                rate = elapsed / i
                left = len(remaining) - i
                avg_cpu_wait = cpu_wait_total / i
                avg_neural = neural_total / i
                print(
                    f"  vnlegal STRICT64 {i}/{len(remaining)} "
                    f"{rate:.2f}s/q eta={rate*left/60:.1f}m "
                    f"| avg_wait_cpu={avg_cpu_wait:.2f}s "
                    f"avg_neural={avg_neural:.2f}s "
                    f"avg_passages={prepared_passages/i:.1f}",
                    flush=True,
                )

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return saved



def print_private_cache_status(cache: Path, ids, candidates) -> None:
    """Read-only cache inventory. Never mutates cache files."""
    specs = [
        ("E5-small", cache / "d1_e5_small_threeview_scores.pkl"),
        ("crossenc", cache / "crossenc_scores.pkl"),
        ("AITeamVN-FT bi", cache / "aiteamvn_ft_scores.pkl"),
        ("Jina-FT", cache / "jina_ft_scores.pkl"),
        ("title_embed", cache / "title_embed_scores.pkl"),
    ]
    print("  cache-preservation audit (read-only):", flush=True)
    for name, path in specs:
        if not path.is_file():
            print(f"    {name:18s} missing -> will create", flush=True)
            continue
        try:
            obj = load_pickle(path)
            covered = sum(
                1 for q in ids
                if q in obj and set(obj.get(q, {})) >= set(candidates[q])
            )
            print(
                f"    {name:18s} {covered}/{len(ids)} complete-query rows | PRESERVE+RESUME | {path.name}",
                flush=True,
            )
        except Exception as exc:
            # Never delete/reset a suspicious cache automatically. Fail loudly instead.
            raise RuntimeError(
                f"Existing cache cannot be read safely: {path}. "
                f"v14-fast refuses to delete/reset it automatically. Error: {exc}"
            ) from exc

    vn = cache / "vnlegal" / "vnlegal_scores_exact64_prefetch.pkl"
    if vn.is_file():
        try:
            obj = load_pickle(vn)
            covered = sum(
                1 for q in ids
                if q in obj and set(obj.get(q, {})) >= set(candidates[q])
            )
            print(f"    {'vnlegal':18s} {covered}/{len(ids)} complete-query rows | PRESERVE+RESUME", flush=True)
        except Exception as exc:
            raise RuntimeError(
                f"Existing vnlegal cache cannot be read safely: {vn}. "
                f"Refusing automatic reset. Error: {exc}"
            ) from exc


def score_bi_channel(root: Path, cache_path: Path, questions, ids, candidates, documents, model_path: Path, passage_batch_size=96):
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
        pv = encode_cls(model, tok, passages, passage_batch_size, 512)
        ds = {d: -1e9 for d in candidates[q]}
        for d, s in zip(owners, pv @ qv):
            ds[d] = max(ds[d], float(s))
        saved[q] = ds
        if i % 25 == 0 or i == len(remaining):
            save_pickle(cache_path, saved)
            print(f"    bi {i}/{len(remaining)} {(time.perf_counter()-started)/i:.2f}s/q", flush=True)
    del model, tok; gc.collect(); torch.cuda.empty_cache()
    return saved


def score_jina_ft_channel(root: Path, cache_path: Path, questions, ids, candidates, documents, batch_size=64):
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


def score_title_channel(root: Path, cache_path: Path, questions, ids, candidates, documents, doc_batch_size=96, query_batch_size=64):
    """Score title channel without ever deleting/resetting an existing cache.

    If a complete cache exists, return it immediately. If an incomplete cache exists,
    keep every existing score, compute the current full title representation once, merge
    only missing q/doc scores in memory, then atomically replace the cache after success.
    An interruption before the final atomic save leaves the old cache untouched.
    """
    from transformers import AutoModel, AutoTokenizer
    from benchmark_aiteamvn_holdouts import encode_cls
    from tune_title_features import title_table

    saved = load_pickle(cache_path) if cache_path.is_file() else {}
    complete = (
        set(saved) >= set(ids)
        and all(set(saved.get(q, {})) >= set(candidates[q]) for q in ids)
    )
    if complete:
        print("  title_embed cache complete", flush=True)
        return saved

    if cache_path.is_file():
        covered_q = sum(
            1 for q in ids
            if set(saved.get(q, {})) >= set(candidates[q])
        )
        print(
            f"  title_embed cache partial {covered_q}/{len(ids)}; preserving existing entries",
            flush=True,
        )

    titles = title_table(documents, ids, candidates)
    model_path = root / "models/AITeamVN_Vietnamese_Embedding"
    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path, dtype=torch.float16).eval().to("cuda")
    dids = list(titles)
    print(f"  title_embed documents={len(dids)}", flush=True)
    tvec = encode_cls(model, tok, [titles[d] or "khong co tieu de" for d in dids], doc_batch_size, 128)
    tmap = dict(zip(dids, tvec))
    qvec = encode_cls(model, tok, [questions[q] for q in ids], query_batch_size, 128)
    qmap = dict(zip(ids, qvec))

    merged = {q: dict(saved.get(q, {})) for q in ids}
    for q in ids:
        row = merged[q]
        for d in candidates[q]:
            if d not in row:
                row[d] = float(qmap[q] @ tmap[d])

    # save_pickle is atomic (tmp + os.replace). Existing cache survives if scoring crashes.
    save_pickle(cache_path, merged)
    del model, tok
    gc.collect(); torch.cuda.empty_cache()
    return merged


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


def validate_d1(submission, ids, valid_docs):
    if set(submission) != set(ids):
        raise RuntimeError("D1 qid population mismatch")
    for q in ids:
        ans = [str(d) for d in submission[q]["answer"]]
        if len(ans) != 5 or len(set(ans)) != 5 or any(d not in valid_docs for d in ans):
            raise RuntimeError(f"Invalid D1 output q={q}: {ans}")


def preflight_d1_only(root: Path, device: str):
    """
    Fail-fast audit for D1-only inference through final submission packaging.

    v14 intentionally does NOT check REL_L0, Fold-0 BGE CE, evidence DB,
    VietLegal-E5 confirmation assets, or any other post-D1 dependency.

    It performs no private-label access and no private inference.
    """
    print("[PRE/D1] Fail-fast audit for D1-only stages...", flush=True)

    errors = []
    notes = {}

    # -------------------------------
    # A. Runtime / Python modules.
    # -------------------------------
    if str(device).startswith("cuda"):
        if not torch.cuda.is_available():
            errors.append("CUDA requested but torch.cuda.is_available() is False")
        else:
            notes["cuda_device"] = torch.cuda.get_device_name(0)

    required_root_modules = [
        "benchmark_e5_holdouts",
        "benchmark_jina_reranker_holdouts",
        "benchmark_aiteamvn_holdouts",
        "run_vnlegal_extra_channel_submission",
        "tune_title_features",
        "tune_expanded_fusion_selection",
        "tune_doctype_features",
        "tune_citation_graph",
    ]
    for mod in required_root_modules:
        try:
            if importlib.util.find_spec(mod) is None:
                errors.append(f"Python module not importable: {mod}")
        except Exception as exc:
            errors.append(f"Python module probe failed: {mod}: {exc}")

    try:
        import safetensors  # noqa: F401
    except Exception as exc:
        errors.append(f"safetensors import failed: {exc}")

    try:
        import transformers  # noqa: F401
    except Exception as exc:
        errors.append(f"transformers import failed: {exc}")

    # -------------------------------
    # B. Historical artifacts / E5.
    # -------------------------------
    required_files = {
        "historical Large-LTR": root / "results/burst_large_ltr/best_model.pkl",
        "historical Legal-LTR": root / "results/burst_legal_features/validation_model.pkl",
        "historical pairwise": root / "results/burst_empirical_pairwise/model.pkl",
        "historical Jina state": root / "results/jina_reranker/burst_pairwise_state.pt",
        "historical public CPU top20": root / "results/burst_gpu_threeview/cpu_top20.pkl",
        "historical GPU scores": root / "results/burst_gpu_threeview/gpu_scores.checkpoint.pkl",
        "historical E5 helper": root / "benchmark_e5_holdouts.py",
    }

    # Stage 6c-6f models.
    required_paths = {
        "AITeamVN FT bi-encoder": root / "models/from_drive/AITeamVN_Vietnamese_Embedding",
        "AITeamVN title/base embedding": root / "models/AITeamVN_Vietnamese_Embedding",
        "AITeamVN crossencoder": root / "models/AITeamVN_Vietnamese_Reranker",
        "Jina base": root / "models/jina-reranker-v2-base-multilingual",
        "Jina FT weights": root / "models/from_drive/jina_finetuned/model.safetensors",
    }

    for label, p in required_files.items():
        if not p.is_file():
            errors.append(f"Missing file [{label}]: {p}")

    for label, p in required_paths.items():
        if not p.exists():
            errors.append(f"Missing path [{label}]: {p}")
        elif p.is_dir() and not any(p.iterdir()):
            errors.append(f"Empty model directory [{label}]: {p}")

    # Exact source helper and exact E5 weights.
    if not errors:
        try:
            load_exact_historical_e5_helper(root)
        except Exception as exc:
            errors.append(f"Historical E5 helper gate failed: {exc}")

        try:
            e5_dir = ensure_historical_e5_small(root)
            notes["historical_e5_dir"] = str(e5_dir)
        except Exception as exc:
            errors.append(f"Historical E5 model gate failed: {exc}")

        try:
            notes["historical_gpu_audit"] = audit_historical_gpu_score_artifact(root)
        except Exception as exc:
            errors.append(f"Historical GPU artifact gate failed: {exc}")

    # Stage 6b is allowed to auto-download, but do it NOW rather than after
    # 2,080 E5 queries so network/model issues fail early.
    if not errors:
        try:
            from run_vnlegal_extra_channel_submission import ensure_vnlegal_model
            ensure_vnlegal_model(root)
            vn_path = root / "models/vnlegal-lal"
            if not vn_path.is_dir() or not any(vn_path.iterdir()):
                raise RuntimeError(f"vnlegal-lal provisioning incomplete: {vn_path}")
            notes["vnlegal_model"] = str(vn_path)
        except Exception as exc:
            errors.append(f"vnlegal-lal pre-provision failed: {exc}")

    # -------------------------------
    # C. v14 stops at D1: no REL/BGE/evidence dependencies.
    # -------------------------------
    if errors:
        print("  PRE-FLIGHT FAILED:", flush=True)
        for i, err in enumerate(errors, 1):
            print(f"    {i:02d}. {err}", flush=True)
        raise RuntimeError(
            f"D1-only preflight found {len(errors)} blocking issue(s). "
            "Nothing expensive was started."
        )

    print(
        "  PRE-FLIGHT PASS: D1 Stage 6b-6f models and exact historical E5 provenance are ready.",
        flush=True,
    )
    if notes.get("cuda_device"):
        print(f"  CUDA: {notes['cuda_device']}", flush=True)

    return {
        "historical_gpu_audit": notes.get("historical_gpu_audit"),
        "notes": notes,
    }



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
    ap.add_argument("--bi-passage-batch-size", type=int, default=96,
                    help="AITeamVN-FT passage encoding batch size; lower to 64/32 if CUDA OOM.")
    ap.add_argument("--jina-ft-batch-size", type=int, default=64,
                    help="Jina-FT pair batch size; lower to 32/16 if CUDA OOM.")
    ap.add_argument("--title-doc-batch-size", type=int, default=96,
                    help="Title embedding document batch size; lower if CUDA OOM.")
    ap.add_argument("--title-query-batch-size", type=int, default=64,
                    help="Title embedding private-query batch size; lower if CUDA OOM.")
    ap.add_argument("--historical-e5-model", default=E5_MODEL_ID, help="Deprecated; exact runner uses restored models/multilingual-e5-small contract.")
    ap.add_argument("--run-cached-public-d1-parity", action="store_true",
                    help="Optional safety audit only; OFF by default in v14-fast.")
    args = ap.parse_args()

    root = args.repo_root.resolve()
    data_dir = root / "DSC2026-LegalIR-main/v4_run/public_test_dataset"
    private_path = data_dir / args.private_file
    if not private_path.is_file():
        raise FileNotFoundError(private_path)
    sys.path.insert(0, str(root))
    from run_burst_expanded_fusion_submission import DocumentStore

    preflight = preflight_d1_only(root, args.device)

    out = root / "results/manual/huy_private_d1_rel_l0_exact_v1"
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

    print("[0/7] FAST START: public CPU reconstruction/parity is intentionally disabled in v14-fast.", flush=True)
    # IMPORTANT: do NOT call validate_restored_public_cpu_parity() here.
    # That audit reconstructs all 1,000 public CPU rankings and normalizes all 8,532 docs
    # on every launch. It has already passed in the strict v13 workflow and contributes
    # nothing to the private score caches we are trying to finish tonight.
    cpu_public_parity = {
        "status": "SKIPPED_V14_FAST_ALREADY_VALIDATED_IN_V13",
        "reason": "Avoid repeated 8,532-doc public reconstruction; private caches remain fingerprint-guarded.",
    }
    historical_gpu_audit = preflight["historical_gpu_audit"]

    print("[1/7] Training production D1 on CAL600...", flush=True)
    scaler, d1_model, d1_meta = train_exact_d1(root, documents)

    if args.run_cached_public_d1_parity:
        print("[2/7] OPTIONAL cached-public D1 parity...", flush=True)
        public_parity = cached_public_d1_parity(root, documents, scaler, d1_model)
    else:
        print("[2/7] cached-public D1 parity skipped (v14-fast default)", flush=True)
        public_parity = {
            "status": "SKIPPED_V14_FAST_DEFAULT",
            "reason": "Enable only with --run-cached-public-d1-parity if desired.",
        }

    print("[3/7] Loading private-official queries...", flush=True)
    ids, questions = load_questions(private_path)
    print(f"  private queries={len(ids)} sha256={sha256(private_path)}", flush=True)

    print("[4/7] Private sparse retrieval + historical CPU base Top20...", flush=True)

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
        cache / "private_base_top20_historical_exact.pkl"
    )

    print("[5/7] Building private candidates + reusing compatible Stage-5 neural scores...", flush=True)
    shared_stage5 = (
        root / "results/manual/huy_private_d1_rel_l0_approx_v1/"
        "cache/candidate_generation"
    )
    if shared_stage5.is_dir():
        cand_cache = shared_stage5
        print(
            f"  reusing Stage-5 cache from previous run -> {cand_cache}",
            flush=True,
        )
    else:
        cand_cache = cache / "candidate_generation"
        cand_cache.mkdir(parents=True, exist_ok=True)
    candidates, expanded, corpus_sparse, expansion_scores, rerank_scores = build_target_candidates(
        root, cand_cache, questions, ids, retrieval, base, documents, args.device
    )
    sizes = [len(candidates[q]) for q in ids]
    print(f"  private candidates min={min(sizes)} mean={np.mean(sizes):.2f} max={max(sizes)}", flush=True)
    print_private_cache_status(cache, ids, candidates)

    print("[6/7] Private D1 score channels (resume/reuse first)...", flush=True)
    print("  [6a] EXACT historical E5-small three-view channel", flush=True)
    e5_scores = score_historical_e5_channel(
        root, cache / "d1_e5_small_threeview_scores.pkl",
        questions, ids, base, documents
    )
    print("  [6b] vnlegal-lal", flush=True)
    vn_dir = cache / "vnlegal"; vn_dir.mkdir(parents=True, exist_ok=True)
    vnlegal_scores = score_vnlegal_channel(
        root, vn_dir, questions, ids, candidates, documents, args.device,
    )
    print("  [6c] AITeamVN crossencoder", flush=True)
    crossenc_scores = score_crossenc_channel(
        root, cache / "crossenc_scores.pkl", questions, ids, candidates, documents
    )
    print("  [6d] AITeamVN fine-tuned bi-encoder", flush=True)
    aiteam_ft_scores = score_bi_channel(
        root, cache / "aiteamvn_ft_scores.pkl", questions, ids, candidates,
        documents, root / "models/from_drive/AITeamVN_Vietnamese_Embedding",
        passage_batch_size=args.bi_passage_batch_size,
    )
    print("  [6e] Jina fine-tuned", flush=True)
    jina_ft_scores = score_jina_ft_channel(
        root, cache / "jina_ft_scores.pkl", questions, ids, candidates, documents,
        batch_size=args.jina_ft_batch_size,
    )
    print("  [6f] title embedding", flush=True)
    title_scores = score_title_channel(
        root, cache / "title_embed_scores.pkl", questions, ids, candidates, documents,
        doc_batch_size=args.title_doc_batch_size,
        query_batch_size=args.title_query_batch_size,
    )

    print("[7/7] Inferring + packaging private D1 Top5...", flush=True)
    d1_top5, d1_scores = infer_target_d1(
        root, documents, ids, questions, candidates, base, expanded,
        corpus_sparse, expansion_scores, rerank_scores, e5_scores, vnlegal_scores,
        crossenc_scores, aiteam_ft_scores, jina_ft_scores, title_scores,
        scaler, d1_model, d1_meta["floors"]
    )
    d1_submission = {q: {"answer": list(d1_top5[q])} for q in ids}
    validate_d1(d1_submission, ids, valid_docs)
    d1_json, d1_zip = out / "D1_PRIVATE_V14_FAST.json", out / "D1_PRIVATE_V14_FAST.zip"
    dump(d1_json, d1_submission); zip_exact(d1_json, d1_zip)
    save_pickle(cache / "d1_private_scores.pkl", {"top5": d1_top5, "decision_scores": d1_scores})
    print(f"  D1 private ZIP={d1_zip}", flush=True)

    print("[DONE] Writing D1-only submission report and exiting...", flush=True)
    report = {
        "schema": "manual.private_d1_fast_v14.report",
        "status": "READY_FOR_PRIVATE_SUBMISSION_D1_ONLY_FAST_V14",
        "target": {
            "file": str(private_path),
            "sha256": sha256(private_path),
            "queries": len(ids),
        },
        "public_downstream_parity": public_parity,
        "public_cpu_restore_parity": cpu_public_parity,
        "historical_gpu_score_artifact": historical_gpu_audit,
        "fts_database": fts_meta,
        "cpu_base_reconstruction": cpu_base_artifacts,
        "cache_policy_v14": {
            "delete_existing_private_caches": False,
            "reset_existing_private_caches": False,
            "resume_partial_query_score_caches": True,
            "atomic_pickle_writes": True,
            "public_cpu_reconstruction": "DISABLED",
            "cached_public_d1_parity": "OPT_IN_ONLY",
        },
        "batch_policy_v14": {
            "note": (
                "Throughput-prioritized v14. Batch sizes intentionally differ from the "
                "strict historical reproduction path and may introduce tiny FP16/CUDA score drift."
            ),
            "aiteam_ft_passage_batch_size": args.bi_passage_batch_size,
            "jina_ft_pair_batch_size": args.jina_ft_batch_size,
            "title_doc_batch_size": args.title_doc_batch_size,
            "title_query_batch_size": args.title_query_batch_size,
        },
        "d1": {
            "training": d1_meta,
            "queries": len(ids),
            "candidate_pool_min": int(min(sizes)),
            "candidate_pool_mean": float(np.mean(sizes)),
            "candidate_pool_max": int(max(sizes)),
            "submission_json": str(d1_json),
            "submission_json_sha256": sha256(d1_json),
            "submission_zip": str(d1_zip),
            "submission_zip_sha256": sha256(d1_zip),
        },
        "omitted_v14": [
            "REL_L0 threshold/action stage",
            "VietLegal-E5 private REL query encoding",
            "Fold-0 BGE cross-encoder scoring",
            "REL adaptive-K packaging",
            "all post-D1 evidence/confirmation stages",
        ],
        "scientific_contract": {
            "private_labels_used": False,
            "d1_trained_on_cal600_labels": True,
            "post_d1_policy": "NONE_D1_TOP5_SUBMITTED_DIRECTLY",
        },
    }
    report_path = out / "PRIVATE_D1_FAST_V14_REPORT.json"
    dump(report_path, report)

    print("=" * 112)
    print("PRIVATE D1-ONLY V14 SUBMISSION READY")
    print(f"queries={len(ids)} | K=5 | REL_L0=DISABLED | Fold0-BGE-CE=DISABLED")
    print(f"D1 ZIP: {d1_zip}")
    print(f"Report: {report_path}")
    print(
        "NOTE: v14 intentionally prioritizes throughput: larger inference batches may "
        "cause tiny FP16/CUDA numerical drift versus the strict historical batch contract."
    )
    print("=" * 112)



if __name__ == "__main__":
    main()
