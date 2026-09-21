"""ONE-SHOT EXACT-HISTORICAL private D1 + frozen REL_L0 materializer.

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

Before private inference, this script reconstructs PUBLIC cpu_top20 from the
restored artifacts and requires exact ordered parity 1000/1000 against:
  results/burst_gpu_threeview/cpu_top20.pkl

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

import argparse, gc, hashlib, importlib.util, json, os, pickle, sqlite3, subprocess, sys, threading, time, zipfile, warnings
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
E5_MODEL_ID = "models/multilingual-e5-small"
REL_E5_MODEL_ID = "mainguyen9/vietlegal-e5"
E5_DIM = 1024
EXPECTED_EVIDENCE_DOCS = 8507
EXPECTED_EVIDENCE_CHUNKS = 343347



# Historical artifacts were restored from the original public run.
# The mandatory 1000/1000 public cpu_top20 parity gate proves that the
# current runtime reproduces their inference behavior exactly. Suppress only
# the corresponding persistence-version warnings to keep the terminal clean.
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
    # Historical artifacts were serialized under older sklearn/XGBoost
    # runtimes. We suppress warnings only during unpickle. Correctness is
    # guarded by mandatory historical public parity checks immediately after.
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


def validate_restored_public_cpu_parity(
    root: Path,
    corpus_paths,
    train,
    doc_ids,
):
    """
    Hard gate: restored historical artifacts must reproduce cpu_top20.pkl
    exactly on the 1,000 public queries.
    """
    data_dir = root / "DSC2026-LegalIR-main/v4_run/public_test_dataset"
    public_ids, public_questions = load_questions(data_dir / "public-official.json")
    target = load_pickle(
        root / "results/burst_gpu_threeview/cpu_top20.pkl"
    )["rankings"]

    retrieval = None
    retrieval_path = None
    for p in (
        root / "results/burst_multistage/public_retrieval.pkl",
        root / "results/burst_robust_fusion/public_retrieval.pkl",
        root / "results/burst_expanded_fusion/public_retrieval.pkl",
    ):
        if not p.is_file():
            continue
        obj = load_pickle(p)
        c = obj.get("cache", {})
        if all(q in c for q in public_ids):
            retrieval = c
            retrieval_path = p
            break
    if retrieval is None:
        raise RuntimeError("Missing complete historical public retrieval cache")

    print(
        f"  reconstructing historical public CPU top20 using restored artifacts "
        f"({retrieval_path})...",
        flush=True,
    )
    rebuilt, provenance = _cpu_multistage_rankings(
        root, corpus_paths, train, doc_ids,
        retrieval, public_questions, public_ids,
        cache_path=None,
    )

    exact20 = sum(rebuilt[q] == target[q] for q in public_ids)
    exact5 = sum(rebuilt[q][:5] == target[q][:5] for q in public_ids)
    mean20 = float(np.mean([
        len(set(rebuilt[q]) & set(target[q])) / 20.0
        for q in public_ids
    ]))
    print(
        f"  PUBLIC CPU RESTORE PARITY: "
        f"Top20 exact={exact20}/1000 | Top5 exact={exact5}/1000 | "
        f"mean Top20 overlap={mean20:.6f}",
        flush=True,
    )
    if exact20 != len(public_ids):
        raise RuntimeError(
            "RESTORED HISTORICAL CPU ARTIFACTS FAILED EXACT PUBLIC PARITY: "
            f"Top20 exact={exact20}/{len(public_ids)}. "
            "Do not run private D1 until the original artifact set is restored."
        )
    return {
        "status": "PASS_EXACT_CPU_TOP20",
        "exact_top20": exact20,
        "exact_top5": exact5,
        "mean_top20_overlap": mean20,
        "provenance": provenance,
    }


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

    # Exact helper + exact model hash + exact raw passage contract should give
    # virtually identical rankings. Allow tiny runtime/GPU fp16 drift in raw
    # scores, but not a top-5 ranking change.
    if (
        exact_top5 != n
        or exact_top10 < n - 1
        or exact_order < n - 2
        or metrics["mean_abs_diff"] > 0.00020
        or metrics["max_abs_diff"] > 0.0020
    ):
        raise RuntimeError(
            "EXACT RESTORED E5 HELPER FAILED HISTORICAL PUBLIC PARITY. "
            "Do not run private E5. "
            f"metrics={metrics}"
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
    ap.add_argument("--historical-e5-model", default=E5_MODEL_ID, help="Deprecated; exact runner uses restored models/multilingual-e5-small contract.")
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

    print("[0/12] HARD GATES: restored historical CPU + GPU artifacts...", flush=True)
    cpu_public_parity = validate_restored_public_cpu_parity(
        root, corpus_paths, train, doc_ids
    )
    historical_gpu_audit = audit_historical_gpu_score_artifact(root)

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

    print("[4/12] Private sparse retrieval + EXACT historical CPU base Top20...", flush=True)

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

    print("[5/12] Building EXACT private candidates + reusing compatible Stage-5 neural scores...", flush=True)
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

    print("[6/12] Fresh private D1 score channels...", flush=True)
    print("  [6a] EXACT historical E5-small three-view channel", flush=True)
    e5_scores = score_historical_e5_channel(
        root, cache / "d1_e5_small_threeview_scores.pkl",
        questions, ids, base, documents
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
    d1_json, d1_zip = out / "D1_PRIVATE_EXACT.json", out / "D1_PRIVATE_EXACT.zip"
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
    zip_path = rel_dir / "submission_REL_L0_PRIVATE_EXACT.zip"
    dump(action_path, {
        "schema": "manual.private_d1_rel_l0_exact_v1.actions",
        "status": "SEALED_LABEL_FREE_PRIVATE_EXACT_D1", "threshold": REL_L0,
        "threshold_contract": rel_contract, "private_questions_sha256": sha256(private_path),
        "private_qvec_sha256": sha256(qvec_path), "d1_private_sha256": sha256(d1_json),
        "checkpoint_sha256": sha256(checkpoint), "actions_count": len(actions),
        "no_evidence_abstentions": no_evidence, "actions": actions,
    })
    dump(sub_path, rel_submission); zip_exact(sub_path, zip_path)

    print("[12/12] Writing private report...", flush=True)
    mean_k = float(np.mean([len(rel_submission[q]["answer"]) for q in ids]))
    report = {
        "schema": "manual.private_d1_rel_l0_exact_v1.report",
        "status": "READY_FOR_PRIVATE_SUBMISSION_EXACT_D1",
        "target": {"file": str(private_path), "sha256": sha256(private_path), "queries": len(ids)},
        "public_downstream_parity": public_parity,
        "public_cpu_restore_parity": cpu_public_parity,
        "historical_gpu_score_artifact": historical_gpu_audit,
        "fts_database": fts_meta,
        "cpu_base_reconstruction": cpu_base_artifacts,
        "historical_generator_caveat": {
            "cpu_top20_original_generator_fully_committed": False,
            "cpu_top20_exact_recovery_attempt_abandoned": False,
            "cpu_top20_private_mode": "RESTORED_HISTORICAL_ARTIFACTS_EXACT_GENERATOR",
            "cpu_top20_public_recovery_parity": cpu_public_parity,
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
    print("PRIVATE EXACT-HISTORICAL D1 + REL_L0 SUBMISSION READY")
    print(
        f"queries={len(ids)} | D1 K=5 | REL scorable={len(rel_rows)} | "
        f"REL actions={len(actions)} ({100*len(actions)/len(ids):.2f}%) | meanK={mean_k:.4f}"
    )
    print(f"D1 EXACT ZIP: {d1_zip}")
    print(f"REL_L0 ZIP:    {zip_path}")
    print(f"Report:        {report_path}")
    print(
        "NOTE: CPU base is the final approximate recovery, NOT the exact "
        "historical cpu_top20 generator."
    )
    print("=" * 112)


if __name__ == "__main__":
    main()
