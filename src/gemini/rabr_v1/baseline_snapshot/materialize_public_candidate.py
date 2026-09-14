"""Materialize the evidence-matched Huy-fasttrack public candidate.

The public endpoint intentionally omits Huy's frozen Jina cross-encoder until
the exact public interface has been scored.  Every remaining neural and sparse
channel already has public cache coverage.  The LR is trained once on 5-way
cross-fitted label-derived features, then refit on all evaluable labels.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import run_huy_5fold_fasttrack as core


sys.path.insert(0, str(core.ROOT))
sys.path.insert(0, str(core.WORKSPACE / "LegalIR/scripts"))
from tune_burst_supervised_profile_bm25 import build_profiles, profile_rank
from exp_final_memory_ltr_probe import MEMORY_NAMES, memory_features, support_index


OUT = core.OUT / "submission_candidate_no_huy_jina"
PUBLIC_JSON = core.DATA / "public-official.json"
PUBLIC_SCORE_DB = core.ROOT / "cache/research_v2_open_rl/v2_anchor_submission_candidate/public_scores.sqlite"
TRAIN_LAL = core.WORKSPACE / "LegalIR/cache/exp109b_encoder_complementarity/embeddings/vnlegal_lal/queries.npz"
PUBLIC_LAL_DIR = core.WORKSPACE / "LegalIR/cache/exp112_task_adaptive_retrieval/public_vectors/lal"
PUBLIC_JINA_DB = core.ROOT / "cache/huy_fasttrack/public_frozen_jina_scores.sqlite"
CANONICAL_CONTEXTS = core.ROOT / "cache/research_v2_forensic/kaggle_input/research-v2-jina-boundary-v4/V2_CONTEXTS.jsonl"
CONFIG = {
    "rank_views": [
        "adapted_e5", "lal_native", "legalir_jina", "huy_profile",
        "legalir_bm25", "legalir_trigram",
    ],
    "score_channels": [
        "adapted_e5", "lal_native", "legalir_bm25", "legalir_trigram",
    ],
    "metadata": ["citation", "lal_memory"],
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def tree_sha256(path: Path) -> str:
    h = hashlib.sha256()
    for child in sorted((p for p in path.rglob("*") if p.is_file()), key=lambda p: p.relative_to(path).as_posix()):
        h.update(child.relative_to(path).as_posix().encode("utf-8"))
        h.update(bytes.fromhex(sha256(child)))
    return h.hexdigest()


def normalize(values):
    values = np.asarray(values, dtype=np.float32)
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)


def load_public():
    payload = json.loads(PUBLIC_JSON.read_text(encoding="utf-8"))
    questions = {str(q): str(row["question"]) for q, row in payload.items()}
    con = sqlite3.connect(f"file:{PUBLIC_SCORE_DB.as_posix()}?mode=ro", uri=True)
    con.execute("PRAGMA query_only=ON")
    pools, e5_scores, lal_scores = defaultdict(list), defaultdict(dict), defaultdict(dict)
    for table, target in (("e5", e5_scores), ("lal", lal_scores)):
        for qid, rank, doc, score in con.execute(
            f"SELECT qid,rank,doc_id,score FROM {table} ORDER BY qid,rank"
        ):
            qid, doc = str(qid), str(doc)
            target[qid][doc] = float(score)
            if table == "e5":
                pools[qid].append(doc)
    integrity = con.execute("PRAGMA integrity_check").fetchone()[0]
    con.close()
    pools = dict(pools)
    if integrity != "ok" or set(pools) != set(questions) or any(len(v) != 50 for v in pools.values()):
        raise RuntimeError("public score database/query/pool contract mismatch")
    for qid in pools:
        if set(pools[qid]) != set(lal_scores[qid]):
            raise RuntimeError(f"public E5/LAL pool mismatch: {qid}")
    return pools, questions, dict(e5_scores), dict(lal_scores), integrity


def source_channel(channel: str, pools):
    con = sqlite3.connect(f"file:{core.SOURCE_DB.as_posix()}?mode=ro", uri=True)
    con.execute("PRAGMA query_only=ON")
    orders, scores = {}, {}
    for qid, docs in pools.items():
        row = con.execute(
            "SELECT payload FROM sources WHERE q=? AND source=?", (qid, channel)
        ).fetchone()
        if row is None:
            raise RuntimeError(f"missing public source {channel}/{qid}")
        values = json.loads(row[0])
        native_rank = {str(x["doc_id"]): int(x["rank"]) for x in values}
        scores[qid] = {str(x["doc_id"]): float(x["score"]) for x in values}
        orders[qid] = sorted(docs, key=lambda d: (native_rank.get(d, 10**9), d))
    integrity = con.execute("PRAGMA integrity_check").fetchone()[0]
    con.close()
    if integrity != "ok":
        raise RuntimeError("LegalIR source database integrity failure")
    return orders, scores


def public_jina_channel(pools):
    con = sqlite3.connect(f"file:{PUBLIC_JINA_DB.as_posix()}?mode=ro", uri=True)
    con.execute("PRAGMA query_only=ON")
    scores = defaultdict(dict)
    for qid, doc, score in con.execute("SELECT qid,doc_id,score FROM scores"):
        scores[str(qid)][str(doc)] = float(score)
    integrity = con.execute("PRAGMA integrity_check").fetchone()[0]
    con.close()
    if integrity != "ok" or any(set(scores[q]) != set(docs) for q, docs in pools.items()):
        raise RuntimeError("public frozen Jina score coverage failure")
    orders = {q: sorted(docs, key=lambda d: (-scores[q][d], d)) for q, docs in pools.items()}
    return orders, dict(scores)


def crossfit_train_label_features(folds, pools, questions, golds, dup, vectors, vector_row):
    profile_orders, memory_rows = {}, {}
    all_qids = set(pools)
    for fold, target_ids in folds.items():
        blocked = set(map(str, dup.get(fold, [])))
        support_ids = sorted(all_qids - set(target_ids) - blocked, key=int)
        profile_model = build_profiles(
            {q: (questions[q], golds[q]) for q in pools}, support_ids
        )
        by_doc, frequency = support_index(golds, support_ids)
        support_index_rows = [vector_row[q] for q in support_ids]
        for qid in target_ids:
            profile_orders[qid] = profile_rank(questions[qid], profile_model, 2, 1.2, .75, .3)
            similarities = vectors[vector_row[qid]] @ vectors[support_index_rows].T
            memory_rows[qid] = memory_features(
                similarities, pools[qid], support_ids, golds, by_doc, frequency
            )
    if set(profile_orders) != set(pools) or set(memory_rows) != set(pools):
        raise RuntimeError("cross-fitted label feature population mismatch")
    return profile_orders, memory_rows


def public_label_features(train_pools, train_questions, golds, public_pools, public_questions,
                          train_vectors, vector_row):
    support_ids = sorted(train_pools, key=int)
    query_records = {q: (train_questions[q], golds[q]) for q in support_ids}
    profile_model = build_profiles(query_records, support_ids)
    by_doc, frequency = support_index(golds, support_ids)
    support = train_vectors[[vector_row[q] for q in support_ids]]
    profile_orders, memory_rows = {}, {}
    missing_vectors = []
    for qid in sorted(public_pools, key=int):
        vector_path = PUBLIC_LAL_DIR / f"{qid}.npy"
        if not vector_path.exists():
            missing_vectors.append(qid)
            continue
        vector = normalize(np.load(vector_path).reshape(1, -1))[0]
        profile_orders[qid] = profile_rank(public_questions[qid], profile_model, 2, 1.2, .75, .3)
        memory_rows[qid] = memory_features(
            vector @ support.T, public_pools[qid], support_ids,
            golds, by_doc, frequency,
        )
    if missing_vectors:
        raise RuntimeError(f"missing {len(missing_vectors)} public LAL vectors")
    return profile_orders, memory_rows


def build_features(pools, rank_orders, score_maps, citation, profile_orders, memory_rows):
    rank_orders = dict(rank_orders)
    rank_orders["huy_profile"] = profile_orders
    rank_features = {name: core.rank_columns(order, pools) for name, order in rank_orders.items()}
    score_features = {name: core.score_columns(score, pools) for name, score in score_maps.items()}
    metadata = {"citation": citation, "lal_memory": memory_rows}
    return core.make_rows(CONFIG, pools, rank_features, score_features, metadata)


def main():
    global OUT, CONFIG
    started = time.perf_counter()
    include_huy_jina = False
    if PUBLIC_JINA_DB.exists():
        probe = sqlite3.connect(f"file:{PUBLIC_JINA_DB.as_posix()}?mode=ro", uri=True)
        counts = probe.execute("SELECT COUNT(*),COUNT(DISTINCT qid) FROM scores").fetchone()
        progress = probe.execute("SELECT COUNT(*) FROM progress").fetchone()[0]
        probe.close()
        include_huy_jina = counts == (50_000, 1000) and progress == 1000
    if include_huy_jina:
        OUT = core.OUT / "submission_candidate_with_huy_jina"
        CONFIG = {
            "rank_views": ["jina_ce", *CONFIG["rank_views"]],
            "score_channels": ["jina_ce", *CONFIG["score_channels"]],
            "metadata": list(CONFIG["metadata"]),
        }
    OUT.mkdir(parents=True, exist_ok=False)
    folds, train_pools, train_questions, golds, e5_orders, e5_scores, dup, _ = core.load_inputs()
    lal_order, lal_score, _ = core.load_source_channel("lal", train_pools)
    jina_order, _, _ = core.load_source_channel("jina", train_pools)
    bm25_order, bm25_score, _ = core.load_source_channel("bm25", train_pools)
    trigram_order, trigram_score, _ = core.load_source_channel("trigram", train_pools)
    if include_huy_jina:
        train_huy_jina_order, train_huy_jina_score, _ = core.load_jina(train_pools)
    train_heads = core.document_heads(train_pools)
    _, train_citation = core.metadata_arrays(train_pools, train_questions, train_heads)

    with np.load(TRAIN_LAL, allow_pickle=False) as archive:
        ids = list(map(str, archive["query_ids"].tolist()))
        train_vectors = normalize(archive["vectors"])
    vector_row = {qid: i for i, qid in enumerate(ids)}
    if set(train_pools) - set(vector_row):
        raise RuntimeError("training LAL query vector coverage failure")
    train_profile, train_memory = crossfit_train_label_features(
        folds, train_pools, train_questions, golds, dup, train_vectors, vector_row
    )
    train_rank_orders = {
        "adapted_e5": e5_orders["adapted_e5"], "lal_native": lal_order,
        "legalir_jina": jina_order, "legalir_bm25": bm25_order,
        "legalir_trigram": trigram_order,
    }
    train_score_maps = {
        "adapted_e5": e5_scores["adapted_e5"], "lal_native": lal_score,
        "legalir_bm25": bm25_score, "legalir_trigram": trigram_score,
    }
    if include_huy_jina:
        train_rank_orders["jina_ce"] = train_huy_jina_order
        train_score_maps["jina_ce"] = train_huy_jina_score
    train_rows = build_features(
        train_pools,
        train_rank_orders, train_score_maps,
        train_citation, train_profile, train_memory,
    )

    train_ids = sorted(train_pools, key=int)
    x = np.vstack([train_rows[q] for q in train_ids])
    y = np.concatenate([
        np.asarray([doc in golds[q] for doc in train_pools[q]], dtype=np.int8)
        for q in train_ids
    ])
    scaler = StandardScaler().fit(x)
    model = LogisticRegression(
        C=.15, class_weight="balanced", solver="liblinear", max_iter=3000,
        random_state=2026,
    ).fit(scaler.transform(x), y)

    public_pools, public_questions, public_e5, public_lal, db_integrity = load_public()
    public_jina, _ = source_channel("jina", public_pools)
    public_bm25, public_bm25_score = source_channel("bm25", public_pools)
    public_trigram, public_trigram_score = source_channel("trigram", public_pools)
    if include_huy_jina:
        public_huy_jina_order, public_huy_jina_score = public_jina_channel(public_pools)
    public_lal_order = {
        q: sorted(docs, key=lambda d: (-public_lal[q][d], d))
        for q, docs in public_pools.items()
    }
    public_heads = core.document_heads(public_pools)
    _, public_citation = core.metadata_arrays(public_pools, public_questions, public_heads)
    public_profile, public_memory = public_label_features(
        train_pools, train_questions, golds, public_pools, public_questions,
        train_vectors, vector_row,
    )
    public_rank_orders = {
        "adapted_e5": {q: list(public_pools[q]) for q in public_pools},
        "lal_native": public_lal_order, "legalir_jina": public_jina,
        "legalir_bm25": public_bm25, "legalir_trigram": public_trigram,
    }
    public_score_maps = {
        "adapted_e5": public_e5, "lal_native": public_lal,
        "legalir_bm25": public_bm25_score, "legalir_trigram": public_trigram_score,
    }
    if include_huy_jina:
        public_rank_orders["jina_ce"] = public_huy_jina_order
        public_score_maps["jina_ce"] = public_huy_jina_score
    public_rows = build_features(
        public_pools,
        public_rank_orders, public_score_maps,
        public_citation, public_profile, public_memory,
    )

    predictions = {}
    for qid in sorted(public_pools, key=int):
        values = model.decision_function(scaler.transform(public_rows[qid]))
        index = np.lexsort((np.asarray(public_pools[qid]), -values))
        predictions[qid] = [public_pools[qid][i] for i in index[:5]]

    canonical = {
        str(row["doc_id"])
        for row in core.read_jsonl(CANONICAL_CONTEXTS)
    }
    valid = (
        len(predictions) == 1000
        and set(predictions) == set(public_questions)
        and all(len(v) == 5 and len(set(v)) == 5 and set(v) <= canonical for v in predictions.values())
    )
    if not valid:
        raise RuntimeError("public prediction structural validation failed")

    prediction_path = OUT / "PUBLIC_PREDICTIONS.jsonl"
    with prediction_path.open("w", encoding="utf-8", newline="\n") as handle:
        for qid in sorted(predictions, key=int):
            handle.write(json.dumps({"qid": qid, "top5": predictions[qid]}, ensure_ascii=False, separators=(",", ":")) + "\n")
    submission = {qid: {"answer": docs} for qid, docs in predictions.items()}
    submission_path = OUT / "submission.json"
    submission_path.write_text(json.dumps(submission, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    zip_path = OUT / "submission.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        archive.write(submission_path, arcname="submission.json")

    evidence_report = json.loads((core.OUT / "HUY_LAL_MEMORY_PORT_REPORT.json").read_text(encoding="utf-8"))
    evidence_name = "profile_memory_plus_sparse_rank_scores" if include_huy_jina else "sparse_winner_no_huy_jina_ce"
    evidence = next(
        item for item in evidence_report["results"]
        if item["name"] == evidence_name
    )
    report = {
        "schema_version": "dsc2026.huy_fasttrack.public_candidate.v1",
        "status": "VALID_LOCAL_CANDIDATE_NOT_UPLOADED",
        "strict_oof_support": evidence["metrics"],
        "config": CONFIG,
        "training": "single full-data LR fit on 5-way cross-fitted profile and LAL-memory features",
        "public_validation": {
            "queries": 1000, "answers_per_query": 5, "all_unique": True,
            "all_canonical": True, "canonical_parents": len(canonical),
            "public_score_db_integrity": db_integrity,
        },
        "system": {
            "neural_models": (["jinaai/jina-reranker-v2-base-multilingual"] if include_huy_jina else []) + [
                "mainguyen9/vietlegal-e5 (adapted query encoder)",
                "darklethelong/vnlegal-lal",
                "jinaai/jina-embeddings-v3",
            ],
            "approx_original_parameters": 2_005_738_368 if include_huy_jina else 1_727_694_720,
            "under_4b": True, "augmentation": False, "adaptive_k": False,
            "candidate_depth": 50,
        },
        "runtime_seconds": time.perf_counter() - started,
        "files": {
            "submission_json": str(submission_path),
            "submission_zip": str(zip_path),
            "predictions": str(prediction_path),
        },
    }
    report_path = OUT / "REPORT.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    input_files = [
        core.FOLDS_PATH, core.POOL_PATH, PUBLIC_JSON, PUBLIC_SCORE_DB,
        TRAIN_LAL, CANONICAL_CONTEXTS, core.SOURCE_DB,
        core.OUT / "HUY_LAL_MEMORY_PORT_REPORT.json", Path(__file__),
        *[path for _, path in core.adapted_paths()],
    ]
    if include_huy_jina:
        input_files.extend([PUBLIC_JINA_DB, core.JINA_DB])
    manifest = {
        "schema_version": "dsc2026.huy_fasttrack.public_candidate_manifest.v1",
        "reproduce": "D:\\Study\\DSC2026\\dsc_env\\Scripts\\python.exe D:\\Study\\DSC2026\\sota\\src\\huy_fasttrack\\materialize_public_candidate.py",
        "inputs": {
            **{str(p): sha256(p) for p in input_files},
            str(PUBLIC_LAL_DIR) + "/**": tree_sha256(PUBLIC_LAL_DIR),
        },
        "outputs": {str(p): sha256(p) for p in (
            prediction_path, submission_path, zip_path, report_path,
        )},
    }
    manifest_path = OUT / "MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "oof": evidence["metrics"], "zip": str(zip_path), "sha256": sha256(zip_path)}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
