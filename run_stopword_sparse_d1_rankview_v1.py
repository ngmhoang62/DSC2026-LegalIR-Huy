#!/usr/bin/env python
"""
HUY DEADLINE AUDIT — STOPWORD-SPARSE AS 6TH D1 RANK VIEW V1
============================================================

CPU-only downstream test of the strongest remaining finding:

Sparse audit:
  CURRENT_ALL_UNIQUE FUSED R@20 = 0.746667
  DROP_STOPWORDS     FUSED R@20 = 0.831667  (+0.085)
  improvement is positive in ALL 4 CAL blocks.

This script asks the only question that matters downstream:
  Does the improved sparse retrieval help the EXISTING D1 candidate pool rank
  gold into Top-5 when used as one additional rank view?

Important:
- candidate pool is EXACTLY unchanged (current cap32 D1 pool);
- no sparse candidate expansion yet;
- no new score channel;
- current 48D D1 remains exact control;
- sparse adds only its rank geometry, producing 50D;
- compares old tokenization vs DROP_STOPWORDS to isolate the tokenizer fix.

No public labels.
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
import sqlite3
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


EXPECTED_R = 0.9569444444444444
EXPECTED_P = 0.20566666666666666
D1_VIEWS = ["base", "expanded", "jina", "dense", "corpus"]
TOKEN_RE = re.compile(r"\w+", re.UNICODE)
STOPWORDS = {
    "bị","các","có","của","cho","được","để","đến","đối","gì","hay","khi","không",
    "là","làm","một","nào","những","như","phải","ra","sẽ","theo","thì","thế",
    "trong","trên","từ","và","về","với","việc","bao","nhiêu","người","quy","định",
}
EXTRA = {
    "aiteamvn_ft": "results/from_drive/aiteamvn_ft_cv.pkl",
    "jina_ft": "results/from_drive/jina_ft_cv.pkl",
    "title_embed": "results/burst_fresh_block/title_embed_scores.pkl",
}


def loadp(root, rel):
    obj = pickle.loads((root / rel).read_bytes())
    if isinstance(obj, dict) and isinstance(obj.get("scores"), dict):
        return obj["scores"]
    return obj


def align(raw, candidates, ids, floor=None):
    if isinstance(raw, dict) and isinstance(raw.get("scores"), dict):
        raw = raw["scores"]
    if floor is None:
        vals = [v for q in raw.values() for v in q.values()]
        floor = min(vals) if vals else -1e9
    return {
        q: {d: float(raw.get(q, {}).get(d, floor)) for d in candidates[q]}
        for q in ids
    }


def make_terms(text, drop_stopwords):
    toks = TOKEN_RE.findall((text or "").lower())
    seen, out = set(), []
    for t in toks:
        if drop_stopwords and t in STOPWORDS:
            continue
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def expr(terms):
    return " OR ".join('"' + t.replace('"', '""') + '"' for t in terms)


def locate_fts_db(root):
    candidates = [
        root / "benchmarks/legalir_full_fts.sqlite",
        root.parent / "LegalIR/benchmarks/legalir_full_fts.sqlite",
        root / "results/manual/huy_sparse_fts_query_tokenization_v2/legalir_full_fts_audit.sqlite",
    ]
    for p in candidates:
        if p.is_file():
            return p
    raise FileNotFoundError(
        "No FTS DB found. Let run_sparse_fts_query_tokenization_v2_autobuild "
        "finish first."
    )


def prepare_world(root):
    from run_burst_expanded_fusion_submission import DocumentStore
    from tune_citation_graph import build_citation_table, citation_features
    from tune_corpus_cap32_fusion import build_training_cap
    from tune_doctype_features import build_type_table, type_features

    docs = DocumentStore(sorted(
        (
            root /
            "DSC2026-LegalIR-main/v4_run/public_test_dataset/selected-contexts"
        ).glob("context_*.json")
    ))

    queries, blocks, ids, candidates, views, base_scores = build_training_cap(
        root, 32,
        "results/corpus_index/holdout_extended_scores_cap32.pkl",
        depth=20,
    )
    gold = {q: set(map(str, queries[q][1])) for q in ids}

    channels = {
        **base_scores,
        "vnlegal_lal": align(
            loadp(root, "results/embedding_finetune/vnlegal_lal_cv_scores.pkl"),
            candidates, ids,
        ),
        "crossenc": align(
            loadp(root, "results/crossenc_fullpool/cv_scores.pkl"),
            candidates, ids, -11.5,
        ),
        **{
            k: align(loadp(root, rel), candidates, ids)
            for k, rel in EXTRA.items()
        },
    }

    tt = build_type_table(root, docs, ids, candidates)
    tr = type_features(candidates, tt, queries, ids)
    own, cited = build_citation_table(docs, ids, candidates)
    cr = citation_features(candidates, own, cited, ids)

    return queries, blocks, ids, candidates, views, channels, gold, tr, cr


def build_sparse_rankings(root, world):
    from benchmark_burst_v4_full_sqlite import (
        retrieve_docs, retrieve_local, fuse, load_dataset
    )

    queries, blocks, ids, candidates, views, channels, gold, tr, cr = world
    db = locate_fts_db(root)
    data_dir = root / "DSC2026-LegalIR-main/v4_run/public_test_dataset"
    docs_all, _ = load_dataset(data_dir)
    doc_ids = [str(d) for d, _ in docs_all]

    conn = sqlite3.connect(f"file:{db.resolve().as_posix()}?mode=ro", uri=True)
    ranks = {"CURRENT_ALL_UNIQUE": {}, "DROP_STOPWORDS": {}}

    try:
        for i, q in enumerate(ids, 1):
            text = queries[q][0]
            for name, drop in (
                ("CURRENT_ALL_UNIQUE", False),
                ("DROP_STOPWORDS", True),
            ):
                terms = make_terms(text, drop)
                e = expr(terms)
                full = retrieve_docs(conn, e, 500)
                local = retrieve_local(conn, e, 2000, second_weight=.3)
                fused = fuse(full, local, local_weight=.9, rrf_k=20)
                # Keep full fused ranking as a rank map source. Only current D1
                # pool docs will receive features downstream.
                ranks[name][q] = [doc_ids[d] for d in fused]
            if i % 50 == 0 or i == len(ids):
                print(f"  sparse ranking {i}/{len(ids)}", flush=True)
    finally:
        conn.close()

    return ranks, db


def run_lobo(world, sparse_rank=None):
    from tune_expanded_fusion_selection import ltr_features

    queries, blocks, ids, candidates, base_views, channels, gold, tr, cr = world

    if sparse_rank is None:
        names = list(D1_VIEWS)
        views = dict(base_views)
    else:
        names = D1_VIEWS + ["sparse_sw"]
        views = dict(base_views)
        views["sparse_sw"] = sparse_rank

    rows, groups = ltr_features(
        views, names, candidates, ids, channels
    )
    for q in ids:
        rows[q] = np.concatenate([rows[q], tr[q], cr[q]], axis=1)

    pred, perq = {}, {}
    for held in sorted(blocks):
        train = sum((blocks[b] for b in blocks if b != held), [])
        X = np.vstack([rows[q] for q in train])
        y = np.concatenate([
            [d in gold[q] for d in groups[q]]
            for q in train
        ]).astype(np.int8)

        sc = StandardScaler().fit(X)
        model = LogisticRegression(
            C=.15,
            class_weight="balanced",
            solver="liblinear",
            max_iter=3000,
            random_state=2026,
        )
        model.fit(sc.transform(X), y)

        for q in blocks[held]:
            s = model.decision_function(sc.transform(rows[q]))
            order = np.argsort(-s, kind="stable")
            pred[q] = [groups[q][i] for i in order[:5]]

    for q in ids:
        perq[q] = len(set(pred[q]) & gold[q]) / len(gold[q])

    return {
        "dim": int(rows[ids[0]].shape[1]),
        "recall": float(np.mean([perq[q] for q in ids])),
        "precision": float(np.mean([
            len(set(pred[q]) & gold[q]) / 5.0 for q in ids
        ])),
        "blocks": {
            b: float(np.mean([perq[q] for q in blocks[b]]))
            for b in sorted(blocks)
        },
        "single": float(np.mean([
            perq[q] for q in ids if len(gold[q]) == 1
        ])),
        "multi": float(np.mean([
            perq[q] for q in ids if len(gold[q]) > 1
        ])),
        "pred": pred,
        "perq": perq,
    }


def compare(base, cand, ids):
    d = np.asarray([cand["perq"][q] - base["perq"][q] for q in ids])
    return {
        "delta_recall": cand["recall"] - base["recall"],
        "delta_precision": cand["precision"] - base["precision"],
        "delta_single": cand["single"] - base["single"],
        "delta_multi": cand["multi"] - base["multi"],
        "wins": int(np.sum(d > 1e-12)),
        "losses": int(np.sum(d < -1e-12)),
        "ties": int(np.sum(np.abs(d) <= 1e-12)),
        "block_deltas": {
            b: cand["blocks"][b] - base["blocks"][b]
            for b in base["blocks"]
        },
        "top5_exact_matches": int(sum(
            cand["pred"][q] == base["pred"][q] for q in ids
        )),
    }


def slim(r):
    return {k: v for k, v in r.items() if k not in ("pred", "perq")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=Path, required=True)
    args = ap.parse_args()
    root = args.repo_root.resolve()
    sys.path.insert(0, str(root))

    print("[1/4] Loading exact D1 world...", flush=True)
    world = prepare_world(root)
    ids = world[2]

    print("[2/4] Rebuilding current vs stopword-filtered sparse rankings...", flush=True)
    sparse, db = build_sparse_rankings(root, world)
    print("  FTS DB:", db, flush=True)

    print("[3/4] Exact LOBO: D1 / +old-sparse / +stopword-sparse...", flush=True)
    base = run_lobo(world, None)
    if (
        abs(base["recall"] - EXPECTED_R) > 1e-12
        or abs(base["precision"] - EXPECTED_P) > 1e-12
        or base["dim"] != 48
    ):
        raise RuntimeError(
            f"D1 parity failed: dim={base['dim']} "
            f"R={base['recall']} P={base['precision']}"
        )

    old = run_lobo(world, sparse["CURRENT_ALL_UNIQUE"])
    sw = run_lobo(world, sparse["DROP_STOPWORDS"])

    cold = compare(base, old, ids)
    csw = compare(base, sw, ids)

    strict = (
        csw["delta_recall"] > 1e-12
        and all(v >= -1e-12 for v in csw["block_deltas"].values())
        and csw["wins"] > csw["losses"]
    )

    report = {
        "schema": "manual.stopword_sparse_d1_rankview_v1",
        "control": slim(base),
        "old_sparse_rankview": {
            **slim(old),
            "comparison": cold,
        },
        "stopword_sparse_rankview": {
            **slim(sw),
            "comparison": csw,
        },
        "policy": {
            "candidate_pool_changed": False,
            "new_score_channel": False,
            "new_rank_view": "FULL+LOCAL FTS RRF after DROP_STOPWORDS",
            "LR_C": .15,
            "public_labels_used": False,
        },
        "verdict": (
            "PROMOTE_STOPWORD_SPARSE_RANKVIEW"
            if strict else "KILL_STOPWORD_SPARSE_RANKVIEW"
        ),
    }

    out = root / "results/manual/huy_stopword_sparse_d1_rankview_v1"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "REPORT.json"
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("[4/4] RESULT")
    print("=" * 116)
    print(
        f"D1           dim={base['dim']} R={base['recall']:.10f} "
        f"P={base['precision']:.10f}"
    )
    for name, r, c in (
        ("OLD_SPARSE", old, cold),
        ("STOPWORD_SW", sw, csw),
    ):
        print(
            f"{name:<13s} dim={r['dim']} "
            f"R={r['recall']:.10f} dR={c['delta_recall']:+.10f} "
            f"P={r['precision']:.10f} dP={c['delta_precision']:+.10f} "
            f"W/L/T={c['wins']}/{c['losses']}/{c['ties']} "
            f"blocks={c['block_deltas']}"
        )
    print("VERDICT:", report["verdict"])
    print("Report:", path)
    print("=" * 116)


if __name__ == "__main__":
    main()
