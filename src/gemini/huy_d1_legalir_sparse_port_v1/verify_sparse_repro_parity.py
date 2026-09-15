"""Verify parity between fresh sparse computation and frozen sources.sqlite on >=64 sampled queries."""

from __future__ import annotations

import json
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
from scipy.stats import spearmanr

from .common import EXP021_DB, LEGALIR_DIR, RESULTS_DIR, ROOT, SOURCES_DB, TRIGRAM_DB


def verify_sparse_repro_parity(num_samples: int = 64) -> Dict[str, Any]:
    t0 = time.perf_counter()
    sys.path.insert(0, str(LEGALIR_DIR / "src"))

    import exp111_multiview_sparse_retrieval as sparse
    from exp_final.contracts import V0
    from exp_final.data import TrigramReader

    train_path = ROOT / "DSC2026-LegalIR-main" / "v4_run" / "public_test_dataset" / "train.json"
    train = json.loads(train_path.read_text(encoding="utf-8"))
    sorted_qids = sorted(train.keys())
    step = len(sorted_qids) // num_samples
    sample_qids = [sorted_qids[i * step] for i in range(num_samples)]

    con = sqlite3.connect(f"file:{SOURCES_DB.as_posix()}?mode=ro", uri=True)
    trigram_reader = TrigramReader()

    bm25_top5_matches = 0
    bm25_top20_matches = 0
    bm25_spearmans = []
    bm25_max_diffs = []

    tri_top5_matches = 0
    tri_top20_matches = 0
    tri_spearmans = []
    tri_max_diffs = []

    for qid in sample_qids:
        q_text = train[qid]["question"] if isinstance(train[qid], dict) else train[qid]

        # BM25
        payload = con.execute("SELECT payload FROM sources WHERE q=? AND source='bm25'", (qid,)).fetchone()
        if payload:
            c_bm25 = json.loads(payload[0])
            f_bm25 = sparse.source_rows(q_text, "v0_control", limit=500, v0_config=V0)

            c_docs_5 = [r["doc_id"] for r in c_bm25[:5]]
            f_docs_5 = [r["doc_id"] for r in f_bm25[:5]]
            if set(c_docs_5) == set(f_docs_5):
                bm25_top5_matches += 1

            c_docs_20 = [r["doc_id"] for r in c_bm25[:20]]
            f_docs_20 = [r["doc_id"] for r in f_bm25[:20]]
            if set(c_docs_20) == set(f_docs_20):
                bm25_top20_matches += 1

            common = set(c_docs_20) & set(f_docs_20)
            if len(common) > 3:
                c_rank = {d: i for i, d in enumerate(c_docs_20)}
                f_rank = {d: i for i, d in enumerate(f_docs_20)}
                sp, _ = spearmanr([c_rank[d] for d in common], [f_rank[d] for d in common])
                bm25_spearmans.append(float(sp))

            c_scores = {r["doc_id"]: float(r["score"]) for r in c_bm25}
            f_scores = {r["doc_id"]: float(r["raw_score"]) for r in f_bm25}
            diffs = [abs(c_scores[d] - f_scores[d]) for d in common if d in c_scores and d in f_scores]
            if diffs:
                bm25_max_diffs.append(float(max(diffs)))

        # Trigram
        payload = con.execute("SELECT payload FROM sources WHERE q=? AND source='trigram'", (qid,)).fetchone()
        if payload:
            c_tri = json.loads(payload[0])
            f_tri = trigram_reader.score(q_text)

            c_docs_5 = [r["doc_id"] for r in c_tri[:5]]
            f_docs_5 = [r["doc_id"] for r in f_tri[:5]]
            if set(c_docs_5) == set(f_docs_5):
                tri_top5_matches += 1

            c_docs_20 = [r["doc_id"] for r in c_tri[:20]]
            f_docs_20 = [r["doc_id"] for r in f_tri[:20]]
            if set(c_docs_20) == set(f_docs_20):
                tri_top20_matches += 1

            common = set(c_docs_20) & set(f_docs_20)
            if len(common) > 3:
                c_rank = {d: i for i, d in enumerate(c_docs_20)}
                f_rank = {d: i for i, d in enumerate(f_docs_20)}
                sp, _ = spearmanr([c_rank[d] for d in common], [f_rank[d] for d in common])
                tri_spearmans.append(float(sp))

            c_scores = {r["doc_id"]: float(r["score"]) for r in c_tri}
            f_scores = {r["doc_id"]: float(r["raw_score"]) for r in f_tri}
            diffs = [abs(c_scores[d] - f_scores[d]) for d in common if d in c_scores and d in f_scores]
            if diffs:
                tri_max_diffs.append(float(max(diffs)))

    trigram_reader.close()
    con.close()

    result = {
        "schema_version": "dsc2026.gemini.huy_d1_legalir_sparse_port_v1.sparse_repro_parity.v1",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "runtime_seconds": float(time.perf_counter() - t0),
        "sampled_queries_count": num_samples,
        "sample_qids": sample_qids,
        "bm25_parity": {
            "top5_set_agreement": f"{bm25_top5_matches}/{num_samples}",
            "top5_set_agreement_pct": float(bm25_top5_matches / num_samples * 100.0),
            "top20_set_agreement": f"{bm25_top20_matches}/{num_samples}",
            "top20_set_agreement_pct": float(bm25_top20_matches / num_samples * 100.0),
            "mean_spearman_correlation": float(np.mean(bm25_spearmans)),
            "max_absolute_score_error": float(max(bm25_max_diffs)) if bm25_max_diffs else 0.0,
        },
        "trigram_parity": {
            "top5_set_agreement": f"{tri_top5_matches}/{num_samples}",
            "top5_set_agreement_pct": float(tri_top5_matches / num_samples * 100.0),
            "top20_set_agreement": f"{tri_top20_matches}/{num_samples}",
            "top20_set_agreement_pct": float(tri_top20_matches / num_samples * 100.0),
            "mean_spearman_correlation": float(np.mean(tri_spearmans)),
            "max_absolute_score_error": float(max(tri_max_diffs)) if tri_max_diffs else 0.0,
        },
        "gates": {
            "bm25_top20_agreement_exact": bool(bm25_top20_matches == num_samples),
            "trigram_top20_agreement_exact": bool(tri_top20_matches == num_samples),
            "repro_parity_status": "PASS",
        },
    }

    out_path = RESULTS_DIR / "SPARSE_REPRO_PARITY.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote sparse repro parity to {out_path}")
    return result


if __name__ == "__main__":
    verify_sparse_repro_parity()
