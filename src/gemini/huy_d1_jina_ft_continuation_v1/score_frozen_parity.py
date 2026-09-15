"""Score sample of CAL queries with unadapted shipped Jina model and assert cache parity."""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path
import numpy as np
import scipy.stats as stats
import torch

ROOT = Path("D:/Study/DSC2026/sota")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from run_burst_expanded_fusion_submission import DocumentStore
from tune_corpus_cap32_fusion import build_training_cap
import src.gemini.huy_d1_jina_ft_continuation_v1.common as common

OUT_PATH = ROOT / "results/gemini/huy_d1_jina_ft_continuation_v1/FROZEN_JINA_SCORE_PARITY.json"
CACHED_JINA_CV = ROOT / "results/from_drive/jina_ft_cv.pkl"


def run_parity(n_queries: int = 64) -> dict:
    common.seed_everything(2026)
    print(f"Running frozen Jina score parity check on {n_queries} CAL queries...", flush=True)

    docs = DocumentStore(
        sorted(
            (
                ROOT
                / "DSC2026-LegalIR-main/v4_run/public_test_dataset/selected-contexts"
            ).glob("context_*.json")
        )
    )
    queries, blocks, all_ids, extended, local, _ = build_training_cap(
        ROOT, 32, "results/corpus_index/holdout_extended_scores_cap32.pkl", depth=20
    )

    with open(CACHED_JINA_CV, "rb") as f:
        cached_scores = pickle.load(f)

    model, tok = common.load_jina_base_with_shipped_weights()
    model._tokenizer = tok
    model.eval().to("cuda")

    sample_qids = [str(q) for q in all_ids[:n_queries]]
    max_score_diff = 0.0
    spearman_corrs = []
    top5_agreements = []
    top20_agreements = []

    for q in sample_qids:
        text = queries[q][0]
        owners, passages = [], []
        for d in extended[q]:
            for p in common.top_passages(text, docs[d], count=2):
                owners.append(d)
                passages.append(p)

        raw = model.compute_score(
            [(text, p) for p in passages], batch_size=32, max_length=512
        )
        if isinstance(raw, float):
            raw = [raw]

        fresh_scores = {}
        for d, s in zip(owners, raw):
            fresh_scores[d] = max(fresh_scores.get(d, -1e9), float(s))

        # Compare with cached
        c_scores = cached_scores[q]
        cands = list(extended[q])
        f_vec = [fresh_scores[d] for d in cands]
        c_vec = [c_scores[d] for d in cands]

        diff = max(abs(f - c) for f, c in zip(f_vec, c_vec))
        max_score_diff = max(max_score_diff, diff)

        rho, _ = stats.spearmanr(f_vec, c_vec)
        if np.isfinite(rho):
            spearman_corrs.append(float(rho))

        # Orderings
        f_order = sorted(cands, key=lambda d: fresh_scores[d], reverse=True)
        c_order = sorted(cands, key=lambda d: c_scores[d], reverse=True)

        top5_agreements.append(set(f_order[:5]) == set(c_order[:5]))
        k20 = min(20, len(cands))
        top20_agreements.append(set(f_order[:k20]) == set(c_order[:k20]))

    mean_top5 = float(np.mean(top5_agreements))
    mean_top20 = float(np.mean(top20_agreements))
    mean_spearman = float(np.mean(spearman_corrs)) if spearman_corrs else 1.0

    print(
        f"Parity on {n_queries} queries: max_diff={max_score_diff:.8f}, "
        f"mean_spearman={mean_spearman:.6f}, top5_agree={mean_top5:.2%}, top20_agree={mean_top20:.2%}",
        flush=True,
    )

    passed = (mean_top20 == 1.0) and (max_score_diff < 1e-4)

    report = {
        "experiment_id": "HUY_D1_JINA_FT_CONTINUATION_V1",
        "sample_query_count": n_queries,
        "max_score_difference": max_score_diff,
        "mean_spearman_rank_correlation": mean_spearman,
        "top5_set_agreement_fraction": mean_top5,
        "top20_set_agreement_fraction": mean_top20,
        "required_top20_agreement": 1.0,
        "status": "PASS" if passed else "BLOCKED_FROZEN_SCORE_PARITY",
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Wrote {OUT_PATH}", flush=True)
    return report


if __name__ == "__main__":
    run_parity()
