"""Validate reconstructed local Jina-FT inference against Huy's immutable cache."""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark_jina_reranker_holdouts import top_passages
from jina_ft_local import load_model_and_tokenizer, score_pairs
from run_burst_expanded_fusion_submission import DocumentStore
from tune_corpus_cap32_fusion import build_training_cap


def main() -> None:
    queries, _, all_ids, extended, _, _ = build_training_cap(
        ROOT,
        32,
        "results/corpus_index/holdout_extended_scores_cap32.pkl",
        depth=20,
    )
    cached = pickle.loads((ROOT / "results/from_drive/jina_ft_cv.pkl").read_bytes())
    docs = DocumentStore(
        sorted(
            (
                ROOT
                / "DSC2026-LegalIR-main/v4_run/public_test_dataset/selected-contexts"
            ).glob("context_*.json")
        )
    )
    # Deterministic coverage across all four ordered blocks, with candidates at
    # head, boundary and tail. No labels are consulted.
    q_indices = [0, 99, 100, 199, 200, 299, 300, 399, 499, 599]
    requests = []
    pair_owners = []
    for qi in q_indices:
        qid = all_ids[qi]
        candidates = extended[qid]
        for di in (0, min(4, len(candidates) - 1), len(candidates) - 1):
            docid = candidates[di]
            passages = top_passages(queries[qid][0], docs[docid], count=4)
            for passage_index, passage in enumerate(passages, 1):
                requests.append((queries[qid][0], passage))
                pair_owners.append((qid, docid, passage_index))

    model, tokenizer = load_model_and_tokenizer()
    raw = score_pairs(model, tokenizer, requests, batch_size=8)
    rebuilt = {}
    for (qid, docid, passage_index), score in zip(pair_owners, raw):
        rebuilt.setdefault((qid, docid), []).append((passage_index, float(score)))

    rows = []
    for (qid, docid), passage_scores in rebuilt.items():
        reference = float(cached[qid][docid])
        maxima = {
            str(k): max(s for i, s in passage_scores if i <= k)
            for k in range(1, len(passage_scores) + 1)
        }
        score = maxima.get("2", maxima["1"])
        rows.append(
            {
                "qid": qid,
                "docid": docid,
                "reference": reference,
                "rebuilt": score,
                "abs_diff": abs(reference - score),
                "max_by_passage_count": maxima,
            }
        )
    count_errors = {
        str(k): [
            abs(r["reference"] - r["max_by_passage_count"].get(str(k), r["max_by_passage_count"][str(len(r["max_by_passage_count"]))]))
            for r in rows
        ]
        for k in range(1, 5)
    }
    result = {
        "pairs": len(rows),
        "max_abs_diff": max(r["abs_diff"] for r in rows),
        "mean_abs_diff": sum(r["abs_diff"] for r in rows) / len(rows),
        "within_1e_3": sum(r["abs_diff"] <= 1e-3 for r in rows),
        "within_1e_2": sum(r["abs_diff"] <= 1e-2 for r in rows),
        "mean_abs_diff_by_passage_count": {
            k: sum(v) / len(v) for k, v in count_errors.items()
        },
        "max_abs_diff_by_passage_count": {k: max(v) for k, v in count_errors.items()},
        "rows": rows,
    }
    out = ROOT / "results/sol_high_rl/jina_ft_local_parity.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))
    print(out)


if __name__ == "__main__":
    main()
