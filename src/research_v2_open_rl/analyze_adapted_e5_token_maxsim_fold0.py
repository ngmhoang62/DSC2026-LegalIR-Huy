"""Cache-only causal forensic for the sealed adapted-E5 token-MaxSim pilot."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

import legal_mlm_query_likelihood_fold0 as common


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    output = root / "results/research_v2_open_rl"
    predictions_path = output / "ADAPTED_E5_TOKEN_MAXSIM_FOLD0_PREDICTIONS.jsonl"
    report_path = output / "ADAPTED_E5_TOKEN_MAXSIM_FOLD0_REPORT.json"
    anchor_path = root / "results/research_v2_post_e5/V2_ADAPTED_E5_LAL_EQUAL_RRF32_PREDICTIONS.jsonl"
    e5_path = root / "results/research_v2_e5_confirmation/fold0_runner_parity/E5_CONFIRMATION_FOLD_0_PREDICTIONS.jsonl"
    jina_path = root / "results/research_v2_forensic/V2_ZERO_SHOT_LEXICAL_PREDICTIONS.jsonl"

    predictions = {str(row["qid"]): row for row in common.read_jsonl(predictions_path)}
    anchor = {str(row["qid"]): row for row in common.read_jsonl(anchor_path) if str(row["qid"]) in predictions}
    e5 = {str(row["qid"]): row for row in common.read_jsonl(e5_path) if str(row["qid"]) in predictions}
    jina = {str(row["qid"]): list(map(str, row["top5"])) for row in common.read_jsonl(jina_path) if str(row["qid"]) in predictions}
    if not (len(predictions) == len(anchor) == len(e5) == len(jina) == 1398):
        raise RuntimeError("qid/cardinality mismatch")

    # Reuse exactly the evaluator's clean-expert union construction.
    class Args:
        pass
    args = Args()
    args.anchor = anchor_path
    args.e5_predictions = e5_path
    args.jina_predictions = jina_path
    args.sources_db = root.parent / "LegalIR/cache/exp112_task_adaptive_retrieval/sources.sqlite"
    pool = {qid: set(map(str, row["ranking"])) for qid, row in predictions.items()}
    clean = common.load_clean_sets(args, pool)

    top5_overlap_e5 = []
    top5_overlap_jina = []
    gold_rank_delta_vs_e5 = []
    clean_missed = 0
    expert_gold_hits = 0
    redundant_hits = 0
    exclusive_hits = 0
    exclusive_units = 0.0
    exclusive_queries = set()
    examples = []
    for qid, row in predictions.items():
        ranking = list(map(str, row["ranking"]))
        top5 = ranking[:5]
        e5_order = list(map(str, e5[qid]["ft_order"]))
        e5_top5 = e5_order[:5]
        top5_overlap_e5.append(len(set(top5) & set(e5_top5)))
        top5_overlap_jina.append(len(set(top5) & set(jina[qid])))
        gold = set(map(str, anchor[qid]["gold"]))
        for doc in gold:
            if doc not in clean[qid]:
                clean_missed += 1
            if doc in top5:
                expert_gold_hits += 1
                if doc in clean[qid]:
                    redundant_hits += 1
                else:
                    exclusive_hits += 1
                    exclusive_units += 1.0 / len(gold)
                    exclusive_queries.add(qid)
                    if len(examples) < 20:
                        examples.append([qid, doc])
            if doc in ranking and doc in e5_order:
                gold_rank_delta_vs_e5.append(e5_order.index(doc) - ranking.index(doc))

    official = json.loads(report_path.read_text(encoding="utf-8"))
    forensic = {
        "schema_version": "dsc2026.research_v2.adapted_e5_token_maxsim_fold0_causal_forensic.v1",
        "status": "COMPLETE_CACHE_ONLY",
        "official_verdict": official["verdict"],
        "official_metrics": official["metrics"],
        "diagnostic_only_not_candidate": {
            "clean_missed_gold_occurrences": clean_missed,
            "expert_top5_gold_hits": expert_gold_hits,
            "expert_top5_gold_hits_already_in_clean_union": redundant_hits,
            "exclusive_rescued_gold_occurrences": exclusive_hits,
            "exclusive_rescue_query_recall_units": exclusive_units,
            "queries_with_exclusive_rescue": len(exclusive_queries),
            "exclusive_examples_qid_doc": examples,
            "mean_top5_intersection_with_adapted_e5": float(np.mean(top5_overlap_e5)),
            "mean_top5_intersection_with_frozen_jina_colbert": float(np.mean(top5_overlap_jina)),
            "median_gold_rank_improvement_vs_mean_pooled_adapted_e5": float(np.median(gold_rank_delta_vs_e5)),
            "mean_gold_rank_improvement_vs_mean_pooled_adapted_e5": float(np.mean(gold_rank_delta_vs_e5)),
        },
        "causal_read": [
            "The diagnostic distinguishes a useful adapted-E5 late-interaction geometry from a broad rank reshuffle that merely rediscovers gold already known by the clean experts.",
            "No alternate evidence count, length, token filter, pooling, direction, score transform, fusion, threshold, or routing decision is evaluated here.",
        ],
        "closed_scope": "Strict Fold-0 adapted-E5 query-token to frozen-E5 lexical-top1 evidence-token mean-MaxSim, direct deterministic Top-5.",
        "hashes": {"report": sha256(report_path), "predictions": sha256(predictions_path)},
    }
    target = output / "ADAPTED_E5_TOKEN_MAXSIM_FOLD0_CAUSAL_FORENSIC.json"
    target.write_text(json.dumps(forensic, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(forensic["diagnostic_only_not_candidate"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
