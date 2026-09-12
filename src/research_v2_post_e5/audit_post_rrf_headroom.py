"""Cache-only incremental headroom after the first successful V2 integration."""

from __future__ import annotations

import json
from pathlib import Path

from research_v2_post_e5 import v2_complementarity_audit as audit


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/research_v2_post_e5"


def main() -> None:
    adapted = audit.load_adapted()
    qids = sorted(adapted, key=int)
    pools = {
        str(row["qid"]): list(map(str, row["doc_ids"]))
        for row in audit.read_jsonl(audit.BUNDLE / "V2_CANDIDATE_POOL.jsonl")
    }
    golds = {qid: set(map(str, adapted[qid]["gold"])) for qid in qids}
    source_predictions, source_gold_ranks, source_provenance = audit.source_views(
        qids, pools, golds
    )
    rrf_rows = {
        str(row["qid"]): row
        for row in audit.read_jsonl(OUT / "V2_ADAPTED_E5_LAL_EQUAL_RRF32_PREDICTIONS.jsonl")
    }
    if set(rrf_rows) != set(adapted):
        raise RuntimeError("RRF/reference population mismatch")
    reference = {
        qid: {
            "qid": qid,
            "fold": rrf_rows[qid]["fold"],
            "gold": rrf_rows[qid]["gold"],
            "ft_order": rrf_rows[qid]["fused_top5"],
        }
        for qid in qids
    }
    experts = {
        "adapted_e5_oof": {qid: adapted[qid]["ft_order"][:5] for qid in qids},
        "frozen_e5_exact_pool": {qid: adapted[qid]["base_order"][:5] for qid in qids},
        "jina_v2_lexical_pool": audit.top5_map(audit.FORENSIC / "V2_ZERO_SHOT_LEXICAL_PREDICTIONS.jsonl"),
        "jina_colbert_pool": source_predictions["jina_pool"],
    }
    analyses = {
        name: audit.analyze(name, predictions, reference, pools, source_gold_ranks)
        for name, predictions in experts.items()
    }
    report = {
        "schema_version": "dsc2026.research_v2.post_rrf_incremental_headroom.v1",
        "status": "COMPLETE_CACHE_ONLY_NO_FIT",
        "reference": "V2_ADAPTED_E5_LAL_EQUAL_RRF32",
        "reference_recall_at_5": 0.917954036141706,
        "experts": analyses,
        "source_database_sha256": source_provenance["database_sha256"],
        "rrf_predictions_sha256": audit.sha256(OUT / "V2_ADAPTED_E5_LAL_EQUAL_RRF32_PREDICTIONS.jsonl"),
        "interpretation": "Diagnostic Top-5 set unions only; no additional integration is executed by this audit.",
    }
    path = OUT / "V2_POST_RRF_COMPLEMENTARITY_ANATOMY.json"
    audit.write_json(path, report)
    print(json.dumps({
        "output": str(path),
        "experts": {
            name: {
                "standalone": value["standalone_recall_at_5"],
                "union_increment": value["union_increment_over_adapted_e5"],
                "per_fold_increment": {
                    fold: metrics["union_increment_over_adapted_e5"]
                    for fold, metrics in value["per_fold"].items()
                },
            }
            for name, value in analyses.items()
        },
    }, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
