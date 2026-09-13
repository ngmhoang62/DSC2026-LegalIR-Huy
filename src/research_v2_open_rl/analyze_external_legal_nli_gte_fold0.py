"""Cache-only causal forensic for the killed external legal-NLI GTE pilot."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def recall(docs, gold) -> float:
    return len(set(docs) & gold) / len(gold)


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    output = root / "results/research_v2_open_rl"
    predictions_path = output / "EXTERNAL_LEGAL_NLI_GTE_FOLD0_PREDICTIONS.jsonl"
    report_path = output / "EXTERNAL_LEGAL_NLI_GTE_FOLD0_REPORT.json"
    anchor_path = root / "results/research_v2_post_e5/V2_ADAPTED_E5_LAL_EQUAL_RRF32_PREDICTIONS.jsonl"
    predictions = {str(row["qid"]): row for row in read_jsonl(predictions_path)}
    anchor = {str(row["qid"]): row for row in read_jsonl(anchor_path) if str(row["qid"]) in predictions}
    if len(predictions) != 1398 or len(anchor) != 1398 or set(predictions) != set(anchor):
        raise RuntimeError("qid mismatch")
    descending = []; ascending = []; descending10 = []; ascending10 = []
    gold_ranks_desc = []; gold_ranks_asc = []; gold_scores = []; nongold_scores = []
    gold_above_query_median = []; ascending_wins = ascending_losses = 0
    score_ranges = []; top5_overlap = []
    for qid, row in predictions.items():
        ranking = list(map(str, row["ranking"])); scores = list(map(float, row["scores"]))
        gold = set(map(str, anchor[qid]["gold"])); base = list(map(str, anchor[qid]["fused_top5"]))
        reverse = list(reversed(ranking))
        dr = recall(ranking[:5], gold); ar = recall(reverse[:5], gold)
        descending.append(dr); ascending.append(ar)
        descending10.append(recall(ranking[:10], gold)); ascending10.append(recall(reverse[:10], gold))
        ascending_wins += ar > dr; ascending_losses += ar < dr
        top5_overlap.append(len(set(ranking[:5]) & set(reverse[:5])))
        median = float(np.median(scores)); score_ranges.append(max(scores) - min(scores))
        mapping = dict(zip(ranking, scores))
        for doc in gold:
            if doc in mapping:
                index = ranking.index(doc)
                gold_ranks_desc.append(index + 1); gold_ranks_asc.append(len(ranking) - index)
                gold_scores.append(mapping[doc]); gold_above_query_median.append(mapping[doc] >= median)
        nongold_scores.extend(mapping[doc] for doc in ranking if doc not in gold)
    official = json.loads(report_path.read_text(encoding="utf-8"))
    forensic = {
        "schema_version": "dsc2026.research_v2.external_legal_nli_gte_causal_forensic.v1",
        "status": "COMPLETE_CACHE_ONLY_KILL_CLOSED",
        "official_verdict": official["verdict"],
        "official": official["metrics"],
        "diagnostic_only_not_candidate": {
            "ascending_score_recall_at_5": float(np.mean(ascending)),
            "descending_score_recall_at_5": float(np.mean(descending)),
            "ascending_score_recall_at_10": float(np.mean(ascending10)),
            "descending_score_recall_at_10": float(np.mean(descending10)),
            "ascending_vs_descending_wins_losses_ties": {
                "wins": ascending_wins, "losses": ascending_losses,
                "ties": len(predictions) - ascending_wins - ascending_losses,
            },
            "top5_intersection_mean": float(np.mean(top5_overlap)),
            "gold_rank_desc_median": float(np.median(gold_ranks_desc)),
            "gold_rank_ascending_median": float(np.median(gold_ranks_asc)),
            "gold_score_mean": float(np.mean(gold_scores)),
            "nongold_score_mean": float(np.mean(nongold_scores)),
            "gold_score_minus_nongold_mean": float(np.mean(gold_scores) - np.mean(nongold_scores)),
            "gold_above_query_median_fraction": float(np.mean(gold_above_query_median)),
            "within_query_score_range_mean": float(np.mean(score_ranges)),
        },
        "causal_read": [
            "The adapter learned its external pairwise objective (training loss fell), but the transferred scalar is strongly anti-informative for the V2 parent boundary.",
            "The fixed external triples are short preselected query-article pairs, whereas the V2 interface asks a scalar trained on those pairs to discriminate lexical windows from many near-neighbor canonical parents.",
            "Any sign swap, alternate checkpoint, evidence count, epoch, loss, or fusion would be a rescue intervention and is not an authorized submission candidate.",
        ],
        "closed_scope": "This exact external-dataset, one-epoch GTE-LoRA, lexical-top1 raw-descending-logit interface.",
        "hashes": {"report": sha256(report_path), "predictions": sha256(predictions_path)},
    }
    target = output / "EXTERNAL_LEGAL_NLI_GTE_FOLD0_CAUSAL_FORENSIC.json"
    target.write_text(json.dumps(forensic, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(forensic["diagnostic_only_not_candidate"], indent=2))


if __name__ == "__main__":
    main()
