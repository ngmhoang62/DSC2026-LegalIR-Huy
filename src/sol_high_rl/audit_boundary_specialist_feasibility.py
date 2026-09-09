"""Audit a non-CAL600 boundary-specialist dataset before any GPU commitment."""
from __future__ import annotations
import json
import pickle
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tune_burst_phrases import multi_rrf  # noqa: E402


def main():
    data = ROOT / "DSC2026-LegalIR-main/v4_run/public_test_dataset"
    raw = json.loads((data / "train.json").read_text(encoding="utf-8"))
    queries = {str(q): (x["question"], {str(d) for d in x["answer"]}) for q, x in raw.items() if x.get("answer")}
    cal = set(json.loads((ROOT / "results/sol_high_rl/CAL600_STRATIFIED_5FOLD_SEED42.json").read_text(encoding="utf-8"))["folds"][f][i]
              for f in [f"fold_{i}" for i in range(5)] for i in range(120))
    saved = pickle.loads((ROOT / "results/burst_large_ltr/retrieval_train1000_tune50_val100.pkl").read_bytes())
    cache = saved["cache"]
    eligible = [q for q in saved["qids"] if q in queries and q not in cal]
    rows = []
    reachable_queries = 0
    positive_occurrences = 0
    hard_negative_occurrences = 0
    rank_bins = Counter()
    multi = 0
    for q in eligible:
        ranking = multi_rrf([[d for d, _ in source] for source in cache[q]], [.063, .357, .28, .30], 5)
        pos = {d: i + 1 for i, d in enumerate(ranking) if d in queries[q][1]}
        negatives = [d for d in ranking[3:20] if d not in queries[q][1]]
        if pos and negatives:
            reachable_queries += 1
            positive_occurrences += len(pos)
            hard_negative_occurrences += len(negatives)
            multi += len(queries[q][1]) > 1
            for rank in pos.values():
                rank_bins["top5" if rank <= 5 else ("rank6_20" if rank <= 20 else "gt20")] += 1
            rows.append({"qid": q, "gold_count": len(queries[q][1]), "reachable_positive_count": len(pos),
                         "hard_negative_count_rank4_20": len(negatives), "positive_ranks": sorted(pos.values())})
    report = {
        "status": "FEASIBLE" if reachable_queries >= 800 else "INSUFFICIENT",
        "source_cache": "results/burst_large_ltr/retrieval_train1000_tune50_val100.pkl",
        "cache_qids": len(saved["qids"]), "eligible_non_cal600_qids": len(eligible),
        "cal600_overlap": len(set(eligible) & cal), "reachable_training_queries": reachable_queries,
        "reachable_multi_gold_queries": multi, "positive_occurrences": positive_occurrences,
        "hard_negative_occurrences_rank4_20": hard_negative_occurrences,
        "positive_rank_bins": dict(rank_bins),
        "multi_gold_safety": "all sibling gold docs excluded from negative list",
        "checkpoint_provenance": {
            "aiteam_training_chunk_policy": "UNKNOWN_NO_LOCAL_MANIFEST",
            "aiteam_negative_mining_policy": "UNKNOWN_NO_LOCAL_MANIFEST",
            "jina_training_chunk_policy": "UNKNOWN_NO_LOCAL_MANIFEST",
            "jina_negative_mining_policy": "UNKNOWN_NO_LOCAL_MANIFEST",
            "interpretation": "cannot claim existing checkpoints lacked hard negatives",
        },
        "provable_new_contract": "diagonal similarity metric learned only from non-CAL current four-source rank4-20 negatives over frozen AITeam full-body chunks",
        "rows": rows,
    }
    path = ROOT / "results/sol_high_rl/BOUNDARY_SPECIALIST_FEASIBILITY.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, ensure_ascii=False, indent=2))

if __name__ == "__main__": main()
