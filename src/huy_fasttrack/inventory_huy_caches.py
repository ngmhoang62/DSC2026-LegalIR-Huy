"""Inventory Huy cached channels against the current 6,991-query V2 population."""

from __future__ import annotations

import json
import pickle
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
FOLDS = ROOT / "results/research_v2_forensic/V2_FOLDS.json"
FILES = [
    "results/aiteamvn_dense/holdout_scores_512.pkl",
    "results/burst_expanded_fusion/corpus_rank_cap32.pkl",
    "results/burst_expanded_fusion/expansion_scores.pkl",
    "results/burst_expanded_fusion/rerank_scores.pkl",
    "results/burst_fresh_block/title_embed_scores.pkl",
    "results/burst_gpu_threeview/cpu_top20.pkl",
    "results/burst_gpu_threeview/gpu_scores.checkpoint.pkl",
    "results/burst_large_ltr/fresh_1251_1350_retrieval.pkl",
    "results/burst_large_ltr/fresh_1351_1450_retrieval.pkl",
    "results/burst_large_ltr/fresh_1451_1750_retrieval.pkl",
    "results/burst_large_ltr/retrieval_train1000_tune50_val100.pkl",
    "results/corpus_index/holdout_dense_rank_cap32.pkl",
    "results/corpus_index/holdout_extended_scores_cap32.pkl",
    "results/crossenc_fullpool/cv_scores.pkl",
    "results/dense_expansion/union50_scores.pkl",
    "results/e5_dense/holdout_scores.pkl",
    "results/embedding_finetune/vnlegal_lal_cv_scores.pkl",
    "results/expanded_rerank/scores.pkl",
    "results/from_drive/aiteamvn_ft_cv.pkl",
    "results/from_drive/jina_ft_cv.pkl",
    "results/jina_reranker/holdout_scores_finetuned.pkl",
    "results/vietnamese_reranker/holdout_scores_512_finetuned.pkl",
    "results/research_v2_post_e5/V2_ADAPTED_E5_LAL_EQUAL_RRF32_PREDICTIONS.jsonl",
    "results/research_v2_e5_confirmation/E5_STRICT_CONFIRMATION_REPORT.json",
]


def possible_qid_map(value, evaluable: set[str]):
    if not isinstance(value, dict) or not value:
        return None
    keys = {str(key) for key in value}
    overlap = keys & evaluable
    if len(overlap) < min(10, len(keys)):
        return None
    return {"keys": len(keys), "evaluable_overlap": len(overlap),
            "coverage": len(overlap) / len(evaluable), "sample": sorted(overlap, key=int)[:3]}


def inspect(value, evaluable: set[str], prefix="root", depth=0):
    found = []
    candidate = possible_qid_map(value, evaluable)
    if candidate is not None:
        found.append({"path": prefix, **candidate})
    if isinstance(value, dict) and depth < 2:
        for key, child in list(value.items())[:100]:
            if isinstance(child, (dict, list, tuple)):
                found.extend(inspect(child, evaluable, f"{prefix}.{key}", depth + 1))
    elif isinstance(value, (list, tuple)) and depth < 2:
        for index, child in enumerate(value[:20]):
            if isinstance(child, (dict, list, tuple)):
                found.extend(inspect(child, evaluable, f"{prefix}[{index}]", depth + 1))
    return found


def main() -> None:
    fold_payload = json.loads(FOLDS.read_text(encoding="utf-8"))
    all_fold_qids = {str(qid) for qids in fold_payload["folds"].values() for qid in qids}
    evaluable = all_fold_qids - set(map(str, fold_payload["population"]["non_evaluable_qids"]))
    started = time.perf_counter(); rows = []
    for relative in FILES:
        path = ROOT / relative
        item = {"path": relative, "exists": path.is_file(),
                "bytes": path.stat().st_size if path.is_file() else None}
        if path.is_file():
            try:
                if path.suffix == ".pkl":
                    value = pickle.loads(path.read_bytes())
                    item["type"] = type(value).__name__
                    item["top_level_keys"] = [str(key) for key in list(value)[:20]] if isinstance(value, dict) else None
                    item["qid_maps"] = inspect(value, evaluable)
                elif path.suffix == ".json":
                    value = json.loads(path.read_text(encoding="utf-8")); item["type"] = type(value).__name__; item["qid_maps"] = inspect(value, evaluable)
                else:
                    qids = set()
                    with path.open("r", encoding="utf-8") as stream:
                        for line in stream:
                            if line.strip(): qids.add(str(json.loads(line)["qid"]))
                    item["type"] = "jsonl"; item["qid_maps"] = [{"path":"root","keys":len(qids),"evaluable_overlap":len(qids&evaluable),"coverage":len(qids&evaluable)/len(evaluable)}]
            except Exception as error:
                item["error"] = repr(error)
        rows.append(item); print(json.dumps(item, ensure_ascii=False), flush=True)
    report = {"schema_version":"dsc2026.huy_fasttrack.cache_inventory.v1",
              "evaluable_queries":len(evaluable),"files":rows,
              "runtime_seconds":time.perf_counter()-started}
    output = ROOT / "results/huy_fasttrack/HUY_CACHE_INVENTORY.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")


if __name__ == "__main__": main()
