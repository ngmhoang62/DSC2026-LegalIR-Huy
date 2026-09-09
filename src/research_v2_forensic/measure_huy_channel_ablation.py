"""Reproduce Huy --all and add paired query diagnostics without production edits."""

from __future__ import annotations

import json
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from run_burst_expanded_fusion_submission import DocumentStore  # noqa: E402
from tune_citation_graph import build_citation_table, citation_features  # noqa: E402
from tune_corpus_cap32_fusion import build_training_cap  # noqa: E402
from tune_doctype_features import build_type_table, type_features  # noqa: E402
from tune_expanded_fusion_selection import ltr_features  # noqa: E402

warnings.filterwarnings("ignore")
NAMES = ["base", "expanded", "jina", "dense", "corpus"]
EXTRA = {
    "aiteamvn_ft": "results/from_drive/aiteamvn_ft_cv.pkl",
    "jina_ft": "results/from_drive/jina_ft_cv.pkl",
    "title_embed": "results/burst_fresh_block/title_embed_scores.pkl",
}


def main() -> None:
    docs = DocumentStore(sorted((ROOT / "DSC2026-LegalIR-main/v4_run/public_test_dataset/selected-contexts").glob("context_*.json")))
    queries, blocks, all_ids, candidates, views, scores = build_training_cap(
        ROOT, 32, "results/corpus_index/holdout_extended_scores_cap32.pkl", depth=20)
    gold = {qid: queries[qid][1] for qid in all_ids}

    def load(rel: str, floor: float | None = None):
        obj = pickle.loads((ROOT / rel).read_bytes())
        if isinstance(obj, dict) and isinstance(obj.get("scores"), dict):
            obj = obj["scores"]
        minimum = floor if floor is not None else min(value for qid in obj for value in obj[qid].values())
        return {qid: {docid: obj.get(qid, {}).get(docid, minimum) for docid in candidates[qid]} for qid in all_ids}

    vnlegal = pickle.loads((ROOT / "results/embedding_finetune/vnlegal_lal_cv_scores.pkl").read_bytes())
    base = {**scores, "vnlegal_lal": vnlegal,
            "crossenc": load("results/crossenc_fullpool/cv_scores.pkl", -11.5)}
    extra = {name: load(rel) for name, rel in EXTRA.items()}
    type_rows = type_features(candidates, build_type_table(ROOT, docs, all_ids, candidates), queries, all_ids)
    own, cited = build_citation_table(docs, all_ids, candidates)
    cite_rows = citation_features(candidates, own, cited, all_ids)

    def fuse(channels):
        rows, groups = ltr_features(views, NAMES, candidates, all_ids, channels)
        for feature_block in (type_rows, cite_rows):
            for qid in rows:
                rows[qid] = np.concatenate([rows[qid], feature_block[qid]], axis=1)
        output = {}
        for held in blocks:
            train_qids = sum((blocks[name] for name in blocks if name != held), [])
            x = np.vstack([rows[qid] for qid in train_qids])
            y = np.concatenate([[docid in gold[qid] for docid in groups[qid]] for qid in train_qids]).astype(np.int8)
            scaler = StandardScaler().fit(x)
            model = LogisticRegression(C=.15, class_weight="balanced", solver="liblinear",
                                       max_iter=3000, random_state=2026).fit(scaler.transform(x), y)
            for qid in blocks[held]:
                probability = model.predict_proba(scaler.transform(rows[qid]))[:, 1]
                output[qid] = [groups[qid][i] for i in np.argsort(-probability)[:5]]
        return output

    predictions = {"reference_7": fuse(base), "full_10": fuse({**base, **extra})}
    for name in EXTRA:
        predictions[f"only_add_{name}"] = fuse({**base, name: extra[name]})
        predictions[f"drop_{name}"] = fuse({**base, **{key: value for key, value in extra.items() if key != name}})

    def qrec(pred, qid):
        return len(gold[qid] & set(pred[qid])) / len(gold[qid])

    def metrics(pred):
        return {
            "pooled_recall_at_5": float(np.mean([qrec(pred, qid) for qid in all_ids])),
            "blocks": {name: float(np.mean([qrec(pred, qid) for qid in qids])) for name, qids in blocks.items()},
        }

    def paired(candidate, anchor):
        left, right = predictions[candidate], predictions[anchor]
        deltas = {qid: qrec(left, qid) - qrec(right, qid) for qid in all_ids}
        return {
            "anchor": anchor,
            "recall_delta": metrics(left)["pooled_recall_at_5"] - metrics(right)["pooled_recall_at_5"],
            "wins": sum(value > 0 for value in deltas.values()),
            "losses": sum(value < 0 for value in deltas.values()),
            "ties": sum(value == 0 for value in deltas.values()),
            "changed_top5_order": sum(left[qid] != right[qid] for qid in all_ids),
            "changed_top5_set": sum(set(left[qid]) != set(right[qid]) for qid in all_ids),
            "per_block_delta": {
                name: float(np.mean([deltas[qid] for qid in qids])) for name, qids in blocks.items()
            },
        }

    report = {
        "schema_version": "dsc2026.research_v2.huy_channel_ablation.v1",
        "scope": "Huy 600-query four-block LOBO; historical diagnostic only",
        "metrics": {name: metrics(pred) for name, pred in predictions.items()},
        "marginal_addition_vs_reference_7": {
            name: paired(f"only_add_{name}", "reference_7") for name in EXTRA
        },
        "leave_one_out_ablation_vs_full_10": {
            name: paired(f"drop_{name}", "full_10") for name in EXTRA
        },
    }
    output = ROOT / "results/research_v2_forensic/HUY_CHANNEL_ABLATION_CLARIFICATION.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

