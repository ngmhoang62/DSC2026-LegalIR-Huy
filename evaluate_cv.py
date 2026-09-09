"""Measure the burst_userft_maxrecall configuration on the 600 CV queries.

Reports Recall@5 at alpha=0 (every query gets 5 documents, so the number is
pure recall, not mixed with an answer-length decision), pooled and per block,
against the 7-channel reference.

Gate used throughout this project: a configuration is only worth shipping if it
beats the reference pooled AND regresses no block. A pooled gain with a block
regression has twice been followed by a leaderboard loss.

    python evaluate_cv.py               the shipped configuration
    python evaluate_cv.py --all         plus the per-channel ablations
"""

from __future__ import annotations

import argparse
import pickle
import warnings
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from run_burst_expanded_fusion_submission import DocumentStore
from tune_citation_graph import build_citation_table, citation_features
from tune_corpus_cap32_fusion import build_training_cap
from tune_doctype_features import build_type_table, type_features
from tune_expanded_fusion_selection import ltr_features

warnings.filterwarnings("ignore")

NAMES = ["base", "expanded", "jina", "dense", "corpus"]
EXTRA = {
    "aiteamvn_ft": "results/from_drive/aiteamvn_ft_cv.pkl",
    "jina_ft": "results/from_drive/jina_ft_cv.pkl",
    "title_embed": "results/burst_fresh_block/title_embed_scores.pkl",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true",
                    help="chay them cac to hop rieng le de doi chieu")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent
    docs = DocumentStore(sorted(
        (root / "DSC2026-LegalIR-main/v4_run/public_test_dataset/selected-contexts")
        .glob("context_*.json")))
    queries, blocks, all_ids, extended, local, scores = build_training_cap(
        root, 32, "results/corpus_index/holdout_extended_scores_cap32.pkl", depth=20)
    gold = {q: queries[q][1] for q in all_ids}
    print(f"CV: {len(all_ids)} queries, blocks "
          f"{ {k: len(v) for k, v in blocks.items()} }, "
          f"pool {sum(len(extended[q]) for q in all_ids) / len(all_ids):.1f} docs/query",
          flush=True)

    def load(rel, floor=None):
        obj = pickle.loads((root / rel).read_bytes())
        if isinstance(obj, dict) and isinstance(obj.get("scores"), dict):
            obj = obj["scores"]
        fl = floor if floor is not None else min(
            v for q in obj for v in obj[q].values())
        return {q: {d: obj.get(q, {}).get(d, fl) for d in extended[q]}
                for q in all_ids}

    vnlegal = pickle.loads(
        (root / "results/embedding_finetune/vnlegal_lal_cv_scores.pkl").read_bytes())
    base_channels = {**scores, "vnlegal_lal": vnlegal,
                     "crossenc": load("results/crossenc_fullpool/cv_scores.pkl", -11.5)}
    extra = {name: load(rel) for name, rel in EXTRA.items()}

    type_rows = type_features(
        extended, build_type_table(root, docs, all_ids, extended), queries, all_ids)
    own, cited = build_citation_table(docs, all_ids, extended)
    cite_rows = citation_features(extended, own, cited, all_ids)

    def recall(pred):
        return float(np.mean([len(gold[q] & set(pred[q])) / len(gold[q])
                              for q in all_ids]))

    def fuse(channels, weight=None):
        rows, groups = ltr_features(local, NAMES, extended, all_ids, channels)
        for block in (type_rows, cite_rows):
            for q in rows:
                rows[q] = np.concatenate([rows[q], block[q]], axis=1)
        out = {}
        for held in blocks:
            train = sum((blocks[n] for n in blocks if n != held), [])
            X = np.vstack([rows[q] for q in train])
            y = np.concatenate([[d in gold[q] for d in groups[q]]
                                for q in train]).astype(np.int8)
            sw = (np.concatenate([np.full(len(groups[q]), 1 / len(groups[q]))
                                  for q in train]) if weight else None)
            scaler = StandardScaler().fit(X)
            model = LogisticRegression(C=.15, class_weight="balanced",
                                       solver="liblinear", max_iter=3000,
                                       random_state=2026)
            model.fit(scaler.transform(X), y, sample_weight=sw)
            for q in blocks[held]:
                p = model.predict_proba(scaler.transform(rows[q]))[:, 1]
                out[q] = [groups[q][i] for i in np.argsort(-p)][:5]
        return out

    ref = fuse(base_channels)
    ref_pooled = recall(ref)
    ref_block = {n: float(np.mean([len(gold[q] & set(ref[q])) / len(gold[q])
                                   for q in blocks[n]])) for n in blocks}
    print(f"\n{'THAM CHIEU 7 kenh (alpha=0)':44s} {ref_pooled:.4f}  "
          + "  ".join(f"{n}:{v:.4f}" for n, v in ref_block.items()), flush=True)
    print(f"GATE: > {ref_pooled:.4f} va khong block nao tut\n", flush=True)

    def report(label, channels, weight=None):
        out = fuse(channels, weight)
        pooled = recall(out)
        per = {n: float(np.mean([len(gold[q] & set(out[q])) / len(gold[q])
                                 for q in blocks[n]])) for n in blocks}
        reg = [n for n in blocks if per[n] < ref_block[n] - 1e-9]
        verdict = ("PASS" if pooled > ref_pooled and not reg
                   else ("REGRESS:" + ",".join(reg) if reg else "khong tang"))
        print(f"{label:44s} {pooled:.4f} ({pooled - ref_pooled:+.4f})  "
              + "  ".join(f"{n}:{per[n]:.4f}" for n in blocks)
              + f"   {verdict}", flush=True)
        return pooled

    shipped = report("burst_userft_maxrecall (bản đã ship)",
                     {**base_channels, **extra})

    if args.all:
        print()
        for name in EXTRA:
            report(f"  chi + {name}", {**base_channels, name: extra[name]})
        report("  cả 3 + trọng số/query", {**base_channels, **extra}, "per_query")
        for drop in EXTRA:
            keep = {k: v for k, v in extra.items() if k != drop}
            report(f"  bỏ {drop}", {**base_channels, **keep})

    print(f"\nKet qua chinh: Recall@5 = {shipped:.4f} "
          f"(mong doi 0.9561)", flush=True)
    print("Luu y: day la CV 600 query, KHONG phai diem leaderboard.", flush=True)
    return 0 if abs(shipped - 0.9561) < 1e-4 else 2


if __name__ == "__main__":
    raise SystemExit(main())
