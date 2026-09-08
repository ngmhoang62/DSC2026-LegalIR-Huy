"""Direction 2 (revised): document-type awareness for the LTR ranker.

Manual inspection of every N=600 ranking failure found the real pattern: 82%
(18/22) have the gold document and the wrongly-ranked top-1 be a DIFFERENT
class of legal instrument (Luat/Nghi dinh/Thong tu/Quyet dinh/Cong van).  The
ranker currently has zero notion of document type; Luat is under-weighted
relative to its true hit rate (37.8% of gold vs 27.3% of the candidate pool)
while Cong van/Quyet dinh are over-weighted.  This adds one-hot document-type
features (from the document header) and a same-block ordering hint to the
existing rank+score feature set and re-measures under 4-fold LOBO.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from run_burst_expanded_fusion_submission import DocumentStore
from tune_burst_pairwise import fixed_metrics
from tune_corpus_dense_fusion import build_training
from tune_expanded_fusion_selection import ltr_features


TYPES = ["LUAT", "NGHIDINH", "THONGTU", "QUYETDINH", "CONGVAN", "KHAC"]


def doc_type(text):
    head = text[:400].upper()
    if re.search(r"LUẬT\s+SỐ|LUẬT\s*[:\n]", head) or "QUỐC HỘI" in head[:100]:
        return "LUAT"
    if "NGHỊ ĐỊNH" in head:
        return "NGHIDINH"
    if "THÔNG TƯ" in head:
        return "THONGTU"
    if "QUYẾT ĐỊNH" in head:
        return "QUYETDINH"
    if re.search(r"V/V|CÔNG VĂN", head):
        return "CONGVAN"
    return "KHAC"


def question_type_hints(question):
    """Cheap lexical cues for which instrument level the question is asking about."""
    q = question.lower()
    return np.asarray([
        float("luật" in q), float("nghị định" in q), float("thông tư" in q),
        float("quyết định" in q), float("công văn" in q or "hướng dẫn" in q),
    ], dtype=np.float32)


def build_type_table(root, docs, ids, candidates):
    """doc -> type for every candidate document that appears anywhere in ids."""
    all_docs = set()
    for q in ids:
        all_docs.update(candidates[q])
    table = {}
    for i, d in enumerate(all_docs, 1):
        table[d] = doc_type(docs[d])
    return table


def type_features(candidates, type_table, queries, ids):
    rows = {}
    for q in ids:
        docs = candidates[q]
        type_counts = {t: 0 for t in TYPES}
        for d in docs:
            type_counts[type_table[d]] += 1
        hints = question_type_hints(queries[q][0])
        feature = []
        for d in docs:
            t = type_table[d]
            onehot = [float(t == name) for name in TYPES]
            pool_share = type_counts[t] / max(len(docs), 1)
            feature.append(onehot + [pool_share] + hints.tolist())
        rows[q] = np.asarray(feature, dtype=np.float32)
    return rows


def lobo(names, local, extended, queries, blocks, scores, extra_rows=None, c=.3):
    rows, groups = ltr_features(local, names, extended,
                                sum(blocks.values(), []), scores)
    if extra_rows is not None:
        for q in rows:
            rows[q] = np.concatenate([rows[q], extra_rows[q]], axis=1)
    ranked = {}
    for held in blocks:
        train = sum((blocks[n] for n in blocks if n != held), [])
        x = np.vstack([rows[q] for q in train])
        y = np.concatenate([[d in queries[q][1] for d in groups[q]]
                            for q in train]).astype(np.int8)
        scaler = StandardScaler().fit(x)
        model = LogisticRegression(C=c, class_weight="balanced", solver="liblinear",
                                   max_iter=3000, random_state=2026)
        model.fit(scaler.transform(x), y)
        for q in blocks[held]:
            value = model.decision_function(scaler.transform(rows[q]))
            ranked[q] = [groups[q][i] for i in np.argsort(-value)]
    m, _ = fixed_metrics(ranked, {q: queries[q] for q in ranked})
    per_block = {n: fixed_metrics(ranked, {q: queries[q] for q in blocks[n]})[0]
                for n in blocks}
    return {"recall": m["Recall@5"], "precision": m["Precision@5"],
            "f2": m["F2@5"], "blocks": per_block}, ranked


def main():
    root = Path(__file__).resolve().parent
    queries, blocks, all_ids, extended, local, scores = build_training(root, depth=20)
    names = ["base", "expanded", "jina", "dense", "corpus"]

    docs = DocumentStore(sorted(
        (root / "DSC2026-LegalIR-main/v4_run/public_test_dataset/selected-contexts")
        .glob("context_*.json")))
    print("Building document-type table", flush=True)
    type_table = build_type_table(root, docs, all_ids, extended)
    print(f"{len(type_table)} unique candidate documents typed", flush=True)

    report = {}
    m0, fused0 = lobo(names, local, extended, queries, blocks, scores)
    report["baseline"] = m0
    print(f"baseline           recall={m0['recall']:.4f} "
          f"precision={m0['precision']:.4f} f2={m0['f2']:.4f}", flush=True)

    type_rows = type_features(extended, type_table, queries, all_ids)
    for c in (.1, .3, 1.0):
        m1, fused1 = lobo(names, local, extended, queries, blocks, scores,
                          extra_rows=type_rows, c=c)
        report[f"plus_doctype_C{c}"] = m1
        print(f"plus_doctype_C{c:<4} recall={m1['recall']:.4f} "
              f"precision={m1['precision']:.4f} f2={m1['f2']:.4f}", flush=True)

    best_c = max((.1, .3, 1.0),
                key=lambda c: report[f"plus_doctype_C{c}"]["recall"])
    _, fused_best = lobo(names, local, extended, queries, blocks, scores,
                         extra_rows=type_rows, c=best_c)

    def hit(f, q):
        return len(set(f[q][:5]) & queries[q][1]) / len(queries[q][1])
    w = l = t = 0
    for q in all_ids:
        a, b = hit(fused0, q), hit(fused_best, q)
        if b > a:
            w += 1
        elif b < a:
            l += 1
        else:
            t += 1
    print(f"\nplus_doctype (best C={best_c}) vs baseline: "
          f"wins={w} losses={l} ties={t}", flush=True)
    report["paired_best_vs_baseline"] = {"wins": w, "losses": l, "ties": t,
                                         "best_c": best_c}

    path = root / "burst_doctype_validation.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {path}", flush=True)


if __name__ == "__main__":
    main()
