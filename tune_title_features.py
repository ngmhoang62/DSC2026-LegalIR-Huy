"""Direction 1: title/subject-line matching for same-type sibling confusion.

Manual inspection of the current failures (after the doctype fix) found a new
dominant pattern: gold and the wrongly-ranked top-1 are the SAME document type
(often both Luat) but cover ADJACENT topics -- e.g. Luat 16/2012/QH13
(advertising) beat the correct Luat 59/2010/QH12 for a question about false
advertising by artists.  Matching against the full document body dilutes this
signal; the document's own subject-declaration line ("...QUY DINH VE ...",
right after the type keyword, before "Can cu") states the exact topic far more
precisely.  This extracts that line and tests token-overlap and embedding
similarity between it and the question as LTR features.
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
from tune_doctype_features import build_type_table, doc_type, type_features
from tune_expanded_fusion_selection import ltr_features


TITLE_RE = re.compile(
    r"(LUẬT|NGHỊ ĐỊNH|THÔNG TƯ|QUYẾT ĐỊNH|BỘ LUẬT|PHÁP LỆNH)\s*\n", re.I)
STOP_RE = re.compile(r"Căn cứ|CĂN CỨ")
STOPWORDS = {
    "bị", "các", "có", "của", "cho", "được", "để", "đến", "đối", "gì",
    "hay", "khi", "không", "là", "làm", "một", "nào", "những", "như",
    "phải", "ra", "sẽ", "theo", "thì", "thế", "trong", "trên", "từ",
    "và", "về", "với", "việc",
}


def extract_title(text, max_len=300, search_window=800):
    head = (text or "")[:search_window]
    m = TITLE_RE.search(head)
    if not m:
        return ""
    start = m.end()
    stop = STOP_RE.search(head[start:])
    end = start + stop.start() if stop else min(start + max_len, len(head))
    title = re.sub(r"\s+", " ", head[start:end]).strip()
    return title[:max_len]


def tokenize(text):
    return {t for t in re.findall(r"[^\W\d_]+", (text or "").lower())
            if len(t) >= 2 and t not in STOPWORDS}


def title_table(docs, ids, candidates):
    all_docs = set()
    for q in ids:
        all_docs.update(candidates[q])
    return {d: extract_title(docs[d]) for d in all_docs}


def title_overlap_features(candidates, titles, queries, ids):
    """Token-overlap between question and each candidate's title (CPU-only)."""
    rows = {}
    for q in ids:
        qtok = tokenize(queries[q][0])
        feature = []
        for d in candidates[q]:
            ttok = tokenize(titles[d])
            has_title = float(bool(ttok))
            if ttok:
                inter = len(qtok & ttok)
                jaccard = inter / max(len(qtok | ttok), 1)
                recall_q = inter / max(len(qtok), 1)
                recall_t = inter / max(len(ttok), 1)
            else:
                jaccard = recall_q = recall_t = 0.0
            feature.append([has_title, jaccard, recall_q, recall_t])
        rows[q] = np.asarray(feature, dtype=np.float32)
    return rows


def lobo(names, local, extended, queries, blocks, scores, extra_rows_list=(), c=.3):
    rows, groups = ltr_features(local, names, extended,
                                sum(blocks.values(), []), scores)
    for extra_rows in extra_rows_list:
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
    print("Building type + title tables", flush=True)
    type_tab = build_type_table(root, docs, all_ids, extended)
    titles = title_table(docs, all_ids, extended)
    coverage = sum(1 for t in titles.values() if t) / max(len(titles), 1)
    print(f"{len(titles)} candidate documents, title extraction rate "
          f"{coverage:.1%}", flush=True)

    type_rows = type_features(extended, type_tab, queries, all_ids)
    title_rows = title_overlap_features(extended, titles, queries, all_ids)

    report = {}
    m0, f0 = lobo(names, local, extended, queries, blocks, scores)
    report["baseline_5view"] = m0
    print(f"baseline_5view        recall={m0['recall']:.4f}", flush=True)

    m1, f1 = lobo(names, local, extended, queries, blocks, scores, [type_rows])
    report["plus_doctype"] = m1
    print(f"plus_doctype           recall={m1['recall']:.4f}", flush=True)

    m2, f2 = lobo(names, local, extended, queries, blocks, scores, [title_rows])
    report["plus_title_only"] = m2
    print(f"plus_title_only        recall={m2['recall']:.4f}", flush=True)

    m3, f3 = lobo(names, local, extended, queries, blocks, scores,
                  [type_rows, title_rows])
    report["plus_doctype_title"] = m3
    print(f"plus_doctype_title      recall={m3['recall']:.4f}", flush=True)

    for c in (.1, .3, 1.0):
        m, _ = lobo(names, local, extended, queries, blocks, scores,
                   [type_rows, title_rows], c=c)
        report[f"doctype_title_C{c}"] = m
        print(f"doctype_title_C{c:<4}   recall={m['recall']:.4f} "
              f"precision={m['precision']:.4f}", flush=True)

    def hit(f, q):
        return len(set(f[q][:5]) & queries[q][1]) / len(queries[q][1])
    best_label = max(report, key=lambda k: report[k]["recall"])
    print(f"\nBest: {best_label} recall={report[best_label]['recall']:.4f}",
          flush=True)
    w = l = t = 0
    fmap = {"baseline_5view": f0, "plus_doctype": f1, "plus_title_only": f2,
           "plus_doctype_title": f3}
    if best_label in fmap:
        for q in all_ids:
            a, b = hit(f1, q), hit(fmap[best_label], q)
            if b > a:
                w += 1
            elif b < a:
                l += 1
            else:
                t += 1
        print(f"{best_label} vs plus_doctype: wins={w} losses={l} ties={t}",
              flush=True)

    path = root / "burst_title_validation.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {path}", flush=True)


if __name__ == "__main__":
    main()
