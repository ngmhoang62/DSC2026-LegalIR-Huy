"""Citation-relationship feature: does one candidate cite another in the SAME
candidate pool as its legal basis (or get cited by one)?

Diagnostic found this in 21% (4/19) of the hardest remaining failures -- e.g.
QID 32546's wrongly-ranked top-1 explicitly cites gold ("Can cu Thong tu so
19/2018/TT-BQP...") in its own preamble.  This is exact, structural
information (which document legally supersedes/depends on which), genuinely
different from doctype (categorical) or embeddings (fuzzy semantic).
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
from tune_doctype_features import build_type_table, type_features
from tune_expanded_fusion_selection import ltr_features


# Luat/Nghi dinh/Thong tu number the year: "16/2012/QH13".  Quyet dinh (and some
# Cong van) drop the year entirely: "1760/QD-BKHCN" -- the year segment must stay
# optional or every Quyet dinh citation is silently missed.
DOC_NUMBER_RE = re.compile(
    r"Số:?\s*(\d+[\/\.](?:\d{4}[\/\.])?[A-ZĐƯƠ\-]+)", re.I)
CITE_RE = re.compile(
    r"Số:?\s*(\d+[\/\.](?:\d{4}[\/\.])?[A-ZĐƯƠ\-]+)", re.I)
PREAMBLE_END = re.compile(r"Điều\s+1\b", re.I)


def own_number(text):
    m = DOC_NUMBER_RE.search((text or "")[:300])
    return m.group(1).upper() if m else None


def cited_numbers(text):
    head = text or ""
    end = PREAMBLE_END.search(head[:3000])
    preamble = head[:end.start()] if end else head[:2000]
    # Skip the document's own declared number line (first match near the top).
    matches = CITE_RE.findall(preamble)
    return {m.upper() for m in matches}


def build_citation_table(docs, ids, candidates):
    all_docs = set()
    for q in ids:
        all_docs.update(candidates[q])
    own, cited = {}, {}
    for d in all_docs:
        text = docs[d]
        own[d] = own_number(text)
        c = cited_numbers(text)
        if own[d] in c:
            c.discard(own[d])
        cited[d] = c
    return own, cited


# Looser than DOC_NUMBER_RE/CITE_RE (no "So:" prefix required): a QUESTION
# naming a document ("...theo Thong tu 31/2022/TT-BTC?") doesn't use the
# formal preamble phrasing a document header does.  tune_query_cites_feature.py
# found 4/600 CV queries match this way, all 4 resolving to an actual gold
# document -- a rare but essentially risk-free signal (fires on an exact
# string match, so it's zero everywhere else) that lifted LOBO Recall@5
# 0.9511->0.9525 with no block regressing.
LOOSE_NUMBER_RE = re.compile(r"(\d+[\/\.](?:\d{4}[\/\.])?[A-ZĐƯƠ\-]{2,})", re.I)


def query_cites_features(candidates, own, question_of, ids):
    """Per-candidate: is this candidate's own document number named directly
    in the query text?  `question_of(q)` returns the question string."""
    rows = {}
    for q in ids:
        cited = {m.upper() for m in LOOSE_NUMBER_RE.findall(question_of(q))}
        feature = []
        for d in candidates[q]:
            n = own.get(d)
            feature.append([1.0 if (n and n in cited) else 0.0])
        rows[q] = np.asarray(feature, dtype=np.float32)
    return rows


def cited_documents(question, number_to_doc):
    """Corpus docs whose own_number is named directly in the question text."""
    cited = {m.upper() for m in LOOSE_NUMBER_RE.findall(question)}
    found = set()
    for n in cited:
        found.update(number_to_doc.get(n, []))
    return found


def citation_features(candidates, own, cited, ids):
    rows = {}
    for q in ids:
        pool = candidates[q]
        pool_numbers = {own[d] for d in pool if own.get(d)}
        feature = []
        for d in pool:
            cites_another = int(bool(cited.get(d, set()) & pool_numbers))
            n_cited_in_pool = len(cited.get(d, set()) & pool_numbers)
            is_cited_by_another = int(any(
                own.get(d) and own[d] in cited.get(other, set())
                for other in pool if other != d))
            n_citing_it = sum(1 for other in pool if other != d and
                             own.get(d) and own[d] in cited.get(other, set()))
            feature.append([cites_another, n_cited_in_pool,
                            is_cited_by_another, n_citing_it])
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
    type_tab = build_type_table(root, docs, all_ids, extended)
    type_rows = type_features(extended, type_tab, queries, all_ids)

    print("Building citation table", flush=True)
    own, cited = build_citation_table(docs, all_ids, extended)
    coverage = sum(1 for v in own.values() if v) / max(len(own), 1)
    has_citations = sum(1 for v in cited.values() if v) / max(len(cited), 1)
    print(f"{len(own)} candidate docs; {coverage:.1%} have a parsed own-number, "
          f"{has_citations:.1%} have at least one parsed citation", flush=True)
    cite_rows = citation_features(extended, own, cited, all_ids)
    nonzero = sum(1 for q in all_ids for row in cite_rows[q] if row.sum() > 0)
    print(f"Non-zero citation-relationship rows: {nonzero}", flush=True)

    report = {}
    m0, f0 = lobo(names, local, extended, queries, blocks, scores)
    report["baseline_no_doctype"] = m0
    print(f"baseline (no doctype)     recall={m0['recall']:.4f}", flush=True)

    m1, f1 = lobo(names, local, extended, queries, blocks, scores, [type_rows])
    report["plus_doctype"] = m1
    print(f"plus_doctype               recall={m1['recall']:.4f}", flush=True)

    m2, f2 = lobo(names, local, extended, queries, blocks, scores, [cite_rows])
    report["plus_citation_only"] = m2
    print(f"plus_citation_only         recall={m2['recall']:.4f}", flush=True)

    for c in (.1, .3, 1.0):
        m, f = lobo(names, local, extended, queries, blocks, scores,
                   [type_rows, cite_rows], c=c)
        report[f"doctype_citation_C{c}"] = m
        print(f"doctype_citation_C{c:<4}    recall={m['recall']:.4f} "
              f"precision={m['precision']:.4f}", flush=True)

    def hit(f, q):
        return len(set(f[q][:5]) & queries[q][1]) / len(queries[q][1])
    best_label = max(report, key=lambda k: report[k]["recall"])
    print(f"\nBest: {best_label} recall={report[best_label]['recall']:.4f}",
          flush=True)
    fmap = {"baseline_no_doctype": f0, "plus_doctype": f1,
           "plus_citation_only": f2}
    if best_label in fmap:
        w = l = t = 0
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

    path = root / "burst_citation_validation.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {path}", flush=True)


if __name__ == "__main__":
    main()
