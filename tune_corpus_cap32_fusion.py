"""Does raising the per-document chunk cap (16 -> 32) in the corpus dense index
improve downstream Recall@5, not just the raw dense-branch R@20?

benchmark_corpus_dense_recall.py showed cap32's own R@20 is much higher than
cap16's (e.g. block d: 0.856 -> 0.927) and it rescues one more gold doc
(block c ceiling 0.99 -> 1.00) that cap16 never reaches at all.  This runs the
same LOBO fusion as production (rank+score+doctype+citation, C=0.15) with the
cap32 corpus channel/view substituted for cap16, to see whether that
translates into a real Recall@5 gain.
"""

from __future__ import annotations

import pickle
from pathlib import Path

from run_burst_expanded_fusion_submission import DocumentStore
from tune_citation_graph import build_citation_table, citation_features
from tune_doctype_features import build_type_table, type_features
from tune_dynamic_threshold import apply_alpha, lobo_probabilities, real_metrics
from tune_expanded_fusion_robust import build_views
from tune_expanded_fusion_selection import holdout_scores


def build_training_cap(root, cap, extended_scores_path, depth=20, expanded_depth=20):
    """cap16 and cap32 each need their OWN extended_scores file: the jina/dense
    scores stored there are keyed to that cap's specific extended candidate set
    (base+expanded union PLUS that cap's corpus top-20), so cap16 and cap32
    must never share the mutable "results/corpus_index/holdout_extended_scores.pkl"
    live path -- pass explicit backup paths instead."""
    queries, blocks, all_ids, candidates, views = build_views(root, expanded_depth=expanded_depth)
    load = lambda p: pickle.loads((root / p).read_bytes())
    e5 = load("results/e5_dense/holdout_scores.pkl")["scores"]
    dense_saved = load(f"results/corpus_index/holdout_dense_rank_cap{cap}.pkl")
    extended_scores = load(extended_scores_path)
    corpus_rank, corpus_score = dense_saved["ranking"], dense_saved["scores"]

    extended = {q: list(dict.fromkeys(list(candidates[q]) + corpus_rank[q][:depth]))
                for q in all_ids}
    local = dict(views)
    for name, table in (("jina", extended_scores["jina"]),
                        ("dense", extended_scores["dense"])):
        local[name] = {q: sorted(extended[q],
                                 key=lambda d: (-table[q].get(d, -1e9), d))
                       for q in all_ids}
    local["corpus"] = {q: [d for d in corpus_rank[q] if d in set(extended[q])]
                       for q in all_ids}
    scores = {"jina": extended_scores["jina"], "dense": extended_scores["dense"],
              "expansion": holdout_scores(root)["expansion"], "e5": e5,
              "corpus": {q: {d: corpus_score[q].get(d, -1.0) for d in extended[q]}
                         for q in all_ids}}
    return queries, blocks, all_ids, extended, local, scores


def main():
    root = Path(__file__).resolve().parent
    names = ["base", "expanded", "jina", "dense", "corpus"]
    docs = DocumentStore(sorted(
        (root / "DSC2026-LegalIR-main/v4_run/public_test_dataset/selected-contexts")
        .glob("context_*.json")))

    print("=== cap16 (current shipped) ===", flush=True)
    q16, b16, ids16, ext16, loc16, sc16 = build_training_cap(
        root, 16, "results/corpus_index/holdout_extended_scores_cap16_backup.pkl",
        depth=20)
    t16 = build_type_table(root, docs, ids16, ext16)
    tr16 = type_features(ext16, t16, q16, ids16)
    o16, c16 = build_citation_table(docs, ids16, ext16)
    cr16 = citation_features(ext16, o16, c16, ids16)
    ranked16, proba16 = lobo_probabilities(names, loc16, ext16, q16, b16, sc16,
                                           [tr16, cr16])
    held16 = {q: q16[q] for q in ranked16}
    print("always5:", real_metrics({q: ranked16[q][:5] for q in ranked16}, held16),
          flush=True)
    print("alpha=0.1:", real_metrics(apply_alpha(ranked16, proba16, 0.1), held16),
          flush=True)

    print("\n=== cap32 (candidate) ===", flush=True)
    q32, b32, ids32, ext32, loc32, sc32 = build_training_cap(
        root, 32, "results/corpus_index/holdout_extended_scores.pkl", depth=20)
    sizes = [len(ext32[q]) for q in ids32]
    print(f"Extended pool size mean={sum(sizes)/len(sizes):.1f} "
          f"(cap16 was {sum(len(ext16[q]) for q in ids16)/len(ids16):.1f})",
          flush=True)
    t32 = build_type_table(root, docs, ids32, ext32)
    tr32 = type_features(ext32, t32, q32, ids32)
    o32, c32 = build_citation_table(docs, ids32, ext32)
    cr32 = citation_features(ext32, o32, c32, ids32)
    ranked32, proba32 = lobo_probabilities(names, loc32, ext32, q32, b32, sc32,
                                           [tr32, cr32])
    held32 = {q: q32[q] for q in ranked32}
    m0 = real_metrics({q: ranked32[q][:5] for q in ranked32}, held32)
    print("always5:", m0, flush=True)
    m1 = real_metrics(apply_alpha(ranked32, proba32, 0.1), held32)
    print("alpha=0.1:", m1, flush=True)

    for name, ids in b32.items():
        gold = {q: q32[q] for q in ids}
        r16 = real_metrics({q: ranked16[q][:5] for q in ids}, gold)
        r32 = real_metrics({q: ranked32[q][:5] for q in ids}, gold)
        print(f"block {name}: cap16 R={r16['Recall']:.4f} -> "
              f"cap32 R={r32['Recall']:.4f}", flush=True)


if __name__ == "__main__":
    main()
