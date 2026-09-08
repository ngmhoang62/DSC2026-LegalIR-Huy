"""Public submission: expanded-candidate multi-view fusion (1-5 docs/query).

Pipeline per query
  1. BURST multistage top-20            (CPU, cached from the three-view run)
  2. raw lexical union top-50           (CPU, from the retrieval cache)
  3. dense expansion over the union     (GPU, Vietnamese_Embedding, 1 passage/doc)
  3b. full-corpus dense retrieval       (GPU, chunk index over all 8,532 contexts)
  4. candidate set = 1 + top-20 of 3 + top-20 of 3b
  5. fine-tuned Jina + Vietnamese dense over candidates (GPU, 2 passages/doc)
  6. cross-view logistic ranker (trained on the labelled holdouts), top five
  7. dynamic-threshold cutoff: the real scoring.py only requires
     0 < len(pred) <= 5 (confirmed by reading it directly), NOT exactly 5 --
     precision is hits/len(pred), so for the ~93% of queries with a single
     gold document, trimming a confidently-wrong tail turns 1-hit-of-5
     (precision 0.20) into 1-hit-of-1 (precision 1.00) at little recall
     cost.  Rank 1 is always kept; ranks 2-5 are kept only if their
     calibrated probability is >= alpha * rank-1's probability.
     alpha=0.15 chosen by leave-one-block-out on N=600, cap=32 pipeline
     (tune_dynamic_threshold.py + the fine-grained cap32 alpha sweep): this
     is the recall-free zone boundary -- Recall stays IDENTICAL to
     always-returning-5 in every one of the 4 blocks up through alpha=0.15
     (pooled 0.9511 both ways: a=0.98, b=0.97, c=0.96, d=0.9322 unchanged),
     while Precision still rises 0.204->0.253 (F2 0.549->0.612) because only
     candidates below 15% of rank-1's probability get cut, and those were
     never the gold doc anyway.  alpha=0.18 is the first step that costs
     real recall (block b drops 0.97->0.955).  Higher alpha buys much more
     precision (e.g. alpha=0.85 -> P~0.61, F2~0.80) but trades away real
     recall; alpha=0.15 was chosen to keep recall at its ceiling per
     explicit instruction, precision raised only as a free side effect.

Stages checkpoint independently, so the run resumes after an interruption.
"""

from __future__ import annotations

import argparse
import gc
import json
import pickle
import re
import sqlite3
import threading
import time
import zipfile
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlparse

import torch
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

from benchmark_aiteamvn_holdouts import encode_cls
from benchmark_dense_expansion_holdouts import raw_union
from benchmark_jina_reranker_holdouts import top_passages
from run_burst_multistage_submission import load_metadata
from tune_burst_memory import build_query_memory
from tune_burst_multistage_posterior import weighted_rrf
from tune_burst_score_ltr import retrieve


VIEWS = ["base", "expanded", "jina", "dense", "corpus"]
CORPUS_DEPTH = 20
# cap=32 (32 chunks/doc in the corpus dense index, vs. the original 16) beats
# cap=16 on LOBO Recall@5 (0.9481 -> 0.9511, no block regressing --
# tune_corpus_cap32_fusion.py): finer per-document chunking both rescues one
# more ceiling-limited gold doc and ranks better even for documents cap16
# could already reach (dense-branch R@20 rose 0.86-0.89 -> 0.93-0.96).
CORPUS_CAP = 32
EXPANSION_CONFIG = {"depth": 50, "passages_per_doc": 1, "max_length": 512,
                    "model": "AITeamVN_Vietnamese_Embedding"}
RERANK_CONFIG = {"expanded_depth": 20, "passages_per_doc": 2, "max_length": 512,
                 "expansion_weights": (.55, .45), "expansion_rrf_k": 10,
                 "models": ["jina-finetuned", "AITeamVN_Vietnamese_Embedding"]}


def prefetch(pool, items, prepare, ahead=2):
    """Yield prepare(item) with a bounded look-ahead.

    Passage selection is pure-Python and costs about as much as the GPU pass it
    feeds, so preparing the next queries during a forward pass nearly halves the
    wall clock.  The look-ahead stays small to keep memory flat.
    """
    pending = deque()
    stream = iter(items)
    for item in stream:
        pending.append(pool.submit(prepare, item))
        if len(pending) > ahead:
            break
    while pending:
        result = pending.popleft().result()
        nxt = next(stream, None)
        if nxt is not None:
            pending.append(pool.submit(prepare, nxt))
        yield result


SLUG_ID_RE = re.compile(r"-\d+$")


def title_from_link(link):
    """20/8532 corpus documents have a completely empty "passage" field (a
    genuine source-data gap, not a loading bug -- confirmed by reading the
    raw JSON directly). Their "link" field's URL slug still carries a
    readable (unaccented, hyphen-joined) title, e.g.
    ".../TCVN-8400-24-2014-Benh-dong-vat-chan-doan-Benh-viem-phe-quan-truyen-nhiem-914993.aspx"
    -> "TCVN 8400 24 2014 Benh dong vat chan doan Benh viem phe quan truyen nhiem".
    Not a full recovery of the lost content, but real topic signal where
    there was previously none at all."""
    if not link:
        return ""
    slug = urlparse(link).path.rsplit("/", 1)[-1]
    slug = re.sub(r"\.aspx$", "", slug, flags=re.I)
    slug = SLUG_ID_RE.sub("", slug)
    return slug.replace("-", " ").strip()


class DocumentStore:
    """Lazy passage text, so a run needs megabytes rather than gigabytes of RAM.

    Holding all 8,532 contexts in memory competes with the reranker weights on a
    machine without a page file; reads are a rounding error next to GPU time.
    """

    def __init__(self, paths, cache_size=512):
        self.paths = {p.stem[len("context_"):]: p for p in paths}
        self.cache = {}
        self.cache_size = cache_size

    def __getitem__(self, doc):
        text = self.cache.get(doc)
        if text is None:
            path = self.paths.get(doc)
            if path is None:
                return ""
            row = json.loads(path.read_text(encoding="utf-8"))
            text = row.get("passage") or title_from_link(row.get("link"))
            if len(self.cache) >= self.cache_size:
                self.cache.clear()
            self.cache[doc] = text
        return text


def load_public_retrieval(root, args, doc_ids, train, public, public_ids):
    """Reuse the cached BURST retrieval lists; complete any missing queries."""
    for name in ("burst_multistage", "burst_robust_fusion"):
        path = root / "results" / name / "public_retrieval.pkl"
        if path.exists():
            cache = pickle.loads(path.read_bytes()).get("cache", {})
            if not any(q not in cache for q in public_ids):
                print(f"Retrieval cache complete: {path}", flush=True)
                return cache
            break
    else:
        cache = {}
    print("Completing public retrieval cache", flush=True)
    conn = sqlite3.connect(args.db)
    build_query_memory(conn, train)
    conn.close()
    missing = [q for q in public_ids if q not in cache]
    local = threading.local()

    def one(qid):
        if not hasattr(local, "conn"):
            local.conn = sqlite3.connect(args.db)
        return qid, retrieve(local.conn, doc_ids, None, public[qid])

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i, (qid, result) in enumerate(pool.map(one, missing), 1):
            cache[qid] = result
            if i % 25 == 0:
                print(f"Retrieved {i}/{len(missing)}", flush=True)
    (root / "results/burst_expanded_fusion").mkdir(parents=True, exist_ok=True)
    (root / "results/burst_expanded_fusion/public_retrieval.pkl").write_bytes(
        pickle.dumps({"qids": public_ids, "cache": cache}, protocol=5))
    return cache


def dense_expansion(root, output, public, public_ids, raw, documents, device):
    """GPU stage A: score the raw union so weak lexical branches can be reordered."""
    path = output / "expansion_scores.pkl"
    scores = {}
    if path.exists():
        saved = pickle.loads(path.read_bytes())
        if saved.get("config") == EXPANSION_CONFIG:
            scores = saved.get("scores", {})
    print(f"Dense expansion cache {len(scores)}/{len(public_ids)}", flush=True)
    if len(scores) >= len(public_ids):
        return scores

    model_path = root / "models" / EXPANSION_CONFIG["model"]
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path, dtype=torch.float16).eval().to(device)

    def prepare(q):
        return q, raw[q], [top_passages(public[q], documents[doc], count=1)[0]
                           for doc in raw[q]]

    remaining = [q for q in public_ids if q not in scores]
    started = time.perf_counter()
    done = 0
    with ThreadPoolExecutor(max_workers=2) as pool:
        for q, docs, passages in prefetch(pool, remaining, prepare):
            qvec = encode_cls(model, tokenizer, [public[q]], 1,
                              EXPANSION_CONFIG["max_length"])[0]
            dvec = encode_cls(model, tokenizer, passages, 32,
                              EXPANSION_CONFIG["max_length"])
            scores[q] = {doc: float(s) for doc, s in zip(docs, dvec @ qvec)}
            done += 1
            if done % 25 == 0:
                path.write_bytes(pickle.dumps(
                    {"config": EXPANSION_CONFIG, "scores": scores}, protocol=5))
                rate = (time.perf_counter() - started) / done
                left = len(remaining) - done
                print(f"Expansion {len(scores)}/{len(public_ids)} "
                      f"({rate:.2f}s/query, eta {rate*left/60:.1f}m)", flush=True)
    path.write_bytes(pickle.dumps({"config": EXPANSION_CONFIG, "scores": scores},
                                  protocol=5))
    del model
    torch.cuda.empty_cache()
    return scores


def rerank(root, output, public, public_ids, candidates, documents, device):
    """GPU stage B: the two neural views over the union candidate set."""
    path = output / "rerank_scores.pkl"
    saved = {"config": RERANK_CONFIG, "jina": {}, "dense": {}}
    if path.exists():
        loaded = pickle.loads(path.read_bytes())
        if loaded.get("config") == RERANK_CONFIG:
            saved = loaded
    # A query needs work when any of its candidates lacks a score, so growing the
    # candidate pool only costs the documents that are actually new.
    remaining = [q for q in public_ids
                 if any(d not in saved["jina"].get(q, {}) or
                        d not in saved["dense"].get(q, {}) for d in candidates[q])]
    print(f"Rerank cache {len(public_ids)-len(remaining)}/{len(public_ids)}", flush=True)
    if not remaining:
        return saved

    jina_path = root / "models/jina-reranker-v2-base-multilingual"
    jina_tok = AutoTokenizer.from_pretrained(jina_path, trust_remote_code=True,
                                             fix_mistral_regex=True)
    jina = AutoModelForSequenceClassification.from_pretrained(
        jina_path, trust_remote_code=True, dtype=torch.bfloat16)
    jina.load_state_dict(torch.load(
        root / "results/jina_reranker/burst_pairwise_state.pt",
        map_location="cpu", weights_only=True)["state_dict"], strict=False)
    jina._tokenizer = jina_tok
    jina.eval().to(device)

    dense_path = root / "models/AITeamVN_Vietnamese_Embedding"
    dense_tok = AutoTokenizer.from_pretrained(dense_path)
    dense = AutoModel.from_pretrained(dense_path, dtype=torch.float16).eval().to(device)

    print(f"Rerankers ready on {torch.cuda.get_device_name(0)}", flush=True)

    def prepare(q):
        question = public[q]
        known = saved["jina"].get(q, {})
        owners, passages = [], []
        for doc in candidates[q]:
            if doc in known and doc in saved["dense"].get(q, {}):
                continue
            for passage in top_passages(question, documents[doc],
                                        count=RERANK_CONFIG["passages_per_doc"]):
                owners.append(doc)
                passages.append(passage)
        return q, owners, passages

    started = time.perf_counter()
    done = 0
    with ThreadPoolExecutor(max_workers=2) as pool:
        for q, owners, passages in prefetch(pool, remaining, prepare):
            question = public[q]
            js = dict(saved["jina"].get(q, {}))
            ds = dict(saved["dense"].get(q, {}))
            if passages:
                jraw = jina.compute_score([(question, p) for p in passages],
                                          batch_size=16,
                                          max_length=RERANK_CONFIG["max_length"])
                qvec = encode_cls(dense, dense_tok, [question], 1,
                                  RERANK_CONFIG["max_length"])[0]
                pvec = encode_cls(dense, dense_tok, passages, 32,
                                  RERANK_CONFIG["max_length"])
                for doc, j, d in zip(owners, jraw, pvec @ qvec):
                    js[doc] = max(js.get(doc, -1e9), float(j))
                    ds[doc] = max(ds.get(doc, -1e9), float(d))
            saved["jina"][q] = {d: js[d] for d in candidates[q]}
            saved["dense"][q] = {d: ds[d] for d in candidates[q]}
            done += 1
            if done % 20 == 0:
                path.write_bytes(pickle.dumps(saved, protocol=5))
                rate = (time.perf_counter() - started) / done
                left = len(remaining) - done
                print(f"Rerank {done}/{len(remaining)} "
                      f"({rate:.2f}s/query, eta {rate*left/60:.1f}m)", flush=True)
    path.write_bytes(pickle.dumps(saved, protocol=5))
    return saved


def corpus_dense(root, output, public, public_ids, device, cap=16, depth=20):
    """Retrieve over the whole corpus with the chunk index built offline.

    Lexical branches miss roughly 2% of gold documents at any depth; this is the
    only branch that can reach them.  Query encoding is cheap and search is one
    matrix product, so the cost is dominated by the one-off index build.
    """
    path = output / f"corpus_rank_cap{cap}.pkl"
    if path.exists():
        saved = pickle.loads(path.read_bytes())
        if saved.get("cap") == cap and saved.get("depth") >= depth:
            print(f"Corpus rank cache {len(saved['ranking'])}/{len(public_ids)}",
                  flush=True)
            if len(saved["ranking"]) >= len(public_ids):
                return saved["ranking"], saved["scores"]

    from benchmark_corpus_dense_recall import load_index, rank_documents

    documents, vectors, owners = load_index(root, cap)
    print(f"Corpus index: {len(documents)} documents, {vectors.shape[0]} chunks",
          flush=True)
    model_path = root / "models/AITeamVN_Vietnamese_Embedding"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(
        model_path, dtype=torch.float16).eval().to(device)
    qvectors = encode_cls(model, tokenizer, [public[q] for q in public_ids], 16, 512)
    del model
    torch.cuda.empty_cache()

    ranking, scores = {}, {}
    started = time.perf_counter()
    for start in range(0, len(public_ids), 100):
        batch = public_ids[start:start + 100]
        ranked = rank_documents(vectors, owners, len(documents),
                                qvectors[start:start + 100], top_k=max(depth, 100))
        for q, (order, best) in zip(batch, ranked):
            ranking[q] = [documents[i] for i in order]
            scores[q] = {documents[i]: float(best[i]) for i in order}
        print(f"Corpus search {min(start+100, len(public_ids))}/{len(public_ids)} "
              f"({time.perf_counter()-started:.1f}s)", flush=True)
    path.write_bytes(pickle.dumps(
        {"cap": cap, "depth": max(depth, 100), "ranking": ranking, "scores": scores},
        protocol=5))
    return ranking, scores


def ltr_fusion(root, names, view_rank, candidates, public_ids, public_scores,
              documents, public_queries):
    """Fit the cross-view ranker on the labelled holdouts, apply it to public.

    Rank features plus per-query standardized model scores, plus:
      - document-type features (Luat/Nghi dinh/Thong tu/Quyet dinh/Cong van):
        79% of N=600 CV ranking failures had gold and the wrongly-ranked top-1
        be a different class of legal instrument.
      - citation-relationship features: whether one candidate cites another (as
        its "Can cu ...") or is cited by one, within the same candidate pool --
        exact structural information distinct from doctype or embeddings, found
        in 21% of the hardest remaining failures (e.g. the wrong top-1
        explicitly citing gold as its legal basis).
    C=0.15 won the leave-one-block-out comparison in tune_citation_graph.py
    after fixing DOC_NUMBER_RE/CITE_RE to also match year-less citation numbers
    ("1760/QD-BKHCN", used by Quyet dinh) instead of only the Luat/Nghi dinh/
    Thong tu year-included format -- citation coverage rose from 70% to 93%,
    and the feature's LOBO margin widened from 5W/4L to 4W/2L with no block
    regressing.

    (A query_cites feature -- does the question itself name a candidate's
    document number? -- was tried and reverted: +0.0014 LOBO Recall@5 on the
    600-query CV holdout, but the live CodaBench leaderboard score came back
    lower than the pre-query_cites submission, so the CV gain didn't
    generalize.  See tune_query_cites_feature.py / tune_citation_graph.py's
    query_cites_features for the reference implementation if revisited.)
    """
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    from tune_citation_graph import build_citation_table, citation_features
    from tune_corpus_dense_fusion import build_training
    from tune_doctype_features import build_type_table, type_features
    from tune_expanded_fusion_selection import ltr_features

    queries, _, holdout_ids, holdout_candidates, holdout_views, training_scores = (
        build_training(root, depth=CORPUS_DEPTH, cap=CORPUS_CAP,
                       extended_scores_path=
                       "results/corpus_index/holdout_extended_scores_cap32.pkl"))
    rows, groups = ltr_features(holdout_views, names, holdout_candidates, holdout_ids,
                                training_scores)
    holdout_types = build_type_table(root, documents, holdout_ids, holdout_candidates)
    holdout_type_rows = type_features(holdout_candidates, holdout_types, queries,
                                      holdout_ids)
    holdout_own, holdout_cited = build_citation_table(documents, holdout_ids,
                                                       holdout_candidates)
    holdout_cite_rows = citation_features(holdout_candidates, holdout_own,
                                          holdout_cited, holdout_ids)
    for q in rows:
        rows[q] = np.concatenate([rows[q], holdout_type_rows[q],
                                  holdout_cite_rows[q]], axis=1)

    x = np.vstack([rows[q] for q in holdout_ids])
    y = np.concatenate([[d in queries[q][1] for d in groups[q]]
                        for q in holdout_ids]).astype(np.int8)
    scaler = StandardScaler().fit(x)
    model = LogisticRegression(C=.15, class_weight="balanced", solver="liblinear",
                               max_iter=3000, random_state=2026)
    model.fit(scaler.transform(x), y)
    print(f"LTR trained on {len(holdout_ids)} queries, {len(y)} candidate rows "
          f"({int(y.sum())} positive)", flush=True)

    public_rows, public_groups = ltr_features(view_rank, names, candidates, public_ids,
                                              public_scores)
    public_types = build_type_table(root, documents, public_ids, candidates)
    public_type_rows = type_features(candidates, public_types, public_queries,
                                     public_ids)
    public_own, public_cited = build_citation_table(documents, public_ids, candidates)
    public_cite_rows = citation_features(candidates, public_own, public_cited,
                                         public_ids)
    for q in public_rows:
        public_rows[q] = np.concatenate([public_rows[q], public_type_rows[q],
                                         public_cite_rows[q]], axis=1)

    fused = {}
    fused_proba = {}
    for q in public_ids:
        proba = model.predict_proba(scaler.transform(public_rows[q]))[:, 1]
        order = np.argsort(-proba)
        fused[q] = [public_groups[q][i] for i in order]
        fused_proba[q] = {public_groups[q][i]: float(proba[i]) for i in order}
    detail = {"model": "LogisticRegression(C=0.15, balanced)",
              "train_queries": len(holdout_ids),
              "features": "rank+score+doctype+citation",
              "coefficients": model.coef_[0].round(4).tolist()}
    return fused, fused_proba, detail


def main():
    root = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path,
                    default=root / "DSC2026-LegalIR-main/v4_run/public_test_dataset")
    ap.add_argument("--db", type=Path,
                    default=root / "benchmarks/legalir_full_fts.sqlite")
    ap.add_argument("--output-dir", type=Path,
                    default=root / "results/burst_expanded_fusion")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--weights", type=Path,
                    default=root / "burst_expanded_fusion_robust.json")
    ap.add_argument("--stack", default="four_view")
    ap.add_argument("--fusion", choices=("ltr", "rrf"), default="ltr")
    ap.add_argument("--stage", choices=("expansion", "candidates", "rerank", "all"),
                    default="all")
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    paths, doc_ids, train, public = load_metadata(args.data_dir)
    public_ids = list(public)
    valid = set(doc_ids)
    retrieval = load_public_retrieval(root, args, doc_ids, train, public, public_ids)

    base = pickle.loads(
        (root / "results/burst_gpu_threeview/cpu_top20.pkl").read_bytes())["rankings"]
    if any(q not in base for q in public_ids):
        raise RuntimeError("Missing CPU multistage candidates; run the three-view job")

    documents = DocumentStore(paths)
    print(f"Document store ready ({len(paths)} contexts, lazy)", flush=True)

    # The retrieval lists are the largest object in the run; keep only the union.
    raw = {q: raw_union(retrieval[q], EXPANSION_CONFIG["depth"]) for q in public_ids}
    retrieval.clear()
    del retrieval
    gc.collect()
    expansion_scores = dense_expansion(root, args.output_dir, public, public_ids,
                                       raw, documents, device)
    if args.stage == "expansion":
        print("Expansion stage complete", flush=True)
        return

    dense_rank = {q: sorted(raw[q], key=lambda d: (-expansion_scores[q][d], d))
                  for q in public_ids}
    expanded = weighted_rrf([raw, dense_rank], RERANK_CONFIG["expansion_weights"],
                            RERANK_CONFIG["expansion_rrf_k"])
    corpus_rank, corpus_score = corpus_dense(root, args.output_dir, public,
                                             public_ids, device,
                                             cap=CORPUS_CAP, depth=CORPUS_DEPTH)
    gc.collect()
    candidates = {q: list(dict.fromkeys(
        list(base[q]) + expanded[q][:RERANK_CONFIG["expanded_depth"]] +
        corpus_rank[q][:CORPUS_DEPTH]))
        for q in public_ids}
    sizes = [len(candidates[q]) for q in public_ids]
    print(f"Candidates min={min(sizes)} mean={sum(sizes)/len(sizes):.1f} "
          f"max={max(sizes)}", flush=True)
    # Past this point only the candidates matter.  Every branch keeps a hundred
    # documents per query, and together they crowd out the reranker weights on
    # this machine: model loading fails while spawning its loader threads.
    # Kept sparse (only docs the corpus search actually returned) so the "corpus"
    # rank view below can tell a real hit from a padded -1.0 fallback; channel
    # construction pads with .get(d, -1.0) at the point of use instead.
    corpus_score = {q: {d: corpus_score[q][d] for d in candidates[q]
                        if d in corpus_score[q]} for q in public_ids}
    expansion_scores = {q: {d: expansion_scores[q][d] for d in candidates[q]
                            if d in expansion_scores[q]} for q in public_ids}
    expanded = {q: expanded[q][:RERANK_CONFIG["expanded_depth"]] for q in public_ids}
    corpus_rank.clear()
    raw.clear()
    dense_rank.clear()
    gc.collect()
    (args.output_dir / "candidates.pkl").write_bytes(pickle.dumps(
        {"candidates": candidates, "expanded": expanded,
         "corpus_score": corpus_score, "expansion": expansion_scores},
        protocol=5))
    if args.stage == "candidates":
        print(f"Candidate stage complete: {args.output_dir/'candidates.pkl'}",
              flush=True)
        return

    scores = rerank(root, args.output_dir, public, public_ids, candidates,
                    documents, device)
    if args.stage == "rerank":
        print("Rerank stage complete", flush=True)
        return

    view_rank = {
        "base": {q: list(base[q]) for q in public_ids},
        "expanded": expanded,
        "jina": {q: sorted(candidates[q], key=lambda d: (-scores["jina"][q][d], d))
                 for q in public_ids},
        "dense": {q: sorted(candidates[q], key=lambda d: (-scores["dense"][q][d], d))
                  for q in public_ids},
        # corpus_rank was freed earlier to keep memory flat; corpus_score (already
        # trimmed to candidates) reconstructs the same ordering.  Only docs the
        # corpus search actually returned are listed -- matching the training-side
        # construction in tune_corpus_dense_fusion.build_training, where absent
        # docs fall back to ltr_features' rank-60 default rather than a padded
        # in-pool rank.
        "corpus": {q: sorted((d for d in candidates[q] if d in corpus_score[q]),
                             key=lambda d: (-corpus_score[q][d], d))
                   for q in public_ids},
    }
    names = VIEWS
    if args.fusion == "ltr":
        # Trained on the 300 labelled holdout queries; best leave-one-block-out score.
        # E5 was scored over the multistage top-20 during the three-view run;
        # reusing it costs nothing and was the only measurable gain left.
        three_view = pickle.loads(
            (root / "results/burst_gpu_threeview/gpu_scores.checkpoint.pkl")
            .read_bytes())["scores"]
        public_scores = {
            "jina": scores["jina"], "dense": scores["dense"],
            "expansion": expansion_scores,
            "e5": {q: three_view[q]["e5"] for q in public_ids},
            "corpus": {q: {d: corpus_score[q].get(d, -1.0) for d in candidates[q]}
                       for q in public_ids},
        }
        public_queries = {q: (public[q], set()) for q in public_ids}
        fused, fused_proba, detail = ltr_fusion(root, names, view_rank, candidates,
                                                public_ids, public_scores, documents,
                                                public_queries)
    else:
        chosen = json.loads(args.weights.read_text(encoding="utf-8"))
        best = chosen["stacks"][args.stack]["top"][0]
        names, weights, rrf_k = best["views"], tuple(best["weights"]), best["rrf_k"]
        fused = weighted_rrf([view_rank[n] for n in names], weights, rrf_k)
        fused_proba = None
        detail = {"weights": list(weights), "rrf_k": rrf_k,
                  "holdout": best.get("blocks")}
    print(f"Fusion {args.fusion}: views={names}", flush=True)

    # alpha=0.15 chosen by LOBO on N=600, cap=32 (tune_dynamic_threshold.py);
    # see module docstring.  Rank 1 always kept, so every query still
    # returns >=1 doc.
    THRESHOLD_ALPHA = 0.15

    predictions = {}
    for q in public_ids:
        final = [d for d in fused[q] if d in valid][:5]
        for pool in (candidates[q], doc_ids):
            for doc in pool:
                if len(final) >= 5:
                    break
                if doc not in final and doc in valid:
                    final.append(doc)
        final = final[:5]
        if fused_proba is not None and final:
            p = [fused_proba[q].get(d, 0.0) for d in final]
            kept = [final[0]]
            for d, s in zip(final[1:], p[1:]):
                if s >= THRESHOLD_ALPHA * p[0]:
                    kept.append(d)
            final = kept
        predictions[q] = {"answer": final}

    if set(predictions) != set(public):
        raise RuntimeError("QID mismatch")
    for q, row in predictions.items():
        answers = row["answer"]
        if not (0 < len(answers) <= 5) or len(set(answers)) != len(answers) or any(
                d not in valid for d in answers):
            raise RuntimeError(f"Invalid prediction: {q}")

    out_json = args.output_dir / "submission.json"
    out_zip = args.output_dir / "submission.zip"
    out_json.write_text(json.dumps(predictions, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.write(out_json, arcname="submission.json")
    lengths = [len(row["answer"]) for row in predictions.values()]
    (args.output_dir / "run_metadata.json").write_text(json.dumps({
        "expansion": EXPANSION_CONFIG, "rerank": RERANK_CONFIG,
        "fusion": args.fusion, "views": names, "detail": detail,
        "queries": len(predictions), "documents_per_query": (
            f"1-5 (dynamic threshold, alpha={THRESHOLD_ALPHA})"
            if fused_proba is not None else 5),
        "mean_documents_per_query": sum(lengths) / len(lengths),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved: {out_zip}", flush=True)


if __name__ == "__main__":
    main()
