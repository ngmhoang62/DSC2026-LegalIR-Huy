"""Shared plumbing for fine-tuning the three BURST reranker channels.

Everything the shipped pipeline computes on CPU or reads from cache is rebuilt
here verbatim, so a fine-tuning run only has to redo the ONE stage it changes:

  layer 1 (candidate generation)  -- read from cache, never recomputed
      BM25 multi-branch retrieval  results/burst_large_ltr/*.pkl
      multistage top-20 ("base")   key order of results/jina_reranker/...pkl
      dense expansion union-50     results/dense_expansion/union50_scores.pkl
      corpus dense rank cap=32     results/corpus_index/holdout_dense_rank_cap32.pkl
  layer 2 (rerank)                -- the fine-tuned model recomputes ITS channel;
                                     the other two channels stay cached
  layer 3 (LTR fusion)            -- refit here, cheap, CPU only
  layer 4 (dynamic threshold)     -- alpha=0.15, as shipped

The candidate pool is therefore identical across every epoch and every model,
which is what makes the per-epoch numbers comparable to each other and to the
shipped baseline.

Paths mirror the local repo layout: ROOT is the folder holding run.py and
results/, WORK is where checkpoints and rankings are written (/kaggle/working).
Both are overridable via --root/--work or the BURST_ROOT/BURST_WORK env vars.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

# Bumped whenever the module gains something a notebook calls.  Kaggle pins a
# notebook to the dataset version attached at the time, so uploading a New
# Version does not update an open notebook -- and the symptom is an
# AttributeError on a function that plainly exists in the file on your disk.
# Cell 1 prints this so the mismatch is visible in one line instead of one
# traceback.
VERSION = "2026.09.03.2"
REQUIRES = {                       # what each notebook needs to exist
    "burst_common": ["progress", "RunRecorder", "build_eval_bundle"],
    "torch_common": ["setup", "wrap_parallel", "scale_for_gpus", "choose_precision",
                     "check_pooling_discriminates", "freeze_lower_layers",
                     "step_if_finite", "guard_parameters", "report_bad_scores",
                     "report_memory_budget", "patch_hub_code_compat", "load_fp32"],
}

KAGGLE_ROOT = Path("/kaggle/input/fine_tune")   # dataset nhtclone/fine_tune -- zip của cả folder ref/
LOCAL_ROOT = Path(__file__).resolve().parent.parent


def check_code_version():
    """Confirm the attached code dataset actually has what the notebook calls."""
    import importlib
    missing = []
    for module_name, names in REQUIRES.items():
        module = importlib.import_module(module_name)
        missing += [f"{module_name}.{n}" for n in names if not hasattr(module, n)]
    if missing:
        raise RuntimeError(
            f"The attached code is out of date -- missing: {', '.join(missing)}.\n"
            f"  Kaggle pins a notebook to the dataset version that was attached "
            f"when you added it. Open the Input panel on the right, remove "
            f"'fine-tune-code' and add it back so it picks up the latest "
            f"version, then restart the kernel and run from cell 1.")
    print(f"code version {VERSION} — tất cả hàm cần thiết đều có", flush=True)


def default_root():
    """BURST_ROOT wins; otherwise the Kaggle dataset mount if it is there.

    Falls back to the folder holding this one, so the same file runs unchanged
    both from the repo and from /kaggle/input/fine_tune/fine_tune/.
    """
    env = os.environ.get("BURST_ROOT")
    if env:
        return Path(env)
    return KAGGLE_ROOT if KAGGLE_ROOT.exists() else LOCAL_ROOT


ROOT = default_root()
WORK = Path(os.environ.get("BURST_WORK", "/kaggle/working"))
DATA_SUBDIR = "DSC2026-LegalIR-main/v4_run/public_test_dataset"

# --------------------------------------------------------------------------
# Progress bars
# --------------------------------------------------------------------------

try:
    from tqdm.auto import tqdm
    HAS_TQDM = True
except ImportError:                                                 # pragma: no cover
    HAS_TQDM = False

# Deliberately slow: a Kaggle "Save & Run All" captures every bar refresh into
# the committed log, so a fast bar buys nothing on screen and buries the lines
# that actually matter.  Override with BURST_PROGRESS_INTERVAL if you want it
# livelier while watching interactively.
PROGRESS_INTERVAL = float(os.environ.get("BURST_PROGRESS_INTERVAL", "2.0"))


class _NullBar:
    """Stand-in with tqdm's surface, for when tqdm is missing."""

    def __init__(self, iterable=None):
        self.iterable = [] if iterable is None else iterable

    def __iter__(self):
        return iter(self.iterable)

    def set_postfix(self, *args, **kwargs):
        pass

    def set_description(self, *args, **kwargs):
        pass

    def update(self, n=1):
        pass

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def progress(iterable=None, desc="", total=None, unit="it", leave=True):
    """A tqdm bar, or a transparent pass-through when tqdm is unavailable."""
    if not HAS_TQDM:
        return _NullBar(iterable)
    return tqdm(iterable, desc=desc, total=total, unit=unit, leave=leave,
                mininterval=PROGRESS_INTERVAL, dynamic_ncols=True)

# --------------------------------------------------------------------------
# Pipeline constants -- copied from the shipped runner, do not drift
# --------------------------------------------------------------------------

CORPUS_DEPTH = 20
CORPUS_CAP = 32
EXPANSION_DEPTH = 50            # benchmark_dense_expansion_holdouts.DEPTH
EXPANSION_RRF_K_RAW = 20        # benchmark_dense_expansion_holdouts.RRF_K_RAW
EXPANDED_DEPTH = 20             # RERANK_CONFIG["expanded_depth"]
EXPANSION_WEIGHTS = (.55, .45)  # RERANK_CONFIG["expansion_weights"]
EXPANSION_RRF_K = 10            # RERANK_CONFIG["expansion_rrf_k"]
PASSAGES_PER_DOC = 2            # RERANK_CONFIG["passages_per_doc"]
MAX_LENGTH = 512                # RERANK_CONFIG["max_length"]

# Six-channel stack of run_vnlegal_extra_channel_submission.py.
VIEWS = ["base", "expanded", "jina", "dense", "corpus", "vnlegal_lal"]
LTR_C = .15
LTR_RANDOM_STATE = 2026
THRESHOLD_ALPHA = .15

# The 600-query leave-one-block-out set (tune_expanded_fusion_robust.build_views).
BLOCK_RANGES = {"a": (750, 850), "b": (1250, 1350),
                "c": (1350, 1450), "d": (1450, 1750)}

# Queries whose raw BM25 retrieval is cached in results/burst_large_ltr/.
RETRIEVAL_CACHE_FILES = {
    "train1000_tune50_val100": "retrieval_train1000_tune50_val100.pkl",   # qids[0:1150]
    "fresh_1251_1350": "fresh_1251_1350_retrieval.pkl",                   # qids[1250:1350]
    "fresh_1351_1450": "fresh_1351_1450_retrieval.pkl",                   # qids[1350:1450]
    "fresh_1451_1750": "fresh_1451_1750_retrieval.pkl",                   # qids[1450:1750]
}

# --------------------------------------------------------------------------
# Text handling -- verbatim from the pipeline
# --------------------------------------------------------------------------

TOKEN_RE = re.compile(r"\w+", re.UNICODE)          # benchmark_burst_v4_full_sqlite
SPACE_RE = re.compile(r"\S+", re.UNICODE)
SLUG_ID_RE = re.compile(r"-\d+$")
STOPWORDS = {
    "bị", "các", "có", "của", "cho", "được", "để", "đến", "đối", "gì",
    "hay", "khi", "không", "là", "làm", "một", "nào", "những", "như",
    "phải", "ra", "sẽ", "theo", "thì", "thế", "trong", "trên", "từ",
    "và", "về", "với", "việc", "bao", "nhiêu", "người", "quy", "định",
}


def tokens(text):
    return TOKEN_RE.findall((text or "").lower())


def title_from_link(link):
    """20/8532 contexts have an empty passage; the link slug still carries a title."""
    if not link:
        return ""
    from urllib.parse import urlparse
    slug = urlparse(link).path.rsplit("/", 1)[-1]
    slug = re.sub(r"\.aspx$", "", slug, flags=re.I)
    slug = SLUG_ID_RE.sub("", slug)
    return slug.replace("-", " ").strip()


def top_passages(question, text, count=PASSAGES_PER_DOC, window=220, overlap=70):
    """Verbatim from benchmark_jina_reranker_holdouts.top_passages.

    Both training and evaluation feed the model exactly these windows, so a
    fine-tuned checkpoint sees at inference the same text distribution it was
    trained on.
    """
    words = SPACE_RE.findall(text or "")
    if len(words) <= window + 80:
        return [" ".join(words)]
    query_tokens = tokens(question)
    content = {t for t in query_tokens if len(t) >= 3 and t not in STOPWORDS}
    numbers = {t for t in query_tokens if any(c.isdigit() for c in t)}
    bigrams = {" ".join(query_tokens[i:i+2]) for i in range(len(query_tokens)-1)}
    header = " ".join(words[:70])
    scored = []
    step = window - overlap
    for start in range(0, len(words), step):
        end = min(start + window, len(words))
        part_words = words[start:end]
        part = " ".join(part_words)
        normalized = tokens(part)
        token_set = set(normalized)
        norm_text = " ".join(normalized)
        coverage = sum(1.0 + .20 * min(normalized.count(t), 3)
                       for t in content if t in token_set)
        numeric = 3.0 * sum(t in token_set for t in numbers)
        phrase = 1.8 * sum(p in norm_text for p in bigrams)
        density = (coverage + numeric + phrase) / math.sqrt(max(len(normalized), 1))
        scored.append((density, coverage + numeric + phrase, -start, part))
        if end == len(words):
            break
    scored.sort(reverse=True)
    passages = []
    for _, _, neg_start, part in scored:
        candidate = part if -neg_start < 70 else header + "\n[ĐOẠN PHÙ HỢP]\n" + part
        if candidate not in passages:
            passages.append(candidate)
        if len(passages) >= count:
            break
    return passages


class DocumentStore:
    """Passage text by document id.

    preload=True keeps all 8,532 passages in RAM (~0.4 GB) -- worth it here
    because training touches most of the corpus many times per epoch, unlike
    the one-pass submission runner which streams them lazily.
    """

    def __init__(self, data_dir, preload=True):
        self.paths = {p.stem[len("context_"):]: p
                      for p in sorted((Path(data_dir) / "selected-contexts")
                                      .glob("context_*.json"))}
        self.cache = {}
        self.preload = preload
        if preload:
            for doc, path in progress(list(self.paths.items()),
                                      desc="documents", unit="doc"):
                self.cache[doc] = self._read(path)

    @staticmethod
    def _read(path):
        row = json.loads(path.read_text(encoding="utf-8"))
        return row.get("passage") or title_from_link(row.get("link"))

    def __getitem__(self, doc):
        text = self.cache.get(doc)
        if text is None:
            path = self.paths.get(doc)
            if path is None:
                return ""
            text = self._read(path)
            if not self.preload:
                if len(self.cache) >= 512:
                    self.cache.clear()
                self.cache[doc] = text
        return text

    def __contains__(self, doc):
        return doc in self.paths

    def ids(self):
        return list(self.paths)


# --------------------------------------------------------------------------
# Ranking / metric helpers -- verbatim
# --------------------------------------------------------------------------

def load_queries(root=ROOT):
    """All labelled queries in train.json (qid -> (question, gold ids)).

    BURST_TRAIN_JSON wins (manual override); otherwise train.json ships inside
    the single `fine_tune` dataset already, at DATA_SUBDIR under `root`.
    """
    env = os.environ.get("BURST_TRAIN_JSON")
    path = Path(env) if env else Path(root) / DATA_SUBDIR / "train.json"
    print(f"[load_queries] train.json <- {path}", flush=True)
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {str(qid): (row["question"], {str(d) for d in row["answer"]})
            for qid, row in raw.items() if row.get("answer")}


def raw_union(cache_row, depth=EXPANSION_DEPTH):
    """benchmark_dense_expansion_holdouts.raw_union -- RRF over the BM25 branches."""
    scores = {}
    for branch in cache_row:
        for rank, item in enumerate(branch[:depth], 1):
            doc = str(item[0])
            scores[doc] = scores.get(doc, 0.0) + 1.0 / (EXPANSION_RRF_K_RAW + rank)
    return sorted(scores, key=lambda d: (-scores[d], d))


def weighted_rrf(rankings, weights, k):
    """tune_burst_multistage_posterior.weighted_rrf"""
    output = {}
    for q in rankings[0]:
        rankmaps = [{d: i + 1 for i, d in enumerate(branch[q])} for branch in rankings]
        docs = set().union(*(r.keys() for r in rankmaps))
        output[q] = sorted(docs, key=lambda d: (
            -sum(w / (k + ranks.get(d, 100000)) for w, ranks in zip(weights, rankmaps)), d
        ))[:100]
    return output


def rank_by(candidates, scores):
    """tune_expanded_fusion_robust.rank_by"""
    return {q: sorted(candidates[q], key=lambda d: (-scores.get(q, {}).get(d, -1e9), d))
            for q in candidates}


def metrics(rankings, queries):
    """benchmark_burst_v4_full_sqlite.metrics"""
    recall5 = recall100 = ndcg10 = mrr10 = precision5 = 0.0
    per_query = []
    for qid, (_, gold) in queries.items():
        got = rankings.get(qid, [])
        h5 = len(gold.intersection(got[:5]))
        r5 = h5 / len(gold)
        recall5 += r5
        precision5 += h5 / 5
        recall100 += len(gold.intersection(got[:100])) / len(gold)
        dcg = sum(1 / math.log2(i + 2) for i, d in enumerate(got[:10]) if d in gold)
        ideal = sum(1 / math.log2(i + 2) for i in range(min(len(gold), 10)))
        ndcg10 += dcg / ideal if ideal else 0
        mrr10 += next((1 / (i + 1) for i, d in enumerate(got[:10]) if d in gold), 0)
        per_query.append(r5)
    n = len(queries)
    p = precision5 / n
    r = recall5 / n
    f2 = 0 if 4 * p + r == 0 else 5 * p * r / (4 * p + r)
    return {"n": n, "Recall@5": r, "Recall@100": recall100 / n,
            "nDCG@10": ndcg10 / n, "MRR@10": mrr10 / n, "F2@5": f2}, per_query


def fixed_metrics(ranked, queries):
    """tune_burst_pairwise.fixed_metrics"""
    m, pq = metrics(ranked, queries)
    hits = sum(len(set(ranked[q][:5]) & queries[q][1]) for q in queries)
    m["Precision@5"] = hits / (5 * len(queries))
    return m, pq


def official_metrics(predictions, queries):
    """The organiser's scoring.py formula, applied to variable-length answers.

    recall/precision are zero for a query whose answer list is empty or >5,
    exactly as DSC2026-LegalIR-main/scoring.py computes them.
    """
    recalls, precisions = [], []
    for q, (_, gold) in queries.items():
        pred = predictions.get(q, [])
        if 0 < len(pred) <= 5:
            hits = len(gold & set(pred))
            recalls.append(hits / len(gold))
            precisions.append(hits / len(pred))
        else:
            recalls.append(0.0)
            precisions.append(0.0)
    r = float(np.mean(recalls))
    p = float(np.mean(precisions))
    f2 = 0.0 if 4 * p + r == 0 else 5 * p * r / (4 * p + r)
    return {"recall": r, "precision": p, "f2": f2,
            "mean_answers": float(np.mean([len(predictions.get(q, [])) for q in queries]))}


def dynamic_threshold(ranked, proba, ids, alpha=THRESHOLD_ALPHA):
    """Layer 4: keep rank 1, then ranks 2-5 whose probability >= alpha * rank-1's."""
    out = {}
    for q in ids:
        final = ranked[q][:5]
        if not final:
            out[q] = []
            continue
        p = [proba[q].get(d, 0.0) for d in final]
        kept = [final[0]]
        for d, s in zip(final[1:], p[1:]):
            if s >= alpha * p[0]:
                kept.append(d)
        out[q] = kept
    return out


# --------------------------------------------------------------------------
# LTR features -- verbatim from the three modules the runner imports
# --------------------------------------------------------------------------

def ltr_features(views, names, candidates, ids, scores=None):
    """tune_expanded_fusion_selection.ltr_features"""
    rows, groups = {}, {}
    for q in ids:
        ranks = [{d: i + 1 for i, d in enumerate(views[n][q])} for n in names]
        docs = candidates[q]
        columns = []
        if scores:
            for view in sorted(scores):
                raw = scores[view].get(q, {})
                values = np.asarray([raw.get(d, np.nan) for d in docs], dtype=np.float64)
                present = values[~np.isnan(values)]
                if present.size:
                    mean, std = present.mean(), present.std() or 1.0
                    top = present.max()
                else:
                    mean, std, top = 0.0, 1.0, 0.0
                filled = np.where(np.isnan(values), mean - 2 * std, values)
                columns.append((filled - mean) / std)
                columns.append((filled - top) / std)
        feature = []
        for i, d in enumerate(docs):
            r = [ranks[j].get(d, 60) for j in range(len(names))]
            row = ([1.0 / (10 + x) for x in r] + [x / 60 for x in r] +
                   [float(min(r)), float(np.mean(r))])
            row.extend(float(column[i]) for column in columns)
            feature.append(row)
        rows[q] = np.asarray(feature, dtype=np.float32)
        groups[q] = docs
    return rows, groups


TYPES = ["LUAT", "NGHIDINH", "THONGTU", "QUYETDINH", "CONGVAN", "KHAC"]


def doc_type(text):
    """tune_doctype_features.doc_type"""
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
    q = question.lower()
    return np.asarray([
        float("luật" in q), float("nghị định" in q), float("thông tư" in q),
        float("quyết định" in q), float("công văn" in q or "hướng dẫn" in q),
    ], dtype=np.float32)


def build_type_table(docs, ids, candidates):
    all_docs = set()
    for q in ids:
        all_docs.update(candidates[q])
    return {d: doc_type(docs[d]) for d in all_docs}


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


DOC_NUMBER_RE = re.compile(r"Số:?\s*(\d+[\/\.](?:\d{4}[\/\.])?[A-ZĐƯƠ\-]+)", re.I)
CITE_RE = DOC_NUMBER_RE
PREAMBLE_END = re.compile(r"Điều\s+1\b", re.I)


def own_number(text):
    m = DOC_NUMBER_RE.search((text or "")[:300])
    return m.group(1).upper() if m else None


def cited_numbers(text):
    head = text or ""
    end = PREAMBLE_END.search(head[:3000])
    preamble = head[:end.start()] if end else head[:2000]
    return {m.upper() for m in CITE_RE.findall(preamble)}


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


# --------------------------------------------------------------------------
# The cached evaluation bundle
# --------------------------------------------------------------------------

def _load(root, relative):
    return pickle.loads((Path(root) / relative).read_bytes())


def load_retrieval_cache(root, qids_wanted=None):
    """Merged BM25 multi-branch retrieval for every query results/ has cached."""
    cache = {}
    for name in RETRIEVAL_CACHE_FILES.values():
        path = Path(root) / "results/burst_large_ltr" / name
        rows = pickle.loads(path.read_bytes())["cache"]
        if qids_wanted is None:
            cache.update(rows)
        else:
            cache.update({q: rows[q] for q in rows if q in qids_wanted})
    return cache


@dataclass
class EvalBundle:
    """Everything the LTR fusion needs for the 600 LOBO queries, from cache."""
    queries: dict
    blocks: dict
    all_ids: list
    extended: dict                  # candidate pool per query (fixed)
    views: dict                     # rank views, one per channel
    scores: dict                    # raw score channels
    documents: DocumentStore
    type_rows: dict = field(default_factory=dict)
    cite_rows: dict = field(default_factory=dict)

    def block_of(self, qid):
        for name, ids in self.blocks.items():
            if qid in ids:
                return name
        return None


def build_eval_bundle(root=ROOT, documents=None, verbose=True):
    """Rebuild the shipped 600-query holdout exactly, entirely from cache.

    Mirrors tune_expanded_fusion_robust.build_views + tune_corpus_dense_fusion.
    build_training(depth=20, cap=32) + the vnlegal_lal channel added by
    run_vnlegal_extra_channel_submission.ltr_fusion_with_vnlegal.
    """
    root = Path(root)
    queries = load_queries(root)
    qids = list(queries)
    blocks = {name: qids[lo:hi] for name, (lo, hi) in BLOCK_RANGES.items()}
    all_ids = sum(blocks.values(), [])

    old_jina = _load(root, "results/jina_reranker/holdout_scores_finetuned.pkl")["scores"]
    expansion = _load(root, "results/dense_expansion/union50_scores.pkl")["scores"]
    e5 = _load(root, "results/e5_dense/holdout_scores.pkl")["scores"]
    extended_scores = _load(root, "results/corpus_index/holdout_extended_scores_cap32.pkl")
    dense_saved = _load(root, f"results/corpus_index/holdout_dense_rank_cap{CORPUS_CAP}.pkl")
    vnlegal = _load(root, "results/embedding_finetune/vnlegal_lal_cv_scores.pkl")
    corpus_rank, corpus_score = dense_saved["ranking"], dense_saved["scores"]

    retrieval = load_retrieval_cache(root, set(all_ids))
    missing = [q for q in all_ids if q not in retrieval]
    if missing:
        raise RuntimeError(f"retrieval cache missing {len(missing)} eval queries")

    raw = {q: raw_union(retrieval[q]) for q in all_ids}
    dense_rank = {q: sorted(raw[q], key=lambda d: (-expansion[q][d], d)) for q in all_ids}
    expanded_full = weighted_rrf([raw, dense_rank], EXPANSION_WEIGHTS, EXPANSION_RRF_K)

    base = {q: list(old_jina[q]) for q in all_ids}          # multistage top-20
    candidates = {q: list(dict.fromkeys(base[q] + expanded_full[q][:EXPANDED_DEPTH]))
                  for q in all_ids}
    extended = {q: list(dict.fromkeys(candidates[q] + corpus_rank[q][:CORPUS_DEPTH]))
                for q in all_ids}

    views = {
        "base": base,
        "expanded": {q: expanded_full[q][:EXPANDED_DEPTH] for q in all_ids},
        "jina": rank_by(extended, extended_scores["jina"]),
        "dense": rank_by(extended, extended_scores["dense"]),
        "corpus": {q: [d for d in corpus_rank[q] if d in set(extended[q])]
                   for q in all_ids},
        "vnlegal_lal": rank_by(extended, vnlegal),
    }
    scores = {
        "jina": extended_scores["jina"],
        "dense": extended_scores["dense"],
        "expansion": expansion,
        "e5": e5,
        "corpus": {q: {d: corpus_score[q].get(d, -1.0) for d in extended[q]}
                   for q in all_ids},
        "vnlegal_lal": vnlegal,
    }

    if documents is None:
        documents = DocumentStore(root / DATA_SUBDIR, preload=True)
    bundle = EvalBundle(queries=queries, blocks=blocks, all_ids=all_ids,
                        extended=extended, views=views, scores=scores,
                        documents=documents)
    type_table = build_type_table(documents, all_ids, extended)
    bundle.type_rows = type_features(extended, type_table, queries, all_ids)
    own, cited = build_citation_table(documents, all_ids, extended)
    bundle.cite_rows = citation_features(extended, own, cited, all_ids)

    if verbose:
        sizes = [len(extended[q]) for q in all_ids]
        ceiling = float(np.mean([len(set(extended[q]) & queries[q][1]) /
                                 len(queries[q][1]) for q in all_ids]))
        print(f"Eval bundle: {len(all_ids)} queries, pool min={min(sizes)} "
              f"mean={np.mean(sizes):.1f} max={max(sizes)}, "
              f"candidate ceiling={ceiling:.4f}", flush=True)
        print("Channel Recall@5 on its own (cached scores):", flush=True)
        for name in sorted(scores):
            coverage = float(np.mean([
                sum(1 for d in extended[q] if d in scores[name].get(q, {})) /
                len(extended[q]) for q in all_ids]))
            print(f"  {name:12s} R@5={channel_metrics(bundle, scores[name])['Recall@5']:.4f}"
                  f"  pool coverage={coverage:.3f}", flush=True)
    return bundle


# --------------------------------------------------------------------------
# Layer 3 + 4 under leave-one-block-out
# --------------------------------------------------------------------------

def lobo_evaluate(bundle, score_overrides=None, view_overrides=None,
                  c=LTR_C, alpha=THRESHOLD_ALPHA):
    """Refit the LTR fusion per held-out block and score its predictions.

    score_overrides/view_overrides replace one channel with freshly computed
    scores; everything else stays on the cached values, so the delta measured
    is the fine-tuned model's alone.
    """
    scores = dict(bundle.scores)
    scores.update(score_overrides or {})
    views = dict(bundle.views)
    views.update(view_overrides or {})

    rows, groups = ltr_features(views, VIEWS, bundle.extended, bundle.all_ids, scores)
    for q in rows:
        rows[q] = np.concatenate([rows[q], bundle.type_rows[q], bundle.cite_rows[q]],
                                 axis=1)

    ranked, proba_map = {}, {}
    for held, held_ids in bundle.blocks.items():
        train = sum((ids for name, ids in bundle.blocks.items() if name != held), [])
        x = np.vstack([rows[q] for q in train])
        y = np.concatenate([[d in bundle.queries[q][1] for d in groups[q]]
                            for q in train]).astype(np.int8)
        scaler = StandardScaler().fit(x)
        model = LogisticRegression(C=c, class_weight="balanced", solver="liblinear",
                                   max_iter=3000, random_state=LTR_RANDOM_STATE)
        model.fit(scaler.transform(x), y)
        for q in held_ids:
            proba = model.predict_proba(scaler.transform(rows[q]))[:, 1]
            order = np.argsort(-proba)
            ranked[q] = [groups[q][i] for i in order]
            proba_map[q] = {groups[q][i]: float(proba[i]) for i in order}

    gold = {q: bundle.queries[q] for q in bundle.all_ids}
    top5, _ = fixed_metrics(ranked, gold)
    predictions = dynamic_threshold(ranked, proba_map, bundle.all_ids, alpha)
    official = official_metrics(predictions, gold)
    per_block = {}
    for name, ids in bundle.blocks.items():
        block_gold = {q: bundle.queries[q] for q in ids}
        bm, _ = fixed_metrics(ranked, block_gold)
        bo = official_metrics(predictions, block_gold)
        per_block[name] = {"Recall@5": bm["Recall@5"], "F2@5": bm["F2@5"],
                           "recall": bo["recall"], "precision": bo["precision"],
                           "f2": bo["f2"]}
    return {
        "Recall@5": top5["Recall@5"], "Precision@5": top5["Precision@5"],
        "F2@5": top5["F2@5"], "nDCG@10": top5["nDCG@10"], "MRR@10": top5["MRR@10"],
        "recall": official["recall"], "precision": official["precision"],
        "f2": official["f2"], "mean_answers": official["mean_answers"],
        "blocks": per_block,
    }, ranked, predictions


def channel_metrics(bundle, channel_scores):
    """Recall@5 of one channel on its own -- a fast sanity signal per epoch."""
    ranked = rank_by(bundle.extended, channel_scores)
    m, _ = fixed_metrics(ranked, {q: bundle.queries[q] for q in bundle.all_ids})
    return {"Recall@5": m["Recall@5"], "nDCG@10": m["nDCG@10"], "MRR@10": m["MRR@10"]}


# --------------------------------------------------------------------------
# Training pool -- labelled queries with cached retrieval, disjoint from eval
# --------------------------------------------------------------------------

@dataclass
class TrainExample:
    qid: str
    question: str
    positives: list
    negatives: list


def build_train_pool(root=ROOT, exclude=(), negatives=48, limit=None, seed=2026,
                     documents=None, verbose=True):
    """(query, gold, hard negatives) triples mined from the cached BM25 pools.

    The negatives are exactly the documents the deployed retrieval ranks highest
    and gets wrong, so no retrieval is re-run and the model trains against the
    distractors it will actually meet at inference.
    """
    root = Path(root)
    queries = load_queries(root)
    qids = list(queries)
    exclude = set(exclude)
    retrieval = load_retrieval_cache(root)
    corpus = set(documents.ids()) if documents is not None else None

    train_ids = [q for q in qids if q in retrieval and q not in exclude]
    rng = np.random.default_rng(seed)
    if limit is not None and limit < len(train_ids):
        keep = rng.choice(len(train_ids), size=limit, replace=False)
        train_ids = [train_ids[i] for i in sorted(keep)]

    examples, skipped = [], 0
    for q in train_ids:
        question, gold = queries[q]
        pool = raw_union(retrieval[q])
        if corpus is not None:
            gold = {d for d in gold if d in corpus}
            pool = [d for d in pool if d in corpus]
        if not gold:
            skipped += 1
            continue
        negs = [d for d in pool if d not in gold][:negatives]
        if not negs:
            skipped += 1
            continue
        examples.append(TrainExample(q, question, sorted(gold), negs))
    if verbose:
        hit = float(np.mean([any(d in e.positives for d in
                                 raw_union(retrieval[e.qid])[:20])
                             for e in examples])) if examples else 0.0
        print(f"Train pool: {len(examples)} queries "
              f"({skipped} skipped), {negatives} hard negatives each, "
              f"gold-in-top20 of the lexical pool = {hit:.3f}", flush=True)
    return examples


# --------------------------------------------------------------------------
# Result bookkeeping
# --------------------------------------------------------------------------

class RunRecorder:
    """Writes checkpoints, rankings and a history JSON under WORK/<tag>/.

    "Best" is the best *epoch of this run*, measured on the 600-query LOBO
    holdout -- not "better than the cached baseline".  The two are separate
    questions and the code answers both: the first trained epoch always writes
    a checkpoint, every later epoch replaces it only if validation improves,
    and each history row carries `beats_baseline` so it stays visible whether
    the fine-tune is actually ahead of the shipped cache.

    The reason to keep them apart: the cached baseline for a channel may come
    from a checkpoint this package does not ship (jina), so a run can be
    training perfectly well and still sit below the cache for all three epochs.
    Tying the save to the baseline would then throw away the only weights the
    run produced.
    """

    def __init__(self, work, tag, baseline, select=("recall", "f2"), resume=True):
        self.dir = Path(work) / tag
        self.dir.mkdir(parents=True, exist_ok=True)
        self.tag = tag
        self.select = select
        self.baseline = baseline
        self.baseline_key = self._key(baseline)
        self.best_key = None            # None = no epoch has been scored yet
        self.best_epoch = None
        self.history = [{"epoch": "baseline", "note": "all channels from results/",
                         **dict(baseline)}]
        self.done_epochs = []
        if resume:
            self._resume()
        self._write_history()

    def _resume(self):
        """Pick up a history.json left by an interrupted run in the same folder.

        A Kaggle session that dies mid-run leaves whatever epochs finished on
        disk; re-running should skip those rather than redo their GPU time.
        Only the recorded metrics come back -- the weights come back separately
        via load_best_state(), and the optimizer state does not, so a resumed
        epoch restarts its optimizer momentum.
        """
        path = self.dir / "history.json"
        if not path.exists():
            return
        try:
            saved = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as error:
            print(f"[{self.tag}] ignoring unreadable history.json: {error}", flush=True)
            return
        rows = saved.get("history", [])
        if not rows or list(saved.get("select", [])) != list(self.select):
            return
        self.history = rows
        self.best_epoch = saved.get("best_epoch")
        for row in rows:
            if row.get("improved") and all(name in row for name in self.select):
                key = self._key(row)
                self.best_key = key if self.best_key is None else max(self.best_key, key)
        self.done_epochs = sorted({row["epoch"] for row in rows
                                   if isinstance(row.get("epoch"), int)})
        best = f"{self.best_key[0]:.4f}" if self.best_key else "chưa có"
        print(f"[{self.tag}] resuming: epochs {self.done_epochs or '-'} already "
              f"recorded, best_epoch={self.best_epoch}, "
              f"best {self.select[0]}={best}", flush=True)

    def already_done(self, epoch):
        return epoch in self.done_epochs

    def _key(self, m):
        return tuple(round(float(m[name]), 6) for name in self.select)

    def consider(self, epoch, metrics, state_dict, ranked, predictions,
                 extra=None, always_write_ranking=False):
        key = self._key(metrics)
        beats_baseline = key > self.baseline_key
        # Epoch 0 is the untrained reference run (--eval-before-training). It is
        # worth recording, but its weights are the stock ones, already on the
        # hub, so it never claims the checkpoint slot or the bar.
        reference = epoch == 0
        improved = not reference and (self.best_key is None or key > self.best_key)
        row = {"epoch": epoch, "improved": improved,
               "beats_baseline": beats_baseline, **dict(metrics)}
        if extra:
            row.update(extra)
        self.history.append(row)
        if isinstance(epoch, int) and epoch not in self.done_epochs:
            self.done_epochs.append(epoch)
        if improved:
            self.best_key, self.best_epoch = key, epoch
            self._save_state(state_dict)
            self._save_rankings(ranked, predictions, epoch)
        elif always_write_ranking or reference:
            self._save_rankings(ranked, predictions, epoch, suffix=f"_epoch{epoch}")
        self._write_history()
        delta = metrics[self.select[0]] - self.baseline[self.select[0]]
        against = "vượt baseline cache" if beats_baseline else "dưới baseline cache"
        if reference:
            verdict = "reference (không lưu trọng số gốc)"
        elif improved:
            verdict = f"SAVED (tốt nhất tới giờ, {against})"
        else:
            verdict = f"not saved (epoch {self.best_epoch} vẫn tốt hơn)"
        print(f"[{self.tag}] epoch {epoch}: {self.select[0]}={metrics[self.select[0]]:.4f} "
              f"({delta:+.4f} vs cached baseline) f2={metrics['f2']:.4f} "
              f"-> {verdict}", flush=True)
        return improved

    def _save_state(self, state_dict):
        """Always write a checkpoint the shipped runner can load unchanged.

        DataParallel prefixes every key with "module.", which would silently
        match nothing under the runner's strict=False load and leave the stock
        weights in place.  Stripping it here means a checkpoint from a 2-GPU
        run is byte-compatible with a 1-GPU one.
        """
        import torch
        path = self.dir / "best_state.pt"
        prefix = "module."
        torch.save({"state_dict": {
            (k[len(prefix):] if k.startswith(prefix) else k): v.detach().to("cpu")
            for k, v in state_dict.items()}}, path)
        print(f"  saved {path}", flush=True)

    def _save_rankings(self, ranked, predictions, epoch, suffix=""):
        (self.dir / f"best_ranking{suffix}.json").write_text(json.dumps(
            {q: ranked[q][:20] for q in ranked}, ensure_ascii=False), encoding="utf-8")
        (self.dir / f"best_predictions{suffix}.json").write_text(json.dumps(
            {q: {"answer": predictions[q]} for q in predictions},
            ensure_ascii=False, indent=2), encoding="utf-8")

    def save_channel_scores(self, scores, name="best_channel_scores.pkl"):
        (self.dir / name).write_bytes(pickle.dumps(scores, protocol=5))

    def load_best_state(self, model):
        """Restore the best weights this folder holds, if any.

        Called before training so a re-run after an interruption continues from
        the best epoch instead of from the stock weights.
        """
        import torch
        path = self.dir / "best_state.pt"
        if not path.exists():
            return False
        saved = torch.load(path, map_location="cpu", weights_only=True)
        target = getattr(model, "module", model)     # unwrap DataParallel
        missing, unexpected = target.load_state_dict(saved["state_dict"], strict=False)
        print(f"[{self.tag}] restored {path} "
              f"(missing={len(missing)} unexpected={len(unexpected)})", flush=True)
        return True

    def package(self, name=None):
        """Zip everything except the checkpoint into one small downloadable file.

        best_state.pt is left out on purpose: it is gigabytes, and the metrics,
        rankings, predictions and channel scores are what you want off the
        machine first if a session is about to be cut off.
        """
        import zipfile
        target = Path(self.dir).parent / (name or f"{self.tag}_results.zip")
        with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(self.dir.iterdir()):
                if path.is_file() and path.suffix != ".pt":
                    archive.write(path, arcname=f"{self.tag}/{path.name}")
        size = target.stat().st_size / 2**20
        print(f"[{self.tag}] packaged {target} ({size:.1f} MB)", flush=True)
        return target

    def _write_history(self):
        (self.dir / "history.json").write_text(json.dumps(
            {"tag": self.tag, "select": list(self.select),
             "baseline": self.baseline, "best_epoch": self.best_epoch,
             "best_beats_baseline": bool(self.best_key and
                                         self.best_key > self.baseline_key),
             "history": self.history}, ensure_ascii=False, indent=2), encoding="utf-8")


def common_args(description):
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--root", type=Path, default=ROOT,
                    help="read-only root holding results/ and DSC2026-LegalIR-main/ "
                         f"(default: {ROOT})")
    ap.add_argument("--work", type=Path, default=WORK,
                    help=f"writable output dir for checkpoints and rankings "
                         f"(default: {WORK})")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--batch-size", type=int, default=4,
                    help="queries per step (each carries 1 positive + negatives)")
    ap.add_argument("--accum", type=int, default=4, help="gradient accumulation steps")
    ap.add_argument("--negatives", type=int, default=7,
                    help="hard negatives sampled per positive per step")
    ap.add_argument("--negative-depth", type=int, default=48,
                    help="how deep into the cached lexical pool negatives are mined")
    ap.add_argument("--max-length", type=int, default=MAX_LENGTH)
    ap.add_argument("--train-queries", type=int, default=None,
                    help="cap the training pool (default: all cached, non-eval)")
    ap.add_argument("--warmup-ratio", type=float, default=.1)
    ap.add_argument("--weight-decay", type=float, default=.01)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--eval-batch-size", type=int, default=32)
    ap.add_argument("--max-gpus", type=int, default=0,
                    help="cap the GPUs used (0 = all). DataParallel keeps a full "
                         "extra gradient buffer plus a model replica on GPU 0, so "
                         "1 is often the only way a big model fits at all")
    ap.add_argument("--train-top-layers", type=int, default=0,
                    help="train only the top N transformer blocks plus the head "
                         "and freeze the rest (0 = train everything). With 1,050 "
                         "training queries a ~570M model is over-parameterised, "
                         "and every frozen tensor is one fewer place to diverge")
    ap.add_argument("--precision", choices=("auto", "fp16", "bf16", "fp32"),
                    default="auto",
                    help="'auto' probes one batch and drops to fp32 if the "
                         "autocast dtype overflows on this model (a T4 has no "
                         "bf16, and bf16-trained models often blow past fp16's "
                         "65504 ceiling); fp32 is ~2x slower but always safe")
    ap.add_argument("--gradient-checkpointing", action="store_true", default=True)
    ap.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing",
                    action="store_false")
    ap.add_argument("--eval-before-training", action="store_true",
                    help="also score the un-finetuned model as epoch 0")
    ap.add_argument("--keep-every-epoch", action="store_true",
                    help="write rankings for every epoch, not just improvements")
    ap.add_argument("--no-resume", dest="resume", action="store_false", default=True,
                    help="ignore any history.json/best_state.pt already in --work "
                         "and start the run from scratch")
    return ap
