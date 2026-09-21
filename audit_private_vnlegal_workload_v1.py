#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import pickle
import re
import statistics
import sys
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

import numpy as np


# Exact Stage-5 candidate-generation constants.
EXPANSION_DEPTH = 50
EXPANDED_DEPTH = 20
EXPANSION_WEIGHTS = (0.55, 0.45)
EXPANSION_RRF_K = 10
CORPUS_DEPTH = 20

SPACE_RE = re.compile(r"\S+", re.UNICODE)
STOPWORDS = {
    "bị", "các", "có", "của", "cho", "được", "để", "đến", "đối", "gì",
    "hay", "khi", "không", "là", "làm", "một", "nào", "những", "như",
    "phải", "ra", "sẽ", "theo", "thì", "thế", "trong", "trên", "từ",
    "và", "về", "với", "việc", "bao", "nhiêu", "người", "quy", "định",
}
SLUG_ID_RE = re.compile(r"-\d+$")


def load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def percentile(values, p):
    return float(np.percentile(np.asarray(values, dtype=np.float64), p))


def statline(values):
    values = list(values)
    return {
        "min": int(min(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p90": percentile(values, 90),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
        "max": int(max(values)),
    }


def fmt_stats(name, values):
    s = statline(values)
    print(
        f"{name:<28} min={s['min']:>4} mean={s['mean']:>7.2f} "
        f"p50={s['median']:>6.1f} p90={s['p90']:>6.1f} "
        f"p95={s['p95']:>6.1f} p99={s['p99']:>6.1f} max={s['max']:>4}"
    )


def load_questions(path: Path):
    obj = json.loads(path.read_text(encoding="utf-8"))

    if isinstance(obj, dict):
        # Common forms: {qid: question} or {qid: {question: ...}}
        ids = []
        q = {}
        for k, v in obj.items():
            qid = str(k)
            if isinstance(v, str):
                text = v
            elif isinstance(v, dict):
                text = (
                    v.get("question")
                    or v.get("query")
                    or v.get("text")
                    or v.get("content")
                )
            else:
                text = None
            if text is None:
                continue
            ids.append(qid)
            q[qid] = str(text)
        if ids:
            return ids, q

    if isinstance(obj, list):
        ids = []
        q = {}
        for row in obj:
            if not isinstance(row, dict):
                continue
            qid = row.get("id", row.get("qid", row.get("query_id")))
            text = (
                row.get("question")
                or row.get("query")
                or row.get("text")
                or row.get("content")
            )
            if qid is None or text is None:
                continue
            qid = str(qid)
            ids.append(qid)
            q[qid] = str(text)
        if ids:
            return ids, q

    raise RuntimeError(f"Unsupported private question schema: {path}")


def title_from_link(link):
    if not link:
        return ""
    slug = urlparse(link).path.rsplit("/", 1)[-1]
    slug = re.sub(r"\.aspx$", "", slug, flags=re.I)
    slug = SLUG_ID_RE.sub("", slug)
    return slug.replace("-", " ").strip()


class DocumentStore:
    def __init__(self, selected_contexts: Path):
        paths = sorted(selected_contexts.glob("context_*.json"))
        if not paths:
            raise FileNotFoundError(f"No context_*.json under {selected_contexts}")
        self.paths = {
            p.stem[len("context_"):]: p
            for p in paths
        }
        self.cache = {}

    def row(self, doc):
        doc = str(doc)
        if doc not in self.cache:
            p = self.paths.get(doc)
            if p is None:
                self.cache[doc] = {}
            else:
                self.cache[doc] = json.loads(p.read_text(encoding="utf-8"))
        return self.cache[doc]

    def vnlegal_text(self, doc):
        row = self.row(doc)
        return row.get("passage") or title_from_link(row.get("link"))

    def e5_text(self, doc):
        row = self.row(doc)
        return row.get("passage") or ""


def simple_tokens(text):
    # benchmark_burst_v4_full_sqlite.tokens is better if importable.
    # This fallback is used only for passage selection if the repo import fails.
    return re.findall(r"\w+", (text or "").lower(), flags=re.UNICODE)


def make_top_passages(root: Path):
    sys.path.insert(0, str(root))
    try:
        from benchmark_jina_reranker_holdouts import top_passages
        return top_passages
    except Exception as exc:
        print(
            f"WARNING: could not import repository top_passages ({exc}); "
            "using embedded equivalent with simplified tokenizer."
        )

    def top_passages(question, text, count=2, window=220, overlap=70):
        words = SPACE_RE.findall(text or "")
        if len(words) <= window + 80:
            return [" ".join(words)]
        query_tokens = simple_tokens(question)
        content = {
            t for t in query_tokens
            if len(t) >= 3 and t not in STOPWORDS
        }
        numbers = {
            t for t in query_tokens
            if any(c.isdigit() for c in t)
        }
        bigrams = {
            " ".join(query_tokens[i:i + 2])
            for i in range(len(query_tokens) - 1)
        }
        header = " ".join(words[:70])
        scored = []
        step = window - overlap
        for start in range(0, len(words), step):
            end = min(start + window, len(words))
            part_words = words[start:end]
            part = " ".join(part_words)
            normalized = simple_tokens(part)
            token_set = set(normalized)
            norm_text = " ".join(normalized)
            coverage = sum(
                1.0 + .20 * min(normalized.count(t), 3)
                for t in content if t in token_set
            )
            numeric = 3.0 * sum(t in token_set for t in numbers)
            phrase = 1.8 * sum(p in norm_text for p in bigrams)
            density = (
                coverage + numeric + phrase
            ) / math.sqrt(max(len(normalized), 1))
            scored.append(
                (density, coverage + numeric + phrase, -start, part)
            )
            if end == len(words):
                break
        scored.sort(reverse=True)
        passages = []
        for _, _, neg_start, part in scored:
            candidate = (
                part
                if -neg_start < 70
                else header + "\n[ĐOẠN PHÙ HỢP]\n" + part
            )
            if candidate not in passages:
                passages.append(candidate)
            if len(passages) >= count:
                break
        return passages

    return top_passages


def raw_union(lists, depth):
    seen = set()
    out = []
    # retrieval[q] historically is a list/tuple of ranking views.
    if isinstance(lists, dict):
        iterable = lists.values()
    else:
        iterable = lists
    for ranking in iterable:
        if isinstance(ranking, dict):
            ranking = list(ranking)
        for d in list(ranking)[:depth]:
            d = str(d)
            if d not in seen:
                seen.add(d)
                out.append(d)
    return out


def weighted_rrf_two(a, b, weights=(0.55, 0.45), k=10):
    score = {}
    for ranking, w in ((a, weights[0]), (b, weights[1])):
        for rank, doc in enumerate(ranking, 1):
            score[doc] = score.get(doc, 0.0) + w / (k + rank)
    # Historical weighted_rrf uses score descending. doc ID tie-break keeps
    # this deterministic; exact ties are rare.
    return sorted(score, key=lambda d: (-score[d], d))


def reconstruct_candidates(root: Path, ids):
    exact_cache = (
        root
        / "results/manual/huy_private_d1_rel_l0_exact_v1/cache"
    )
    stage5 = (
        root
        / "results/manual/huy_private_d1_rel_l0_approx_v1/"
          "cache/candidate_generation"
    )

    retrieval_path = exact_cache / "private_retrieval.pkl"
    base_path = exact_cache / "private_base_top20_historical_exact.pkl"
    expansion_path = stage5 / "expansion_scores.pkl"
    corpus_path = stage5 / "corpus_rank_cap32.pkl"

    required = [
        retrieval_path, base_path, expansion_path, corpus_path
    ]
    missing = [p for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing cache(s):\n  " + "\n  ".join(map(str, missing))
        )

    retrieval_obj = load_pickle(retrieval_path)
    retrieval = retrieval_obj.get("cache", retrieval_obj)

    base_obj = load_pickle(base_path)
    base = base_obj.get("rankings", base_obj)

    expansion_obj = load_pickle(expansion_path)
    expansion_scores = expansion_obj.get("scores", expansion_obj)

    corpus_obj = load_pickle(corpus_path)
    corpus_rank = corpus_obj.get("ranking", corpus_obj)

    for q in ids:
        for label, mapping in (
            ("retrieval", retrieval),
            ("base", base),
            ("expansion", expansion_scores),
            ("corpus", corpus_rank),
        ):
            if q not in mapping:
                raise RuntimeError(f"{label} cache missing qid={q}")

    expanded = {}
    candidates = {}

    for q in ids:
        raw = raw_union(retrieval[q], EXPANSION_DEPTH)
        dense_rank = sorted(
            raw,
            key=lambda d: (
                -float(expansion_scores[q].get(d, -1e30)),
                d,
            ),
        )
        expanded_all = weighted_rrf_two(
            raw, dense_rank, EXPANSION_WEIGHTS, EXPANSION_RRF_K
        )
        expanded[q] = expanded_all[:EXPANDED_DEPTH]

        candidates[q] = list(dict.fromkeys(
            [str(d) for d in base[q][:20]]
            + [str(d) for d in expanded[q]]
            + [str(d) for d in corpus_rank[q][:CORPUS_DEPTH]]
        ))

    return base, expanded, corpus_rank, candidates


def passage_workload(ids, questions, base, candidates, docs, top_passages):
    e5_counts = {}
    vn_counts = {}
    vn_passages = {}

    for i, q in enumerate(ids, 1):
        question = questions[q]

        ec = 0
        for d in base[q][:20]:
            ec += len(
                top_passages(
                    question,
                    docs.e5_text(d),
                    count=2,
                )
            )
        e5_counts[q] = ec

        plist = []
        for d in candidates[q]:
            plist.extend(
                top_passages(
                    question,
                    docs.vnlegal_text(d),
                    count=2,
                )
            )
        vn_counts[q] = len(plist)
        vn_passages[q] = plist

        if i % 250 == 0 or i == len(ids):
            print(f"  passage scan {i}/{len(ids)}", flush=True)

    return e5_counts, vn_counts, vn_passages


def forward_stats(counts, batch):
    vals = list(counts.values())
    batches = [math.ceil(v / batch) for v in vals]
    return {
        "batch": batch,
        "total_passage_forwards": int(sum(batches)),
        "mean_passage_forwards_per_query": float(np.mean(batches)),
        "queries_1_forward": int(sum(x <= batch for x in vals)),
        "queries_2plus": int(sum(x > batch for x in vals)),
        "queries_3plus": int(sum(x > 2 * batch for x in vals)),
    }


def tokenizer_probe(
    root: Path,
    ids,
    vn_passages,
    sample_n,
    batches,
):
    if sample_n <= 0:
        return None

    model_dir = root / "models/vnlegal-lal"
    if not model_dir.is_dir():
        print(f"Tokenizer probe skipped: missing {model_dir}")
        return None

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_dir)

    if sample_n >= len(ids):
        sample_ids = list(ids)
    else:
        idx = np.linspace(0, len(ids) - 1, sample_n, dtype=int)
        sample_ids = [ids[int(i)] for i in idx]

    lengths_by_q = {}
    all_lengths = []

    print(
        f"\nTokenizing vnlegal passage sample: "
        f"{len(sample_ids)} queries on CPU...",
        flush=True,
    )
    for i, q in enumerate(sample_ids, 1):
        passages = vn_passages[q]
        enc = tok(
            passages,
            add_special_tokens=True,
            truncation=True,
            max_length=512,
            padding=False,
            return_length=True,
        )
        lengths = enc["length"]
        if isinstance(lengths, np.ndarray):
            lengths = lengths.tolist()
        lengths = [int(x) for x in lengths]
        lengths_by_q[q] = lengths
        all_lengths.extend(lengths)

        if i % 50 == 0 or i == len(sample_ids):
            print(f"  tokenizer {i}/{len(sample_ids)}", flush=True)

    out = {
        "sample_queries": len(sample_ids),
        "passage_token_stats": statline(all_lengths),
        "batch_padding": {},
    }

    for bs in batches:
        true_tokens = 0
        padded_tokens = 0
        forward_batches = 0
        for q in sample_ids:
            lens = lengths_by_q[q]
            true_tokens += sum(lens)
            for st in range(0, len(lens), bs):
                chunk = lens[st:st + bs]
                forward_batches += 1
                padded_tokens += max(chunk) * len(chunk)

        out["batch_padding"][bs] = {
            "forward_batches": forward_batches,
            "true_tokens": true_tokens,
            "padded_token_positions": padded_tokens,
            "padding_multiplier": (
                padded_tokens / true_tokens if true_tokens else None
            ),
        }

    return out


def main():
    ap = argparse.ArgumentParser(
        description=(
            "CPU-only audit of private D1 candidate/passage workload. "
            "Does not load neural model weights or modify caches."
        )
    )
    ap.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
    )
    ap.add_argument(
        "--tokenize-sample",
        type=int,
        default=200,
        help=(
            "Number of evenly spread queries for CPU tokenizer length/padding "
            "analysis. 0 disables tokenization."
        ),
    )
    args = ap.parse_args()

    root = args.repo_root.resolve()
    data = (
        root
        / "DSC2026-LegalIR-main/v4_run/public_test_dataset"
    )
    private_path = data / "private-official.json"
    selected = data / "selected-contexts"

    ids, questions = load_questions(private_path)
    print(f"Private queries: {len(ids)}")

    base, expanded, corpus_rank, candidates = reconstruct_candidates(
        root, ids
    )

    cand_sizes = [len(candidates[q]) for q in ids]
    base_sizes = [len(base[q][:20]) for q in ids]

    print("\n=== Candidate pool ===")
    fmt_stats("Historical E5 base docs/q", base_sizes)
    fmt_stats("D1 candidate docs/q", cand_sizes)

    # Provenance contribution counts.
    new_from_expanded = []
    new_from_corpus = []
    for q in ids:
        b = set(map(str, base[q][:20]))
        e = [str(d) for d in expanded[q]]
        c = [str(d) for d in corpus_rank[q][:20]]
        new_from_expanded.append(len([d for d in e if d not in b]))
        be = b | set(e)
        new_from_corpus.append(len([d for d in c if d not in be]))

    fmt_stats("Novel expanded docs/q", new_from_expanded)
    fmt_stats("Novel corpus docs/q", new_from_corpus)

    docs = DocumentStore(selected)
    top_passages = make_top_passages(root)

    print("\nScanning exact top_passages(count=2) workload...")
    e5_counts, vn_counts, vn_passages = passage_workload(
        ids, questions, base, candidates, docs, top_passages
    )

    print("\n=== Actual passage workload ===")
    fmt_stats("E5 passages/query", e5_counts.values())
    fmt_stats("vnlegal passages/query", vn_counts.values())

    total_e5_passages = sum(e5_counts.values())
    total_vn_passages = sum(vn_counts.values())
    print(f"\nE5 total passages:       {total_e5_passages:,}")
    print(f"vnlegal total passages:  {total_vn_passages:,}")
    print(f"Passage-count multiplier: {total_vn_passages/total_e5_passages:.2f}x")

    print("\n=== Passage forward-pass count (query encode NOT included) ===")
    for bs in (32, 48, 64, 80, 96, 128):
        s = forward_stats(vn_counts, bs)
        print(
            f"batch={bs:>3}: forwards={s['total_passage_forwards']:>5} "
            f"mean/q={s['mean_passage_forwards_per_query']:.3f} "
            f"one-forward={s['queries_1_forward']:>4}/{len(ids)} "
            f">1={s['queries_2plus']:>4} >2={s['queries_3plus']:>4}"
        )

    print(
        "\nRemember: vnlegal also performs one separate query forward per query "
        f"= {len(ids):,} additional batch-1 forwards."
    )

    # Distribution around batch boundaries.
    v = np.asarray(list(vn_counts.values()))
    print("\n=== Where passage counts sit relative to batches ===")
    for threshold in (32, 48, 64, 80, 96, 128, 160, 192):
        print(
            f"<= {threshold:>3} passages: "
            f"{int(np.sum(v <= threshold)):>4}/{len(v)} "
            f"({100*np.mean(v <= threshold):5.1f}%)"
        )

    tok = tokenizer_probe(
        root,
        ids,
        vn_passages,
        args.tokenize_sample,
        batches=(64, 80, 96, 128),
    )

    if tok:
        print("\n=== vnlegal tokenizer sample ===")
        s = tok["passage_token_stats"]
        print(
            "Passage token length: "
            f"min={s['min']} mean={s['mean']:.1f} "
            f"p50={s['median']:.1f} p90={s['p90']:.1f} "
            f"p95={s['p95']:.1f} p99={s['p99']:.1f} max={s['max']}"
        )
        print(
            f"Sample queries: {tok['sample_queries']} "
            "(evenly spread across private set)"
        )
        for bs, row in tok["batch_padding"].items():
            print(
                f"batch={bs:>3}: forwards={row['forward_batches']:>4} "
                f"padding-multiplier={row['padding_multiplier']:.3f}x "
                f"padded-token-positions={row['padded_token_positions']:,}"
            )

    # Save a compact JSON report next to script invocation location.
    report = {
        "queries": len(ids),
        "candidate_docs": statline(cand_sizes),
        "e5_passages": statline(e5_counts.values()),
        "vnlegal_passages": statline(vn_counts.values()),
        "total_e5_passages": total_e5_passages,
        "total_vnlegal_passages": total_vn_passages,
        "passage_multiplier": total_vn_passages / total_e5_passages,
        "vnlegal_forward_stats": {
            str(bs): forward_stats(vn_counts, bs)
            for bs in (32, 48, 64, 80, 96, 128)
        },
        "tokenizer_probe": tok,
    }
    report_path = Path("private_workload_audit.json").resolve()
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nSaved report: {report_path}")


if __name__ == "__main__":
    main()
