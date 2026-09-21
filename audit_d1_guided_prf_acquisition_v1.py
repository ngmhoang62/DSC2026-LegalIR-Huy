#!/usr/bin/env python
"""
D1-GUIDED PSEUDO-RELEVANCE FEEDBACK (PRF) — ACQUISITION AUDIT
==============================================================

CPU-only. No private labels. No neural inference.

Core hypothesis
---------------
The current D1 selector may be near saturation, but the D1 candidate pool still
has an acquisition ceiling: some gold documents never enter the pool.

Use D1's own high-confidence top documents as pseudo relevance feedback:

    query
      -> exact OOF D1 ranking
      -> top-3 / top-5 predicted documents
      -> extract corpus-discriminative legal terms
      -> expand the query
      -> FTS5 retrieval over all 8,532 documents
      -> ask whether OUTSIDE-POOL gold documents are rescued

This is fundamentally different from:
  * adding BM25/trigram rank views to D1;
  * query bootstrap/memory;
  * raw section expansion;
  * selector/boundary swaps.

The first gate is ACQUISITION ONLY. We do NOT force PRF candidates into Top-5.

Feedback term scoring
---------------------
For each pseudo-relevant document:
  - tokenize with the corpus unicode tokenizer;
  - rank-weight by D1 position;
  - score candidate terms by:
        IDF(term) * support_fraction * weighted_log_tf
  - exclude original query terms, boilerplate/stop terms,
    numbers, ultra-common terms, and corpus singletons.

Predeclared configs
-------------------
T3_COMMON12 : D1 top3, term must occur in >=2 feedback docs, add 12 terms
T3_BROAD12  : D1 top3, term may occur in >=1 feedback doc, add 12 terms
T5_COMMON16 : D1 top5, term must occur in >=2 feedback docs, add 16 terms
T5_BROAD16  : D1 top5, term may occur in >=2 feedback docs, add 16 terms
              (top5 broad still requires support>=2 to limit drift)

Retrieval branches per config
-----------------------------
DOC_EXPANDED  : document-level FTS on original query + PRF terms
LOCAL_EXPANDED: chunk-level FTS on original query + PRF terms
DOC_PRF_ONLY  : document-level FTS using only PRF terms

For acquisition evaluation, we inspect novel documents after removing anything
already in the authoritative D1 candidate pool.

Mandatory parity
----------------
Exact OOF D1 Recall@5 = 0.9569444444444444

Outputs
-------
results/manual/huy_d1_guided_prf_acquisition_v1/
  REPORT.json
  RESCUE_CASES.json
  PROMISING_CONFIG.json  (only if acquisition gate passes)

Gate (acquisition, not final submission)
----------------------------------------
A config/branch is "promising" if at novel depth <=20:
  - rescues >=3 outside-pool missed-gold occurrences;
  - rescues span >=2 CAL blocks;
  - candidate oracle gain >= +0.003;
  - median novel docs/query at that depth <=20 (by construction).

Strong signal:
  >=4 outside-pool occurrences rescued at depth <=20.

A pass only authorizes a second-stage selector/reranker experiment.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sqlite3
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


EXPECTED_D1 = 0.9569444444444444
SEED = 2026
D1_VIEWS = ["base", "expanded", "jina", "dense", "corpus"]
NOVEL_DEPTHS = [5, 10, 20, 50]

CONFIGS = {
    "T3_COMMON12": {
        "feedback_k": 3,
        "min_support": 2,
        "terms": 12,
    },
    "T3_BROAD12": {
        "feedback_k": 3,
        "min_support": 1,
        "terms": 12,
    },
    "T5_COMMON16": {
        "feedback_k": 5,
        "min_support": 2,
        "terms": 16,
    },
    "T5_BROAD16": {
        "feedback_k": 5,
        "min_support": 2,
        "terms": 16,
    },
}

STOP = {
    # Vietnamese function words
    "và","là","có","của","được","cho","trong","theo","với","thì","khi",
    "các","một","những","này","đó","về","để","từ","tại","hay","như","nào",
    "gì","bao","nhiêu","mà","do","bởi","trên","dưới","giữa","sau","trước",
    "đến","ra","vào","nếu","hoặc","cũng","đang","sẽ","đã","bị","phải",
    "không","chưa","nên","cần","việc","đối","đối với",
    # legal boilerplate that is often too broad for PRF
    "điều","khoản","điểm","chương","mục","luật","quy","định","nghị",
    "định","thông","tư","quyết","nghị quyết","văn","bản","pháp","luật",
    "cơ","quan","nhà","nước","việt","nam","trường","hợp","thực","hiện",
    "theo","quyền","trách","nhiệm",
}

TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def tokens(text):
    return TOKEN_RE.findall((text or "").lower())


def clean_term(t):
    if len(t) < 3:
        return False
    if t in STOP:
        return False
    if t.isdigit():
        return False
    if all(ch == "_" for ch in t):
        return False
    return True


def fts_expr(term_list):
    seen = set()
    parts = []
    for t in term_list:
        if not t or t in seen:
            continue
        seen.add(t)
        parts.append('"' + t.replace('"', '""') + '"')
    return " OR ".join(parts)


def load_context_text(path: Path):
    row = json.loads(path.read_text(encoding="utf-8"))
    text = row.get("passage") or ""
    if not text:
        link = row.get("link") or ""
        slug = link.rsplit("/", 1)[-1]
        slug = re.sub(r"\.aspx$", "", slug, flags=re.I)
        slug = re.sub(r"-\d+$", "", slug)
        text = slug.replace("-", " ")
    return text


def build_doc_paths(context_dir: Path):
    out = {}
    for p in context_dir.glob("context_*.json"):
        did = p.stem[len("context_"):]
        out[str(did)] = p
    return out


def build_df(doc_paths):
    """
    Stream corpus once; retain only document frequency Counter.
    Avoid keeping the ~hundreds-MB corpus text in RAM.
    """
    df = Counter()
    total_tokens = 0
    for i, (did, p) in enumerate(sorted(doc_paths.items()), 1):
        ts = set(t for t in tokens(load_context_text(p)) if clean_term(t))
        df.update(ts)
        total_tokens += len(ts)
        if i % 1000 == 0:
            print(f"    DF scanned {i}/{len(doc_paths)} docs", flush=True)
    return df


class DocTokenCache:
    def __init__(self, doc_paths):
        self.doc_paths = doc_paths
        self.cache = {}

    def get(self, did):
        did = str(did)
        if did not in self.cache:
            p = self.doc_paths.get(did)
            if p is None:
                raise KeyError(f"Missing selected-context for doc {did}")
            self.cache[did] = tokens(load_context_text(p))
        return self.cache[did]


def build_d1_rows(local_views, full_channels, extended, all_ids, type_rows, cite_rows):
    from tune_expanded_fusion_selection import ltr_features
    rows0, groups = ltr_features(
        local_views, D1_VIEWS, extended, all_ids, full_channels
    )
    rows = {
        q: np.concatenate(
            [rows0[q], type_rows[q], cite_rows[q]], axis=1
        ).astype(np.float32, copy=False)
        for q in all_ids
    }
    for q in all_ids:
        if rows[q].shape[1] != 48:
            raise RuntimeError(f"Expected D1 48D, q={q} got {rows[q].shape}")
    return rows, groups


def fit_d1_lobo(blocks, all_ids, rows, groups, gold):
    pred = {}
    full = {}
    scoremaps = {}

    for held in sorted(blocks):
        train = sum(
            (list(blocks[b]) for b in sorted(blocks) if b != held),
            [],
        )
        test = list(blocks[held])

        X = np.vstack([rows[q] for q in train])
        y = np.concatenate([
            [d in gold[q] for d in groups[q]]
            for q in train
        ]).astype(np.int8)

        scaler = StandardScaler().fit(X)
        model = LogisticRegression(
            C=.15,
            class_weight="balanced",
            solver="liblinear",
            max_iter=3000,
            random_state=SEED,
        ).fit(scaler.transform(X), y)

        for q in test:
            s = np.asarray(
                model.decision_function(scaler.transform(rows[q])),
                dtype=np.float64,
            )
            idx = np.argsort(-s, kind="stable")
            order = [groups[q][i] for i in idx]
            pred[q] = order[:5]
            full[q] = order
            scoremaps[q] = {
                groups[q][i]: float(s[i])
                for i in range(len(groups[q]))
            }

    recalls = []
    for q in all_ids:
        recalls.append(
            len(set(pred[q]) & set(gold[q])) / max(1, len(gold[q]))
        )
    return pred, full, scoremaps, float(np.mean(recalls))


def feedback_terms(
    question,
    feedback_docs,
    token_cache,
    df,
    n_docs,
    min_support,
    n_terms,
):
    qterms = set(t for t in tokens(question) if clean_term(t))

    support = Counter()
    weighted_tf = defaultdict(float)

    for rank, did in enumerate(feedback_docs, 1):
        ts = token_cache.get(did)
        c = Counter(t for t in ts if clean_term(t))
        if not c:
            continue

        # D1 rank-weighted pseudo relevance.
        rw = 1.0 / math.sqrt(rank)
        for t, tf in c.items():
            support[t] += 1
            weighted_tf[t] += rw * math.log1p(tf)

    scored = []
    for t, wtf in weighted_tf.items():
        if t in qterms:
            continue
        sup = support[t]
        if sup < min_support:
            continue

        dfi = int(df.get(t, 0))
        # Reject singletons (over-specific names/noise) and ultra-common terms.
        if dfi < 2:
            continue
        if dfi / n_docs > 0.20:
            continue

        idf = math.log((n_docs + 1.0) / (dfi + 1.0)) + 1.0
        support_frac = sup / max(1, len(feedback_docs))
        score = idf * support_frac * wtf
        scored.append((float(score), t, sup, dfi))

    scored.sort(key=lambda x: (-x[0], x[1]))
    return scored[:n_terms]


def retrieve_doc(conn, expr, doc_ids, limit=100):
    if not expr:
        return []
    rows = conn.execute(
        "SELECT rowid,-bm25(docs_fts) AS score FROM docs_fts "
        "WHERE docs_fts MATCH ? ORDER BY bm25(docs_fts) LIMIT ?",
        (expr, limit),
    ).fetchall()
    return [(doc_ids[int(rowid)-1], float(score)) for rowid, score in rows]


def retrieve_local(conn, expr, doc_ids, limit_rows=1500, second_weight=.3):
    if not expr:
        return []
    rows = conn.execute(
        "SELECT o.doc_idx,-bm25(chunks_fts) AS score FROM chunks_fts "
        "JOIN chunk_owner o ON o.rowid=chunks_fts.rowid "
        "WHERE chunks_fts MATCH ? ORDER BY bm25(chunks_fts) LIMIT ?",
        (expr, limit_rows),
    ).fetchall()

    best = {}
    for doc_idx, score in rows:
        did = doc_ids[int(doc_idx)]
        score = float(score)
        a, b = best.get(did, (0.0, 0.0))
        if score > a:
            a, b = score, a
        elif score > b:
            b = score
        best[did] = (a, b)

    return sorted(
        ((d, a + second_weight*b) for d, (a, b) in best.items()),
        key=lambda x: (-x[1], x[0]),
    )


def novel_rank(ranked_docs, existing_pool):
    return [d for d in ranked_docs if d not in existing_pool]


def candidate_oracle(candidates, gold, ids):
    return float(np.mean([
        len(set(candidates[q]) & set(gold[q])) / max(1, len(gold[q]))
        for q in ids
    ]))


def block_of_map(blocks):
    return {
        q: b
        for b, ids in blocks.items()
        for q in ids
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=Path, required=True)
    ap.add_argument(
        "--fts-db",
        type=Path,
        default=None,
        help="Default: <repo>/benchmarks/legalir_full_fts.sqlite",
    )
    args = ap.parse_args()

    root = args.repo_root.expanduser().resolve()
    sys.path.insert(0, str(root))

    fts_db = (
        args.fts_db.expanduser().resolve()
        if args.fts_db is not None
        else root / "benchmarks/legalir_full_fts.sqlite"
    )
    if not fts_db.is_file():
        raise FileNotFoundError(
            f"Missing FTS DB: {fts_db}\n"
            "This audit intentionally reuses the repo's existing FTS index."
        )

    print("[1/8] Loading authoritative D1 CAL600 world...", flush=True)
    from src.gemini.huy_vnlegal_rank_ablation_v1.evaluate_ablation_cal import (
        load_cal_inputs,
    )
    (
        queries,
        blocks,
        all_ids,
        extended,
        local_views,
        full_channels,
        gold,
        _vnlegal,
        type_rows,
        cite_rows,
    ) = load_cal_inputs()

    print("[2/8] Reconstructing exact OOF D1 ranking...", flush=True)
    rows, groups = build_d1_rows(
        local_views,
        full_channels,
        extended,
        all_ids,
        type_rows,
        cite_rows,
    )
    d1_top5, d1_full, d1_scores, d1_recall = fit_d1_lobo(
        blocks, all_ids, rows, groups, gold
    )
    print(f"  D1 OOF Recall@5={d1_recall:.10f}", flush=True)
    if abs(d1_recall - EXPECTED_D1) > 1e-9:
        raise RuntimeError(
            f"D1 parity failed: {d1_recall} != {EXPECTED_D1}"
        )

    print("[3/8] Auditing current candidate acquisition ceiling...", flush=True)
    base_oracle = candidate_oracle(extended, gold, all_ids)
    outside = []
    for q in all_ids:
        pool = set(extended[q])
        for g in gold[q]:
            if g not in pool:
                outside.append({
                    "qid": q,
                    "gold_doc": g,
                    "block": block_of_map(blocks)[q],
                    "question": queries[q][0],
                    "d1_top5": d1_top5[q],
                })
    print(
        f"  candidate oracle={base_oracle:.10f} "
        f"outside-pool gold occurrences={len(outside)}",
        flush=True,
    )

    print("[4/8] Preparing corpus DF statistics for feedback-term scoring...", flush=True)
    context_dir = (
        root
        / "DSC2026-LegalIR-main/v4_run/public_test_dataset/selected-contexts"
    )
    doc_paths = build_doc_paths(context_dir)
    if len(doc_paths) != 8532:
        raise RuntimeError(
            f"Expected 8532 selected-context docs, got {len(doc_paths)}"
        )
    df = build_df(doc_paths)
    token_cache = DocTokenCache(doc_paths)
    n_docs = len(doc_paths)
    print(f"  DF vocabulary={len(df):,}", flush=True)

    # Exact FTS doc row order must match sorted context files used by DB builder.
    sorted_paths = sorted(context_dir.glob("context_*.json"))
    doc_ids = [
        p.stem[len("context_"):]
        for p in sorted_paths
    ]
    if set(doc_ids) != set(doc_paths):
        raise RuntimeError("doc-id mapping mismatch")

    print("[5/8] Running D1-guided PRF retrieval...", flush=True)
    conn = sqlite3.connect(str(fts_db))
    branches = ["DOC_EXPANDED", "LOCAL_EXPANDED", "DOC_PRF_ONLY"]

    rankings = {
        cfg: {branch: {} for branch in branches}
        for cfg in CONFIGS
    }
    term_trace = {cfg: {} for cfg in CONFIGS}

    t0 = time.perf_counter()

    for iq, q in enumerate(all_ids, 1):
        question = queries[q][0]
        qterms = [t for t in tokens(question) if t]
        pool = set(extended[q])

        for cfg_name, cfg in CONFIGS.items():
            feedback = d1_full[q][:cfg["feedback_k"]]
            chosen = feedback_terms(
                question,
                feedback,
                token_cache,
                df,
                n_docs,
                cfg["min_support"],
                cfg["terms"],
            )
            exp_terms = [t for _, t, _, _ in chosen]

            expr_exp = fts_expr(qterms + exp_terms)
            expr_prf = fts_expr(exp_terms)

            doc_exp = retrieve_doc(
                conn, expr_exp, doc_ids, limit=120
            )
            local_exp = retrieve_local(
                conn, expr_exp, doc_ids, limit_rows=1800, second_weight=.3
            )
            doc_prf = retrieve_doc(
                conn, expr_prf, doc_ids, limit=120
            )

            rankings[cfg_name]["DOC_EXPANDED"][q] = novel_rank(
                [d for d, _ in doc_exp],
                pool,
            )
            rankings[cfg_name]["LOCAL_EXPANDED"][q] = novel_rank(
                [d for d, _ in local_exp],
                pool,
            )
            rankings[cfg_name]["DOC_PRF_ONLY"][q] = novel_rank(
                [d for d, _ in doc_prf],
                pool,
            )

            term_trace[cfg_name][q] = {
                "feedback_docs": list(feedback),
                "terms": [
                    {
                        "term": t,
                        "score": score,
                        "support": sup,
                        "df": dfi,
                    }
                    for score, t, sup, dfi in chosen
                ],
            }

        if iq % 50 == 0:
            print(
                f"  PRF {iq}/{len(all_ids)} "
                f"elapsed={(time.perf_counter()-t0)/60:.1f}m",
                flush=True,
            )

    conn.close()

    print("[6/8] Measuring outside-pool rescue and oracle lift...", flush=True)
    blockmap = block_of_map(blocks)
    results = {}
    rescue_cases = []

    outside_pairs = {(x["qid"], x["gold_doc"]) for x in outside}

    for cfg_name in CONFIGS:
        results[cfg_name] = {}

        for branch in branches:
            by_depth = {}

            for depth in NOVEL_DEPTHS:
                augmented = {}
                rescued = []

                for q in all_ids:
                    novel = rankings[cfg_name][branch][q][:depth]
                    augmented[q] = list(extended[q]) + novel

                    for g in gold[q]:
                        if (q, g) not in outside_pairs:
                            continue
                        if g in novel:
                            rescued.append((q, g))

                oracle = candidate_oracle(
                    augmented, gold, all_ids
                )
                blocks_hit = sorted(set(blockmap[q] for q, _ in rescued))
                multi_rescue = sum(
                    1 for q, g in rescued
                    if len(gold[q]) > 1
                )

                row = {
                    "novel_depth": depth,
                    "rescued_outside_occurrences": len(rescued),
                    "rescued_unique_queries": len(set(q for q, _ in rescued)),
                    "rescued_blocks": blocks_hit,
                    "rescued_block_count": len(blocks_hit),
                    "rescued_multi_gold_occurrences": multi_rescue,
                    "candidate_oracle": oracle,
                    "oracle_gain": oracle - base_oracle,
                    "mean_novel_available": float(np.mean([
                        min(depth, len(rankings[cfg_name][branch][q]))
                        for q in all_ids
                    ])),
                    "median_novel_available": float(np.median([
                        min(depth, len(rankings[cfg_name][branch][q]))
                        for q in all_ids
                    ])),
                    "rescued_pairs": [
                        {"qid": q, "gold_doc": g, "block": blockmap[q]}
                        for q, g in sorted(rescued)
                    ],
                }
                by_depth[str(depth)] = row

                for q, g in rescued:
                    rescue_cases.append({
                        "config": cfg_name,
                        "branch": branch,
                        "depth": depth,
                        "qid": q,
                        "gold_doc": g,
                        "block": blockmap[q],
                        "question": queries[q][0],
                        "d1_top5": d1_top5[q],
                        "feedback_docs": term_trace[cfg_name][q]["feedback_docs"],
                        "feedback_terms": term_trace[cfg_name][q]["terms"],
                        "novel_rank": (
                            rankings[cfg_name][branch][q].index(g) + 1
                        ),
                    })

            results[cfg_name][branch] = by_depth

            d20 = by_depth["20"]
            print(
                f"  {cfg_name:12s} {branch:14s} "
                f"@20 rescue={d20['rescued_outside_occurrences']:2d}/"
                f"{len(outside)} "
                f"blocks={d20['rescued_blocks']} "
                f"oracleΔ={d20['oracle_gain']:+.6f}",
                flush=True,
            )

    print("[7/8] Acquisition gate / plateau check...", flush=True)
    candidates = []

    for cfg_name in CONFIGS:
        for branch in branches:
            for depth in [5, 10, 20]:
                row = results[cfg_name][branch][str(depth)]
                passed = (
                    row["rescued_outside_occurrences"] >= 3
                    and row["rescued_block_count"] >= 2
                    and row["oracle_gain"] >= 0.003 - 1e-12
                )
                strength = (
                    "STRONG"
                    if row["rescued_outside_occurrences"] >= 4 and passed
                    else "PROMISING"
                    if passed
                    else "FAIL"
                )
                if passed:
                    candidates.append({
                        "config": cfg_name,
                        "branch": branch,
                        "depth": depth,
                        "strength": strength,
                        **row,
                    })

    # Consistency: how many fixed config/branch arms rescue >=3 at depth <=20.
    broad_support = sum(
        1
        for cfg_name in CONFIGS
        for branch in branches
        if results[cfg_name][branch]["20"]["rescued_outside_occurrences"] >= 3
    )

    recommended = None
    if candidates:
        recommended = max(
            candidates,
            key=lambda x: (
                x["rescued_outside_occurrences"],
                x["rescued_block_count"],
                x["oracle_gain"],
                -x["depth"],
            ),
        )

    verdict = (
        "PROMISING_D1_GUIDED_PRF_ACQUISITION"
        if recommended is not None
        else "KILL_D1_GUIDED_PRF_ACQUISITION"
    )

    print(
        f"  broad_support arms with >=3 rescues @20: "
        f"{broad_support}/{len(CONFIGS)*len(branches)}",
        flush=True,
    )
    print(
        f"  recommended={None if recommended is None else (recommended['config'], recommended['branch'], recommended['depth'], recommended['rescued_outside_occurrences'])}",
        flush=True,
    )

    print("[8/8] Writing report...", flush=True)
    out = root / "results/manual/huy_d1_guided_prf_acquisition_v1"
    out.mkdir(parents=True, exist_ok=True)

    # Deduplicate rescue cases for readability by exact config/branch/depth/q/g.
    uniq = {}
    for x in rescue_cases:
        key = (
            x["config"], x["branch"], x["depth"],
            x["qid"], x["gold_doc"],
        )
        uniq[key] = x

    report = {
        "schema": "manual.d1_guided_prf_acquisition_v1",
        "status": verdict,
        "hypothesis": (
            "Use exact OOF D1 top documents as pseudo relevance feedback to "
            "extract discriminative corpus terms and retrieve novel sibling "
            "documents outside the current D1 candidate pool."
        ),
        "d1_recall": d1_recall,
        "base_candidate_oracle": base_oracle,
        "outside_pool_gold_occurrences": len(outside),
        "outside_pool_cases": outside,
        "configs": CONFIGS,
        "branches": branches,
        "novel_depths": NOVEL_DEPTHS,
        "results": results,
        "broad_support_arms_ge3_rescues_at20": broad_support,
        "gate": {
            "rescued_outside_occurrences_ge": 3,
            "rescued_blocks_ge": 2,
            "oracle_gain_ge": 0.003,
            "strong_rescue_ge": 4,
            "note": (
                "This is acquisition-only. A pass does not authorize direct "
                "Top-5 insertion; it authorizes a second-stage certification/"
                "reranking experiment."
            ),
        },
        "recommended": recommended,
        "elapsed_seconds": time.perf_counter() - t0,
    }

    report_path = out / "REPORT.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    rescue_path = out / "RESCUE_CASES.json"
    rescue_path.write_text(
        json.dumps(list(uniq.values()), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    promising_path = out / "PROMISING_CONFIG.json"
    if recommended is not None:
        promising_path.write_text(
            json.dumps({
                "schema": "manual.d1_guided_prf_acquisition_choice.v1",
                "recommended": recommended,
                "broad_support_arms_ge3_rescues_at20": broad_support,
                "next_step": (
                    "Build a SECOND-STAGE certification/reranking audit only "
                    "for the small PRF novel candidate set. Do not directly "
                    "union all PRF docs into D1 Top-5."
                ),
                "source_report": str(report_path),
            }, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    elif promising_path.exists():
        promising_path.unlink()

    print("=" * 116)
    print("VERDICT:", verdict)
    print("OUTSIDE-POOL GOLD:", len(outside))
    print("BASE ORACLE:", f"{base_oracle:.10f}")
    print("BROAD SUPPORT:", broad_support)
    if recommended is not None:
        print(
            "RECOMMENDED:",
            recommended["config"],
            recommended["branch"],
            f"depth={recommended['depth']}",
            f"rescues={recommended['rescued_outside_occurrences']}",
            f"oracleΔ={recommended['oracle_gain']:+.10f}",
            f"blocks={recommended['rescued_blocks']}",
        )
        print("Promising config:", promising_path)
    print("Report:", report_path)
    print("Rescue cases:", rescue_path)
    print("=" * 116)


if __name__ == "__main__":
    main()
