#!/usr/bin/env python
"""
D1 DOCUMENT-ANCHORED LEGAL GRAPH ACQUISITION AUDIT V1
======================================================
CPU-only. No private labels. No neural inference.

Existing query-anchored legal-reference expansion only triggers when the query
itself contains an explicit legal reference. This experiment instead anchors on
D1's own top documents and traverses the corpus legal-reference graph.

Arms:
  REL_OUT_T3      : top3 D1 anchors, outgoing relation-specific edges
  REL_BIDIR_T3    : top3, incoming + outgoing relation-specific edges
  REF_OUT_T3      : top3, outgoing all reference-mention edges
  REF_BIDIR_T3    : top3, incoming + outgoing all reference-mention edges
  REF_BIDIR_T5    : top5, incoming + outgoing all reference-mention edges
  REL2_BIDIR_T3   : top3, relation-specific bidirectional graph, <=2 hops

Acquisition only: never insert these candidates directly into Top-5 from this
experiment. A pass only authorizes a second-stage certification/ranking audit.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

EXPECTED_D1 = 0.9569444444444444
EXPECTED_ORACLE = 0.9847222222222222
EXPECTED_OUTSIDE = 15
SEED = 2026
D1_VIEWS = ["base", "expanded", "jina", "dense", "corpus"]
DEPTHS = [5, 10, 20, 50]

AMEND_TRIGGERS = [
    "sửa đổi", "bổ sung", "thay thế", "bãi bỏ",
    "sua doi", "bo sung", "thay the", "bai bo",
]
GUIDE_TRIGGERS = [
    "hướng dẫn thi hành",
    "quy định chi tiết và hướng dẫn thi hành",
    "quy định chi tiết",
    "hướng dẫn", "thi hành",
    "huong dan thi hanh", "quy dinh chi tiet", "huong dan", "thi hanh",
]

ARMS = {
    "REL_OUT_T3": dict(anchor_k=3, relation_only=True, bidir=False, max_hops=1),
    "REL_BIDIR_T3": dict(anchor_k=3, relation_only=True, bidir=True, max_hops=1),
    "REF_OUT_T3": dict(anchor_k=3, relation_only=False, bidir=False, max_hops=1),
    "REF_BIDIR_T3": dict(anchor_k=3, relation_only=False, bidir=True, max_hops=1),
    "REF_BIDIR_T5": dict(anchor_k=5, relation_only=False, bidir=True, max_hops=1),
    "REL2_BIDIR_T3": dict(anchor_k=3, relation_only=True, bidir=True, max_hops=2),
}


def numeric_doc_key(x):
    sx = str(x)
    return (0, int(sx)) if sx.isdigit() else (1, sx)


def load_corpus(context_dir: Path):
    corpus = {}
    for p in sorted(context_dir.glob("context_*.json")):
        did = p.stem[len("context_"):]
        row = json.loads(p.read_text(encoding="utf-8"))
        corpus[str(did)] = {
            "id": str(did),
            "passage": row.get("passage") or "",
            "link": row.get("link") or "",
        }
    return corpus


def build_reference_graph(corpus):
    from src.gemini.huy_d1_query_anchored_legal_ref_expansion_v1.legal_ref_indexer import (
        REF_REGEX, canonicalize_ref, extract_doc_own_reference,
    )

    ref_to_docs = defaultdict(list)
    doc_to_ref = {}
    for did, item in corpus.items():
        own = extract_doc_own_reference(item["passage"], item["link"])
        if own:
            doc_to_ref[did] = own
            ref_to_docs[own].append(did)
    for ref in ref_to_docs:
        ref_to_docs[ref].sort(key=numeric_doc_key)

    edge_agg = {}
    relation_mentions = generic_mentions = mapped_mentions = 0

    for i, (src, item) in enumerate(corpus.items(), 1):
        passage = item["passage"]
        own = doc_to_ref.get(src)
        for m in REF_REGEX.finditer(passage):
            target_ref = canonicalize_ref(m.group(0))
            if own and target_ref == own:
                continue
            targets = ref_to_docs.get(target_ref, [])
            if not targets:
                continue

            mapped_mentions += 1
            start, end = m.span()
            ctx = (
                passage[max(0, start - 100):start] + " " +
                passage[end:min(len(passage), end + 100)]
            ).lower()
            relation_family = None
            if any(t in ctx for t in AMEND_TRIGGERS):
                relation_family = "AMENDMENT_REPLACEMENT_REPEAL"
            elif any(t in ctx for t in GUIDE_TRIGGERS):
                relation_family = "IMPLEMENTATION_GUIDANCE"
            is_relation = relation_family is not None
            relation_mentions += int(is_relation)
            generic_mentions += int(not is_relation)
            is_header = start < 1200

            for dst in targets:
                if dst == src:
                    continue
                key = (src, dst)
                e = edge_agg.setdefault(key, {
                    "src": src, "dst": dst,
                    "mentions": 0,
                    "relation_mentions": 0,
                    "header_mentions": 0,
                    "families": set(), "refs": set(),
                })
                e["mentions"] += 1
                e["relation_mentions"] += int(is_relation)
                e["header_mentions"] += int(is_header)
                if relation_family:
                    e["families"].add(relation_family)
                e["refs"].add(target_ref)

        if i % 1000 == 0:
            print(f"    graph parsed {i}/{len(corpus)} docs", flush=True)

    outgoing, incoming = defaultdict(list), defaultdict(list)
    for e in edge_agg.values():
        row = {
            **e,
            "is_relation": e["relation_mentions"] > 0,
            "is_header": e["header_mentions"] > 0,
            "families": sorted(e["families"]),
            "refs": sorted(e["refs"]),
        }
        outgoing[e["src"]].append(row)
        incoming[e["dst"]].append(row)

    audit = {
        "documents": len(corpus),
        "documents_with_own_ref": len(doc_to_ref),
        "unique_refs": len(ref_to_docs),
        "directed_doc_edges": len(edge_agg),
        "nodes_with_outgoing": len(outgoing),
        "nodes_with_incoming": len(incoming),
        "mapped_reference_mentions": mapped_mentions,
        "relation_mentions": relation_mentions,
        "generic_mentions": generic_mentions,
    }
    return dict(outgoing), dict(incoming), audit


def build_d1_rows(local_views, full_channels, extended, all_ids, type_rows, cite_rows):
    from tune_expanded_fusion_selection import ltr_features
    rows0, groups = ltr_features(
        local_views, D1_VIEWS, extended, all_ids, full_channels
    )
    rows = {
        q: np.concatenate([rows0[q], type_rows[q], cite_rows[q]], axis=1).astype(np.float32, copy=False)
        for q in all_ids
    }
    for q in all_ids:
        if rows[q].shape[1] != 48:
            raise RuntimeError(f"Expected exact D1 48D; q={q}, shape={rows[q].shape}")
    return rows, groups


def exact_oof_d1(blocks, all_ids, rows, groups, gold):
    pred, full, scoremaps = {}, {}, {}
    for held in sorted(blocks):
        train = sum((list(blocks[b]) for b in sorted(blocks) if b != held), [])
        test = list(blocks[held])
        X = np.vstack([rows[q] for q in train])
        y = np.concatenate([[d in gold[q] for d in groups[q]] for q in train]).astype(np.int8)
        scaler = StandardScaler().fit(X)
        model = LogisticRegression(
            C=.15, class_weight="balanced", solver="liblinear",
            max_iter=3000, random_state=SEED,
        ).fit(scaler.transform(X), y)
        for q in test:
            s = np.asarray(model.decision_function(scaler.transform(rows[q])), dtype=np.float64)
            idx = np.argsort(-s, kind="stable")
            order = [groups[q][i] for i in idx]
            pred[q] = order[:5]
            full[q] = order
            scoremaps[q] = {groups[q][i]: float(s[i]) for i in range(len(groups[q]))}
    recall = float(np.mean([
        len(set(pred[q]) & set(gold[q])) / max(1, len(gold[q])) for q in all_ids
    ]))
    return pred, full, scoremaps, recall


def candidate_oracle(pool, gold, ids):
    return float(np.mean([
        len(set(pool[q]) & set(gold[q])) / max(1, len(gold[q])) for q in ids
    ]))


def add_support(rec, anchor_rank, hop, edge, direction):
    rec["anchors"].add(anchor_rank)
    rec["best_anchor_rank"] = min(rec["best_anchor_rank"], anchor_rank)
    rec["best_hop"] = min(rec["best_hop"], hop)
    rec["edge_support"] += int(edge["mentions"])
    rec["relation_support"] += int(edge["relation_mentions"])
    rec["header_support"] += int(edge["header_mentions"])
    rec["directions"].add(direction)
    rec["families"].update(edge["families"])
    rec["refs"].update(edge["refs"])


def graph_candidates_for_query(anchors, existing_pool, outgoing, incoming, relation_only, bidir, max_hops):
    records = {}

    def ensure(d):
        return records.setdefault(d, {
            "doc_id": d, "anchors": set(),
            "best_anchor_rank": 10**9, "best_hop": 10**9,
            "edge_support": 0, "relation_support": 0, "header_support": 0,
            "directions": set(), "families": set(), "refs": set(),
        })

    frontier = [(d, r) for r, d in enumerate(anchors, 1)]
    seen = set((d, r, 0) for d, r in frontier)

    for hop in range(1, max_hops + 1):
        nxt = []
        for node, anchor_rank in frontier:
            edges = []
            for e in outgoing.get(node, []):
                if not relation_only or e["is_relation"]:
                    edges.append((e["dst"], e, "OUTGOING"))
            if bidir:
                for e in incoming.get(node, []):
                    if not relation_only or e["is_relation"]:
                        edges.append((e["src"], e, "INCOMING"))

            for dst, edge, direction in edges:
                if dst in anchors:
                    continue
                rec = ensure(dst)
                add_support(rec, anchor_rank, hop, edge, direction)
                if hop < max_hops:
                    state = (dst, anchor_rank, hop)
                    if state not in seen:
                        seen.add(state)
                        nxt.append((dst, anchor_rank))
        frontier = nxt
        if not frontier:
            break

    vals = [r for d, r in records.items() if d not in existing_pool]
    vals.sort(key=lambda r: (
        r["best_hop"], -len(r["anchors"]), -r["relation_support"],
        -r["header_support"], r["best_anchor_rank"], -r["edge_support"],
        numeric_doc_key(r["doc_id"]),
    ))
    for r in vals:
        r["anchor_support"] = len(r["anchors"])
        r["anchors"] = sorted(r["anchors"])
        r["directions"] = sorted(r["directions"])
        r["families"] = sorted(r["families"])
        r["refs"] = sorted(r["refs"])
    return vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=Path, required=True)
    args = ap.parse_args()
    root = args.repo_root.expanduser().resolve()
    sys.path.insert(0, str(root))

    print("[1/7] Loading authoritative CAL600 D1 world...", flush=True)
    from src.gemini.huy_vnlegal_rank_ablation_v1.evaluate_ablation_cal import load_cal_inputs
    (
        queries, blocks, all_ids, extended, local_views, full_channels, gold,
        _vnlegal, type_rows, cite_rows,
    ) = load_cal_inputs()

    print("[2/7] Reconstructing exact OOF D1 ranking...", flush=True)
    rows, groups = build_d1_rows(local_views, full_channels, extended, all_ids, type_rows, cite_rows)
    d1_top5, d1_full, d1_scores, d1_recall = exact_oof_d1(blocks, all_ids, rows, groups, gold)
    print(f"  D1 Recall@5={d1_recall:.10f}", flush=True)
    if abs(d1_recall - EXPECTED_D1) > 1e-9:
        raise RuntimeError(f"D1 parity failed: {d1_recall} != {EXPECTED_D1}")

    print("[3/7] Auditing baseline candidate ceiling...", flush=True)
    base_oracle = candidate_oracle(extended, gold, all_ids)
    bmap = {q: b for b, ids in blocks.items() for q in ids}
    outside = []
    for q in all_ids:
        pool = set(extended[q])
        for g in gold[q]:
            if g not in pool:
                outside.append({
                    "qid": q, "gold_doc": g, "block": bmap[q],
                    "question": queries[q][0], "d1_top5": d1_top5[q],
                })
    print(f"  candidate oracle={base_oracle:.10f} outside-pool gold occurrences={len(outside)}", flush=True)
    if abs(base_oracle - EXPECTED_ORACLE) > 1e-9:
        raise RuntimeError(f"Candidate oracle parity failed: {base_oracle} != {EXPECTED_ORACLE}")
    if len(outside) != EXPECTED_OUTSIDE:
        raise RuntimeError(f"Outside-pool count parity failed: {len(outside)} != {EXPECTED_OUTSIDE}")

    print("[4/7] Building corpus legal-reference graph...", flush=True)
    context_dir = root / "DSC2026-LegalIR-main/v4_run/public_test_dataset/selected-contexts"
    corpus = load_corpus(context_dir)
    if len(corpus) != 8532:
        raise RuntimeError(f"Expected 8532 corpus docs, got {len(corpus)}")
    t0 = time.perf_counter()
    outgoing, incoming, graph_audit = build_reference_graph(corpus)
    print(
        "  graph: "
        f"own-ref docs={graph_audit['documents_with_own_ref']} "
        f"unique refs={graph_audit['unique_refs']} "
        f"edges={graph_audit['directed_doc_edges']} "
        f"relation mentions={graph_audit['relation_mentions']} "
        f"generic mentions={graph_audit['generic_mentions']}",
        flush=True,
    )

    print("[5/7] Generating document-anchored graph candidates...", flush=True)
    candidates = {arm: {} for arm in ARMS}
    traces = {arm: {} for arm in ARMS}
    for i, q in enumerate(all_ids, 1):
        pool = set(extended[q])
        for arm, cfg in ARMS.items():
            anchors = d1_full[q][:cfg["anchor_k"]]
            recs = graph_candidates_for_query(
                anchors, pool, outgoing, incoming,
                cfg["relation_only"], cfg["bidir"], cfg["max_hops"],
            )
            candidates[arm][q] = [r["doc_id"] for r in recs]
            traces[arm][q] = {"anchors": list(anchors), "candidate_details": recs[:50]}
        if i % 100 == 0:
            print(f"  queries {i}/{len(all_ids)}", flush=True)

    print("[6/7] Measuring novel acquisition and oracle lift...", flush=True)
    outside_pairs = {(x["qid"], x["gold_doc"]) for x in outside}
    results, rescue_cases = {}, []
    for arm in ARMS:
        results[arm] = {}
        for depth in DEPTHS:
            augmented, rescued, counts = {}, [], []
            for q in all_ids:
                novel = candidates[arm][q][:depth]
                counts.append(len(novel))
                augmented[q] = list(extended[q]) + novel
                for g in gold[q]:
                    if (q, g) in outside_pairs and g in novel:
                        rescued.append((q, g))
            oracle = candidate_oracle(augmented, gold, all_ids)
            blocks_hit = sorted(set(bmap[q] for q, _ in rescued))
            row = {
                "depth": depth,
                "rescued_outside_occurrences": len(rescued),
                "rescued_unique_queries": len(set(q for q, _ in rescued)),
                "rescued_blocks": blocks_hit,
                "rescued_block_count": len(blocks_hit),
                "rescued_multi_gold_occurrences": int(sum(len(gold[q]) > 1 for q, _ in rescued)),
                "candidate_oracle": oracle,
                "oracle_gain": oracle - base_oracle,
                "queries_with_any_novel": int(sum(x > 0 for x in counts)),
                "mean_novel_available": float(np.mean(counts)),
                "median_novel_available": float(np.median(counts)),
                "p95_novel_available": float(np.percentile(counts, 95)),
                "max_novel_available": int(max(counts)),
                "rescued_pairs": [dict(qid=q, gold_doc=g, block=bmap[q]) for q, g in sorted(rescued)],
            }
            results[arm][str(depth)] = row
            for q, g in rescued:
                detail = next((x for x in traces[arm][q]["candidate_details"] if x["doc_id"] == g), None)
                rescue_cases.append({
                    "arm": arm, "depth": depth, "qid": q, "gold_doc": g,
                    "block": bmap[q], "question": queries[q][0],
                    "d1_top5": d1_top5[q], "anchors": traces[arm][q]["anchors"],
                    "novel_rank": candidates[arm][q].index(g) + 1,
                    "graph_detail": detail,
                })
        r20 = results[arm]["20"]
        print(
            f"  {arm:18s} @20 rescue={r20['rescued_outside_occurrences']:2d}/{len(outside)} "
            f"queries={r20['rescued_unique_queries']:2d} blocks={r20['rescued_blocks']} "
            f"oracleΔ={r20['oracle_gain']:+.6f} novelQ={r20['queries_with_any_novel']}/{len(all_ids)} "
            f"meanNovel={r20['mean_novel_available']:.2f}",
            flush=True,
        )

    print("[7/7] Acquisition gate + report...", flush=True)
    eligible = []
    for arm in ARMS:
        for depth in [5, 10, 20]:
            r = results[arm][str(depth)]
            passed = (
                r["rescued_outside_occurrences"] >= 3
                and r["rescued_block_count"] >= 2
                and r["oracle_gain"] >= 0.003 - 1e-12
            )
            if passed:
                eligible.append({
                    "arm": arm, "depth": depth,
                    "strength": "STRONG" if r["rescued_outside_occurrences"] >= 4 else "PROMISING",
                    **r,
                })

    broad_support = int(sum(results[arm]["20"]["rescued_outside_occurrences"] >= 3 for arm in ARMS))
    recommended = None
    if eligible:
        recommended = max(eligible, key=lambda x: (
            x["rescued_outside_occurrences"], x["rescued_block_count"],
            x["oracle_gain"], -x["mean_novel_available"], -x["depth"],
        ))
    verdict = "PROMISING_DOCUMENT_ANCHORED_LEGAL_GRAPH" if recommended else "KILL_DOCUMENT_ANCHORED_LEGAL_GRAPH"

    out = root / "results/manual/huy_d1_document_graph_acquisition_v1"
    out.mkdir(parents=True, exist_ok=True)
    uniq = {}
    for x in rescue_cases:
        key = (x["arm"], x["qid"], x["gold_doc"])
        if key not in uniq or x["depth"] < uniq[key]["depth"]:
            uniq[key] = x

    report = {
        "schema": "manual.d1_document_graph_acquisition_v1",
        "status": verdict,
        "private_labels_used": False,
        "hypothesis": "Use exact OOF D1 top documents as anchors for legal-reference graph expansion.",
        "d1_recall": d1_recall,
        "base_candidate_oracle": base_oracle,
        "outside_pool_gold_occurrences": len(outside),
        "outside_pool_cases": outside,
        "graph_audit": graph_audit,
        "arms": ARMS,
        "depths": DEPTHS,
        "results": results,
        "broad_support_arms_ge3_rescues_at20": broad_support,
        "recommended": recommended,
        "elapsed_seconds": time.perf_counter() - t0,
    }
    report_path = out / "REPORT.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    rescue_path = out / "RESCUE_CASES.json"
    rescue_path.write_text(json.dumps(list(uniq.values()), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    promising_path = out / "PROMISING_CONFIG.json"
    if recommended:
        promising_path.write_text(json.dumps({
            "schema": "manual.d1_document_graph_choice.v1",
            "recommended": recommended,
            "source_report": str(report_path),
            "next_step": "Build conservative certification/reranking only for graph-novel candidates.",
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    elif promising_path.exists():
        promising_path.unlink()

    print("=" * 118)
    print("VERDICT:", verdict)
    print("D1:", f"{d1_recall:.10f}")
    print("BASE ORACLE:", f"{base_oracle:.10f}")
    print("OUTSIDE-POOL GOLD:", len(outside))
    print("BROAD SUPPORT:", f"{broad_support}/{len(ARMS)}")
    if recommended:
        print(
            "RECOMMENDED:", recommended["arm"], f"depth={recommended['depth']}",
            f"rescues={recommended['rescued_outside_occurrences']}",
            f"oracleΔ={recommended['oracle_gain']:+.10f}",
            f"blocks={recommended['rescued_blocks']}",
        )
        print("Promising:", promising_path)
    print("Report:", report_path)
    print("Rescue cases:", rescue_path)
    print("=" * 118)


if __name__ == "__main__":
    main()
