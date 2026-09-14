"""Diagnostic: Boundary Relation Coverage.

Checks whether label-free relation graph touches boundary opportunity queries
(defender rank 5 non-gold, challenger rank 6-8 gold) and defense cases
(defender rank 5 gold, challenger rank 6-8 non-gold).
Writes BOUNDARY_RELATION_COVERAGE.json.
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parents[2]
BASE_SNAPSHOT_DIR = CURRENT_DIR / "baseline_snapshot"
sys.path.insert(0, str(BASE_SNAPSHOT_DIR))

import run_huy_5fold_fasttrack as core

RESULTS_DIR = REPO_ROOT / "results/gemini/rabr_v1"
CACHE_DIR = RESULTS_DIR / "cache"
GRAPH_FILE = RESULTS_DIR / "RELATION_GRAPH.jsonl"
PREDICTIONS_FILE = CACHE_DIR / "BASELINE_PREDICTIONS_AND_SCORES.jsonl"


def load_graph():
    forward = defaultdict(list)
    backward = defaultdict(list)
    undirected = defaultdict(set)
    edge_types = Counter()

    with GRAPH_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            src, dst, rel = str(row["src_doc"]), str(row["dst_doc"]), row["relation_type"]
            forward[src].append((dst, rel))
            backward[dst].append((src, rel))
            undirected[src].add(dst)
            undirected[dst].add(src)
            edge_types[rel] += 1

    # Connected components
    visited = set()
    components = {}
    comp_id = 0
    for node in list(undirected.keys()):
        if node not in visited:
            queue = [node]
            visited.add(node)
            comp_nodes = {node}
            while queue:
                curr = queue.pop()
                for neighbor in undirected[curr]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        comp_nodes.add(neighbor)
                        queue.append(neighbor)
            for c_node in comp_nodes:
                components[c_node] = comp_id
            comp_id += 1

    return forward, backward, undirected, components, edge_types


def main():
    started = time.perf_counter()
    print("=== Step 5: Boundary Relation Coverage Diagnostic ===", flush=True)

    folds, pools, questions, golds, _, _, _, _ = core.load_inputs()
    forward, backward, undirected, components, edge_counts = load_graph()

    predictions = {}
    scores = {}
    with PREDICTIONS_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            qid = str(row["qid"])
            predictions[qid] = row["order"]
            scores[qid] = row["scores"]

    total_queries = len(predictions)
    total_opportunities = 0
    opp_by_challenger_rank = Counter()
    total_defense_cases = 0

    opp_qids = []
    def_qids = []

    # Oracle computation
    baseline_hits = 0.0
    oracle_hits = 0.0
    total_gold_count = 0

    opp_touched_any = []
    opp_touched_replaces_repeals = []
    opp_touched_amends_guides = []
    opp_touched_cites = []

    opp_pair_replaces = []
    opp_pair_repeals = []
    opp_pair_amends = []
    opp_pair_guides = []
    opp_pair_cites = []

    opp_top4_replaces = []
    opp_top4_amends = []
    opp_top4_guides = []
    opp_top4_cites = []
    opp_top4_same_component = []

    def_touched_any = []
    def_touched_replaces_repeals = []
    def_touched_amends_guides = []
    def_touched_cites = []

    for qid, order in predictions.items():
        gold = golds[qid]
        top4 = order[:4]
        defender = order[4]
        challengers = order[5:8]  # ranks 6, 7, 8

        top4_hits = len(set(top4) & gold)
        defender_hit = float(defender in gold)
        base_hits = top4_hits + defender_hit
        baseline_hits += base_hits / len(gold)
        total_gold_count += len(gold)

        is_opp = (defender not in gold) and any(c in gold for c in challengers)
        is_def = (defender in gold) and any(c not in gold for c in challengers)

        # Oracle choice from ranks 5-8
        slot5_candidates = [defender] + challengers
        gold_slot5 = [c for c in slot5_candidates if c in gold]
        if gold_slot5:
            oracle_hits += (top4_hits + 1.0) / len(gold)
        else:
            oracle_hits += top4_hits / len(gold)

        if is_opp:
            total_opportunities += 1
            opp_qids.append(qid)
            for idx, c in enumerate(challengers):
                if c in gold:
                    opp_by_challenger_rank[idx + 6] += 1

            # Check relation signals for this opportunity
            # 1. Direct relations between challengers and defender
            has_pair_rel = False
            has_pair_rep = False
            has_pair_amend = False
            has_pair_guide = False
            has_pair_cite = False

            # Check defender-challenger edges
            for c in challengers:
                for dst, r in forward[c]:
                    if dst == defender:
                        has_pair_rel = True
                        if r == "REPLACES": has_pair_rep = True
                        elif r == "REPEALS": has_pair_rep = True
                        elif r == "AMENDS": has_pair_amend = True
                        elif r == "GUIDES": has_pair_guide = True
                        elif r == "CITES": has_pair_cite = True
                for dst, r in forward[defender]:
                    if dst == c:
                        has_pair_rel = True
                        if r == "REPLACES": has_pair_rep = True
                        elif r == "REPEALS": has_pair_rep = True
                        elif r == "AMENDS": has_pair_amend = True
                        elif r == "GUIDES": has_pair_guide = True
                        elif r == "CITES": has_pair_cite = True

            # 2. Relations to Top-4
            top4_set = set(top4)
            has_top4_rel = False
            has_top4_rep = False
            has_top4_amend = False
            has_top4_guide = False
            has_top4_cite = False
            has_top4_same_comp = False

            for cand in [defender] + challengers:
                if components.get(cand) is not None:
                    if any(components.get(t) == components.get(cand) for t in top4 if components.get(t) is not None):
                        has_top4_same_comp = True

                for dst, r in forward[cand]:
                    if dst in top4_set:
                        has_top4_rel = True
                        if r in ("REPLACES", "REPEALS"): has_top4_rep = True
                        elif r == "AMENDS": has_top4_amend = True
                        elif r == "GUIDES": has_top4_guide = True
                        elif r == "CITES": has_top4_cite = True
                for src, r in backward[cand]:
                    if src in top4_set:
                        has_top4_rel = True
                        if r in ("REPLACES", "REPEALS"): has_top4_rep = True
                        elif r == "AMENDS": has_top4_amend = True
                        elif r == "GUIDES": has_top4_guide = True
                        elif r == "CITES": has_top4_cite = True

            if has_pair_rel or has_top4_rel or has_top4_same_comp:
                opp_touched_any.append(qid)
            if has_pair_rep or has_top4_rep:
                opp_touched_replaces_repeals.append(qid)
            if has_pair_amend or has_pair_guide or has_top4_amend or has_top4_guide:
                opp_touched_amends_guides.append(qid)
            if has_pair_cite or has_top4_cite:
                opp_touched_cites.append(qid)

            if has_pair_rep: opp_pair_replaces.append(qid)
            if has_pair_amend: opp_pair_amends.append(qid)
            if has_pair_guide: opp_pair_guides.append(qid)
            if has_pair_cite: opp_pair_cites.append(qid)
            if has_top4_same_comp: opp_top4_same_component.append(qid)

        if is_def:
            total_defense_cases += 1
            def_qids.append(qid)

            has_def_rel = False
            has_def_rep = False
            has_def_amend = False
            has_def_cite = False

            top4_set = set(top4)
            for cand in [defender] + challengers:
                for dst, r in forward[cand]:
                    if dst in top4_set or dst == defender or dst in challengers:
                        has_def_rel = True
                        if r in ("REPLACES", "REPEALS"): has_def_rep = True
                        elif r in ("AMENDS", "GUIDES"): has_def_amend = True
                        elif r == "CITES": has_def_cite = True

            if has_def_rel: def_touched_any.append(qid)
            if has_def_rep: def_touched_replaces_repeals.append(qid)
            if has_def_amend: def_touched_amends_guides.append(qid)
            if has_def_cite: def_touched_cites.append(qid)

    baseline_recall_at_5 = baseline_hits / total_queries
    oracle_recall_at_5 = oracle_hits / total_queries
    oracle_headroom = oracle_recall_at_5 - baseline_recall_at_5

    coverage_report = {
        "schema_version": "dsc2026.gemini.rabr_v1.boundary_coverage.v1",
        "total_queries": total_queries,
        "baseline_recall_at_5": baseline_recall_at_5,
        "oracle_recall_at_5": oracle_recall_at_5,
        "oracle_headroom_delta": oracle_headroom,
        "total_opportunity_queries": total_opportunities,
        "gold_opportunities_by_challenger_rank": {
            "rank_6": opp_by_challenger_rank[6],
            "rank_7": opp_by_challenger_rank[7],
            "rank_8": opp_by_challenger_rank[8],
        },
        "total_defense_cases": total_defense_cases,
        "opportunities_touched": {
            "touched_by_any_graph_signal": len(opp_touched_any),
            "touched_by_replaces_repeals": len(opp_touched_replaces_repeals),
            "touched_by_amends_guides": len(opp_touched_amends_guides),
            "touched_by_cites": len(opp_touched_cites),
            "pair_replaces_repeals": len(opp_pair_replaces),
            "pair_amends": len(opp_pair_amends),
            "pair_guides": len(opp_pair_guides),
            "pair_cites": len(opp_pair_cites),
            "same_component_with_top4": len(opp_top4_same_component),
        },
        "defense_cases_touched": {
            "touched_by_any_graph_signal": len(def_touched_any),
            "touched_by_replaces_repeals": len(def_touched_replaces_repeals),
            "touched_by_amends_guides": len(def_touched_amends_guides),
            "touched_by_cites": len(def_touched_cites),
        },
        "coverage_meets_threshold": len(opp_touched_any) >= 10,
        "runtime_seconds": time.perf_counter() - started,
    }

    out_path = RESULTS_DIR / "BOUNDARY_RELATION_COVERAGE.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(coverage_report, f, indent=2, ensure_ascii=False)
    print(f"Wrote {out_path}", flush=True)
    print("Coverage summary:", json.dumps(coverage_report, indent=2), flush=True)

    if len(opp_touched_any) < 10:
        abort_path = RESULTS_DIR / "LOW_COVERAGE_ABORT.md"
        abort_path.write_text(
            f"# LOW COVERAGE ABORT\n\nGraph touches only {len(opp_touched_any)} recoverable boundary opportunity queries (< 10 threshold).\n",
            encoding="utf-8",
        )
        print(f"STOPPING: Graph touches < 10 opportunities ({len(opp_touched_any)}). Wrote {abort_path}", flush=True)
        sys.exit(0)
    else:
        print(f"CONTINUING: Graph touches {len(opp_touched_any)} opportunities (>= 10 threshold).", flush=True)


if __name__ == "__main__":
    main()
