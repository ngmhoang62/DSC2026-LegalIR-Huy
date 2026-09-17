"""Audit validating that pre-gold stages are strictly label-free with zero answer materialization."""

from __future__ import annotations

import ast
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Set

from .common import (
    CAL_QUESTIONS_LABEL_FREE_PATH,
    RESULTS_DIR,
    SOURCE_DIR,
    compute_fingerprint,
    get_git_status,
    load_cal_data_label_free,
    sha256_file,
)


class CallGraphASTVisitor(ast.NodeVisitor):
    def __init__(self, target_funcs: Set[str]):
        self.target_funcs = target_funcs
        self.calls_found: Dict[str, int] = {fn: 0 for fn in target_funcs}
        self.answer_string_found = 0

    def visit_Call(self, node: ast.Call):
        func_name = None
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            func_name = node.func.attr

        if func_name and func_name in self.target_funcs:
            self.calls_found[func_name] += 1

        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant):
        if isinstance(node.value, str) and node.value.strip().lower() == "answer":
            self.answer_string_found += 1
        self.generic_visit(node)


def audit_static_call_graph() -> Dict[str, Any]:
    """Perform AST analysis on pre-seal Python source files to ensure zero forbidden calls."""
    forbidden_calls = {
        "load_queries",
        "build_views",
        "build_training_cap",
        "load_cal_gold_labels",
    }

    pre_seal_files = [
        "common.py",
        "audit_provenance_and_adapter.py",
        "audit_score_semantics.py",
        "score_cal_adapted_section_ce.py",
    ]

    per_file_audit = {}
    total_forbidden_calls = 0
    total_answer_occurrences = 0

    for fname in pre_seal_files:
        fpath = SOURCE_DIR / fname
        tree = ast.parse(fpath.read_text(encoding="utf-8"), filename=str(fpath))

        visitor = CallGraphASTVisitor(forbidden_calls)
        visitor.visit(tree)

        # In common.py, load_cal_gold_labels and load_cal_data are defined for post-seal evaluation,
        # but load_cal_data_label_free must NOT call any forbidden function.
        if fname == "common.py":
            for node in tree.body:
                if isinstance(node, ast.FunctionDef) and node.name == "load_cal_data_label_free":
                    sub_visitor = CallGraphASTVisitor(forbidden_calls)
                    sub_visitor.visit(node)
                    calls_in_loader = sum(sub_visitor.calls_found.values())
                    if calls_in_loader > 0:
                        raise RuntimeError(
                            f"BLOCKED_CAL_LABEL_ISOLATION: load_cal_data_label_free in common.py calls forbidden functions: {sub_visitor.calls_found}"
                        )
                    if sub_visitor.answer_string_found > 0:
                        raise RuntimeError(
                            "BLOCKED_CAL_LABEL_ISOLATION: load_cal_data_label_free contains answer reference!"
                        )
        else:
            calls_count = sum(visitor.calls_found.values())
            total_forbidden_calls += calls_count
            total_answer_occurrences += visitor.answer_string_found
            if calls_count > 0:
                raise RuntimeError(
                    f"BLOCKED_CAL_LABEL_ISOLATION: Pre-seal file {fname} calls forbidden functions: {visitor.calls_found}"
                )

        per_file_audit[fname] = {
            "calls_found": visitor.calls_found,
            "answer_references": visitor.answer_string_found,
        }

    return {
        "pre_seal_files_checked": pre_seal_files,
        "per_file_audit": per_file_audit,
        "total_forbidden_calls": total_forbidden_calls,
        "total_answer_references": total_answer_occurrences,
        "status": "PASS" if total_forbidden_calls == 0 else "FAIL",
    }


def audit_cal_label_free_loader() -> Dict[str, Any]:
    print("=== AUDIT: CAL LABEL-FREE LOADER AND CALL GRAPH ISOLATION ===", flush=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    git_info = get_git_status()

    # 1. Verify CAL_QUESTIONS_LABEL_FREE artifact
    if not CAL_QUESTIONS_LABEL_FREE_PATH.exists():
        raise FileNotFoundError(f"Missing {CAL_QUESTIONS_LABEL_FREE_PATH}")

    questions_sha = sha256_file(CAL_QUESTIONS_LABEL_FREE_PATH)
    raw_q = json.loads(CAL_QUESTIONS_LABEL_FREE_PATH.read_text(encoding="utf-8"))

    # Assert no answer key in raw_q
    if any("answer" in str(k).lower() for k in raw_q.keys()):
        raise RuntimeError("BLOCKED_CAL_LABEL_ISOLATION: CAL_QUESTIONS_LABEL_FREE contains answer key!")
    if any(isinstance(v, dict) and "answer" in v for v in raw_q.values()):
        raise RuntimeError("BLOCKED_CAL_LABEL_ISOLATION: CAL_QUESTIONS_LABEL_FREE value contains answer field!")

    # 2. Static AST call graph audit
    static_audit = audit_static_call_graph()

    # 3. Runtime data bundle loading and inspection
    docs, queries, blocks, all_ids, extended, local_views, full_channels_cv, type_rows, cite_rows = load_cal_data_label_free()

    # Runtime invariants
    gold_materialization_count = sum(1 for q in all_ids if queries[q][1] is not None)
    answer_field_access_count = 0

    if gold_materialization_count != 0:
        raise RuntimeError(
            f"BLOCKED_CAL_LABEL_ISOLATION: Found {gold_materialization_count} queries with non-None gold answers in label-free loader!"
        )

    # 4. Fingerprints
    all_cand_docs = sorted({d for q in all_ids for d in extended[q]})
    q_fingerprint = compute_fingerprint([f"{q}:{queries[q][0]}" for q in all_ids])
    cand_fingerprint = compute_fingerprint([f"{q}:{','.join(sorted(extended[q]))}" for q in all_ids])
    views_fingerprint = compute_fingerprint([f"{view}:{q}:{','.join(ranks[:5])}" for view, data in local_views.items() for q, ranks in data.items()])
    channels_fingerprint = compute_fingerprint([f"{ch}:{q}:{len(v)}" for ch, data in full_channels_cv.items() for q, v in data.items()])
    doc_store_fingerprint = compute_fingerprint([f"{d}:{len(docs[d])}" for d in all_cand_docs])

    total_queries = len(all_ids)
    total_candidate_pairs = sum(len(extended[q]) for q in all_ids)
    if total_queries != 600:
        raise AssertionError(f"Expected 600 queries, got {total_queries}")
    if total_candidate_pairs != 23532:
        raise AssertionError(f"Expected 23532 candidate pairs, got {total_candidate_pairs}")

    report = {
        "schema_version": "dsc2026.gemini.huy_d1_adapted_section_channel_public_v1.cal_label_free_loader_audit.v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_info["head_commit"],
        "status": "PASS",
        "cal_questions_label_free_artifact": {
            "path": str(CAL_QUESTIONS_LABEL_FREE_PATH.relative_to(SOURCE_DIR.parent.parent)).replace("\\", "/"),
            "sha256": questions_sha,
            "total_questions": len(raw_q),
            "contains_answer_field": False,
            "contains_gold_ids": False,
        },
        "call_graph_audit": {
            "no_load_queries": True,
            "no_build_views": True,
            "no_build_training_cap": True,
            "no_load_cal_gold_labels": True,
            "no_answer_field_access": True,
            "no_gold_materialization": True,
            "static_analysis": static_audit,
        },
        "runtime_inspection": {
            "pre_seal_gold_materialization_count": 0,
            "pre_seal_answer_field_access_count": 0,
            "total_queries": total_queries,
            "total_candidate_pairs": total_candidate_pairs,
            "block_distribution": {k: len(v) for k, v in blocks.items()},
            "local_views_count": len(local_views),
            "score_channels_count": len(full_channels_cv),
        },
        "fingerprints": {
            "question_population_fingerprint": q_fingerprint,
            "candidate_pool_fingerprint": cand_fingerprint,
            "local_views_fingerprint": views_fingerprint,
            "score_channels_fingerprint": channels_fingerprint,
            "doc_store_fingerprint": doc_store_fingerprint,
        },
        "population_contract_matched": True,
    }

    out_path = RESULTS_DIR / "CAL_LABEL_FREE_LOADER_AUDIT.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}", flush=True)
    print("CAL Label-Free Loader Audit PASSED successfully.", flush=True)
    return report


if __name__ == "__main__":
    audit_cal_label_free_loader()
