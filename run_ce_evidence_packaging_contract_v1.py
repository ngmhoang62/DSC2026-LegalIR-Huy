#!/usr/bin/env python
"""
HUY PIPELINE AUDIT P — CE EVIDENCE PACKAGING CONTRACT V1
========================================================

CPU-oriented audit of the exact D1 CAL Top5 packages consumed by the frozen
BGE cross-encoder / Evidence layer.

No model forward pass. No public labels.

Audits:
- exact D1 Top5 population vs V2 renderer coverage;
- Evidence.package(qid, doc_id) success/failure by rank;
- untruncated tokenizer length of the actual packaged query/evidence pair;
- >512 truncation rate by rank (Top1-4 vs rank5);
- evidence-package text length / duplication;
- whether missing package coverage is symmetric or rank5-skewed;
- the known one CAL query outside strict V2 is explicitly reported.

Runtime dependencies already used by the CE experiments:
  ../run_noncal_trainable_ce_boundary_v3_fixed.py
  ../LegalIR/src/exp_final/evidence.py

This audit does NOT instantiate the cross-encoder model.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer


MAXLEN = 512


def loadmod(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_d1_top5(root: Path):
    p = (
        root
        / "results/gemini/huy_d1_legal_section_evidence_v1/"
        "S0_S1_CAL_PREDICTIONS.jsonl"
    )
    out = {}
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            out[str(r["qid"])] = [str(x) for x in r["s0_top5"]]
    if len(out) != 600:
        raise RuntimeError(f"Expected exact D1 CAL600 population, got {len(out)}")
    return out, p


def extract_pair(pkg):
    """Best-effort extraction of the exact text pair handed to CrossEncoder."""
    if isinstance(pkg, (tuple, list)) and len(pkg) == 2:
        if isinstance(pkg[0], str) and isinstance(pkg[1], str):
            return pkg[0], pkg[1], "tuple2"

    if isinstance(pkg, dict):
        q = None
        p = None
        for k in ("query", "question", "q", "text_a"):
            if isinstance(pkg.get(k), str):
                q = pkg[k]
                break
        for k in ("passage", "evidence", "document", "text", "text_b", "context"):
            if isinstance(pkg.get(k), str):
                p = pkg[k]
                break
        if q is not None and p is not None:
            return q, p, "dict"

    for qattr in ("query", "question", "text_a"):
        for pattr in ("passage", "evidence", "text", "text_b", "context"):
            q = getattr(pkg, qattr, None)
            p = getattr(pkg, pattr, None)
            if isinstance(q, str) and isinstance(p, str):
                return q, p, f"attrs:{qattr}/{pattr}"

    raise TypeError(
        f"Unknown Evidence.package return type/shape: {type(pkg).__name__}"
    )


def summarize(x):
    arr = np.asarray(x, dtype=np.float64)
    if not len(arr):
        return {}
    return {
        "n": int(len(arr)),
        "mean": float(arr.mean()),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(arr.max()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=Path, required=True)
    args = ap.parse_args()

    root = args.repo_root.resolve()
    sibling = root.parent / "LegalIR"
    base_script = root.parent / "run_noncal_trainable_ce_boundary_v3_fixed.py"

    if not base_script.is_file():
        raise FileNotFoundError(
            f"Missing historical CE contract script: {base_script}"
        )
    if not (sibling / "src").is_dir():
        raise FileNotFoundError(sibling / "src")

    print("[1/5] Loading exact D1 Top5 + frozen V2 RenderData...", flush=True)
    d1, d1_path = load_d1_top5(root)

    m = loadmod(base_script, "ce_packaging_contract_base")
    cal_ids, _ = m.get_cal_ids_label_free(root)
    world = m.load_noncal_world(root, sibling, set(cal_ids))

    if "render" not in world:
        raise RuntimeError(
            f"load_noncal_world() did not return render; keys={sorted(world)}"
        )
    render = world["render"]

    sys.path[:0] = [str(sibling), str(sibling / "src")]
    from exp_final.evidence import Evidence

    tok_path = root / "models/jina-reranker-v2-base-multilingual"
    if not tok_path.is_dir():
        raise FileNotFoundError(tok_path)
    tok = AutoTokenizer.from_pretrained(
        tok_path,
        trust_remote_code=True,
        fix_mistral_regex=True,
        local_files_only=True,
    )
    tok.model_max_length = 10**9

    evidence = Evidence(render, tok)

    print("[2/5] Determining exact renderer coverage...", flush=True)
    # RenderData implementations used in this project keep either _qrow or
    # question dict. Use both without assuming one exact class definition.
    qrow = getattr(render, "_qrow", None)
    if isinstance(qrow, dict):
        render_qids = set(map(str, qrow))
    else:
        questions = getattr(render, "questions", None)
        if isinstance(questions, dict):
            render_qids = set(map(str, questions))
        else:
            render_qids = set()

    if not render_qids:
        raise RuntimeError(
            "Could not infer qid coverage from RenderData (_qrow/questions missing)"
        )

    covered = sorted(set(d1) & render_qids)
    outside = sorted(set(d1) - render_qids)
    print(
        f"  exact D1 qids={len(d1)} renderer-covered={len(covered)} "
        f"outside-renderer={len(outside)}",
        flush=True,
    )

    print("[3/5] Packaging exact D1 Top5 with Evidence.package()...", flush=True)

    rows = []
    failures = []
    shape_counts = Counter()

    for qi, q in enumerate(sorted(d1, key=lambda x: int(x)), 1):
        if q not in render_qids:
            for rank, d in enumerate(d1[q], 1):
                failures.append({
                    "qid": q,
                    "doc_id": d,
                    "rank": rank,
                    "reason": "QID_OUTSIDE_RENDERER",
                })
            continue

        for rank, d in enumerate(d1[q], 1):
            try:
                pkg = evidence.package(q, d)
                qa, pb, shape = extract_pair(pkg)
                shape_counts[shape] += 1

                enc = tok(
                    qa,
                    pb,
                    add_special_tokens=True,
                    truncation=False,
                )
                n_tokens = len(enc["input_ids"])

                rows.append({
                    "qid": q,
                    "doc_id": d,
                    "rank": rank,
                    "query_chars": len(qa),
                    "evidence_chars": len(pb),
                    "pair_tokens": n_tokens,
                    "over512": bool(n_tokens > MAXLEN),
                    "package_shape": shape,
                    "evidence_sha1_like": hash(pb),
                })
            except Exception as e:
                failures.append({
                    "qid": q,
                    "doc_id": d,
                    "rank": rank,
                    "reason": type(e).__name__,
                    "error": str(e),
                })

        if qi % 50 == 0 or qi == len(d1):
            print(
                f"  queries {qi}/{len(d1)} "
                f"success_pairs={len(rows)} failures={len(failures)}",
                flush=True,
            )

    try:
        evidence.db.close()
    except Exception:
        pass

    print("[4/5] Summarizing package/truncation asymmetry...", flush=True)

    by_rank = {}
    for rank in range(1, 6):
        rr = [x for x in rows if x["rank"] == rank]
        lens = [x["pair_tokens"] for x in rr]
        by_rank[str(rank)] = {
            "successful_packages": len(rr),
            "pair_tokens": summarize(lens),
            "over512": sum(x["over512"] for x in rr),
            "over512_rate": (
                float(np.mean([x["over512"] for x in rr])) if rr else None
            ),
            "evidence_chars": summarize([x["evidence_chars"] for x in rr]),
        }

    top14 = [x for x in rows if x["rank"] <= 4]
    rank5 = [x for x in rows if x["rank"] == 5]

    failure_by_rank = Counter(x["rank"] for x in failures)
    failure_reason = Counter(x["reason"] for x in failures)

    # Duplicate evidence text within the same query/top5 can signal packaging
    # collapse: multiple candidate docs effectively receive the same CE input.
    dup_queries = 0
    dup_extra_pairs = 0
    for q in d1:
        qr = [x for x in rows if x["qid"] == q]
        sigs = [x["evidence_sha1_like"] for x in qr]
        extra = len(sigs) - len(set(sigs))
        if extra > 0:
            dup_queries += 1
            dup_extra_pairs += extra

    report = {
        "schema": "manual.ce_evidence_packaging_contract_v1",
        "population": {
            "exact_d1_queries": len(d1),
            "renderer_covered_queries": len(covered),
            "outside_renderer_queries": outside,
            "expected_top5_pairs": len(d1) * 5,
            "successful_packages": len(rows),
            "failed_packages": len(failures),
        },
        "package_shapes": dict(shape_counts),
        "by_rank": by_rank,
        "top1_4": {
            "successful_packages": len(top14),
            "over512": sum(x["over512"] for x in top14),
            "over512_rate": (
                float(np.mean([x["over512"] for x in top14])) if top14 else None
            ),
        },
        "rank5": {
            "successful_packages": len(rank5),
            "over512": sum(x["over512"] for x in rank5),
            "over512_rate": (
                float(np.mean([x["over512"] for x in rank5])) if rank5 else None
            ),
        },
        "failures_by_rank": {str(k): v for k, v in sorted(failure_by_rank.items())},
        "failure_reasons": dict(failure_reason),
        "duplicate_packaged_evidence": {
            "queries_with_duplicate_evidence_text_within_top5": dup_queries,
            "extra_duplicate_pairs": dup_extra_pairs,
        },
        "risk_flags": {
            "rank5_missingness_asymmetric": bool(
                failure_by_rank.get(5, 0)
                > max(
                    [failure_by_rank.get(r, 0) for r in (1, 2, 3, 4)]
                    or [0]
                )
            ),
            "rank5_truncation_material": bool(
                rank5
                and np.mean([x["over512"] for x in rank5]) >= .01
            ),
            "top14_truncation_material": bool(
                top14
                and np.mean([x["over512"] for x in top14]) >= .01
            ),
            "duplicate_package_material": bool(dup_queries >= 5),
        },
        "provenance": {
            "d1_top5": str(d1_path),
            "ce_contract_script": str(base_script),
            "evidence_db": str(
                sibling / "cache/exp_final_retrieval/evidence.sqlite"
            ),
        },
        "public_labels_used": False,
        "cal_gold_used": False,
        "model_forward_performed": False,
        "rows": rows,
        "failures": failures,
    }

    out = root / "results/manual/huy_ce_evidence_packaging_contract_v1"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "REPORT.json"
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("[5/5] RESULT")
    print("=" * 116)
    print(
        f"Renderer coverage: {len(covered)}/{len(d1)} queries; "
        f"outside={outside}"
    )
    print(
        f"Evidence.package success={len(rows)}/{len(d1)*5}; "
        f"failures={len(failures)} reasons={dict(failure_reason)}"
    )
    print(
        "Top1-4 >512: "
        f"{report['top1_4']['over512']}/{len(top14)} "
        f"({100*(report['top1_4']['over512_rate'] or 0):.2f}%)"
    )
    print(
        "Rank5 >512: "
        f"{report['rank5']['over512']}/{len(rank5)} "
        f"({100*(report['rank5']['over512_rate'] or 0):.2f}%)"
    )
    print(
        "Duplicate packaged evidence within Top5: "
        f"queries={dup_queries} extra_pairs={dup_extra_pairs}"
    )
    print("Risk flags:", report["risk_flags"])
    print("Report:", path)
    print("=" * 116)


if __name__ == "__main__":
    main()
