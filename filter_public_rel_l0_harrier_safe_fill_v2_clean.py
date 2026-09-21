#!/usr/bin/env python
"""
HUY DEADLINE — FILTERED PUBLIC REL_L0 + HAR(R)IER SAFE FILL V2
==============================================================

Purpose:
  Rebuild the Harrier safe-fill submission while EXCLUDING every public query
  whose normalized question text occurs in Harrier TRAIN or WARMUP results.

Why:
  split-content audit found real public-content overlap with Harrier train/warmup.
  We want an uncontaminated public probe that can say something about
  generalization, not merely exploit overlap.

Contract:
  - Start from exact public D1.
  - Discover exact REL_L0 public artifact (133 K4 + 867 K5, each K4 == D1[:4]).
  - Only on REL_L0-safe AND content-clean public queries:
      D1[:4] + first Harrier top50 document outside original D1 top5.
  - On contaminated or non-safe queries: keep exact D1 top5.
  - No public labels are read.

Input produced by:
  run_public_rel_l0_harrier_safe_fill_v1.py
    results/manual/huy_public_rel_l0_harrier_safe_fill_v1/
      PUBLIC_HARRIER_GLOBAL_TOP50.json
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
import zipfile
from pathlib import Path

WS_RE = re.compile(r"\s+")


def norm(s):
    s = unicodedata.normalize("NFKC", str(s or "")).lower().strip()
    return WS_RE.sub(" ", s)


def get_answer(v):
    if isinstance(v, dict):
        return [str(x) for x in v.get("answer", [])]
    if isinstance(v, list):
        return [str(x) for x in v]
    raise TypeError(type(v))


def find_named(base: Path, name: str):
    p = base / name
    if p.is_file():
        return p
    hits = list(base.rglob(name))
    return hits[0] if hits else None


def extract_questions(path: Path):
    obj = json.loads(path.read_text(encoding="utf-8"))
    out = {}
    for q, v in obj.items():
        if isinstance(v, dict):
            text = v.get("question") or v.get("query") or v.get("text") or ""
        else:
            text = str(v)
        out[str(q)] = str(text)
    return out


def discover_rel_l0(root: Path, d1: dict, explicit: Path | None):
    def validate(payload, source):
        if not isinstance(payload, dict) or set(payload) != set(d1):
            return None
        safe = []
        for q in d1:
            a = get_answer(payload[q])
            b = get_answer(d1[q])
            if len(a) == 4 and a == b[:4]:
                safe.append(q)
            elif len(a) == 5 and a == b:
                pass
            else:
                return None
        if len(safe) == 133:
            return {"source": str(source), "safe_qids": safe}
        return None

    if explicit:
        if explicit.suffix.lower() == ".zip":
            with zipfile.ZipFile(explicit) as z:
                payload = json.loads(z.read("submission.json").decode("utf-8"))
        else:
            payload = json.loads(explicit.read_text(encoding="utf-8"))
        got = validate(payload, explicit)
        if not got:
            raise RuntimeError("Explicit REL_L0 artifact failed 133/867 contract")
        return got

    candidates = []
    for p in (root / "results").rglob("*"):
        if not p.is_file() or p.suffix.lower() not in (".json", ".zip"):
            continue
        lo = p.name.lower()
        if "rel_l0" in lo or "rank5" in lo or "veto" in lo:
            candidates.append(p)
    candidates.sort(
        key=lambda p: (
            0 if "rel_l0" in p.name.lower() else 1,
            -p.stat().st_mtime,
        )
    )

    for p in candidates:
        try:
            if p.suffix.lower() == ".zip":
                with zipfile.ZipFile(p) as z:
                    if "submission.json" not in z.namelist():
                        continue
                    payload = json.loads(
                        z.read("submission.json").decode("utf-8")
                    )
            else:
                if p.stat().st_size > 20_000_000:
                    continue
                payload = json.loads(p.read_text(encoding="utf-8"))
            got = validate(payload, p)
            if got:
                return got
        except Exception:
            continue

    raise FileNotFoundError(
        "Could not auto-discover REL_L0 public artifact. "
        "Pass --rel-l0-json."
    )


def write_zip(json_path: Path, zip_path: Path, payload):
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("submission.json", json_path.read_bytes())
    with zipfile.ZipFile(zip_path) as z:
        assert z.namelist() == ["submission.json"]
        assert z.read("submission.json") == json_path.read_bytes()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=Path, required=True)
    ap.add_argument("--model-root", type=Path, required=True)
    ap.add_argument("--rel-l0-json", type=Path, default=None)
    ap.add_argument(
        "--harrier-ranking-json",
        type=Path,
        default=None,
    )
    args = ap.parse_args()

    root = args.repo_root.resolve()
    model_root = args.model_root.resolve()

    d1_path = (
        root
        / "results/gemini/huy_vnlegal_rank_ablation_v1/"
        "CANDIDATE_D1_VNLEGAL_SCORE_ONLY.json"
    )
    d1 = json.loads(d1_path.read_text(encoding="utf-8"))
    if len(d1) != 1000:
        raise RuntimeError(f"Expected D1 1000 qids, got {len(d1)}")

    rel = discover_rel_l0(root, d1, args.rel_l0_json)
    safe = set(rel["safe_qids"])

    rank_path = args.harrier_ranking_json or (
        root
        / "results/manual/huy_public_rel_l0_harrier_safe_fill_v1/"
        "PUBLIC_HARRIER_GLOBAL_TOP50.json"
    )
    if not rank_path.is_file():
        raise FileNotFoundError(
            f"Harrier ranking not ready yet: {rank_path}\n"
            "Wait for run_public_rel_l0_harrier_safe_fill_v1.py to finish."
        )
    raw_rank = json.loads(rank_path.read_text(encoding="utf-8"))
    rankings = {
        str(q): [str(x["ctx_id"]) for x in v.get("results", [])]
        for q, v in raw_rank.items()
    }

    public_path = (
        root
        / "DSC2026-LegalIR-main/v4_run/public_test_dataset/"
        "public-official.json"
    )
    public = extract_questions(public_path)

    train_path = find_named(model_root, "train_finetuned_best.json")
    warm_path = find_named(model_root, "warmup_finetuned_best.json")
    if not train_path or not warm_path:
        raise FileNotFoundError("Missing Harrier train/warmup result files")
    train = extract_questions(train_path)
    warm = extract_questions(warm_path)

    train_texts = {norm(t) for t in train.values() if norm(t)}
    warm_texts = {norm(t) for t in warm.values() if norm(t)}

    contaminated = {}
    for q, text in public.items():
        nt = norm(text)
        sources = []
        if nt and nt in train_texts:
            sources.append("TRAIN")
        if nt and nt in warm_texts:
            sources.append("WARMUP")
        if sources:
            contaminated[q] = sources

    safe_contaminated = sorted(safe & set(contaminated))
    clean_safe = sorted(safe - set(contaminated))

    # Validate corpus IDs.
    ctx_dir = (
        root
        / "DSC2026-LegalIR-main/v4_run/public_test_dataset/"
        "selected-contexts"
    )
    valid_docs = {
        p.stem[len("context_"):]
        for p in ctx_dir.glob("context_*.json")
    }

    candidate = {
        q: {"answer": list(get_answer(d1[q]))}
        for q in d1
    }
    actions = []
    abstain = []

    for q in clean_safe:
        base = get_answer(d1[q])
        forbidden = set(base)
        challenger = next(
            (d for d in rankings.get(q, []) if d not in forbidden),
            None,
        )
        if challenger is None:
            abstain.append(q)
            continue
        candidate[q] = {"answer": base[:4] + [challenger]}
        actions.append({
            "qid": q,
            "dropped_d1_rank5": base[4],
            "challenger": challenger,
            "harrier_rank": rankings[q].index(challenger) + 1,
        })

    for q, row in candidate.items():
        a = get_answer(row)
        if len(a) != 5 or len(set(a)) != 5:
            raise RuntimeError(f"Invalid K5 q={q}: {a}")
        bad = [d for d in a if d not in valid_docs]
        if bad:
            raise RuntimeError(f"Invalid docs q={q}: {bad}")

    out = (
        root
        / "results/manual/huy_public_rel_l0_harrier_safe_fill_v2_clean"
    )
    out.mkdir(parents=True, exist_ok=True)
    out_json = out / "CANDIDATE_D1_REL_L0_HARRIER_CLEAN_SAFE_FILL.json"
    out_zip = out / "CANDIDATE_D1_REL_L0_HARRIER_CLEAN_SAFE_FILL.zip"
    write_zip(out_json, out_zip, candidate)

    changed = [
        q for q in d1
        if get_answer(candidate[q]) != get_answer(d1[q])
    ]

    report = {
        "schema": "manual.public_rel_l0_harrier_clean_safe_fill_v2",
        "d1_path": str(d1_path),
        "rel_l0_source": rel["source"],
        "harrier_ranking_path": str(rank_path),
        "train_result_path": str(train_path),
        "warmup_result_path": str(warm_path),
        "public_total": len(public),
        "public_content_contaminated_total": len(contaminated),
        "contaminated_by_source_counts": {
            "TRAIN": sum("TRAIN" in v for v in contaminated.values()),
            "WARMUP": sum("WARMUP" in v for v in contaminated.values()),
            "BOTH": sum(
                "TRAIN" in v and "WARMUP" in v
                for v in contaminated.values()
            ),
        },
        "rel_l0_safe_total": len(safe),
        "rel_l0_safe_contaminated": len(safe_contaminated),
        "rel_l0_safe_clean": len(clean_safe),
        "clean_safe_fill_actions": len(actions),
        "clean_safe_abstentions": len(abstain),
        "changed_vs_d1": len(changed),
        "safe_contaminated_qids": safe_contaminated,
        "contaminated_public": contaminated,
        "actions": actions,
        "abstain_qids": abstain,
        "public_labels_used": False,
        "submission_json": str(out_json),
        "submission_zip": str(out_zip),
        "interpretation": (
            "Only exact NFKC+lower+whitespace-normalized question-text overlap "
            "with Harrier train/warmup is excluded. This is a conservative "
            "content-leakage filter; semantic near-duplicates may still exist."
        ),
    }
    rp = out / "REPORT.json"
    rp.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("=" * 116)
    print(f"Public contaminated questions : {len(contaminated)}/1000")
    print(
        "  TRAIN/WARMUP/BOTH          : "
        f"{report['contaminated_by_source_counts']['TRAIN']}/"
        f"{report['contaminated_by_source_counts']['WARMUP']}/"
        f"{report['contaminated_by_source_counts']['BOTH']}"
    )
    print(f"REL_L0 safe total             : {len(safe)}")
    print(f"REL_L0 safe contaminated      : {len(safe_contaminated)}")
    print(f"REL_L0 safe CLEAN             : {len(clean_safe)}")
    print(f"Clean Harrier fill actions    : {len(actions)}")
    print(f"Changed vs exact D1           : {len(changed)}")
    print("SUBMIT CLEAN ZIP:", out_zip)
    print("Report:", rp)
    print("=" * 116)


if __name__ == "__main__":
    main()
