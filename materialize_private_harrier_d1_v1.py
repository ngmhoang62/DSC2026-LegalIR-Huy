#!/usr/bin/env python
"""
MATERIALIZE PRIVATE D1 + HAR(R)IER RANK-SPECIALIST SUBMISSION
=============================================================

CPU-only. No private labels. No neural inference. No REL_L0.

Requires:
  - confirmed D1 v14 private JSON + decision score cache
  - CAL_HARRIER_POLICY.json from audit_harrier_d1_gate_v1.py
  - PRIVATE_HARRIER_TOP200.json or PRIVATE_HARRIER_PARENT_RANKS.pkl

Safety:
  1. exact D1 ordered-top5 parity between JSON and cache is mandatory;
  2. policy may only promote an existing D1 candidate from ranks 6..TOPN;
  3. ranks 1..4 are frozen;
  4. at most one swap/query;
  5. output has exactly 2080 qids, K=5, no duplicates;
  6. confirmed D1 artifacts are never overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import zipfile
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def dump(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def zip_exact(json_path: Path, zip_path: Path):
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(
        zip_path,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as zf:
        zf.write(json_path, arcname="submission.json")
    with zipfile.ZipFile(zip_path, "r") as zf:
        if zf.namelist() != ["submission.json"]:
            raise RuntimeError("ZIP member contract failed")
        if zf.read("submission.json") != json_path.read_bytes():
            raise RuntimeError("ZIP byte parity failed")


def load_parent_ranks(path: Path):
    if path.suffix.lower() in (".pkl", ".pickle"):
        obj = pickle.loads(path.read_bytes())
        return {
            str(q): {str(d): int(r) for d, r in row.items()}
            for q, row in obj.items()
        }

    raw = json.loads(path.read_text(encoding="utf-8"))
    out = {}
    for q, row in raw.items():
        rank = {}
        pos = 0
        for x in row.get("results", []):
            d = str(x.get("ctx_id", x.get("doc_id", "")))
            if not d or d in rank:
                continue
            pos += 1
            rank[d] = pos
        out[str(q)] = rank
    return out


def apply_policy(
    d1_order,
    h_rank,
    *,
    topn,
    hmax,
    gap,
    missing_rank=1000,
):
    top5 = list(d1_order[:5])
    if len(d1_order) < 6:
        return top5, None

    defender = top5[4]
    defender_hr = int(h_rank.get(defender, missing_rank))

    eligible = []
    for i in range(5, min(len(d1_order), int(topn))):
        doc = d1_order[i]
        hr = int(h_rank.get(doc, missing_rank))
        if hr <= int(hmax) and (defender_hr - hr) >= int(gap):
            eligible.append((hr, i + 1, doc))

    if not eligible:
        return top5, None

    hr, d1_rank, challenger = min(eligible)
    return top5[:4] + [challenger], {
        "defender": defender,
        "challenger": challenger,
        "defender_harrier_rank": defender_hr,
        "challenger_harrier_rank": hr,
        "challenger_d1_rank": d1_rank,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=Path, required=True)
    ap.add_argument("--policy", type=Path, default=None)
    ap.add_argument("--harrier-private", type=Path, default=None)
    args = ap.parse_args()

    root = args.repo_root.expanduser().resolve()
    d1_dir = root / "results/manual/huy_private_d1_rel_l0_exact_v1"
    d1_json_path = d1_dir / "D1_PRIVATE_V14_FAST.json"
    d1_cache_path = d1_dir / "cache/d1_private_scores.pkl"

    if not d1_json_path.is_file():
        raise FileNotFoundError(d1_json_path)
    if not d1_cache_path.is_file():
        raise FileNotFoundError(d1_cache_path)

    policy_path = (
        args.policy.expanduser().resolve()
        if args.policy is not None
        else root / "results/manual/huy_harrier_d1_gate_v1/CAL_HARRIER_POLICY.json"
    )
    if not policy_path.is_file():
        raise FileNotFoundError(
            f"Promoted Harrier policy missing: {policy_path}. "
            "Run CPU gate first; do not materialize a killed branch."
        )

    if args.harrier_private is not None:
        harrier_path = args.harrier_private.expanduser().resolve()
    else:
        default_pkl = (
            root
            / "results/manual/huy_private_harrier_ft_v1/"
            "PRIVATE_HARRIER_PARENT_RANKS.pkl"
        )
        default_json = (
            root
            / "results/manual/huy_private_harrier_ft_v1/"
            "PRIVATE_HARRIER_TOP200.json"
        )
        harrier_path = default_pkl if default_pkl.is_file() else default_json

    if not harrier_path.is_file():
        raise FileNotFoundError(harrier_path)

    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    if policy.get("policy_type") != "d1_rank5_single_swap":
        raise RuntimeError(
            f"Unsupported policy_type={policy.get('policy_type')}"
        )

    d1_json = json.loads(d1_json_path.read_text(encoding="utf-8"))
    cache = pickle.loads(d1_cache_path.read_bytes())
    if not isinstance(cache, dict):
        raise RuntimeError("Unsupported D1 score cache")
    d1_top5 = cache.get("top5")
    d1_scores = cache.get("decision_scores")
    if not isinstance(d1_top5, dict) or not isinstance(d1_scores, dict):
        raise RuntimeError(
            "d1_private_scores.pkl must contain top5 + decision_scores"
        )

    ids = sorted(d1_json)
    if len(ids) != 2080:
        raise RuntimeError(f"Expected 2080 D1 qids, got {len(ids)}")

    # Mandatory ordered top5 parity against the confirmed submission.
    parity_fail = []
    for q in ids:
        json_top5 = list(map(str, d1_json[q]["answer"]))
        cache_top5 = list(map(str, d1_top5[q]))
        if json_top5 != cache_top5:
            parity_fail.append((q, json_top5, cache_top5))
    if parity_fail:
        raise RuntimeError(
            f"D1 ordered Top5 parity failed for {len(parity_fail)} qids; "
            f"first={parity_fail[:3]}"
        )
    print("[PASS] D1 ordered Top5 parity 2080/2080", flush=True)

    harrier = load_parent_ranks(harrier_path)
    missing_qids = [q for q in ids if q not in harrier]
    if missing_qids:
        raise RuntimeError(
            f"Harrier private missing {len(missing_qids)} qids; "
            f"first={missing_qids[:10]}"
        )

    cfg = {
        "topn": int(policy["topn"]),
        "hmax": int(policy["hmax"]),
        "gap": int(policy["gap"]),
        "missing_rank": int(policy.get("missing_rank", 1000)),
    }

    submission = {}
    changes = {}
    for q in ids:
        score_row = {
            str(d): float(s)
            for d, s in d1_scores[q].items()
        }
        order = sorted(
            score_row,
            key=lambda d: (-score_row[d], d),
        )
        # Full-score cache must reproduce confirmed D1 top5 as an extra guard.
        if order[:5] != list(map(str, d1_top5[q])):
            raise RuntimeError(
                f"D1 full decision-score order != cached top5 for q={q}"
            )

        top5, change = apply_policy(
            order,
            harrier[q],
            **cfg,
        )
        if len(top5) != 5 or len(set(top5)) != 5:
            raise RuntimeError(f"Invalid top5 q={q}: {top5}")

        submission[q] = {"answer": top5}
        if change is not None:
            changes[q] = change

    if set(submission) != set(d1_json):
        raise RuntimeError("Submission qid population changed")

    out = root / "results/manual/huy_private_harrier_d1_v1"
    out.mkdir(parents=True, exist_ok=True)

    json_path = out / "D1_HARRIER_RANK_SPECIALIST_V1.json"
    zip_path = out / "D1_HARRIER_RANK_SPECIALIST_V1.zip"
    dump(json_path, submission)
    zip_exact(json_path, zip_path)

    report = {
        "schema": "manual.private_harrier_d1_v1",
        "status": "READY_FOR_PRIVATE_SUBMISSION",
        "private_labels_used": False,
        "rel_l0_used": False,
        "base": {
            "json": str(d1_json_path),
            "json_sha256": sha256(d1_json_path),
            "score_cache": str(d1_cache_path),
            "score_cache_sha256": sha256(d1_cache_path),
            "ordered_top5_parity": "PASS_2080_2080",
        },
        "harrier": {
            "path": str(harrier_path),
            "sha256": sha256(harrier_path),
        },
        "policy": policy,
        "private_application": {
            "queries": len(ids),
            "changed_queries": len(changes),
            "changed_fraction": len(changes) / len(ids),
            "changes": changes,
        },
        "submission": {
            "json": str(json_path),
            "json_sha256": sha256(json_path),
            "zip": str(zip_path),
            "zip_sha256": sha256(zip_path),
        },
    }
    report_path = out / "PRIVATE_HARRIER_D1_REPORT.json"
    dump(report_path, report)

    print("=" * 108)
    print("D1 + HAR(R)IER RANK-SPECIALIST READY")
    print("Policy:", cfg)
    print(f"Changed queries: {len(changes)}/{len(ids)}")
    print("ZIP:", zip_path)
    print("Report:", report_path)
    print("=" * 108)


if __name__ == "__main__":
    main()
