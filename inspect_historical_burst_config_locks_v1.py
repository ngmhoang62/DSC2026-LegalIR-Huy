#!/usr/bin/env python
"""
FORENSIC: inspect local historical BURST config locks
=====================================================

Reads only local artifacts. Does not modify anything.

Purpose:
- inspect top-level metadata in cpu_top20.pkl / gpu_scores.checkpoint.pkl
- inspect untracked historical tuner reports if they still exist locally
- compare historical best configs against the newly rebuilt model artifacts
- verify whether run_burst_multistage_submission.py's hard-coded stack matches
  the historical multistage validation report

Run:
  python ../inspect_historical_burst_config_locks_v1.py \
    --repo-root /d/Study/DSC2026/sota
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path


def load_pkl(p: Path):
    return pickle.loads(p.read_bytes())


def compact(x, depth=0):
    if depth >= 3:
        if isinstance(x, dict):
            return f"<dict:{len(x)}>"
        if isinstance(x, (list, tuple)):
            return f"<{type(x).__name__}:{len(x)}>"
        return x
    if isinstance(x, dict):
        out = {}
        for k, v in x.items():
            if k in {"model", "scaler", "predictions", "rankings", "scores", "cache"}:
                if isinstance(v, dict):
                    out[k] = f"<dict:{len(v)}>"
                else:
                    out[k] = f"<{type(v).__name__}>"
            else:
                out[k] = compact(v, depth + 1)
        return out
    if isinstance(x, (list, tuple)):
        if len(x) > 20:
            return [compact(v, depth + 1) for v in x[:5]] + [f"... <{len(x)} total>"]
        return [compact(v, depth + 1) for v in x]
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    return f"<{type(x).__name__}>"


def jprint(title, obj):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)
    print(json.dumps(compact(obj), ensure_ascii=False, indent=2, default=str))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=Path, required=True)
    args = ap.parse_args()
    root = args.repo_root.resolve()

    print("Repo:", root)

    # 1. Inspect historical output artifacts themselves.
    artifact_paths = [
        root / "results/burst_gpu_threeview/cpu_top20.pkl",
        root / "results/burst_gpu_threeview/gpu_scores.checkpoint.pkl",
    ]
    for p in artifact_paths:
        if not p.is_file():
            print(f"\nMISSING artifact: {p}")
            continue
        obj = load_pkl(p)
        jprint(f"ARTIFACT {p.relative_to(root)}", obj)

        if isinstance(obj, dict):
            print("Top-level keys:", list(obj.keys()))
            if "rankings" in obj and isinstance(obj["rankings"], dict):
                q = next(iter(obj["rankings"]))
                print("Sample ranking qid:", q)
                print("Sample ranking length:", len(obj["rankings"][q]))
                print("Sample ranking head:", obj["rankings"][q][:20])
            if "scores" in obj and isinstance(obj["scores"], dict):
                q = next(iter(obj["scores"]))
                print("Sample score qid:", q)
                row = obj["scores"][q]
                if isinstance(row, dict):
                    print("Sample per-query score keys:", list(row.keys())[:30])

    # 2. Historical validation reports may be untracked and therefore absent
    #    from GitHub while still present locally.
    report_paths = {
        "large": root / "burst_large_ltr_validation.json",
        "pair": root / "burst_empirical_pairwise_validation.json",
        "multi": root / "burst_multistage_posterior_validation.json",
    }
    reports = {}
    for name, p in report_paths.items():
        if p.is_file():
            reports[name] = json.loads(p.read_text(encoding="utf-8"))
            jprint(f"LOCAL VALIDATION REPORT: {p.name}", reports[name])
        else:
            print(f"\nMISSING local validation report: {p}")

    # 3. Current rebuilt artifact metadata.
    model_paths = {
        "large": root / "results/burst_large_ltr/best_model.pkl",
        "pair": root / "results/burst_empirical_pairwise/model.pkl",
        "legal": root / "results/burst_legal_features/validation_model.pkl",
    }
    models = {}
    for name, p in model_paths.items():
        if p.is_file():
            models[name] = load_pkl(p)
            jprint(f"CURRENT/REBUILT MODEL: {p.relative_to(root)}", models[name])
        else:
            print(f"\nMISSING current model artifact: {p}")

    # 4. Targeted comparison.
    print("\n" + "#" * 100)
    print("TARGETED CONFIG COMPARISON")
    print("#" * 100)

    # large LTR
    hist_large = None
    if "large" in reports:
        r = reports["large"]
        hist_large = {
            "kind": r.get("best", {}).get("kind"),
            "params": r.get("best", {}).get("params"),
            "blend_alpha": r.get("blend", {}).get("new_alpha"),
            "blend_k": r.get("blend", {}).get("rrf_k"),
        }
    cur_large = None
    if "large" in models:
        m = models["large"]
        cur_large = {
            "kind": m.get("kind"),
            "params": m.get("params"),
            "blend_alpha": m.get("blend_alpha"),
            "blend_k": m.get("blend_k"),
        }

    print("\nLARGE LTR")
    print("  historical report:", hist_large)
    print("  rebuilt artifact: ", cur_large)
    print("  exact config match:", hist_large == cur_large if hist_large is not None else "UNKNOWN")

    # empirical pairwise
    hist_pair = None
    if "pair" in reports:
        best = reports["pair"].get("best", {})
        hist_pair = {
            "pairwise": best.get("pairwise"),
            "fusion": best.get("fusion"),
            "safe": best.get("safe"),
            "gain": best.get("total_validation_recall_gain"),
        }
    cur_pair = None
    if "pair" in models:
        best = models["pair"].get("report", {})
        cur_pair = {
            "pairwise": best.get("pairwise"),
            "fusion": best.get("fusion"),
            "safe": best.get("safe"),
            "gain": best.get("total_validation_recall_gain"),
        }

    print("\nEMPIRICAL PAIRWISE")
    print("  historical report:", hist_pair)
    print("  rebuilt artifact: ", cur_pair)
    print("  exact config match:", hist_pair == cur_pair if hist_pair is not None else "UNKNOWN")

    # multistage
    hist_multi = None
    if "multi" in reports:
        best = reports["multi"].get("best", {})
        hist_multi = {
            "weights": best.get("weights_baseline_pairwise_profile_graph"),
            "rrf_k": best.get("rrf_k"),
            "safe": best.get("safe"),
            "gain": best.get("total_validation_recall_gain"),
        }

    hardcoded_multi = {
        "weights": [0.40, 0.30, 0.15, 0.15],
        "rrf_k": 0,
    }
    print("\nMULTISTAGE FINAL STACK")
    print("  historical report:", hist_multi)
    print("  current hard-coded:", hardcoded_multi)
    if hist_multi is not None:
        hw = hist_multi.get("weights")
        hk = hist_multi.get("rrf_k")
        match = (
            hw is not None
            and list(hw) == hardcoded_multi["weights"]
            and hk == hardcoded_multi["rrf_k"]
        )
        print("  hard-coded config matches historical report:", match)
    else:
        print("  hard-coded config match: UNKNOWN")

    # 5. Bottom-line diagnostic.
    flags = []
    if hist_large is not None and hist_large != cur_large:
        flags.append("LARGE_LTR_CONFIG_DRIFT")
    if hist_pair is not None and hist_pair != cur_pair:
        flags.append("PAIRWISE_CONFIG_DRIFT")
    if hist_multi is not None:
        hw = hist_multi.get("weights")
        hk = hist_multi.get("rrf_k")
        if hw is not None and (
            list(hw) != hardcoded_multi["weights"]
            or hk != hardcoded_multi["rrf_k"]
        ):
            flags.append("MULTISTAGE_CONFIG_DRIFT")

    print("\n" + "#" * 100)
    print("FORENSIC VERDICT")
    print("#" * 100)
    if flags:
        print("FOUND CONFIG DRIFT:", ", ".join(flags))
        print(
            "Next step: rebuild locked to the surviving historical report(s), "
            "then rerun public CPU/D1 parity."
        )
    elif reports:
        print(
            "No obvious config drift in surviving reports. "
            "Next step: reverse-engineer ranking-generation semantics / historical "
            "training cache differences against cpu_top20.pkl."
        )
    else:
        print(
            "No historical validation reports survive locally. "
            "Next step: output-match cpu_top20.pkl using generated component rankings."
        )


if __name__ == "__main__":
    main()
