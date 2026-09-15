import sys
import json
import time
from pathlib import Path
from statistics import mean
import torch
from torch.nn import functional as F
import numpy as np

REPO_ROOT = Path("d:/Study/DSC2026/sota").resolve()
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from src.research_v2_e5_transfer import e5_transfer_runner as core
from src.research_v2_e5_confirmation.e5_confirmation_runner import ConfirmationData

BUNDLE_DIR = REPO_ROOT / "cache" / "research_v2_e5_confirmation" / "bundle-v1"
RESULTS_DIR = REPO_ROOT / "results" / "gemini" / "huy_e5_transplant_corrected_v2"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

FOLD_INFO = {
    "fold_0": {
        "checkpoint": REPO_ROOT / "results/research_v2_e5_transfer/research_v2_e5_transfer_fold0/training/epoch-2.pt",
        "sealed_predictions": REPO_ROOT / "results/research_v2_e5_confirmation/fold0_runner_parity/E5_CONFIRMATION_FOLD_0_PREDICTIONS.jsonl",
    },
    "fold_1": {
        "checkpoint": REPO_ROOT / "results/research_v2_e5_confirmation/fold_1/training/epoch-2.pt",
        "sealed_predictions": REPO_ROOT / "results/research_v2_e5_confirmation/fold_1/score/E5_CONFIRMATION_FOLD_1_PREDICTIONS.jsonl",
    },
    "fold_2": {
        "checkpoint": REPO_ROOT / "results/research_v2_e5_confirmation/fold_2/training/epoch-2.pt",
        "sealed_predictions": REPO_ROOT / "results/research_v2_e5_confirmation/fold_2/score/E5_CONFIRMATION_FOLD_2_PREDICTIONS.jsonl",
    },
    "fold_3": {
        "checkpoint": REPO_ROOT / "results/research_v2_e5_confirmation/fold_3/training/epoch-2.pt",
        "sealed_predictions": REPO_ROOT / "results/research_v2_e5_confirmation/fold_3/score/E5_CONFIRMATION_FOLD_3_PREDICTIONS.jsonl",
    },
    "fold_4": {
        "checkpoint": REPO_ROOT / "results/research_v2_e5_confirmation/fold_4/training/epoch-2.pt",
        "sealed_predictions": REPO_ROOT / "results/research_v2_e5_confirmation/fold_4/score/E5_CONFIRMATION_FOLD_4_PREDICTIONS.jsonl",
    },
}

def run_parity():
    print("=== VERIFYING PARITY AGAINST SEALED STRICT-V2 SCORES ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load data bundle once
    print("Loading confirmation data bundle...", flush=True)
    data = ConfirmationData(BUNDLE_DIR, held_fold="fold_0")
    bank = core.ParentBank(data.vectors, data.parent, device=device)
    print("Corpus bank loaded.")

    audit_folds = {}
    all_adapted_top5_match = True
    all_frozen_top5_match = True
    global_adapted_max_err = 0.0
    global_frozen_max_err = 0.0

    for fold_name, info in FOLD_INFO.items():
        print(f"\n--- Checking {fold_name} ---")
        # Load sealed predictions
        sealed_rows = {}
        with open(info["sealed_predictions"], "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    sealed_rows[str(row["qid"])] = row

        # Deterministically select first 12 queries
        sample_qids = sorted(list(sealed_rows.keys()))[:12]
        print(f"Selected {len(sample_qids)} audit queries: {sample_qids[:4]}...")

        # Instantiate canonical QueryEncoder with held fold checkpoint
        model = core.QueryEncoder(BUNDLE_DIR / "vietlegal-e5", device=device, checkpoint_path=info["checkpoint"])
        model.eval()

        fold_adapted_max_err = 0.0
        fold_frozen_max_err = 0.0
        adapted_top5_matches = 0
        frozen_top5_matches = 0
        cosines = []
        l2_deltas = []
        adapted_diff_from_frozen_count = 0

        query_audits = []

        for qid in sample_qids:
            sealed = sealed_rows[qid]
            docs = data.pool[qid]
            question = data.questions[qid]

            with torch.no_grad():
                adapted_vec = model([question])[0]
                with model.adapter_disabled():
                    frozen_vec = model([question])[0]

            # Compare query vectors
            cos_sim = float(F.cosine_similarity(adapted_vec.unsqueeze(0), frozen_vec.unsqueeze(0)).item())
            l2_delta = float(torch.norm(adapted_vec - frozen_vec).item())
            cosines.append(cos_sim)
            l2_deltas.append(l2_delta)

            # Score candidates
            adapted_scores = bank.score_pool(adapted_vec, docs, data)
            frozen_scores = bank.score_pool(frozen_vec, docs, data)

            adapted_order = core.ranking(docs, adapted_scores)
            frozen_order = core.ranking(docs, frozen_scores)

            if adapted_order != frozen_order:
                adapted_diff_from_frozen_count += 1

            # Check vs sealed (mapping doc IDs to scores properly)
            calc_adapted_map = {doc_id: s for doc_id, s in zip(docs, adapted_scores)}
            calc_frozen_map = {doc_id: s for doc_id, s in zip(docs, frozen_scores)}
            sealed_adapted_map = {doc_id: s for doc_id, s in zip(sealed["ft_order"], sealed["ft_scores"])}
            sealed_frozen_map = {doc_id: s for doc_id, s in zip(sealed["base_order"], sealed["base_scores"])}

            # True numerical errors per doc
            a_err = max(abs(calc_adapted_map[d] - sealed_adapted_map[d]) for d in docs)
            f_err = max(abs(calc_frozen_map[d] - sealed_frozen_map[d]) for d in docs)
            fold_adapted_max_err = max(fold_adapted_max_err, a_err)
            fold_frozen_max_err = max(fold_frozen_max_err, f_err)

            # Top-5 set checks
            sealed_adapted_top5 = set(sealed["ft_order"][:5])
            calc_adapted_top5 = set(adapted_order[:5])
            adapted_match = (calc_adapted_top5 == sealed_adapted_top5)
            if adapted_match:
                adapted_top5_matches += 1

            sealed_frozen_top5 = set(sealed["base_order"][:5])
            calc_frozen_top5 = set(frozen_order[:5])
            frozen_match = (calc_frozen_top5 == sealed_frozen_top5)
            if frozen_match:
                frozen_top5_matches += 1

            query_audits.append({
                "qid": qid,
                "adapted_score_max_abs_err": a_err,
                "frozen_score_max_abs_err": f_err,
                "adapted_top5_set_match": adapted_match,
                "frozen_top5_set_match": frozen_match,
                "cosine_adapted_vs_frozen": cos_sim,
                "l2_delta_adapted_vs_frozen": l2_delta,
                "adapted_order_differs_from_frozen": (adapted_order != frozen_order)
            })

        print(f"  Adapted score max abs err: {fold_adapted_max_err:.2e}")
        print(f"  Frozen score max abs err: {fold_frozen_max_err:.2e}")
        print(f"  Adapted Top-5 set match: {adapted_top5_matches}/{len(sample_qids)} ({adapted_top5_matches/len(sample_qids)*100:.1f}%)")
        print(f"  Frozen Top-5 set match: {frozen_top5_matches}/{len(sample_qids)} ({frozen_top5_matches/len(sample_qids)*100:.1f}%)")
        print(f"  Mean cosine(adapted, frozen): {mean(cosines):.4f}")
        print(f"  Mean L2(adapted, frozen): {mean(l2_deltas):.4f}")
        print(f"  Adapted ranking differs from frozen: {adapted_diff_from_frozen_count}/{len(sample_qids)}")

        global_adapted_max_err = max(global_adapted_max_err, fold_adapted_max_err)
        global_frozen_max_err = max(global_frozen_max_err, fold_frozen_max_err)
        if adapted_top5_matches < len(sample_qids):
            all_adapted_top5_match = False
        if frozen_top5_matches < len(sample_qids):
            all_frozen_top5_match = False

        audit_folds[fold_name] = {
            "checkpoint": str(info["checkpoint"]),
            "sealed_predictions": str(info["sealed_predictions"]),
            "sample_queries_count": len(sample_qids),
            "adapted_score_max_abs_err": fold_adapted_max_err,
            "frozen_score_max_abs_err": fold_frozen_max_err,
            "adapted_top5_set_match_pct": adapted_top5_matches / len(sample_qids) * 100,
            "frozen_top5_set_match_pct": frozen_top5_matches / len(sample_qids) * 100,
            "mean_cosine_adapted_vs_frozen": mean(cosines),
            "mean_l2_delta_adapted_vs_frozen": mean(l2_deltas),
            "max_l2_delta_adapted_vs_frozen": max(l2_deltas),
            "fraction_queries_ranking_differs": adapted_diff_from_frozen_count / len(sample_qids),
            "queries": query_audits
        }

    # Summary
    pass_gate = all_adapted_top5_match and all_frozen_top5_match and (global_adapted_max_err < 1e-4) and (global_frozen_max_err < 1e-4)
    print("\n=== PARITY AUDIT SUMMARY ===")
    print(f"All Adapted Top-5 Set Matches 100%: {all_adapted_top5_match}")
    print(f"All Frozen Top-5 Set Matches 100%: {all_frozen_top5_match}")
    print(f"Global Max Adapted Score Error: {global_adapted_max_err:.2e}")
    print(f"Global Max Frozen Score Error: {global_frozen_max_err:.2e}")
    print(f"Parity Gate Status: {'PASS' if pass_gate else 'FAIL'}")

    report = {
        "schema_version": "dsc2026.gemini.huy_e5_transplant_corrected_v2.e5_canonical_parity_audit.v1",
        "parity_gate_passed": pass_gate,
        "global_summary": {
            "all_adapted_top5_match_100_pct": all_adapted_top5_match,
            "all_frozen_top5_match_100_pct": all_frozen_top5_match,
            "global_max_adapted_score_abs_err": global_adapted_max_err,
            "global_max_frozen_score_abs_err": global_frozen_max_err,
        },
        "folds": audit_folds
    }

    out_file = RESULTS_DIR / "E5_CANONICAL_PARITY_AUDIT.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Wrote report to {out_file}")

    if not pass_gate:
        raise RuntimeError("CANONICAL PARITY GATE FAILED! STOPPING EXPERIMENT.")

if __name__ == "__main__":
    run_parity()
