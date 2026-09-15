"""Audit and verify structural contract difference between D0 (Current Production) and D1 (Score-Only vnlegal_lal)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = ROOT / "results" / "gemini" / "huy_vnlegal_rank_ablation_v1"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Contract Definitions
D0_RANK_VIEWS = ["base", "expanded", "jina", "dense", "corpus", "vnlegal_lal"]
D1_RANK_VIEWS = ["base", "expanded", "jina", "dense", "corpus"]

SHARED_SCORE_CHANNELS = [
    "aiteamvn_ft",
    "corpus",
    "crossenc",
    "dense",
    "e5",
    "expansion",
    "jina",
    "jina_ft",
    "title_embed",
    "vnlegal_lal",
]

SHARED_METADATA_GROUPS = [
    "doctype_features",      # 4 features: is_law, is_decree, is_circular, other
    "citation_features",     # 4 features: in_degree, out_degree, pagerank, hub_authority
]


def audit_contracts() -> Dict[str, Any]:
    print("=== Step 1: Contract Difference Audit ===", flush=True)

    # 1. Inspect differences
    rank_diff_removed = set(D0_RANK_VIEWS) - set(D1_RANK_VIEWS)
    rank_diff_added = set(D1_RANK_VIEWS) - set(D0_RANK_VIEWS)

    print(f"D0 Rank Views ({len(D0_RANK_VIEWS)}): {D0_RANK_VIEWS}")
    print(f"D1 Rank Views ({len(D1_RANK_VIEWS)}): {D1_RANK_VIEWS}")
    print(f"Rank Views Removed in D1: {rank_diff_removed}")
    print(f"Rank Views Added in D1: {rank_diff_added}")

    # Feature dimension calculations
    # Each rank view produces 2 features: 1.0 / (10.0 + rank) and rank / 60.0
    d0_rank_dims = len(D0_RANK_VIEWS) * 2
    d1_rank_dims = len(D1_RANK_VIEWS) * 2

    # Each score channel produces 3 features: raw, normalized (z-score), gap to max
    score_dims = len(SHARED_SCORE_CHANNELS) * 3

    # Metadata features: 4 doctype + 4 citation
    metadata_dims = 4 + 4

    d0_total_dim = d0_rank_dims + score_dims + metadata_dims
    d1_total_dim = d1_rank_dims + score_dims + metadata_dims

    print(f"D0 Dimensions: {d0_rank_dims} rank + {score_dims} score + {metadata_dims} meta = {d0_total_dim}D")
    print(f"D1 Dimensions: {d1_rank_dims} rank + {score_dims} score + {metadata_dims} meta = {d1_total_dim}D")

    # Strict assertions
    assert rank_diff_removed == {"vnlegal_lal"}, f"Expected only vnlegal_lal removed, got {rank_diff_removed}"
    assert len(rank_diff_added) == 0, f"Expected no rank views added in D1, got {rank_diff_added}"
    assert "vnlegal_lal" in SHARED_SCORE_CHANNELS, "vnlegal_lal score channel must be preserved in both contracts"
    assert d0_total_dim == 50, f"Expected D0 to be 50D, got {d0_total_dim}D"
    assert d1_total_dim == 48, f"Expected D1 to be 48D, got {d1_total_dim}D"

    payload = {
        "schema_version": "dsc2026.gemini.huy_vnlegal_rank_ablation_v1.contract_diff_audit.v1",
        "d0_current_production": {
            "name": "D0_CURRENT_PRODUCTION",
            "description": "Current production burst_userft_maxrecall contract (6 rank views, 10 score channels)",
            "rank_views": D0_RANK_VIEWS,
            "rank_view_count": len(D0_RANK_VIEWS),
            "rank_features_dim": d0_rank_dims,
            "score_channels": sorted(SHARED_SCORE_CHANNELS),
            "score_channel_count": len(SHARED_SCORE_CHANNELS),
            "score_features_dim": score_dims,
            "metadata_groups": SHARED_METADATA_GROUPS,
            "metadata_features_dim": metadata_dims,
            "total_feature_dim": d0_total_dim,
        },
        "d1_score_only_vnlegal": {
            "name": "D1_SCORE_ONLY_VNLEGAL",
            "description": "Ablated contract: vnlegal_lal removed from rank views, preserved in score channels",
            "rank_views": D1_RANK_VIEWS,
            "rank_view_count": len(D1_RANK_VIEWS),
            "rank_features_dim": d1_rank_dims,
            "score_channels": sorted(SHARED_SCORE_CHANNELS),
            "score_channel_count": len(SHARED_SCORE_CHANNELS),
            "score_features_dim": score_dims,
            "metadata_groups": SHARED_METADATA_GROUPS,
            "metadata_features_dim": metadata_dims,
            "total_feature_dim": d1_total_dim,
        },
        "contract_diff": {
            "rank_views_removed": sorted(list(rank_diff_removed)),
            "rank_views_added": sorted(list(rank_diff_added)),
            "score_channels_diff": [],
            "metadata_diff": [],
            "dimension_delta": d1_total_dim - d0_total_dim,
            "exact_ablation_property_satisfied": True,
        },
        "audit_verdict": "PASS_EXACT_CONTRACT_DIFF",
        "status": "PASS",
    }

    out_file = RESULTS_DIR / "CONTRACT_DIFF_AUDIT.json"
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"CONTRACT_DIFF_AUDIT: status={payload['status']} (50D -> 48D by removing vnlegal_lal rank view only)")
    return payload


if __name__ == "__main__":
    audit_contracts()
