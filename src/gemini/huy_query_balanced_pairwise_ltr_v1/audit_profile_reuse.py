"""Audit duplicate links in V2 baseline and verify profile BM25 rankings cache provenance."""

from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path
from typing import Any, Dict, Set, Tuple

ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = ROOT / "results" / "gemini" / "huy_query_balanced_pairwise_ltr_v1"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

PROFILE_CACHE_PATH = ROOT / "results" / "gemini" / "huy_fulltrain_profile_port_v1" / "PROFILE_BM25_RANKINGS.pkl"
EXPECTED_PROFILE_SHA256 = "a240d000b9e1d342bf50a8f2a935b39bce1f91114dff5d33996f2a69f2395826"
V2_BASELINE_PATH = ROOT / "results" / "research_v2_forensic" / "V2_EXECUTABLE_BASELINE.json"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def audit_duplicate_links() -> Dict[str, Any]:
    if not V2_BASELINE_PATH.exists():
        raise FileNotFoundError(f"Missing {V2_BASELINE_PATH}")

    data = json.loads(V2_BASELINE_PATH.read_text(encoding="utf-8"))
    dup = data.get("duplicate_contamination", {})

    exact_norm = dup.get("exact_normalized", {})
    near_dup = dup.get("near_duplicate_char_tfidf", {})

    exact_groups = exact_norm.get("groups", 0)
    exact_examples = exact_norm.get("examples", [])
    near_pairs = near_dup.get("pairs", 0)
    near_examples = near_dup.get("examples", [])

    assert exact_groups == 16, f"Expected 16 exact groups, got {exact_groups}"
    assert len(exact_examples) == 16, f"Expected 16 exact examples, got {len(exact_examples)}"
    assert near_pairs == 20, f"Expected 20 near pairs, got {near_pairs}"
    assert len(near_examples) == 20, f"Expected 20 near examples, got {len(near_examples)}"

    directed_links: Set[Tuple[str, str]] = set()

    for ex in exact_examples:
        qids = ex.get("qids", [])
        for i in range(len(qids)):
            for j in range(len(qids)):
                if i != j:
                    directed_links.add((str(qids[i]), str(qids[j])))

    for ex in near_examples:
        qa = str(ex.get("qid_a"))
        qb = str(ex.get("qid_b"))
        directed_links.add((qa, qb))
        directed_links.add((qb, qa))

    num_directed_links = len(directed_links)
    assert num_directed_links == 50, f"Expected 50 directed links, got {num_directed_links}"

    return {
        "v2_baseline_path": str(V2_BASELINE_PATH.relative_to(ROOT)).replace("\\", "/"),
        "exact_normalized_groups": exact_groups,
        "exact_normalized_examples": len(exact_examples),
        "near_duplicate_pairs": near_pairs,
        "near_duplicate_examples": len(near_examples),
        "unique_bidirectional_directed_links": num_directed_links,
        "assertion_passed": True,
    }


def audit_profile_cache() -> Dict[str, Any]:
    if not PROFILE_CACHE_PATH.exists():
        raise FileNotFoundError(f"Missing {PROFILE_CACHE_PATH}")

    actual_sha = sha256_file(PROFILE_CACHE_PATH)
    sha_matches = actual_sha == EXPECTED_PROFILE_SHA256
    assert sha_matches, f"SHA256 mismatch for {PROFILE_CACHE_PATH}: {actual_sha} vs {EXPECTED_PROFILE_SHA256}"

    obj = pickle.loads(PROFILE_CACHE_PATH.read_bytes())
    required_keys = [
        "nested_cal_rankings",
        "cal_oof_rankings",
        "cal_lexical_support",
        "public_rankings",
        "public_lexical_support",
    ]
    for k in required_keys:
        assert k in obj, f"Key {k} missing from profile cache"

    nested_cal = obj["nested_cal_rankings"]
    expected_blocks = {"a", "b", "c", "d"}
    assert set(nested_cal.keys()) == expected_blocks, f"Unexpected nested cal blocks: {set(nested_cal.keys())}"

    for b in expected_blocks:
        assert "held_ranks" in nested_cal[b], f"held_ranks missing for block {b}"
        assert "train_ranks" in nested_cal[b], f"train_ranks missing for block {b}"

    cal_oof = obj["cal_oof_rankings"]
    assert len(cal_oof) == 600, f"Expected 600 cal oof rankings, got {len(cal_oof)}"

    pub_rankings = obj["public_rankings"]
    assert len(pub_rankings) == 1000, f"Expected 1000 public rankings, got {len(pub_rankings)}"

    return {
        "profile_cache_path": str(PROFILE_CACHE_PATH.relative_to(ROOT)).replace("\\", "/"),
        "sha256": actual_sha,
        "expected_sha256": EXPECTED_PROFILE_SHA256,
        "sha256_matches": sha_matches,
        "cal_oof_queries": len(cal_oof),
        "public_queries": len(pub_rankings),
        "nested_blocks": sorted(list(nested_cal.keys())),
        "provenance_verified": True,
    }


def main() -> Dict[str, Any]:
    dup_audit = audit_duplicate_links()
    cache_audit = audit_profile_cache()

    result = {
        "schema_version": "dsc2026.gemini.huy_query_balanced_pairwise_ltr_v1.profile_reuse_audit.v1",
        "duplicate_links_audit": dup_audit,
        "profile_cache_audit": cache_audit,
        "status": "PASS",
    }

    out_file = RESULTS_DIR / "PROFILE_REUSE_AUDIT.json"
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"PROFILE_REUSE_AUDIT: status={result['status']} (50 directed links verified, profile cache SHA matched)")
    return result


if __name__ == "__main__":
    main()
