"""Stage 1: Audit Source Provenance and Frozen Jina Cross-Encoder Model."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from .common import (
    REPO_JINA,
    RESULTS_DIR,
    SRC_DIR,
    WEIGHTS_JINA_FT,
    get_git_status,
    get_source_files_sha256,
    load_jina_crossencoder,
    sha256_file,
)


def audit_source_and_model_provenance() -> Dict[str, Any]:
    print("=== STAGE 1: AUDIT SOURCE AND FROZEN MODEL PROVENANCE ===", flush=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    git_info = get_git_status()

    # 1. Source Provenance
    source_files = get_source_files_sha256()
    source_provenance = {
        "schema_version": "dsc2026.gemini.huy_d1_frozen_section_public_v1.source_provenance.v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git": git_info,
        "source_files": source_files,
    }
    src_prov_path = RESULTS_DIR / "SOURCE_PROVENANCE.json"
    src_prov_path.write_text(json.dumps(source_provenance, indent=2), encoding="utf-8")
    print(f"Wrote {src_prov_path}", flush=True)

    # 2. Frozen Model Provenance
    assert REPO_JINA.exists(), f"Repo Jina missing: {REPO_JINA}"
    assert WEIGHTS_JINA_FT.exists(), f"Weights Jina FT missing: {WEIGHTS_JINA_FT}"

    model, tok, prov = load_jina_crossencoder()
    parser_path = SRC_DIR / "legal_section_parser.py"

    frozen_model_provenance = {
        "schema_version": "dsc2026.gemini.huy_d1_frozen_section_public_v1.frozen_model_provenance.v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_info["head_commit"],
        "status": "PASS",
        "model_provenance": prov,
        "parser_sha256": sha256_file(parser_path),
        "chunk_config": {
            "max_chunk_words": 220,
            "overlap_words": 60,
            "count_sections": 2,
            "max_length": 512,
            "aggregation": "MAX",
            "score_semantics": "raw_frozen_ce_score",
        },
    }

    model_prov_path = RESULTS_DIR / "FROZEN_MODEL_PROVENANCE.json"
    model_prov_path.write_text(json.dumps(frozen_model_provenance, indent=2), encoding="utf-8")
    print(f"Wrote {model_prov_path}", flush=True)

    del model
    del tok
    return {"source": source_provenance, "model": frozen_model_provenance}


if __name__ == "__main__":
    audit_source_and_model_provenance()
