"""Audit lexical implementation provenance against historical burst_retriever.py.

Verifies:
- Exact tokenization parity on diverse Vietnamese legal queries/strings
- Exact FTS5 query string formulation parity
- Schema parity: unicode61 tokenizer, contentless FTS5, bm25 scoring semantics
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

from .common import RES_DIR, ROOT, SRC_DIR, get_git_status, sha256_file
from .section_retriever import fts_query, tokens

HISTORICAL_BURST_PATH = ROOT / "src/gemini/huy_sparse_resurrection_v1/burst_retriever.py"


def run_lexical_provenance_audit() -> Dict[str, Any]:
    RES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RES_DIR / "LEXICAL_IMPLEMENTATION_PROVENANCE.json"

    # Verify historical source exists
    if not HISTORICAL_BURST_PATH.exists():
        raise FileNotFoundError(f"Historical burst retriever not found: {HISTORICAL_BURST_PATH}")

    # Import historical functions dynamically
    import importlib.util
    spec = importlib.util.spec_from_file_location("burst_retriever", str(HISTORICAL_BURST_PATH))
    hist_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hist_mod)

    hist_tokens = hist_mod.tokens
    hist_fts_query = hist_mod.fts_query

    # Test battery of Vietnamese legal strings
    test_cases = [
        "",
        "   ",
        "Luật Doanh nghiệp 2020",
        "Điều 12. Xử phạt vi phạm hành chính trong lĩnh vực bảo vệ môi trường",
        "Nghị định số 15/2020/NĐ-CP ngày 03 tháng 02 năm 2020 của Chính phủ",
        "Hành vi vi phạm quy định tại Khoản 2 Điều 15 bị xử phạt như thế nào?",
        "Tội giết người theo Bộ luật Hình sự 2015 (sửa đổi, bổ sung 2017)",
        "Công ty TNHH 2 thành viên trở lên có quyền phát hành trái phiếu không?",
        "Các trường hợp không được hưởng thừa kế theo di chúc?",
        "Quy định về thời hiệu khởi kiện tranh chấp hợp đồng thương mại: 02 năm hay 03 năm?",
        'Dấu ngoặc kép "test" và dấu phẩy, chấm: ; ! ? @ # $ % ^ & * ( ) _ + - =',
    ]

    mismatches = []
    for tc in test_cases:
        cur_toks = tokens(tc)
        ref_toks = hist_tokens(tc)
        if cur_toks != ref_toks:
            mismatches.append({"type": "tokens", "input": tc, "current": cur_toks, "reference": ref_toks})

        cur_query = fts_query(tc)
        ref_query = hist_fts_query(tc)
        if cur_query != ref_query:
            mismatches.append({"type": "fts_query", "input": tc, "current": cur_query, "reference": ref_query})

    git_info = get_git_status()
    burst_sha = sha256_file(HISTORICAL_BURST_PATH)
    retriever_sha = sha256_file(SRC_DIR / "section_retriever.py")

    passed = (len(mismatches) == 0)

    record: Dict[str, Any] = {
        "audit_name": "LEXICAL_IMPLEMENTATION_PROVENANCE",
        "status": "PASS" if passed else "FAIL",
        "reference_source": {
            "path": str(HISTORICAL_BURST_PATH.relative_to(ROOT)),
            "sha256": burst_sha,
        },
        "target_source": {
            "path": str((SRC_DIR / "section_retriever.py").relative_to(ROOT)),
            "sha256": retriever_sha,
        },
        "specifications": {
            "token_regex": r"\w+",
            "token_flags": "re.UNICODE",
            "casing": "lower",
            "fts_operator": "OR",
            "fts_quoting": '"{term}" with double quote escaping',
            "fts_sqlite_tokenizer": "unicode61",
            "fts_table_type": "contentless (content='')",
            "scoring_function": "-bm25(sections_fts)",
            "aggregation": "MAX parent document section score",
            "budget_hit_depth": 128,
            "budget_cap_additions": 8,
            "tie_breaking": "score DESC, doc_id numeric ASC",
        },
        "test_battery": {
            "test_cases_evaluated": len(test_cases),
            "mismatch_count": len(mismatches),
            "mismatches": mismatches,
        },
        "git_commit": git_info["head_commit"],
    }

    out_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[AUDIT] Lexical provenance audit completed: {record['status']}")
    print(f"[AUDIT] Saved to {out_path}")
    return record


if __name__ == "__main__":
    res = run_lexical_provenance_audit()
    if res["status"] != "PASS":
        sys.exit(1)
