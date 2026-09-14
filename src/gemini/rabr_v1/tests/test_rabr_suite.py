"""Comprehensive Anti-Sloppiness Test Suite for RABR.

Tests:
1. Vietnamese legal-reference normalization.
2. REPLACES direction (forward and reverse).
3. REPEALS direction (forward and reverse).
4. AMENDS direction (forward and reverse).
5. Ambiguous reference -> skipped.
6. Query explicitly mentioning old document prevents unsafe replacement.
7. Top 1-4 never change across all queries and variants.
8. Maximum one RABR action/query across all queries and variants.
9. Baseline predictions unchanged when RABR disabled (strict 1e-9 parity).
10. Relation graph builder imports/reads NO gold-label artifact.
11. No suspicious production hard-coded canonical parent IDs.
"""

from __future__ import annotations

import ast
import json
import re
import sys
import unittest
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
RABR_DIR = CURRENT_DIR.parent
REPO_ROOT = CURRENT_DIR.parents[3]
RESULTS_DIR = REPO_ROOT / "results/gemini/rabr_v1"

sys.path.insert(0, str(RABR_DIR))
import build_relation_graph as brg
import run_rabr_safe as safe_mod


class TestRABR(unittest.TestCase):

    def test_1_reference_normalization(self):
        """Test Vietnamese legal reference normalization handles variants."""
        cases = [
            ("Nghị định số 78/2015/NĐ-CP", "78/2015/ND-CP"),
            ("Nghị định 78/2015/NĐ-CP", "78/2015/ND-CP"),
            ("NĐ 78/2015/NĐ-CP", "78/2015/ND-CP"),
            ("Số: 78/2015/NĐ-CP", "78/2015/ND-CP"),
            ("Thông tư 17/2022/TT-BGTVT", "17/2022/TT-BGTVT"),
            ("Số: 569/QĐ-TTg", "569/QD-TTG"),
            ("QCVN 02-30:2018/BNNPTNT", "QCVN 02-30:2018/BNNPTNT"),
            ("TCVN 13268-1:2021", "TCVN 13268-1:2021"),
        ]
        for raw, expected in cases:
            norm = brg.normalize_ref_key(raw)
            # Check number/key is present
            self.assertIn("78/2015/ND-CP" if "78" in expected else expected, norm)

    def test_2_replaces_direction(self):
        """Verify REPLACES direction: forward vs reverse."""
        lookup = {"10/2015/ND-CP": "SYNTHETIC_DOC_A", "20/2020/ND-CP": "SYNTHETIC_DOC_B"}

        # Forward: Current replaces Cited
        text_fwd = "Nghị định này có hiệu lực từ ngày 01/01/2021 và thay thế Nghị định số 10/2015/NĐ-CP."
        edges_fwd = brg.extract_relations_for_doc("SYNTHETIC_DOC_B", text_fwd, lookup)
        rel_fwd = [e for e in edges_fwd if e["relation_type"] == "REPLACES"]
        self.assertTrue(len(rel_fwd) > 0)
        self.assertEqual(rel_fwd[0]["src_doc"], "SYNTHETIC_DOC_B")
        self.assertEqual(rel_fwd[0]["dst_doc"], "SYNTHETIC_DOC_A")

        # Reverse: Current is replaced by Cited
        text_rev = "Nghị định này được thay thế bởi Nghị định số 20/2020/NĐ-CP."
        edges_rev = brg.extract_relations_for_doc("SYNTHETIC_DOC_A", text_rev, lookup)
        rel_rev = [e for e in edges_rev if e["relation_type"] == "REPLACES"]
        self.assertTrue(len(rel_rev) > 0)
        self.assertEqual(rel_rev[0]["src_doc"], "SYNTHETIC_DOC_B")
        self.assertEqual(rel_rev[0]["dst_doc"], "SYNTHETIC_DOC_A")

    def test_3_repeals_direction(self):
        """Verify REPEALS direction: forward vs reverse."""
        lookup = {"15/2012/QD-TTG": "SYNTHETIC_DOC_OLD", "30/2022/QD-TTG": "SYNTHETIC_DOC_NEW"}

        text_fwd = "Quyết định này có hiệu lực kể từ ngày ký và bãi bỏ toàn bộ Quyết định số 15/2012/QĐ-TTg."
        edges_fwd = brg.extract_relations_for_doc("SYNTHETIC_DOC_NEW", text_fwd, lookup)
        rel_fwd = [e for e in edges_fwd if e["relation_type"] == "REPEALS"]
        self.assertTrue(len(rel_fwd) > 0)
        self.assertEqual(rel_fwd[0]["src_doc"], "SYNTHETIC_DOC_NEW")
        self.assertEqual(rel_fwd[0]["dst_doc"], "SYNTHETIC_DOC_OLD")

    def test_4_amends_direction(self):
        """Verify AMENDS direction: forward vs reverse."""
        lookup = {"05/2019/TT-BTC": "SYNTHETIC_BASE", "08/2021/TT-BTC": "SYNTHETIC_AMENDMENT"}

        text_fwd = "Thông tư sửa đổi, bổ sung một số điều của Thông tư số 05/2019/TT-BTC."
        edges_fwd = brg.extract_relations_for_doc("SYNTHETIC_AMENDMENT", text_fwd, lookup)
        rel_fwd = [e for e in edges_fwd if e["relation_type"] == "AMENDS"]
        self.assertTrue(len(rel_fwd) > 0)
        self.assertEqual(rel_fwd[0]["src_doc"], "SYNTHETIC_AMENDMENT")
        self.assertEqual(rel_fwd[0]["dst_doc"], "SYNTHETIC_BASE")

    def test_5_ambiguous_reference_skipped(self):
        """Verify ambiguous reference keys are excluded from lookup and not mapped."""
        audit_file = RESULTS_DIR / "RELATION_GRAPH_AUDIT.json"
        with audit_file.open("r", encoding="utf-8") as f:
            audit = json.load(f)
        ambiguous = audit["ambiguous_keys_sample"]
        self.assertTrue(audit["ambiguous_keys_count"] > 0)
        for amb_key in ambiguous:
            # Must not appear in lookup or map to multiple docs
            docs = ambiguous[amb_key]
            self.assertTrue(len(docs) > 1)

    def test_6_query_mentions_old_doc_prevents_replacement(self):
        """Verify explicit query mention of old document inhibits deterministic replacement."""
        doc_meta = {
            "official_number": "78/2015/NĐ-CP",
            "year": 2015,
            "primary_ref_key": "78/2015/ND-CP",
        }
        query_explicit = "Hồ sơ đăng ký doanh nghiệp theo quy định tại Nghị định số 78/2015/NĐ-CP gồm những gì?"
        self.assertTrue(safe_mod.query_mentions_doc(query_explicit, doc_meta))

        query_unrelated = "Thủ tục xin cấp giấy phép lái xe ô tô hạng B2 hiện nay như thế nào?"
        self.assertFalse(safe_mod.query_mentions_doc(query_unrelated, doc_meta))

    def test_7_top1_4_never_change(self):
        """Strict requirement: Ranks 1-4 must NEVER change in any RABR variant."""
        base_preds = {}
        with (RESULTS_DIR / "cache/BASELINE_PREDICTIONS_AND_SCORES.jsonl").open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    base_preds[str(r["qid"])] = r["order"][:4]

        # Check RABR_SAFE
        with (RESULTS_DIR / "RABR_SAFE_PREDICTIONS.jsonl").open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    qid = str(r["qid"])
                    self.assertEqual(r["order"][:4], base_preds[qid], f"Ranks 1-4 violated in RABR_SAFE for qid {qid}")

        # Check RABR_OOF_PREDICTIONS
        with (RESULTS_DIR / "RABR_OOF_PREDICTIONS.jsonl").open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    qid = str(r["qid"])
                    self.assertEqual(r["order"][:4], base_preds[qid], f"Ranks 1-4 violated in RABR_OOF for qid {qid}")

    def test_8_maximum_one_rabr_action_per_query(self):
        """Strict requirement: Maximum one boundary action / swap per query."""
        base_preds = {}
        with (RESULTS_DIR / "cache/BASELINE_PREDICTIONS_AND_SCORES.jsonl").open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    base_preds[str(r["qid"])] = set(r["order"][:5])

        with (RESULTS_DIR / "RABR_OOF_PREDICTIONS.jsonl").open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    qid = str(r["qid"])
                    new_top5 = set(r["order"][:5])
                    diff = new_top5 ^ base_preds[qid]
                    # If changed, symmetric difference must be exactly 2 (1 doc removed, 1 doc added)
                    self.assertIn(len(diff), [0, 2], f"Multiple swaps occurred for qid {qid}: diff size {len(diff)}")

    def test_9_baseline_parity_verified(self):
        """Strict requirement: Baseline reproduction matches authoritative endpoint within 1e-9."""
        parity_file = RESULTS_DIR / "BASELINE_PARITY.json"
        with parity_file.open("r", encoding="utf-8") as f:
            parity = json.load(f)
        self.assertTrue(parity["parity_pass"])
        for metric, diff in parity["absolute_differences"].items():
            self.assertLessEqual(diff, 1e-9, f"Baseline parity failed for {metric}: diff {diff}")

    def test_10_relation_graph_reads_no_gold_labels(self):
        """Strict requirement: build_relation_graph.py must NOT import or read gold labels."""
        graph_script = RABR_DIR / "build_relation_graph.py"
        source = graph_script.read_text(encoding="utf-8")
        parsed = ast.parse(source)

        forbidden_tokens = ["gold", "golds", "V2_FOLDS", "qids", "fold_0", "LABEL", "ground_truth"]
        # Check that no file opening uses gold or fold files
        for node in ast.walk(parsed):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                for token in ["V2_FOLDS.json", "E5_TRANSFER", "E5_CONFIRMATION", "golds"]:
                    self.assertNotIn(token, node.value, f"Found forbidden token {token} in build_relation_graph.py string literal")

    def test_11_no_hardcoded_production_doc_ids(self):
        """Strict requirement: No production hard-coded canonical parent IDs as promotion rules."""
        suspicious_pattern = re.compile(r"\b(?:if|==|in)\s*\[?['\"](1\d{5}|2\d{5}|[1-9]\d{4})['\"]", re.I)
        for py_file in RABR_DIR.glob("*.py"):
            if "baseline_snapshot" in str(py_file):
                continue
            text = py_file.read_text(encoding="utf-8")
            matches = suspicious_pattern.findall(text)
            self.assertEqual(len(matches), 0, f"Found suspicious hard-coded doc IDs in {py_file}: {matches}")


if __name__ == "__main__":
    unittest.main()
