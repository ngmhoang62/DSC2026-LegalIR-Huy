#!/usr/bin/env python
"""
HAR(R)IER SPLIT CONTENT-OVERLAP AUDIT V2
=========================================
QID overlap alone is not leakage evidence. This audit compares actual question
text across:
  - Harrier train_finetuned_best
  - Harrier warmup_finetuned_best
  - CAL600 authoritative queries
  - public-official 1000 queries

Reports:
  1) same-QID + exact/normalized same question
  2) cross-QID normalized duplicate questions
"""

from __future__ import annotations
import argparse, json, re, sys, unicodedata
from pathlib import Path

WS = re.compile(r"\s+")

def norm(s):
    s = unicodedata.normalize("NFKC", str(s or "")).lower().strip()
    return WS.sub(" ", s)

def find_named(base, name):
    p = base / name
    if p.is_file(): return p
    hits = list(base.rglob(name))
    return hits[0] if hits else None

def extract_questions(path):
    obj = json.loads(path.read_text(encoding="utf-8"))
    out = {}
    for q, v in obj.items():
        if isinstance(v, dict):
            text = v.get("question") or v.get("query") or v.get("text") or ""
        else:
            text = str(v)
        out[str(q)] = str(text)
    return out

def pair_report(a_name, a, b_name, b):
    same_qid = set(a) & set(b)
    exact_same = [q for q in same_qid if a[q] == b[q]]
    norm_same = [q for q in same_qid if norm(a[q]) == norm(b[q]) and norm(a[q])]
    same_qid_diff = [q for q in same_qid if norm(a[q]) != norm(b[q])]

    # Cross-QID content duplicates.
    inv_a = {}
    for q,t in a.items():
        nt = norm(t)
        if nt: inv_a.setdefault(nt, []).append(q)
    inv_b = {}
    for q,t in b.items():
        nt = norm(t)
        if nt: inv_b.setdefault(nt, []).append(q)
    shared_texts = set(inv_a) & set(inv_b)
    cross_pairs = []
    for t in shared_texts:
        for qa in inv_a[t]:
            for qb in inv_b[t]:
                if qa != qb:
                    cross_pairs.append((qa,qb,t[:120]))

    return {
        "pair": f"{a_name} vs {b_name}",
        "same_qid_count": len(same_qid),
        "same_qid_exact_question": len(exact_same),
        "same_qid_normalized_question": len(norm_same),
        "same_qid_different_question": len(same_qid_diff),
        "cross_qid_same_normalized_question_pairs": len(cross_pairs),
        "sample_same_qid_same": norm_same[:20],
        "sample_same_qid_different": [
            {"qid":q, a_name:a[q][:160], b_name:b[q][:160]}
            for q in same_qid_diff[:10]
        ],
        "sample_cross_qid_same": [
            {"qid_a":qa,"qid_b":qb,"text":t}
            for qa,qb,t in cross_pairs[:20]
        ],
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--repo-root",type=Path,required=True)
    ap.add_argument("--model-root",type=Path,required=True)
    args=ap.parse_args()
    root=args.repo_root.resolve()
    model_root=args.model_root.resolve()
    sys.path.insert(0,str(root))

    from src.gemini.huy_d1_legal_section_evidence_v1.common import load_cal_data

    train_path=find_named(model_root,"train_finetuned_best.json")
    warm_path=find_named(model_root,"warmup_finetuned_best.json")
    if not train_path or not warm_path:
        raise FileNotFoundError("Missing train/warmup result json")

    train=extract_questions(train_path)
    warm=extract_questions(warm_path)

    public_path=root/"DSC2026-LegalIR-main/v4_run/public_test_dataset/public-official.json"
    public=extract_questions(public_path)

    _, queries, _, cal_ids, *_ = load_cal_data()
    cal={str(q):str(queries[q][0]) for q in cal_ids}

    reports=[
        pair_report("train",train,"CAL",cal),
        pair_report("warmup",warm,"CAL",cal),
        pair_report("train",train,"PUBLIC",public),
        pair_report("warmup",warm,"PUBLIC",public),
    ]

    out=root/"results/manual/huy_harrier_split_content_overlap_audit_v2"
    out.mkdir(parents=True,exist_ok=True)
    rp=out/"REPORT.json"
    rp.write_text(json.dumps({"reports":reports},ensure_ascii=False,indent=2),encoding="utf-8")

    print("="*110)
    for r in reports:
        print(
            f"{r['pair']:<24s} sameQID={r['same_qid_count']:4d} | "
            f"sameText(norm)={r['same_qid_normalized_question']:4d} | "
            f"sameQID-diffText={r['same_qid_different_question']:4d} | "
            f"crossQID-sameText={r['cross_qid_same_normalized_question_pairs']:4d}"
        )
    print("Report:",rp)
    print("="*110)

if __name__=="__main__":
    main()
