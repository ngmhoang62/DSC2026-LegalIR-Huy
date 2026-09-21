#!/usr/bin/env python
"""
HUY PIPELINE AUDIT R — PUBLIC/PRIVATE DEPLOYMENT CONTRACT V1
============================================================

CPU-only reproducibility / packaging audit.

Audits the two currently trusted public arms:
  A) exact D1 fixed-K5 champion
  B) REL_L0 variable-K4/5 precision arm

Also checks the historical exact-citation/legal-ref public hedge and reports
whether it is byte/semantic duplicate of an already-submitted arm.

Checks:
- 1000 exact public qids, no answer labels in public-official input;
- public qids do not intersect labelled train qids;
- all predicted doc ids exist in selected-contexts;
- K constraints, duplicates, rank-preservation;
- ZIP contains exactly submission.json and byte-parity holds;
- D1 vs REL_L0 churn is rank5-drop-only;
- direct-reference/legal-ref hedge duplication;
- hashes of the key artifacts/checkpoint/source files needed for later private
  rematerialization.

No public/private gold labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path


def sha256(p: Path):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(8<<20),b""):
            h.update(b)
    return h.hexdigest()


def load_json(p):
    return json.loads(p.read_text(encoding="utf-8"))


def load_zip(p):
    with zipfile.ZipFile(p,"r") as z:
        names=z.namelist()
        if names!=["submission.json"]:
            raise RuntimeError(f"{p}: expected exactly ['submission.json'], got {names}")
        raw=z.read("submission.json")
    return json.loads(raw.decode("utf-8")),raw


def normalize_submission(raw):
    return {str(q):[str(x) for x in row["answer"]] for q,row in raw.items()}


def validate(name,sub,qids,valid_docs,allowed_k):
    issues=[]
    kcounts={}
    for q in qids:
        if q not in sub:
            issues.append({"qid":q,"issue":"MISSING_QUERY"})
            continue
        ans=sub[q]
        kcounts[len(ans)]=kcounts.get(len(ans),0)+1
        if len(ans) not in allowed_k:
            issues.append({"qid":q,"issue":"INVALID_K","answer":ans})
        if len(ans)!=len(set(ans)):
            issues.append({"qid":q,"issue":"DUPLICATE_DOCS","answer":ans})
        bad=[d for d in ans if d not in valid_docs]
        if bad:
            issues.append({"qid":q,"issue":"UNKNOWN_DOCS","docs":bad})
    extra=sorted(set(sub)-set(qids))
    if extra:
        issues.append({"issue":"EXTRA_QIDS","sample":extra[:20],"count":len(extra)})
    return {"name":name,"K_counts":kcounts,"issues":issues,"valid":not issues}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--repo-root",type=Path,required=True)
    args=ap.parse_args()
    root=args.repo_root.resolve()

    print("[1/5] Auditing public/train populations and valid docs...",flush=True)
    data=root/"DSC2026-LegalIR-main/v4_run/public_test_dataset"
    pub_path=data/"public-official.json"
    train_path=data/"train.json"
    pub=load_json(pub_path)
    train=load_json(train_path)
    public_qids=list(map(str,pub.keys()))
    train_qids=set(map(str,train.keys()))
    overlap=sorted(set(public_qids)&train_qids)

    public_answer_fields=[
        q for q,v in pub.items()
        if isinstance(v,dict) and any(k in v for k in ("answer","answers","gold","label"))
    ]

    valid_docs=set()
    for p in (data/"selected-contexts").glob("context_*.json"):
        r=load_json(p); valid_docs.add(str(r["id"]))
    if len(public_qids)!=1000:
        raise RuntimeError(f"Expected 1000 public qids, got {len(public_qids)}")
    if len(valid_docs)!=8532:
        raise RuntimeError(f"Expected 8532 valid docs, got {len(valid_docs)}")

    print("[2/5] Validating D1 + REL_L0 artifacts...",flush=True)
    d1_path=root/"results/gemini/huy_vnlegal_rank_ablation_v1/CANDIDATE_D1_VNLEGAL_SCORE_ONLY.json"
    rel_zip=root/"results/manual/huy_public_d1_ce_rank5_veto_v1/REL_L0/submission_REL_L0.zip"
    if not d1_path.is_file(): raise FileNotFoundError(d1_path)
    if not rel_zip.is_file(): raise FileNotFoundError(rel_zip)

    d1=normalize_submission(load_json(d1_path))
    rel_raw,rel_bytes=load_zip(rel_zip)
    rel=normalize_submission(rel_raw)

    vd1=validate("D1",d1,public_qids,valid_docs,{5})
    vrel=validate("REL_L0",rel,public_qids,valid_docs,{4,5})

    churn=[]
    contract_viol=[]
    for q in public_qids:
        if d1[q]==rel[q]: continue
        churn.append(q)
        if not (len(d1[q])==5 and len(rel[q])==4 and rel[q]==d1[q][:4]):
            contract_viol.append({"qid":q,"d1":d1[q],"rel":rel[q]})

    print("[3/5] Checking ZIP byte parity + legal-ref/exact-citation duplicate arms...",flush=True)
    # Re-serialize nothing: byte-parity means zip payload itself can be hashed and
    # parsed; if a co-located submission.json exists, compare exactly.
    rel_json_path=rel_zip.parent/"submission.json"
    rel_byte_parity=None
    if rel_json_path.is_file():
        rel_byte_parity=(rel_json_path.read_bytes()==rel_bytes)

    optional_candidates=[
        root/"results/gemini/huy_d1_exact_citation_public_v1/CANDIDATE_D1_EXACT_CITATION.json",
        root/"results/manual/huy_public_legal_ref_challenger_certificate_rescue_v1/submission.json",
    ]
    hedge_compare=[]
    for p in optional_candidates:
        if not p.is_file(): continue
        x=normalize_submission(load_json(p))
        same_d1=(x==d1)
        same_rel=(x==rel)
        diff=sum(x.get(q)!=d1.get(q) for q in public_qids)
        hedge_compare.append({
            "path":str(p),"sha256":sha256(p),
            "identical_to_D1":same_d1,
            "identical_to_REL_L0":same_rel,
            "changed_queries_vs_D1":diff,
        })
    if len(hedge_compare)>=2:
        a=normalize_submission(load_json(Path(hedge_compare[0]["path"])))
        b=normalize_submission(load_json(Path(hedge_compare[1]["path"])))
        hedge_pair_identical=(a==b)
    else:
        hedge_pair_identical=None

    print("[4/5] Hashing private-rematerialization dependencies...",flush=True)
    deps=[
        d1_path,
        rel_zip,
        root/"results/manual/huy_noncal_trainable_ce_boundary_v1/oof/fold_0/training/model.pt",
        root/"src/gemini/huy_d1_query_anchored_legal_ref_expansion_v1/legal_ref_indexer.py",
        root/"src/gemini/huy_d1_query_anchored_legal_ref_expansion_v1/relation_graph.py",
        root/"src/gemini/huy_d1_query_anchored_legal_ref_expansion_v1/query_anchored_generator.py",
        root/"cache/research_v2_e5_confirmation/bundle-v1/embeddings.f16.npy",
    ]
    dependency_receipts={}
    for p in deps:
        dependency_receipts[str(p)]={
            "exists":p.is_file(),
            "size":p.stat().st_size if p.is_file() else None,
            "sha256":sha256(p) if p.is_file() else None,
        }

    critical_missing=[
        p for p in (
            d1_path,
            rel_zip,
            root/"results/manual/huy_noncal_trainable_ce_boundary_v1/oof/fold_0/training/model.pt",
        ) if not p.is_file()
    ]

    pass_contract=(
        not overlap
        and not public_answer_fields
        and vd1["valid"] and vrel["valid"]
        and not contract_viol
        and not critical_missing
    )

    report={
        "schema":"manual.public_private_deployment_contract_v1",
        "population":{
            "public_queries":len(public_qids),
            "train_queries":len(train_qids),
            "public_train_qid_overlap":overlap,
            "public_records_with_label_like_fields":public_answer_fields,
            "valid_documents":len(valid_docs),
        },
        "D1_validation":vd1,
        "REL_L0_validation":vrel,
        "REL_L0_vs_D1":{
            "changed_queries":len(churn),
            "changed_qids":churn,
            "contract_violations":contract_viol,
            "required_semantics":"rank5 drop only; top1-4 preserved",
            "zip_payload_matches_neighbor_submission_json":rel_byte_parity,
        },
        "hedge_artifacts":hedge_compare,
        "exact_citation_vs_legal_ref_semantically_identical":hedge_pair_identical,
        "dependency_receipts":dependency_receipts,
        "critical_missing":[str(p) for p in critical_missing],
        "private_plan_contract":{
            "arm_A":"exact D1 fixed K5 anchor",
            "arm_B":"REL_L0 drop-rank5-only precision arm",
            "arm_C":"DIRECT_REFERENCE_MATCH challenger-certificate recall hedge; relation-neighbor disabled",
            "no_private_label_tuning":True,
            "max_K":5,
        },
        "verdict":"PASS_PUBLIC_PRIVATE_DEPLOYMENT_CONTRACT" if pass_contract else "FAIL_OR_INCOMPLETE_DEPLOYMENT_CONTRACT",
        "gold_labels_accessed":False,
    }

    out=root/"results/manual/huy_public_private_deployment_contract_v1"
    out.mkdir(parents=True,exist_ok=True)
    path=out/"REPORT.json"
    path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")

    print("[5/5] RESULT"); print("="*116)
    print(
        f"Public/train qid overlap={len(overlap)}; "
        f"public label-like records={len(public_answer_fields)}; valid_docs={len(valid_docs)}"
    )
    print(
        f"D1 valid={vd1['valid']} K={vd1['K_counts']} | "
        f"REL_L0 valid={vrel['valid']} K={vrel['K_counts']}"
    )
    print(
        f"REL_L0 churn={len(churn)} rank5-drop contract violations={len(contract_viol)} "
        f"zip-neighbor-byte-parity={rel_byte_parity}"
    )
    print("Hedge artifacts:",hedge_compare)
    print("Exact-citation vs legal-ref identical:",hedge_pair_identical)
    print("Critical missing deps:",[str(p) for p in critical_missing] or "NONE")
    print("VERDICT:",report["verdict"])
    print("Report:",path); print("="*116)


if __name__=="__main__":
    main()
