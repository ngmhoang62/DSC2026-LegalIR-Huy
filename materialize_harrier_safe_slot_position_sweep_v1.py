#!/usr/bin/env python
"""
MATERIALIZE CLEAN REL_L0 SAFE-SLOT HAR(R)IER POSITION VARIANTS
==============================================================

Uses existing PUBLIC_HARRIER_GLOBAL_TOP50.json; no GPU.

For each clean REL_L0-safe public query, create separate submissions using the
N-th Harrier document that is outside the original D1 Top5.

Because these queries are a subset of the already validated REL_L0 K4 public
population, each position variant preserves D1 top1-4 and cannot lose recall
relative to the K4 construction if the REL_L0 contract holds.

No public labels are read.
"""

from __future__ import annotations
import argparse,json,re,unicodedata,zipfile
from pathlib import Path

WS=re.compile(r"\s+")

def norm(s):
    return WS.sub(" ",unicodedata.normalize("NFKC",str(s or "")).lower().strip())

def ans(v):
    return [str(x) for x in (v.get("answer",[]) if isinstance(v,dict) else v)]

def find_named(base,name):
    p=base/name
    if p.is_file(): return p
    h=list(base.rglob(name))
    return h[0] if h else None

def questions(path):
    obj=json.loads(path.read_text(encoding="utf-8"))
    return {str(q):str(v.get("question") or v.get("query") or v.get("text") or "")
            if isinstance(v,dict) else str(v) for q,v in obj.items()}

def discover_rel(root,d1):
    for p in sorted((root/"results").rglob("*"),key=lambda x:-x.stat().st_mtime if x.is_file() else 0):
        if not p.is_file() or p.suffix.lower() not in (".json",".zip"): continue
        lo=p.name.lower()
        if not any(x in lo for x in ("rel_l0","rank5","veto")): continue
        try:
            if p.suffix.lower()==".zip":
                with zipfile.ZipFile(p) as z:
                    if "submission.json" not in z.namelist(): continue
                    x=json.loads(z.read("submission.json"))
            else:
                if p.stat().st_size>20_000_000: continue
                x=json.loads(p.read_text(encoding="utf-8"))
            if set(x)!=set(d1): continue
            safe=[]
            ok=True
            for q in d1:
                a,b=ans(x[q]),ans(d1[q])
                if len(a)==4 and a==b[:4]: safe.append(q)
                elif len(a)==5 and a==b: pass
                else: ok=False; break
            if ok and len(safe)==133:
                return p,safe
        except Exception: pass
    raise RuntimeError("REL_L0 artifact not found")

def write(payload,jp,zp):
    jp.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8")
    with zipfile.ZipFile(zp,"w",zipfile.ZIP_DEFLATED) as z:
        z.writestr("submission.json",jp.read_bytes())

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--repo-root",type=Path,required=True)
    ap.add_argument("--model-root",type=Path,required=True)
    ap.add_argument("--positions",default="2,3,4,5,6,8,10")
    args=ap.parse_args()
    root=args.repo_root.resolve(); mr=args.model_root.resolve()

    d1=json.loads((root/"results/gemini/huy_vnlegal_rank_ablation_v1/CANDIDATE_D1_VNLEGAL_SCORE_ONLY.json").read_text(encoding="utf-8"))
    rel_path,safe=discover_rel(root,d1); safe=set(safe)

    rank=json.loads((root/"results/manual/huy_public_rel_l0_harrier_safe_fill_v1/PUBLIC_HARRIER_GLOBAL_TOP50.json").read_text(encoding="utf-8"))
    pub=questions(root/"DSC2026-LegalIR-main/v4_run/public_test_dataset/public-official.json")
    train=questions(find_named(mr,"train_finetuned_best.json"))
    warm=questions(find_named(mr,"warmup_finetuned_best.json"))
    contaminated_texts={norm(x) for x in list(train.values())+list(warm.values()) if norm(x)}
    clean_safe=[q for q in safe if norm(pub[q]) not in contaminated_texts]

    valid={p.stem[len("context_"):] for p in
           (root/"DSC2026-LegalIR-main/v4_run/public_test_dataset/selected-contexts").glob("context_*.json")}

    positions=[int(x) for x in args.positions.split(",") if x.strip()]
    out=root/"results/manual/huy_public_rel_l0_harrier_position_sweep_v1"
    out.mkdir(parents=True,exist_ok=True)
    report={"rel_source":str(rel_path),"clean_safe":len(clean_safe),"variants":{}}

    for pos in positions:
        payload={q:{"answer":ans(v)} for q,v in d1.items()}
        actions=[]
        for q in clean_safe:
            base=ans(d1[q]); forbidden=set(base)
            outsiders=[]
            for item in rank[q].get("results",[]):
                d=str(item["ctx_id"])
                if d not in forbidden and d not in outsiders:
                    outsiders.append(d)
            if len(outsiders)<pos: continue
            c=outsiders[pos-1]
            payload[q]={"answer":base[:4]+[c]}
            actions.append({"qid":q,"challenger":c,"outside_d1_top5_position":pos})
        for q,v in payload.items():
            a=ans(v)
            assert len(a)==5 and len(set(a))==5
            assert all(d in valid for d in a)
        stem=f"CANDIDATE_D1_REL_L0_HARRIER_CLEAN_POS{pos}"
        jp=out/f"{stem}.json"; zp=out/f"{stem}.zip"
        write(payload,jp,zp)
        report["variants"][str(pos)]={"actions":len(actions),"zip":str(zp)}
        print(f"POS{pos:<2d}: actions={len(actions)} -> {zp}")

    (out/"REPORT.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print("Clean safe population:",len(clean_safe))
    print("Report:",out/"REPORT.json")

if __name__=="__main__":
    main()
