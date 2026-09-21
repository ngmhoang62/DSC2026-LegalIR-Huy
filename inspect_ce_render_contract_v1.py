#!/usr/bin/env python
from __future__ import annotations
import argparse, importlib.util, json, sys
from pathlib import Path
import numpy as np

def loadmod(path: Path):
    spec=importlib.util.spec_from_file_location("cebase",path)
    m=importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(m)
    return m

def summarize(name, value):
    out={"type":type(value).__name__}
    try:
        out["len"]=len(value)
    except Exception:
        pass
    if isinstance(value,(str,int,float,bool,type(None))):
        out["value"]=value
    elif isinstance(value,Path):
        out["value"]=str(value)
    elif isinstance(value,np.ndarray):
        out.update({"shape":list(value.shape),"dtype":str(value.dtype)})
    elif isinstance(value,dict):
        out["keys_sample"]=list(map(str,list(value.keys())[:10]))
        if value:
            v=next(iter(value.values()))
            if isinstance(v,np.ndarray):
                out["value_sample_shape"]=list(v.shape)
                out["value_sample_dtype"]=str(v.dtype)
            else:
                out["value_sample_type"]=type(v).__name__
    elif isinstance(value,(list,tuple)):
        out["sample_types"]=[type(x).__name__ for x in value[:5]]
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--repo-root",type=Path,required=True)
    args=ap.parse_args()
    root=args.repo_root.resolve()
    sibling=root.parent/"LegalIR"
    base=root.parent/"run_noncal_trainable_ce_boundary_v3_fixed.py"
    if not base.is_file():
        raise FileNotFoundError(base)
    m=loadmod(base)
    sys.path[:0]=[str(root),str(root/"src"),str(sibling),str(sibling/"src")]
    cal_ids,_=m.get_cal_ids_label_free(root)
    world=m.load_noncal_world(root,sibling,set(cal_ids))
    r=world["render"]

    print("RENDER_CLASS",type(r).__module__,type(r).__name__)
    print("RENDER_DICT_KEYS",sorted(getattr(r,"__dict__",{}).keys()))
    report={}
    for k,v in getattr(r,"__dict__",{}).items():
        report[k]=summarize(k,v)
    print(json.dumps(report,ensure_ascii=False,indent=2,default=str))

    print("\nMETHODS")
    for name in ("matrix","query_vector","questions","chunk_ids","fingerprint"):
        try:
            attr=getattr(r,name)
            if callable(attr):
                print(name,"CALLABLE")
            else:
                print(name,summarize(name,attr))
        except Exception as e:
            print(name,"ERROR",repr(e))

    # Probe one known nonCAL qid to expose vector shapes only.
    q=world["noncal"][0]
    print("\nPROBE_QID",q)
    try:
        v=r.query_vector(q,"e5")
        print("query_vector_e5",summarize("v",v))
    except Exception as e:
        print("query_vector_e5 ERROR",repr(e))
    try:
        x=r.matrix("e5")
        print("matrix_e5",summarize("x",x))
    except Exception as e:
        print("matrix_e5 ERROR",repr(e))

if __name__=="__main__":
    main()
