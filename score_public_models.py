"""Score the PUBLIC candidate pool with a supplied bi-encoder or Jina-family
cross-encoder, so a configuration validated on CV can be turned into a
submission.

The pool comes from the cached full-pool cross-encoder scores, whose key set is
exactly the public candidate set the runner builds, so candidate generation is
not repeated. Jina-family weights are overlaid onto the repo's working Jina
code, since the supplied copies ship modeling code that no longer imports on
transformers v5.
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import torch
from transformers import (AutoModel, AutoModelForSequenceClassification,
                          AutoTokenizer)

from benchmark_aiteamvn_holdouts import encode_cls
from benchmark_jina_reranker_holdouts import top_passages
from run_burst_expanded_fusion_submission import DocumentStore

REPO_JINA = "models/jina-reranker-v2-base-multilingual"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["bi", "jina"], required=True)
    ap.add_argument("--model-path", default="")
    ap.add_argument("--weights", default="",
                    help="jina only: safetensors or .pt state dict to overlay")
    ap.add_argument("--out", required=True)
    ap.add_argument("--passages", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    root = Path(__file__).resolve().parent
    data = root / "DSC2026-LegalIR-main/v4_run/public_test_dataset"
    docs = DocumentStore(sorted((data / "selected-contexts").glob("context_*.json")))
    public = json.loads((data / "public-official.json").read_text(encoding="utf-8"))
    public = {q: (v["question"] if isinstance(v, dict) else v) for q, v in public.items()}
    pool = pickle.loads(
        (root / "results/crossenc_fullpool/public_scores.pkl").read_bytes())["scores"]
    ids = [q for q in public if q in pool]

    out_path = root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    saved = pickle.loads(out_path.read_bytes()) if out_path.exists() else {}
    todo = [q for q in ids if any(d not in saved.get(q, {}) for d in pool[q])]
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(saved)} cached, scoring {len(todo)}/{len(ids)} public queries",
          flush=True)
    if not todo:
        return

    if args.kind == "bi":
        path = root / args.model_path
        tok = AutoTokenizer.from_pretrained(path)
        model = AutoModel.from_pretrained(
            path, dtype=torch.float16, low_cpu_mem_usage=True).eval().to("cuda")
    else:
        path = root / (args.model_path or REPO_JINA)
        tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True,
                                            fix_mistral_regex=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            path, trust_remote_code=True, dtype=torch.float16)
        if args.weights.endswith(".safetensors"):
            from safetensors.torch import load_file
            state = {k: v.to(torch.float16)
                     for k, v in load_file(root / args.weights).items()}
        else:
            state = torch.load(root / args.weights, map_location="cpu",
                               weights_only=True)["state_dict"]
        missing, unexpected = model.load_state_dict(state, strict=False)
        assert not unexpected, f"unexpected keys: {unexpected[:5]}"
        print(f"overlaid {len(state)} tensors from {args.weights} "
              f"({len(missing)} repo tensors untouched)", flush=True)
        model._tokenizer = tok
        model = model.eval().to("cuda")
    print(f"ready on {torch.cuda.get_device_name(0)}", flush=True)

    started = time.time()
    for i, q in enumerate(todo, 1):
        text = public[q]
        owners, passages = [], []
        for d in pool[q]:
            for p in top_passages(text, docs[d], count=args.passages):
                owners.append(d)
                passages.append(p)
        ds = dict(saved.get(q, {}))
        if passages:
            if args.kind == "bi":
                qv = encode_cls(model, tok, [text], 1, 512)[0]
                pv = encode_cls(model, tok, passages, 32, 512)
                vals = pv @ qv
            else:
                vals = model.compute_score([(text, p) for p in passages],
                                           batch_size=args.batch_size, max_length=512)
                if isinstance(vals, float):
                    vals = [vals]
            for d, s in zip(owners, vals):
                ds[d] = max(ds.get(d, -1e9), float(s))
        saved[q] = ds
        if i % 50 == 0:
            out_path.write_bytes(pickle.dumps(saved, protocol=5))
            rate = (time.time() - started) / i
            print(f"  {i}/{len(todo)} {rate:.2f}s/q "
                  f"eta {rate * (len(todo) - i) / 60:.1f}m", flush=True)
    out_path.write_bytes(pickle.dumps(saved, protocol=5))
    print(f"Saved {out_path} ({len(saved)} queries)", flush=True)


if __name__ == "__main__":
    main()
