"""Score the CV pool with the user-supplied fine-tuned Jina cross-encoder.

The supplied directory ships its own copy of Jina's custom modeling code, which
imports `create_position_ids_from_input_ids` -- removed in transformers v5, so
it cannot be loaded directly. The architecture is byte-identical to the repo's
working jina-reranker-v2-base-multilingual (XLMRobertaForSequenceClassification,
768x12, vocab 250002, 153 tensors including the classifier head), so the fix is
to instantiate the repo's model and overlay the supplied weights -- the same
pattern the project already uses for its own fine-tuned Jina checkpoint.
"""

from __future__ import annotations

import argparse
import pickle
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from benchmark_jina_reranker_holdouts import top_passages
from run_burst_expanded_fusion_submission import DocumentStore, prefetch
from tune_corpus_cap32_fusion import build_training_cap

REPO_JINA = "models/jina-reranker-v2-base-multilingual"
WEIGHTS = "models/from_drive/jina_finetuned/model.safetensors"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--passages", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--weights", default=WEIGHTS)
    ap.add_argument("--out", default="results/from_drive/jina_ft_cv.pkl")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    root = Path(__file__).resolve().parent
    docs = DocumentStore(sorted(
        (root / "DSC2026-LegalIR-main/v4_run/public_test_dataset/selected-contexts")
        .glob("context_*.json")))
    queries, blocks, all_ids, extended, local, _ = build_training_cap(
        root, 32, "results/corpus_index/holdout_extended_scores_cap32.pkl", depth=20)

    out_path = root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    saved = pickle.loads(out_path.read_bytes()) if out_path.exists() else {}
    todo = [q for q in all_ids if any(d not in saved.get(q, {}) for d in extended[q])]
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(saved)} cached, scoring {len(todo)} queries", flush=True)
    if not todo:
        return

    tok = AutoTokenizer.from_pretrained(root / REPO_JINA, trust_remote_code=True,
                                        fix_mistral_regex=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        root / REPO_JINA, trust_remote_code=True, dtype=torch.float16)
    state = load_file(root / args.weights)
    missing, unexpected = model.load_state_dict(
        {k: v.to(torch.float16) for k, v in state.items()}, strict=False)
    assert not unexpected, f"unexpected keys: {unexpected[:5]}"
    print(f"overlaid {len(state)} tensors from {args.weights} "
          f"({len(missing)} repo tensors left untouched)", flush=True)
    model._tokenizer = tok
    model.eval().to("cuda")
    print(f"ready on {torch.cuda.get_device_name(0)}", flush=True)

    def prepare(q):
        known = saved.get(q, {})
        text = queries[q][0]
        owners, passages = [], []
        for d in extended[q]:
            if d in known:
                continue
            for p in top_passages(text, docs[d], count=args.passages):
                owners.append(d)
                passages.append(p)
        return q, owners, passages

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=2) as tp:
        for i, (q, owners, passages) in enumerate(prefetch(tp, todo, prepare), 1):
            ds = dict(saved.get(q, {}))
            if passages:
                text = queries[q][0]
                raw = model.compute_score([(text, p) for p in passages],
                                          batch_size=args.batch_size, max_length=512)
                if isinstance(raw, float):
                    raw = [raw]
                for d, s in zip(owners, raw):
                    ds[d] = max(ds.get(d, -1e9), float(s))
            saved[q] = ds
            if i % 25 == 0:
                out_path.write_bytes(pickle.dumps(saved, protocol=5))
                rate = (time.perf_counter() - started) / i
                print(f"  {i}/{len(todo)} {rate:.2f}s/q "
                      f"eta {rate * (len(todo) - i) / 60:.1f}m", flush=True)
    out_path.write_bytes(pickle.dumps(saved, protocol=5))
    print(f"Saved {out_path} ({len(saved)} queries)", flush=True)


if __name__ == "__main__":
    main()
