"""Score full CAL600 candidate pool using the adapted Jina continuation model."""

from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path
import torch
from peft import PeftModel

ROOT = Path("D:/Study/DSC2026/sota")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from run_burst_expanded_fusion_submission import DocumentStore
from tune_corpus_cap32_fusion import build_training_cap
import src.gemini.huy_d1_jina_ft_continuation_v1.common as common

RES_DIR = ROOT / "results/gemini/huy_d1_jina_ft_continuation_v1"
ADAPTER_DIR = RES_DIR / "jina_ft_continued_adapter"
OUT_CV_PKL = RES_DIR / "jina_ft_continued_cv.pkl"


def run_scoring(batch_size: int = 16) -> dict:
    common.seed_everything(2026)
    print("Loading documents and CAL600 pool...", flush=True)
    docs = DocumentStore(
        sorted(
            (
                ROOT
                / "DSC2026-LegalIR-main/v4_run/public_test_dataset/selected-contexts"
            ).glob("context_*.json")
        )
    )
    queries, blocks, all_ids, extended, local, _ = build_training_cap(
        ROOT, 32, "results/corpus_index/holdout_extended_scores_cap32.pkl", depth=20
    )

    batch_size = 16
    print(f"Loading adapted Jina model from {ADAPTER_DIR} (infer_bs={batch_size})...", flush=True)
    base_model, tok = common.load_jina_base_with_shipped_weights()
    model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
    common.patch_tuple_returning_lora(model)
    model.eval().to("cuda")

    saved_scores = {}
    t0 = time.perf_counter()
    total_queries = len(all_ids)

    for i, q in enumerate(all_ids, 1):
        q_str = str(q)
        text = queries[q_str][0]
        owners, passages = [], []
        for d in extended[q_str]:
            p_list = common.top_passages(text, docs[d], count=2, window=220, overlap=70)
            for p in p_list:
                owners.append(d)
                passages.append(p)

        # Batch scoring
        sentence_pairs = [(text, p) for p in passages]
        raw_scores = []
        with torch.no_grad():
            for b_start in range(0, len(sentence_pairs), batch_size):
                b_chunk = sentence_pairs[b_start : b_start + batch_size]
                inputs = tok(
                    [c[0] for c in b_chunk],
                    [c[1] for c in b_chunk],
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    max_length=512,
                ).to("cuda")
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    logits = model(**inputs).logits.view(-1).float()
                    sig = torch.sigmoid(logits).cpu().numpy().tolist()
                    if isinstance(sig, float):
                        sig = [sig]
                    raw_scores.extend(sig)

        ds = {}
        for d, s in zip(owners, raw_scores):
            ds[d] = max(ds.get(d, -1e9), float(s))

        saved_scores[q_str] = ds

        if i % 50 == 0 or i == total_queries:
            elapsed = time.perf_counter() - t0
            rate = elapsed / i
            eta_m = (rate * (total_queries - i)) / 60.0
            print(
                f"  Scored {i}/{total_queries} queries ({rate:.2f} s/q, eta {eta_m:.1f} m)",
                flush=True,
            )

    OUT_CV_PKL.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CV_PKL, "wb") as f:
        pickle.dump(saved_scores, f, protocol=5)

    print(f"Saved {OUT_CV_PKL} ({len(saved_scores)} queries)", flush=True)
    return {"status": "SUCCESS", "query_count": len(saved_scores), "path": str(OUT_CV_PKL)}


if __name__ == "__main__":
    run_scoring()
