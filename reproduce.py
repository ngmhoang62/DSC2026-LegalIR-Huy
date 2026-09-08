"""Rebuild the burst_userft_maxrecall submission from this directory alone.

Configuration: the production 6-channel LTR fusion + the full-pool
cross-encoder + three extra score channels (aiteamvn_ft, jina_ft, title_embed),
LTR trained on the 600 CV queries, threshold removed (alpha=0) so every query
gets 5 documents. CV Recall@5 = 0.9561.

Everything the run needs is bundled: the corpus, the query files, and the
cached per-channel scores. No model weights, no GPU and no network are
required, because every stage's scores are already computed -- the run is pure
fusion plus selection.

    python reproduce.py            rebuild and verify the submission
    python reproduce.py --check    verify the bundle is complete, then exit
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXPECTED_MD5 = "2fb9a8a3b7"  # first 10 hex chars of submission.json

REQUIRED = [
    "DSC2026-LegalIR-main/v4_run/public_test_dataset/public-official.json",
    "DSC2026-LegalIR-main/v4_run/public_test_dataset/train.json",
    "DSC2026-LegalIR-main/v4_run/public_test_dataset/selected-contexts",
    "results/aiteamvn_dense/holdout_scores_512.pkl",
    "results/burst_expanded_fusion/corpus_rank_cap32.pkl",
    "results/burst_expanded_fusion/expansion_scores.pkl",
    "results/burst_expanded_fusion/rerank_scores.pkl",
    "results/burst_fresh_block/title_embed_public.pkl",
    "results/burst_fresh_block/title_embed_scores.pkl",
    "results/burst_gpu_threeview/cpu_top20.pkl",
    "results/burst_gpu_threeview/gpu_scores.checkpoint.pkl",
    "results/burst_large_ltr/fresh_1251_1350_retrieval.pkl",
    "results/burst_large_ltr/fresh_1351_1450_retrieval.pkl",
    "results/burst_large_ltr/fresh_1451_1750_retrieval.pkl",
    "results/burst_large_ltr/retrieval_train1000_tune50_val100.pkl",
    "results/burst_robust_fusion/public_retrieval.pkl",
    "results/burst_userft_maxrecall/vnlegal_scores.pkl",
    "results/corpus_index/holdout_dense_rank_cap32.pkl",
    "results/corpus_index/holdout_extended_scores_cap32.pkl",
    "results/crossenc_fullpool/cv_scores.pkl",
    "results/crossenc_fullpool/public_scores.pkl",
    "results/dense_expansion/union50_scores.pkl",
    "results/e5_dense/holdout_scores.pkl",
    "results/embedding_finetune/vnlegal_lal_cv_scores.pkl",
    "results/expanded_rerank/scores.pkl",
    "results/from_drive/aiteamvn_ft_cv.pkl",
    "results/from_drive/aiteamvn_ft_public.pkl",
    "results/from_drive/jina_ft_cv.pkl",
    "results/from_drive/jina_ft_public.pkl",
    "results/jina_reranker/holdout_scores_finetuned.pkl",
    "results/vietnamese_reranker/holdout_scores_512_finetuned.pkl",
]

ARGS = [
    "--crossenc", "--alpha", "0",
    "--output-dir", "results/burst_userft_maxrecall",
    "--extra-channel",
    "aiteamvn_ft=results/from_drive/aiteamvn_ft_cv.pkl,"
    "results/from_drive/aiteamvn_ft_public.pkl",
    "--extra-channel",
    "jina_ft=results/from_drive/jina_ft_cv.pkl,"
    "results/from_drive/jina_ft_public.pkl",
    "--extra-channel",
    "title_embed=results/burst_fresh_block/title_embed_scores.pkl,"
    "results/burst_fresh_block/title_embed_public.pkl",
]


def check() -> bool:
    missing = []
    for rel in REQUIRED:
        p = ROOT / rel
        if not p.exists():
            missing.append(rel)
            print(f"  MISSING            {rel}")
            continue
        if p.is_dir():
            files = [f for f in p.rglob("*") if f.is_file()]
            size = sum(f.stat().st_size for f in files)
            print(f"  ok      {size / 2**20:8.1f} MB  {rel}  ({len(files)} files)")
        else:
            print(f"  ok      {p.stat().st_size / 2**20:8.1f} MB  {rel}")
    if missing:
        print(f"\n{len(missing)} required path(s) missing -- bundle incomplete.")
        return False
    print("\nAll required inputs present.")
    return True


def main() -> int:
    print("=== checking bundled inputs ===")
    if not check():
        return 1
    if "--check" in sys.argv:
        return 0

    print("\n=== rebuilding submission ===", flush=True)
    sys.path.insert(0, str(ROOT))
    import run_vnlegal_extra_channel_submission as runner

    # Every vnlegal-lal score is already cached, so the model is never loaded --
    # but the runner would still download 1.2 GB from HuggingFace before
    # discovering that. Neutralised so this bundle needs no network at all.
    runner.ensure_vnlegal_model = lambda root: print(
        "  vnlegal-lal: not needed, its scores are cached", flush=True)

    sys.argv = ["run_vnlegal_extra_channel_submission.py"] + ARGS
    cwd = os.getcwd()
    os.chdir(ROOT)
    try:
        runner.main()
    finally:
        os.chdir(cwd)

    out = ROOT / "results/burst_userft_maxrecall/submission.json"
    digest = hashlib.md5(out.read_bytes()).hexdigest()[:10]
    print(f"\nsubmission.json md5[:10] = {digest}  (expected {EXPECTED_MD5})")
    if digest == EXPECTED_MD5:
        print("MATCH -- reproduced exactly")
        print(f"submit: {ROOT / 'results/burst_userft_maxrecall/submission.zip'}")
        return 0
    print("DIFFERENT -- the bundle produced another answer set; see README.md")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
