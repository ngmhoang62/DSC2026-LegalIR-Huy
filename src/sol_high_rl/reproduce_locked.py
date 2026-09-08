"""Reproduce the immutable baseline into the sol_high_rl namespace."""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/sol_high_rl/baseline_reproduction"
BASE = ROOT / "results/burst_userft_maxrecall"
EXPECTED_MD5_PREFIX = "2fb9a8a3b7"


def digest(path: Path, algorithm: str = "sha256") -> str:
    h = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    if OUT.resolve() == BASE.resolve():
        raise RuntimeError("Refusing to overwrite immutable baseline")
    OUT.mkdir(parents=True, exist_ok=True)
    before = {name: digest(BASE / name) for name in
              ("submission.json", "submission.zip", "run_metadata.json", "vnlegal_scores.pkl")}
    shutil.copy2(BASE / "vnlegal_scores.pkl", OUT / "vnlegal_scores.pkl")

    sys.path.insert(0, str(ROOT))
    import run_vnlegal_extra_channel_submission as runner

    runner.ensure_vnlegal_model = lambda root: print(
        "vnlegal-lal model load disabled: immutable cached scores copied", flush=True)
    sys.argv = [
        "run_vnlegal_extra_channel_submission.py",
        "--crossenc", "--alpha", "0",
        "--output-dir", str(OUT),
        "--extra-channel",
        "aiteamvn_ft=results/from_drive/aiteamvn_ft_cv.pkl,results/from_drive/aiteamvn_ft_public.pkl",
        "--extra-channel",
        "jina_ft=results/from_drive/jina_ft_cv.pkl,results/from_drive/jina_ft_public.pkl",
        "--extra-channel",
        "title_embed=results/burst_fresh_block/title_embed_scores.pkl,results/burst_fresh_block/title_embed_public.pkl",
    ]
    runner.main()

    submission = OUT / "submission.json"
    payload = json.loads(submission.read_text(encoding="utf-8"))
    corpus_ids = {
        p.stem[len("context_"):]
        for p in (ROOT / "DSC2026-LegalIR-main/v4_run/public_test_dataset/selected-contexts")
        .glob("context_*.json")
    }
    invalid = {}
    for q, row in payload.items():
        docs = list(map(str, row.get("answer", [])))
        if len(docs) != 5 or len(set(docs)) != 5 or any(d not in corpus_ids for d in docs):
            invalid[q] = docs
    with zipfile.ZipFile(OUT / "submission.zip") as archive:
        zip_names = archive.namelist()
        zip_payload = archive.read("submission.json")
    after = {name: digest(BASE / name) for name in before}
    manifest = {
        "status": "MATCH" if digest(submission, "md5")[:10] == EXPECTED_MD5_PREFIX else "DIFFERENT",
        "expected_md5_prefix": EXPECTED_MD5_PREFIX,
        "submission_json_md5": digest(submission, "md5"),
        "submission_json_sha256": digest(submission),
        "submission_zip_sha256": digest(OUT / "submission.zip"),
        "queries": len(payload),
        "exactly_five_unique_valid_docs": not invalid and len(payload) == 1000,
        "invalid_queries": invalid,
        "zip_names": zip_names,
        "zip_json_matches_file": hashlib.sha256(zip_payload).hexdigest() == digest(submission),
        "immutable_baseline_hashes_before": before,
        "immutable_baseline_hashes_after": after,
        "immutable_baseline_unchanged": before == after,
        "reproduce_command": ".\\dsc_env_huy\\Scripts\\python.exe -u src/sol_high_rl/reproduce_locked.py",
        "uploaded": False,
    }
    path = OUT / "REPRODUCTION_MANIFEST.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    if manifest["status"] != "MATCH" or not manifest["exactly_five_unique_valid_docs"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
