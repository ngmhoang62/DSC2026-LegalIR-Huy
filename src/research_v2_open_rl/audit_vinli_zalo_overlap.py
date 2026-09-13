"""Audit declared external training data for direct Research V2 query overlap."""

from __future__ import annotations

import gzip
import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize(text: str, strip_accents: bool = False) -> str:
    value = unicodedata.normalize("NFKC", str(text)).lower().replace("_", " ")
    if strip_accents:
        value = "".join(ch for ch in unicodedata.normalize("NFD", value) if unicodedata.category(ch) != "Mn")
        value = value.replace("đ", "d")
    return " ".join(re.sub(r"[^\w]+", " ", value, flags=re.UNICODE).split())


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    workspace = root.parent
    dataset = root / "cache/research_v2_open_rl/external_audits/vinli_zalo/law_vi.jsonl.gz"
    train_path = workspace / "LegalIR/public_test_dataset/train.json"
    folds_path = root / "results/research_v2_forensic/V2_FOLDS.json"
    output = root / "results/research_v2_open_rl/EXTERNAL_LEGAL_NLI_OVERLAP_AUDIT.json"

    external_counts: Counter[str] = Counter()
    external_ascii_counts: Counter[str] = Counter()
    rows = 0
    with gzip.open(dataset, "rt", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            external_counts[normalize(row["query"])] += 1
            external_ascii_counts[normalize(row["query"], strip_accents=True)] += 1
            rows += 1

    train = json.loads(train_path.read_text(encoding="utf-8"))
    folds = json.loads(folds_path.read_text(encoding="utf-8"))["folds"]
    qid_fold = {str(qid): fold for fold, qids in folds.items() for qid in qids}
    exact = []
    accentless = []
    for qid, row in train.items():
        query = str(row.get("question", row.get("query", "")))
        key = normalize(query)
        ascii_key = normalize(query, strip_accents=True)
        item = {
            "qid": str(qid),
            "fold": qid_fold[str(qid)],
            "query": query,
            "external_rows": external_counts[key],
        }
        if external_counts[key]:
            exact.append(item)
        if external_ascii_counts[ascii_key]:
            accentless.append(item)

    by_fold = Counter(item["fold"] for item in exact)
    qid_digest = hashlib.sha256(
        ("\n".join(sorted((item["qid"] for item in exact), key=int)) + "\n").encode("utf-8")
    ).hexdigest()
    status = "PROVENANCE_UNSAFE_DIRECT_QUERY_OVERLAP" if exact else "NO_EXACT_QUERY_OVERLAP_REQUIRES_DEEPER_AUDIT"
    report = {
        "schema_version": "dsc2026.research_v2.external_legal_nli_overlap_audit.v1",
        "status": status,
        "candidate_model": "ngdangkhanh/vietnamese-law-rerank-model",
        "declared_training_dataset": "anti-ai/ViNLI-Zalo-supervised",
        "dataset_revision": "e41cae3",
        "model_card_claim": "The checkpoint was fine-tuned on anti-ai/ViNLI-Zalo-supervised.",
        "dataset": {
            "path": str(dataset),
            "sha256": sha256(dataset),
            "rows": rows,
            "unique_normalized_queries": len(external_counts),
            "license": "MIT",
        },
        "research_v2": {
            "train_path": str(train_path),
            "train_sha256": sha256(train_path),
            "folds_sha256": sha256(folds_path),
            "queries": len(train),
        },
        "overlap": {
            "exact_normalized_queries": len(exact),
            "accentless_normalized_queries": len(accentless),
            "per_fold_exact": dict(sorted(by_fold.items())),
            "exact_qids_sha256": qid_digest,
            "sample": sorted(exact, key=lambda item: int(item["qid"]))[:25],
        },
        "decision": (
            "REJECT_MODEL_WITHOUT_DOWNLOAD_OR_INFERENCE"
            if exact
            else "CONTINUE_PROVENANCE_AUDIT"
        ),
        "reason": (
            "A checkpoint trained on direct Research V2 query text is not clean frozen external supervision; "
            "using it would contaminate held-fold evaluation even if document labels were transformed."
            if exact
            else "Exact matching did not establish overlap; provenance is not yet cleared."
        ),
        "sources": [
            "https://huggingface.co/ngdangkhanh/vietnamese-law-rerank-model",
            "https://huggingface.co/datasets/anti-ai/ViNLI-Zalo-supervised",
        ],
    }
    output.write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"status": status, "rows": rows, "overlap": report["overlap"], "decision": report["decision"]}, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
