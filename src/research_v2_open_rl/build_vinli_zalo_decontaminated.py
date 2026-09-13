"""Create a removal-only external legal-NLI corpus after conservative V2 decontamination."""

from __future__ import annotations

import gzip
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


THRESHOLD = 0.75


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


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
    source = root / "cache/research_v2_open_rl/external_audits/vinli_zalo/law_vi.jsonl.gz"
    clean_path = root / "cache/research_v2_open_rl/external_audits/vinli_zalo/law_vi_decontaminated.jsonl"
    report_path = root / "results/research_v2_open_rl/EXTERNAL_LEGAL_NLI_DECONTAMINATION_AUDIT.json"
    train_path = workspace / "LegalIR/public_test_dataset/train.json"
    folds_path = root / "results/research_v2_forensic/V2_FOLDS.json"

    rows: list[dict] = []
    by_query: dict[str, list[int]] = defaultdict(list)
    raw_query: dict[str, str] = {}
    with gzip.open(source, "rt", encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            row = json.loads(line)
            if set(row) != {"query", "positive", "hard_neg"}:
                raise RuntimeError(f"unexpected source schema at row {index}")
            key = normalize(row["query"])
            rows.append(row)
            by_query[key].append(index)
            raw_query.setdefault(key, str(row["query"]))

    train = json.loads(train_path.read_text(encoding="utf-8"))
    local_qids = sorted(train, key=lambda value: int(value))
    local_queries = [normalize(train[qid].get("question", train[qid].get("query", ""))) for qid in local_qids]
    local_ascii = {normalize(train[qid].get("question", train[qid].get("query", "")), True) for qid in local_qids}
    external_queries = sorted(by_query)

    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), lowercase=False, dtype=np.float32)
    matrix = vectorizer.fit_transform(local_queries + external_queries)
    local_matrix = matrix[: len(local_queries)]
    external_matrix = matrix[len(local_queries) :]
    nearest: list[tuple[float, int]] = []
    for start in range(0, len(external_queries), 128):
        similarities = (external_matrix[start : start + 128] @ local_matrix.T).toarray()
        indices = similarities.argmax(axis=1)
        values = similarities[np.arange(len(indices)), indices]
        nearest.extend((float(value), int(index)) for value, index in zip(values, indices))

    contaminated: set[str] = set()
    evidence: list[dict] = []
    for query, (similarity, local_index) in zip(external_queries, nearest):
        accentless_exact = normalize(raw_query[query], True) in local_ascii
        if similarity >= THRESHOLD or accentless_exact:
            contaminated.add(query)
            evidence.append(
                {
                    "external_query": raw_query[query],
                    "normalized_query": query,
                    "source_rows": len(by_query[query]),
                    "nearest_v2_qid": str(local_qids[local_index]),
                    "nearest_v2_query": str(train[local_qids[local_index]].get("question", train[local_qids[local_index]].get("query", ""))),
                    "char_tfidf_cosine": similarity,
                    "accentless_exact": accentless_exact,
                }
            )

    with clean_path.open("w", encoding="utf-8", newline="\n") as sink:
        for row in rows:
            if normalize(row["query"]) not in contaminated:
                sink.write(canonical_json(row) + "\n")

    kept = sum(len(indices) for query, indices in by_query.items() if query not in contaminated)
    removed = len(rows) - kept
    similarity_bins = Counter()
    for similarity, _ in nearest:
        label = ">=0.90" if similarity >= 0.90 else "0.75-0.90" if similarity >= 0.75 else "0.60-0.75" if similarity >= 0.60 else "<0.60"
        similarity_bins[label] += 1
    report = {
        "schema_version": "dsc2026.research_v2.external_legal_nli_decontamination.v1",
        "status": "PASS_REMOVAL_ONLY_NO_AUGMENTATION" if kept > 0 and removed > 0 else "FAIL_CLOSED",
        "source": {
            "dataset": "anti-ai/ViNLI-Zalo-supervised",
            "revision": "e41cae3",
            "path": str(source),
            "sha256": sha256(source),
            "rows": len(rows),
            "unique_queries": len(external_queries),
            "license": "MIT",
        },
        "v2_guard": {
            "train_path": str(train_path),
            "train_sha256": sha256(train_path),
            "folds_sha256": sha256(folds_path),
            "queries": len(local_queries),
        },
        "method": {
            "normalization": "NFKC lowercase underscore-to-space punctuation-to-space whitespace-collapse",
            "near_duplicate_metric": "character_wb TF-IDF cosine, ngrams 3-5",
            "remove_if_cosine_gte": THRESHOLD,
            "remove_if_accentless_normalized_exact": True,
            "threshold_role": "conservative provenance filter only; never a ranking or metric rule",
        },
        "result": {
            "contaminated_unique_queries_removed": len(contaminated),
            "source_rows_removed": removed,
            "source_rows_kept": kept,
            "rows_added": 0,
            "rows_modified": 0,
            "augmentation": False,
            "similarity_bins_unique_external_queries": dict(sorted(similarity_bins.items())),
            "removed_evidence": sorted(evidence, key=lambda item: (-item["char_tfidf_cosine"], item["normalized_query"])),
        },
        "output": {"path": str(clean_path), "sha256": sha256(clean_path), "rows": kept},
        "training_authorization": "EXTERNAL_ONLY_FROZEN_CONFIGURATION_ALLOWED" if kept > 0 and removed > 0 else "FORBIDDEN",
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({
        "status": report["status"],
        "unique_removed": len(contaminated),
        "rows_removed": removed,
        "rows_kept": kept,
        "similarity_bins": dict(similarity_bins),
        "output_sha256": report["output"]["sha256"],
    }, indent=2))


if __name__ == "__main__":
    main()
