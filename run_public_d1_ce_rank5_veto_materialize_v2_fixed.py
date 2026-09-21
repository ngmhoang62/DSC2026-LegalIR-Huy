#!/usr/bin/env python
"""
PUBLIC D1 + FOLD0 BGE CE RANK5 VETO MATERIALIZER V1
====================================================

Creates ready-to-submit variable-K public candidates using the already-trained
Fold0 BGE cross-encoder as a DROP-ONLY semantic verifier.

Scientific contract
-------------------
- Public base is the exact D1 champion:
    results/gemini/huy_vnlegal_rank_ablation_v1/
    CANDIDATE_D1_VNLEGAL_SCORE_ONLY.json
- CE checkpoint is the completed Fold0 checkpoint trained on nonCAL only.
- Evidence rendering is distribution-matched to Fold0:
    * same frozen 343347 x 1024 VietLegal-E5 chunk bank
    * public queries encoded with the EXACT EXP-021 frozen query encoder:
      mainguyen9/vietlegal-e5, prefix "query: ", attention-mask mean pooling,
      L2 normalization
    * same sibling exp_final Evidence.package() implementation/database
- Thresholds are re-derived ONLY from deterministic Fold0 DEV and asserted
  against the already-observed frozen REL_L0 threshold.
- No CAL600 labels are read anywhere in this script.
- No public labels exist/read.
- DROP rank5 only. Ranks 1-4 are immutable.
- If any D1 Top5 document has no frozen evidence, abstain on the whole query.

Outputs
-------
Four ready ZIPs from ONE public CE scoring pass:
  REL_L0, REL_L025, REL_L05, REL_L10

Primary planned public probe:
  REL_L0
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import os
import sqlite3
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import torch

SALT = "dsc2026-endgame-ce-rank5-veto-v1"

ARMS = {
    "REL_L0": 0.0,
    "REL_L025": 0.25,
    "REL_L05": 0.5,
    "REL_L10": 1.0,
}

EXPECTED_REL_L0_THRESHOLD = -3.0393552780151367
EXPECTED_FOLD0_DEV_GOLD_RANK5 = 17
EXPECTED_PUBLIC_QUERIES = 1000
EXPECTED_DOCS = 8507
EXPECTED_CHUNKS = 343347
E5_DIM = 1024
E5_MODEL_ID = "mainguyen9/vietlegal-e5"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def digest_json(value) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("cebase_public", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def robust_scale(values) -> float:
    x = np.asarray(values, dtype=np.float64)
    med = np.median(x)
    mad = np.median(np.abs(x - med)) * 1.4826
    q75, q25 = np.percentile(x, [75, 25])
    iqr = (q75 - q25) / 1.349
    std = np.std(x)
    return max(float(mad), float(iqr), float(std), 1e-6)


def split_fold0(qids):
    dev, confirm = [], []
    for q in qids:
        h = int(hashlib.sha256(f"{SALT}|{q}".encode()).hexdigest(), 16)
        (dev if h % 2 == 0 else confirm).append(q)
    return dev, confirm


def derive_frozen_thresholds(base_module, world, root: Path):
    """
    Reproduce the already-frozen Fold0 DEV threshold without touching CAL labels.
    """
    fold0 = [
        q for q in world["noncal"]
        if world["folds"][q] == "fold_0"
    ]
    dev, confirm = split_fold0(fold0)

    score_dir = (
        root
        / "results/manual/huy_noncal_trainable_ce_boundary_v1/"
        "oof/fold_0/scores"
    )
    gold_rel = []

    for q in dev:
        top5 = world["base"][q][:5]
        if top5[4] not in world["gold"][q]:
            continue
        p = score_dir / f"{q}.json"
        if not p.is_file():
            raise FileNotFoundError(p)
        obj = json.loads(p.read_text(encoding="utf-8"))
        scores = {str(d): float(s) for d, s in obj["scores"].items()}
        missing = [d for d in top5 if d not in scores]
        if missing:
            raise RuntimeError(
                f"Fold0 score cache lacks base Top5 q={q}: {missing}"
            )
        rel = scores[top5[4]] - float(
            np.median([scores[d] for d in top5[:4]])
        )
        gold_rel.append(float(rel))

    if len(gold_rel) != EXPECTED_FOLD0_DEV_GOLD_RANK5:
        raise RuntimeError(
            "Frozen threshold population drift: "
            f"gold rank5 DEV={len(gold_rel)} expected="
            f"{EXPECTED_FOLD0_DEV_GOLD_RANK5}"
        )

    minimum = float(min(gold_rel))
    scale = robust_scale(gold_rel)

    if abs(minimum - EXPECTED_REL_L0_THRESHOLD) > 1e-9:
        raise RuntimeError(
            "Frozen REL_L0 threshold drift: "
            f"observed={minimum:.16f} "
            f"expected={EXPECTED_REL_L0_THRESHOLD:.16f}"
        )

    thresholds = {
        arm: float(minimum - lam * scale)
        for arm, lam in ARMS.items()
    }
    return {
        "fold0_n": len(fold0),
        "dev_n": len(dev),
        "confirm_n": len(confirm),
        "gold_rank5_dev_n": len(gold_rel),
        "gold_rank5_dev_min": minimum,
        "robust_scale": scale,
        "thresholds": thresholds,
        "split_salt": SALT,
    }


def load_public_inputs(root: Path):
    public_path = (
        root
        / "DSC2026-LegalIR-main/v4_run/public_test_dataset/"
        "public-official.json"
    )
    d1_path = (
        root
        / "results/gemini/huy_vnlegal_rank_ablation_v1/"
        "CANDIDATE_D1_VNLEGAL_SCORE_ONLY.json"
    )
    if not public_path.is_file():
        raise FileNotFoundError(public_path)
    if not d1_path.is_file():
        raise FileNotFoundError(d1_path)

    public_raw = json.loads(public_path.read_text(encoding="utf-8"))
    questions = {
        str(q): str(row["question"])
        for q, row in public_raw.items()
    }
    if len(questions) != EXPECTED_PUBLIC_QUERIES:
        raise RuntimeError(
            f"Expected {EXPECTED_PUBLIC_QUERIES} public queries, "
            f"got {len(questions)}"
        )

    d1_raw = json.loads(d1_path.read_text(encoding="utf-8"))
    base = {}
    for q, row in d1_raw.items():
        q = str(q)
        if isinstance(row, dict):
            if "answer" not in row:
                raise RuntimeError(
                    f"Unexpected D1 champion row q={q}: {list(row)}"
                )
            row = row["answer"]
        docs = [str(d) for d in row]
        if len(docs) != 5 or len(set(docs)) != 5:
            raise RuntimeError(f"Invalid exact public D1 Top5 q={q}: {docs}")
        base[q] = docs

    if set(base) != set(questions):
        raise RuntimeError(
            "Exact public D1 qids do not equal public-official qids"
        )

    ids = list(map(str, public_raw.keys()))
    return ids, questions, base, public_path, d1_path


def encode_public_queries(
    *,
    questions,
    ids,
    model_dir: Path,
    output_dir: Path,
    batch_size: int,
    device: str,
):
    """
    Exact EXP-021 query encoder:
      "query: " prefix
      AutoModel
      attention-mask mean pooling
      L2 normalize
      float32
    """
    from transformers import AutoModel, AutoTokenizer

    output_dir.mkdir(parents=True, exist_ok=True)
    ids_path = output_dir / "public_query_ids.json"
    matrix_path = output_dir / "public_queries.f32.npy"
    manifest_path = output_dir / "PUBLIC_QUERY_VECTOR_MANIFEST.json"

    source_contract = {
        "model_id": E5_MODEL_ID,
        "model_dir_sha256": tree_sha256(model_dir),
        "query_prefix": "query: ",
        "pooling": "attention_mask_mean",
        "normalize": "l2",
        "dimension": E5_DIM,
        "dtype": "float32",
        "public_questions_fingerprint": digest_json(
            [(q, questions[q]) for q in ids]
        ),
    }
    contract_fp = digest_json(source_contract)

    if matrix_path.is_file() and ids_path.is_file() and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        cached_ids = [str(x) for x in json.loads(ids_path.read_text(encoding="utf-8"))]
        matrix = np.load(matrix_path, mmap_mode="r")
        if (
            manifest.get("contract_fingerprint") == contract_fp
            and cached_ids == ids
            and matrix.shape == (len(ids), E5_DIM)
            and matrix.dtype == np.float32
        ):
            print("  Reusing exact public E5 query-vector cache.", flush=True)
            return matrix_path, ids_path, manifest_path

    print(
        f"  Encoding {len(ids)} public queries with frozen {E5_MODEL_ID}...",
        flush=True,
    )
    tok = AutoTokenizer.from_pretrained(
        str(model_dir),
        local_files_only=True,
        use_fast=True,
    )
    model = AutoModel.from_pretrained(
        str(model_dir),
        local_files_only=True,
    ).to(device).eval()

    matrix = np.lib.format.open_memmap(
        matrix_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(ids), E5_DIM),
    )

    started = time.perf_counter()
    for start in range(0, len(ids), batch_size):
        batch_ids = ids[start:start + batch_size]
        texts = ["query: " + questions[q] for q in batch_ids]
        encoded = tok(
            texts,
            add_special_tokens=True,
            truncation=False,
            padding=True,
            return_tensors="pt",
        )
        lengths = encoded["attention_mask"].sum(dim=1)
        if int(lengths.max()) > 512:
            raise RuntimeError(
                f"Public query exceeds frozen E5 512-token contract: "
                f"batch={batch_ids}"
            )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        with torch.inference_mode():
            hidden = model(**encoded).last_hidden_state
            weights = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (
                (hidden * weights).sum(1)
                / weights.sum(1).clamp_min(1e-9)
            )
            vectors = torch.nn.functional.normalize(
                pooled, p=2, dim=1
            ).float().cpu().numpy().astype(np.float32)

        matrix[start:start + len(batch_ids)] = vectors

        done = start + len(batch_ids)
        if done % 160 == 0 or done == len(ids):
            print(
                f"    public qvec {done}/{len(ids)} "
                f"qps={done/max(time.perf_counter()-started,1e-9):.2f}",
                flush=True,
            )

    matrix.flush()
    del matrix
    ids_path.write_text(
        json.dumps(ids, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    manifest = {
        **source_contract,
        "contract_fingerprint": contract_fp,
        "matrix_sha256": sha256(matrix_path),
        "ids_sha256": sha256(ids_path),
    }
    dump(manifest_path, manifest)

    del model, tok
    gc.collect()
    torch.cuda.empty_cache()
    return matrix_path, ids_path, manifest_path


def tree_sha256(path: Path) -> str:
    h = hashlib.sha256()
    files = sorted(
        (p for p in path.rglob("*") if p.is_file()),
        key=lambda p: p.relative_to(path).as_posix(),
    )
    if not files:
        raise RuntimeError(f"Empty model directory: {path}")
    for p in files:
        h.update(p.relative_to(path).as_posix().encode("utf-8"))
        h.update(bytes.fromhex(sha256(p)))
    return h.hexdigest()


class PublicRenderData:
    """
    Distribution-matched public variant of Fold0 RenderData.
    """

    def __init__(
        self,
        *,
        bundle: Path,
        questions,
        qvec_path: Path,
        qids_path: Path,
        sibling_evidence_db: Path,
    ):
        self.questions = dict(questions)

        con = sqlite3.connect(
            f"file:{sibling_evidence_db.resolve().as_posix()}"
            "?mode=ro&immutable=1",
            uri=True,
        )
        row = con.execute(
            "SELECT signature FROM complete WHERE kind='inventory'"
        ).fetchone()
        con.close()
        if row is None:
            raise RuntimeError("Sibling Evidence inventory signature missing")
        self.fingerprint = str(row[0])

        chunk_rows = list(read_jsonl(bundle / "chunk_ids.jsonl"))
        self.chunk_ids = [str(r["chunk_id"]) for r in chunk_rows]
        parents = [str(r["doc_id"]) for r in chunk_rows]
        if len(self.chunk_ids) != EXPECTED_CHUNKS:
            raise RuntimeError(
                f"Chunk count drift: {len(self.chunk_ids)}"
            )

        self.doc_ids = sorted(set(parents))
        if len(self.doc_ids) != EXPECTED_DOCS:
            raise RuntimeError(f"Parent count drift: {len(self.doc_ids)}")

        self.doc_row = {d: i for i, d in enumerate(self.doc_ids)}
        positions = [[] for _ in self.doc_ids]
        for i, d in enumerate(parents):
            positions[self.doc_row[d]].append(i)
        self.positions = [
            np.asarray(v, dtype=np.int64)
            for v in positions
        ]

        self._matrix = np.load(
            bundle / "embeddings.f16.npy",
            mmap_mode="r",
        )
        if self._matrix.shape != (EXPECTED_CHUNKS, E5_DIM):
            raise RuntimeError(
                f"Frozen chunk matrix shape drift: {self._matrix.shape}"
            )

        qids = [
            str(x)
            for x in json.loads(qids_path.read_text(encoding="utf-8"))
        ]
        self._qvec = np.load(qvec_path, mmap_mode="r")
        self._qrow = {q: i for i, q in enumerate(qids)}

        if set(self.questions) != set(self._qrow):
            raise RuntimeError("Public qvec cache population mismatch")
        if self._qvec.shape != (len(qids), E5_DIM):
            raise RuntimeError(
                f"Public qvec matrix shape drift: {self._qvec.shape}"
            )

    def matrix(self, source: str):
        if source != "e5":
            raise ValueError(source)
        return self._matrix

    def query_vector(self, qid: str, source: str):
        if source != "e5":
            raise ValueError(source)
        v = np.asarray(
            self._qvec[self._qrow[qid]],
            dtype=np.float32,
        )
        return v / max(float(np.linalg.norm(v)), 1e-12)


@torch.inference_mode()
def score_exact_public_d1(
    *,
    sibling: Path,
    render,
    ids,
    base,
    checkpoint: Path,
    output_dir: Path,
    pair_microbatch: int,
):
    sys.path[:0] = [str(sibling), str(sibling / "src")]
    from exp_final.cross_encoder import CrossEncoder
    from exp_final.evidence import Evidence

    model = CrossEncoder(checkpoint)
    model.eval()
    evidence = Evidence(render, model.tokenizer)

    score_dir = output_dir / "ce_scores"
    score_dir.mkdir(parents=True, exist_ok=True)

    ckpt_sha = sha256(checkpoint)
    qvec_fp = sha256(
        output_dir / "public_query_vectors/public_queries.f32.npy"
    )
    rows = {}
    no_evidence = {}
    started = time.perf_counter()

    try:
        for i, q in enumerate(ids, 1):
            top5 = base[q]

            missing_docs = [
                d for d in top5
                if evidence.db.execute(
                    "SELECT 1 FROM chunks WHERE doc=? LIMIT 1",
                    (d,),
                ).fetchone() is None
            ]
            if missing_docs:
                no_evidence[q] = missing_docs
                continue

            signature = digest_json(
                [
                    "public-exact-d1-ce-top5-v1",
                    ckpt_sha,
                    qvec_fp,
                    q,
                    top5,
                ]
            )
            cache_path = score_dir / f"{q}.json"

            if cache_path.is_file():
                obj = json.loads(
                    cache_path.read_text(encoding="utf-8")
                )
                if obj.get("signature") != signature:
                    raise RuntimeError(
                        f"Public CE cache signature mismatch q={q}"
                    )
                scores = {
                    str(d): float(s)
                    for d, s in obj["scores"].items()
                }
            else:
                values = []
                for st in range(0, 5, pair_microbatch):
                    docs = top5[st:st + pair_microbatch]
                    pairs = [
                        evidence.package(q, d)
                        for d in docs
                    ]
                    values.extend(
                        model(pairs).detach().cpu().tolist()
                    )
                scores = {
                    d: float(s)
                    for d, s in zip(top5, values)
                }
                dump(
                    cache_path,
                    {
                        "signature": signature,
                        "top5": top5,
                        "scores": scores,
                    },
                )

            rel = (
                scores[top5[4]]
                - float(np.median([scores[d] for d in top5[:4]]))
            )
            rows[q] = {
                "top5": top5,
                "scores": scores,
                "rel_top4_med": float(rel),
            }

            if i % 50 == 0 or i == len(ids):
                print(
                    f"    CE public scan {i}/{len(ids)} "
                    f"scored={len(rows)} "
                    f"abstain_no_evidence={len(no_evidence)} "
                    f"qps={i/max(time.perf_counter()-started,1e-9):.3f}",
                    flush=True,
                )
    finally:
        evidence.db.close()
        del evidence, model
        gc.collect()
        torch.cuda.empty_cache()

    return rows, no_evidence


def validate_submission(
    submission,
    *,
    ids,
    base,
    actions,
):
    if set(submission) != set(ids):
        raise RuntimeError("Submission qid population mismatch")

    k_hist = {}
    for q in ids:
        row = submission[q]
        if not isinstance(row, dict) or "answer" not in row:
            raise RuntimeError(f"Bad row schema q={q}")
        ans = [str(d) for d in row["answer"]]
        if len(ans) not in (4, 5):
            raise RuntimeError(f"Unexpected K q={q}: {len(ans)}")
        if len(set(ans)) != len(ans):
            raise RuntimeError(f"Duplicate output q={q}")
        # Output may contain a competition-valid D1 document that is outside
        # the frozen CE evidence universe.  That is fine: this policy never
        # introduces documents; it can only preserve D1 or drop D1 rank5.
        if not set(ans) <= set(base[q]):
            raise RuntimeError(f"Output introduced non-D1 doc q={q}")
        if q in actions:
            if ans != base[q][:4]:
                raise RuntimeError(
                    f"Action query is not exact rank5 drop q={q}"
                )
        else:
            if ans != base[q]:
                raise RuntimeError(
                    f"Abstention query changed unexpectedly q={q}"
                )
        k_hist[str(len(ans))] = k_hist.get(str(len(ans)), 0) + 1

    return k_hist


def zip_exact(json_path: Path, zip_path: Path):
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(
        zip_path,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as zf:
        zf.write(json_path, arcname="submission.json")
    with zipfile.ZipFile(zip_path, "r") as zf:
        if zf.namelist() != ["submission.json"]:
            raise RuntimeError("ZIP member contract failed")
        if zf.read("submission.json") != json_path.read_bytes():
            raise RuntimeError("ZIP byte parity failed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=Path, required=True)
    ap.add_argument("--pair-microbatch", type=int, default=4)
    ap.add_argument("--e5-batch-size", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    root = args.repo_root.resolve()
    sibling = root.parent / "LegalIR"
    base_script = (
        root.parent
        / "run_noncal_trainable_ce_boundary_v3_fixed.py"
    )
    if not base_script.is_file():
        raise FileNotFoundError(base_script)

    bundle = (
        root
        / "cache/research_v2_e5_confirmation/bundle-v1"
    )
    model_dir = bundle / "vietlegal-e5"
    checkpoint = (
        root
        / "results/manual/huy_noncal_trainable_ce_boundary_v1/"
        "oof/fold_0/training/model.pt"
    )
    sibling_evidence_db = (
        sibling / "cache/exp_final_retrieval/evidence.sqlite"
    )

    for p in (
        bundle / "embeddings.f16.npy",
        bundle / "chunk_ids.jsonl",
        model_dir,
        checkpoint,
        sibling_evidence_db,
    ):
        if not p.exists():
            raise FileNotFoundError(p)

    out = (
        root
        / "results/manual/huy_public_d1_ce_rank5_veto_v1"
    )
    out.mkdir(parents=True, exist_ok=True)

    print("[1/7] Loading exact public D1 and public questions...")
    ids, questions, base, public_path, d1_path = load_public_inputs(root)
    print(
        f"  public={len(ids)} exact_D1={len(base)} "
        f"D1_sha256={sha256(d1_path)}",
        flush=True,
    )

    print("[2/7] Re-deriving frozen thresholds from Fold0 DEV only...")
    m = load_module(base_script)
    sys.path[:0] = [
        str(root),
        str(root / "src"),
        str(sibling),
        str(sibling / "src"),
    ]
    cal_ids, _ = m.get_cal_ids_label_free(root)
    world = m.load_noncal_world(root, sibling, set(cal_ids))
    threshold_contract = derive_frozen_thresholds(
        m, world, root
    )
    print(
        json.dumps(
            threshold_contract,
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )

    print("[3/7] Encoding public queries with exact frozen EXP-021 E5...")
    qvec_dir = out / "public_query_vectors"
    qvec_path, qids_path, qvec_manifest = encode_public_queries(
        questions=questions,
        ids=ids,
        model_dir=model_dir,
        output_dir=qvec_dir,
        batch_size=args.e5_batch_size,
        device=args.device,
    )

    print("[4/7] Building distribution-matched public Evidence renderer...")
    render = PublicRenderData(
        bundle=bundle,
        questions=questions,
        qvec_path=qvec_path,
        qids_path=qids_path,
        sibling_evidence_db=sibling_evidence_db,
    )

    canonical = set(render.doc_ids)
    if any(not set(base[q]) <= canonical for q in ids):
        bad = [
            q for q in ids
            if not set(base[q]) <= canonical
        ]
        print(
            f"  note: {len(bad)} public D1 queries contain "
            "docs outside frozen canonical Evidence inventory; "
            "they will be conservative abstentions if evidence is absent.",
            flush=True,
        )

    print("[5/7] Scoring exact public D1 Top5 with frozen Fold0 BGE CE...")
    rows, no_evidence = score_exact_public_d1(
        sibling=sibling,
        render=render,
        ids=ids,
        base=base,
        checkpoint=checkpoint,
        output_dir=out,
        pair_microbatch=args.pair_microbatch,
    )
    print(
        f"  CE scorable={len(rows)} "
        f"no-evidence abstentions={len(no_evidence)}",
        flush=True,
    )

    print("[6/7] Freezing actions and materializing ZIP candidates...")
    thresholds = threshold_contract["thresholds"]
    arms_report = {}

    for arm, threshold in thresholds.items():
        arm_dir = out / arm
        arm_dir.mkdir(parents=True, exist_ok=True)

        actions = {}
        submission = {}

        for q in ids:
            if (
                q in rows
                and rows[q]["rel_top4_med"] < threshold
            ):
                submission[q] = {"answer": list(base[q][:4])}
                actions[q] = {
                    "qid": q,
                    "removed_rank5": base[q][4],
                    "rel_top4_med": rows[q]["rel_top4_med"],
                    "threshold": threshold,
                    "original_top5": base[q],
                    "new_answer": base[q][:4],
                }
            else:
                submission[q] = {"answer": list(base[q])}

        k_hist = validate_submission(
            submission,
            ids=ids,
            base=base,
            actions=actions,
        )

        action_path = arm_dir / "PUBLIC_ACTIONS_LABEL_FREE.json"
        sub_path = arm_dir / "submission.json"
        zip_path = arm_dir / f"submission_{arm}.zip"

        dump(
            action_path,
            {
                "schema": "manual.public_d1_ce_rank5_veto_v1.actions",
                "status": "SEALED_LABEL_FREE",
                "arm": arm,
                "threshold": threshold,
                "threshold_source": (
                    "Fold0 deterministic DEV only; "
                    "frozen before public scoring"
                ),
                "checkpoint_sha256": sha256(checkpoint),
                "exact_d1_sha256": sha256(d1_path),
                "public_questions_sha256": sha256(public_path),
                "public_qvec_sha256": sha256(qvec_path),
                "actions_count": len(actions),
                "no_evidence_abstentions": no_evidence,
                "actions": actions,
            },
        )
        dump(sub_path, submission)
        zip_exact(sub_path, zip_path)

        arms_report[arm] = {
            "threshold": threshold,
            "actions": len(actions),
            "prune_rate": len(actions) / len(ids),
            "mean_k": (
                sum(len(submission[q]["answer"]) for q in ids)
                / len(ids)
            ),
            "k_histogram": k_hist,
            "submission_json": str(sub_path),
            "submission_zip": str(zip_path),
            "submission_json_sha256": sha256(sub_path),
            "submission_zip_sha256": sha256(zip_path),
            "actions_sha256": sha256(action_path),
        }

    print("[7/7] Writing production report...")
    report = {
        "schema": "manual.public_d1_ce_rank5_veto_v1.report",
        "status": "READY_LOCAL_CANDIDATES_NOT_UPLOADED",
        "primary_arm": "REL_L0",
        "scientific_contract": {
            "public_base": "exact D1 champion",
            "ce_checkpoint": "Fold0 nonCAL-only BGE CE",
            "query_encoder": E5_MODEL_ID,
            "query_prefix": "query: ",
            "query_pooling": "attention-mask mean + L2",
            "chunk_bank": "same frozen EXP-021 343347x1024 bank",
            "evidence_renderer": "exp_final.evidence.Evidence.package",
            "policy": "drop exact D1 rank5 only",
            "cal_gold_used": False,
            "public_labels_used": False,
        },
        "threshold_contract": threshold_contract,
        "population": {
            "public_queries": len(ids),
            "ce_scorable": len(rows),
            "no_evidence_abstentions": len(no_evidence),
        },
        "arms": arms_report,
        "provenance": {
            "base_script": str(base_script),
            "base_script_sha256": sha256(base_script),
            "exact_d1": str(d1_path),
            "exact_d1_sha256": sha256(d1_path),
            "public_questions": str(public_path),
            "public_questions_sha256": sha256(public_path),
            "fold0_checkpoint": str(checkpoint),
            "fold0_checkpoint_sha256": sha256(checkpoint),
            "public_qvec_manifest": str(qvec_manifest),
            "public_qvec_sha256": sha256(qvec_path),
            "bundle_chunk_bank_sha256": sha256(
                bundle / "embeddings.f16.npy"
            ),
            "bundle_chunk_ids_sha256": sha256(
                bundle / "chunk_ids.jsonl"
            ),
        },
    }
    report_path = out / "PUBLIC_CE_VETO_REPORT.json"
    dump(report_path, report)

    print("=" * 108)
    print("PUBLIC CE RANK5 VETO CANDIDATES")
    print(
        f"Exact D1 public queries={len(ids)} | "
        f"CE scorable={len(rows)} | "
        f"no-evidence abstain={len(no_evidence)}"
    )
    for arm in ARMS:
        r = arms_report[arm]
        print(
            f"{arm:9s} threshold={r['threshold']:+.6f} "
            f"actions={r['actions']:4d} "
            f"rate={100*r['prune_rate']:.2f}% "
            f"meanK={r['mean_k']:.4f}"
        )
        print(f"           ZIP={r['submission_zip']}")
    print("PRIMARY PUBLIC PROBE: REL_L0")
    print(f"Report: {report_path}")
    print("=" * 108)


if __name__ == "__main__":
    main()
