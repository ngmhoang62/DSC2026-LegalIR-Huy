"""Generate the user-facing Kaggle 2xT4 notebook artifact."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TARGET = ROOT / "results/research_v2_open_rl/E5_PASSAGE_ALIGNMENT_2XT4_KAGGLE.ipynb"


def markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": text.splitlines(keepends=True)}


cells = [
    markdown("""# DSC2026 Research V2 — sealed E5 passage alignment on 2×T4

Attach the private Kaggle Dataset created from
`kaggle_bundle_e5_passage_alignment_2xt4_v1`, select the **GPU T4 x2**
accelerator, then run all cells in order. The notebook is resumable inside the
same Kaggle session. It stops before training if input hashes, numerical parity,
two-GPU availability, memory, or projected 3-hour cost gates fail.

The final evaluator is executed exactly once. Do not change batch size, dtype,
attention implementation, passages, loss, optimizer, seed, epochs, or gates.
No internet or model download is used when the Kaggle image already provides
PyTorch, Transformers and PEFT.
"""),
    code("""from pathlib import Path
import hashlib, importlib.util, json, os, platform, subprocess, sys, time

os.environ.update({
    "TOKENIZERS_PARALLELISM": "false",
    "PYTHONHASHSEED": "113",
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "OMP_NUM_THREADS": "2",
    "MKL_NUM_THREADS": "2",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
})

manifests = list(Path("/kaggle/input").rglob("BUNDLE_MANIFEST.json"))
assert len(manifests) == 1, f"Expected exactly one attached sealed bundle, found: {manifests}"
INPUT_ROOT = manifests[0].parent
OUTPUT_ROOT = Path("/kaggle/working/research_v2_e5_passage_alignment_2xt4")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
RUNNER = INPUT_ROOT / "code/kaggle_e5_passage_alignment_2xt4.py"
assert RUNNER.is_file()
print("INPUT_ROOT =", INPUT_ROOT)
print("OUTPUT_ROOT =", OUTPUT_ROOT)
"""),
    code("""required = ["torch", "transformers", "peft", "numpy", "safetensors", "sentencepiece"]
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    raise RuntimeError(
        f"Missing {missing}. Enable Kaggle internet only for this install, run: "
        f"{sys.executable} -m pip install -r {INPUT_ROOT / 'code/requirements.txt'}, then restart the kernel."
    )

import numpy, peft, torch, transformers
assert torch.cuda.device_count() == 2, f"Select GPU T4 x2; detected {torch.cuda.device_count()} GPU(s)"
environment = {
    "python": sys.version,
    "platform": platform.platform(),
    "torch": torch.__version__,
    "transformers": transformers.__version__,
    "peft": peft.__version__,
    "numpy": numpy.__version__,
    "cuda": torch.version.cuda,
    "gpus": [torch.cuda.get_device_name(i) for i in range(2)],
}
(OUTPUT_ROOT / "ENVIRONMENT.json").write_text(json.dumps(environment, indent=2) + "\\n")
print(json.dumps(environment, indent=2))
"""),
    code("""def run_logged(name, command):
    log_path = OUTPUT_ROOT / f"{name}.log"
    print("RUN:", " ".join(map(str, command)), flush=True)
    with log_path.open("a", encoding="utf-8", buffering=1) as log:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, bufsize=1, env=os.environ.copy())
        for line in process.stdout:
            print(line, end="")
            log.write(line)
        code = process.wait()
    if code != 0:
        raise RuntimeError(f"{name} failed with exit code {code}; see {log_path}")

BASE = [sys.executable, str(RUNNER)]
ARGS = ["--input-root", str(INPUT_ROOT), "--output-root", str(OUTPUT_ROOT)]
TORCHRUN = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2", str(RUNNER)]
"""),
    markdown("""## 1. Fail-closed input, numerical and cost gates

This reads no Fold-0 metric. The parity stage compares scalar/batched execution,
FP32/FP16 loss and adapter gradients, exact replay, memory and projected total
runtime. Any failed sealed gate stops here.
"""),
    code("""run_logged("01_preflight", BASE + ["preflight"] + ARGS)
run_logged("02_parity", BASE + ["parity"] + ARGS)
gate = json.loads((OUTPUT_ROOT / "RUNTIME_PARITY_AND_COST_GATE.json").read_text())
assert gate["status"] == "PASS_RUNTIME_AND_COST_NO_HELD_METRIC"
print("Projected total minutes:", gate["cost"]["projected_total_seconds"] / 60)
"""),
    markdown("""## 2. Strict Fold-0 passage-adapter training

Two DDP ranks process the globally sealed effective batch of 16. Checkpoints,
optimizer/scheduler state and rank-specific RNG states are saved every ten
updates; rerunning this cell resumes rather than restarts.
"""),
    code("""run_logged("03_train", TORCHRUN + ["train"] + ARGS)
training = json.loads((OUTPUT_ROOT / "training/_SUCCESS.json").read_text())
assert training["status"] == "COMPLETE_EPOCH2" and training["updates"] == 700
print(json.dumps(training, indent=2))
"""),
    markdown("""## 3. Two-GPU Fold-0 scoring and fail-closed merge

Each GPU owns a stable half of the 1,398 queries and writes a separate resumable
SQLite shard. Merge requires all 146,249 sequences and 73,128 parent scores.
No evaluation metric is read in this section.
"""),
    code("""run_logged("04_score", TORCHRUN + ["score"] + ARGS)
run_logged("05_merge", BASE + ["merge"] + ARGS)
merged = json.loads((OUTPUT_ROOT / "SCORE_MERGE_REPORT.json").read_text())
assert merged["status"] == "COMPLETE_FAIL_CLOSED"
print(json.dumps(merged, indent=2))
"""),
    markdown("""## 4. One and only one sealed evaluation

This is the first cell that reads Fold-0 gold metrics. If a report already
exists, evaluation is skipped and only verification proceeds. The original
PASS/KILL gate is applied verbatim; there is no rescue grid.
"""),
    code("""report_path = OUTPUT_ROOT / "E5_PASSAGE_ALIGNMENT_TOP2MEAN_FOLD0_REPORT.json"
if report_path.exists():
    print("Sealed report already exists; refusing a second evaluation and proceeding to verification.")
else:
    run_logged("06_evaluate_once", BASE + ["evaluate"] + ARGS)
run_logged("07_verify", BASE + ["verify"] + ARGS)
report = json.loads(report_path.read_text())
print(json.dumps({"verdict": report["verdict"], "metrics": report["metrics"]}, indent=2))
"""),
    markdown("""## 5. Package outputs for return

Download the printed ZIP from Kaggle Output and return it unchanged. It includes
the adapter checkpoint, both score locks, strict report, predictions, manifests,
environment and execution logs.
"""),
    code("""import zipfile

zip_path = Path("/kaggle/working/E5_PASSAGE_ALIGNMENT_2XT4_RESULTS.zip")
with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
    for path in sorted(item for item in OUTPUT_ROOT.rglob("*") if item.is_file()):
        archive.write(path, arcname=path.relative_to(OUTPUT_ROOT).as_posix())
    provenance = [
        INPUT_ROOT / "BUNDLE_MANIFEST.json",
        INPUT_ROOT / "code/kaggle_e5_passage_alignment_2xt4.py",
        INPUT_ROOT / "preregistration/E5_PASSAGE_ALIGNMENT_TOP2MEAN_FOLD0_PREREGISTRATION.json",
        INPUT_ROOT / "preregistration/E5_PASSAGE_ALIGNMENT_2XT4_RUNTIME_PREREGISTRATION.json",
    ]
    for path in provenance:
        archive.write(path, arcname="provenance/" + path.name)

def file_sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()

receipt = {"path": str(zip_path), "bytes": zip_path.stat().st_size,
           "sha256": file_sha256(zip_path), "verdict": report["verdict"]}
Path("/kaggle/working/E5_PASSAGE_ALIGNMENT_2XT4_RESULTS_RECEIPT.json").write_text(
    json.dumps(receipt, indent=2) + "\\n"
)
print(json.dumps(receipt, indent=2))
"""),
]

notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "kaggle": {"accelerator": "nvidiaTeslaT4", "dataSources": []},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

TARGET.parent.mkdir(parents=True, exist_ok=True)
TARGET.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(TARGET)
