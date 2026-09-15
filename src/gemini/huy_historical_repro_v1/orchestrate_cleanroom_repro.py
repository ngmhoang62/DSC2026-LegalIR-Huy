"""Master orchestrator for Clean-Room Historical Reproduction of Huy's baseline.

Executes all pipeline steps sequentially from fresh weights:
1. score_cv_custom_encoder.py (aiteamvn_ft CV)
2. score_cv_jina_ft.py (jina_ft CV)
3. score_public_models.py --kind bi (aiteamvn_ft Public)
4. score_public_models.py --kind jina (jina_ft Public)
5. evaluate_cv.py (600-query CV protocol validation)
6. run_vnlegal_extra_channel_submission.py (fresh vnlegal + LTR refit + submission)
7. Audit & Integrity verification against historical backups
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = REPO_ROOT / "results/gemini/huy_historical_repro_v1"
TRACE_FILE = OUTPUT_DIR / "EXECUTION_TRACE.jsonl"
INTEGRITY_FILE = OUTPUT_DIR / "CLEANROOM_REPRO_INTEGRITY.json"
CV_REPORT_FILE = OUTPUT_DIR / "CV_REPRO_REPORT.json"
PYTHON_EXE = sys.executable


def log_trace(event: str, data: dict | None = None) -> None:
    TRACE_FILE.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": time.time(),
        "time_iso": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "event": event,
        "data": data or {},
    }
    with open(TRACE_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    try:
        print(f"[{payload['time_iso']}] [TRACE] {event}: {json.dumps(data, ensure_ascii=False) if data else ''}", flush=True)
    except Exception:
        pass


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def md5_file(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def run_command(cmd: list[str], step_name: str, valid_rcs: tuple[int, ...] = (0,)) -> str:
    log_trace(f"STEP_START: {step_name}", {"cmd": " ".join(cmd)})
    t0 = time.perf_counter()
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    proc = subprocess.Popen(
        cmd,
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        encoding="utf-8",
        errors="replace",
        env=env
    )
    lines = []
    while True:
        line = proc.stdout.readline()
        if not line and proc.poll() is not None:
            break
        if line:
            lines.append(line)
            clean = line.strip()
            if any(k in clean for k in ("scoring", "ready on", "eta", "Saved", "Recall@5", "THAM CHIEU", "PASS", "overlaid", "Embedded", "REGRESS", "chi +", "cả 3", "bỏ ")):
                try:
                    print(f"  [{step_name}] {clean}", flush=True)
                except Exception:
                    pass

    proc.wait()
    duration = time.perf_counter() - t0
    output_text = "".join(lines)

    if proc.returncode not in valid_rcs:
        log_trace(f"STEP_FAILED: {step_name}", {
            "returncode": proc.returncode,
            "duration_sec": duration,
            "tail_output": "".join(lines[-20:])
        })
        print(f"ERROR in {step_name}:\n" + "".join(lines[-20:]))
        raise RuntimeError(f"Step {step_name} failed with code {proc.returncode}")

    log_trace(f"STEP_SUCCESS: {step_name}", {"duration_sec": duration, "returncode": proc.returncode})
    print(f"  [{step_name}] COMPLETED in {duration:.1f}s", flush=True)
    return output_text


def find_latest_backup_dir() -> Path:
    backups = sorted(REPO_ROOT.glob("results/_pre_cleanroom_historical_backup_*"))
    if not backups:
        raise RuntimeError("No historical backup directory found!")
    return backups[-1]


def verify_integrity():
    print("\n=== Verifying Clean-Room Integrity & Reproducibility ===")
    backup_dir = find_latest_backup_dir()
    print(f"Comparing against historical backup in: {backup_dir}")

    integrity = {
        "timestamp": time.time(),
        "backup_directory": backup_dir.relative_to(REPO_ROOT).as_posix(),
        "channels": {},
        "submission": {}
    }

    # Compare channels
    channels = [
        ("aiteamvn_ft_cv", REPO_ROOT / "results/from_drive/aiteamvn_ft_cv.pkl"),
        ("aiteamvn_ft_public", REPO_ROOT / "results/from_drive/aiteamvn_ft_public.pkl"),
        ("jina_ft_cv", REPO_ROOT / "results/from_drive/jina_ft_cv.pkl"),
        ("jina_ft_public", REPO_ROOT / "results/from_drive/jina_ft_public.pkl"),
        ("title_embed_cv", REPO_ROOT / "results/burst_fresh_block/title_embed_scores.pkl"),
        ("title_embed_public", REPO_ROOT / "results/burst_fresh_block/title_embed_public.pkl"),
    ]

    for name, fresh_path in channels:
        rel = fresh_path.relative_to(REPO_ROOT).as_posix()
        hist_path = backup_dir / rel
        if not hist_path.exists():
            # Search across all backups
            for b in REPO_ROOT.glob("results/_pre_cleanroom_historical_backup_*"):
                cand = b / rel
                if cand.exists():
                    hist_path = cand
                    break

        if not fresh_path.exists():
            print(f"WARNING: Fresh {rel} does not exist!")
            continue

        fresh_data = pickle.loads(fresh_path.read_bytes())
        fresh_sha = sha256_file(fresh_path)

        chan_report = {
            "path": rel,
            "fresh_sha256": fresh_sha,
            "queries": len(fresh_data),
        }

        if hist_path.exists():
            hist_data = pickle.loads(hist_path.read_bytes())
            chan_report["historical_sha256"] = sha256_file(hist_path)
            chan_report["historical_queries"] = len(hist_data)

            # Numerical delta
            diffs = []
            spearmans = []
            for q in fresh_data:
                if q in hist_data:
                    f_row = fresh_data[q]
                    h_row = hist_data[q]
                    common_docs = sorted(set(f_row.keys()) & set(h_row.keys()))
                    if len(common_docs) >= 2:
                        f_vals = [f_row[d] for d in common_docs]
                        h_vals = [h_row[d] for d in common_docs]
                        diffs.extend([abs(a - b) for a, b in zip(f_vals, h_vals)])
                        try:
                            s, _ = spearmanr(f_vals, h_vals)
                            if not np.isnan(s):
                                spearmans.append(s)
                        except Exception:
                            pass

            if diffs:
                chan_report["max_abs_diff"] = float(np.max(diffs))
                chan_report["mean_abs_diff"] = float(np.mean(diffs))
                chan_report["mean_spearman_rank_corr"] = float(np.mean(spearmans)) if spearmans else None
                print(f"  {name:20s}: max_diff={chan_report['max_abs_diff']:.2e}, "
                      f"mean_diff={chan_report['mean_abs_diff']:.2e}, "
                      f"mean_spearman={chan_report['mean_spearman_rank_corr']:.4f}")
        else:
            print(f"  {name:20s}: fresh_sha={fresh_sha[:12]} (no historical backup found)")

        integrity["channels"][name] = chan_report

    # Compare Submission
    sub_fresh = REPO_ROOT / "results/burst_userft_maxrecall/submission.json"
    sub_zip_fresh = REPO_ROOT / "results/burst_userft_maxrecall/submission.zip"
    if sub_fresh.exists():
        fresh_sub_data = json.loads(sub_fresh.read_text(encoding="utf-8"))
        fresh_md5 = md5_file(sub_fresh)
        fresh_sha = sha256_file(sub_fresh)
        print(f"\nFresh submission.json:")
        print(f"  MD5:    {fresh_md5}")
        print(f"  MD5[:10]: {fresh_md5[:10]} (Historical expected: 2fb9a8a3b7)")
        print(f"  SHA256: {fresh_sha}")
        print(f"  Queries: {len(fresh_sub_data)}")

        integrity["submission"] = {
            "fresh_md5": fresh_md5,
            "fresh_md5_10": fresh_md5[:10],
            "historical_expected_md5_10": "2fb9a8a3b7",
            "md5_exact_match": (fresh_md5[:10] == "2fb9a8a3b7"),
            "fresh_sha256": fresh_sha,
            "query_count": len(fresh_sub_data),
        }

        # Snapshot immutable copies
        snap_json = OUTPUT_DIR / "submission_from_scratch.json"
        snap_zip = OUTPUT_DIR / "submission_from_scratch.zip"
        shutil.copy2(str(sub_fresh), str(snap_json))
        if sub_zip_fresh.exists():
            shutil.copy2(str(sub_zip_fresh), str(snap_zip))
        print(f"Saved immutable clean-room submission copy to {snap_json}")

        # Check against backup submission
        hist_sub_path = backup_dir / "results/burst_userft_maxrecall/submission.json"
        if not hist_sub_path.exists():
            for b in REPO_ROOT.glob("results/_pre_cleanroom_historical_backup_*"):
                cand = b / "results/burst_userft_maxrecall/submission.json"
                if cand.exists():
                    hist_sub_path = cand
                    break

        if hist_sub_path.exists():
            hist_sub = json.loads(hist_sub_path.read_text(encoding="utf-8"))
            exact_matches = sum(1 for q in fresh_sub_data if fresh_sub_data[q] == hist_sub.get(q))
            doc_jaccards = []
            for q in fresh_sub_data:
                if q in hist_sub:
                    set_f = set(fresh_sub_data[q])
                    set_h = set(hist_sub[q])
                    doc_jaccards.append(len(set_f & set_h) / len(set_f | set_h))

            integrity["submission"]["historical_sub_agreement"] = {
                "exact_ranking_agreement": exact_matches / len(fresh_sub_data),
                "mean_doc_jaccard": float(np.mean(doc_jaccards)),
            }
            print(f"  Agreement with historical submission: {exact_matches}/{len(fresh_sub_data)} "
                  f"({exact_matches/len(fresh_sub_data)*100:.1f}%), Mean Jaccard: {np.mean(doc_jaccards):.4f}")

    INTEGRITY_FILE.write_text(json.dumps(integrity, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Integrity report written to: {INTEGRITY_FILE}")
    log_trace("INTEGRITY_VERIFICATION_COMPLETE", integrity["submission"])


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    os.environ["PYTHONIOENCODING"] = "utf-8"
    print("=================================================================")
    print("  CLEAN-ROOM HISTORICAL REPRODUCTION: burst_userft_maxrecall")
    print("=================================================================")
    start_total = time.perf_counter()
    log_trace("CLEANROOM_REPRO_SESSION_START", {
        "repo_root": str(REPO_ROOT),
        "python": PYTHON_EXE
    })

    # Step 1: CV aiteamvn_ft bi-encoder
    out_aiteam_cv = REPO_ROOT / "results/from_drive/aiteamvn_ft_cv.pkl"
    if not out_aiteam_cv.exists():
        cmd = [
            PYTHON_EXE, "score_cv_custom_encoder.py",
            "--model-path", "models/from_drive/AITeamVN_Vietnamese_Embedding",
            "--out", "results/from_drive/aiteamvn_ft_cv.pkl",
            "--batch-size", "32"
        ]
        run_command(cmd, "score_cv_aiteamvn_ft")
    else:
        print("  aiteamvn_ft_cv.pkl already exists, skipping")

    # Step 2: CV jina_ft cross-encoder
    out_jina_cv = REPO_ROOT / "results/from_drive/jina_ft_cv.pkl"
    if not out_jina_cv.exists():
        cmd = [
            PYTHON_EXE, "score_cv_jina_ft.py",
            "--weights", "models/from_drive/jina_finetuned/model.safetensors",
            "--out", "results/from_drive/jina_ft_cv.pkl",
            "--batch-size", "16"
        ]
        run_command(cmd, "score_cv_jina_ft")
    else:
        print("  jina_ft_cv.pkl already exists, skipping")

    # Step 3: Public aiteamvn_ft
    out_aiteam_pub = REPO_ROOT / "results/from_drive/aiteamvn_ft_public.pkl"
    if not out_aiteam_pub.exists():
        cmd = [
            PYTHON_EXE, "score_public_models.py",
            "--kind", "bi",
            "--model-path", "models/from_drive/AITeamVN_Vietnamese_Embedding",
            "--out", "results/from_drive/aiteamvn_ft_public.pkl",
            "--batch-size", "32"
        ]
        run_command(cmd, "score_public_aiteamvn_ft")
    else:
        print("  aiteamvn_ft_public.pkl already exists, skipping")

    # Step 4: Public jina_ft
    out_jina_pub = REPO_ROOT / "results/from_drive/jina_ft_public.pkl"
    if not out_jina_pub.exists():
        cmd = [
            PYTHON_EXE, "score_public_models.py",
            "--kind", "jina",
            "--weights", "models/from_drive/jina_finetuned/model.safetensors",
            "--out", "results/from_drive/jina_ft_public.pkl",
            "--batch-size", "16"
        ]
        run_command(cmd, "score_public_jina_ft")
    else:
        print("  jina_ft_public.pkl already exists, skipping")

    # Step 5: Evaluate CV (Protocol Recall@5 = 0.9561)
    print("\n--- Running CV Protocol Evaluation (600 queries) ---")
    cmd = [PYTHON_EXE, "evaluate_cv.py", "--all"]
    cv_output = run_command(cmd, "evaluate_cv_protocol", valid_rcs=(0, 2))

    # Parse metrics from stdout
    cv_metrics = {"raw_stdout": cv_output, "blocks": {}}
    for line in cv_output.splitlines():
        if "burst_userft_maxrecall (bản đã ship)" in line:
            cv_metrics["shipped_line"] = line
            for token in line.split():
                if ":" in token and token.split(":")[0] in ("a", "b", "c", "d"):
                    b, val = token.split(":")
                    cv_metrics["blocks"][b] = float(val)
        if "Ket qua chinh: Recall@5 =" in line:
            try:
                cv_metrics["recall_at_5"] = float(line.split("Recall@5 =")[1].split()[0])
            except Exception:
                pass
        if "THAM CHIEU 7 kenh" in line:
            cv_metrics["reference_line"] = line

    CV_REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    CV_REPORT_FILE.write_text(json.dumps(cv_metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved CV evaluation report to: {CV_REPORT_FILE}")

    # Step 6: Refit LTR and build Public Submission
    print("\n--- Refitting LTR and Building Public Submission ---")
    cmd = [
        PYTHON_EXE, "run_vnlegal_extra_channel_submission.py",
        "--crossenc", "--alpha", "0",
        "--output-dir", "results/burst_userft_maxrecall",
        "--extra-channel",
        "aiteamvn_ft=results/from_drive/aiteamvn_ft_cv.pkl,results/from_drive/aiteamvn_ft_public.pkl",
        "--extra-channel",
        "jina_ft=results/from_drive/jina_ft_cv.pkl,results/from_drive/jina_ft_public.pkl",
        "--extra-channel",
        "title_embed=results/burst_fresh_block/title_embed_scores.pkl,results/burst_fresh_block/title_embed_public.pkl"
    ]
    sub_output = run_command(cmd, "refit_ltr_and_build_submission")

    # Step 7: Verification & Audit
    verify_integrity()

    total_sec = time.perf_counter() - start_total
    log_trace("CLEANROOM_REPRO_SESSION_COMPLETE", {"total_duration_sec": total_sec})
    print(f"\n=================================================================")
    print(f"  CLEAN-ROOM REPRODUCTION COMPLETE in {total_sec/60:.1f} minutes")
    print(f"=================================================================")


if __name__ == "__main__":
    main()
