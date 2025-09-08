#!/usr/bin/env python3
"""
Re-run SEMANTICRUN_TIGHT_Q25 with article context:
 - Cleans articles per date (if raw present)
 - Morning: generate paragraph predictions with `sampler_profile=tight`, using article excerpts in prompts
 - Evening: evaluate (group ranking with fallback)
 - Night: prepare GSPO training JSON
 - Train GSPO and update per-run final_model

Defaults mimic the prior SEMANTICRUN_TIGHT_Q25 setup but attach article context.
This writes to `timestamped_storage_SEMANTICRUN_TIGHT_Q25_ARTICLES/` to avoid
overwriting the original run.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = REPO_ROOT / "training" / "logs"
RUN_SUFFIX = "SEMANTICRUN_TIGHT_Q25_ARTICLES"


def _storage_dir(suffix: str) -> Path:
    return REPO_ROOT / f"timestamped_storage_{suffix}"


def _base_storage() -> Path:
    return REPO_ROOT / "timestamped_storage"


def _ckpt_dir(suffix: str) -> Path:
    return REPO_ROOT / "training" / "checkpoints" / f"gspo_{suffix}"


def _list_headline_dates() -> list[str]:
    dates: list[str] = []
    for p in sorted(_base_storage().glob("*_headlines.json")):
        try:
            dates.append(p.name.split("_")[0])
        except Exception:
            continue
    return dates


def _run_and_log(cmd: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as f:
        proc = subprocess.Popen(
            cmd,
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            try:
                f.write(line)
            except Exception:
                pass
        proc.stdout.close()
        return proc.wait()


def main() -> int:
    ap = argparse.ArgumentParser(description="Re-run SEMANTICRUN_TIGHT_Q25 with article context")
    ap.add_argument("--start_date", type=str, default=None, help="YYYYMMDD inclusive")
    ap.add_argument("--end_date", type=str, default=None, help="YYYYMMDD inclusive")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--resume_from_last_model", action="store_true")
    args = ap.parse_args()

    # Ensure run namespace
    os.environ["VARRO_RUN_DIR_SUFFIX"] = RUN_SUFFIX
    _storage_dir(RUN_SUFFIX).mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR = _ckpt_dir(RUN_SUFFIX)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    # Freeze bucket weights to SEMANTICRUN_TIGHT_Q25 defaults: LLM=0, Q≈0.25, SEM≈0.75
    q = 0.25
    for bucket in ("LOW", "MID", "HIGH"):
        os.environ[f"VARRO_BUCKET_{bucket}_LLM"] = "0.0"
        os.environ[f"VARRO_BUCKET_{bucket}_Q"] = f"{q:.6f}"
        os.environ[f"VARRO_BUCKET_{bucket}_SEM"] = f"{(1.0-q):.6f}"

    dates = _list_headline_dates()
    if args.start_date:
        dates = [d for d in dates if d >= args.start_date]
    if args.end_date:
        dates = [d for d in dates if d <= args.end_date]
    if not dates:
        print("No dates found to process.")
        return 1

    # Optionally resume from last per-run model
    trained_model = None
    if args.resume_from_last_model:
        latest = CKPT_DIR / "final_model"
        if latest.exists():
            trained_model = str(latest)

    for i, d in enumerate(dates):
        # 0) Clean articles for this date (if raw exists)
        rc = _run_and_log([sys.executable, str(REPO_ROOT / "scripts" / "clean_articles.py"), "--date", d], LOG_DIR / f"{RUN_SUFFIX}_{d}_clean_articles.log")
        if rc != 0:
            print(f"Warning: cleaning articles failed rc={rc} for {d}; continuing")

        # 1) Morning predictions (paragraph + article context, tight sampler)
        cmd = [
            sys.executable, "run_daily_pipeline.py",
            "--mode", "morning",
            "--date", d,
            "--seed", str(args.seed),
            "--sampler_profile", "tight",
            "--output_format", "paragraph",
        ]
        if trained_model:
            cmd += ["--trained-model", trained_model]
        rc = _run_and_log(cmd, LOG_DIR / f"{RUN_SUFFIX}_{d}_morning.log")
        if rc != 0:
            print(f"Morning failed for {d} (rc={rc})")
            return rc

        # 2) Evening (evaluate)
        rc = _run_and_log([sys.executable, "run_daily_pipeline.py", "--mode", "evening", "--date", d, "--seed", str(args.seed)], LOG_DIR / f"{RUN_SUFFIX}_{d}_evening.log")
        if rc != 0:
            print(f"Evening failed for {d} (rc={rc})")
            return rc

        # 3) Night prep (training JSON)
        rc = _run_and_log([sys.executable, "run_daily_pipeline.py", "--mode", "night", "--date", d], LOG_DIR / f"{RUN_SUFFIX}_{d}_night_prep.log")
        if rc != 0:
            print(f"Night prep failed for {d} (rc={rc})")
            return rc

        # 4) Train (response-only) into per-run checkpoint dir
        train_cmd = [
            sys.executable, "run_gspo_training.py",
            "--training_data", f"training/gspo_training_{d}.json",
            "--epochs", "1",
            "--save_every", "50",
            "--response_only_loss", "1",
            "--ema_baseline", "0",
            "--checkpoint_dir", str(CKPT_DIR),
        ]
        if trained_model:
            train_cmd += ["--load_checkpoint", trained_model]
        rc = _run_and_log(train_cmd, LOG_DIR / f"{RUN_SUFFIX}_{d}_train_resp_only.log")
        if rc != 0:
            print(f"Training failed for {d} (rc={rc})")
            return rc

        # Update latest for next loop
        trained_model = str(CKPT_DIR / "final_model")
        # Snapshot into dated dir
        try:
            dst = CKPT_DIR / f"final_model_{d}"
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(CKPT_DIR / "final_model", dst)
        except Exception as e:
            print(f"Warning: snapshot failed for {d}: {e}")

    print(f"{RUN_SUFFIX} run completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

