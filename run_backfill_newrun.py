#!/usr/bin/env python3
"""
NEWRUN Backfill Orchestrator

Runs the full daily pipeline from the earliest available headlines date
through the latest, using ONLY existing headlines, and saving outputs
under a separate NEWRUN storage namespace.

Outputs:
- Predictions/Evaluations: timestamped_storage_NEWRUN/<date>_*.json
- Training JSON: training/gspo_training_<date>.json
- Checkpoints: training/checkpoints/gspo_NEWRUN/final_model
- Logs: training/logs/NEWRUN_<date>_*.log
"""

import os
import sys
import json
import glob
import argparse
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
BASE_STORAGE = REPO_ROOT / "timestamped_storage"
NEWRUN_STORAGE = REPO_ROOT / "timestamped_storage_NEWRUN"
LOG_DIR = REPO_ROOT / "training" / "logs"
CKPT_DIR = REPO_ROOT / "training" / "checkpoints" / "gspo_NEWRUN"


def ensure_dirs(run_suffix: str = "NEWRUN"):
    os.environ["VARRO_RUN_DIR_SUFFIX"] = os.environ.get("VARRO_RUN_DIR_SUFFIX", run_suffix)
    NEWRUN_STORAGE.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)


def list_headline_dates() -> list[str]:
    dates = []
    for p in sorted(BASE_STORAGE.glob("*_headlines.json")):
        try:
            dates.append(p.name.split("_")[0])
        except Exception:
            continue
    return dates


def run_and_log(cmd: list[str], log_path: Path) -> int:
    """Run a command, stream output to stdout and write to a log file."""
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
                # Best-effort logging; continue streaming to stdout
                pass
        proc.stdout.close()
        return proc.wait()


def file_exists(path: Path) -> bool:
    return path.exists()


def filter_dates(all_dates: list[str], start_date: str | None, end_date: str | None) -> list[str]:
    if start_date:
        all_dates = [d for d in all_dates if d >= start_date]
    if end_date:
        all_dates = [d for d in all_dates if d <= end_date]
    return all_dates


def main():
    parser = argparse.ArgumentParser(description="Run NEWRUN backfill or continuation with controls")
    parser.add_argument("--start_date", type=str, default=None, help="Start date (YYYYMMDD) inclusive")
    parser.add_argument("--end_date", type=str, default=None, help="End date (YYYYMMDD) inclusive")
    parser.add_argument("--force_predictions", action="store_true", help="Regenerate predictions even if present")
    parser.add_argument("--force_evaluations", action="store_true", help="Re-evaluate even if present")
    parser.add_argument("--force_training", action="store_true", help="Retrain even if training JSON or checkpoint exists")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed for generation/eval")
    parser.add_argument("--resume_from_last_model", action="store_true", help="Resume using latest NEWRUN final_model if present")
    parser.add_argument("--run_suffix", type=str, default="NEWRUN", help="Suffix for timestamped_storage_<suffix> namespace")
    parser.add_argument("--override_headlines_date", type=str, default=None, help="Use this date's headlines when evaluating prediction_date")

    args = parser.parse_args()

    ensure_dirs(args.run_suffix)

    dates = list_headline_dates()
    if not dates:
        print("No headline dates found in timestamped_storage.")
        sys.exit(1)

    dates = filter_dates(dates, args.start_date, args.end_date)
    if not dates:
        print("No dates to process after filtering.")
        sys.exit(0)

    # Rolling pointer to latest NEWRUN model
    trained_model = None
    if args.resume_from_last_model:
        latest = CKPT_DIR / "final_model"
        if latest.exists():
            trained_model = str(latest)

    for i, d in enumerate(dates):
        # Morning: generate predictions using existing headlines
        pred_path = NEWRUN_STORAGE / f"{d}_predictions.json"
        if args.force_predictions and pred_path.exists():
            pred_path.unlink(missing_ok=True)
        if args.force_predictions or not file_exists(pred_path):
            cmd = [sys.executable, "run_daily_pipeline.py", "--mode", "morning", "--date", d, "--seed", str(args.seed)]
            if trained_model:
                cmd += ["--trained-model", trained_model]
            rc = run_and_log(cmd, LOG_DIR / f"{args.run_suffix}_{d}_morning.log")
            if rc != 0:
                print(f"Morning failed for {d} (rc={rc}). Aborting.")
                sys.exit(rc)

        # Run evening + night + training for all but the last date.
        # If an override headlines date is provided, also process the last date using the override.
        if (i < len(dates) - 1) or (args.override_headlines_date is not None):
            eval_path = NEWRUN_STORAGE / f"{d}_evaluations.json"
            if args.force_evaluations and eval_path.exists():
                eval_path.unlink(missing_ok=True)
            if args.force_evaluations or not file_exists(eval_path):
                evening_cmd = [sys.executable, "run_daily_pipeline.py", "--mode", "evening", "--date", d, "--seed", str(args.seed)]
                if args.override_headlines_date:
                    evening_cmd += ["--override_headlines_date", args.override_headlines_date]
                rc = run_and_log(evening_cmd, LOG_DIR / f"{args.run_suffix}_{d}_evening.log")
                if rc != 0:
                    print(f"Evening failed for {d} (rc={rc}). Aborting.")
                    sys.exit(rc)

            training_json = REPO_ROOT / f"training/gspo_training_{d}.json"
            if args.force_training and training_json.exists():
                training_json.unlink(missing_ok=True)
            if args.force_training or not file_exists(training_json):
                rc = run_and_log([sys.executable, "run_daily_pipeline.py", "--mode", "night", "--date", d],
                                 LOG_DIR / f"{args.run_suffix}_{d}_night_prep.log")
                if rc != 0:
                    print(f"Night prep failed for {d} (rc={rc}). Aborting.")
                    sys.exit(rc)

            # Train (response-only) and save under NEWRUN checkpoints
            train_cmd = [
                sys.executable, "run_gspo_training.py",
                "--training_data", str(training_json),
                "--epochs", "1",
                "--save_every", "50",
                "--response_only_loss", "1",
                "--ema_baseline", "0",
                "--checkpoint_dir", str(CKPT_DIR),
            ]
            if trained_model:
                train_cmd += ["--load_checkpoint", trained_model]

            rc = run_and_log(train_cmd, LOG_DIR / f"{args.run_suffix}_{d}_train_resp_only.log")
            if rc != 0:
                print(f"Training failed for {d} (rc={rc}). Aborting.")
                sys.exit(rc)

            # Update pointer for next day
            trained_model = str(CKPT_DIR / "final_model")

    print(f"{args.run_suffix} run completed.")


if __name__ == "__main__":
    main()


