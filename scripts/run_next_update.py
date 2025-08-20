#!/usr/bin/env python3
"""
Run the next-day update to bring the pipeline up to date:
- For yesterday: evening (evaluate with today's headlines) + night (prep training JSON) + training
- For today: morning predictions only

This wraps run_backfill_newrun.py so you get live streaming output and a simple one-command entrypoint.
"""

import argparse
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description="Run next-day pipeline update (yesterday..today)")
    parser.add_argument("--run_suffix", type=str, default="NEWCOMPOSITERUN", help="Storage namespace suffix")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    parser.add_argument("--resume_from_last_model", action="store_true", help="Resume from latest <run_suffix> final_model")
    parser.add_argument("--force_evaluations", action="store_true", help="Force re-evaluations for yesterday")
    parser.add_argument("--force_training", action="store_true", help="Force re-building training JSON + retrain for yesterday")
    parser.add_argument("--force_predictions", action="store_true", help="Force re-generating predictions (usually unnecessary)")
    args = parser.parse_args()

    today = datetime.now().strftime("%Y%m%d")
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")

    repo_root = Path(__file__).resolve().parent.parent
    cmd = [
        sys.executable,
        str(repo_root / "scripts" / "run_backfill_newrun.py"),
        "--start_date", yesterday,
        "--end_date", today,
        "--seed", str(args.seed),
        "--run_suffix", args.run_suffix,
    ]
    if args.resume_from_last_model:
        cmd.append("--resume_from_last_model")
    if args.force_evaluations:
        cmd.append("--force_evaluations")
    if args.force_training:
        cmd.append("--force_training")
    if args.force_predictions:
        cmd.append("--force_predictions")

    print("Running:", " ".join(cmd))
    proc = subprocess.Popen(cmd, cwd=repo_root)
    return proc.wait()


if __name__ == "__main__":
    sys.exit(main())

