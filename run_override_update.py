#!/usr/bin/env python3
"""
One-shot override update:
 - Evaluate predictions from --prediction_date using headlines from --headlines_date
 - Prepare training JSON and train GSPO (updates NEWRUN final_model)
 - Generate morning predictions for --headlines_date with the updated model

Usage example:
  python run_override_update.py \
    --prediction_date 20250808 \
    --headlines_date 20250810 \
    --resume_from_last_model \
    --run_suffix NEWRUN \
    --seed 1234
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
LOG_DIR = REPO_ROOT / "training" / "logs"
CKPT_DIR = REPO_ROOT / "training" / "checkpoints" / "gspo_NEWRUN"


def run_and_stream(cmd: list[str], log_path: Path) -> int:
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
    parser = argparse.ArgumentParser(description="Evaluate D with headlines from H, train, then generate H preds")
    parser.add_argument("--prediction_date", required=True, help="Prediction date (YYYYMMDD) to evaluate")
    parser.add_argument("--headlines_date", required=True, help="Headlines date (YYYYMMDD) to use for evaluation and to generate morning preds")
    parser.add_argument("--run_suffix", default="NEWRUN", help="Storage namespace suffix")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    parser.add_argument("--resume_from_last_model", action="store_true", help="Load latest NEWRUN final_model for training start and for predictions")

    args = parser.parse_args()

    os.environ["VARRO_RUN_DIR_SUFFIX"] = os.environ.get("VARRO_RUN_DIR_SUFFIX", args.run_suffix)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Evening evaluation for prediction_date using override headlines_date
    evening_cmd = [
        sys.executable, "run_daily_pipeline.py",
        "--mode", "evening",
        "--date", args.prediction_date,
        "--override_headlines_date", args.headlines_date,
        "--seed", str(args.seed),
    ]
    rc = run_and_stream(evening_cmd, LOG_DIR / f"{args.run_suffix}_{args.prediction_date}_evening_override_{args.headlines_date}.log")
    if rc != 0:
        print(f"Evening failed (rc={rc})")
        return rc

    # 2) Night prep (training JSON)
    night_cmd = [
        sys.executable, "run_daily_pipeline.py",
        "--mode", "night",
        "--date", args.prediction_date,
    ]
    rc = run_and_stream(night_cmd, LOG_DIR / f"{args.run_suffix}_{args.prediction_date}_night_prep.log")
    if rc != 0:
        print(f"Night prep failed (rc={rc})")
        return rc

    # 3) Train GSPO for prediction_date
    train_cmd = [
        sys.executable, "run_gspo_training.py",
        "--training_data", f"training/gspo_training_{args.prediction_date}.json",
        "--epochs", "1",
        "--save_every", "50",
        "--response_only_loss", "1",
        "--ema_baseline", "0",
        "--checkpoint_dir", str(CKPT_DIR),
    ]
    if args.resume_from_last_model:
        train_cmd += ["--load_checkpoint", str(CKPT_DIR / "final_model")]
    rc = run_and_stream(train_cmd, LOG_DIR / f"{args.run_suffix}_{args.prediction_date}_train_resp_only.log")
    if rc != 0:
        print(f"Training failed (rc={rc})")
        return rc

    # 4) Morning predictions for headlines_date using updated model
    morning_cmd = [
        sys.executable, "run_daily_pipeline.py",
        "--mode", "morning",
        "--date", args.headlines_date,
        "--seed", str(args.seed),
    ]
    if args.resume_from_last_model:
        morning_cmd += ["--trained-model", str(CKPT_DIR / "final_model")]
    rc = run_and_stream(morning_cmd, LOG_DIR / f"{args.run_suffix}_{args.headlines_date}_morning.log")
    if rc != 0:
        print(f"Morning failed (rc={rc})")
        return rc

    print("Override update completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


