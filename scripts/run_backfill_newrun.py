#!/usr/bin/env python3
"""
Run-suffixed Backfill Orchestrator

Runs the full daily pipeline from the earliest available headlines date
through the latest, using ONLY existing headlines, and saving outputs
under a separate per-run storage namespace.

Outputs:
- Predictions/Evaluations: timestamped_storage_<RUN_SUFFIX>/<date>_*.json
- Training JSON: training/gspo_training_<date>.json
- Checkpoints: training/checkpoints/gspo_<RUN_SUFFIX>/final_model
- Logs: training/logs/<RUN_SUFFIX>_<date>_*.log
"""

import os
import sys
import json
import glob
import argparse
import subprocess
import shutil
from statistics import median
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
BASE_STORAGE = REPO_ROOT / "timestamped_storage"
def get_storage_dir(run_suffix: str) -> Path:
    suffix = (run_suffix or "NEWCOMPOSITERUN").strip()
    return REPO_ROOT / f"timestamped_storage_{suffix}"
LOG_DIR = REPO_ROOT / "training" / "logs"

def get_checkpoint_dir(run_suffix: str) -> Path:
    """Return the per-run checkpoint directory (e.g., gspo_COMPOSITERUN)."""
    normalized = (run_suffix or "NEWCOMPOSITERUN").strip()
    return REPO_ROOT / "training" / "checkpoints" / f"gspo_{normalized}"


def ensure_dirs(run_suffix: str = "NEWCOMPOSITERUN") -> Path:
    # Force storage namespace to match provided run suffix
    os.environ["VARRO_RUN_DIR_SUFFIX"] = run_suffix
    get_storage_dir(run_suffix).mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_dir = get_checkpoint_dir(run_suffix)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    return ckpt_dir


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
    parser = argparse.ArgumentParser(description="Run backfill or continuation with controls (per run-suffix)")
    parser.add_argument("--start_date", type=str, default=None, help="Start date (YYYYMMDD) inclusive")
    parser.add_argument("--end_date", type=str, default=None, help="End date (YYYYMMDD) inclusive")
    parser.add_argument("--force_predictions", action="store_true", help="Regenerate predictions even if present")
    parser.add_argument("--force_evaluations", action="store_true", help="Re-evaluate even if present")
    parser.add_argument("--force_training", action="store_true", help="Retrain even if training JSON or checkpoint exists")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed for generation/eval")
    parser.add_argument("--resume_from_last_model", action="store_true", help="Resume using latest <run_suffix> final_model if present")
    parser.add_argument("--run_suffix", type=str, default="NEWCOMPOSITERUN", help="Suffix for timestamped_storage_<suffix> namespace")
    parser.add_argument("--override_headlines_date", type=str, default=None, help="Use this date's headlines when evaluating prediction_date")
    parser.add_argument("--sampler_profile", type=str, default="default", choices=["loose", "default", "tight"], help="Sampler profile for rollout generation entropy")
    parser.add_argument("--auto_profile", action="store_true", help="Automatically choose sampler profile based on prior-day metrics")
    parser.add_argument("--horizon", type=str, default=None, choices=["next_day", "next_month", "next_year", "next_2days", "next_3days"], help="Optional prediction horizon directive to forward to morning step")
    parser.add_argument("--horizons", type=str, default=None, help="Comma-separated list of horizons to generate (e.g., next_day,next_2days,next_3days)")
    parser.add_argument("--evaluate_due", action="store_true", help="Evening step evaluates all rollouts maturing on the given date")
    # Auto schedule for format weight (Q) and learning rate (LR)
    parser.add_argument("--auto_q_lr", action="store_true", help="Automatically anneal format weight (Q) and learning rate across days")
    parser.add_argument("--q_start", type=float, default=0.10, help="Starting format weight Q when --auto_q_lr is set")
    parser.add_argument("--q_floor", type=float, default=0.05, help="Minimum format weight Q when --auto_q_lr is set")
    parser.add_argument("--q_step", type=float, default=0.02, help="Daily decrement for Q when --auto_q_lr is set")
    parser.add_argument("--lr_start", type=float, default=1e-6, help="Starting learning rate when --auto_q_lr is set")
    parser.add_argument("--lr_floor", type=float, default=5e-7, help="Minimum learning rate when --auto_q_lr is set")
    parser.add_argument("--lr_decay", type=float, default=0.90, help="Daily multiplicative decay for LR when --auto_q_lr is set")
    parser.add_argument("--low_llm_alpha", type=float, default=0.20, help="Low-bucket LLM absolute weight once picker improves (when --auto_q_lr)")
    parser.add_argument("--llm_picks_threshold", type=int, default=4, help="Median llm_picks threshold to enable low-bucket LLM weight (when --auto_q_lr)")

    args = parser.parse_args()

    ckpt_dir = ensure_dirs(args.run_suffix)
    storage_dir = get_storage_dir(args.run_suffix)

    dates = list_headline_dates()
    if not dates:
        print("No headline dates found in timestamped_storage.")
        sys.exit(1)

    dates = filter_dates(dates, args.start_date, args.end_date)
    if not dates:
        print("No dates to process after filtering.")
        sys.exit(0)

    # Rolling pointer to latest <run_suffix> model
    trained_model = None
    if args.resume_from_last_model:
        latest = ckpt_dir / "final_model"
        if latest.exists():
            trained_model = str(latest)

    # Helper: choose sampler profile based on previous day's metrics (paragraph mode)
    def choose_profile(prev_date: str, default_profile: str = "default") -> str:
        pred_file = storage_dir / f"{prev_date}_predictions.json"
        eval_file = storage_dir / f"{prev_date}_evaluations.json"
        avg_quality = None  # Q
        avg_eval = None     # E
        try:
            if pred_file.exists():
                data = json.load(open(pred_file))
                total = 0
                s = 0.0
                for item in data.get('predictions', []):
                    for r in item.get('rollouts', []):
                        val = r.get('immediate_reward')
                        if isinstance(val, (int, float)):
                            s += float(val)
                            total += 1
                if total > 0:
                    avg_quality = s / total
        except Exception:
            pass
        try:
            if eval_file.exists():
                ed = json.load(open(eval_file))
                summary = ed.get('summary', {})
                aos = summary.get('avg_outcome_score')
                if isinstance(aos, (int, float)):
                    avg_eval = float(aos)
        except Exception:
            pass
        Q = avg_quality if avg_quality is not None else 0.0
        E = avg_eval if avg_eval is not None else 0.0
        if Q < 0.35:
            return 'default'
        if Q > 0.55 and E > 0.25:
            return 'loose'
        return default_profile

    # Helpers for auto Q/LR schedule (metric-responsive)
    def _median_llm_picks(prev_date: str) -> int:
        try:
            td_path = REPO_ROOT / f"training/gspo_training_{prev_date}.json"
            if not td_path.exists():
                return 0
            data = json.load(open(td_path))
            picks = []
            for ex in data.get('training_data', []):
                lp = ex.get('llm_picks')
                if isinstance(lp, int):
                    picks.append(lp)
            return int(median(picks)) if picks else 0
        except Exception:
            return 0

    def _avg_composite(storage: Path, date: str) -> float:
        try:
            p = storage / f"{date}_evaluations.json"
            if not p.exists():
                return float('nan')
            data = json.load(open(p))
            s = data.get('summary', {})
            v = s.get('avg_composite_reward')
            return float(v) if isinstance(v, (int, float)) else float('nan')
        except Exception:
            return float('nan')

    q_curr = args.q_start if args.auto_q_lr else None
    lr_curr = args.lr_start if args.auto_q_lr else None
    no_improve_days = 0

    def _set_bucket_envs(q: float, low_llm: float | None = None):
        """Export per-bucket absolute weights using current Q and desired low-bucket LLM share.
        If low_llm is None, default to 0.0 (pure semantic) until picker improves.
        Mid/high keep their L shares; SEM is adjusted to 1-Q-L.
        """
        low_llm = 0.0 if low_llm is None else max(0.0, min(1.0, low_llm))
        # Low bucket: default L=low_llm, S=1-Q-L
        low_sem = max(0.0, 1.0 - q - low_llm)
        os.environ["VARRO_BUCKET_LOW_LLM"] = f"{low_llm:.6f}"
        os.environ["VARRO_BUCKET_LOW_SEM"] = f"{low_sem:.6f}"
        os.environ["VARRO_BUCKET_LOW_Q"] = f"{q:.6f}"

        # Mid bucket: L fixed at 0.30
        mid_llm = 0.30
        mid_sem = max(0.0, 1.0 - q - mid_llm)
        os.environ["VARRO_BUCKET_MID_LLM"] = f"{mid_llm:.6f}"
        os.environ["VARRO_BUCKET_MID_SEM"] = f"{mid_sem:.6f}"
        os.environ["VARRO_BUCKET_MID_Q"] = f"{q:.6f}"

        # High bucket: L fixed at 0.475
        high_llm = 0.475
        high_sem = max(0.0, 1.0 - q - high_llm)
        os.environ["VARRO_BUCKET_HIGH_LLM"] = f"{high_llm:.6f}"
        os.environ["VARRO_BUCKET_HIGH_SEM"] = f"{high_sem:.6f}"
        os.environ["VARRO_BUCKET_HIGH_Q"] = f"{q:.6f}"

    for i, d in enumerate(dates):
        # Determine scheduled Q/LR and whether to enable low-bucket LLM weight
        current_q = q_curr if args.auto_q_lr else None
        current_lr = lr_curr if args.auto_q_lr else None
        low_llm = None
        if args.auto_q_lr:
            prev_picks_med = _median_llm_picks(dates[i-1]) if i > 0 else 0
            if prev_picks_med >= args.llm_picks_threshold:
                low_llm = args.low_llm_alpha
            _set_bucket_envs(current_q, low_llm)
            # Informational print
            print(f"[auto_q_lr] {d}: Q={current_q:.3f}, LR={current_lr:.2e}, low_llm={0.0 if low_llm is None else low_llm:.3f}")
        # Select sampler profile
        profile = args.sampler_profile
        if args.auto_profile:
            # Use previous day if available; else default
            from datetime import datetime, timedelta
            try:
                prev = (datetime.strptime(d, '%Y%m%d') - timedelta(days=1)).strftime('%Y%m%d')
                profile = choose_profile(prev, default_profile=args.sampler_profile)
            except Exception:
                profile = args.sampler_profile
        # Morning: generate predictions using existing headlines
        pred_path = storage_dir / f"{d}_predictions.json"
        if args.force_predictions and pred_path.exists():
            pred_path.unlink(missing_ok=True)
        if args.force_predictions or not file_exists(pred_path):
            cmd = [sys.executable, "run_daily_pipeline.py", "--mode", "morning", "--date", d, "--seed", str(args.seed), "--sampler_profile", profile]
            if args.horizons:
                cmd += ["--horizons", args.horizons]
            elif args.horizon:
                cmd += ["--horizon", args.horizon]
            if trained_model:
                cmd += ["--trained-model", trained_model]
            rc = run_and_log(cmd, LOG_DIR / f"{args.run_suffix}_{d}_morning.log")
            if rc != 0:
                print(f"Morning failed for {d} (rc={rc}). Aborting.")
                sys.exit(rc)

        # Run evening + night + training for all but the last date.
        # If an override headlines date is provided, also process the last date using the override.
        if (i < len(dates) - 1) or (args.override_headlines_date is not None):
            eval_path = storage_dir / f"{d}_evaluations.json"
            if args.force_evaluations and eval_path.exists():
                eval_path.unlink(missing_ok=True)
            if args.force_evaluations or not file_exists(eval_path):
                evening_cmd = [sys.executable, "run_daily_pipeline.py", "--mode", "evening", "--date", d, "--seed", str(args.seed)]
                if args.evaluate_due:
                    evening_cmd += ["--evaluate_due"]
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
                "--checkpoint_dir", str(ckpt_dir),
            ]
            if args.auto_q_lr and current_lr is not None:
                train_cmd += ["--learning_rate", f"{current_lr}"]
            if trained_model:
                train_cmd += ["--load_checkpoint", trained_model]

            rc = run_and_log(train_cmd, LOG_DIR / f"{args.run_suffix}_{d}_train_resp_only.log")
            if rc != 0:
                print(f"Training failed for {d} (rc={rc}). Aborting.")
                sys.exit(rc)

            # Update pointer for next day
            trained_model = str(ckpt_dir / "final_model")

            # Snapshot the final_model into a dated directory for this prediction date
            try:
                src = ckpt_dir / "final_model"
                dst = ckpt_dir / f"final_model_{d}"
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)
            except Exception as e:
                print(f"Warning: failed to snapshot final_model to final_model_{d}: {e}")

            # After training: update Q/LR for next day if auto schedule is enabled
            if args.auto_q_lr:
                # Check composite movement vs previous day to drive decay
                prev_date = dates[i]
                prev2_date = dates[i-1] if i > 0 else None
                comp_prev = _avg_composite(storage_dir, prev_date)
                comp_prev2 = _avg_composite(storage_dir, prev2_date) if prev2_date else float('nan')
                moved = False
                if not (comp_prev != comp_prev):  # not NaN
                    if not (comp_prev2 != comp_prev2):
                        if comp_prev - comp_prev2 >= 0.01:
                            moved = True
                    else:
                        # If no baseline, treat first measured day as moved
                        moved = True
                # Decide Q decay
                if moved and q_curr > args.q_floor:
                    q_curr = max(args.q_floor, q_curr - args.q_step)
                    no_improve_days = 0
                    print(f"[auto_q_lr] Decayed Q due to composite improvement to {q_curr:.3f} (comp {comp_prev2:.3f} -> {comp_prev:.3f})")
                else:
                    no_improve_days += 1
                # Decide LR decay
                if no_improve_days >= 2 and lr_curr > args.lr_floor:
                    lr_curr = max(args.lr_floor, lr_curr * args.lr_decay)
                    print(f"[auto_q_lr] Decayed LR after {no_improve_days} non-improving day(s) to {lr_curr:.2e}")

    print(f"{args.run_suffix} run completed.")


if __name__ == "__main__":
    main()

