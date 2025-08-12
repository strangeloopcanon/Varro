"""Generate figures for the paper from run logs and snapshot the inputs.

Outputs (saved under paper/figs/):
- reward_curve.png – average reward vs. training step (from training_state.json)
- kl_stability.png – loss proxy over steps (from training_state.json)
- pipeline_throughput.png – evaluated vs. dropped roll-outs per day (from timestamped storage)

Additionally, this script snapshots the source data into paper/data/ so the
paper folder is self-contained for archiving/printing:
- paper/data/training_state.json – copy of the training state used for curves
- paper/data/throughput.csv – per-day counts used for the throughput plot
"""

from __future__ import annotations

import re
import shutil
import json
import pathlib
from typing import List, Dict, Any

import matplotlib.pyplot as plt

# Matplotlib style: colour-blind friendly and larger fonts
plt.rcParams.update({
    "font.size": 12,
    "axes.prop_cycle": plt.cycler(
        "color", ["#377eb8", "#e41a1c", "#4daf4a", "#984ea3", "#ff7f00"]
    ),
    "figure.dpi": 120,
})

ROOT = pathlib.Path(__file__).resolve().parents[2]
FIG_DIR = ROOT / "paper" / "figs"
FIG_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = ROOT / "paper" / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
def _find_training_state() -> pathlib.Path | None:
    """Find a plausible training_state.json to use for curves.

    Preference order:
      1) training/checkpoints/gspo_NEWRUN/final_model/training_state.json
      2) training/checkpoints/gspo/final_model/training_state.json
    """
    cand1 = ROOT / "training" / "checkpoints" / "gspo_NEWRUN" / "final_model" / "training_state.json"
    if cand1.exists():
        return cand1
    cand2 = ROOT / "training" / "checkpoints" / "gspo" / "final_model" / "training_state.json"
    if cand2.exists():
        return cand2
    return None


def _load_training_state(path: pathlib.Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        state = json.load(f)
    # snapshot a copy into paper/data
    shutil.copy2(path, DATA_DIR / "training_state.json")
    return state


def _collect_throughput(storage_dir: pathlib.Path) -> List[Dict[str, Any]]:
    """Collect per-day throughput from timestamped storage.

    For each predictions JSON in storage_dir, attempt to find matching
    evaluations CSV. Return a list of dicts with date, headlines, rollouts,
    evaluated, dropped, avg_reward.
    """
    import csv
    rows: List[Dict[str, Any]] = []
    preds = sorted(storage_dir.glob("*_predictions.json"))
    for pf in preds:
        date = pf.name.split("_")[0]
        try:
            pdata = json.loads(pf.read_text())
        except Exception:
            continue
        headlines = len(pdata.get("predictions", []))
        rollouts = 0
        for item in pdata.get("predictions", []):
            if isinstance(item, dict):
                rollouts += item.get("total_rollouts", len(item.get("rollouts", [])))
        csv_path = storage_dir / f"{date}_evaluations.csv"
        evaluated = 0
        avg_rew = None
        if csv_path.exists():
            with open(csv_path, newline="") as f:
                r = csv.DictReader(f)
                arr = [float(row.get("reward", 0.0)) for row in r]
            evaluated = len(arr)
            if evaluated:
                avg_rew = sum(arr) / evaluated
        dropped = max(0, rollouts - evaluated)
        rows.append({
            "date": date,
            "headlines": headlines,
            "rollouts": rollouts,
            "evaluated": evaluated,
            "dropped": dropped,
            "avg_reward": round(avg_rew, 6) if avg_rew is not None else None,
        })
    # Save snapshot CSV
    import csv as csvm
    out_csv = DATA_DIR / "throughput.csv"
    with open(out_csv, "w", newline="") as f:
        w = csvm.DictWriter(f, fieldnames=["date", "headlines", "rollouts", "evaluated", "dropped", "avg_reward"])
        w.writeheader()
        for row in rows:
            w.writerow(row)
    return rows


def reward_curve_from_state(state: Dict[str, Any]):
    rewards = state.get("total_rewards", [])
    if not rewards:
        return
    plt.figure(figsize=(4, 3))
    plt.plot(range(1, len(rewards) + 1), rewards, linewidth=1.2)
    plt.xlabel("Training step")
    plt.ylabel("Average reward")
    plt.title("Reward progression")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "reward_curve.png", dpi=300)


def kl_curve_from_state(state: Dict[str, Any]):
    kl = state.get("kl_history", [])
    if not kl:
        return
    plt.figure(figsize=(4, 3))
    plt.plot(range(1, len(kl) + 1), kl, linewidth=1.0, color="orange")
    plt.xlabel("Training step")
    plt.ylabel("Loss proxy")
    plt.title("Training stability")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "kl_stability.png", dpi=300)


def throughput_bars(rows: List[Dict[str, Any]]):
    if not rows:
        return
    # Sort by date string
    rows = sorted(rows, key=lambda r: r.get("date", ""))
    plt.figure(figsize=(4, 3))
    evaluated = [r.get("evaluated", 0) for r in rows]
    dropped = [r.get("dropped", 0) for r in rows]
    labels = [r.get("date", "") for r in rows]
    x = range(len(labels))
    width = 0.6
    plt.bar(x, evaluated, width, label="Evaluated")
    plt.bar(x, dropped, width, bottom=evaluated, label="Dropped", color="#cccccc")
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.xlabel("Date")
    plt.ylabel("Count")
    plt.title("Evaluator throughput")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "pipeline_throughput.png", dpi=300)


def main():
    # 1) Training curves from training_state.json
    ts_path = _find_training_state()
    if ts_path is None:
        print("No training_state.json found; skipping reward and stability figures.")
        state = None
    else:
        state = _load_training_state(ts_path)
        reward_curve_from_state(state)
        kl_curve_from_state(state)

    # 2) Throughput from timestamped storage (prefer NEWRUN)
    storage = ROOT / "timestamped_storage_NEWRUN"
    if not storage.exists():
        storage = ROOT / "timestamped_storage"
    rows = _collect_throughput(storage)
    throughput_bars(rows)

    print("Figures saved to", FIG_DIR)


if __name__ == "__main__":
    main()
