"""Generate figures for the paper.

Assumes TRAINING_SUMMARY_REPORT.md contains the same tables as in
Section 1 of the manuscript.  The script extracts numbers with simple
regex and produces three PNGs in paper/figs/:

* reward_curve.png – reward vs. training step
* pipeline_throughput.png – evaluated vs. dropped roll-outs per day
* kl_stability.png – KL divergence over steps

The plots are meant for rough illustration; they can be replaced by
more rigorous notebooks later.
"""

from __future__ import annotations

import re
import pathlib
from typing import List

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
REPORT = ROOT / "TRAINING_SUMMARY_REPORT.md"
FIG_DIR = ROOT / "paper" / "figs"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def _parse_table(lines: List[str]) -> List[List[str]]:
    """Parse a simple pipe-separated markdown table."""
    rows = []
    for ln in lines:
        if not ln.strip() or ln.startswith("|-"):
            continue
        if ln.strip().startswith("|"):
            cells = [c.strip() for c in ln.strip().strip("|").split("|")]
            rows.append(cells)
    return rows


def extract_summary():
    """Extract per-day metrics from TRAINING_SUMMARY_REPORT.md."""
    if not REPORT.exists():
        raise SystemExit("TRAINING_SUMMARY_REPORT.md not found; cannot build figures.")

    lines = REPORT.read_text().splitlines()

    day_blocks: List[dict] = []
    current_day = None
    for ln in lines:
        m = re.match(r"### Day (\d+): ([A-Za-z]+) (\d+)[a-z]{2}, (\d{4})", ln)
        if m:
            current_day = {
                "day": int(m.group(1)),
                "date": f"{m.group(2)} {m.group(3)}",
            }
            day_blocks.append(current_day)
        else:
            # look for metrics
            if current_day is None:
                continue
            m2 = re.search(r"Average reward: ([0-9.]+)", ln)
            if m2:
                current_day["avg_reward"] = float(m2.group(1))
            m3 = re.search(r"Average KL: ([0-9.]+)", ln)
            if m3:
                current_day["kl"] = float(m3.group(1))
            m4 = re.search(r"Headlines collected: (\d+)", ln)
            if m4:
                current_day["headlines"] = int(m4.group(1))
            m5 = re.search(r"Predictions generated: (\d+)", ln)
            if m5:
                current_day["rollouts"] = int(m5.group(1))
            m6 = re.search(r"Evaluations completed: (\d+)", ln)
            if m6:
                current_day["evaluated"] = int(m6.group(1))

    return day_blocks


def reward_curve(days):
    plt.figure(figsize=(4, 3))
    plt.plot([d["day"] for d in days], [d["avg_reward"] for d in days], marker="o")
    plt.xlabel("Day")
    plt.ylabel("Average normalised reward")
    plt.title("Reward progression")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "reward_curve.png", dpi=300)


def kl_curve(days):
    plt.figure(figsize=(4, 3))
    plt.plot([d["day"] for d in days], [d["kl"] for d in days], marker="o", color="orange")
    plt.xlabel("Day")
    plt.ylabel("KL Divergence")
    plt.title("KL stability")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "kl_stability.png", dpi=300)


def throughput_bars(days):
    plt.figure(figsize=(4, 3))
    evaluated = [d.get("evaluated", 0) for d in days]
    dropped = [d.get("rollouts", 0) - d.get("evaluated", 0) for d in days]
    labels = [d["day"] for d in days]
    width = 0.5
    plt.bar(labels, evaluated, width, label="Evaluated")
    plt.bar(labels, dropped, width, bottom=evaluated, label="Dropped", color="#cccccc")
    plt.xlabel("Day")
    plt.ylabel("Count")
    plt.title("Evaluator throughput")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "pipeline_throughput.png", dpi=300)


def main():
    days = extract_summary()
    if not days:
        print("No data parsed.")
        return

    reward_curve(days)
    kl_curve(days)
    throughput_bars(days)
    print("Figures saved to", FIG_DIR)


if __name__ == "__main__":
    main()
