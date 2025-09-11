#!/usr/bin/env python3
"""
Make paper figures for the manuscript.

Reads the latest reports/CROSS_RUN_DAILY_METRICS_*.csv and produces:

1) Article-aware validation trends comparing SEMANTICRUN_TIGHT_Q25 vs
   SEMANTICRUN_TIGHT_Q25_ARTICLES for:
   - Daily paragraph quality
   - Daily zeros share
   Output: paper/figs/article_aware_validation.png

2) Cross-run overview bar charts (overall averages computed from the CSV):
   - Quality avg (0–1)
   - Zeros share (0–1)
   - Leak share (0–1)
   Output: paper/figs/cross_run_overview.png
"""

from __future__ import annotations

import csv
import glob
import os
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
FIGS = ROOT / "paper" / "figs"


def latest_cross_run_csv() -> Path:
    files = sorted(glob.glob(str(REPORTS / "CROSS_RUN_DAILY_METRICS_*.csv")))
    if not files:
        raise SystemExit("No cross-run CSV found in reports/")
    return Path(files[-1])


def load_series(csv_path: Path, runs: list[str]) -> dict[str, list[tuple[datetime, float, float, float]]]:
    # Returns mapping run -> list of (date, quality, zeros, leak)
    series: dict[str, list[tuple[datetime, float, float, float]]] = {r: [] for r in runs}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            run = row.get("run")
            if run not in series:
                continue
            try:
                dt = datetime.strptime(row["date"], "%Y%m%d")
                q = float(row["quality_avg"]) if row.get("quality_avg") else 0.0
                z = float(row["zeros_share"]) if row.get("zeros_share") else 0.0
                l = float(row["leak_share"]) if row.get("leak_share") else 0.0
            except Exception:
                continue
            series[run].append((dt, q, z, l))
    # sort by date
    for r in runs:
        series[r].sort(key=lambda t: t[0])
    return series


def plot_article_validation(series: dict[str, list[tuple[datetime, float, float, float]]], out_path: Path):
    runs = list(series.keys())
    colors = {
        "SEMANTICRUN_TIGHT_Q25": "#1f77b4",  # blue
        "SEMANTICRUN_TIGHT_Q25_ARTICLES": "#ff7f0e",  # orange
    }
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))

    # Quality subplot
    ax = axes[0]
    for r in runs:
        xs = [d for d, *_ in series[r]]
        qs = [q for _, q, *_ in series[r]]
        ax.plot(xs, qs, label=r, color=colors.get(r), marker="o", linewidth=1.8, markersize=3.5)
    ax.set_title("Daily Paragraph Quality")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("Date")
    ax.set_ylabel("Quality avg (0–1)")
    ax.legend(fontsize=8)

    # Zeros subplot
    ax2 = axes[1]
    for r in runs:
        xs = [d for d, *_ in series[r]]
        zs = [z for _, _, z, _ in series[r]]
        ax2.plot(xs, zs, label=r, color=colors.get(r), marker="o", linewidth=1.8, markersize=3.5)
    ax2.set_title("Daily Zeros Share")
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Share (0–1)")
    ax2.legend(fontsize=8)

    fig.tight_layout()
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(out_path)


def compute_overall_averages(csv_path: Path) -> list[tuple[str, float, float, float]]:
    """Return list of (run, q_avg, zeros, leak) ordered by a sensible default."""
    import csv as _csv
    from collections import defaultdict

    sums = defaultdict(lambda: {"q": 0.0, "z": 0.0, "l": 0.0, "n": 0})
    with open(csv_path, newline="") as f:
        reader = _csv.DictReader(f)
        for row in reader:
            run = row.get("run")
            if not run:
                continue
            try:
                q = float(row.get("quality_avg") or 0.0)
                z = float(row.get("zeros_share") or 0.0)
                l = float(row.get("leak_share") or 0.0)
            except Exception:
                continue
            sums[run]["q"] += q
            sums[run]["z"] += z
            sums[run]["l"] += l
            sums[run]["n"] += 1

    def avg(d: dict) -> tuple[float, float, float]:
        n = max(1, d.get("n", 0))
        return d["q"] / n, d["z"] / n, d["l"] / n

    # Preferred ordering for readability
    preferred = [
        "COMPOSITERUN",
        "NEWCOMPOSITERUN",
        "NEWCOMPOSITERUN2",
        "SEMANTICRUN",
        "SEMANTICRUN_TIGHT_Q25",
        "SEMANTICRUN_TIGHT_Q25_ARTICLES",
    ]
    runs = list(sums.keys())
    ordered = [r for r in preferred if r in runs] + [r for r in runs if r not in preferred]

    out: list[tuple[str, float, float, float]] = []
    for r in ordered:
        q, z, l = avg(sums[r])
        out.append((r, q, z, l))
    return out


def plot_cross_run_overview(overall: list[tuple[str, float, float, float]], out_path: Path):
    runs = [r for r, *_ in overall]
    q = [v for _, v, *_ in overall]
    z = [v for *_, v, _ in overall]
    l = [v for *_, v in overall]

    import numpy as _np
    x = _np.arange(len(runs))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10.5, 4.5))
    ax.bar(x - width, q, width, label="Quality", color="#1f77b4")
    ax.bar(x, z, width, label="Zeros", color="#ff7f0e")
    ax.bar(x + width, l, width, label="Leak", color="#2ca02c")
    ax.set_xticks(x)
    ax.set_xticklabels(runs, rotation=20, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Share / Score (0–1)")
    ax.set_title("Cross‑Run Overview (overall averages)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(out_path)


def main():
    csv_path = latest_cross_run_csv()
    # 1) Article-aware validation trends
    runs = ["SEMANTICRUN_TIGHT_Q25", "SEMANTICRUN_TIGHT_Q25_ARTICLES"]
    series = load_series(csv_path, runs)
    out1 = FIGS / "article_aware_validation.png"
    plot_article_validation(series, out1)

    # 2) Cross-run overview bars
    overall = compute_overall_averages(csv_path)
    out2 = FIGS / "cross_run_overview.png"
    plot_cross_run_overview(overall, out2)


if __name__ == "__main__":
    main()
