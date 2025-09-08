#!/usr/bin/env python3
"""
Make paper figures for the article-aware validation.

Reads the latest reports/CROSS_RUN_DAILY_METRICS_*.csv and plots
SEMANTICRUN_TIGHT_Q25 vs SEMANTICRUN_TIGHT_Q25_ARTICLES for:
 - Daily paragraph quality
 - Daily zeros share (very-low optional overlay)

Outputs: paper/figs/article_aware_validation.png
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


def main():
    csv_path = latest_cross_run_csv()
    runs = ["SEMANTICRUN_TIGHT_Q25", "SEMANTICRUN_TIGHT_Q25_ARTICLES"]
    series = load_series(csv_path, runs)
    out = FIGS / "article_aware_validation.png"
    plot_article_validation(series, out)


if __name__ == "__main__":
    main()

