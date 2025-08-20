#!/usr/bin/env python3
"""
Generate cross-run metrics CSV, plots, and a canonical synthesis report.
Outputs:
 - reports/CROSS_RUN_DAILY_METRICS_YYYYMMDD.csv
 - reports/cross_run_daily_quality.png
 - reports/cross_run_daily_leak.png
 - reports/CROSS_RUN_COMPARISON_YYYYMMDD.md (summary)
 - reports/ALL_RUNS_SYNTHESIS_SO_WHAT.md (canonical synthesis)
"""
import csv
import datetime as dt
import json
import math
import os
import re
import sys
from glob import glob
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


RUNS = [
    'NEWCOMPOSITERUN',
    'NEWCOMPOSITERUN2',
    'SEMANTICRUN',
    'SEMANTICRUN_TIGHT_Q25',
    'COMPOSITERUN',
]

LEAK_PATTERNS = [
    r"\bkeep (it|to)\b",
    r"\bdo not\b",
    r"\bavoid\b",
    r"\buse (emojis|markdown|bold|bullet|headers|one paragraph)\b",
    r"\banswer:\b",
    r"\bfinal answer\b",
    r"\bnote:\b",
    r"\bthe following (is|are)\b",
    r"\bstart with\b",
    r"\bmake sure\b",
    r"\bin this forecast\b",
    r"\byou must\b",
    r"\bthe user wants\b",
]


def compute_daily_metrics():
    rx = re.compile("|".join(LEAK_PATTERNS), re.I)
    rows = []
    for run in RUNS:
        for f in sorted(glob(f'timestamped_storage_{run}/*_predictions.json')):
            try:
                with open(f) as fh:
                    j = json.load(fh)
            except Exception:
                continue
            date = j.get('date') or os.path.basename(f)[:8]
            total = 0
            words = []
            zeros = 0
            low = 0
            qvals = []
            leaks = 0
            for p in j.get('predictions', []):
                for r in p.get('rollouts', []):
                    pred = (r.get('prediction') or '').strip()
                    v = r.get('immediate_reward')
                    if pred:
                        total += 1
                        words.append(len(pred.split()))
                        if rx.search(pred):
                            leaks += 1
                    if isinstance(v, (int, float)):
                        qvals.append(float(v))
                        if float(v) == 0.0:
                            zeros += 1
                        if float(v) < 0.2:
                            low += 1
            if total == 0:
                continue
            avg_q = sum(qvals) / len(qvals) if qvals else 0.0
            avg_w = sum(words) / len(words) if words else 0.0
            leak = leaks / total if total else 0.0
            zeros_share = zeros / len(qvals) if qvals else 0.0
            low_share = low / len(qvals) if qvals else 0.0
            rows.append((date, run, total, avg_q, zeros_share, low_share, leak, avg_w))
    rows.sort(key=lambda x: (x[0], x[1]))
    return rows


def write_csv(rows, out_csv):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, 'w', newline='') as out:
        wr = csv.writer(out)
        wr.writerow(['date','run','rollouts_N','quality_avg','zeros_share','very_low_lt_0_2_share','leak_share','avg_words'])
        for row in rows:
            date, run, N, qa, zs, ls, lk, w = row
            wr.writerow([date, run, N, f"{qa:.6f}", f"{zs:.6f}", f"{ls:.6f}", f"{lk:.6f}", f"{w:.2f}"])


def compute_overall_averages(rows):
    agg = defaultdict(lambda: defaultdict(list))
    for date, run, N, qa, zs, ls, lk, w in rows:
        agg[run]['qa'].append(qa)
        agg[run]['zs'].append(zs)
        agg[run]['ls'].append(ls)
        agg[run]['lk'].append(lk)
        agg[run]['w'].append(w)
    out = {}
    for run, m in agg.items():
        def mean(x):
            return sum(x)/len(x) if x else float('nan')
        out[run] = {
            'quality': mean(m['qa']),
            'zeros': mean(m['zs']),
            '<0.2': mean(m['ls']),
            'leak': mean(m['lk']),
            'words': mean(m['w']),
        }
    return out


def plot_from_csv(csv_path, quality_png, leak_png):
    # Read csv to dict by run
    by_run = defaultdict(list)
    with open(csv_path, newline='') as fh:
        dr = csv.DictReader(fh)
        for row in dr:
            by_run[row['run']].append(row)
    for series in by_run.values():
        series.sort(key=lambda r: r['date'])

    colors = {
        'SEMANTICRUN_TIGHT_Q25': '#2ca02c',
        'SEMANTICRUN': '#1f77b4',
        'NEWCOMPOSITERUN': '#ff7f0e',
        'NEWCOMPOSITERUN2': '#d62728',
        'COMPOSITERUN': '#9467bd',
    }

    # Quality plot
    plt.figure(figsize=(9,5))
    for run in ['SEMANTICRUN_TIGHT_Q25','SEMANTICRUN','NEWCOMPOSITERUN','NEWCOMPOSITERUN2','COMPOSITERUN']:
        series = by_run.get(run, [])
        if not series:
            continue
        xs = [dt.datetime.strptime(r['date'], '%Y%m%d') for r in series]
        ys = [float(r['quality_avg']) for r in series]
        plt.plot(xs, ys, label=run, color=colors.get(run))
    plt.title('Daily Paragraph Quality (immediate_reward) by Run')
    plt.xlabel('Date'); plt.ylabel('Quality avg (0–1)'); plt.ylim(0,1)
    plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout(); plt.savefig(quality_png, dpi=150); plt.close()

    # Leak plot
    plt.figure(figsize=(9,5))
    for run in ['SEMANTICRUN_TIGHT_Q25','SEMANTICRUN','NEWCOMPOSITERUN','NEWCOMPOSITERUN2','COMPOSITERUN']:
        series = by_run.get(run, [])
        if not series:
            continue
        xs = [dt.datetime.strptime(r['date'], '%Y%m%d') for r in series]
        ys = [float(r['leak_share']) for r in series]
        plt.plot(xs, ys, label=run, color=colors.get(run))
    plt.title('Daily Meta Leak (heuristic) by Run')
    plt.xlabel('Date'); plt.ylabel('Leak share (0–1)'); plt.ylim(0,1)
    plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout(); plt.savefig(leak_png, dpi=150); plt.close()


def write_cross_run_comparison(md_path, csv_path, averages):
    lines = []
    lines.append("### Cross-Run Daily Metrics (latest)")
    lines.append("")
    lines.append(f"- File: `{csv_path}`")
    lines.append("- Plots: `reports/cross_run_daily_quality.png`, `reports/cross_run_daily_leak.png`")
    lines.append("")
    lines.append("#### Per-run overall averages")
    for run in ['COMPOSITERUN','NEWCOMPOSITERUN','NEWCOMPOSITERUN2','SEMANTICRUN','SEMANTICRUN_TIGHT_Q25']:
        if run not in averages: continue
        a = averages[run]
        lines.append(f"- {run}: quality={a['quality']:.3f}, zeros={a['zeros']:.3f}, <0.2={a['<0.2']:.3f}, leak={a['leak']:.3f}, words={a['words']:.1f}")
    lines.append("")
    lines.append("Notes")
    lines.append("- Leak is a conservative heuristic that flags meta echoes; use for comparison/trend, not absolute cleanliness.")
    lines.append("- One-line runs are not quality-comparable with paragraph metrics.")
    os.makedirs(os.path.dirname(md_path), exist_ok=True)
    with open(md_path, 'w') as f:
        f.write("\n".join(lines) + "\n")


def write_synthesis(md_path, csv_path, averages):
    # Keep this concise and aligned with our prior synthesis
    lines = []
    lines.append("### All Runs Synthesis & So‑What (latest)")
    lines.append("")
    lines.append("#### Executive TL;DR")
    lines.append("- Paragraph forecasting with a positive prompt, `LLM=0 / Semantic≈0.75 / Format(Q)≈0.25`, and `sampler_profile=tight` yields highest quality and lowest failure rates.")
    lines.append("- Leakage persists due to scorer–detector misalignment; align penalties or pre-strip echoes to improve hygiene without sacrificing specificity.")
    lines.append("- One-line strictness bottlenecks learning; needs validator-assisted decoding and validity-gated rewards to become useful.")
    lines.append("")
    lines.append("#### Quantitative Recap (overall averages)")
    lines.append(f"- Source CSV: `{csv_path}`")
    for run in ['COMPOSITERUN','NEWCOMPOSITERUN','NEWCOMPOSITERUN2','SEMANTICRUN','SEMANTICRUN_TIGHT_Q25']:
        if run not in averages: continue
        a = averages[run]
        lines.append(f"- {run}: quality={a['quality']:.3f}, zeros={a['zeros']:.3f}, <0.2={a['<0.2']:.3f}, leak={a['leak']:.3f}, words={a['words']:.1f}")
    lines.append("")
    lines.append("#### Defaults to Operationalize")
    lines.append("- Paragraph: `LLM=0`, `Semantic≈0.75`, `Format(Q)≈0.25`, `sampler=tight`, `tokens≈160–180`; positive 3–5 sentence prompt with ‘Go.’")
    lines.append("- Scorer alignment: penalize/strip echo stems; monitor quality/leak daily.")
    lines.append("- One-line: sample-N + validator selection, retries, and validity-gated rewards.")
    os.makedirs(os.path.dirname(md_path), exist_ok=True)
    with open(md_path, 'w') as f:
        f.write("\n".join(lines) + "\n")


def main():
    today = dt.datetime.now().strftime('%Y%m%d')
    rows = compute_daily_metrics()
    csv_path = f'reports/CROSS_RUN_DAILY_METRICS_{today}.csv'
    write_csv(rows, csv_path)
    # Plots
    plot_from_csv(csv_path, 'reports/cross_run_daily_quality.png', 'reports/cross_run_daily_leak.png')
    # Averages and reports
    avgs = compute_overall_averages(rows)
    # Dated and undated comparison reports
    write_cross_run_comparison(f'reports/CROSS_RUN_COMPARISON_{today}.md', csv_path, avgs)
    write_cross_run_comparison('reports/CROSS_RUN_COMPARISON.md', csv_path, avgs)
    write_synthesis('reports/ALL_RUNS_SYNTHESIS_SO_WHAT.md', csv_path, avgs)
    print(csv_path)
    print('reports/cross_run_daily_quality.png')
    print('reports/cross_run_daily_leak.png')
    print('reports/ALL_RUNS_SYNTHESIS_SO_WHAT.md')


if __name__ == '__main__':
    main()
