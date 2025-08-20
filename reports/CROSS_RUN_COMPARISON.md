### Cross-Run Daily Metrics (latest)

- File: `reports/CROSS_RUN_DAILY_METRICS_20250819.csv`
- Plots: `reports/cross_run_daily_quality.png`, `reports/cross_run_daily_leak.png`

#### Per-run overall averages
- COMPOSITERUN: quality=0.000, zeros=1.000, <0.2=1.000, leak=0.132, words=28.9
- NEWCOMPOSITERUN: quality=0.462, zeros=0.100, <0.2=0.132, leak=0.693, words=124.5
- NEWCOMPOSITERUN2: quality=0.242, zeros=0.432, <0.2=0.472, leak=0.708, words=120.8
- SEMANTICRUN: quality=0.441, zeros=0.116, <0.2=0.161, leak=0.708, words=123.8
- SEMANTICRUN_TIGHT_Q25: quality=0.643, zeros=0.013, <0.2=0.029, leak=0.200, words=129.2

Notes
- Leak is a conservative heuristic that flags meta echoes; use for comparison/trend, not absolute cleanliness.
- One-line runs are not quality-comparable with paragraph metrics.
