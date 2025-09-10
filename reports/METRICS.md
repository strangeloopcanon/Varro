### Metrics and Definitions (cross-run reports)

- Quality (0..1): Evaluator-derived paragraph quality score using a rubric favoring specific, falsifiable, time‑bound, readable forecasts. Used as a primary metric for paragraph runs.
- Zeros: Share of evaluated items with invalid or degenerate outputs (e.g., extraction failed). Lower is better.
- Very‑low <0.2: Share of items with Quality < 0.2. Acts as a tail/degeneracy indicator.
- Leak: Heuristic meta‑echo/hygiene flag based on a conservative stem list (e.g., instruction echoes). Good for trend and comparisons, not absolute cleanliness.
- Words: Average word count of rollouts; a rough shape/verbosity proxy.

Ranking and rewards
- Grouping: Each headline forms a group with up to 8 rollouts (A–H).
- Primary selection: Deterministic A→H sequential mapping is enforced for valid constrained selections.
- Fallback: If constrained selection is invalid, a deterministic semantic ranking fallback is applied to derive order/scores.
- Normalization: Rewards are normalized within each group before GSPO training.

Artifacts
- Latest CSV: `reports/CROSS_RUN_DAILY_METRICS_YYYYMMDD.csv` (see `reports/CROSS_RUN_COMPARISON.md`).
- Plots: `reports/cross_run_daily_quality.png`, `reports/cross_run_daily_leak.png`.

Notes
- Leak is intentionally conservative; use it for ordering and trend rather than absolute claims.
- Paragraph metrics are not directly comparable to strict one‑line runs without validator‑assisted decoding and validity‑gated rewards.

