# Roadmap (Next Steps)

Concise, living checklist to guide the next iterations. Use GitHub Issues for details; keep this file high-level.

## 1) Pipeline hardening and ops
- [ ] Night stage: optionally auto-run GSPO training after preparing training/gspo_training_<date>.json (training JSON prepared; training not yet invoked)
- [x] Evaluations CSV: write timestamped_storage/<date>_evaluations.csv for analysis
- [ ] Simple dashboard/notebook to plot daily counts and avg reward from CSVs
- [ ] Add minimal unit tests for storage read/write and date-range utilities

## 2) Evaluator resiliency and quality
- [x] Deterministic fallback when stochastic selection fails to yield a valid letter
- [ ] Log a small sample of failed rounds for forensics (inputs + outputs, redacted) (basic logging present; add sampling/redaction)
- [ ] Unit test _extract_single_letter with tricky cases (JSON/brackets/parentheses/“A.” line-start)
- [ ] Optional: add failure reason tags and a daily failure summary

## 3) Training experiments (GSPO)
- [x] Toggle: response-only loss masking (default ON)
- [x] Toggle: EMA reward baseline (default OFF)
- [ ] Optional: add length-normalization for response loss (toggle)
- [x] Auto-log both raw reward and advantage when EMA is enabled
- [ ] Wire pipeline night stage to invoke run_gspo_training.py with toggles via config

## 4) Data and scale-up
- [ ] 30-day extended run; monitor evaluator completion and reward trends
- [ ] Improve de-duplication (optional fuzzy match) and source coverage via config/rss_sources.json
- [ ] Add a small deterministic sample bundle and mini E2E test script (sample data present; add E2E script)

## 5) Baselines (lower priority)
- [x] Supervised MLE baseline (minimal pass) using current training data
- [x] KTO preference baseline (pair construction from rank-derived rewards)
- [ ] Compare zero-shot vs GSPO vs MLE/KTO on mean reward and hit-rate

## 6) Optional paper polish
- [ ] Archive final paper assets; keep code/docs aligned with the deployed pipeline

Legend: [x] done, [ ] planned
