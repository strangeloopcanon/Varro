### Scripts

Helper/utility entry points moved under `scripts/` to keep the repo root tidy. Root stubs remain for backward compatibility and print a deprecation notice.

File tree
- `manage_models.py`: List/info/archive checkpoints.
- `run_backfill_newrun.py`: Backfill/continue runs over a date range with per-run suffix.
- `run_baseline_comparisons.py`: Train/evaluate MLE and KTO baselines.
- `run_next_update.py`: Convenience wrapper to run yesterday→today update.
- `run_override_update.py`: Evaluate D with headlines from H, train, then generate H predictions.

Quick commands
- Model versions: `python scripts/manage_models.py --list`
- Backfill example: `python scripts/run_backfill_newrun.py --start_date 20250804 --end_date 20250810 --run_suffix NEWCOMPOSITERUN --resume_from_last_model`
- Next-day update: `python scripts/run_next_update.py --run_suffix NEWCOMPOSITERUN --resume_from_last_model`
- Override update: `python scripts/run_override_update.py --prediction_date 20250808 --headlines_date 20250810 --resume_from_last_model`
- Baselines: `python scripts/run_baseline_comparisons.py`

