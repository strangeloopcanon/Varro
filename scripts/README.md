### Scripts

Helper/utility entry points moved under `scripts/` to keep the repo root tidy. Root stubs remain for backward compatibility and print a deprecation notice.

File tree
- `manage_models.py`: List/info/archive checkpoints.
- `run_backfill_newrun.py`: Backfill/continue runs over a date range with per-run suffix.
- `run_baseline_comparisons.py`: Train/evaluate MLE and KTO baselines.
- `run_next_update.py`: Convenience wrapper to run yesterday→today update.
- `run_override_update.py`: Evaluate D with headlines from H, train, then generate H predictions.
- `run_semantic_with_articles.py`: Backfill dates with article‑aware prompting (attaches cleaned excerpts).
- `clean_articles.py`: Pre-clean article HTML into excerpt text for morning prompts.
- `ab_compare_qwen3_strict.py`: A/B compare two checkpoints on fixed headlines (strict 1.7B harness).

Quick commands
- Model versions: `python scripts/manage_models.py --list`
- Backfill example: `python scripts/run_backfill_newrun.py --start_date 20250804 --end_date 20250810 --run_suffix NEWCOMPOSITERUN --resume_from_last_model`
- Next-day update: `python scripts/run_next_update.py --run_suffix NEWCOMPOSITERUN --resume_from_last_model`
- Override update: `python scripts/run_override_update.py --prediction_date 20250808 --headlines_date 20250810 --resume_from_last_model`
- Baselines: `python scripts/run_baseline_comparisons.py`
 - Article‑aware backfill: `python scripts/run_semantic_with_articles.py --start_date 20250829 --end_date 20250907 --resume_from_last_model`
 - Clean articles: `python scripts/clean_articles.py --input timestamped_storage_<SUFFIX>/<DATE> --output timestamped_storage_<SUFFIX>/<DATE>`
 - A/B compare (strict harness):
   `python scripts/ab_compare_qwen3_strict.py --date 20250904 --ckpt_a training/checkpoints/gspo_QWEN3_17B_STRICT_CHAT_ART_RERUN/final_model_20250829 --ckpt_b training/checkpoints/gspo_QWEN3_17B_STRICT_CHAT_ART_RERUN/final_model --run_suffix QWEN3_17B_STRICT_CHAT_ART_RERUN --num_rollouts 2 --sampler_profile tight --limit 12 --with_articles`
