# Model Card — Varro (LLMONLY legacy namespace)

- Checkpoint: `timestamped_storage_LLMONLY/checkpoints/final_model`
- Last updated: 2025-09-10

## Intended Use
- Legacy baseline checkpoint for LLM‑only runs; useful for sanity checks or A/B against GSPO‑trained models.

## Notes
- This directory mirrors the expected tokenizer/model layout used elsewhere (MLX/Qwen family).
- For current GSPO‑trained models and metrics, prefer the `training/checkpoints/gspo_*` namespaces and see:
  - `reports/CROSS_RUN_COMPARISON.md`
  - `reports/METRICS.md`

## How to Use
- Morning predictions: `python run_daily_pipeline.py --mode morning --trained-model timestamped_storage_LLMONLY/checkpoints/final_model`

