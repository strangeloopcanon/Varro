# Session Notes — 2025-08-18

Concise record so we can resume smoothly next time.

## Agreed Command
```
python scripts/run_backfill_newrun.py \
  --start_date 20250802 \
  --end_date 20250811 \
  --force_predictions \
  --force_evaluations \
  --force_training \
  --seed 1234 \
  --sampler_profile tight \
  --auto_profile \
  --auto_q_lr \
  --run_suffix NEWCOMPOSITERUN
```

- "auto gl" referred to `--auto_q_lr` (auto-anneals Q and LR).
- Defaults (overridable): `--q_start 0.10 --q_floor 0.05 --q_step 0.02 --lr_start 1e-6 --lr_floor 5e-7 --lr_decay 0.90`.

## Outputs & Logs
- Predictions/Evals: `timestamped_storage_NEWCOMPOSITERUN/<date>_*.json`
- Training JSON: `training/gspo_training_<date>.json`
- Checkpoints: `training/checkpoints/gspo_NEWCOMPOSITERUN/final_model[_<date>]`
- Logs: `training/logs/NEWCOMPOSITERUN_<date>_*.log`

## Options We May Toggle Next
- Resume from last model: `--resume_from_last_model`
- Evaluate due instruments: `--evaluate_due`
- Horizons: `--horizon next_day` or `--horizons next_day,next_2days,next_3days`

## Resume Guidance
- Re-running the same command is idempotent by file gating; use `--force_*` flags only when you want regeneration.
- Environment weights for buckets are auto-set when `--auto_q_lr` is enabled.
