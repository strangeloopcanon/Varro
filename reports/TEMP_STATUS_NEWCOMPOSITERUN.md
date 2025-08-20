# TEMP STATUS — NEWCOMPOSITERUN (Evaluator + Rewards)

Updated: 2025-08-18

## Summary
- Fixed evening evaluator bias and improved determinism.
- Disabled trade-thinking by default (broad forecasts focus).
- Added bucketed weighting in training to adapt LLM vs semantic based on ranking strength.
- Cleaned NEWCOMPOSITERUN artifacts; ready for re-run.
- Added automated Q/LR scheduler for backfills (optional) to bootstrap structure adherence, then anneal.

## Code Changes
- outcome_tracking/llm_outcome_evaluator.py
  - Constrained-first, single-letter selection with deterministic per-round shuffle (removes A-position bias).
  - Full-group deterministic semantic ranking if any round fails or <4 picks.
  - Adds `llm_picks` (count of valid selections) and `ranking_source` ('llm' | 'semantic') per evaluation row.
  - Trade-thinking/composite gated behind `VARRO_ENABLE_TRADE_THINKING` (default off).
- outcome_tracking/evaluation_storage.py
  - Builds GSPO training data with bucketed weights per example using `llm_picks`:
    - picks ≤ 2: L=0.00, S=0.95, Q=0.05
    - picks 3–5: L=0.30, S=0.65, Q=0.05
    - picks ≥ 6: L=0.475, S=0.475, Q=0.05
  - Derives effective `outcome_alpha` and `rubric_weight` per example from these absolute weights.
  - Stores diagnostics: `llm_picks`, `bucket`, `outcome_alpha`, `rubric_weight`.
- prediction_generation/adaptive_rollout_generator.py
  - `trade_thinking_score` computed only if `VARRO_ENABLE_TRADE_THINKING=1`.
- reports/Evaluation_Fallback_Incident_20250802_20250811.md added (incident + remediation write-up).

### New: Auto Q/LR Scheduler (backfill)
- scripts/run_backfill_newrun.py
  - New flag `--auto_q_lr` that, when set, automatically schedules per-day format weight (Q) and learning rate (LR), and enables low-bucket LLM weighting once the picker improves.
  - Metric-responsive behavior:
    - Q decays by `q_step` when avg_composite_reward improves ≥ +0.01 vs previous day (floor `q_floor`).
    - LR decays by `lr_decay` after 2 consecutive non-improving days (floor `lr_floor`).
    - Low-bucket LLM weight set to `low_llm_alpha` once previous-day median `llm_picks ≥ llm_picks_threshold`.
  - Defaults (tunable via flags): `q_start=0.10`, `q_floor=0.05`, `q_step=0.02`, `lr_start=1e-6`, `lr_floor=5e-7`, `lr_decay=0.90`, `llm_picks_threshold=4`, `low_llm_alpha=0.20`.
  - Exports per-bucket envs each day (LOW/MID/HIGH) and passes `--learning_rate` to training.

## Behavior Now
- Evening (per headline group of 8):
  - Deterministic shuffle → constrained A–H choice (greedy). If failure at any round or <4 total picks → switch to semantic ranking for the whole group.
  - Rank → reward: 1..8 → 1.0 .. 0.0.
  - Each row carries `llm_picks`, `ranking_source`, and the headlines used.
- Night/Training prep:
  - Computes final reward via per-example bucketed weights: final = Q*quality + (1-Q)*(α*LLM + (1-α)*semantic), where Q and α are derived from the bucket’s absolute weights.
  - Optional (when `--auto_q_lr`): applies scheduled Q and LR per day; enables low-bucket LLM weighting once the picker improves.

## Config (optional overrides)
- Bucket weights (absolute L/S/Q per bucket):
  - `VARRO_BUCKET_LOW_LLM` (def 0.0), `VARRO_BUCKET_LOW_SEM` (0.95), `VARRO_BUCKET_LOW_Q` (0.05)
  - `VARRO_BUCKET_MID_LLM` (0.30), `VARRO_BUCKET_MID_SEM` (0.65), `VARRO_BUCKET_MID_Q` (0.05)
  - `VARRO_BUCKET_HIGH_LLM` (0.475), `VARRO_BUCKET_HIGH_SEM` (0.475), `VARRO_BUCKET_HIGH_Q` (0.05)
- Global fallbacks (used only if bucket envs unset/invalid): `VARRO_RUBRIC_WEIGHT` (def 0.5), `VARRO_OUTCOME_ALPHA` (def 0.7)
- Trade-thinking (default off): `VARRO_ENABLE_TRADE_THINKING=0`; composite preference: `VARRO_PREFER_COMPOSITE=0`.

## Re-run Command
Recommended (bootstrap structure with tight decoding; automated Q/LR):
```
python scripts/run_backfill_newrun.py \
  --start_date 20250802 --end_date 20250811 \
  --force_predictions --force_evaluations --force_training \
  --seed 1234 --sampler_profile tight \
  --run_suffix NEWCOMPOSITERUN2 --auto_q_lr
```
Optional flags to tune schedule:
```
# Examples (defaults shown)
  --q_start 0.10 --q_floor 0.05 --q_step 0.02 \
  --lr_start 1e-6 --lr_floor 5e-7 --lr_decay 0.90 \
  --llm_picks_threshold 4 --low_llm_alpha 0.20
```
Optional envs (example):
```
export VARRO_ENABLE_TRADE_THINKING=0
export VARRO_PREFER_COMPOSITE=0
# Customize buckets if desired; otherwise defaults apply
# export VARRO_BUCKET_LOW_LLM=0.0 VARRO_BUCKET_LOW_SEM=0.95 VARRO_BUCKET_LOW_Q=0.05
# export VARRO_BUCKET_MID_LLM=0.30 VARRO_BUCKET_MID_SEM=0.65 VARRO_BUCKET_MID_Q=0.05
# export VARRO_BUCKET_HIGH_LLM=0.475 VARRO_BUCKET_HIGH_SEM=0.475 VARRO_BUCKET_HIGH_Q=0.05
```

## What to Watch For
- Evening logs:
  - "Round N mapping: {...}" (shuffle active)
  - "Deterministic semantic ranking applied" lines should appear only when LLM fails or picks < 4.
- Training JSON (e.g., `training/gspo_training_YYYYMMDD.json`):
  - Per-example `llm_picks`, `bucket`, `outcome_alpha`, `rubric_weight`, `reward`.

## Open Items / Next Iterations
- Optional: auto-tune bucket weights based on per-day fallback rate.
- Optional: add summary metric in evaluation summaries for percent of groups using semantic fallback.
- Optional: revisit immediate quality weight (Q) if overfitting to format is observed.
