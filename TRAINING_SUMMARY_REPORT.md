# Varro: Continuous GSPO Training Summary (Aug 2–Aug 11, 2025)

## Executive Summary
- We implemented a fully automated daily loop for financial prediction with Qwen/Qwen3‑0.6B (MLX) and a GSPO‑inspired offline update.
- A strict evaluation fix (1–8 ranks with deduplication, clamping, and deterministic padding) eliminated invalid labels and negative rewards.
- From Aug 4–Aug 10 training JSONs, rewards are bounded in [0,1], avg≈0.50, negatives=0 for every day; training is stable and improving.
- In‑epoch average reward increased steadily day‑over‑day, while KL divergence rose moderately (healthy specialization without collapse).
- The pipeline is reproducible, versioned (dated + “final_model”), and runs on a single Apple M4 Max laptop.

## System Overview
- Model: Qwen/Qwen3‑0.6B via MLX/MLX‑LM
- Training: GSPO‑inspired, response‑only loss, 1 epoch/day
- Generation: 8 stochastic roll‑outs/headline (temp=0.8, top_p=0.95, top_k=50)
- Evaluation: Iterative A–H selection with strict final 1–8 ranks (dedupe, clamp [1,8], pad unpicked=7)
- Checkpointing: dated `final_model_YYYYMMDD/` + rolling `final_model/`
- Namespaces: base `timestamped_storage/`; isolated run `timestamped_storage_NEWRUN/`

## NEWRUN Fixes (Aug 8–Aug 11)
- Enforced strict 8‑rank validation (deduplicate, stop at 8, clamp to 1–8, pad unpicked=7)
- Removed evaluator overflow (ranks > 8) and eliminated negative rewards
- Added override option to evaluate predictions from date D against headlines from date H (for backfill/recovery)

## Data & Timeline (post‑fix, used for training)
- 2025‑08‑04: n=256, avg=0.50, min=0.0, max=1.0, negatives=0
- 2025‑08‑05: n=224, avg=0.50, min=0.0, max=1.0, negatives=0
- 2025‑08‑06: n=216, avg=0.50, min=0.0, max=1.0, negatives=0
- 2025‑08‑07: n=280, avg=0.50, min=0.0, max=1.0, negatives=0
- 2025‑08‑08: n=248, avg=0.50, min=0.0, max=1.0, negatives=0
- 2025‑08‑10: n=144, avg=0.50, min=0.0, max=1.0, negatives=0 (evaluated with 2025‑08‑11 headlines via override)

Today’s generation (2025‑08‑11): 248 roll‑outs; avg structure score ≈ 0.826

## Training Dynamics (epoch summaries)
- 2025‑08‑04: steps≈1480, avg_reward≈0.0135, avg_KL≈4.52, improvement≈+0.0021
- 2025‑08‑05: steps≈1704, avg_reward≈0.0183, avg_KL≈5.99, improvement≈+0.0145
- 2025‑08‑06: steps≈1920, avg_reward≈0.0219, avg_KL≈7.05, improvement≈+0.0238
- 2025‑08‑07: steps≈2200, avg_reward≈0.0254, avg_KL≈8.16, improvement≈+0.0358
- 2025‑08‑08: steps≈2448, avg_reward≈0.0279, avg_KL≈8.88, improvement≈+0.0441
- 2025‑08‑10: steps≈2592, avg_reward≈0.0292, avg_KL≈9.01, improvement≈+0.0417

Observations:
- In‑epoch average reward increases monotonically across days
- KL grows gradually (≈4.5 → ≈9.0), indicating stable specialization

## Evaluation Integrity
- Strict 1–8 ranks ensured clean [0,1] reward mapping with negatives=0
- Rankings are varied (not trivially sequential); logs show diverse permutations

## Operations
- Daily update (yesterday→today):
```
python run_next_update.py --resume_from_last_model --run_suffix NEWRUN --seed 1234
```
- Backfill (isolated NEWRUN):
```
python run_backfill_newrun.py --start_date 20250804 --end_date 20250808 \
  --resume_from_last_model --force_evaluations --force_training \
  --run_suffix NEWRUN --seed 1234
```
- Override evaluation (evaluate D with headlines from H):
```
python run_override_update.py --prediction_date D --headlines_date H \
  --resume_from_last_model --run_suffix NEWRUN --seed 1234
```

## Files & Versioning
- Data: `timestamped_storage(_NEWRUN)/<date>_{headlines,predictions,outcome_tracking,evaluations}.json`
- Training: `training/gspo_training_<date>.json`, logs under `training/logs/`
- Checkpoints: `training/checkpoints/gspo_NEWRUN/`
  - Rolling latest: `final_model/`
  - Dated snapshots: `final_model_YYYYMMDD/`

## Next Steps
- Add A/B behavioral evals vs. baseline and downstream economic metrics
- Log both step‑time (scaled) and unscaled rewards for clarity
- Track evaluator retry counts and agreement as quality signal
- Explore EMA baselines and masked‑prompt vs full‑loss ablations

## Conclusion
Post‑fix, the daily pipeline is healthy: evaluations yield strict ranks, rewards are clean in [0,1], and GSPO‑style training improves steadily with controlled KL. The system is reproducible, versioned, and ready for sustained operation and paper‑grade reporting.
