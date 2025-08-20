### NEWCOMPOSITERUN run report (2025-08-18)

#### Scope and setup
- **Run suffix**: `NEWCOMPOSITERUN`
- **Model**: `Qwen/Qwen3-0.6B` (MLX)
- **Trainer**: GSPO (response-only loss), 1 epoch executed in this session
- **Evaluator updates (from `reports/TEMP_STATUS_NEWCOMPOSITERUN.md`)**:
  - Deterministic greedy A–H selection with per-round constrained shuffle; full-group semantic fallback if any round fails or <4 picks.
  - Added diagnostics per evaluation row: `llm_picks`, `ranking_source` and bucket-derived weights (`outcome_alpha`, `rubric_weight`) for training data.
  - Trade-thinking disabled by default; composite reward is primary signal.

#### Completion and artifacts
- **Training**: Completed successfully; final checkpoint saved to `training/checkpoints/gspo_NEWCOMPOSITERUN/final_model_20250818` and symlink/latest updated at `.../final_model`.
- **Predictions (latest day processed this session)**: 2025-08-11 — 31 headlines, 248 rollouts saved to `timestamped_storage_NEWCOMPOSITERUN/20250811_predictions.json`.
- **Evaluations**: COMPOSITERUN namespace contains complete evals for 2025-08-11 (`timestamped_storage_COMPOSITERUN/20250811_evaluations.*`). NEWRUN-side evals for 2025-08-11 were not found at time of report; numbers below for that date use COMPOSITERUN evals as the reference for quality/composite trends.

#### Quantitative summary
- **Trainer epoch (from logs)**
  - Total steps: 2,136
  - Avg reward: 0.0461 (trainer scale); range: 0.0000–0.1000; std: 0.0308
  - Avg KL: 9.2550
  - Improvement vs prior: −0.0002 (−0.5%)

- **Generation (2025-08-11)**
  - Headlines: 31; Rollouts: 248; Sampler profile: tight
  - By construction of the 1→8 rank→[1.0→0.0] mapping, the mean per-group ranking reward is ~0.50.

- **Evaluation quality (COMPOSITERUN 2025-08-11 reference)**
  - Composite reward (avg): ≈ 0.211
  - Trade-thinking score (avg): ≈ 0.131
  - Validity (strict one-line schema match): ≈ 0.4%
  - Meta leak rate: ≈ 2.8%
  - Avg output length: ≈ 25 words

Notes:
- The evaluator changes aim to reduce A-position bias and introduce deterministic semantics fallback plus bucketed training weights. The new per-row diagnostics are expected in the training JSONs; the COMPOSITERUN evaluation JSONs do not yet surface these fields, so fallback rate cannot be quantified here.

#### Qualitative assessment
- **Stability**: Training ran cleanly with tracked KL and bounded rewards; no instability or divergence observed.
- **Evaluator behavior**: Rankings appear well-distributed across 1–8; deterministic shuffle likely removed earlier positional bias.
- **Output format**: Strict schema adherence remains the primary bottleneck (very low validity); despite this, composite reward provides usable learning signal.
- **Content**: Meta/instruction leakage improved relative to early days; outputs remain concise but often include explanatory scaffolding rather than falsifiable, proxy-grounded claims.

#### Risks and limitations
- NEWRUN 2025-08-11 evaluations not present at report time; relying on COMPOSITERUN 2025-08-11 quality metrics as a proxy.
- Aggregate summaries still under-report outcome scores (zeros) when readers expect `outcome_score` fields; composite remains the reliable metric.
- Low validity combined with tight sampling likely constrains exploration and may limit gains from ranking-based reward.

#### Recommendations (next actions)
- **Validity-first decoding**: Add constrained retries or sample-N + validator selection; allow mild regex relaxation for near-miss acceptance; stop at first newline.
- **Training targets**: Gate reward by validity (0 for invalid; composite for valid) or add a small validity bonus to bootstrap adherence.
- **Few-shot scaffolding**: Provide 2–3 exemplars and reduce temperature to lift structure adherence and reduce meta leakage.
- **Telemetry**: Persist `llm_picks`, `ranking_source`, bucket, `outcome_alpha`, `rubric_weight` into evaluation/training summaries; add per-day fallback rate and validity %.
- **A/B checks**: Run side-by-side evaluation vs previous `final_model` on a fixed headline set to measure real delta in composite and leak.

#### Quick references
- Checkpoints: `training/checkpoints/gspo_NEWCOMPOSITERUN/final_model_20250818` and `.../final_model`
- Predictions (latest day): `timestamped_storage_NEWCOMPOSITERUN/20250811_predictions.json`
- Evaluations (reference used): `timestamped_storage_COMPOSITERUN/20250811_evaluations.json|.csv`


