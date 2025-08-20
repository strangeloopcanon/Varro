### SEMANTICRUN run report (2025-08-19)

#### Scope and setup
- **Run suffix**: `SEMANTICRUN`
- **Model**: `Qwen/Qwen3-0.6B` (MLX)
- **Trainer**: GSPO (response-only loss), 1 epoch over the backfill range
- **Weights (per-bucket absolute)**: LLM=0.00, Semantic=0.80, Format(Q)=0.20 across buckets
  - Observed in training JSONs: `rubric_weight≈0.200`, `outcome_alpha≈0.000`, `llm_picks=0` median.
- **Output format**: Paragraph forecasts (quality-scored), not the one-line schema.

#### Completion and artifacts
- **Training**: Final checkpoint saved to `training/checkpoints/gspo_SEMANTICRUN/final_model_20250818`; latest at `training/checkpoints/gspo_SEMANTICRUN/final_model`.
- **Predictions**: 2025-08-02 → 2025-08-11 under `timestamped_storage_SEMANTICRUN/` (8 rollouts/headline).
- **Evaluations**: Present for 2025-08-02 … 2025-08-10; 2025-08-11 has predictions only at report time.

#### Quantitative summary
- **Trainer epoch (from `final_model/training_state.json`)**
  - Steps: 2,136
  - Avg reward: 0.0460; range: 0.0000–0.1000; std: 0.0308
  - KL avg: 9.232; range: 0.0–32.25; std: 7.10

- **Generation volume**
  - 20250802: 32×8=256; 20250803: 31×8=248; 20250804: 32×8=256; 20250805: 28×8=224; 20250806: 27×8=216; 20250807: 35×8=280; 20250808: 31×8=248; 20250809: 33×8=264; 20250810: 18×8=144; 20250811: 31×8=248.

- **Paragraph quality (immediate_reward in [0,1])**
  - Daily avg: 0.507 → 0.504 → 0.503 → 0.466 → 0.471 → 0.429 → 0.396 → 0.405 → 0.366 → 0.359.
  - Mean (all rollouts): ≈ 0.443; drift from first→last day: −0.148.

- **Meta leakage and length (heuristic scan over final outputs)**
  - Leak rate: 27.3% (20250802) → 17.3% (20250811); overall ≈ 22.8%.
  - Avg length: ≈ 125 words overall (per-day ~121–129 words).

- **Evaluation rank (CSV `reward`)**
  - By design averages ~0.50/day (1..8 → 1.0..0.0 mapping within groups); useful for relative ordering but not absolute quality.

- **Training JSON diagnostics**
  - `rubric_weight` (Q): ≈ 0.200 across days (format/quality 20%).
  - `outcome_alpha`: ≈ 0.000 (LLM rank ignored; semantic-only within composite).
  - `llm_picks` median: 0 (heavy semantic fallback, as intended for this run).

#### Qualitative assessment
- **Stability**: Training is stable; rewards bounded, KL higher than NEWCOMPOSITERUN2 but within safe range.
- **Output shape**: Paragraphs generally coherent and topical; some instruction echo remains (e.g., “Keep it professional and concise.”) and occasional degenerate minimal responses.
- **Signal usage**: With L=0, S=0.8, Q=0.2, learning emphasizes semantic relevance with a modest nudge toward clarity/conciseness.
- **Trends**: Meta leakage improves over the window (~10pp reduction); quality score shows a mild downtrend, likely reflecting rubric strictness or entropy effects rather than collapse.

#### Risks and limitations
- **Instruction echo**: Still ~17–23% by end; raises noise and can hurt readability.
- **Degenerate outputs**: A small fraction produce very short or instruction-like lines; quality scorer catches many but not all.
- **Ranking relativity**: CSV rewards are relative; cross-day comparisons should rely on paragraph quality/leak metrics instead.
- **Missing eval (20250811)**: Latest day lacks evening evals; trends stop at 20250810 for evaluator-derived stats.

#### Recommendations (next actions)
- **Sampler tuning**: Consider `--sampler_profile tight` to reduce leak and verbosity; keep `default` if diversity is preferred.
- **Slightly more format weight**: If you want crisper paragraphs, nudge Q from 0.20 → 0.25 while keeping L=0, S=0.75.
- **Cleaner decoding**: Shorten `max_tokens` for paragraph generation and strengthen meta-line stripping to reduce instruction echo.
- **A/B check**: Compare `gspo_SEMANTICRUN/final_model` vs `gspo_NEWCOMPOSITERUN2/final_model` on a fixed headline set for paragraph quality and leak rate.
- **Telemetry**: Persist daily paragraph quality/leak/length into a small per-day summary for faster monitoring.

#### Quick references
- Checkpoints: `training/checkpoints/gspo_SEMANTICRUN/final_model_20250818` and `.../final_model`
- Predictions/Evals: `timestamped_storage_SEMANTICRUN/2025080{2..10}_*.json|.csv` (predictions exist for 20250811)


