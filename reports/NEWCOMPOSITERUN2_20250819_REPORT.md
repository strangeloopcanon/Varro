### NEWCOMPOSITERUN2 run report (2025-08-19)

#### Scope and setup
- **Run suffix**: `NEWCOMPOSITERUN2`
- **Model**: `Qwen/Qwen3-0.6B` (MLX)
- **Trainer**: GSPO (response-only loss), 1 epoch in this session
- **Evaluator/flow updates**: Deterministic A–H picker with semantic fallback; bucketed weighting exported to training JSONs (see `reports/TEMP_STATUS_NEWCOMPOSITERUN.md`). Trade-thinking disabled by default; composite-style rank reward drives learning.

#### Completion and artifacts
- **Training**: Completed; final checkpoint saved to `training/checkpoints/gspo_NEWCOMPOSITERUN2/final_model_20250818` and latest symlink at `training/checkpoints/gspo_NEWCOMPOSITERUN2/final_model`.
- **Predictions (covered)**: 2025-08-02 → 2025-08-11. Each day generated 8 rollouts per headline.
- **Evaluations**: Present for 2025-08-02 … 2025-08-10 (`timestamped_storage_NEWCOMPOSITERUN2/*_evaluations.(json|csv)`). 2025-08-11 has predictions only at report time.

#### Quantitative summary
- **Trainer epoch (from `final_model/training_state.json`)**
  - Total steps: 2,136
  - Avg reward: 0.0447; range: 0.0000–0.1000; std: 0.0310
  - Avg KL: 4.786; range: 0.0–34.5; std: 5.290

- **Generation volume**
  - 20250802: headlines=32, rollouts=256
  - 20250803: headlines=31, rollouts=248
  - 20250804: headlines=32, rollouts=256
  - 20250805: headlines=28, rollouts=224
  - 20250806: headlines=27, rollouts=216
  - 20250807: headlines=35, rollouts=280
  - 20250808: headlines=31, rollouts=248
  - 20250809: headlines=33, rollouts=264
  - 20250810: headlines=18, rollouts=144
  - 20250811: headlines=31, rollouts=248 (no eval yet)

- **Evaluation/quality (NEWCOMPOSITERUN2)**
  - Validity (strict one-line schema match): ≈ 0% across all days (regex against the target template).
  - Meta leakage (heuristic pattern scan over final outputs):
    - 20250802: 26.2% → trending down to 10.1% by 20250811; mean over 20250802–20250810 ≈ 14.5%.
  - Output length: overall average ≈ 123 words (per-day ≈ 119–128 words).
  - Rank-based reward (CSV `reward`): by construction averages ~0.50 per group/day; not directly informative about absolute quality.
  - Training JSON quality_score (proxy for immediate/format signal) — daily averages:
    - 20250802: 0.525; 20250803: 0.434; 20250804: 0.333; 20250805: 0.274; 20250806: 0.228; 20250807: 0.197; 20250808: 0.138; 20250809: 0.092; 20250810: 0.123.
  - Evaluator outcome fields in training JSONs average ≈ 0.50 (reflecting rank mapping); treat as cosmetic for now.

Notes:
- The evaluator exports bucket diagnostics (`llm_picks`, `bucket`, `outcome_alpha`, `rubric_weight`) into training JSONs. In this run, `llm_picks` recorded as 0 for most rows, indicating heavy reliance on semantic fallback under current constraints.

#### Qualitative assessment
- **Stability**: Training ran cleanly; rewards/KL bounded and consistent with prior sessions at similar scale.
- **Format adherence**: Final outputs almost never match the strict one-line schema; the model continues to produce paragraphs with instruction echo. This depresses the immediate/quality signal and undermines conversion of rank signals into useful targets.
- **Leakage**: Meta/instructional content is present but improved over the period (≈26% → ≈10%). Still materially higher than the best recent COMPOSITERUN references (~3%).
- **Content shape**: Outputs are long (≈120+ words) and repetitive; semantic consistency signals are present in training JSONs, but decoding remains too loose for the desired template.

#### Risks and limitations
- The average evaluator outcome reported in summaries (~0.5) is an artifact of rank→[1..8] mapping; it should not be interpreted as absolute quality.
- With validity ≈ 0%, rank-based training likely encodes stylistic artifacts from invalid strings; this can slow or misdirect learning.
- Heavy semantic fallback suggests the constrained picker frequently fails given current outputs; this limits exposure to stronger LLM ranking signals.

#### Recommendations (next actions)
- **Validity-first decoding**: Add constrained retries (stop at first newline, enforce `Domain=...; Proxy=...;` anchors); sample-N + validator selection; mild regex relaxation for near-miss acceptance.
- **Gate training by validity**: Set reward=0 for invalid outputs; use composite/rank-based signal only for valid strings. Optionally add a small validity bonus to bootstrap adherence.
- **Sampler tightening**: Lower temperature and reduce max tokens until valid rate >20%; gradually relax once structure sticks.
- **Few-shot scaffolding**: Prepend 2–3 exemplars of the exact line format to reduce instruction echo and verbosity.
- **Telemetry**: Persist per-day validity %, meta leak %, and fallback rate into a daily summary to make trend tracking obvious.
- **A/B check**: Evaluate `gspo_NEWCOMPOSITERUN2/final_model` vs the previous best `final_model` on a fixed headline set for leak rate and any uptick in valid structured lines.

#### Quick references
- Checkpoints: `training/checkpoints/gspo_NEWCOMPOSITERUN2/final_model_20250818` and `training/checkpoints/gspo_NEWCOMPOSITERUN2/final_model`
- Predictions (latest day): `timestamped_storage_NEWCOMPOSITERUN2/20250811_predictions.json`
- Evaluations (days present): `timestamped_storage_NEWCOMPOSITERUN2/2025080{2..10}_evaluations.json|.csv`


