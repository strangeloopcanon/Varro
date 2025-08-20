### COMPOSITERUN training backfill report (20250802–20250811)

#### Scope and artifacts
- **Namespace**: `timestamped_storage_COMPOSITERUN/`
- **Checkpoints**: `training/checkpoints/gspo_COMPOSITERUN/`
- **Logs**: `training/logs/`
- Completed pipeline for 10 days with predictions, evaluations, GSPO data, and per‑day model snapshots.

#### Completion status
- All days processed; per‑day logs present without fatal errors.
- Checkpoints:
  - **Latest**: `training/checkpoints/gspo_COMPOSITERUN/final_model`
  - **Snapshots**: `final_model_20250802 … final_model_20250811`

#### Quantitative results
- **Totals**: 298 headlines, 2,384 rollouts, 2,384 evaluations.
- **Averages (period)**: composite reward ≈ 0.225; trade‑thinking ≈ 0.147; validity (avg_immediate_reward) ≈ ~0%.
- **Sampler profile**: tight for all days (no auto‑relaxation observed).

| Date | Headlines | Rollouts | Sampler | Validity (avg_immediate_reward) | Composite reward | Trade-thinking | Evaluations |
|---|---:|---:|---|---:|---:|---:|---:|
| 20250802 | 32 | 256 | tight | 0.0000 | 0.2933 | 0.2188 | 256 |
| 20250803 | 31 | 248 | tight | 0.0000 | 0.2498 | 0.1565 | 248 |
| 20250804 | 32 | 256 | tight | 0.0000 | 0.2372 | 0.1516 | 256 |
| 20250805 | 28 | 224 | tight | 0.0000 | 0.2560 | 0.1652 | 224 |
| 20250806 | 27 | 216 | tight | 0.0000 | 0.2284 | 0.1375 | 216 |
| 20250807 | 35 | 280 | tight | 0.0000 | 0.2206 | 0.1471 | 280 |
| 20250808 | 31 | 248 | tight | 0.0000 | 0.1921 | 0.1371 | 248 |
| 20250809 | 33 | 264 | tight | 0.0000 | 0.1888 | 0.1174 | 264 |
| 20250810 | 18 | 144 | tight | 0.0000 | 0.1759 | 0.1069 | 144 |
| 20250811 | 31 | 248 | tight | 0.0040 | 0.2110 | 0.1315 | 248 |

Notes:
- Predictions summaries show per‑day sampler usage: all “tight”.
- Evaluations summaries report `avg_outcome_score` and `avg_normalized_score` as 0.0 (see limitations below). Composite reward and trade‑thinking metrics are populated and reliable.

#### Trends and observations
- **Composite reward trend**: Down from ~0.293 (20250802) toward ~0.176 (20250810) with a small rebound on 20250811 (~0.211).
- **Validity**: Near‑zero across days; even with tight sampling and a single retry, the one‑line schema rarely matches.
- **Auto‑profile**: No evidence of relaxing beyond “tight”, consistent with low measured validity and leak heuristics.
- **Diversity**: Tight sampling plus low validity likely constrained exploration; stack‑ranking signal may be learning on low‑diversity, often invalid responses.

#### Qualitative assessment
- **Format adherence** is the main bottleneck. The validator and retry path are strict, causing almost all outputs to be marked invalid, starving the system of a strong “validity” signal and leaving auto‑profile stuck in “tight”.
- **Reward shaping** weights composite (stack‑ranking) but ignores validity in training targets. Learning from invalid strings risks encoding noise or brittle patterns.
- **Evaluator reporting mismatch**: evaluations write `reward`/`composite_reward`, but summaries compute `avg_outcome_score` and `avg_normalized_score` from absent fields, yielding 0.0 (cosmetic, but confusing). Composite and trade‑thinking are correct.
- **Model/entropy balance**: Qwen3‑0.6B plus tight decoding can under‑explore; without a validity gate, improvements don’t propagate.

#### Risks and limitations
- Reported `avg_outcome_score` and `avg_normalized_score` are zeros due to field mismatch; rely on `avg_composite_reward` and `avg_trade_thinking_score` instead.
- Validity metric (avg_immediate_reward) shows near‑zero; this is a true pipeline limitation, not just reporting.

#### Recommendations
- **Improve validity generation**
  - Add 2–3 iterative constrained retries with slot filling when format fails; accept near‑misses via slightly relaxed regex.
  - Beam/rerank or sample‑N with a validity scorer to select a compliant line.
- **Incorporate validity into training targets**
  - Gate reward: set reward=0 for invalid responses; otherwise use composite.
  - Optionally blend a small validity bonus to bootstrap adherence.
- **Adjust diversity policy**
  - Keep day‑1 “tight,” but allow auto‑profile to move to “default” when validity ≥ 10% and leak ≤ 10%.
  - A/B within day: run a subset of headlines with “default” to recover diversity; compare composite and leak.
- **Fix evaluation summaries**
  - When saving evaluations, mirror `reward` into `outcome_score` and `normalized_score`, or update the summary reader to use `reward` if present. This will restore meaningful averages.
- **Telemetry and observability**
  - Produce a per‑day run summary (validity %, composite, trade‑thinking, sampler) and a run‑level report file.
  - Track training step stats (loss/avg reward) per day for trend lines.

#### Artifact references (examples)
- Predictions: `timestamped_storage_COMPOSITERUN/2025080x_predictions.json`
- Evaluations: `timestamped_storage_COMPOSITERUN/2025080x_evaluations.json`
- Checkpoints: `training/checkpoints/gspo_COMPOSITERUN/final_model*`
- Logs: `training/logs/COMPOSITERUN_*`


