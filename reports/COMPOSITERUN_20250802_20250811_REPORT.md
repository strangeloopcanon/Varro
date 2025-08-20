## COMPOSITERUN Training Report (2025-08-02 to 2025-08-11)

### Scope
- Dates: 20250802–20250811
- Namespace: `timestamped_storage_COMPOSITERUN`
- Checkpoints: `training/checkpoints/gspo_<SUFFIX>/final_model[_<date>]` (e.g., `gspo_COMPOSITERUN`, `gspo_NEWCOMPOSITERUN`)
- Total evaluations: 2,384

### High-level outcome
- Composite reward present and non-zero across all days.
- LLM outcome scores recorded as zeros (current evaluator behavior), so avg_outcome_score/normalized are 0.0 in summaries.
- Training executed with 2,896 steps; latest `training_state.json` shows avg reward ≈ 0.0484 with KL tracked.

### Per-day evaluation summary (averages)
- 20250802: composite=0.225, trade_thinking=0.168, n=256
- 20250803: composite=0.243, trade_thinking=0.138, n=248
- 20250804: composite=0.221, trade_thinking=0.127, n=256
- 20250805: composite=0.222, trade_thinking=0.131, n=224
- 20250806: composite=0.251, trade_thinking=0.158, n=216
- 20250807: composite=0.232, trade_thinking=0.153, n=280
- 20250808: composite=0.196, trade_thinking=0.129, n=248
- 20250809: composite=0.199, trade_thinking=0.139, n=264
- 20250810: composite=0.194, trade_thinking=0.122, n=144
- 20250811: composite=0.211, trade_thinking=0.131, n=248

Overall averages:
- avg_composite_reward ≈ 0.220
- avg_trade_thinking_score ≈ 0.141
- avg_outcome_score = 0.0 (composite used as learning signal)

### Observations
- Composite reward stable around 0.20–0.25; modest peak on 20250806.
- Trade-thinking proxy low (0.12–0.17), consistent with concise one-line outputs.
- Evaluator summaries show score_distribution entirely in 0–3 bucket (reflects evaluator mapping rather than real accuracy). Effective reward optimized is composite.
- Training state:
  - steps: 2,896
  - avg_reward ≈ 0.0484 (trainer scale)
  - KL tracked; no instability indicated.

### Artifacts present
- Predictions: all dates present under `timestamped_storage_COMPOSITERUN/<date>_predictions.json`.
- Evaluations: all dates present under `timestamped_storage_COMPOSITERUN/<date>_evaluations.json`.
- Training JSONs: all dates present under `training/gspo_training_<date>.json`.
- Checkpoints: `training/checkpoints/gspo_<SUFFIX>/final_model_<date>` per day and rolling `final_model`.

### Recommendations
- Update evaluator to emit calibrated numeric outcome score (0–10); otherwise composite remains sole signal.
- Add few-shot examples; reduce temperature to tighten schema adherence and content quality.
- Keep composite reward with multiplicative penalty; continue tuning `alpha`, `gamma`, `w_meta`, `w_repeat`.
- Consider a small SFT bootstrap (500–2k curated one-line examples) to lift content before RL.

### Repro notes
- Aggregation: per-day averages over `composite_reward` and `trade_thinking_score` from `*_evaluations.json`; trainer stats from latest `training_state.json`.

### Quantitative analysis (post-run)
- Composite reward trend: slope ≈ -0.004 per day (slight drift down across the window).
- Daily metrics (valid_rate = fraction matching one-line schema; leak_rate = meta/instruction leakage rate; avg_len in words):
  - 20250802: comp=0.225, p50=0.158, p90=0.551, n=256, valid=0.000, leak=0.113, len=24.7
  - 20250803: comp=0.243, p50=0.257, p90=0.554, n=248, valid=0.000, leak=0.097, len=26.4
  - 20250804: comp=0.221, p50=0.178, p90=0.516, n=256, valid=0.004, leak=0.082, len=20.0
  - 20250805: comp=0.222, p50=0.209, p90=0.501, n=224, valid=0.000, leak=0.076, len=21.5
  - 20250806: comp=0.251, p50=0.209, p90=0.554, n=216, valid=0.000, leak=0.069, len=22.0
  - 20250807: comp=0.232, p50=0.178, p90=0.554, n=280, valid=0.000, leak=0.061, len=25.6
  - 20250808: comp=0.196, p50=0.000, p90=0.554, n=248, valid=0.000, leak=0.065, len=20.2
  - 20250809: comp=0.199, p50=0.000, p90=0.609, n=264, valid=0.000, leak=0.068, len=22.0
  - 20250810: comp=0.194, p50=0.128, p90=0.501, n=144, valid=0.000, leak=0.028, len=21.6
  - 20250811: comp=0.211, p50=0.128, p90=0.551, n=248, valid=0.000, leak=0.028, len=25.1
- Overall signal: composite is the main learning signal; valid_rate stayed ≈0 (strict schema adherence not learned), leak_rate improved from ~11% to ~3% by 20250810–20250811; content length remained concise (20–26 words average).

### Qualitative spot-check
- Many outputs still contain explanatory/meta preambles or placeholders; Domains/Proxies often off-list; claims sometimes restate headlines or include invented specifics.
- Compared to pre-run: modest reduction in meta leakage (confirmed by decreasing leak_rate), but no consistent shift toward falsifiable market claims or allowed proxy usage.

### What an engineer would want next
- Tighten generation controls: few-shot exemplars (2–3), lower temperature, and add stop at first newline.
- Strengthen reward: keep multiplicative composite; add hard 0/1 structure reward and increase penalty for meta leakage; optionally gate selection by validator.
- Curriculum: start with MarketMove-only headlines and allowed proxies to simplify target space.
- Bootstrap SFT: 500–2k curated one-line exemplars to establish schema and content prior to RL.
- Evaluator: emit calibrated 0–10 scores (not all zeros) to enable outcome correlation analyses.
