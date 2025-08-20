### SEMANTICRUN_TIGHT_Q25 run report (2025-08-19)

#### Scope and setup
- **Run suffix**: `SEMANTICRUN_TIGHT_Q25`
- **Model**: `Qwen/Qwen3-0.6B` (MLX)
- **Trainer**: GSPO (response-only loss), 1 epoch across the backfill window
- **Weights (absolute, all buckets)**: LLM=0.00, Semantic=0.75, Format(Q)=0.25
  - Observed in training JSONs: `rubric_weight≈0.25`, `outcome_alpha≈0.00`, `llm_picks` median = 0.
- **Decoding**: `--sampler_profile tight` (lower entropy)
- **Paragraph prompt**: stricter plain-text paragraph spec + “Go.” exhortation (see `config/prompt_templates.json` key `paragraph_prompt`).

#### Completion and artifacts
- **Training**: Final checkpoint saved to `training/checkpoints/gspo_SEMANTICRUN_TIGHT_Q25/final_model_20250819`; latest at `.../final_model`.
- **Predictions**: 2025-08-02 → 2025-08-11 under `timestamped_storage_SEMANTICRUN_TIGHT_Q25/`.
- **Evaluations**: Present for 2025-08-02 … 2025-08-10; 2025-08-11 has predictions only at report time.

#### Quantitative summary
- **Trainer epoch** (`final_model/training_state.json`)
  - Steps: 2,136
  - Avg reward: 0.0441; range: 0.0000–0.1000; std: 0.0298
  - KL avg: 7.323; range: 0.0–45.0; std: 6.918

- **Generation volume** (headlines×8 rollouts)
  - 20250802: 32×8=256; 20250803: 31×8=248; 20250804: 32×8=256; 20250805: 28×8=224; 20250806: 27×8=216; 20250807: 35×8=280; 20250808: 31×8=248; 20250809: 33×8=264; 20250810: 18×8=144; 20250811: 31×8=248.

- **Paragraph quality (immediate_reward in [0,1])**
  - Daily avg: 0.641, 0.618, 0.625, 0.672, 0.671, 0.650, 0.637, 0.628, 0.646, 0.638
  - Overall avg (all rollouts): ≈ 0.643

- **Meta leakage (heuristic) and length**
  - Leak rate per day: 53.9%, 51.6%, 51.2%, 45.5%, 38.4%, 45.0%, 39.9%, 33.3%, 28.5%, 33.9%
  - Overall leak rate: ≈ 42.8%
  - Avg length: ≈ 136 words overall (per-day ≈ 133–138 words)

- **Evaluation rank (CSV `reward`)**
  - Averages ~0.50/day (relative within group; not a measure of absolute quality).

#### Qualitative assessment
- **Stability**: Training steady; KL between NEWCOMPOSITERUN2 (lower) and NEWCOMPOSITERUN (higher). No divergence.
- **Output shape**: Paragraphs are more assertive and specific; higher immediate quality suggests stronger adherence to the “start with the claim, include assets/direction/size/timeframe/driver” pattern.
- **Leakage**: Heuristic leak rate increased versus prior runs — many outputs still echo instruction-like phrases. Despite `tight` sampling and stricter prompt, the model frequently mirrors constraints.
- **Length**: Slightly longer paragraphs (~+10 words vs prior paragraph runs), consistent with richer detail; can be trimmed.

#### Risks and limitations
- **Instruction echo**: Elevated leak indicates the model often copies parts of the instruction (despite explicit “do not”). This is a known limitation at 0.6B scale under constrained prompts.
- **Heuristic bias**: Leak metric is conservative; some counted phrases may be benign. However, qualitative spot checks confirm visible echo in a material fraction of outputs.
- **Cross-day comparability**: CSV rank rewards remain relative; rely on paragraph quality + leak + length for absolute trends.

#### Recommendations (next actions)
- **Prompt nudge**: Remove the “Do not …” sentence (negative examples are often copied). Replace with positive constraints only; keep “Go.”
- **Sampler + length**: Keep `tight`; cap paragraph generation to ~180 tokens for the paragraph path to reduce ramble.
- **Cleaning**: Strengthen regex filtering for common echo phrases before quality scoring; add penalties for lines beginning with meta language.
- **Weights**: Keep L=0.00; try S=0.70 / Q=0.30 if you want crisper style; if leak remains high, nudge to Q=0.35 temporarily.
- **A/B**: Evaluate `gspo_SEMANTICRUN_TIGHT_Q25/final_model` vs earlier runs on a fixed headline set for paragraph quality, leak, and length.

#### Quick references
- Checkpoints: `training/checkpoints/gspo_SEMANTICRUN_TIGHT_Q25/final_model_20250819` and `.../final_model`
- Predictions/Evals: `timestamped_storage_SEMANTICRUN_TIGHT_Q25/2025080{2..10}_*.json|.csv` (predictions for 20250811)

