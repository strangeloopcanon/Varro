# Varro: Training a Small Model to Forecast from Daily News

## Abstract
We study a daily pipeline that teaches a 0.6B model to produce short‑horizon, market‑relevant forecasts from news headlines. We compare strict one‑line schemas to structured paragraphs and evaluate design levers: prompting, decoding entropy, and reward shaping. Our best observed recipe uses a concise paragraph prompt with tight sampling and a blended reward dominated by semantic consistency and a modest format quality term.

## Methods (Brief)
- Morning: Collect headlines, generate 8 paragraph forecasts per headline (tight sampler; ~160–180 tokens), optionally attach cleaned article excerpts.
- Evening: Build outcome groups and stack‑rank 8 rollouts; fall back to deterministic semantic ranking when constrained selection is invalid.
- Night: Construct GSPO training JSON (quality + outcome blend) and train step‑by‑step.

## Results (Prior Headline‑Only Baselines)
- Paragraph prompting with `LLM=0 / Semantic≈0.75 / Format(Q)≈0.25` and `sampler_profile=tight` delivered the highest paragraph quality and lowest failure rates among tested settings.
- One‑line strictness bottlenecked progress due to near‑zero validity without validator‑assisted decoding.

## Article‑Aware Validation (Aug 29 → Sep 7)
We stress‑tested our best headline‑only setting by augmenting prompts with a cleaned article excerpt when available.

- Run suffix: `SEMANTICRUN_TIGHT_Q25_ARTICLES`. Dates: 20250829–20250907.
- System robustness: Completed end‑to‑end with predictions, evaluations, training JSONs, and checkpoints saved.
- Excerpt coverage: ≈66% overall; some days were lower due to link canonicalization mismatches.
- Topic alignment: High semantic consistency vs next‑day headlines (≈0.93 avg).
- Paragraph quality: Underperformed the headline‑only tight baseline; zero/very‑low shares were higher. Qualitative inspection shows prompt scaffolding echoes (e.g., “Write the forecast paragraph.”) slipping through paragraph cleaning.

So‑What: Our core conclusion is robust in the harder setting—tight paragraph prompting remains the best‑performing recipe under headline‑only inputs. Adding article excerpts does not improve measured paragraph quality without better excerpt matching and stricter paragraph hygiene. The program still “works” (remains topically aligned and trains), but to claim gains in this richer input regime we must raise excerpt coverage and tighten paragraph cleaning.

## Discussion
- Decoding controls and reward alignment remain primary levers at small scale.
- For article‑aware prompting, hygiene must be stricter to prevent scaffold echoes from degrading measured quality.
- Future: Normalize links and implement fallback (domain+title) match; extend paragraph cleaner to strip echoed scaffolding, then reassess over a 3‑day slice.

## Data & Artifacts
- Storage: `timestamped_storage_<SUFFIX>/` (article run: `SEMANTICRUN_TIGHT_Q25_ARTICLES`).
- Checkpoints: `training/checkpoints/gspo_<SUFFIX>/`.
- Cross‑run reports: `reports/CROSS_RUN_DAILY_METRICS_*.csv`, plots, and comparison/synthesis MDs.

