### Article‑Aware Validation (Aug 29 → Sep 7)

Goal
- Validate that our conclusions (tight paragraph prompting performs best) hold under a harder setting where prompts include a cleaned article excerpt in addition to the headline.

Setup
- Run suffix: `SEMANTICRUN_TIGHT_Q25_ARTICLES` (separate namespace).
- Dates: 20250829–20250907.
- Same recipe as tight baseline: `LLM=0 / Semantic≈0.75 / Format(Q)≈0.25 / sampler_profile=tight`.
- Morning attaches cleaned article excerpts when links match; evening stack‑ranks; night builds GSPO training JSON and trains.

Observations
- Completed end‑to‑end with predictions, evaluations, GSPO JSON, and checkpoints saved.
- Article excerpt coverage ≈66% overall (varies by day; some low‑coverage days due to link mismatches).
- Topic alignment remained strong (semantic consistency ≈0.93 avg vs next‑day headlines).
- Paragraph quality underperformed the headline‑only tight baseline; zero/very‑low shares were higher. Qualitative review shows prompt scaffolding echoes in some outputs.

So‑What
- Our core conclusion is robust: the tight paragraph recipe remains the best observed configuration under headline‑only prompting. Adding article excerpts did not improve measured paragraph quality without additional hygiene and better excerpt coverage.
- To make article‑aware prompting competitive, we need (i) higher excerpt matching/coverage and (ii) stricter paragraph hygiene (strip scaffold echoes, title quotes). With those in place, we can re‑assess over a shorter slice.

Pointers
- Storage: `timestamped_storage_SEMANTICRUN_TIGHT_Q25_ARTICLES/`
- Checkpoints: `training/checkpoints/gspo_SEMANTICRUN_TIGHT_Q25_ARTICLES/`
- Cross‑run CSV/plots include this run.

