### Default Settings + Rationale (Paragraph + One‑Line)

Scope: Default recipes for paragraph forecasting and, separately, the one‑line schema, plus a short rationale grounded in observed metrics across runs (2025‑08‑02 → 2025‑08‑11).

#### Paragraph Forecasting — Defaults
- Prompt: 3–5 sentences; name 2–3 assets/sectors; direction + rough magnitude; concrete timeframe; 1–2 key drivers; plain text only; end with a short “Go.” exhortation.
- Weights: LLM rank off; Semantic ≈ 0.75; Format(Q) ≈ 0.25.
- Decoding: `sampler_profile=tight`; cap `max_tokens` ≈ 160–180 for the paragraph path.
- Scorer alignment: Penalize or strip canonical echo stems (e.g., “Keep it…”, “Do not…”, “Answer:”, “Note:”, “Start with…”) before/within quality scoring.
- Monitoring: Persist daily `quality_avg`, `zeros_share`, `<0.2_share`, `leak_share`, `avg_words`; keep `tight` until zeros ≤ 3% and leak ≤ 15%, then consider limited `default` sampling on a subset.

#### One‑Line Schema — Defaults (when re‑enabling)
- Constrained decoding: sample‑N + validator selection; stop at first newline; 1–3 guided retries with slot anchoring; accept near‑misses with mild regex relaxation.
- Reward gating: Set reward=0 for invalid lines; use composite/semantic signals only for valid strings; optionally add a small validity bonus.
- Decoding: Start `sampler_profile=tight`; allow auto‑relax to `default` when validity ≥ 10% and leak ≤ 10%.
- Telemetry: Track per‑day validity %, composite, trade‑thinking, sampler transitions.

#### Rationale (why these defaults)
- Positive prompt > prohibitions: Negative instructions were echoed (leak); short, affirmative specs with “Go.” improved adherence and scores.
- Semantic + Format(Q): A modest Q increase (0.20→0.25) with `tight` decoding materially reduced zeros/very‑low cases and raised mean quality without washing out specificity.
- LLM rank off: Removed positional artifacts, stabilized training; CSV ranks are relative and not an absolute quality proxy.
- Entropy control: `tight` narrows the distribution around rubric‑satisfying modes, boosting quality and shrinking tails; moderate KL (≈7–9) coincided with best outcomes.
- Scorer–detector alignment: High quality coexisted with “leak” because penalties were mild and the detector broad; penalizing/stripping echo stems makes quality and hygiene move together.
- One‑line specifics: Strict regex without retries/starving validity taught on invalid strings; validator‑assisted decoding and validity‑gated rewards unlock progress.

#### Quick Links
- Daily metrics CSV: `reports/CROSS_RUN_DAILY_METRICS_20250819.csv`
- Comparison report: `reports/CROSS_RUN_COMPARISON_20250819.md`
- Plots: `reports/cross_run_daily_quality.png`, `reports/cross_run_daily_leak.png`

