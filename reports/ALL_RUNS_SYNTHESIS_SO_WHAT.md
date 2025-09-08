### All Runs Synthesis & So‑What (latest)

#### Executive TL;DR
- Paragraph forecasting with a positive prompt, `LLM=0 / Semantic≈0.75 / Format(Q)≈0.25`, and `sampler_profile=tight` yields highest quality and lowest failure rates.
- Leakage persists due to scorer–detector misalignment; align penalties or pre-strip echoes to improve hygiene without sacrificing specificity.
- One-line strictness bottlenecks learning; needs validator-assisted decoding and validity-gated rewards to become useful.

#### Quantitative Recap (overall averages)
- Source CSV: `reports/CROSS_RUN_DAILY_METRICS_20250907.csv`
- COMPOSITERUN: quality=0.000, zeros=1.000, <0.2=1.000, leak=0.132, words=28.9
- NEWCOMPOSITERUN: quality=0.462, zeros=0.100, <0.2=0.132, leak=0.693, words=124.5
- NEWCOMPOSITERUN2: quality=0.242, zeros=0.432, <0.2=0.472, leak=0.708, words=120.8
- SEMANTICRUN: quality=0.441, zeros=0.116, <0.2=0.161, leak=0.708, words=123.8
- SEMANTICRUN_TIGHT_Q25: quality=0.643, zeros=0.013, <0.2=0.029, leak=0.200, words=129.2
- SEMANTICRUN_TIGHT_Q25_ARTICLES: quality=0.527, zeros=0.164, <0.2=0.184, leak=0.218, words=128.1

#### Defaults to Operationalize
- Paragraph: `LLM=0`, `Semantic≈0.75`, `Format(Q)≈0.25`, `sampler=tight`, `tokens≈160–180`; positive 3–5 sentence prompt with ‘Go.’
- Scorer alignment: penalize/strip echo stems; monitor quality/leak daily.
- One-line: sample-N + validator selection, retries, and validity-gated rewards.
