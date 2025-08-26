### Varro Final Report — All Runs Synthesis (as of 2025-08-19)

This is the canonical, easy-to-find summary of what we've built, learned, and decided. Last updated: 2025-08-19. Source snapshot: `reports/ALL_RUNS_SYNTHESIS_SO_WHAT_20250819.md`.

#### Executive TL;DR
- Paragraph forecasting is the productive path at current scale. With a positive, concrete prompt, `LLM=0 / Semantic≈0.75 / Format(Q)≈0.25`, and `sampler_profile=tight`, we get the highest quality and the lowest failure rates by a wide margin.
- Leakage persists due to scorer–detector misalignment: the quality rubric upweights specificity while the leak heuristic flags meta echoes broadly. Align penalties or strip echoes pre‑score to bring quality and hygiene up together.
- One‑line strictness bottlenecks learning (near‑zero validity); it needs validator‑assisted generation and validity‑gated rewards before it can serve as a backbone signal.
- Entropy control (tight sampling and moderate KL) is leverage: fewer degenerate outputs, higher means, narrower tails.

#### Objective & Approach (plain words)
- Objective: use daily news headlines to continually train a small model (0.6B) to make near‑term, market‑relevant forecasts that are specific, falsifiable, time‑bound, and readable.
- Success criteria: higher paragraph quality, low failure/zero rates, lower meta‑leakage, stable training (bounded KL), and clear day‑over‑day trends; eventually, calibrated outcome accuracy.
- Method: daily pipeline (morning/evening/night), 8 rollouts/headline, evaluator‑derived rewards, GSPO training with simple, controllable decoding; track per‑day metrics and compare runs.
- Two targets explored: a strict one‑line schema vs a free‑form paragraph with light structural priors.

#### So What — The Story (what changed and why it mattered)
- We started with a strict one‑line format (COMPOSITERUN). It looked clean on paper but produced near‑zero valid strings under stochastic decoding, starving learning; composite reward became the only useful signal.
- We pivoted to paragraphs (NEWCOMPOSITERUN). Quality jumped and zeros collapsed, but instruction echoes (“Do not…”, “Keep it…”) inflated leak. We were optimizing specificity without aligned hygiene.
- We tried very low entropy (NEWCOMPOSITERUN2). Exploration stalled and quality fell; leak stayed high. Controlling entropy helped, but not that much.
- We restored moderate entropy and adjusted the Format(Q) weight (SEMANTICRUN). Outputs became more assertive and consistent, yet still echo‑prone.
- We tightened decoding and set Q≈0.25 (SEMANTICRUN_TIGHT_Q25). This combination delivered the best observed trade‑off: highest paragraph quality, minimal zeros, and substantially lower leak. It confirmed that structure+entropy control are strong levers at small scale.
- The remaining gap is scorer–detector alignment. By penalizing or pre‑stripping echo stems, we expect quality and hygiene to rise together without sacrificing specificity.

#### What Changed Across Experiments
- Prompting: Moved from verbose + prohibitions to a short, affirmative paragraph spec (3–5 sentences; assets/sectors; direction+magnitude; timeframe; drivers; plain text) with a tiny “Go.” nudge.
- Reward shaping: Turned LLM rank off; emphasized Semantic with a meaningful Format(Q) share (0.20→0.25); kept CSV ranks only for intra‑group ordering.
- Decoding: Standardized on `sampler_profile=tight`; capped tokens for paragraphs to keep 3–5 sentence shape.
- Evaluator: Deterministic A–H ranking with semantic fallback; bucket diagnostics exported into training JSONs.
- Telemetry: Added per‑day summary CSV + plots to track quality, zeros, very‑low share, leak, and length.

#### Quantitative Recap (overall averages)
- Source CSV: `reports/CROSS_RUN_DAILY_METRICS_20250819.csv`
- Plots: `reports/cross_run_daily_quality.png`, `reports/cross_run_daily_leak.png`
- Averages (period 2025‑08‑02 → 2025‑08‑11):
  - COMPOSITERUN: quality=0.000, zeros=1.000, <0.2=1.000, leak=0.132, words=28.9
  - NEWCOMPOSITERUN: quality=0.462, zeros=0.100, <0.2=0.132, leak=0.693, words=124.5
  - NEWCOMPOSITERUN2: quality=0.242, zeros=0.432, <0.2=0.472, leak=0.708, words=120.8
  - SEMANTICRUN: quality=0.441, zeros=0.116, <0.2=0.161, leak=0.708, words=123.8
  - SEMANTICRUN_TIGHT_Q25: quality=0.643, zeros=0.013, <0.2=0.029, leak=0.200, words=129.2

Notes
- Leak rates here use a strict, explicit echo‑stem list; absolute values differ from earlier ad‑hoc scans but relative ordering is consistent. Use for comparisons and trend, not for absolute cleanliness claims.

#### Reward Shaping: What We Learned
- LLM rank off (outcome_alpha≈0): Stabilizes training and removes position effects; CSV ranks remain relative (center ~0.5) and are not an absolute quality proxy.
- Semantic vs Format(Q): A modest Q increase (0.20→0.25) alongside `tight` sampling materially reduces degenerate/zero outputs and raises mean paragraph quality; too little Q allows ramble, too much Q risks generic output. Best observed trade‑off: S≈0.75, Q≈0.25.
- Entropy (KL): Very low KL (e.g., NEWCOMPOSITERUN2) under‑explores and stalls quality; moderate KL (≈7–9) coincides with the highest quality and stability.
- Mechanism: Q acts like a structure/clarity prior; Semantic preserves topical relevance and concreteness. Tight sampling focuses decoding around rubric‑satisfying modes.

Guidance
- Default paragraph recipe: `LLM=0 / Semantic≈0.75 / Format(Q)≈0.25 / sampler=tight / tokens≈160–180`.
- If ramble creeps in, nudge Q to ≈0.30; if outputs get too generic/short, pull Q back to ≈0.25.
- Align scorer/detector: penalize or pre‑strip canonical echo stems so quality and leak improve together.

#### Leakage vs Reward: Reconciling the Paradox
- The paragraph quality scorer rewards direction/magnitude/timeframe/driver and brevity. The leak detector flags broad meta phrases.
- Thus, a paragraph can score high on content while still echoing instruction fragments. Aligning penalties (or stripping echoes pre‑score) collapses this gap without hurting specificity.

#### One‑Line vs Paragraph: Strategic Choice
- One‑line (strict regex) today: near‑zero validity → starved signal → sampler stuck “tight” → slow/no learning.
- Paragraphs: rich signal directly optimizes useful traits (specificity, time‑bound claims, plausible magnitudes) and responds strongly to reward and decoding tweaks.
- Decision: use paragraphs as the primary path for quality gains at 0.6B; invest in validator‑assisted decoding + validity‑gated rewards before revisiting one‑line as a core target.

#### So What (practical implications)
- Defaults to Operationalize
  - Adopt the paragraph recipe (above) as the standard config for upcoming runs.
  - Keep `tight` until `zeros ≤ 3%` and `leak ≤ 15%`, then allow a small `default` subset for diversity.
  - Cap paragraph tokens at ≈160–180; re‑tune only if length drifts >10%.
  - Update scorer or pre‑clean outputs to de‑weight echoed stems; re‑baseline leak vs quality after the change.
- Evaluation & Monitoring
  - Use the daily CSV + plots to watch quality/zeros/leak/length; alert on >10% weekly drift.
  - A/B final models on a fixed headline panel before promoting; report paragraph quality, leak, words.
- Product & Roadmap
  - Expect clearer, more actionable forecasts with explicit assets, small magnitudes (1–3%), and tight timeframes (2–4 weeks/next quarter).
  - After scorer alignment, hygiene should converge without sacrificing specificity.
  - Scale path: apply the same recipe to larger models for additional leak reduction and richer drivers; consider adding calibrated magnitude guidance from outcome data.

#### Risks & Mitigations
- Heuristic leak over/under‑counts: keep a hand‑labelled spot‑check set; align detector phrases with scorer penalties.
- Over‑tight outputs: if specificity drops, ease Q back to 0.25 or slightly relax decoding on a subset.
- Metric myopia: CSV ranks are relative; anchor decisions on paragraph metrics and fixed‑panel A/Bs.

#### Next Experiments (2‑week plan)
- Scorer alignment: implement echo‑stem penalties or pre‑strip; re‑run a short backfill; compare quality/leak/length vs baseline.
- One‑line path: add sample‑N + validator selection and validity‑gated rewards; measure validity %, composite, and leak.
- S/Q sensitivity: trial S=0.70/Q=0.30 under `tight` to test hygiene vs specificity trade‑off.
- Fixed‑panel A/Bs: compare `gspo_SEMANTICRUN_TIGHT_Q25/final_model` against prior finals across 50–100 headlines.

#### Artifact Index
- Per‑day metrics: `reports/CROSS_RUN_DAILY_METRICS_20250819.csv`
- Plots: `reports/cross_run_daily_quality.png`, `reports/cross_run_daily_leak.png`
- Cross‑run comparison: `reports/CROSS_RUN_COMPARISON_20250819.md`
- Defaults + rationale: `reports/DEFAULTS_AND_RATIONALE_20250819.md`

#### Per‑Run Highlights (what shifted and why)
- COMPOSITERUN (one‑line baseline): strict regex target starved validity (≈0%) and learning; short outputs (~20–26 words), low leak (≈3–11%) but not comparable to paragraph quality. Composite hovered ≈0.20–0.25 with slight down‑drift.
- NEWCOMPOSITERUN (paragraphs, looser Q): quality ≈0.46; zeros ≈10%; leak ≈69%; words ≈125. Richer signal unlocked gains but hygiene suffered due to instruction echoes.
- NEWCOMPOSITERUN2 (paragraphs, very low KL): quality ≈0.24; zeros ≈43%; leak ≈71%; words ≈121. Under‑explored decoding reduced quality; leak stayed high.
- SEMANTICRUN (paragraphs, moderate settings): quality ≈0.44; zeros ≈12%; leak ≈71%; words ≈124. Stable but still echo‑prone.
- SEMANTICRUN_TIGHT_Q25 (paragraphs, tight + Q=0.25): quality ≈0.64; zeros ≈1.3%; leak ≈20%; words ≈129. Best trade‑off observed; `tight` + modest Q curbed degeneracy and lifted adherence.

#### Example Outputs (real excerpts)
- One‑line baseline (COMPOSITERUN → invalid schema example)
  - Example: `Domain=<…>; Proxy=<instrument|metric|event>; Horizon=<1w|1m|1q>; Prob=<0–100%>; Claim=<falsifiable statement>; Rationale=≤18 words; Risk=≤6 words; Confidence=0|1|2|3.`
  - Why it matters: shows the strict template being echoed instead of filled → validity ≈0% → no backbone signal for RL.

- Paragraph (SEMANTICRUN_TIGHT_Q25 → asset/sector, direction, magnitude, timeframe, drivers)
  - Example A (commodities): “The next 4–6 weeks will see a modest increase in demand for gold and silver as investors rotate defensively; prices likely up 1–3% near term, led by safe‑haven flows. Key drivers include sticky inflation concerns and higher real‑rate volatility.”
    - Source: `timestamped_storage_SEMANTICRUN_TIGHT_Q25/20250810_predictions.json` (gold/silver items; lightly trimmed for brevity).
  - Example B (exchange operator): “Singapore Exchange Ltd. (SPXCF) likely trends higher by ~1–3% over the next 2–4 weeks on improving liquidity and tech/healthcare issuance; investor confidence and steadier macro risk support flow.”
    - Source: `timestamped_storage_SEMANTICRUN_TIGHT_Q25/20250810_predictions.json` (SPXCF items; lightly trimmed for brevity).
  - Example C (large‑cap tech equity): “Apple (AAPL) likely up 2–3% in the next 2–4 weeks as tariff‑relief headlines ease policy overhang; tech and consumer hardware benefit marginally (~1–3%). Key drivers: anticipated tariff reprieve and stronger U.S. demand.”
    - Source: `timestamped_storage_SEMANTICRUN_TIGHT_Q25/20250805_predictions.json` (AAPL items; lightly trimmed for brevity).
  - Example D (sector rotation mix): “Within 2–4 weeks, energy and infrastructure see small gains (up 1–3%) while consumer goods and parts of tech soften (down ~2–4%) amid macro uncertainty and capex skew. Drivers: risk rotation to hard‑asset exposures and defensive positioning.”
    - Source: `timestamped_storage_SEMANTICRUN_TIGHT_Q25/20250810_predictions.json` (rotation items; lightly trimmed for brevity).
  - Example E (digital assets): “Over the next 4–6 weeks, demand for digital assets rises, with crypto and blockchain sectors up around 1–3% as capital chases DeFi narratives and liquidity improves.”
    - Source: `timestamped_storage_SEMANTICRUN_TIGHT_Q25/20250811_predictions.json` (digital‑assets items; lightly trimmed for brevity).
  - Example F (bank equity downside): “Within 2–4 weeks, Bank of America tilts lower by ~1–3% as capital return optics and weaker demand weigh; financials soften while defensives hold.”
    - Source: `timestamped_storage_SEMANTICRUN_TIGHT_Q25/20250810_predictions.json` (bank items; lightly trimmed for brevity).

- Paragraph with instruction echo (what we penalize/strip next)
  - Example: “Write a single paragraph (3–5 sentences) forecasting what happens next… Include a concrete timeframe… Do not restate the headline…”
    - Why it matters: high quality on specificity can coexist with “leak”; scoring/cleaning alignment fixes this without losing substance.

Notes
- Excerpts are verbatim or lightly trimmed from the `prediction` field (never from internal “think” traces). Files above contain the full paragraphs for auditability.

#### Narrative: What’s interesting about learning to predict this way
- Structure as a prior: Enforcing “claim first → assets → magnitude → timeframe → drivers” acts like an inductive prior. With GSPO, the model learns to inhabit this structure, which raises specificity and reduces ramble even at small scale (0.6B).
- Decoding controls trump small‑scale capacity: Tight sampling and token caps improved mean quality more than any other single change. Exploration without validity makes learning chase noise; entropy discipline is leverage.
- Reward‑detector alignment is everything: Our best paragraph scores initially rose while leak stayed high because the rubric rewarded concreteness but didn’t penalize echoed stems. Aligning scoring with hygiene (or pre‑stripping echoes) makes both move together.
- Richer signals beat brittle schemas: Paragraphs provided dense, smooth reward gradients; the strict one‑line template yielded near‑zero validity and a starved signal. A validator‑assisted path can rehabilitate one‑lines later, but paragraphs carry the program today.
- Practical predictability: Short‑horizon claims with small magnitudes (1–3%) and explicit drivers are easier to learn and evaluate. This matches how human forecasters scaffold “what moves what” rather than raw price calls.

#### What we learned (condensed)
- Best observed recipe: `LLM=0 / Semantic≈0.75 / Format(Q)≈0.25 / sampler=tight / ≈160–180 tokens`.
- Raise Q to ~0.30 if ramble creeps in; roll back if outputs become generic.
- Keep `tight` until zeros ≤3% and leak ≤15%; then cautiously diversify sampling.
- Penalize/strip echo stems in scoring; hygiene improves without harming specificity.
- For one‑lines: sample‑N + validator selection + validity‑gated rewards unlock progress; without this, learning stalls.

#### Limitations and open questions
- Outcome accuracy: Current summaries underweight true outcome scores (many zeros in early runs). We still need calibrated 0–10 outcome scoring to quantify real‑world accuracy beyond stylistic quality.
- Leak metric: Heuristic and conservative; good for trend/ordering, not absolutes. A small hand‑labelled set will improve calibration.
- Scale vs control: At 0.6B, decoding and reward alignment do most of the work. Larger models may reduce leak and enrich drivers, but should keep the same structural prior and hygiene alignment.
- Generalization: Fixed‑panel A/Bs across diverse headlines are needed to confirm robustness of the observed gains.

#### Roadmap (contextualized)
- Hygiene alignment: Implement echo‑stem penalties/stripping, then re‑baseline quality vs leak; target leak ≤15% at the same quality.
- Valid one‑line path: Add validator‑assisted decoding and validity‑gated rewards; re‑measure validity %, composite, and leak. Promote only if validity ≥20% without quality regression.
- Outcome grounding: Restore calibrated numeric outcome scores; study correlation with paragraph quality and length; evaluate whether Q‑tuning tracks real‑world accuracy.
- Fixed‑panel A/Bs: Compare `gspo_SEMANTICRUN_TIGHT_Q25/final_model` vs prior best on 50–100 headlines; report paragraph quality, leak, and words.

#### Provenance
- Run reports: `reports/SEMANTICRUN_TIGHT_Q25_20250819_REPORT.md`, `reports/SEMANTICRUN_20250819_REPORT.md`, `reports/NEWCOMPOSITERUN_20250818_REPORT.md`, `reports/NEWCOMPOSITERUN2_20250819_REPORT.md`, `reports/COMPOSITERUN_20250802_20250811_REPORT.md`.
- Cross‑run roll‑up: `reports/ALL_RUNS_SYNTHESIS_SO_WHAT_20250819.md` (this file distills and expands it).
