### All Runs Synthesis & So‑What (2025‑08‑19)

#### Executive TL;DR
- Paragraph forecasting is the productive path at current scale. With a positive, concrete prompt, `LLM=0 / Semantic≈0.75 / Format(Q)≈0.25`, and `sampler_profile=tight`, we get the highest quality and the lowest failure rates by a wide margin.
- Leakage persists due to scorer–detector misalignment: the quality rubric upweights specificity while the leak heuristic flags meta echoes broadly. Align penalties or strip echoes pre‑score to bring quality and hygiene up together.
- One‑line strictness bottlenecks learning (near‑zero validity); it needs validator‑assisted generation and validity‑gated rewards before it can serve as a backbone signal.
- Entropy control (tight sampling and moderate KL) is leverage: fewer degenerate outputs, higher means, narrower tails.

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

