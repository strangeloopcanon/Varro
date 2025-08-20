# Evaluation Fallback Incident (Aug 2–11, 2025)

Summary: During evening evaluations, the LLM frequently failed to output a single-letter choice (A–H). The constrained retry then biased selections heavily toward the first available option (A), creating near-linear rankings. This undermines evaluation signal quality and leaks ordering bias into rewards and training.

Scope
- Dates reviewed: 2025-08-02 to 2025-08-11
- Namespaces: `timestamped_storage_<SUFFIX>/` (e.g., `COMPOSITERUN`, `NEWCOMPOSITERUN`)
- Logs scanned: `training/logs/<SUFFIX>_*_evening.log`

Symptoms
- Very high counts of "No valid letter found" warnings (typically 5 per round).
- Selections skew heavily to letter `A` (first option), indicating fallback-driven ordering.
- Few to no groups dropped for insufficient rankings, meaning fallback masked the failure by forcing progress.

Observed Metrics
- COMPOSITERUN (10 evening logs)
  - Groups evaluated: 298
  - Total rounds recorded: 2,303 (vs. 2,384 expected → ~3.4% short)
  - Warnings ("No valid letter found"): 9,345
  - Letter selection distribution: A=1,784 (77%), C=313; B=66; D=38; E=46; F=26; G=26; H=4
  - Typical per-day pattern (e.g., 20250802): groups=32, rounds=224, warnings=1,120 (exactly 5 retries/round), A-rate≈0.86
- NEWRUN (6 evening logs)
  - Groups evaluated: 185
  - Total rounds recorded: 1,674
  - Warnings: 4,627
  - Letter selection distribution: A=1,240 (74%), C=206; B=148; D/E/F/G sparse
  - Some later days show improvement (e.g., 20250806 A≈0.68; 20250807 A≈0.58) but bias persists.

Impact
- Rankings for many headline groups are driven by fallback position rather than true LLM preference.
- Rewards derived from ranks (1.0 → 0.0) become noisy and biased, weakening training signal quality.

Root Cause
- The evaluator first uses an unconstrained prompt and often fails to emit a valid single letter.
- The constrained retry—intended as a recovery—frequently produces `A`, and the final hard fallback chooses the first available option, reinforcing linear bias.

Remediation (Implemented)
- Change evaluator to use a constrained, single-letter prompt first.
- If any selection fails (no valid letter), abort LLM selection for the whole group and switch to a deterministic, non-LLM fallback ranking based on semantic consistency between forecast and due-day headlines (TF–IDF cosine-like similarity), assigning ranks 1..8 by descending similarity.
- Do not use the "pick first available" fallback; prefer skipping to deterministic ranking to avoid ordering bias.

Next Steps
- Re-run evening evaluations after patch (with `--force_evaluations`) for new runs (e.g., `NEWCOMPOSITERUN`).
- Spot-check logs: selection letter distribution should no longer be dominated by `A`; deterministic fallback should be rare and explicitly logged.
- Consider adding a metric to summaries: percent of groups using deterministic fallback.

Files/Paths Referenced
- Logs: `training/logs/*_evening.log`
- Evaluator: `outcome_tracking/llm_outcome_evaluator.py`
- Semantic similarity helper: `outcome_tracking/evaluation_storage.py` (`_semantic_consistency`)

