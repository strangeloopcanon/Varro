# Next-Step Research Checklist

This file records the high-priority tasks that will bring the paper and codebase from “good” to camera-ready.

## 1  Baseline & Ablation Experiments

- [ ] Train **Supervised-MLE** baseline on the 622 scored roll-outs; log reward, hit-rate, KL.  
- [ ] Train **KTO offline-RL** baseline on the same data; collect identical metrics.  
- [ ] Run **reward-mix ablation** (structure-only, outcome-only, 30/70 mix, 70/30 mix) and populate a comparison matrix.

## 2  Economic Back-Test

- [ ] Parse the “Trade Recommendation” sections into long/short calls on ETF proxies (SPY, TLT, GLD, USO).  
- [ ] Compute cumulative return, Sharpe ratio, and max draw-down versus buy-and-hold and zero-shot Qwen.  
- [ ] Add equity curve figure to `paper/figs/` and Discussion section.

## 3  Robustness & Error Analysis

- [ ] Re-score Day-1 roll-outs with the *latest* evaluator to isolate model improvements.  
- [ ] Manually tag ~50 lowest-reward examples; classify error types and create a pie chart + qualitative table.

## 4  30-Day Extended Run

- [ ] Let the daily pipeline run for three additional weeks; retrain GSPO on the enlarged dataset.  
- [ ] Update all figures and statistics to reflect ~5 k examples; examine learning-curve saturation.

## 5  Figure & Paper Polish

- [ ] Export Mermaid architecture diagram as `paper/figs/architecture.png` via mermaid-cli.  
- [ ] Replace placeholder BibTeX keys with real citations.  
- [ ] Trim manuscript to venue page limit; move overflow to appendix.

## 6  Reproducibility & Release

- [x] Publish a small CC-BY sample dataset under `data/sample/`.
- [ ] Create a Colab notebook that recreates one GSPO epoch and regen figures.  
- [ ] Add acknowledgements, funding & license statements.

---
Maintainers: check off items as they complete them so future contributors (human or LLM) can immediately see project status.
