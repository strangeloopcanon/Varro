# Sample Data Directory

This directory contains a small synthetic sample dataset (Apache‑2.0, same as the repo) to illustrate Varro’s file formats.

## Contents

- `20250101_headlines.json`: One example financial headline.
- `20250101_predictions.json`: Eight sample rollouts for the headline.
- `20250101_outcome_tracking.json`: Next-day outcomes for evaluation.
- `20250101_evaluations.json`: Sample evaluation scores (0–10 scale).

## Usage

These files are primarily for browsing and understanding the expected JSON shapes.

Optional (turn sample predictions/evaluations into a GSPO training JSON without running models):

```bash
export VARRO_RUN_DIR_SUFFIX=SAMPLE
mkdir -p timestamped_storage_SAMPLE
cp data/sample/20250101_*.json timestamped_storage_SAMPLE/

python run_daily_pipeline.py --mode night --date 20250101
```
