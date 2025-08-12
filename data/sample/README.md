# Sample Data Directory

This directory contains a small CC‑BY sample dataset to illustrate the Varro GSPO pipeline.

## Contents

- `20250101_headlines.json`: One example financial headline.
- `20250101_predictions.json`: Eight sample rollouts for the headline.
- `20250101_outcome_tracking.json`: Next-day outcomes for evaluation.
- `20250101_evaluations.json`: Sample evaluation scores (0–10 scale).

## Usage

To experiment with the pipeline without live RSS feeds, copy or link this folder into your timestamped storage and prediction directories,
then run the corresponding pipeline stages. For example:

```bash
# Morning stage: use sample headlines
python run_daily_pipeline.py --mode morning --input_dir data/sample

# Evening stage: evaluate sample predictions
python run_daily_pipeline.py --mode evening --input_dir data/sample

# Night stage: train on sample evaluations
python run_daily_pipeline.py --mode night --input_dir data/sample
```
