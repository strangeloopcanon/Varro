# Future Predictions (A/B)

This folder holds six consolidated JSONs from 2025-08-11.
Copy the needed file back to a run dir before evaluation (examples below).

Examples:
- export VARRO_RUN_DIR_SUFFIX=AB_BASE_DAY && mkdir -p timestamped_storage_AB_BASE_DAY && cp future_predictions/20250811_predictions_AB_BASE_DAY.json timestamped_storage_AB_BASE_DAY/20250811_predictions.json && python run_daily_pipeline.py --mode evening --date 20250811
- export VARRO_RUN_DIR_SUFFIX=AB_LATEST_MONTH && mkdir -p timestamped_storage_AB_LATEST_MONTH && cp future_predictions/20250811_predictions_AB_LATEST_MONTH.json timestamped_storage_AB_LATEST_MONTH/20250811_predictions.json && python run_daily_pipeline.py --mode evening --date 20250811 --override_headlines_date 20250911
- export VARRO_RUN_DIR_SUFFIX=AB_LATEST_YEAR && mkdir -p timestamped_storage_AB_LATEST_YEAR && cp future_predictions/20250811_predictions_AB_LATEST_YEAR.json timestamped_storage_AB_LATEST_YEAR/20250811_predictions.json && python run_daily_pipeline.py --mode evening --date 20250811 --override_headlines_date 20260811
