#!/usr/bin/env python3
"""
Daily Pipeline Orchestration
Main script that orchestrates the complete daily prediction and training pipeline.
"""

import json
import logging
import os
import sys
from datetime import datetime, timedelta
from typing import List, Dict, Any
from pathlib import Path

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_collection.enhanced_rss_collector import EnhancedRSSCollector
from data_collection.timestamped_storage import TimestampedStorage
from prediction_generation.adaptive_rollout_generator import AdaptiveRolloutGenerator
from prediction_generation.prediction_storage import PredictionStorage
from outcome_tracking.outcome_tracker import OutcomeTracker
from outcome_tracking.llm_outcome_evaluator import LLMOutcomeEvaluator
from outcome_tracking.evaluation_storage import EvaluationStorage

# Configure logging for the entry script
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


class DailyPipeline:
    """Orchestrates the complete daily prediction and training pipeline."""

    def __init__(self, trained_model_path: str = None, seed: int = None, num_rollouts: int = 8, horizon: str | None = None, sampler_profile: str = "default", auto_profile: bool = False, output_format: str = "one_line"):
        self.rss_collector = EnhancedRSSCollector()
        self.storage = TimestampedStorage()
        self.output_format = (output_format or "one_line")

        # Use trained model if provided, otherwise use base model
        if trained_model_path:
            logger.info(f"Using trained model from: {trained_model_path}")
            self.prediction_generator = AdaptiveRolloutGenerator(checkpoint_path=trained_model_path, sampler_profile=sampler_profile, output_format=self.output_format)
        else:
            logger.info("Using base model for predictions")
            self.prediction_generator = AdaptiveRolloutGenerator(sampler_profile=sampler_profile, output_format=self.output_format)

        self.prediction_storage = PredictionStorage()
        self.outcome_tracker = OutcomeTracker()
        self.evaluator = LLMOutcomeEvaluator()
        self.evaluation_storage = EvaluationStorage()

        # Generation settings
        self.num_rollouts = num_rollouts
        self.horizon = horizon
        self.auto_profile = auto_profile
        self.default_sampler_profile = sampler_profile

        # Best-effort global seeding
        if seed is not None:
            try:
                import random
                import numpy as np

                random.seed(seed)
                np.random.seed(seed)
                logger.info(f"Global seed set to {seed}")
            except Exception:
                logger.warning("Could not fully set global seeds; continuing without strict determinism.")

        # Load configuration
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from files."""
        config = {}

        # Load training config
        try:
            with open("config/training_config.json", 'r') as f:
                config["training"] = json.load(f)
        except FileNotFoundError:
            logger.warning("Training config not found, using defaults")
            config["training"] = {
                "model_name": "Qwen/Qwen3-0.6B",
                "sampler_config": {"temperature": 0.8, "top_p": 0.95, "top_k": 50},
                "rollout_config": {"basic_threshold": 0.6, "max_tokens": 512, "num_rollouts": 8},
                "training_config": {"learning_rate": 5e-7, "rollouts_per_step": 8, "save_every": 1},
            }

        return config

    def run_morning_pipeline(self, date: str = None):
        """Run morning pipeline: collect headlines and generate predictions."""
        if date is None:
            date = datetime.now().strftime("%Y%m%d")

        logger.info(f"Starting morning pipeline for {date}")

        try:
            # Optional: auto-select sampler profile based on previous day's metrics
            if self.auto_profile:
                try:
                    prev_date = (datetime.strptime(date, "%Y%m%d") - timedelta(days=1)).strftime("%Y%m%d")
                    profile = self._choose_sampler_profile(prev_date, self.default_sampler_profile)
                    if profile != getattr(self.prediction_generator, 'sampler_profile', 'default'):
                        logger.info(f"Auto-profile selected '{profile}' (was '{self.prediction_generator.sampler_profile}') based on {prev_date} metrics")
                    else:
                        logger.info(f"Auto-profile using '{profile}' based on {prev_date} metrics")
                    # Apply profile to generator
                    self.prediction_generator.sampler_profile = profile
                    self.prediction_generator.sampler = self.prediction_generator._create_sampler(profile)
                except Exception as e:
                    logger.warning(f"Auto-profile selection failed, using default profile '{self.default_sampler_profile}': {e}")

            # Step 1: Load existing headlines if available; otherwise collect
            logger.info("Step 1: Loading existing headlines (or collecting if missing)...")
            existing = self.storage.load_data("headlines", date)
            if existing and isinstance(existing, dict) and "headlines" in existing:
                headlines = existing["headlines"]
                logger.info(f"Using existing headlines for {date} ({len(headlines)})")
            else:
                headlines = self.rss_collector.collect_headlines()
                if not headlines:
                    logger.error("No headlines collected!")
                    return False
                # Save headlines only if collected anew
                self.rss_collector.save_headlines(headlines, date)
                logger.info(f"Collected and saved {len(headlines)} headlines")

            # Step 1b: Scrape and save article info (non-breaking separate file)
            try:
                existing_articles = self.storage.load_data("articles", date)
                if existing_articles and isinstance(existing_articles, dict) and existing_articles.get("articles"):
                    logger.info(f"Using existing articles for {date} ({len(existing_articles['articles'])})")
                else:
                    arts = self.rss_collector.collect_article_info(headlines)
                    if arts:
                        self.rss_collector.save_articles(arts, date)
                        logger.info(f"Collected and saved {len(arts)} articles")
                    else:
                        logger.info("No articles scraped (scraper unavailable or zero results)")
            except Exception as e:
                logger.warning(f"Article scraping step failed; continuing without articles: {e}")

            # Step 1c: Attach cleaned article excerpts to headlines when available
            try:
                from data_collection.article_cleaning import build_link_to_excerpt_map
                link_map = build_link_to_excerpt_map(date, self.storage)
                if link_map:
                    attached = 0
                    for h in headlines:
                        link = h.get("link")
                        if link and link in link_map:
                            h["article_excerpt"] = link_map[link]
                            attached += 1
                    logger.info(f"Attached article excerpts to {attached} headlines (cleaned or raw)")
                else:
                    logger.info("No article excerpts available to attach")
            except Exception as e:
                logger.warning(f"Failed to attach article excerpts: {e}")

            # Step 2: Generate predictions (support multi-horizon)
            logger.info("Step 2: Generating predictions...")
            horizons = getattr(self, 'horizons', None)
            combined: List[Dict[str, Any]] = []
            if horizons:
                for h in horizons:
                    logger.info(f"Generating predictions for horizon={h}...")
                    preds_h = self.prediction_generator.generate_daily_predictions(headlines, num_rollouts=self.num_rollouts, horizon=h)
                    combined.extend(preds_h)
            else:
                preds_h = self.prediction_generator.generate_daily_predictions(headlines, num_rollouts=self.num_rollouts, horizon=self.horizon)
                combined.extend(preds_h)
            predictions = combined
            # Enrich with horizon wiring: offset_days and matures_on for each rollout
            def _offset_for(h: str | None) -> int:
                if not h:
                    return 1
                h = str(h).strip().lower()
                return 1 if h == "next_day" else (2 if h in {"next_2days", "next2days", "two_days"} else (3 if h in {"next_3days", "next3days", "three_days"} else 1))
            def _add_days(date_str: str, n: int) -> str:
                from datetime import datetime, timedelta
                try:
                    dt = datetime.strptime(date_str, "%Y%m%d")
                    return (dt + timedelta(days=int(n))).strftime("%Y%m%d")
                except Exception:
                    return date_str
            for pred in predictions:
                h = pred.get("horizon") or self.horizon or "next_day"
                off = _offset_for(h)
                mat = _add_days(date, off)
                pred["horizon"] = h
                pred["offset_days"] = off
                pred["matures_on"] = mat
                for r in pred.get("rollouts", []):
                    r["horizon"] = h
                    r["offset_days"] = off
                    r["matures_on"] = mat

            if not predictions:
                logger.error("No predictions generated!")
                return False

            # Save predictions
            self.prediction_storage.save_daily_predictions(predictions, date)
            logger.info(
                f"Generated and saved {len(predictions)} predictions with {self._count_total_rollouts(predictions)} rollouts"
            )

            logger.info(f"Morning pipeline completed successfully for {date}")
            return True

        except Exception as e:
            logger.error(f"Error in morning pipeline: {e}")
            return False

    def run_evening_pipeline(self, prediction_date: str = None, override_headlines_date: str = None, evaluate_due: bool = False):
        """Run evening pipeline: evaluate previous day's predictions."""
        if prediction_date is None:
            # Use previous day's date
            yesterday = datetime.now() - timedelta(days=1)
            prediction_date = yesterday.strftime("%Y%m%d")

        logger.info(f"Starting evening pipeline for predictions from {prediction_date}")

        try:
            # Step 1: Track outcomes
            logger.info("Step 1: Tracking outcomes...")
            if evaluate_due:
                outcomes = self.outcome_tracker.track_outcomes_due_on(prediction_date)
            else:
                outcomes = self.outcome_tracker.track_outcomes_for_date(
                    prediction_date,
                    next_headlines_date=override_headlines_date,
                )

            if not outcomes:
                logger.warning(f"No outcomes to track for {prediction_date}")
                return False

            # Save outcome tracking
            self.outcome_tracker.save_outcome_tracking(outcomes, prediction_date)
            logger.info(f"Tracked {len(outcomes)} outcomes")

            # Step 2: Evaluate predictions
            logger.info("Step 2: Evaluating predictions...")
            evaluations = self.evaluator.batch_evaluate(outcomes)

            if not evaluations:
                logger.error("No evaluations generated!")
                return False

            # Save evaluations
            self.evaluation_storage.save_evaluations(evaluations, prediction_date)
            logger.info(f"Evaluated {len(evaluations)} predictions")

            # Step 3: Prepare training data (no write; canonical file is saved during night training)
            logger.info("Step 3: Preparing training data (no write; night step saves canonical file)...")
            training_data = self.evaluation_storage.get_gspo_training_data(prediction_date, prediction_date)
            if training_data:
                logger.info(f"Prepared {len(training_data)} training examples (GSPO-ready)")
            else:
                logger.warning("No training data prepared")

            logger.info(f"Evening pipeline completed successfully for {prediction_date}")
            return True

        except Exception as e:
            logger.error(f"Error in evening pipeline: {e}")
            return False

    def run_night_training(self, prediction_date: str = None):
        """Run night training: train GSPO model with evaluation results."""
        if prediction_date is None:
            # Use previous day's date
            yesterday = datetime.now() - timedelta(days=1)
            prediction_date = yesterday.strftime("%Y%m%d")

        logger.info(f"Starting night training for predictions from {prediction_date}")

        try:
            # Load training data
            training_data = self.evaluation_storage.get_gspo_training_data(prediction_date, prediction_date)

            if not training_data:
                logger.warning(f"No training data available for {prediction_date}")
                return False

            # TODO: Integrate GSPO training here; for now, save prepared data
            logger.info(f"Training data prepared: {len(training_data)} examples")
            avg_reward = sum(item['reward'] for item in training_data) / len(training_data)
            logger.info(f"Average reward: {avg_reward:.3f}")

            training_filename = f"training/gspo_training_{prediction_date}.json"
            os.makedirs("training", exist_ok=True)

            with open(training_filename, 'w') as f:
                json.dump({
                    "date": prediction_date,
                    "training_data": training_data,
                    "created_at": datetime.now().isoformat(),
                }, f, indent=2)

            logger.info(f"Training data saved to {training_filename}")
            logger.info(f"Night training completed for {prediction_date}")
            return True

        except Exception as e:
            logger.error(f"Error in night training: {e}")
            return False

    def run_full_daily_cycle(self, date: str = None):
        """Run the complete daily cycle."""
        if date is None:
            date = datetime.now().strftime("%Y%m%d")

        logger.info(f"Starting full daily cycle for {date}")

        # Morning pipeline
        if not self.run_morning_pipeline(date):
            logger.error("Morning pipeline failed!")
            return False

        # Evening pipeline (evaluate previous day)
        yesterday = datetime.strptime(date, "%Y%m%d") - timedelta(days=1)
        yesterday_str = yesterday.strftime("%Y%m%d")

        if not self.run_evening_pipeline(yesterday_str):
            logger.warning("Evening pipeline failed!")

        # Night training
        if not self.run_night_training(yesterday_str):
            logger.warning("Night training failed!")

        logger.info(f"Full daily cycle completed for {date}")
        return True

    def _count_total_rollouts(self, predictions: List[Dict[str, Any]]) -> int:
        """Count total number of rollouts across all predictions."""
        total = 0
        for prediction in predictions:
            total += len(prediction.get("rollouts", []))
        return total

    def get_pipeline_status(self, date: str = None) -> Dict[str, Any]:
        """Get status of pipeline for a specific date."""
        if date is None:
            date = datetime.now().strftime("%Y%m%d")

        return {
            "date": date,
            "has_headlines": self.storage.load_data("headlines", date) is not None,
            "has_predictions": self.storage.load_data("predictions", date) is not None,
            "has_outcomes": self.storage.load_data("outcome_tracking", date) is not None,
            "has_evaluations": self.storage.load_data("evaluations", date) is not None,
        }

    def _choose_sampler_profile(self, prev_date: str, default_profile: str = "default") -> str:
        """Choose a sampler profile using paragraph-era metrics (Q and E).

        Q: average immediate_reward from predictions (rubric quality in [0,1])
        E: evaluation summary avg_outcome_score (evaluator reward in [0,1])
        """
        avg_quality = None  # Q
        avg_eval = None     # E
        try:
            preds = self.storage.load_data("predictions", prev_date)
            if preds and isinstance(preds, dict) and "predictions" in preds:
                total = 0
                s = 0.0
                for item in preds.get("predictions", []):
                    for r in item.get("rollouts", []):
                        val = r.get("immediate_reward")
                        if isinstance(val, (int, float)):
                            s += float(val)
                            total += 1
                if total > 0:
                    avg_quality = s / total
        except Exception:
            pass
        try:
            evals = self.storage.load_data("evaluations", prev_date)
            if evals and isinstance(evals, dict):
                summary = evals.get("summary", {})
                # After our fix, avg_outcome_score falls back to reward; use it
                aos = summary.get("avg_outcome_score")
                if isinstance(aos, (int, float)):
                    avg_eval = float(aos)
        except Exception:
            pass

        Q = avg_quality if avg_quality is not None else 0.0
        E = avg_eval if avg_eval is not None else 0.0

        # Policy: encourage moderate exploration when quality is low; allow loose only when both are healthy
        if Q < 0.35:
            return "default"
        if Q > 0.55 and E > 0.25:
            return "loose"
        return default_profile


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run the daily Varro pipeline")
    parser.add_argument("--mode", choices=["full", "morning", "evening", "night"], default="full",
                        help="Which part of the pipeline to run")
    parser.add_argument("--date", type=str, default=None,
                        help="Date in YYYYMMDD format. Morning uses this date; evening/night use it as prediction date")
    parser.add_argument("--trained-model", type=str, default=None,
                        help="Optional path to a trained model checkpoint for prediction generation")
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed for reproducibility")
    parser.add_argument("--num_rollouts", type=int, default=8, help="Number of rollouts per headline (default 8)")
    parser.add_argument("--horizon", type=str, default=None, choices=["next_day", "next_month", "next_year", "next_2days", "next_3days"], help="Optional single prediction horizon directive")
    parser.add_argument("--horizons", type=str, default=None, help="Comma-separated list of horizons to generate (e.g., next_day,next_2days,next_3days)")
    parser.add_argument("--sampler_profile", type=str, default="default", choices=["loose", "default", "tight"], help="Sampler profile for rollout generation entropy")
    parser.add_argument("--output_format", type=str, default="paragraph", choices=["one_line", "paragraph"], help="Output format for predictions (schema line or paragraph forecast)")
    parser.add_argument("--override_headlines_date", type=str, default=None,
                        help="Optional override: use this date's headlines when evaluating prediction_date")
    parser.add_argument("--evaluate_due", action="store_true", help="Evaluate all rollouts maturing on the given date (multi-horizon support)")
    parser.add_argument("--use_composite_reward", action="store_true",
                        help="Prefer composite_reward over LLM-only reward when available (doc flag; behavior handled in evaluation storage)")
    parser.add_argument("--auto_profile", action="store_true", help="Automatically choose sampler profile based on previous day metrics (morning only)")

    args = parser.parse_args()

    pipeline = DailyPipeline(trained_model_path=args.trained_model, seed=args.seed, num_rollouts=args.num_rollouts, horizon=args.horizon, sampler_profile=args.sampler_profile, auto_profile=args.auto_profile, output_format=args.output_format)
    if args.horizons:
        setattr(pipeline, "horizons", [h.strip() for h in args.horizons.split(',') if h.strip()])

    ok = True
    if args.mode == "full":
        ok = pipeline.run_full_daily_cycle(args.date)
    elif args.mode == "morning":
        ok = pipeline.run_morning_pipeline(args.date)
    elif args.mode == "evening":
        ok = pipeline.run_evening_pipeline(args.date, override_headlines_date=args.override_headlines_date, evaluate_due=bool(args.evaluate_due))
    elif args.mode == "night":
        ok = pipeline.run_night_training(args.date)

    if not ok:
        logger.error("Pipeline failed!")
        sys.exit(1)
    else:
        logger.info("Pipeline finished successfully")

    # Optional: auto-generate cross-run reports
    try:
        auto = os.getenv('VARRO_UPDATE_ALL_RUNS_REPORT', '0').lower() in {'1','true','yes'}
        if auto:
            logger.info("Generating cross-run metrics and synthesis report (VARRO_UPDATE_ALL_RUNS_REPORT=1)...")
            import subprocess, shlex
            script = os.path.join(os.path.dirname(__file__), 'analysis', 'generate_all_runs_report.py')
            if os.path.exists(script):
                subprocess.run([sys.executable, script], check=False)
            else:
                logger.warning(f"Report generator not found at {script}")
    except Exception as e:
        logger.warning(f"Auto-report generation failed: {e}")


if __name__ == "__main__":
    main()
