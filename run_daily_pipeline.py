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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DailyPipeline:
    """Orchestrates the complete daily prediction and training pipeline."""
    
    def __init__(self, trained_model_path: str = None):
        self.rss_collector = EnhancedRSSCollector()
        self.storage = TimestampedStorage()
        
        # Use trained model if provided, otherwise use base model
        if trained_model_path:
            logger.info(f"Using trained model from: {trained_model_path}")
            self.prediction_generator = AdaptiveRolloutGenerator(checkpoint_path=trained_model_path)
        else:
            logger.info("Using base model for predictions")
            self.prediction_generator = AdaptiveRolloutGenerator()
            
        self.prediction_storage = PredictionStorage()
        self.outcome_tracker = OutcomeTracker()
        self.evaluator = LLMOutcomeEvaluator()
        self.evaluation_storage = EvaluationStorage()
        
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
                "training_config": {"learning_rate": 5e-7, "rollouts_per_step": 8, "save_every": 1}
            }
        
        return config
    
    def run_morning_pipeline(self, date: str = None):
        """Run morning pipeline: collect headlines and generate predictions."""
        if date is None:
            date = datetime.now().strftime("%Y%m%d")
        
        logger.info(f"Starting morning pipeline for {date}")
        
        try:
            # Step 1: Collect headlines
            logger.info("Step 1: Collecting headlines...")
            headlines = self.rss_collector.collect_headlines()
            
            if not headlines:
                logger.error("No headlines collected!")
                return False
            
            # Save headlines
            self.rss_collector.save_headlines(headlines, date)
            logger.info(f"Collected and saved {len(headlines)} headlines")
            
            # Step 2: Generate predictions
            logger.info("Step 2: Generating predictions...")
            predictions = self.prediction_generator.generate_daily_predictions(headlines)
            
            if not predictions:
                logger.error("No predictions generated!")
                return False
            
            # Save predictions
            self.prediction_storage.save_daily_predictions(predictions, date)
            logger.info(f"Generated and saved {len(predictions)} predictions with {self._count_total_rollouts(predictions)} rollouts")
            
            logger.info(f"Morning pipeline completed successfully for {date}")
            return True
            
        except Exception as e:
            logger.error(f"Error in morning pipeline: {e}")
            return False
    
    def run_evening_pipeline(self, prediction_date: str = None):
        """Run evening pipeline: evaluate previous day's predictions."""
        if prediction_date is None:
            # Use previous day's date
            yesterday = datetime.now() - timedelta(days=1)
            prediction_date = yesterday.strftime("%Y%m%d")
        
        logger.info(f"Starting evening pipeline for predictions from {prediction_date}")
        
        try:
            # Step 1: Track outcomes
            logger.info("Step 1: Tracking outcomes...")
            outcomes = self.outcome_tracker.track_outcomes_for_date(prediction_date)
            
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
            
            # Step 3: Prepare training data
            logger.info("Step 3: Preparing training data...")
            training_data = self.evaluation_storage.get_gspo_training_data(prediction_date, prediction_date)
            
            if training_data:
                # Save GSPO training data
                filename = self.evaluation_storage.save_gspo_training_data(training_data)
                logger.info(f"Prepared {len(training_data)} training examples")
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
            
            # TODO: Implement GSPO training
            # For now, just log the training data
            logger.info(f"Training data prepared: {len(training_data)} examples")
            logger.info(f"Average reward: {sum(item['reward'] for item in training_data) / len(training_data):.3f}")
            
            # Save training data for manual training
            training_filename = f"training/gspo_training_{prediction_date}.json"
            os.makedirs("training", exist_ok=True)
            
            with open(training_filename, 'w') as f:
                json.dump({
                    "date": prediction_date,
                    "training_data": training_data,
                    "created_at": datetime.now().isoformat()
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
        morning_success = self.run_morning_pipeline(date)
        if not morning_success:
            logger.error("Morning pipeline failed!")
            return False
        
        # Evening pipeline (evaluate previous day)
        yesterday = datetime.strptime(date, "%Y%m%d") - timedelta(days=1)
        yesterday_str = yesterday.strftime("%Y%m%d")
        
        evening_success = self.run_evening_pipeline(yesterday_str)
        if not evening_success:
            logger.warning("Evening pipeline failed!")
        
        # Night training
        night_success = self.run_night_training(yesterday_str)
        if not night_success:
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
        
        status = {
            "date": date,
            "headlines": self.storage.load_data("headlines", date) is not None,
            "predictions": self.storage.load_data("predictions", date) is not None,
            "outcome_tracking": self.storage.load_data("outcome_tracking", date) is not None,
            "evaluations": self.storage.load_data("evaluations", date) is not None
        }
        
        return status
    
    def cleanup_old_data(self, days_to_keep: int = 30):
        """Clean up old data files."""
        self.storage.cleanup_old_data(days_to_keep)
        logger.info(f"Cleaned up data older than {days_to_keep} days")

def main():
    """Main function for running the daily pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run the daily prediction pipeline")
    parser.add_argument("--mode", choices=["morning", "evening", "night", "full"], 
                       default="full", help="Pipeline mode to run")
    parser.add_argument("--date", help="Date to process (YYYYMMDD format)")
    parser.add_argument("--cleanup", action="store_true", help="Clean up old data")
    parser.add_argument("--trained-model", help="Path to trained model checkpoint")
    
    args = parser.parse_args()
    
    pipeline = DailyPipeline(trained_model_path=args.trained_model)
    
    if args.cleanup:
        pipeline.cleanup_old_data()
        return
    
    date = args.date
    if not date:
        date = datetime.now().strftime("%Y%m%d")
    
    logger.info(f"Running pipeline in {args.mode} mode for date {date}")
    
    if args.mode == "morning":
        success = pipeline.run_morning_pipeline(date)
    elif args.mode == "evening":
        success = pipeline.run_evening_pipeline(date)
    elif args.mode == "night":
        success = pipeline.run_night_training(date)
    else:  # full
        success = pipeline.run_full_daily_cycle(date)
    
    if success:
        logger.info("Pipeline completed successfully")
        
        # Show status
        status = pipeline.get_pipeline_status(date)
        logger.info(f"Pipeline status: {status}")
    else:
        logger.error("Pipeline failed!")
        sys.exit(1)

if __name__ == "__main__":
    main() 