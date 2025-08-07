#!/usr/bin/env python3
"""
Prediction Storage
Store and manage daily predictions with preparation for next-day evaluation.
"""

import json
import logging
import os
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path

from data_collection.timestamped_storage import TimestampedStorage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PredictionStorage:
    """Manages storage and retrieval of daily predictions."""
    
    def __init__(self):
        self.storage = TimestampedStorage()
        
        # Statistics tracking
        self.stats = {
            "total_predictions": 0,
            "total_rollouts": 0,
            "methods_used": {},
            "avg_rewards": []
        }
    
    def save_daily_predictions(self, predictions: List[Dict[str, Any]], date: str = None):
        """Save daily predictions to timestamped storage."""
        if date is None:
            date = datetime.now().strftime("%Y%m%d")
        
        # Prepare data for storage
        data = {
            "date": date,
            "total_headlines": len(predictions),
            "predictions": predictions,
            "summary": self._create_prediction_summary(predictions),
            "saved_at": datetime.now().isoformat()
        }
        
        # Save to timestamped storage
        filename = self.storage.save_data(data, "predictions", date)
        
        # Update statistics
        self._update_stats(predictions)
        
        logger.info(f"Saved {len(predictions)} predictions with {self._count_total_rollouts(predictions)} rollouts")
        return filename
    
    def load_daily_predictions(self, date: str) -> Optional[Dict[str, Any]]:
        """Load daily predictions for a specific date."""
        return self.storage.load_data("predictions", date)
    
    def get_pending_evaluations(self, date: str) -> List[Dict[str, Any]]:
        """Get predictions that need evaluation for a specific date."""
        predictions_data = self.load_daily_predictions(date)
        
        if not predictions_data:
            logger.warning(f"No predictions found for date: {date}")
            return []
        
        # Flatten all rollouts for evaluation
        pending_evaluations = []
        
        for prediction in predictions_data["predictions"]:
            headline = prediction["headline"]
            
            for rollout in prediction["rollouts"]:
                evaluation_item = {
                    "prediction_id": f"{date}_{prediction['headline'][:20].replace(' ', '_')}_{rollout['rollout_id']}",
                    "headline": headline,
                    "original_prediction": rollout["prediction"],
                    "rollout_id": rollout["rollout_id"],
                    "method": rollout["method"],
                    "immediate_reward": rollout["immediate_reward"],
                    "timestamp": rollout["timestamp"],
                    "date": date
                }
                
                # Add perspective if available
                if "perspective" in rollout:
                    evaluation_item["perspective"] = rollout["perspective"]
                
                pending_evaluations.append(evaluation_item)
        
        logger.info(f"Found {len(pending_evaluations)} pending evaluations for date: {date}")
        return pending_evaluations
    
    def save_evaluations(self, evaluations: List[Dict[str, Any]], date: str):
        """Save evaluation results."""
        data = {
            "date": date,
            "total_evaluations": len(evaluations),
            "evaluations": evaluations,
            "summary": self._create_evaluation_summary(evaluations),
            "saved_at": datetime.now().isoformat()
        }
        
        filename = self.storage.save_data(data, "evaluations", date)
        
        logger.info(f"Saved {len(evaluations)} evaluations for date: {date}")
        return filename
    
    def get_training_data(self, start_date: str, end_date: str) -> List[Dict[str, Any]]:
        """Get training data for a date range."""
        training_data = []
        
        # Get all predictions and evaluations in date range
        predictions_data = self.storage.get_date_range(start_date, end_date, "predictions")
        evaluations_data = self.storage.get_date_range(start_date, end_date, "evaluations")
        
        # Create lookup for evaluations
        evaluation_lookup = {}
        for eval_data in evaluations_data:
            if "evaluations" in eval_data:
                for evaluation in eval_data["evaluations"]:
                    prediction_id = evaluation.get("prediction_id")
                    if prediction_id:
                        evaluation_lookup[prediction_id] = evaluation
        
        # Combine predictions with evaluations
        for pred_data in predictions_data:
            if "predictions" in pred_data:
                for prediction in pred_data["predictions"]:
                    headline = prediction["headline"]
                    
                    for rollout in prediction["rollouts"]:
                        prediction_id = f"{pred_data['date']}_{headline[:20].replace(' ', '_')}_{rollout['rollout_id']}"
                        
                        training_item = {
                            "headline": headline,
                            "prompt": self._create_prediction_prompt(headline),
                            "prediction": rollout["prediction"],
                            "immediate_reward": rollout["immediate_reward"],
                            "method": rollout["method"],
                            "date": pred_data["date"]
                        }
                        
                        # Add evaluation if available
                        if prediction_id in evaluation_lookup:
                            evaluation = evaluation_lookup[prediction_id]
                            training_item["outcome_score"] = evaluation.get("outcome_score")
                            training_item["normalized_score"] = evaluation.get("normalized_score")
                            training_item["explanation"] = evaluation.get("explanation")
                        
                        training_data.append(training_item)
        
        logger.info(f"Created {len(training_data)} training examples from {start_date} to {end_date}")
        return training_data
    
    def _create_prediction_prompt(self, headline: str) -> str:
        """Create prediction prompt for a headline."""
        try:
            with open("config/prompt_templates.json", 'r') as f:
                prompt_config = json.load(f)
            return prompt_config["basic_prompt"].format(headline=headline)
        except Exception as e:
            logger.error(f"Error loading prompt template: {e}")
            # Fallback to hardcoded prompt
            return f"""
Headline: "{headline}"

Based on this news, what do you think will happen and what's the trade?

Please structure your response as follows:
1. **Market Impact**: How will this affect markets broadly?
2. **Specific Assets**: Which specific assets will be most affected?
3. **Trade Recommendation**: What specific trade would you make?
4. **Timeframe**: When do you expect this to play out?
5. **Risk Factors**: What could go wrong with this prediction?
6. **World View**: What world does this news imagine or create? What narrative does it suggest?

Think like a trader - be specific about what to buy/sell and why.
"""
    
    def _create_prediction_summary(self, predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create summary statistics for predictions."""
        total_rollouts = self._count_total_rollouts(predictions)
        
        # Count methods used
        methods_used = {}
        for prediction in predictions:
            for rollout in prediction["rollouts"]:
                method = rollout["method"]
                methods_used[method] = methods_used.get(method, 0) + 1
        
        # Calculate average rewards
        all_rewards = []
        for prediction in predictions:
            for rollout in prediction["rollouts"]:
                all_rewards.append(rollout["immediate_reward"])
        
        avg_reward = sum(all_rewards) / len(all_rewards) if all_rewards else 0.0
        
        return {
            "total_headlines": len(predictions),
            "total_rollouts": total_rollouts,
            "methods_used": methods_used,
            "avg_immediate_reward": avg_reward,
            "reward_range": {
                "min": min(all_rewards) if all_rewards else 0.0,
                "max": max(all_rewards) if all_rewards else 0.0
            }
        }
    
    def _create_evaluation_summary(self, evaluations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create summary statistics for evaluations."""
        if not evaluations:
            return {"error": "No evaluations provided"}
        
        # Calculate outcome scores
        outcome_scores = [eval.get("outcome_score", 0) for eval in evaluations]
        normalized_scores = [eval.get("normalized_score", 0) for eval in evaluations]
        
        # Count methods
        methods_used = {}
        for evaluation in evaluations:
            method = evaluation.get("method", "unknown")
            methods_used[method] = methods_used.get(method, 0) + 1
        
        return {
            "total_evaluations": len(evaluations),
            "avg_outcome_score": sum(outcome_scores) / len(outcome_scores) if outcome_scores else 0.0,
            "avg_normalized_score": sum(normalized_scores) / len(normalized_scores) if normalized_scores else 0.0,
            "score_range": {
                "min": min(outcome_scores) if outcome_scores else 0.0,
                "max": max(outcome_scores) if outcome_scores else 0.0
            },
            "methods_used": methods_used
        }
    
    def _count_total_rollouts(self, predictions: List[Dict[str, Any]]) -> int:
        """Count total number of rollouts across all predictions."""
        total = 0
        for prediction in predictions:
            total += len(prediction.get("rollouts", []))
        return total
    
    def _update_stats(self, predictions: List[Dict[str, Any]]):
        """Update internal statistics."""
        self.stats["total_predictions"] += len(predictions)
        
        total_rollouts = self._count_total_rollouts(predictions)
        self.stats["total_rollouts"] += total_rollouts
        
        # Update methods used
        for prediction in predictions:
            for rollout in prediction["rollouts"]:
                method = rollout["method"]
                self.stats["methods_used"][method] = self.stats["methods_used"].get(method, 0) + 1
        
        # Update average rewards
        all_rewards = []
        for prediction in predictions:
            for rollout in prediction["rollouts"]:
                all_rewards.append(rollout["immediate_reward"])
        
        if all_rewards:
            self.stats["avg_rewards"].append(sum(all_rewards) / len(all_rewards))
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get comprehensive storage statistics."""
        storage_stats = self.storage.get_storage_stats()
        
        return {
            "storage": storage_stats,
            "predictions": self.stats,
            "avg_reward_over_time": self.stats["avg_rewards"] if self.stats["avg_rewards"] else []
        }
    
    def cleanup_old_predictions(self, days_to_keep: int = 30):
        """Clean up old prediction data."""
        self.storage.cleanup_old_data(days_to_keep)
        logger.info(f"Cleaned up predictions older than {days_to_keep} days")

def main():
    """Test the prediction storage system."""
    storage = PredictionStorage()
    
    # Test with sample predictions
    test_predictions = [
        {
            "headline": "Fed signals potential rate cut in March",
            "rollouts": [
                {
                    "rollout_id": 0,
                    "prediction": "Bonds will rally, buy TLT calls...",
                    "method": "basic_stochastic",
                    "immediate_reward": 0.85,
                    "timestamp": datetime.now().isoformat()
                },
                {
                    "rollout_id": 1,
                    "prediction": "Dollar will weaken, short USD/JPY...",
                    "method": "basic_stochastic",
                    "immediate_reward": 0.78,
                    "timestamp": datetime.now().isoformat()
                }
            ]
        }
    ]
    
    # Save predictions
    filename = storage.save_daily_predictions(test_predictions, "20240115")
    print(f"Saved predictions to: {filename}")
    
    # Load predictions
    loaded_data = storage.load_daily_predictions("20240115")
    print(f"Loaded data: {loaded_data is not None}")
    
    # Get pending evaluations
    pending = storage.get_pending_evaluations("20240115")
    print(f"Pending evaluations: {len(pending)}")

if __name__ == "__main__":
    main() 