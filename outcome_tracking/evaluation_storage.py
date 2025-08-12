#!/usr/bin/env python3
"""
Evaluation Storage
Store evaluation results and prepare training data for GSPO training.
"""

import json
import logging
import os
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path

from data_collection.timestamped_storage import TimestampedStorage

logger = logging.getLogger(__name__)

class EvaluationStorage:
    """Manages storage and retrieval of evaluation results."""
    
    def __init__(self):
        self.storage = TimestampedStorage()
        
        # Statistics tracking
        self.stats = {
            "total_evaluations": 0,
            "avg_outcome_scores": [],
            "methods_evaluated": {},
            "score_distributions": []
        }
    
    def save_evaluations(self, evaluations: List[Dict[str, Any]], date: str):
        """Save evaluation results for a specific date."""
        data = {
            "date": date,
            "total_evaluations": len(evaluations),
            "evaluations": evaluations,
            "summary": self._create_evaluation_summary(evaluations),
            "saved_at": datetime.now().isoformat()
        }
        
        filename = self.storage.save_data(data, "evaluations", date)
        
        # Update statistics
        self._update_stats(evaluations)
        # Also write a flat CSV for observability
        try:
            self._write_csv_summary(evaluations, date)
        except Exception as e:
            logger.warning(f"Failed to write CSV summary: {e}")
        
        logger.info(f"Saved {len(evaluations)} evaluations for date: {date}")
        return filename
    
    def load_evaluations(self, date: str) -> Optional[Dict[str, Any]]:
        """Load evaluation results for a specific date."""
        return self.storage.load_data("evaluations", date)
    
    def get_training_data(self, start_date: str, end_date: str) -> List[Dict[str, Any]]:
        """Get training data combining predictions and evaluations."""
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
                            "date": pred_data["date"],
                            "rollout_id": rollout["rollout_id"]
                        }
                        
                        # Add evaluation if available
                        if prediction_id in evaluation_lookup:
                            evaluation = evaluation_lookup[prediction_id]
                            training_item["outcome_score"] = evaluation.get("reward")  # Use reward as outcome_score
                            training_item["normalized_score"] = evaluation.get("reward")  # Use reward as normalized_score
                            training_item["explanation"] = evaluation.get("explanation")
                            training_item["evaluated"] = True
                        else:
                            training_item["evaluated"] = False
                        
                        training_data.append(training_item)
        
        logger.info(f"Created {len(training_data)} training examples from {start_date} to {end_date}")
        return training_data
    
    def get_gspo_training_data(self, start_date: str, end_date: str) -> List[Dict[str, Any]]:
        """Get training data specifically formatted for GSPO training."""
        training_data = self.get_training_data(start_date, end_date)
        # Build a quick evaluation lookup again for composite/trade-thinking carry-over
        eval_lookup = {}
        evaluations_data = self.storage.get_date_range(start_date, end_date, "evaluations")
        for ed in evaluations_data:
            if "evaluations" in ed:
                for ev in ed["evaluations"]:
                    pid = ev.get("prediction_id")
                    if pid:
                        eval_lookup[pid] = ev
        
        gspo_data = []
        for item in training_data:
            # Only include items with evaluations
            if item.get("evaluated", False):
                gspo_item = {
                    "prompt": item["prompt"],
                    "response": item["prediction"],
                    # Default: normalized_score (LLM reward); will be overridden by composite if present
                    "reward": item["normalized_score"],
                    "headline": item["headline"],
                    "method": item["method"],
                    "date": item["date"],
                    "immediate_reward": item["immediate_reward"],
                    "outcome_score": item["outcome_score"],
                    "explanation": item["explanation"]
                }
                # If evaluation row had composite/trade-thinking, prefer composite for reward and carry both
                pred_id = f"{item['date']}_{item['headline'][:20].replace(' ', '_')}_{item['rollout_id']}"
                ev = eval_lookup.get(pred_id)
                if isinstance(ev, dict):
                    comp = ev.get("composite_reward")
                    tscore = ev.get("trade_thinking_score")
                    if isinstance(comp, (int, float)):
                        gspo_item["reward"] = comp
                        gspo_item["composite_reward"] = comp
                    if isinstance(tscore, (int, float)):
                        gspo_item["trade_thinking_score"] = tscore
                gspo_data.append(gspo_item)
        
        logger.info(f"Created {len(gspo_data)} GSPO training examples")
        return gspo_data

    def _write_csv_summary(self, evaluations: List[Dict[str, Any]], date: str):
        """Write evaluations as a CSV file for quick analysis."""
        import csv
        out_dir = Path(self.storage.storage_dir)
        csv_path = out_dir / f"{date}_evaluations.csv"
        fields = [
            "prediction_id", "headline", "rollout_id", "method", "ranking", "reward",
            "date", "status"
        ]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for ev in evaluations:
                row = {k: ev.get(k) for k in fields}
                writer.writerow(row)
    
    def save_gspo_training_data(self, training_data: List[Dict[str, Any]], filename: str = None):
        """Save GSPO training data to file."""
        if filename is None:
            filename = f"training/gspo_training_data_{datetime.now().strftime('%Y%m%d')}.json"
        
        os.makedirs("training", exist_ok=True)
        
        data = {
            "created_at": datetime.now().isoformat(),
            "total_examples": len(training_data),
            "training_data": training_data,
            "summary": self._create_gspo_summary(training_data)
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved GSPO training data to {filename}")
        return filename
    
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
    
    def _create_evaluation_summary(self, evaluations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create summary statistics for evaluations."""
        if not evaluations:
            return {"error": "No evaluations provided"}
        
        # Calculate outcome scores
        outcome_scores = [eval.get("outcome_score", 0) for eval in evaluations]
        normalized_scores = [eval.get("normalized_score", 0) for eval in evaluations]
        composite_scores = [ev.get("composite_reward") for ev in evaluations if isinstance(ev.get("composite_reward"), (int, float))]
        trade_scores = [ev.get("trade_thinking_score") for ev in evaluations if isinstance(ev.get("trade_thinking_score"), (int, float))]
        
        # Count methods
        methods_used = {}
        for evaluation in evaluations:
            method = evaluation.get("method", "unknown")
            methods_used[method] = methods_used.get(method, 0) + 1
        
        # Calculate score distribution
        score_ranges = {
            "0-3": sum(1 for score in outcome_scores if 0 <= score <= 3),
            "4-6": sum(1 for score in outcome_scores if 4 <= score <= 6),
            "7-8": sum(1 for score in outcome_scores if 7 <= score <= 8),
            "9-10": sum(1 for score in outcome_scores if 9 <= score <= 10)
        }
        
        return {
            "total_evaluations": len(evaluations),
            "avg_outcome_score": sum(outcome_scores) / len(outcome_scores) if outcome_scores else 0.0,
            "avg_normalized_score": sum(normalized_scores) / len(normalized_scores) if normalized_scores else 0.0,
            "avg_composite_reward": (sum(composite_scores) / len(composite_scores)) if composite_scores else 0.0,
            "avg_trade_thinking_score": (sum(trade_scores) / len(trade_scores)) if trade_scores else 0.0,
            "score_distribution": score_ranges,
            "methods_used": methods_used,
            "score_range": {
                "min": min(outcome_scores) if outcome_scores else 0.0,
                "max": max(outcome_scores) if outcome_scores else 0.0
            }
        }
    
    def _create_gspo_summary(self, training_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create summary for GSPO training data."""
        if not training_data:
            return {"error": "No training data provided"}
        
        # Extract rewards (filter out None values)
        rewards = [item.get("reward", 0) for item in training_data if item.get("reward") is not None]
        immediate_rewards = [item.get("immediate_reward", 0) for item in training_data if item.get("immediate_reward") is not None]
        outcome_scores = [item.get("outcome_score", 0) for item in training_data if item.get("outcome_score") is not None]
        
        # Count methods
        methods_used = {}
        for item in training_data:
            method = item.get("method", "unknown")
            methods_used[method] = methods_used.get(method, 0) + 1
        
        return {
            "total_examples": len(training_data),
            "avg_reward": sum(rewards) / len(rewards) if rewards else 0.0,
            "avg_immediate_reward": sum(immediate_rewards) / len(immediate_rewards) if immediate_rewards else 0.0,
            "avg_outcome_score": sum(outcome_scores) / len(outcome_scores) if outcome_scores else 0.0,
            "methods_used": methods_used,
            "reward_range": {
                "min": min(rewards) if rewards else 0.0,
                "max": max(rewards) if rewards else 0.0
            }
        }
    
    def _update_stats(self, evaluations: List[Dict[str, Any]]):
        """Update internal statistics."""
        self.stats["total_evaluations"] += len(evaluations)
        
        # Update average outcome scores
        outcome_scores = [eval.get("outcome_score", 0) for eval in evaluations]
        if outcome_scores:
            self.stats["avg_outcome_scores"].append(sum(outcome_scores) / len(outcome_scores))
        
        # Update methods used
        for evaluation in evaluations:
            method = evaluation.get("method", "unknown")
            self.stats["methods_evaluated"][method] = self.stats["methods_evaluated"].get(method, 0) + 1
        
        # Update score distributions
        score_ranges = {
            "0-3": sum(1 for score in outcome_scores if 0 <= score <= 3),
            "4-6": sum(1 for score in outcome_scores if 4 <= score <= 6),
            "7-8": sum(1 for score in outcome_scores if 7 <= score <= 8),
            "9-10": sum(1 for score in outcome_scores if 9 <= score <= 10)
        }
        self.stats["score_distributions"].append(score_ranges)
    
    def get_evaluation_statistics(self, start_date: str, end_date: str) -> Dict[str, Any]:
        """Get comprehensive evaluation statistics."""
        evaluations_data = self.storage.get_date_range(start_date, end_date, "evaluations")
        
        total_evaluations = 0
        all_outcome_scores = []
        all_normalized_scores = []
        methods_used = {}
        
        for data in evaluations_data:
            if "evaluations" in data:
                for evaluation in data["evaluations"]:
                    total_evaluations += 1
                    
                    outcome_score = evaluation.get("outcome_score", 0)
                    normalized_score = evaluation.get("normalized_score", 0)
                    
                    all_outcome_scores.append(outcome_score)
                    all_normalized_scores.append(normalized_score)
                    
                    method = evaluation.get("method", "unknown")
                    methods_used[method] = methods_used.get(method, 0) + 1
        
        return {
            "date_range": {"start": start_date, "end": end_date},
            "total_evaluations": total_evaluations,
            "avg_outcome_score": sum(all_outcome_scores) / len(all_outcome_scores) if all_outcome_scores else 0.0,
            "avg_normalized_score": sum(all_normalized_scores) / len(all_normalized_scores) if all_normalized_scores else 0.0,
            "methods_used": methods_used,
            "score_range": {
                "min": min(all_outcome_scores) if all_outcome_scores else 0.0,
                "max": max(all_outcome_scores) if all_outcome_scores else 0.0
            }
        }
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get comprehensive storage statistics."""
        storage_stats = self.storage.get_storage_stats()
        
        return {
            "storage": storage_stats,
            "evaluations": self.stats,
            "avg_outcome_score_over_time": self.stats["avg_outcome_scores"] if self.stats["avg_outcome_scores"] else []
        }

def main():
    """Test the evaluation storage system."""
    storage = EvaluationStorage()
    
    # Test with sample evaluations
    test_evaluations = [
        {
            "prediction_id": "test_001",
            "outcome_score": 8.5,
            "normalized_score": 0.85,
            "explanation": "Prediction was mostly accurate",
            "method": "basic_stochastic",
            "headline": "Fed signals rate cut"
        },
        {
            "prediction_id": "test_002",
            "outcome_score": 6.0,
            "normalized_score": 0.6,
            "explanation": "Prediction was partially correct",
            "method": "advanced_perspective",
            "headline": "Apple reports earnings"
        }
    ]
    
    # Save evaluations
    filename = storage.save_evaluations(test_evaluations, "20240115")
    print(f"Saved evaluations to: {filename}")
    
    # Load evaluations
    loaded_data = storage.load_evaluations("20240115")
    print(f"Loaded data: {loaded_data is not None}")
    
    # Get training data
    training_data = storage.get_training_data("20240115", "20240116")
    print(f"Training data: {len(training_data)} examples")
    
    # Get GSPO training data
    gspo_data = storage.get_gspo_training_data("20240115", "20240116")
    print(f"GSPO training data: {len(gspo_data)} examples")

if __name__ == "__main__":
    main() 
