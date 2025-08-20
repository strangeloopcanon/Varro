#!/usr/bin/env python3
"""
Outcome Tracker
Track prediction outcomes by matching previous day's predictions with next day's headlines.
"""

import json
import logging
import os
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from pathlib import Path

from data_collection.timestamped_storage import TimestampedStorage
from data_collection.enhanced_rss_collector import EnhancedRSSCollector

logger = logging.getLogger(__name__)

class OutcomeTracker:
    """Tracks prediction outcomes by matching predictions with next-day headlines."""
    
    def __init__(self):
        self.storage = TimestampedStorage()
        self.rss_collector = EnhancedRSSCollector()
    
    def track_outcomes_for_date(self, prediction_date: str, next_headlines_date: Optional[str] = None) -> List[Dict[str, Any]]:
        """Track outcomes for predictions made on a specific date.
        If next_headlines_date is provided, use headlines from that date (override next-day logic).
        """
        logger.info(f"Tracking outcomes for predictions from {prediction_date}")
        
        # Load previous day's predictions
        predictions_data = self.storage.load_data("predictions", prediction_date)
        if not predictions_data:
            logger.warning(f"No predictions found for date: {prediction_date}")
            return []
        
        # Create a memoized headlines getter for arbitrary target dates
        headlines_cache: Dict[str, List[Dict[str, Any]]] = {}
        def _get_headlines_for(target_date: str) -> List[Dict[str, Any]]:
            if target_date in headlines_cache:
                return headlines_cache[target_date]
            if next_headlines_date is not None:
                # If override is provided, use it for all targets
                target = next_headlines_date
            else:
                target = target_date
            heads = self._get_next_day_headlines(target)
            headlines_cache[target] = heads or []
            return headlines_cache[target]

        # Create outcome tracking data (per-rollout maturity aware)
        outcomes = self._create_outcome_tracking(predictions_data, prediction_date, _get_headlines_for)
        
        logger.info(f"Created {len(outcomes)} outcome tracking entries for {prediction_date}")
        return outcomes
    
    def _get_next_date(self, date_str: str) -> str:
        """Get the next date in YYYYMMDD format."""
        date_obj = datetime.strptime(date_str, "%Y%m%d")
        next_date = date_obj + timedelta(days=1)
        return next_date.strftime("%Y%m%d")
    
    def _get_next_day_headlines(self, next_date: str) -> List[Dict[str, Any]]:
        """Get headlines for the next day."""
        # Try to load existing headlines
        headlines_data = self.storage.load_data("headlines", next_date)
        
        if headlines_data and "headlines" in headlines_data:
            logger.info(f"Loaded {len(headlines_data['headlines'])} existing headlines for {next_date}")
            return headlines_data["headlines"]
        
        # If no existing headlines, collect them
        logger.info(f"Collecting headlines for {next_date}")
        headlines = self.rss_collector.collect_headlines()
        
        # Save headlines for future use
        self.rss_collector.save_headlines(headlines, next_date)
        
        return headlines
    
    def _create_outcome_tracking(self, predictions_data: Dict[str, Any], 
                                prediction_date: str,
                                get_headlines_for) -> List[Dict[str, Any]]:
        """Create outcome tracking data by matching predictions with appropriate headlines per rollout.

        If a rollout has 'matures_on', use that date to fetch headlines; otherwise, default to prediction_date+1.
        """
        outcomes = []
        
        # Extract all rollouts from predictions
        all_rollouts = []
        for prediction in predictions_data["predictions"]:
            headline = prediction["headline"]
            
            for rollout in prediction["rollouts"]:
                # Determine maturity date for this rollout
                target_date = rollout.get("matures_on")
                if not target_date:
                    target_date = self._get_next_date(prediction_date)

                rollout_data = {
                    "prediction_id": f"{prediction_date}_{(rollout.get('horizon') or 'next_day')}_{headline[:20].replace(' ', '_')}_{rollout['rollout_id']}",
                    "headline": headline,
                    # Evaluator should rank the final, cleaned forecast paragraph
                    "original_prediction": rollout.get("prediction"),
                    "rollout_id": rollout["rollout_id"],
                    "method": rollout["method"],
                    "immediate_reward": rollout["immediate_reward"],
                        # Optional: carry trade-thinking score for composite reward downstream
                        **({"trade_thinking_score": rollout.get("trade_thinking_score")} if "trade_thinking_score" in rollout else {}),
                    "timestamp": rollout["timestamp"],
                    "date": prediction_date,
                    "horizon": rollout.get("horizon"),
                    "offset_days": rollout.get("offset_days"),
                    "matures_on": target_date
                }
                
                # Add perspective if available
                if "perspective" in rollout:
                    rollout_data["perspective"] = rollout["perspective"]
                # Mark whether this rollout used deliberate thinking
                if "uses_think" in rollout:
                    rollout_data["uses_think"] = rollout["uses_think"]
                
                all_rollouts.append(rollout_data)
        
        # Create outcome tracking for each rollout
        for rollout in all_rollouts:
            outcome = {
                **rollout,
                "next_day_headlines": get_headlines_for(rollout.get("matures_on")),
                "headlines_count": len(get_headlines_for(rollout.get("matures_on"))),
                "tracking_created_at": datetime.now().isoformat(),
                "status": "pending_evaluation"
            }
            
            outcomes.append(outcome)
        
        return outcomes
    
    def save_outcome_tracking(self, outcomes: List[Dict[str, Any]], prediction_date: str):
        """Save outcome tracking data."""
        data = {
            "date": prediction_date,
            "total_outcomes": len(outcomes),
            "outcomes": outcomes,
            "summary": self._create_outcome_summary(outcomes),
            "saved_at": datetime.now().isoformat()
        }
        
        filename = self.storage.save_data(data, "outcome_tracking", prediction_date)
        
        logger.info(f"Saved {len(outcomes)} outcome tracking entries for {prediction_date}")
        return filename
    
    def load_outcome_tracking(self, prediction_date: str) -> Optional[Dict[str, Any]]:
        """Load outcome tracking data for a specific date."""
        return self.storage.load_data("outcome_tracking", prediction_date)

    def track_outcomes_due_on(self, due_date: str, lookback_days: int = 7) -> List[Dict[str, Any]]:
        """Collect and prepare outcomes for all rollouts that mature on a given date.

        Scans predictions from the previous `lookback_days` calendar days and selects
        rollouts whose `matures_on == due_date`. Useful for multi-horizon evaluation.
        """
        try:
            from datetime import datetime, timedelta
            end = datetime.strptime(due_date, "%Y%m%d")
            start = end - timedelta(days=max(1, int(lookback_days)))
            start_str = start.strftime("%Y%m%d")
            preds_range = self.storage.get_date_range(start_str, due_date, "predictions")
        except Exception as e:
            logger.error(f"Failed to scan predictions for due date {due_date}: {e}")
            return []

        # Build a memoized headlines getter for the due date
        cache: Dict[str, List[Dict[str, Any]]] = {}
        def _get_heads(d: str) -> List[Dict[str, Any]]:
            if d in cache:
                return cache[d]
            cache[d] = self._get_next_day_headlines(d) or []
            return cache[d]

        outcomes: List[Dict[str, Any]] = []
        for pd in preds_range:
            pdate = pd.get("date")
            if not pdate or "predictions" not in pd:
                continue
            for prediction in pd["predictions"]:
                headline = prediction.get("headline", "")
                for r in prediction.get("rollouts", []):
                    if r.get("matures_on") != due_date:
                        continue
                    record = {
                        "prediction_id": f"{pdate}_{headline[:20].replace(' ', '_')}_{r.get('rollout_id')}",
                        "headline": headline,
                        "original_prediction": r.get("prediction"),
                        "rollout_id": r.get("rollout_id"),
                        "method": r.get("method"),
                        "immediate_reward": r.get("immediate_reward", 0.0),
                        "timestamp": r.get("timestamp"),
                        "date": pdate,
                        "horizon": r.get("horizon"),
                        "offset_days": r.get("offset_days"),
                        "matures_on": r.get("matures_on"),
                        "next_day_headlines": _get_heads(due_date),
                        "headlines_count": len(_get_heads(due_date)),
                        "tracking_created_at": datetime.now().isoformat(),
                        "status": "pending_evaluation"
                    }
                    if "trade_thinking_score" in r:
                        record["trade_thinking_score"] = r["trade_thinking_score"]
                    if "uses_think" in r:
                        record["uses_think"] = r["uses_think"]
                    outcomes.append(record)

        logger.info(f"Prepared {len(outcomes)} outcomes due on {due_date}")
        return outcomes
    
    def get_pending_evaluations(self, prediction_date: str) -> List[Dict[str, Any]]:
        """Get outcome tracking data that needs evaluation."""
        tracking_data = self.load_outcome_tracking(prediction_date)
        
        if not tracking_data:
            logger.warning(f"No outcome tracking found for date: {prediction_date}")
            return []
        
        # Filter for pending evaluations
        pending = []
        for outcome in tracking_data["outcomes"]:
            if outcome.get("status") == "pending_evaluation":
                pending.append(outcome)
        
        logger.info(f"Found {len(pending)} pending evaluations for {prediction_date}")
        return pending
    
    def update_evaluation_status(self, prediction_date: str, prediction_id: str, status: str):
        """Update evaluation status for a specific prediction."""
        tracking_data = self.load_outcome_tracking(prediction_date)
        
        if not tracking_data:
            logger.warning(f"No outcome tracking found for date: {prediction_date}")
            return
        
        # Find and update the specific outcome
        for outcome in tracking_data["outcomes"]:
            if outcome["prediction_id"] == prediction_id:
                outcome["status"] = status
                outcome["status_updated_at"] = datetime.now().isoformat()
                break
        
        # Save updated tracking data
        self.storage.save_data(tracking_data, "outcome_tracking", prediction_date)
        
        logger.info(f"Updated status for {prediction_id} to {status}")
    
    def _create_outcome_summary(self, outcomes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create summary statistics for outcome tracking."""
        if not outcomes:
            return {"error": "No outcomes provided"}
        
        # Count by method
        methods_used = {}
        for outcome in outcomes:
            method = outcome.get("method", "unknown")
            methods_used[method] = methods_used.get(method, 0) + 1
        
        # Calculate average immediate rewards
        immediate_rewards = [outcome.get("immediate_reward", 0) for outcome in outcomes]
        avg_immediate_reward = sum(immediate_rewards) / len(immediate_rewards) if immediate_rewards else 0.0
        
        # Count headlines
        total_headlines = sum(outcome.get("headlines_count", 0) for outcome in outcomes)
        avg_headlines = total_headlines / len(outcomes) if outcomes else 0
        
        return {
            "total_outcomes": len(outcomes),
            "methods_used": methods_used,
            "avg_immediate_reward": avg_immediate_reward,
            "total_headlines": total_headlines,
            "avg_headlines_per_outcome": avg_headlines,
            "reward_range": {
                "min": min(immediate_rewards) if immediate_rewards else 0.0,
                "max": max(immediate_rewards) if immediate_rewards else 0.0
            }
        }
    
    def get_outcome_statistics(self, start_date: str, end_date: str) -> Dict[str, Any]:
        """Get statistics for outcome tracking across a date range."""
        tracking_data = self.storage.get_date_range(start_date, end_date, "outcome_tracking")
        
        total_outcomes = 0
        total_headlines = 0
        methods_used = {}
        immediate_rewards = []
        
        for data in tracking_data:
            if "outcomes" in data:
                for outcome in data["outcomes"]:
                    total_outcomes += 1
                    total_headlines += outcome.get("headlines_count", 0)
                    
                    method = outcome.get("method", "unknown")
                    methods_used[method] = methods_used.get(method, 0) + 1
                    
                    immediate_rewards.append(outcome.get("immediate_reward", 0))
        
        return {
            "date_range": {"start": start_date, "end": end_date},
            "total_outcomes": total_outcomes,
            "total_headlines": total_headlines,
            "methods_used": methods_used,
            "avg_immediate_reward": sum(immediate_rewards) / len(immediate_rewards) if immediate_rewards else 0.0,
            "reward_range": {
                "min": min(immediate_rewards) if immediate_rewards else 0.0,
                "max": max(immediate_rewards) if immediate_rewards else 0.0
            }
        }

def main():
    """Test the outcome tracker."""
    tracker = OutcomeTracker()
    
    # Test with a sample date
    test_date = "20240115"
    
    # Track outcomes
    outcomes = tracker.track_outcomes_for_date(test_date)
    
    if outcomes:
        # Save outcome tracking
        filename = tracker.save_outcome_tracking(outcomes, test_date)
        print(f"Saved outcome tracking to: {filename}")
        
        # Get pending evaluations
        pending = tracker.get_pending_evaluations(test_date)
        print(f"Pending evaluations: {len(pending)}")
        
        # Get statistics
        stats = tracker.get_outcome_statistics("20240115", "20240116")
        print(f"Statistics: {stats}")
    else:
        print("No outcomes to track")

if __name__ == "__main__":
    main() 
