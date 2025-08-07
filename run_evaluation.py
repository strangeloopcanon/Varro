#!/usr/bin/env python3
"""
Run evaluations using the new find_best approach
"""

import json
import logging
from outcome_tracking.llm_outcome_evaluator import LLMOutcomeEvaluator
from data_collection.timestamped_storage import TimestampedStorage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_evaluation(date: str):
    """Run evaluation for the specified date."""
    storage = TimestampedStorage()
    
    # Load outcome tracking data
    outcome_data = storage.load_data('outcome_tracking', date)
    if not outcome_data or 'outcome_tracking' not in outcome_data:
        logger.error(f"No outcome tracking data found for {date}")
        return
    
    outcome_tracking = outcome_data['outcome_tracking']
    logger.info(f"Loaded {len(outcome_tracking)} outcome tracking entries for {date}")
    
    # Initialize evaluator with find_best approach
    evaluator = LLMOutcomeEvaluator()
    
    # Run evaluations
    evaluations = evaluator.evaluate_outcomes(outcome_tracking)
    
    if evaluations:
        # Save results
        storage.save_data({'evaluations': evaluations}, 'evaluations', date)
        logger.info(f"Saved {len(evaluations)} evaluations for {date}")
        
        # Print summary
        successful_groups = len(evaluations) // 8
        logger.info(f"Successfully evaluated {successful_groups} prediction groups")
        
        # Check for non-sequential rankings
        sequential_count = 0
        non_sequential_count = 0
        
        for i in range(0, len(evaluations), 8):
            group = evaluations[i:i+8]
            rankings = [eval['ranking'] for eval in group]
            if rankings == list(range(1, 9)):
                sequential_count += 1
            else:
                non_sequential_count += 1
        
        logger.info(f"Sequential rankings: {sequential_count}")
        logger.info(f"Non-sequential rankings: {non_sequential_count}")
        
    else:
        logger.warning("No evaluations completed")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        date = sys.argv[1]
    else:
        date = "20250802"
    
    run_evaluation(date) 