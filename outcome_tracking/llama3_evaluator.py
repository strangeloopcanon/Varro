#!/usr/bin/env python3
"""
Llama3-based Outcome Evaluator for better ranking results
"""

import json
import logging
import requests
import re
from typing import List, Dict, Any
from outcome_tracking.llm_outcome_evaluator import LLMOutcomeEvaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Llama3OutcomeEvaluator(LLMOutcomeEvaluator):
    """Llama3-based evaluator using Ollama API."""
    
    def __init__(self, model_name: str = "phi4-mini:latest"):
        self.model_name = model_name
        self.base_url = "http://localhost:11434"
        self._load_evaluation_prompt()
    
    def _load_evaluation_prompt(self):
        """Load evaluation prompt from config or use default."""
        try:
            with open("config/prompt_templates.json", 'r') as f:
                prompts = json.load(f)
            self.evaluation_prompt = prompts.get("evaluation_prompt", self._get_default_prompt())
        except Exception as e:
            logger.warning(f"Could not load evaluation prompt from config: {e}")
            self.evaluation_prompt = self._get_default_prompt()
    
    def _get_default_prompt(self) -> str:
        """Get default evaluation prompt."""
        return """You are evaluating 8 predictions about market events. Each prediction is labeled A-H.

Given the actual headlines that occurred the next day, rank the TOP 4 most accurate predictions from A-H.

IMPORTANT: Respond with ONLY the ranking in this exact format: [1,2,3,4,5,6,7,8]
Where the top 4 positions (1-4) correspond to the most accurate predictions, and positions 5-8 get rank 5.

Example: If predictions A, C, F, B are most accurate, respond: [1,4,3,6,5,2,5,5]"""

    def _call_llama3(self, prompt: str) -> str:
        """Call Llama3 via Ollama API."""
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "temperature": 0.1,
                    "top_p": 0.9
                },
                timeout=120
            )
            
            if response.status_code == 200:
                return response.json()["response"]
            else:
                logger.error(f"Ollama API error: {response.status_code}")
                return ""
                
        except Exception as e:
            logger.error(f"Error calling Llama3: {e}")
            return ""

    def evaluate_prediction_group(self, predictions: List[str], next_day_headlines: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Evaluate a group of 8 predictions using Llama3."""
        if len(predictions) != 8:
            logger.error(f"Expected 8 predictions, got {len(predictions)}")
            return None
        
        # Format predictions as A-H
        prediction_text = ""
        for i, pred in enumerate(predictions):
            letter = chr(65 + i)  # A, B, C, D, E, F, G, H
            prediction_text += f"{letter}. {pred}\n\n"
        
        # Format next-day headlines
        headlines_text = self._format_headlines_for_evaluation(next_day_headlines)
        
        # Create evaluation prompt
        evaluation_prompt = f"{self.evaluation_prompt}\n\nPREDICTIONS:\n{prediction_text}\n\nACTUAL HEADLINES:\n{headlines_text}\n\nRank the TOP 4 most accurate predictions:"
        
        # Get ranking from Llama3
        ranking_response = self._call_llama3(evaluation_prompt)
        
        if not ranking_response:
            logger.error("Failed to get response from Llama3")
            return None
        
        # Extract ranking
        ranking = self._extract_ranking(ranking_response)
        if not ranking:
            logger.error(f"Failed to extract ranking from response: {ranking_response[:100]}...")
            return None
        
        # Convert ranking to rewards
        rewards = self._convert_ranking_to_rewards(ranking)
        
        # Create evaluation results
        evaluations = []
        for i, (prediction, reward) in enumerate(zip(predictions, rewards)):
            evaluation = {
                "prediction": prediction,
                "ranking": ranking[i],
                "reward": reward,
                "evaluation_method": "llama3_stack_ranking"
            }
            evaluations.append(evaluation)
        
        logger.info(f"Successfully evaluated group with ranking: {ranking}")
        return evaluations

    def _extract_ranking(self, response: str) -> List[int]:
        """Extract ranking from response with improved parsing."""
        try:
            # Find array pattern
            start = response.find('[')
            end = response.rfind(']') + 1
            
            if start != -1 and end > start:
                array_str = response[start:end]
                
                # Try to parse as JSON
                try:
                    ranking = json.loads(array_str)
                except json.JSONDecodeError:
                    # Try to extract numbers manually with more flexible patterns
                    numbers = re.findall(r'\d+', array_str)
                    if len(numbers) >= 8:
                        # Take first 8 numbers
                        ranking = [int(n) for n in numbers[:8]]
                    else:
                        # Try to find numbers in the entire response
                        all_numbers = re.findall(r'\d+', response)
                        if len(all_numbers) >= 8:
                            ranking = [int(n) for n in all_numbers[:8]]
                        else:
                            return None
                
                # Validate ranking
                if (len(ranking) == 8 and 
                    all(isinstance(x, int) for x in ranking) and
                    min(ranking) >= 1 and max(ranking) <= 8):
                    return ranking
                    
            return None
                
        except Exception as e:
            logger.error(f"Error extracting ranking: {e}")
            return None

    def _convert_ranking_to_rewards(self, ranking: List[int]) -> List[float]:
        """Convert ranking to dense rewards."""
        N = len(ranking)
        rewards = []
        
        for rank in ranking:
            # Dense reward: r_k = 1 - (k-1)/(N-1)
            reward = 1.0 - (rank - 1) / (N - 1)
            rewards.append(reward)
        
        return rewards

    def evaluate_outcomes(self, outcome_tracking: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Evaluate multiple outcomes using Llama3."""
        evaluations = []
        
        # Group predictions by headline (all 8 rollouts for each headline)
        headline_groups = {}
        for outcome in outcome_tracking:
            headline = outcome["headline"]
            if headline not in headline_groups:
                headline_groups[headline] = []
            headline_groups[headline].append(outcome)
        
        # Evaluate each group of 8 rollouts together
        for headline, group_outcomes in headline_groups.items():
            logger.info(f"Evaluating group for headline: {headline[:50]}...")
            
            # Extract predictions and next-day headlines
            predictions = [outcome["original_prediction"] for outcome in group_outcomes]
            next_day_headlines = group_outcomes[0]["next_day_headlines"]  # Same for all in group
            
            # Evaluate the group with stack ranking
            group_evaluations = self.evaluate_prediction_group(predictions, next_day_headlines)
            
            # Check if evaluation failed
            if group_evaluations is None:
                logger.error(f"Evaluation failed for headline: {headline[:50]}...")
                # Skip this group - don't add to evaluations
                continue
            
            # Combine with outcome tracking data
            for outcome, evaluation in zip(group_outcomes, group_evaluations):
                evaluation_result = {
                    **outcome,
                    **evaluation,
                    "status": "evaluated"
                }
                evaluations.append(evaluation_result)
        
        logger.info(f"Evaluated {len(evaluations)} predictions in {len(headline_groups)} groups")
        return evaluations 