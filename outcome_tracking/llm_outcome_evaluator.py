#!/usr/bin/env python3
"""
LLM Outcome Evaluator
Evaluate predictions against next-day headlines using LLM-based evaluation.
"""

import json
import logging
import re
from datetime import datetime
from typing import List, Dict, Any
import mlx.core as mx
from mlx_lm import load, generate as mlx_generate
from mlx_lm.sample_utils import make_sampler
try:
    from analysis.trade_thinking import score_trade_thinking
except Exception:
    score_trade_thinking = None

logger = logging.getLogger(__name__)

class LLMOutcomeEvaluator:
    """Evaluates predictions against next-day headlines using LLM."""
    
    def __init__(self, model_name: str = "Qwen/Qwen3-0.6B"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.sampler = None
        
        # Load model and tokenizer
        self._load_model()
        
        # Load evaluation prompt from config
        self._load_evaluation_prompt()
    
    def _load_model(self):
        """Load the MLX model and tokenizer."""
        try:
            logger.info(f"Loading model: {self.model_name}")
            self.model, self.tokenizer = load(self.model_name)
            
            # Create sampler (stochastic phase)
            self.sampler = make_sampler(temp=0.3, top_p=0.9, top_k=50)
            
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def _load_evaluation_prompt(self):
        """Load evaluation prompt from config file."""
        try:
            with open("config/prompt_templates.json", 'r') as f:
                prompt_config = json.load(f)
            
            self.evaluation_prompt = prompt_config["evaluation_prompt"]
            logger.info("Loaded evaluation prompt from config")
            
        except Exception as e:
            logger.error(f"Error loading evaluation prompt: {e}")
            # Fallback to hardcoded prompt
            self.evaluation_prompt = """Prediction: {prediction}

Next day's headlines:
{headlines}

Based on these headlines, how accurate was this prediction?

Rate from 0-10 and explain:
- 0-3: Prediction was wrong or irrelevant
- 4-6: Prediction was partially correct or too vague  
- 7-8: Prediction was mostly accurate
- 9-10: Prediction was highly accurate and specific

Consider:
- Did the predicted market impact happen?
- Were the specific assets mentioned affected as predicted?
- Was the timeframe reasonable?
- Did the trade recommendation make sense given what actually happened?

Give your score and explanation."""
    
    def evaluate_prediction_group(self, predictions: List[str], next_day_headlines: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Evaluate a group of predictions using 'find the best' approach 8 times."""
        try:
            # Format headlines for evaluation
            headlines_text = self._format_headlines_for_evaluation(next_day_headlines)
            
            # Initialize tracking
            rankings = []
            used_predictions = set()
            
            # Run 8 rounds of "find the best"
            for round_num in range(8):
                # Create available predictions list with sequential letters
                # Now store original indices to avoid brittle string matching
                available_predictions = []  # List[Tuple[str, int, str]] = (letter, idx, pred)
                available_letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
                
                for i, pred in enumerate(predictions):
                    if i not in used_predictions:
                        # Map to next available letter sequentially
                        letter = available_letters[len(available_predictions)]
                        available_predictions.append((letter, i, pred))
                
                if not available_predictions:
                    break

                # Deterministic early-exit: if only one option remains, select it without querying the LLM
                if len(available_predictions) == 1:
                    only_letter, only_idx, only_pred = available_predictions[0]
                    rankings.append(only_idx)
                    used_predictions.add(only_idx)
                    logger.info(f"Round {round_num + 1}: Only one option left, selected {only_letter} (prediction {only_idx})")
                    continue
                
                # Build prompt for this round (truncate predictions to first 200 chars)
                predictions_text = "\n".join([f"({letter}) {pred[:200]}..." for letter, _, pred in available_predictions])
                
                find_best_prompt = f"""You are an expert financial analyst. Your task is to find the SINGLE BEST prediction from the available options that most accurately matches the provided headlines.

HEADLINES:
{headlines_text}

AVAILABLE PREDICTIONS (choose from these only):
{predictions_text}

Which prediction is MOST ACCURATE given these headlines?

Use this format:
<think>
[Your reasoning here]
</think>
[Single letter answer]

Answer:"""
                
                # Generate response with retry logic
                max_retries = 5
                selected_letter = None
                
                for attempt in range(max_retries):
                    response = mlx_generate(
                        self.model,
                        self.tokenizer,
                        find_best_prompt,
                        max_tokens=4,  # Keep very short to bias toward a single letter
                        sampler=self.sampler
                    )
                    
                    # Extract letter from response
                    selected_letter = self._extract_single_letter(response)
                    
                    # Check if letter is in available predictions for this round
                    available_letters = [letter for letter, _, _ in available_predictions]
                    
                    if selected_letter and selected_letter in available_letters:
                        # Directly map letter to its original prediction index
                        selected_prediction_idx = None
                        for letter, idx, _ in available_predictions:
                            if letter == selected_letter:
                                selected_prediction_idx = idx
                                break
                        if selected_prediction_idx is not None and selected_prediction_idx not in used_predictions:
                            rankings.append(selected_prediction_idx)
                            used_predictions.add(selected_prediction_idx)
                            logger.info(f"Round {round_num + 1}: Selected {selected_letter} (prediction {selected_prediction_idx})")
                            break
                        else:
                            logger.warning(f"Invalid selection {selected_letter} on attempt {attempt + 1}, retrying...")
                    else:
                        logger.warning(f"No valid letter found in response on attempt {attempt + 1}, retrying...")
                        logger.info(f"Raw response: '{response}'")
                        logger.info(f"Available letters: {available_letters}")
                
                if not selected_letter or selected_letter not in available_letters:
                    # Deterministic fallback: ask again with constrained instruction and greedy decoding
                    constrained_prompt = (
                        f"HEADLINES (context omitted)\n\n"
                        f"Available options: {', '.join(available_letters)}\n"
                        "Reply with exactly ONE letter from the options above.\n"
                        "Answer:"
                    )
                    deterministic_resp = mlx_generate(
                        self.model,
                        self.tokenizer,
                        constrained_prompt,
                        max_tokens=2,
                        sampler=None  # Greedy decoding
                    )
                    selected_letter = self._extract_single_letter(deterministic_resp)

                if not selected_letter or selected_letter not in available_letters:
                    # Final deterministic fallback: pick the first available option to ensure progress
                    selected_letter = available_letters[0]
                    logger.warning(f"Falling back to first available option: {selected_letter} in round {round_num + 1}")

                # Map the (possibly fallback) selected_letter to original prediction index
                selected_prediction_idx = None
                for letter, idx, _ in available_predictions:
                    if letter == selected_letter:
                        selected_prediction_idx = idx
                        break

                if selected_prediction_idx is not None:
                    rankings.append(selected_prediction_idx)
                    used_predictions.add(selected_prediction_idx)
                    logger.info(f"Round {round_num + 1}: Selected {selected_letter} (prediction {selected_prediction_idx})")
                else:
                    logger.error(f"Failed to map selected letter {selected_letter} to a prediction index in round {round_num + 1}")
                    break
            
            # Convert rankings to final format with strict validation (exactly 8, 1..8 ranks)
            def build_final_ranking(selected_indices: list[int]) -> list[int]:
                # Deduplicate while preserving order
                unique = []
                seen = set()
                for idx in selected_indices:
                    if 0 <= idx < 8 and idx not in seen:
                        unique.append(idx)
                        seen.add(idx)
                # Fill missing with remaining indices
                remaining = [i for i in range(8) if i not in seen]
                # Truncate to at most 8
                while len(unique) > 8:
                    unique.pop()
                # Compose final order: first picks + fillers
                final_order = unique + remaining
                final_order = final_order[:8]
                # Assign ranks: picked ones get 1..len(unique); fillers get rank 7
                final_ranking_local = [0] * 8
                picked_set = set(unique)
                for pos, pred_idx in enumerate(final_order):
                    if pred_idx in picked_set:
                        rank_val = pos + 1
                    else:
                        rank_val = 7
                    # Clamp rank to [1,8]
                    if rank_val < 1:
                        rank_val = 1
                    if rank_val > 8:
                        rank_val = 8
                    final_ranking_local[pred_idx] = rank_val
                # Final safety: replace any unset (0) with worst rank 8
                for i in range(8):
                    if final_ranking_local[i] == 0:
                        final_ranking_local[i] = 8
                return final_ranking_local

            if len(rankings) >= 4:
                final_ranking = build_final_ranking(rankings)
                logger.info(f"Generated validated ranking: {final_ranking}")
                return self._create_evaluation_results(predictions, final_ranking, next_day_headlines)
            else:
                logger.error(f"Only got {len(rankings)} rankings, need at least 4")
                return []
                
        except Exception as e:
            logger.error(f"Error in evaluate_prediction_group: {e}")
            return []
    
    def _extract_single_letter(self, response: str) -> str:
        """Extract a single letter (A-H) from the response."""
        # Clean response
        response_clean = response.replace("ANSWER", "").replace("answer", "").strip()
        
        # Handle empty responses
        if not response_clean:
            return None
        
        # Extract letter after </think> tag if present
        think_match = re.search(r'</think>\s*([A-H])', response_clean, re.IGNORECASE)
        if think_match:
            return think_match.group(1).upper()
        
        # Remove <think> tags and their content
        response_clean = re.sub(r'<think>.*?</think>', '', response_clean, flags=re.DOTALL | re.IGNORECASE)
        
        # Remove markdown code blocks
        response_clean = re.sub(r'```.*?```', '', response_clean, flags=re.DOTALL)
        response_clean = re.sub(r'```.*', '', response_clean, flags=re.DOTALL)
        
        # Remove JSON content
        response_clean = re.sub(r'\{.*?\}', '', response_clean, flags=re.DOTALL)
        response_clean = re.sub(r'```json.*?```', '', response_clean, flags=re.DOTALL)
        
        # Remove end-of-text tokens and content after them
        response_clean = re.sub(r'<\|endoftext\|>.*', '', response_clean)
        
        # Remove common extra text patterns
        response_clean = re.sub(r'\*\*Note:\*\*.*', '', response_clean, flags=re.IGNORECASE)
        response_clean = re.sub(r'Note:.*', '', response_clean, flags=re.IGNORECASE)
        response_clean = re.sub(r'Explanation:.*', '', response_clean, flags=re.IGNORECASE)
        response_clean = re.sub(r'Reasoning:.*', '', response_clean, flags=re.IGNORECASE)
        response_clean = re.sub(r'Answer:.*', '', response_clean, flags=re.IGNORECASE)
        response_clean = response_clean.strip()
        
        # Look for boxed format first
        if "\\boxed{" in response_clean:
            start = response_clean.find("\\boxed{") + 7
            end = response_clean.find("}", start)
            if start > 6 and end > start:
                letter = response_clean[start:end].strip().upper()
                if letter in 'ABCDEFGH':
                    return letter
        
        # Look for numbered responses like [1], [2], etc. and convert to letters
        number_match = re.search(r'\[(\d+)\]', response_clean)
        if number_match:
            try:
                number = int(number_match.group(1))
                if 1 <= number <= 8:
                    # Convert number to letter (1=A, 2=B, etc.)
                    letter = chr(64 + number)  # ASCII: A=65, B=66, etc.
                    return letter
            except ValueError:
                pass
        
        # Look for standalone numbers (1, 2, 3, etc.) and convert to letters
        number_match = re.search(r'\b(\d+)\b', response_clean)
        if number_match:
            try:
                number = int(number_match.group(1))
                if 1 <= number <= 8:
                    # Convert number to letter (1=A, 2=B, etc.)
                    letter = chr(64 + number)  # ASCII: A=65, B=66, etc.
                    return letter
            except ValueError:
                pass
        
        # Look for letters in parentheses like (A), (B), etc.
        paren_match = re.search(r'\(([A-H])\)', response_clean)
        if paren_match:
            return paren_match.group(1)
        
        # Look for letters in brackets like [A], [B], etc.
        bracket_matches = re.findall(r'\[([A-H])\]', response_clean)
        if bracket_matches:
            # Take the last letter found in brackets
            last_letter = bracket_matches[-1]
            return last_letter
        
        # Also handle formats like: "A." or "A -" at line start
        start_match = re.search(r'^[\s\-\*\(\[]*([A-H])[\)\]\.:\-\s]', response_clean, re.IGNORECASE)
        if start_match:
            return start_match.group(1).upper()

        # Look for "The answer is X" pattern
        answer_match = re.search(r'answer is ([A-H])', response_clean, re.IGNORECASE)
        if answer_match:
            return answer_match.group(1).upper()
        
        # Look for "I choose X" pattern
        choose_match = re.search(r'i choose ([A-H])', response_clean, re.IGNORECASE)
        if choose_match:
            return choose_match.group(1).upper()
        
        # Look for "Option X" pattern
        option_match = re.search(r'option ([A-H])', response_clean, re.IGNORECASE)
        if option_match:
            return option_match.group(1).upper()
        
        # Look for "X is the best" pattern
        best_match = re.search(r'([A-H]) is the best', response_clean, re.IGNORECASE)
        if best_match:
            return best_match.group(1).upper()
        
        # Look for "Selected X" pattern
        selected_match = re.search(r'selected ([A-H])', response_clean, re.IGNORECASE)
        if selected_match:
            return selected_match.group(1).upper()
        
        # Look for JSON with different key names
        json_answer_match = re.search(r'"answer":\s*"([A-H])"', response_clean, re.IGNORECASE)
        if json_answer_match:
            return json_answer_match.group(1).upper()
        
        json_prediction_match = re.search(r'"prediction":\s*"([A-H])"', response_clean, re.IGNORECASE)
        if json_prediction_match:
            return json_prediction_match.group(1).upper()
        
        json_correct_match = re.search(r'"correct_prediction":\s*"([A-H])"', response_clean, re.IGNORECASE)
        if json_correct_match:
            return json_correct_match.group(1).upper()
        
        # Handle 'M' as a potential typo for 'A' (common keyboard error)
        if 'M' in response_clean and 'A' not in response_clean and 'B' not in response_clean and 'C' not in response_clean and 'D' not in response_clean and 'E' not in response_clean and 'F' not in response_clean and 'G' not in response_clean and 'H' not in response_clean:
            return 'A'  # Assume 'M' is a typo for 'A'
        
        # Find the last letter in the response (most likely the answer)
        letters = [char for char in response_clean if char in 'ABCDEFGH']
        if letters:
            return letters[-1]  # Take the last letter found
        
        return None
    
    def _create_evaluation_results(self, predictions: List[str], ranking: List[int], next_day_headlines: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create evaluation results from ranking."""
        results = []
        for i, prediction in enumerate(predictions):
            # Calculate reward for this individual ranking
            rank = ranking[i]
            N = 8  # Total number of predictions
            reward = 1.0 - (rank - 1) / (N - 1) if N > 1 else 1.0

            # Optional: trade-thinking score from text
            s = score_trade_thinking(prediction, horizon="next_day") if score_trade_thinking is not None else None
            # Composite reward (soft AND) if trade-thinking available
            composite = None
            if s is not None:
                w = 0.7  # weight on LLM reward
                # Guard rails
                r = max(0.0, min(1.0, reward))
                s = max(0.0, min(1.0, s))
                composite = (r ** w) * (s ** (1.0 - w))
            
            result = {
                'prediction': prediction,
                'ranking': ranking[i],
                'reward': reward,
                **({ 'trade_thinking_score': s } if s is not None else {}),
                **({ 'composite_reward': composite } if composite is not None else {}),
                'headlines': next_day_headlines,
                'evaluation_method': 'find_best_8_rounds'
            }
            results.append(result)
        return results
    
    def evaluate_outcomes(self, outcome_tracking: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Evaluate multiple outcomes from outcome tracking with stack ranking."""
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
    
    def _format_headlines_for_evaluation(self, headlines: List[Dict[str, Any]]) -> str:
        """Format headlines for evaluation prompt."""
        if not headlines:
            return "No headlines available for evaluation."
        
        # Take first 10 headlines to avoid token limits
        limited_headlines = headlines[:10]
        
        formatted_headlines = []
        for i, headline in enumerate(limited_headlines, 1):
            text = headline.get("text", "")
            source = headline.get("source", "Unknown")
            formatted_headlines.append(f"{i}. {text} ({source})")
        
        return "\n".join(formatted_headlines)
    
    def _extract_ranking(self, ranking_response: str, num_predictions: int) -> List[int]:
        """Extract ranking from LLM response with letter-based format."""
        try:
            import re
            
            # Clean the response
            response = ranking_response.strip()
            
            # Look for top 4 letter-based ranking patterns like "A,B,C,D" or "F,B,D,A"
            letter_pattern_4 = r'([A-H]),\s*([A-H]),\s*([A-H]),\s*([A-H])'
            match = re.search(letter_pattern_4, response)
            
            if match:
                letters = match.groups()
                # Convert letters to numbers (A=1, B=2, etc.)
                letter_to_num = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'H': 8}
                top_4_ranking = [letter_to_num[letter] for letter in letters]
                
                # Validate top 4 ranking (should be 4 unique numbers from 1-8)
                if len(top_4_ranking) == 4 and len(set(top_4_ranking)) == 4 and min(top_4_ranking) >= 1 and max(top_4_ranking) <= 8:
                    # Convert to full 8-item ranking: top 4 as ranked (1-4), bottom 4 as sequential
                    # The top_4_ranking contains the original positions that should get ranks 1-4
                    # We need to create a mapping: original_position -> rank
                    position_to_rank = {}
                    for rank, original_pos in enumerate(top_4_ranking, 1):
                        position_to_rank[original_pos] = rank
                    
                    # Create full ranking: top 4 get ranks 1-4, rest get rank 5 (flat)
                    ranking = []
                    for pos in range(1, 9):
                        if pos in position_to_rank:
                            ranking.append(position_to_rank[pos])  # 1-4 for top 4
                        else:
                            ranking.append(5)  # 5 for all others (flat)
                    logger.info("Successfully extracted top 4 ranking from letter format")
                    return ranking
            
            # Look for top 4 letters without spaces
            letter_pattern_4_no_spaces = r'([A-H]),([A-H]),([A-H]),([A-H])'
            match = re.search(letter_pattern_4_no_spaces, response)
            
            if match:
                letters = match.groups()
                letter_to_num = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'H': 8}
                top_4_ranking = [letter_to_num[letter] for letter in letters]
                
                if len(top_4_ranking) == 4 and len(set(top_4_ranking)) == 4 and min(top_4_ranking) >= 1 and max(top_4_ranking) <= 8:
                    # Convert to full 8-item ranking: top 4 as ranked (1-4), bottom 4 as sequential
                    # The top_4_ranking contains the original positions that should get ranks 1-4
                    # We need to create a mapping: original_position -> rank
                    position_to_rank = {}
                    for rank, original_pos in enumerate(top_4_ranking, 1):
                        position_to_rank[original_pos] = rank
                    
                    # Create full ranking: top 4 get ranks 1-4, rest get rank 5 (flat)
                    ranking = []
                    for pos in range(1, 9):
                        if pos in position_to_rank:
                            ranking.append(position_to_rank[pos])  # 1-4 for top 4
                        else:
                            ranking.append(5)  # 5 for all others (flat)
                    logger.info("Successfully extracted top 4 ranking from letter format (no spaces)")
                    return ranking
            
            # Look for just letters in sequence
            letters_found = re.findall(r'\b([A-H])\b', response)
            if len(letters_found) >= num_predictions:
                letter_to_num = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'H': 8}
                ranking = [letter_to_num[letter] for letter in letters_found[:num_predictions]]
                
                if len(set(ranking)) == num_predictions and min(ranking) >= 1 and max(ranking) <= num_predictions:
                    logger.info("Successfully extracted ranking from letter sequence")
                    return ranking
            
            # Final fallback: sequential ranking
            logger.warning(f"Could not extract ranking from response: '{response[:100]}...', using sequential fallback")
            return list(range(1, num_predictions + 1))
            
        except Exception as e:
            logger.error(f"Error extracting ranking: {e}")
            return list(range(1, num_predictions + 1))
    
    def _convert_ranking_to_rewards(self, ranking: List[int]) -> List[float]:
        """Convert ranking to rewards using the specified formula."""
        N = len(ranking)
        rewards = []
        
        for rank in ranking:
            # Simple linear: r_k = 1 - (k-1)/(N-1)
            reward = 1.0 - (rank - 1) / (N - 1)
            rewards.append(reward)
        
        return rewards
    
    def _extract_score_and_explanation(self, evaluation: str) -> tuple:
        """Extract score and explanation from LLM evaluation."""
        try:
            # Look for score patterns
            score_patterns = [
                r"score[:\s]*(\d+(?:\.\d+)?)",
                r"rating[:\s]*(\d+(?:\.\d+)?)",
                r"(\d+(?:\.\d+)?)/10",
                r"(\d+(?:\.\d+)?) out of 10",
                r"(\d+(?:\.\d+)?) out of ten"
            ]
            
            score = None
            for pattern in score_patterns:
                match = re.search(pattern, evaluation, re.IGNORECASE)
                if match:
                    score = float(match.group(1))
                    break
            
            # If no score found, try to extract from text
            if score is None:
                # Look for numbers that could be scores
                numbers = re.findall(r'\b(\d+(?:\.\d+)?)\b', evaluation)
                if numbers:
                    # Take the first number that could be a score (0-10 range)
                    for num in numbers:
                        num_float = float(num)
                        if 0 <= num_float <= 10:
                            score = num_float
                            break
            
            # Default score if none found
            if score is None:
                score = 5.0  # Neutral score
            
            # Extract explanation (everything after the score)
            explanation = evaluation
            
            return score, explanation
            
        except Exception as e:
            logger.error(f"Error extracting score and explanation: {e}")
            return 5.0, f"Error parsing evaluation: {str(e)}"
    
    def batch_evaluate(self, outcome_tracking: List[Dict[str, Any]], 
                      batch_size: int = 5) -> List[Dict[str, Any]]:
        """Evaluate outcomes with group-first batching.

        Behavior change: We first group all outcomes by headline (ensuring each group
        contains up to 8 rollouts), then batch by groups. This prevents fragmenting a
        headline's rollouts across different batches, which previously caused partial
        evaluations and failures.

        Args:
            outcome_tracking: Flat list of outcome rows for a date.
            batch_size: Number of headline groups per batch (default: 5).

        Returns:
            A flat list of evaluation results.
        """
        if not outcome_tracking:
            return []

        # Group first across the entire dataset
        headline_groups: Dict[str, List[Dict[str, Any]]] = {}
        for outcome in outcome_tracking:
            headline = outcome.get("headline")
            if not headline:
                # Skip malformed rows without headline key
                continue
            headline_groups.setdefault(headline, []).append(outcome)

        grouped = list(headline_groups.values())
        all_evaluations: List[Dict[str, Any]] = []

        # Batch by groups (headlines)
        for i in range(0, len(grouped), batch_size):
            batch_groups = grouped[i:i + batch_size]
            # Flatten outcomes for this batch so evaluate_outcomes can re-group and process
            batch_flat = [row for group in batch_groups for row in group]
            logger.info(
                f"Evaluating batch {i//batch_size + 1} (groups: {len(batch_groups)}, predictions: {len(batch_flat)})"
            )
            batch_evaluations = self.evaluate_outcomes(batch_flat)
            all_evaluations.extend(batch_evaluations)

        return all_evaluations
    
    def get_evaluation_summary(self, evaluations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get summary statistics for evaluations."""
        if not evaluations:
            return {"error": "No evaluations provided"}
        
        # Extract scores
        outcome_scores = [eval.get("outcome_score", 0) for eval in evaluations]
        normalized_scores = [eval.get("normalized_score", 0) for eval in evaluations]
        
        # Calculate statistics
        avg_outcome_score = sum(outcome_scores) / len(outcome_scores) if outcome_scores else 0.0
        avg_normalized_score = sum(normalized_scores) / len(normalized_scores) if normalized_scores else 0.0
        
        # Count by score ranges
        score_ranges = {
            "0-3": sum(1 for score in outcome_scores if 0 <= score <= 3),
            "4-6": sum(1 for score in outcome_scores if 4 <= score <= 6),
            "7-8": sum(1 for score in outcome_scores if 7 <= score <= 8),
            "9-10": sum(1 for score in outcome_scores if 9 <= score <= 10)
        }
        
        # Count by method
        methods_used = {}
        for evaluation in evaluations:
            method = evaluation.get("method", "unknown")
            methods_used[method] = methods_used.get(method, 0) + 1
        
        return {
            "total_evaluations": len(evaluations),
            "avg_outcome_score": avg_outcome_score,
            "avg_normalized_score": avg_normalized_score,
            "score_distribution": score_ranges,
            "methods_used": methods_used,
            "score_range": {
                "min": min(outcome_scores) if outcome_scores else 0.0,
                "max": max(outcome_scores) if outcome_scores else 0.0
            }
        }

def main():
    """Test the LLM outcome evaluator."""
    evaluator = LLMOutcomeEvaluator()
    
    # Test with sample data
    test_prediction = """
**Market Impact**: Fed actions will significantly impact bond yields and risk assets.
**Specific Assets**: Treasury bonds, bank stocks, and the dollar will be most affected.
**Trade Recommendation**: Buy TLT calls if dovish, short bank stocks if hawkish.
**Timeframe**: Next 1-3 days
**Risk Factors**: Fed could walk back comments or data could surprise.
**World View**: This suggests the Fed is either maintaining or changing its monetary stance.
"""
    
    test_headlines = [
        {"text": "Fed Chair Powell signals potential rate cuts in March", "source": "Reuters"},
        {"text": "Treasury yields fall to 3-month lows", "source": "MarketWatch"},
        {"text": "Tech stocks rally on Fed dovishness", "source": "CNN"},
        {"text": "Bond market celebrates Fed pivot", "source": "Bloomberg"},
        {"text": "S&P 500 hits new highs on Fed comments", "source": "CNBC"}
    ]
    
    # Evaluate prediction
    evaluation = evaluator.evaluate_prediction(test_prediction, test_headlines)
    
    print(f"Evaluation result: {evaluation}")
    
    # Test batch evaluation
    test_outcomes = [
        {
            "prediction_id": "test_001",
            "original_prediction": test_prediction,
            "next_day_headlines": test_headlines,
            "method": "basic_stochastic"
        }
    ]
    
    batch_evaluations = evaluator.batch_evaluate(test_outcomes)
    print(f"Batch evaluations: {len(batch_evaluations)}")
    
    # Get summary
    summary = evaluator.get_evaluation_summary(batch_evaluations)
    print(f"Evaluation summary: {summary}")

if __name__ == "__main__":
    main() 
