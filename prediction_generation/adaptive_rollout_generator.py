#!/usr/bin/env python3
"""
Adaptive Rollout Generator
Generates 8 rollouts per headline with basic stochastic or advanced perspective-based approaches.
"""

import json
import logging
import os
from datetime import datetime
from typing import List, Dict, Any
import mlx.core as mx
from mlx_lm import load, generate as mlx_generate
from mlx_lm.sample_utils import make_sampler
import re
try:
    # Optional: trade-thinking rubric scorer (analysis/trade_thinking.py)
    from analysis.trade_thinking import score_trade_thinking
except Exception:
    score_trade_thinking = None

# Removed diversity assessor import since we always use same prompt

logger = logging.getLogger(__name__)

class AdaptiveRolloutGenerator:
    """Generates diverse rollouts for financial predictions."""
    
    def __init__(self, model_name: str = "Qwen/Qwen3-0.6B", checkpoint_path: str = None):
        self.model_name = model_name
        self.checkpoint_path = checkpoint_path
        self.model = None
        self.tokenizer = None
        self.sampler = None
        # Removed diversity assessor since we always use same prompt
        
        # Load model and tokenizer
        self._load_model()
        
        # Load prompt templates from config
        self._load_prompt_templates()
    
    def _load_model(self):
        """Load the MLX model and tokenizer."""
        try:
            if self.checkpoint_path:
                logger.info(f"Loading trained model from checkpoint: {self.checkpoint_path}")
                # Load base model first, then load trained weights
                self.model, self.tokenizer = load(self.model_name)
                self.model.load_weights(os.path.join(self.checkpoint_path, "model.safetensors.npz"))
                logger.info("Loaded trained weights successfully")
            else:
                logger.info(f"Loading base model: {self.model_name}")
                self.model, self.tokenizer = load(self.model_name)
            
            # Create sampler
            self.sampler = make_sampler(temp=0.8, top_p=0.95, top_k=50)
            
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def _load_prompt_templates(self):
        """Load prompt templates from config file."""
        try:
            with open("config/prompt_templates.json", 'r') as f:
                prompt_config = json.load(f)
            
            self.basic_prompt = prompt_config["basic_prompt"]
            self.perspective_prompts = prompt_config["perspective_prompts"]
            
            logger.info("Loaded prompt templates from config")
            
        except Exception as e:
            logger.error(f"Error loading prompt templates: {e}")
            # Fallback to hardcoded prompts
            self.basic_prompt = """Headline: "{headline}"

Based on this news, what do you think will happen and what's the trade?

Please structure your response as follows:
1. **Market Impact**: How will this affect markets broadly?
2. **Specific Assets**: Which specific assets will be most affected?
3. **Trade Recommendation**: What specific trade would you make?
4. **Timeframe**: When do you expect this to play out?
5. **Risk Factors**: What could go wrong with this prediction?
6. **World View**: What world does this news imagine or create? What narrative does it suggest?

Think like a trader - be specific about what to buy/sell and why."""
            
            self.perspective_prompts = {
                "fundamental_analyst": "Headline: \"{headline}\"\n\nThink like a fundamental analyst...",
                "technical_trader": "Headline: \"{headline}\"\n\nThink like a technical trader...",
                "macro_economist": "Headline: \"{headline}\"\n\nThink like a macro economist..."
            }
    
    def generate_rollouts_for_headline(self, headline: str, num_rollouts: int = 8, horizon: str | None = None) -> List[Dict[str, Any]]:
        """Generate N rollouts for a single headline using same prompt with sampling diversity.

        Args:
            headline: Headline text to condition on
            num_rollouts: Number of rollouts to generate (default 8)
            horizon: Optional horizon directive: "next_day", "next_month", or "next_year"
        """
        logger.info(f"Generating rollouts for headline: {headline[:50]}...")

        rollouts = self._generate_basic_rollouts(headline, num_rollouts=num_rollouts, horizon=horizon)
        logger.info(f"Generated {len(rollouts)} rollouts using same prompt with sampling diversity")
        return rollouts
    
    def _generate_basic_rollouts(self, headline: str, num_rollouts: int = 8, horizon: str | None = None) -> List[Dict[str, Any]]:
        """Generate N rollouts using basic stochastic approach with optional horizon directive."""
        rollouts: List[Dict[str, Any]] = []

        # Optional horizon directive to bias timeframe
        horizon_prefix = self._build_horizon_directive(horizon) if horizon else ""

        for i in range(max(1, int(num_rollouts))):
            # Use same prompt, let MLX sampler create diversity
            base_prompt = self.basic_prompt.format(headline=headline)
            prompt = f"{horizon_prefix}{base_prompt}" if horizon_prefix else base_prompt

            try:
                raw_prediction = mlx_generate(
                    self.model,
                    self.tokenizer,
                    prompt,
                    max_tokens=512,
                    sampler=self.sampler
                )

                # Clean prompt-echo and scaffold noise
                prediction = self._clean_prediction_response(raw_prediction)

                # Calculate immediate structure-based reward
                immediate_reward = self._calculate_structure_reward(prediction)

                # Optional: trade-thinking score (for later analysis). Does not affect selection or reward.
                trade_thinking = None
                if score_trade_thinking is not None:
                    trade_thinking = score_trade_thinking(prediction, horizon=horizon)

                rollouts.append({
                    "rollout_id": i,
                    "prediction": prediction,
                    "method": "basic_stochastic",
                    "immediate_reward": immediate_reward,
                    **({"trade_thinking_score": trade_thinking} if trade_thinking is not None else {}),
                    "timestamp": datetime.now().isoformat(),
                    **({"horizon": horizon} if horizon else {})
                })

            except Exception as e:
                logger.error(f"Error generating rollout {i}: {e}")
                continue

        return rollouts
    
    def _generate_advanced_rollouts(self, headline: str) -> List[Dict[str, Any]]:
        """Generate 8 rollouts using different perspectives."""
        rollouts = []
        perspectives = list(self.perspective_prompts.keys())
        
        for i in range(8):
            perspective = perspectives[i]
            prompt_template = self.perspective_prompts[perspective]
            prompt = prompt_template.format(headline=headline)
            
            try:
                prediction = mlx_generate(
                    self.model,
                    self.tokenizer,
                    prompt,
                    max_tokens=512,
                    sampler=self.sampler
                )
                
                # Calculate immediate structure-based reward
                immediate_reward = self._calculate_structure_reward(prediction)
                
                rollouts.append({
                    "rollout_id": i,
                    "prediction": prediction,
                    "method": "advanced_perspective",
                    "perspective": perspective,
                    "immediate_reward": immediate_reward,
                    "timestamp": datetime.now().isoformat()
                })
                
            except Exception as e:
                logger.error(f"Error generating rollout {i} with perspective {perspective}: {e}")
                continue
        
        return rollouts
    
    def _calculate_structure_reward(self, response: str) -> float:
        """Calculate structure-based reward for a response."""
        required_sections = [
            "Market Impact",
            "Specific Assets", 
            "Trade Recommendation",
            "Timeframe",
            "Risk Factors",
            "World View"
        ]
        
        section_score = 0
        for section in required_sections:
            if section in response:
                section_score += 1
        
        return section_score / len(required_sections)
    
    def generate_daily_predictions(self, headlines: List[Dict[str, Any]], num_rollouts: int = 8, horizon: str | None = None) -> List[Dict[str, Any]]:
        """Generate predictions for all headlines.

        Args:
            headlines: list of headline dicts with key "text"
            num_rollouts: number of rollouts per headline
            horizon: optional horizon directive ("next_day" | "next_month" | "next_year")
        """
        daily_predictions: List[Dict[str, Any]] = []

        for headline_data in headlines:
            headline_text = headline_data["text"]

            rollouts = self.generate_rollouts_for_headline(headline_text, num_rollouts=num_rollouts, horizon=horizon)

            prediction_entry = {
                "headline": headline_text,
                "headline_data": headline_data,
                "rollouts": rollouts,
                "total_rollouts": len(rollouts),
                "methods_used": list(set(r["method"] for r in rollouts)),
                "avg_immediate_reward": sum(r["immediate_reward"] for r in rollouts) / len(rollouts) if rollouts else 0.0,
                "generated_at": datetime.now().isoformat(),
                **({"horizon": horizon} if horizon else {})
            }

            daily_predictions.append(prediction_entry)

            logger.info(f"Generated {len(rollouts)} rollouts for headline: {headline_text[:50]}...")

        return daily_predictions

    def _build_horizon_directive(self, horizon: str) -> str:
        """Return a directive prefix to bias the model toward a given prediction horizon."""
        mapping = {
            "next_day": "You must generate a prediction calibrated to the next 1 trading day. Be explicit and concrete about the 24–48h horizon.\n\n",
            "next_month": "You must generate a prediction calibrated to roughly the next 1 month. Focus on medium-term drivers and positioning.\n\n",
            "next_year": "You must generate a prediction calibrated to roughly the next 12 months. Focus on longer-term fundamentals and regime shifts.\n\n",
        }
        if not horizon:
            return ""
        key = horizon.strip().lower()
        return mapping.get(key, "")

    def _clean_prediction_response(self, text: str) -> str:
        """Remove prompt scaffolding and instruction echo from model outputs.

        Heuristics:
        - Drop fenced code blocks and backticks
        - Drop lines that start with meta-instructions (e.g., 'Also,', 'Please', 'Ensure', 'Use ... markdown', 'Answer:')
        - If structured section headers exist, keep from the first header onwards
        - Collapse excessive whitespace
        """
        if not text:
            return text

        # Remove code fences and inline backticks
        text = re.sub(r"```[\s\S]*?```", " ", text)
        text = text.replace("```", " ").replace("`", " ")

        # Split into lines and filter instruction-like lines
        meta_patterns = [
            r"^\s*(Also|Please|Ensure|Now|Note|Remember|Make sure|Use|Start with|Answer|Your response)\b",
            r"markdown",
            r"format",
            r"in English",
            r"one paragraph",
        ]
        def is_meta(line: str) -> bool:
            low = line.strip()
            if not low:
                return False
            for pat in meta_patterns:
                if re.search(pat, low, flags=re.IGNORECASE):
                    return True
            return False

        lines = [ln for ln in text.splitlines() if not is_meta(ln)]
        cleaned = "\n".join(lines).strip()

        # If structured sections exist, keep from first section onwards
        sections = ["Market Impact", "Specific Assets", "Trade Recommendation", "Timeframe", "Risk Factors", "World View"]
        first_idx = None
        for sec in sections:
            m = re.search(re.escape(sec), cleaned, flags=re.IGNORECASE)
            if m:
                if first_idx is None or m.start() < first_idx:
                    first_idx = m.start()
        if first_idx is not None:
            cleaned = cleaned[first_idx:]

        # Collapse multiple newlines/spaces
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)

        return cleaned.strip()

def main():
    """Test the adaptive rollout generator."""
    # Test with sample headlines
    test_headlines = [
        {"text": "Fed signals potential rate cut in March", "source": "Reuters"},
        {"text": "Apple reports strong earnings beat", "source": "CNN"},
        {"text": "Oil prices surge on supply concerns", "source": "MarketWatch"}
    ]
    
    generator = AdaptiveRolloutGenerator()
    
    # Generate predictions
    predictions = generator.generate_daily_predictions(test_headlines)
    
    print(f"Generated predictions for {len(predictions)} headlines")
    
    # Show sample rollouts
    for i, pred in enumerate(predictions):
        print(f"\nHeadline {i+1}: {pred['headline'][:50]}...")
        print(f"Rollouts: {pred['total_rollouts']}")
        print(f"Methods: {pred['methods_used']}")
        print(f"Avg reward: {pred['avg_immediate_reward']:.3f}")

if __name__ == "__main__":
    main() 
