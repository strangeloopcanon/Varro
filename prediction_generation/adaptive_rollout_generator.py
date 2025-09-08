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
    """Generates diverse rollouts for financial predictions.

    Supports two output formats:
    - "one_line": original schema line (loosely validated)
    - "paragraph": a short free-form forecast paragraph (quality-scored)
    """
    
    def __init__(self, model_name: str = "Qwen/Qwen3-0.6B", checkpoint_path: str = None, sampler_profile: str = "default", output_format: str = "paragraph"):
        self.model_name = model_name
        self.checkpoint_path = checkpoint_path
        self.model = None
        self.tokenizer = None
        self.sampler = None
        self.sampler_profile = (sampler_profile or "default").lower()
        self.output_format = (output_format or "one_line").lower()
        # Feature flags
        self.enable_trade_thinking = str(os.environ.get("VARRO_ENABLE_TRADE_THINKING", "0")).lower() in {"1", "true", "yes", "on"}
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
            
            # Create sampler per profile
            self.sampler = self._create_sampler(self.sampler_profile)
            
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
            self.basic_examples_note = prompt_config.get("basic_examples_note", "")
            # Optional paragraph prompt for free-form forecasts
            self.paragraph_prompt = prompt_config.get(
                "paragraph_prompt",
                (
                    "Headline: \"{headline}\"\n\n"
                    "Write a concise forecast (3–5 sentences) describing what is likely to happen next,\n"
                    "naming specific assets or sectors affected, expected direction/magnitude, a rough timeframe,\n"
                    "and the key driver(s). Avoid meta-instructions and boilerplate."
                ),
            )
            
            # Optional: article-aware paragraph prompt
            self.paragraph_with_article_prompt = prompt_config.get(
                "paragraph_with_article_prompt",
                (
                    "Headline: \"{headline}\"\n\n"
                    "Context (article, cleaned excerpt):\n{article_excerpt}\n\n"
                    "Write a single paragraph (3–5 sentences) forecasting what happens next using the headline and context."
                    " State the main prediction, name affected assets/sectors with direction/magnitude, a rough timeframe, and key driver(s)."
                    " Use only the content; do not include instructions or boilerplate."
                ),
            )
            logger.info("Loaded prompt templates from config")
            
        except Exception as e:
            logger.error(f"Error loading prompt templates: {e}")
            # Fallback to hardcoded one-line schema prompt
            self.basic_prompt = (
                "You convert ONE news headline into a falsifiable forecast. Output exactly ONE line using: "
                "Domain=<…>; Proxy=<instrument|metric|event>; Horizon=<1w|1m|1q>; Prob=<0–100%>; "
                "Claim=<falsifiable statement>; Rationale=<…>; Risk=<…>; Confidence=<0|1|2|3>.\n\n"
                "Headline: \"{headline}\""
            )
            
            self.perspective_prompts = {
                "fundamental_analyst": "Headline: \"{headline}\"\n\nThink like a fundamental analyst...",
                "technical_trader": "Headline: \"{headline}\"\n\nThink like a technical trader...",
                "macro_economist": "Headline: \"{headline}\"\n\nThink like a macro economist..."
            }
    
    def generate_rollouts_for_headline(self, headline: str, num_rollouts: int = 8, horizon: str | None = None, article_excerpt: str | None = None) -> List[Dict[str, Any]]:
        """Generate N rollouts for a single headline using same prompt with sampling diversity.

        Args:
            headline: Headline text to condition on
            num_rollouts: Number of rollouts to generate (default 8)
            horizon: Optional horizon directive: "next_day", "next_month", or "next_year"
        """
        logger.info(f"Generating rollouts for headline: {headline[:50]}...")

        rollouts = self._generate_basic_rollouts(headline, num_rollouts=num_rollouts, horizon=horizon, article_excerpt=article_excerpt)
        logger.info(f"Generated {len(rollouts)} rollouts using same prompt with sampling diversity")
        return rollouts

    def _create_sampler(self, profile: str):
        """Return an MLX sampler configured by profile."""
        profiles = {
            "loose": {"temp": 0.9, "top_p": 0.95, "top_k": 50},
            "default": {"temp": 0.7, "top_p": 0.9, "top_k": 50},
            "tight": {"temp": 0.5, "top_p": 0.85, "top_k": 50},
        }
        cfg = profiles.get(profile, profiles["default"])
        return make_sampler(temp=cfg["temp"], top_p=cfg["top_p"], top_k=cfg["top_k"])
    
    def _generate_basic_rollouts(self, headline: str, num_rollouts: int = 8, horizon: str | None = None, article_excerpt: str | None = None) -> List[Dict[str, Any]]:
        """Generate N rollouts using basic stochastic approach with optional horizon directive.

        Behavior depends on output_format:
        - one_line: generate schema line with light retries; immediate_reward = 1 if valid else 0
        - paragraph: generate short paragraph; immediate_reward = rubric-based quality score in [0,1]
        """
        rollouts: List[Dict[str, Any]] = []

        # Optional horizon directive to bias timeframe
        horizon_prefix = self._build_horizon_directive(horizon) if horizon else ""

        for i in range(max(1, int(num_rollouts))):
            # Use same prompt, let MLX sampler create diversity
            # Choose prompt by output format
            uses_think = False
            if self.output_format == "paragraph":
                if article_excerpt:
                    base_prompt = self.paragraph_with_article_prompt.format(headline=headline, article_excerpt=article_excerpt)
                else:
                    base_prompt = self.paragraph_prompt.format(headline=headline)
                # For a small subset of rollouts, encourage brief reasoning in <think>...</think>
                # Default policy: ~2 of 8 rollouts (25%) use think when num_rollouts>=8
                try:
                    n_think = max(0, min(2, int(0.25 * max(1, int(num_rollouts)))))
                except Exception:
                    n_think = 2
                uses_think = (i < n_think)
                if uses_think:
                    base_prompt = (
                        "Before writing the final forecast, think briefly inside <think>...</think> and do not include instructions.\n"
                        "After </think>, write the concise 3–5 sentence forecast paragraph.\n\n"
                    ) + base_prompt
            else:
                base_prompt = self.basic_prompt.format(headline=headline)
                if getattr(self, 'basic_examples_note', None):
                    base_prompt = f"{base_prompt}\n\n{self.basic_examples_note}"
            prompt = f"{horizon_prefix}{base_prompt}" if horizon_prefix else base_prompt

            try:
                raw_prediction = mlx_generate(
                    self.model,
                    self.tokenizer,
                    prompt,
                    max_tokens=512,
                    sampler=self.sampler
                )

                # Format-specific post-processing and immediate reward
                if self.output_format == "paragraph":
                    cleaned = self._clean_paragraph_response(raw_prediction)
                    prediction = cleaned
                    quality = self._score_paragraph_quality(prediction)
                    immediate_reward = quality
                else:
                    # Clean, normalize, validate; single corrective retry if invalid
                    prediction = self._clean_prediction_response(raw_prediction)
                    prediction = self._normalize_prediction(prediction)
                    valid = self._is_valid_prediction(prediction)
                    if not valid:
                        retry_prompt = (
                            "Your previous output did not follow the required format.\n"
                            "Output exactly ONE line using this template and nothing else:\n"
                            "Domain=<…>; Proxy=<instrument|metric|event>; Horizon=<1w|1m|1q>; Prob=<0–100%>; "
                            "Claim=<falsifiable statement>; Rationale=<≤18 words>; Risk=<≤6 words>; Confidence=<0|1|2|3>.\n\n"
                            f"Headline: \"{headline}\""
                        )
                        retry_text = mlx_generate(
                            self.model,
                            self.tokenizer,
                            retry_prompt,
                            max_tokens=256,
                            sampler=self.sampler
                        )
                        prediction = self._normalize_prediction(self._clean_prediction_response(retry_text))
                        valid = self._is_valid_prediction(prediction)
                    immediate_reward = 1.0 if valid else 0.0

                # Optional: trade-thinking score (for later analysis). Does not affect selection or reward.
                trade_thinking = None
                if self.enable_trade_thinking and score_trade_thinking is not None:
                    trade_thinking = score_trade_thinking(prediction, horizon=horizon)

                rollouts.append({
                    "rollout_id": i,
                    "prediction": prediction,
                    **({"raw_prediction": raw_prediction} if self.output_format == "paragraph" else {}),
                    **({"uses_think": True} if uses_think else {}),
                    "method": "basic_stochastic",
                    "immediate_reward": immediate_reward,
                    **({"trade_thinking_score": trade_thinking} if (self.enable_trade_thinking and trade_thinking is not None) else {}),
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

                # Clean/normalize/validate like basic path
                prediction = self._normalize_prediction(self._clean_prediction_response(prediction))
                valid = self._is_valid_prediction(prediction)
                immediate_reward = 1.0 if valid else 0.0
                
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
        """Compatibility shim; now aligned to one-line template validity."""
        return 1.0 if self._is_valid_prediction(response) else 0.0
    
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
            article_excerpt = headline_data.get("article_excerpt")

            rollouts = self.generate_rollouts_for_headline(headline_text, num_rollouts=num_rollouts, horizon=horizon, article_excerpt=article_excerpt)

            prediction_entry = {
                "headline": headline_text,
                "headline_data": headline_data,
                "rollouts": rollouts,
                "total_rollouts": len(rollouts),
                "methods_used": list(set(r["method"] for r in rollouts)),
                "avg_immediate_reward": sum(r["immediate_reward"] for r in rollouts) / len(rollouts) if rollouts else 0.0,
                "sampler_profile": getattr(self, 'sampler_profile', 'default'),
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
            "next_2days": "You must generate a prediction calibrated to the next 2 trading days. Be explicit about near-term catalysts.\n\n",
            "next_3days": "You must generate a prediction calibrated to the next 3 trading days. Be explicit about near-term catalysts and positioning.\n\n",
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
            r"the output should",
            r"first line",
            r"bold|italic|centered|paragraph|list|bullet",
            r"^Headline:\s*\".*\"$",
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

        # Collapse multiple newlines/spaces
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)

        return cleaned.strip()

    def _normalize_prediction(self, text: str) -> str:
        """Normalize model output to a single line. Extract first candidate line.

        Heuristics:
        - Prefer first line starting with "Domain=" or "No forecast:"
        - Otherwise, take the first non-empty line and collapse whitespace to single spaces
        """
        if not text:
            return ""
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        for ln in lines:
            if ln.startswith("Domain=") or ln.startswith("No forecast:"):
                return re.sub(r"\s+", " ", ln)
        # fallback to first non-empty line
        return re.sub(r"\s+", " ", lines[0]) if lines else ""

    def _is_valid_prediction(self, text: str) -> bool:
        """Looser validator for one-line schema. Accepts minor variations."""
        if not text or "\n" in text.strip():
            return False
        t = text.strip()
        pattern = re.compile(
            r'^Domain=.*?;\s*'
            r'Proxy=.*?;\s*'
            r'Horizon=(1d|1w|1m|1q);\s*'
            r'Prob=\s*(100|[0-9]?\d)\s*%?;\s*'
            r'Claim=.*?;\s*'
            r'Rationale=.{1,160}?;\s*'
            r'Risk=.{1,60}?;\s*'
            r'Confidence=[0-3]\.?'  # optional trailing period
            r'$'
        )
        return bool(pattern.match(t))

    # ---------------- Paragraph helpers ----------------
    def _clean_paragraph_response(self, text: str) -> str:
        """Clean LLM output to a concise paragraph (remove meta, code fences, headers)."""
        if not text:
            return ""
        # Remove private chain-of-thought blocks
        import re as _re
        text = _re.sub(r"<think>[\s\S]*?</think>", " ", text, flags=_re.IGNORECASE)
        text = re.sub(r"```[\s\S]*?```", " ", text)
        text = text.replace("```", " ").replace("`", " ")
        lines = [ln.strip() for ln in text.splitlines()]
        meta = re.compile(r"^(Also|Please|Ensure|Now|Note|Remember|Make sure|Use|Start with|Answer|The output should|Format:|Headline:)\b", re.I)
        lines = [ln for ln in lines if ln and not meta.search(ln)]
        out = " ".join(lines)
        out = re.sub(r"\s+", " ", out).strip()
        return out[:800]

    def _score_paragraph_quality(self, text: str) -> float:
        """Heuristic rubric scoring in [0,1] favoring concrete forecasts.

        Components:
        - directionality present (up/down/rise/fall, tighten/ease, etc.)
        - mentions of specific assets/sectors/tickers
        - timeframe language (e.g., days, weeks, months, Qx, by Friday)
        - no obvious meta/instruction leakage
        - brevity (3–6 sentences, 40–140 words preferred)
        """
        if not text:
            return 0.0
        t = text.lower()
        # Directionality
        dir_words = ["rise", "rises", "rally", "up", "gain", "higher", "fall", "falls", "drop", "lower", "selloff", "tighten", "ease", "widen", "narrow"]
        has_dir = any(w in t for w in dir_words)
        # Assets/sectors (very rough cues)
        asset_cues = ["usd", "treasury", "bond", "oil", "wti", "brent", "gold", "btc", "s&p", "nasdaq", "dow", "ftse", "euro", "yen", "tech", "energy", "financials", "xl", "qqq", "spy"]
        has_asset = any(w in t for w in asset_cues) or bool(re.search(r"\b[A-Z]{2,5}\b", text))
        # Timeframe
        time_cues = ["day", "days", "week", "weeks", "month", "months", "quarter", "q1", "q2", "q3", "q4", "by friday", "by monday", "next week", "next month"]
        has_time = any(w in t for w in time_cues)
        # Meta leak
        meta_leak = self._paragraph_meta_leak(text)
        # Length / concision
        words = len(text.split())
        sentences = max(1, text.count('.') + text.count('!') + text.count('?'))
        length_score = 1.0
        if words < 25:
            length_score = 0.5
        elif words > 180:
            length_score = 0.6
        elif not (3 <= sentences <= 6):
            length_score = 0.8
        # Combine
        base = 0.0
        base += 0.35 if has_dir else 0.0
        base += 0.30 if has_asset else 0.0
        base += 0.20 if has_time else 0.0
        base *= length_score
        # Penalize meta leakage
        penalty = 0.3 if meta_leak else 0.0
        score = max(0.0, min(1.0, base * (1.0 - penalty)))
        return score

    def _paragraph_meta_leak(self, text: str) -> bool:
        patt = re.compile(r"(the output should|as an ai|please|ensure|format:|domain=|proxy=|horizon=|prob=|rationale=|risk=|confidence=)", re.I)
        return bool(patt.search(text or ""))

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
