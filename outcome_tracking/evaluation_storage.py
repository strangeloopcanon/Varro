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
                        prediction_id = f"{pred_data['date']}_{(rollout.get('horizon') or 'next_day')}_{headline[:20].replace(' ', '_')}_{rollout['rollout_id']}"
                        
                        training_item = {
                            "headline": headline,
                            "prompt": self._create_prediction_prompt(headline),
                            "prediction": rollout["prediction"],
                            "immediate_reward": rollout["immediate_reward"],
                            "method": rollout["method"],
                            "date": pred_data["date"],
                            "rollout_id": rollout["rollout_id"],
                            # Carry horizon/maturity metadata when present
                            **({"horizon": rollout.get("horizon")} if "horizon" in rollout else {}),
                            **({"offset_days": rollout.get("offset_days")} if "offset_days" in rollout else {}),
                            **({"matures_on": rollout.get("matures_on")} if "matures_on" in rollout else {}),
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
        # Preference for composite vs raw evaluator reward is configurable
        import os as _os
        prefer_composite = str(_os.environ.get("VARRO_PREFER_COMPOSITE", "0")).lower() in {"1", "true", "yes", "on"}
        # Default global weights (still used if bucket envs are not set). We will compute per-example effective weights.
        try:
            import os as _os
            default_w_rubric = float(_os.environ.get("VARRO_RUBRIC_WEIGHT", "0.5"))
        except Exception:
            default_w_rubric = 0.5
        default_w_rubric = max(0.0, min(1.0, default_w_rubric))
        try:
            import os as _os
            default_outcome_alpha = float(_os.environ.get("VARRO_OUTCOME_ALPHA", "0.7"))
        except Exception:
            default_outcome_alpha = 0.7
        default_outcome_alpha = max(0.0, min(1.0, default_outcome_alpha))

        # Bucketed absolute weights (LLM, semantic, quality) derived from llm_picks per headline group
        import os as _os
        def _get_bucket_weights(picks: int):
            # Defaults: low (<=2), mid (3-5), high (>=6)
            try:
                low_llm = float(_os.environ.get("VARRO_BUCKET_LOW_LLM", "0.0"))
                low_sem = float(_os.environ.get("VARRO_BUCKET_LOW_SEM", "0.95"))
                low_q   = float(_os.environ.get("VARRO_BUCKET_LOW_Q",   "0.05"))
                mid_llm = float(_os.environ.get("VARRO_BUCKET_MID_LLM", "0.30"))
                mid_sem = float(_os.environ.get("VARRO_BUCKET_MID_SEM", "0.65"))
                mid_q   = float(_os.environ.get("VARRO_BUCKET_MID_Q",   "0.05"))
                high_llm= float(_os.environ.get("VARRO_BUCKET_HIGH_LLM","0.475"))
                high_sem= float(_os.environ.get("VARRO_BUCKET_HIGH_SEM","0.475"))
                high_q  = float(_os.environ.get("VARRO_BUCKET_HIGH_Q",  "0.05"))
            except Exception:
                low_llm, low_sem, low_q = 0.0, 0.95, 0.05
                mid_llm, mid_sem, mid_q = 0.30, 0.65, 0.05
                high_llm, high_sem, high_q = 0.475, 0.475, 0.05
            # Clamp and normalize if needed
            def _norm(llm, sem, q):
                llm = max(0.0, min(1.0, llm))
                sem = max(0.0, min(1.0, sem))
                q   = max(0.0, min(1.0, q))
                s = llm + sem + q
                if s <= 0:
                    return 0.0, 1.0, 0.0
                # If sum != 1, scale llm and sem to make room for q
                if abs(s - 1.0) > 1e-6:
                    rem = max(1e-8, (llm + sem))
                    target = max(0.0, 1.0 - q)
                    if rem > 0:
                        llm = (llm / rem) * target
                        sem = (sem / rem) * target
                    else:
                        llm, sem = 0.0, target
                return llm, sem, q
            if picks <= 2:
                return _norm(low_llm, low_sem, low_q), 'low'
            if 3 <= picks <= 5:
                return _norm(mid_llm, mid_sem, mid_q), 'mid'
            return _norm(high_llm, high_sem, high_q), 'high'
        for item in training_data:
            # Only include items with evaluations
            if item.get("evaluated", False):
                gspo_item = {
                    "prompt": item["prompt"],
                    "response": item["prediction"],
                    # Reward is blended later; initialize placeholders
                    "reward": None,
                    "headline": item["headline"],
                    "method": item["method"],
                    "date": item["date"],
                    "immediate_reward": item["immediate_reward"],
                    "outcome_score": item["outcome_score"],
                    "explanation": item["explanation"]
                }
                # If evaluation row had composite/trade-thinking, optionally carry and prefer composite
                pred_id = f"{item['date']}_{(item.get('horizon') or 'next_day')}_{item['headline'][:20].replace(' ', '_')}_{item['rollout_id']}"
                ev = eval_lookup.get(pred_id)
                if isinstance(ev, dict):
                    comp = ev.get("composite_reward")
                    tscore = ev.get("trade_thinking_score")
                    if isinstance(comp, (int, float)):
                        gspo_item["composite_reward"] = comp
                    if isinstance(tscore, (int, float)):
                        gspo_item["trade_thinking_score"] = tscore
                # Compute blended reward:  w_rubric * quality + w_outcome * outcome
                quality = 0.0
                try:
                    quality = float(item.get("immediate_reward", 0) or 0)
                    quality = max(0.0, min(1.0, quality))
                except Exception:
                    quality = 0.0
                # Outcome component: evaluator and semantic consistency (cosine) blended
                evaluator_outcome = 0.0
                semantic_consistency = 0.0
                if isinstance(ev, dict):
                    if prefer_composite and isinstance(ev.get("composite_reward"), (int, float)):
                        evaluator_outcome = float(ev.get("composite_reward"))
                    elif isinstance(ev.get("reward"), (int, float)):
                        evaluator_outcome = float(ev.get("reward"))
                    # Compute semantic consistency of response vs next-day headlines
                    try:
                        headlines_list = ev.get("headlines") or []
                        semantic_consistency = self._semantic_consistency(gspo_item["response"], headlines_list)
                    except Exception:
                        semantic_consistency = 0.0
                # Determine bucket weights based on llm picks metadata in evaluation
                picks = 0
                bucket = 'high'
                if isinstance(ev, dict):
                    try:
                        picks = int(ev.get('llm_picks', 0) or 0)
                    except Exception:
                        picks = 0
                (w_llm, w_sem, w_q), bucket = _get_bucket_weights(picks)
                # Fall back to global defaults if bucket envs are unset (rare)
                eff_wq = w_q if (w_llm + w_sem + w_q) > 0 else default_w_rubric
                denom = max(1e-8, (1.0 - eff_wq))
                eff_alpha = (w_llm / denom) if (w_llm + w_sem + w_q) > 0 else default_outcome_alpha
                eff_alpha = max(0.0, min(1.0, eff_alpha))
                # Outcome component with bucketed alpha
                outcome_component = eff_alpha * evaluator_outcome + (1.0 - eff_alpha) * semantic_consistency
                # Final reward blend with bucketed quality weight
                final_reward = eff_wq * quality + (1.0 - eff_wq) * outcome_component
                gspo_item["reward"] = max(0.0, min(1.0, final_reward))
                gspo_item["quality_score"] = quality
                gspo_item["outcome_component"] = outcome_component
                gspo_item["evaluator_outcome"] = evaluator_outcome
                gspo_item["semantic_consistency"] = semantic_consistency
                gspo_item["outcome_alpha"] = eff_alpha
                gspo_item["rubric_weight"] = eff_wq
                gspo_item["llm_picks"] = picks
                gspo_item["bucket"] = bucket
                gspo_data.append(gspo_item)

        logger.info(f"Created {len(gspo_data)} GSPO training examples")
        return gspo_data

    # ---------------------- Semantic Consistency ----------------------
    def _semantic_consistency(self, response_text: str, headlines: List[Dict[str, Any]]) -> float:
        """Return a cosine-like similarity between the forecast paragraph and next-day headlines.

        Lightweight TF-IDF (batch-local) with simple tokenization, no external deps.
        Uses max cosine over the provided headlines; returns value in [0,1].
        """
        import numpy as _np

        def _tokens(s: str) -> list[str]:
            if not s:
                return []
            import re as _re
            s = s.lower()
            # remove code fences and punctuation to a degree
            s = _re.sub(r"```[\s\S]*?```", " ", s)
            s = _re.sub(r"[^a-z0-9\s]+", " ", s)
            toks = [t for t in s.split() if len(t) >= 3]
            return toks

        # Build corpus: response + all headlines
        docs = []
        docs.append(_tokens(response_text))
        for h in headlines or []:
            docs.append(_tokens(h.get("text", "")))

        if len(docs) <= 1:
            return 0.0

        # Vocabulary
        vocab = {}
        for doc in docs:
            for t in doc:
                if t not in vocab:
                    vocab[t] = len(vocab)

        V = len(vocab)
        N = len(docs)
        if V == 0:
            return 0.0

        # Term frequency vectors
        tf = _np.zeros((N, V), dtype=_np.float32)
        df = _np.zeros(V, dtype=_np.float32)
        for i, doc in enumerate(docs):
            if not doc:
                continue
            counts = {}
            for t in doc:
                j = vocab[t]
                counts[j] = counts.get(j, 0) + 1
            for j, c in counts.items():
                tf[i, j] = c
            for j in counts.keys():
                df[j] += 1

        # TF normalization (log-scaled)
        tf = 1.0 + _np.log1p(tf)
        # IDF
        idf = _np.log((N + 1) / (1.0 + df)) + 1.0
        # TF-IDF
        x = tf * idf  # shape (N, V)

        # Normalize vectors
        def _norm_rows(a):
            n = _np.linalg.norm(a, axis=1, keepdims=True) + 1e-8
            return a / n
        x = _norm_rows(x)

        # Cosine between response (row 0) and each headline (rows 1:)
        sims = (x[1:] @ x[0])  # shape (N-1,)
        if sims.size == 0:
            return 0.0
        # Take max sim; clamp to [0,1]
        val = float(_np.clip(_np.max(sims), 0.0, 1.0))
        return val

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
        
        # Calculate outcome scores, falling back to raw reward when fields are absent
        outcome_scores = [
            (ev.get("outcome_score") if ev.get("outcome_score") is not None else ev.get("reward", 0))
            for ev in evaluations
        ]
        normalized_scores = [
            (ev.get("normalized_score") if ev.get("normalized_score") is not None else ev.get("reward", 0))
            for ev in evaluations
        ]
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
