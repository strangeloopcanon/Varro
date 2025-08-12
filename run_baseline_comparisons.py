#!/usr/bin/env python3
"""
Baseline Comparison Script
Run supervised MLE and KTO baselines for comparison with GSPO results.
"""

import json
import logging
import os
from datetime import datetime
from typing import List, Dict, Any
import mlx.core as mx
import mlx.optimizers as optimizers
from mlx_lm import load, generate as mlx_generate
from mlx_lm.sample_utils import make_sampler

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

class BaselineTrainer:
    """Train baseline models for comparison with GSPO."""
    
    def __init__(self, model_name: str = "Qwen/Qwen3-0.6B"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.optimizer = None
        
        # Load model and tokenizer
        self._load_model()
        
    def _load_model(self):
        """Load the MLX model and tokenizer."""
        try:
            logger.info(f"Loading model: {self.model_name}")
            self.model, self.tokenizer = load(self.model_name)
            
            # Initialize optimizer
            self.optimizer = optimizers.Adam(learning_rate=1e-6)
            
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def load_training_data(self, data_path: str) -> List[Dict[str, Any]]:
        """Load training data from GSPO training file."""
        try:
            with open(data_path, 'r') as f:
                data = json.load(f)
            
            # Extract training examples from the nested structure
            if 'training_data' in data:
                training_examples = data['training_data']
            else:
                training_examples = data
                
            logger.info(f"Loaded {len(training_examples)} training examples")
            return training_examples
        except Exception as e:
            logger.error(f"Error loading training data: {e}")
            raise
    
    def train_supervised_mle(self, training_data: List[Dict[str, Any]], epochs: int = 1) -> Dict[str, float]:
        """Train supervised MLE baseline."""
        logger.info("Training supervised MLE baseline...")
        
        # Prepare training data (ignore rewards, just use prompts and responses)
        training_examples = []
        for item in training_data:
            prompt = item.get('prompt', '')
            response = item.get('response', '')
            if prompt and response:
                training_examples.append({
                    'text': prompt + response,
                    'target': response
                })
        
        logger.info(f"Training on {len(training_examples)} examples")
        
        # Simple MLE training (maximize likelihood of responses)
        total_loss = 0.0
        num_steps = 0
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            for example in training_examples:
                # Tokenize
                tokens = self.tokenizer.encode(example['text'])
                target_tokens = self.tokenizer.encode(example['target'])
                
                if len(tokens) < 2:
                    continue
                
                # Forward pass
                logits = self.model(mx.array(tokens[:-1]))
                loss = mx.nn.losses.cross_entropy(logits, mx.array(tokens[1:]))
                
                # Backward pass
                gradients = mx.grad(self.model)(loss)
                self.optimizer.update(self.model, gradients)
                
                epoch_loss += float(loss)
                num_steps += 1
            
            avg_epoch_loss = epoch_loss / len(training_examples)
            logger.info(f"Epoch {epoch + 1}: Average loss = {avg_epoch_loss:.4f}")
            total_loss += epoch_loss
        
        avg_loss = total_loss / num_steps if num_steps > 0 else 0.0
        
        # Evaluate on training data (simple perplexity)
        perplexity = self._evaluate_perplexity(training_examples)
        
        return {
            'avg_loss': avg_loss,
            'perplexity': perplexity,
            'num_examples': len(training_examples)
        }
    
    def train_kto(self, training_data: List[Dict[str, Any]], epochs: int = 1) -> Dict[str, float]:
        """Train KTO (Kahneman-Tversky Optimization) baseline."""
        logger.info("Training KTO baseline...")
        
        # KTO uses preference pairs - create from high/low reward examples
        high_reward_examples = [item for item in training_data if item.get('reward', 0) > 0.06]
        low_reward_examples = [item for item in training_data if item.get('reward', 0) < 0.04]
        
        # Create preference pairs
        preference_pairs = []
        for high in high_reward_examples[:len(low_reward_examples)]:
            for low in low_reward_examples:
                preference_pairs.append({
                    'chosen': high,
                    'rejected': low
                })
        
        logger.info(f"Training on {len(preference_pairs)} preference pairs")
        
        # KTO training (simplified)
        total_loss = 0.0
        num_steps = 0
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            for pair in preference_pairs:
                # Tokenize chosen and rejected responses
                chosen_tokens = self.tokenizer.encode(pair['chosen']['prompt'] + pair['chosen']['response'])
                rejected_tokens = self.tokenizer.encode(pair['rejected']['prompt'] + pair['rejected']['response'])
                
                if len(chosen_tokens) < 2 or len(rejected_tokens) < 2:
                    continue
                
                # Compute log probabilities
                chosen_logits = self.model(mx.array(chosen_tokens[:-1]))
                rejected_logits = self.model(mx.array(rejected_tokens[:-1]))
                
                chosen_logp = mx.nn.losses.cross_entropy(chosen_logits, mx.array(chosen_tokens[1:]))
                rejected_logp = mx.nn.losses.cross_entropy(rejected_logits, mx.array(rejected_tokens[1:]))
                
                # KTO loss: -log(sigmoid(beta * (logp_chosen - logp_rejected)))
                beta = 0.1
                logp_diff = chosen_logp - rejected_logp
                kto_loss = -mx.log(mx.sigmoid(beta * logp_diff))
                
                # Backward pass
                gradients = mx.grad(self.model)(kto_loss)
                self.optimizer.update(self.model, gradients)
                
                epoch_loss += float(kto_loss)
                num_steps += 1
            
            avg_epoch_loss = epoch_loss / len(preference_pairs)
            logger.info(f"Epoch {epoch + 1}: Average KTO loss = {avg_epoch_loss:.4f}")
            total_loss += epoch_loss
        
        avg_loss = total_loss / num_steps if num_steps > 0 else 0.0
        
        return {
            'avg_kto_loss': avg_loss,
            'num_pairs': len(preference_pairs),
            'num_examples': len(training_data)
        }
    
    def _evaluate_perplexity(self, examples: List[Dict[str, Any]]) -> float:
        """Evaluate perplexity on training examples."""
        total_logp = 0.0
        total_tokens = 0
        
        for example in examples:
            tokens = self.tokenizer.encode(example['text'])
            if len(tokens) < 2:
                continue
            
            logits = self.model(mx.array(tokens[:-1]))
            logp = mx.nn.losses.cross_entropy(logits, mx.array(tokens[1:]))
            
            total_logp += float(logp)
            total_tokens += len(tokens) - 1
        
        avg_logp = total_logp / total_tokens if total_tokens > 0 else 0.0
        perplexity = mx.exp(avg_logp)
        
        return float(perplexity)
    
    def save_baseline_results(self, results: Dict[str, Any], baseline_type: str):
        """Save baseline training results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"training/baseline_{baseline_type}_{timestamp}.json"
        
        os.makedirs("training", exist_ok=True)
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Saved {baseline_type} results to {filename}")
        return filename

def main():
    """Run baseline comparisons."""
    trainer = BaselineTrainer()
    
    # Load training data
    training_data_path = "training/gspo_training_data_20250806.json"
    if not os.path.exists(training_data_path):
        logger.error(f"Training data not found: {training_data_path}")
        return
    
    training_data = trainer.load_training_data(training_data_path)
    
    # Train supervised MLE baseline
    logger.info("=" * 50)
    logger.info("TRAINING SUPERVISED MLE BASELINE")
    logger.info("=" * 50)
    
    mle_results = trainer.train_supervised_mle(training_data, epochs=1)
    mle_file = trainer.save_baseline_results(mle_results, "mle")
    
    # Train KTO baseline
    logger.info("=" * 50)
    logger.info("TRAINING KTO BASELINE")
    logger.info("=" * 50)
    
    kto_results = trainer.train_kto(training_data, epochs=1)
    kto_file = trainer.save_baseline_results(kto_results, "kto")
    
    # Print summary
    logger.info("=" * 50)
    logger.info("BASELINE COMPARISON SUMMARY")
    logger.info("=" * 50)
    logger.info(f"Supervised MLE: {mle_results}")
    logger.info(f"KTO: {kto_results}")
    logger.info("=" * 50)

if __name__ == "__main__":
    main()
