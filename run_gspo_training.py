#!/usr/bin/env python3
"""
GSPO Training Script for Meta-Learning with Qwen and MLX
"""

import json
import logging
import os
import argparse
from datetime import datetime
from typing import List, Dict, Any

from mlx_lm import load
from gspo_core import GSPOTrainer, EvaluationScoreReward

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class GSPOTrainingConfig:
    """Configuration for GSPO training with MLX."""
    
    def __init__(self, training_data_path: str = None):
        # Training parameters
        self.num_epochs = 1  # Process all data once
        self.steps_per_epoch = None  # Will be set to number of training examples
        self.save_every = 50  # Save every 50 steps
        
        # Model parameters
        self.model_name = "Qwen/Qwen3-0.6B"  # Using the 0.6B model as specified
        self.max_tokens = 512
        
        # GSPO parameters
        self.learning_rate = 1e-7  # Lower learning rate for stability
        self.temperature = 0.8
        self.top_p = 0.95
        self.top_k = 50
        
        # Data paths
        self.training_data_path = training_data_path or "training/gspo_training_20250802.json"
        
        # Output paths
        self.checkpoint_dir = "training/checkpoints/gspo"
        self.output_dir = "training/outputs/gspo"
        
        # Checkpoint loading
        self.load_checkpoint = None

def load_training_data(data_path: str) -> List[Dict[str, Any]]:
    """Load training data from JSON file."""
    try:
        with open(data_path, 'r') as f:
            data = json.load(f)
        
        training_data = data.get('training_data', [])
        logger.info(f"Loaded {len(training_data)} training examples from {data_path}")
        
        return training_data
        
    except Exception as e:
        logger.error(f"Error loading training data: {e}")
        return []

def create_base_model(model_name: str):
    """Create base model and tokenizer."""
    try:
        logger.info(f"Loading base Qwen model: {model_name}")
        model, tokenizer = load(model_name)
        logger.info("Model loaded successfully")
        return model, tokenizer
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None, None

def analyze_training_performance(stats: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze training performance statistics."""
    try:
        rewards = stats.get('total_rewards', [])
        kl_divergences = stats.get('kl_history', [])
        
        if not rewards:
            return None
        
        # Calculate reward statistics
        avg_reward = sum(rewards) / len(rewards)
        min_reward = min(rewards)
        max_reward = max(rewards)
        
        # Calculate KL divergence statistics
        avg_kl = sum(kl_divergences) / len(kl_divergences) if kl_divergences else 0.0
        max_kl = max(kl_divergences) if kl_divergences else 0.0
        
        # Calculate improvement
        if len(rewards) >= 2:
            first_half = rewards[:len(rewards)//2]
            second_half = rewards[len(rewards)//2:]
            first_avg = sum(first_half) / len(first_half)
            second_avg = sum(second_half) / len(second_half)
            improvement = second_avg - first_avg
            improvement_pct = (improvement / first_avg * 100) if first_avg > 0 else 0
        else:
            improvement = 0.0
            improvement_pct = 0.0
        
        return {
            "total_steps": len(rewards),
            "average_reward": avg_reward,
            "min_reward": min_reward,
            "max_reward": max_reward,
            "reward_std": (sum((r - avg_reward)**2 for r in rewards) / len(rewards))**0.5,
            "average_kl": avg_kl,
            "max_kl": max_kl,
            "improvement": improvement,
            "improvement_percent": improvement_pct
        }
        
    except Exception as e:
        logger.error(f"Error analyzing training performance: {e}")
        return None

def train(config: GSPOTrainingConfig):
    """Execute GSPO training with MLX."""
    try:
        logger.info("Starting GSPO training with MLX...")
        
        # Load training data
        training_data = load_training_data(config.training_data_path)
        if not training_data:
            logger.error("No training data loaded!")
            return
        
        # Create base model
        model, tokenizer = create_base_model(config.model_name)
        if model is None or tokenizer is None:
            logger.error("Failed to load model!")
            return
        
        # Create reward function that uses evaluation scores
        reward_function = EvaluationScoreReward()
        
        # Set steps per epoch to number of training examples
        config.steps_per_epoch = len(training_data)
        
        # Create GSPO trainer
        trainer = GSPOTrainer(
            model=model,
            tokenizer=tokenizer,
            tasks=training_data,
            reward_function=reward_function,
            learning_rate=config.learning_rate,
            save_dir=config.checkpoint_dir,
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k
        )
        
        # Load from checkpoint if specified
        if config.load_checkpoint:
            logger.info(f"Loading from checkpoint: {config.load_checkpoint}")
            trainer.load_model(config.load_checkpoint)
        
        logger.info("Starting GSPO training with MLX...")
        logger.info(f"Training for {config.num_epochs} epochs, {config.steps_per_epoch} steps each")
        logger.info(f"Features: MLX sampler (temp={config.temperature}, top_p={config.top_p}, top_k={config.top_k})")
        logger.info(f"Model will be saved every {config.save_every} steps")
        
        # Training loop
        for epoch in range(config.num_epochs):
            logger.info(f"Starting epoch {epoch + 1}/{config.num_epochs}")
            
            for step in range(config.steps_per_epoch):
                # Execute training step with single example
                reward = trainer.train_step_single(step)
                
                # Log progress
                logger.info(f"Step {step + 1}/{config.steps_per_epoch}, Reward: {reward:.4f}")
                
                # Save checkpoint
                if (step + 1) % config.save_every == 0:
                    logger.info(f"Saving checkpoint at step {step + 1}...")
                    trainer.save_model(step + 1, epoch + 1)
            
            # End of epoch analysis
            stats = trainer.get_training_stats()
            analysis = analyze_training_performance(stats)
            
            if analysis:
                logger.info("=== EPOCH SUMMARY ===")
                logger.info(f"Total steps: {analysis['total_steps']}")
                logger.info(f"Average reward: {analysis['average_reward']:.4f}")
                logger.info(f"Reward range: {analysis['min_reward']:.4f} - {analysis['max_reward']:.4f}")
                logger.info(f"Reward std: {analysis['reward_std']:.4f}")
                logger.info(f"Average KL: {analysis['average_kl']:.4f}")
                logger.info(f"Improvement: {analysis['improvement']:.4f} ({analysis['improvement_percent']:.1f}%)")
        
        logger.info("Training completed successfully!")
        
        # Save final model to dated directory
        date_str = datetime.now().strftime("%Y%m%d")
        final_model_dir = os.path.join(config.checkpoint_dir, f"final_model_{date_str}")
        logger.info(f"Saving final model to {final_model_dir}...")
        trainer.save_model("final", 1, final_model_dir)
        
        # Also save to generic final_model for latest
        latest_model_dir = os.path.join(config.checkpoint_dir, "final_model")
        logger.info(f"Saving latest model to {latest_model_dir}...")
        trainer.save_model("final", 1, latest_model_dir)
        
    except Exception as e:
        logger.error(f"Error during training: {e}")

def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train GSPO model with MLX")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--steps_per_epoch", type=int, default=None, help="Steps per epoch (auto-set to number of examples)")
    parser.add_argument("--learning_rate", type=float, default=5e-7, help="Learning rate")
    parser.add_argument("--training_data", type=str, default="training/gspo_training_20250802.json", 
                       help="Path to training data file")
    parser.add_argument("--save_every", type=int, default=10, help="Save model every N steps")
    parser.add_argument("--load_checkpoint", type=str, help="Path to checkpoint to load from")
    
    args = parser.parse_args()
    
    # Create config
    config = GSPOTrainingConfig()
    config.num_epochs = args.epochs
    config.steps_per_epoch = args.steps_per_epoch
    config.learning_rate = args.learning_rate
    config.training_data_path = args.training_data
    config.save_every = args.save_every
    config.load_checkpoint = args.load_checkpoint
    
    # Run training
    train(config)

if __name__ == "__main__":
    main()