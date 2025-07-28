#!/usr/bin/env python3
"""
GSPO Trainer with MLX for Meta-Learning Tasks
"""

import json
import logging
import os
from typing import List, Dict, Any, Tuple
from collections import defaultdict

import mlx.core as mx
import mlx.nn as nn
from mlx.optimizers import Adam
from mlx_lm import load, generate as mlx_generate
from mlx_lm.sample_utils import make_sampler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MetaLearningReward:
    """Reward function for meta-learning tasks."""
    
    def __init__(self):
        # Define the required sections for a good meta-learning response
        self.required_sections = [
            "Goal Hypothesis",
            "Prior Knowledge", 
            "Reward Model",
            "Exploration Plan",
            "Prediction"
        ]
    
    def calculate_reward(self, task: Dict[str, Any], response: str) -> float:
        """
        Calculate a reward for a meta-learning response based on:
        1. Presence of required sections
        2. Length of response (longer might be more detailed)
        3. Coherence (checking for repeated content)
        """
        # Section score
        section_score = 0
        for section in self.required_sections:
            if section in response:
                section_score += 1
        section_reward = section_score / len(self.required_sections)
        
        # Length score (normalized)
        length_score = min(len(response) / 500, 1.0)
        
        # Coherence score (penalize repetition)
        words = response.split()
        unique_words = len(set(words))
        total_words = len(words)
        coherence_score = unique_words / total_words if total_words > 0 else 0
        
        # Combine scores with weights
        reward = section_reward * 0.5 + length_score * 0.25 + coherence_score * 0.25
        
        return reward

class GSPOTrainer:
    """GSPO trainer with MLX for meta-learning tasks."""
    
    def __init__(self, model, tokenizer, tasks: List[Dict[str, Any]], reward_function,
                 learning_rate=5e-6, rollouts_per_step=4, save_dir="training/checkpoints",
                 temperature=0.8, top_p=0.95, top_k=50):
        self.model = model
        self.tokenizer = tokenizer
        self.tasks = tasks
        self.reward_function = reward_function
        self.learning_rate = learning_rate
        self.rollouts_per_step = rollouts_per_step
        self.save_dir = save_dir
        
        # MLX sampler for non-deterministic generation
        self.sampler = make_sampler(temp=temperature, top_p=top_p, top_k=top_k)
        
        # Initialize optimizer
        self.optimizer = Adam(learning_rate=learning_rate)
        
        # Group tasks by headline for GSPO
        self.task_groups = defaultdict(list)
        for task in tasks:
            headline = task.get('headline', 'unknown')
            self.task_groups[headline].append(task)
        
        # Training statistics
        self.step_count = 0
        self.total_rewards = []
        self.kl_history = []
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        logger.info(f"Initialized GSPO trainer with MLX sampler")
        logger.info(f"  - Temperature: {temperature}")
        logger.info(f"  - Top-p: {top_p}")
        logger.info(f"  - Top-k: {top_k}")
        logger.info(f"  - Rollouts per step: {rollouts_per_step}")
        logger.info(f"  - Number of task groups: {len(self.task_groups)}")
    
    def compute_policy_gradients(self, group_prompts: List[str], group_responses: List[List[str]], 
                               group_rewards: List[List[float]]) -> Tuple[mx.array, float]:
        """Compute policy gradients for GSPO using grouped sequences."""
        try:
            def loss_fn():
                """Loss function for policy gradient computation."""
                total_loss = mx.array(0.0)
                total_groups = 0
                
                # Process each group
                for prompt, responses, rewards in zip(group_prompts, group_responses, group_rewards):
                    if not responses or not any(r.strip() for r in responses):
                        continue
                    
                    # Normalize rewards within the group
                    if len(set(rewards)) > 1:
                        normalized_rewards = [(r - min(rewards)) / (max(rewards) - min(rewards) + 1e-8) for r in rewards]
                    else:
                        normalized_rewards = [0.5] * len(rewards)
                    
                    group_loss = mx.array(0.0)
                    
                    # Process each response in the group
                    for i, response in enumerate(responses):
                        if not response.strip():
                            continue
                            
                        # Tokenize prompt and response
                        prompt_tokens = self.tokenizer.encode(prompt)
                        response_tokens = self.tokenizer.encode(response)
                        
                        # Combine tokens
                        input_tokens = prompt_tokens + response_tokens
                        input_array = mx.array(input_tokens, dtype=mx.int32)
                        
                        # Get model logits
                        logits = self.model(input_array[None, :])  # Add batch dimension
                        
                        # Compute log probabilities for the response tokens
                        log_probs = nn.log_softmax(logits, axis=-1)
                        
                        # Extract log probabilities for the response tokens
                        response_log_probs = []
                        for j in range(len(prompt_tokens), len(input_tokens) - 1):
                            token_id = input_tokens[j + 1]  # Predict next token
                            log_prob = log_probs[0, j, token_id]
                            response_log_probs.append(log_prob)
                        
                        # Policy gradient loss: -reward * sum(log_prob)
                        if response_log_probs:
                            log_prob_sum = sum(response_log_probs)
                            group_loss += -normalized_rewards[i] * log_prob_sum
                    
                    total_loss += group_loss / len(responses) if responses else mx.array(0.0)
                    total_groups += 1
                
                return total_loss / total_groups if total_groups > 0 else mx.array(0.0)
            
            # Compute gradients
            loss, gradients = nn.value_and_grad(self.model, loss_fn)()
            
            return gradients, loss.item()
            
        except Exception as e:
            logger.error(f"Error in policy gradient computation: {e}")
            return {}, 0.0
    
    def update_model_parameters(self, gradients):
        """Update model parameters using computed gradients."""
        try:
            # Apply optimizer step
            self.optimizer.update(self.model, gradients)
            mx.eval(self.model.parameters())
            logger.debug(f"Updated model parameters for step {self.step_count}")
            
        except Exception as e:
            logger.error(f"Error updating model parameters: {e}")
    
    def train_step(self) -> float:
        """Execute one training step with non-deterministic rollouts."""
        try:
            # Select a random group of tasks
            group_keys = list(self.task_groups.keys())
            selected_group_key = group_keys[self.step_count % len(group_keys)]
            selected_group = self.task_groups[selected_group_key]
            
            logger.info(f"Step {self.step_count + 1}: Selected group with {len(selected_group)} tasks")
            
            # Generate responses for each task in the group
            group_prompts = []
            group_responses = []
            group_rewards = []
            
            for task in selected_group:
                prompt = task['prompt']
                responses = []
                rewards = []
                
                # Generate diverse responses using MLX sampler
                for i in range(self.rollouts_per_step):
                    # Generate response with sampler for non-deterministic behavior
                    response = mlx_generate(
                        self.model, 
                        self.tokenizer, 
                        prompt, 
                        max_tokens=512,
                        sampler=self.sampler
                    )
                    responses.append(response)
                    
                    # Calculate reward
                    reward = self.reward_function.calculate_reward(task, response)
                    rewards.append(reward)
                    
                    logger.debug(f"Task: {task.get('headline', 'unknown')[:30]}..., Rollout {i+1}: reward = {reward:.4f}")
                
                group_prompts.append(prompt)
                group_responses.append(responses)
                group_rewards.append(rewards)
            
            # Compute policy gradients
            gradients, loss = self.compute_policy_gradients(group_prompts, group_responses, group_rewards)
            
            # Update model parameters
            self.update_model_parameters(gradients)
            
            # Calculate average reward across all responses
            all_rewards = [r for rewards_list in group_rewards for r in rewards_list]
            avg_reward = sum(all_rewards) / len(all_rewards) if all_rewards else 0.0
            
            # Update statistics
            self.step_count += 1
            self.total_rewards.append(avg_reward)
            self.kl_history.append(loss)  # Using loss as a proxy for KL divergence
            
            logger.info(f"Step {self.step_count}: Avg reward = {avg_reward:.4f}, Loss = {loss:.4f}")
            
            return avg_reward
            
        except Exception as e:
            logger.error(f"Error in training step: {e}")
            return 0.0
    
    def save_model(self, step: int, epoch: int = 1):
        """Save model weights and training state."""
        try:
            checkpoint_dir = os.path.join(self.save_dir, f"gspo_step_{step}")
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            # Save MLX model weights
            model_path = os.path.join(checkpoint_dir, "model.safetensors.npz")
            self.model.save_weights(model_path)
            logger.info(f"Saved MLX model weights to {model_path}")
            
            # Save tokenizer
            self.tokenizer.save_pretrained(checkpoint_dir)
            logger.info(f"Saved tokenizer to {checkpoint_dir}")
            
            # Save training state
            state = {
                'step': step,
                'epoch': epoch,
                'total_rewards': self.total_rewards,
                'kl_history': self.kl_history,
                'learning_rate': self.learning_rate
            }
            
            state_path = os.path.join(checkpoint_dir, "training_state.json")
            with open(state_path, 'w') as f:
                json.dump(state, f, indent=2)
            
            logger.info(f"Saved complete checkpoint to {checkpoint_dir}")
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
    
    def load_model(self, checkpoint_path: str):
        """Load model weights and training state."""
        try:
            # Load MLX model weights
            model_path = os.path.join(checkpoint_path, "model.safetensors.npz")
            self.model.load_weights(model_path)
            logger.info(f"Loaded MLX model weights from {model_path}")
            
            # Load training state
            state_path = os.path.join(checkpoint_path, "training_state.json")
            if os.path.exists(state_path):
                with open(state_path, 'r') as f:
                    state = json.load(f)
                
                self.step_count = state.get('step', 0)
                self.total_rewards = state.get('total_rewards', [])
                self.kl_history = state.get('kl_history', [])
                logger.info(f"Loaded training state from {state_path}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
    
    def evaluate(self, tasks: List[Dict[str, Any]]) -> float:
        """Evaluate model on test tasks."""
        try:
            total_reward = 0.0
            num_tasks = len(tasks)
            
            for task in tasks:
                prompt = task['prompt']
                
                # Generate response with sampler
                response = mlx_generate(
                    self.model, 
                    self.tokenizer, 
                    prompt, 
                    max_tokens=512,
                    sampler=self.sampler
                )
                
                reward = self.reward_function.calculate_reward(task, response)
                total_reward += reward
            
            avg_reward = total_reward / num_tasks
            logger.info(f"Test evaluation - Average reward: {avg_reward:.4f}")
            
            return avg_reward
            
        except Exception as e:
            logger.error(f"Error in evaluation: {e}")
            return 0.0
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get training statistics."""
        return {
            'step_count': self.step_count,
            'total_rewards': self.total_rewards,
            'kl_history': self.kl_history,
            'avg_reward': sum(self.total_rewards) / len(self.total_rewards) if self.total_rewards else 0.0,
            'avg_kl': sum(self.kl_history) / len(self.kl_history) if self.kl_history else 0.0
        }