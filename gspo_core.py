#!/usr/bin/env python3
"""
GSPO Trainer for Meta-Learning with Pre-computed Rollouts and Evaluation Scores
"""

import json
import logging
import os
import re
from collections import defaultdict, Counter
from typing import List, Dict, Any, Tuple

import mlx.core as mx
from mlx.optimizers import Adam
from mlx_lm import load, generate as mlx_generate
import mlx.nn as nn

logger = logging.getLogger(__name__)

class EvaluationScoreReward:
    """Reward function that uses pre-computed evaluation scores with optional quality penalty.

    The base reward is derived from pre-computed outcome scores (0–10 → 0–1).
    Optionally, a composite penalty focused on instruction/meta leakage and
    repetition/boilerplate is applied multiplicatively:
    r_final = clamp01(r_eval * (1 - alpha * penalty^gamma)).
    """

    def __init__(self,
                 enable_quality_penalty: bool | None = None,
                 alpha: float | None = None,
                 use_meta_classifier: bool | None = None):
        # Feature toggles (env overridable)
        env_enable = os.environ.get("VARRO_ENABLE_QUALITY_PENALTY", "1").strip()
        self.enable_quality_penalty = (
            enable_quality_penalty
            if enable_quality_penalty is not None
            else env_enable not in {"0", "false", "False"}
        )

        try:
            self.alpha = float(os.environ.get("VARRO_PENALTY_ALPHA", "0.30")) if alpha is None else float(alpha)
        except Exception:
            self.alpha = 0.30

        # Nonlinear penalty dampening exponent (gamma)
        try:
            self.gamma = float(os.environ.get("VARRO_PENALTY_GAMMA", "0.7"))
        except Exception:
            self.gamma = 0.7

        env_cls = os.environ.get("VARRO_META_CLASSIFIER", "1").strip()
        self.use_meta_classifier = (
            use_meta_classifier
            if use_meta_classifier is not None
            else env_cls not in {"0", "false", "False"}
        )

        # Penalty component weights
        self.w_meta = float(os.environ.get("VARRO_PENALTY_W_META", "0.3"))
        self.w_repeat = float(os.environ.get("VARRO_PENALTY_W_REPEAT", "0.7"))

        # Pre-compiled patterns for instruction/meta leakage
        patterns = [
            # Common meta/instruction phrases
            r"\bthe answer must\b",
            r"\bthe answer should\b",
            r"\bfinal answer\b",
            r"\banswer:\b",
            r"\bas an ai\b",
            r"\bi need to\b",
            r"\blet me\b",
            r"\bwe need to\b",
            r"\bin this answer\b",
            r"\byour answer\b",
            r"\bbased on this news,\b",
            r"\bassistant\b",
            r"\bteacher\b",
            r"\brubric\b",
            r"\binstructions?\b",
            r"\bdo not\s+.*\b",
            # Code fences / raw blocks
            r"```",
            r"^\s*json\s*$",
            # Note: do not flag section headers as meta; handle via repetition penalty instead
        ]
        self.meta_re = re.compile("|".join(patterns), flags=re.IGNORECASE | re.MULTILINE)

    # ---------------------- Penalty components ----------------------
    def _score_meta_leak_regex(self, text: str) -> float:
        """Return 0/1 based on presence of obvious meta/instruction leakage markers."""
        return 1.0 if self.meta_re.search(text or "") else 0.0

    def _score_meta_leak_classifier(self, text: str) -> float:
        """A tiny heuristic classifier that estimates meta-leak probability in [0,1].

        Uses a few weak features (presence counts of cue phrases, fence tokens,
        excessive headings). This is deliberately simple and dependency-free.
        """
        if not text:
            return 0.0
        t = text.lower()
        cues = [
            "the answer must", "final answer", "as an ai", "assistant", "teacher",
            "rubric", "the answer should", "section must", "must include",
        ]
        score = 0.0
        for cue in cues:
            if cue in t:
                score += 0.2
        # Code-fence density
        fence_count = t.count("```")
        if fence_count >= 1:
            score += 0.2
        if fence_count >= 3:
            score += 0.2
        # Excessive numbered headers
        header_like = len(re.findall(r"^\s*\d+\s*[\).-]", text, flags=re.MULTILINE))
        if header_like >= 6:
            score += 0.2
        return max(0.0, min(1.0, score))

    def _score_repetition(self, text: str) -> float:
        """Combined 2-gram and 3-gram repetition ratio in [0,1]."""
        if not text:
            return 0.0
        tokens = text.split()
        def rep_ratio(n: int) -> float:
            if len(tokens) < n:
                return 0.0
            ngrams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
            counts = Counter(ngrams)
            repeated = sum(c - 1 for c in counts.values() if c > 1)
            return min(1.0, repeated / max(1, len(ngrams)))
        rep2 = rep_ratio(2)
        rep3 = rep_ratio(3)
        # Use max to strongly penalize boilerplate repetitions at any scale
        return max(rep2, rep3)

    def _quality_penalty(self, text: str) -> float:
        if not self.enable_quality_penalty:
            return 0.0
        v_meta_regex = self._score_meta_leak_regex(text)
        v_meta_cls = self._score_meta_leak_classifier(text) if self.use_meta_classifier else 0.0
        # Combine meta signals (take max to be conservative)
        v_meta = max(v_meta_regex, v_meta_cls)
        v_rep = self._score_repetition(text)

        # Multiplicative combiner to keep penalty in [0,1]
        keep = (1.0 - self.w_meta * v_meta) * (1.0 - self.w_repeat * v_rep)
        penalty = 1.0 - max(0.0, min(1.0, keep))

        logger.debug(
            f"Quality penalty components: meta_regex={v_meta_regex:.3f}, meta_cls={v_meta_cls:.3f}, "
            f"v_meta={v_meta:.3f}, v_rep={v_rep:.3f}, penalty={penalty:.3f}"
        )
        return penalty

    # --------------------------- API ---------------------------
    def calculate_reward(self, task: Dict[str, Any], response: str) -> float:
        """Return final reward in [0,1] with optional quality penalty applied."""
        outcome_score = task.get("outcome_score", 0.0)
        r_eval = max(0.0, min(1.0, float(outcome_score) / 10.0))

        if not self.enable_quality_penalty:
            return r_eval

        penalty = self._quality_penalty(response)
        # Nonlinear dampening to preserve signal, multiplicative application
        try:
            penalty_term = (penalty ** self.gamma)
        except Exception:
            penalty_term = penalty
        r_final = r_eval * (1.0 - self.alpha * penalty_term)
        r_final = max(0.0, min(1.0, r_final))

        logger.debug(
            f"Reward: r_eval={r_eval:.3f}, alpha={self.alpha:.3f}, gamma={self.gamma:.3f}, penalty={penalty:.3f}, r_final={r_final:.3f}"
        )
        return r_final

class GSPOTrainer:
    """GSPO trainer using pre-computed rollouts and evaluation scores."""
    
    def __init__(self, model, tokenizer, tasks: List[Dict[str, Any]], reward_function,
                 learning_rate=5e-6, save_dir="training/checkpoints",
                 temperature=0.8, top_p=0.95, top_k=50,
                 response_only_loss: bool = True,
                 ema_baseline: bool = False,
                 ema_momentum: float = 0.9):
        self.model = model
        self.tokenizer = tokenizer
        self.tasks = tasks
        self.reward_function = reward_function
        self.learning_rate = learning_rate
        self.save_dir = save_dir
        self.response_only_loss = response_only_loss
        self.ema_baseline_enabled = ema_baseline
        self.ema_momentum = ema_momentum
        self._ema_value = 0.0
        
        # MLX sampler for non-deterministic generation (for evaluation only)
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
        
        logger.info(f"Initialized GSPO trainer with {len(tasks)} pre-computed examples")
        logger.info(f"  - Number of headlines: {len(self.task_groups)}")
        logger.info(f"  - Average rollouts per headline: {len(tasks) // len(self.task_groups)}")

    def compute_policy_gradients(self, group_prompts: List[str], group_responses: List[List[str]], 
                               group_rewards: List[List[float]]) -> Tuple[mx.array, float]:
        """Compute policy gradients using pre-computed responses and rewards."""
        try:
            def loss_fn():
                total_loss = mx.array(0.0)
                
                for prompt, responses, rewards in zip(group_prompts, group_responses, group_rewards):
                    # Tokenize the prompt
                    logger.debug(f"Encoding prompt: {prompt[:100]}...")
                    prompt_tokens = self.tokenizer.encode(prompt)
                    logger.debug(f"Prompt tokens: {prompt_tokens[:10]}...")
                    prompt_array = mx.array(prompt_tokens, dtype=mx.int32)
                    
                    # Calculate log probabilities for each response
                    for response, reward in zip(responses, rewards):
                        # Tokenize the response
                        logger.debug(f"Encoding response: {response[:100]}...")
                        response_tokens = self.tokenizer.encode(response)
                        logger.debug(f"Response tokens: {response_tokens[:10]}...")
                        response_array = mx.array(response_tokens, dtype=mx.int32)
                        
                        # Debug: check types
                        logger.debug(f"prompt_array type: {type(prompt_array)}, shape: {prompt_array.shape}")
                        logger.debug(f"response_array type: {type(response_array)}, shape: {response_array.shape}")
                        logger.debug(f"reward type: {type(reward)}, value: {reward}")
                        
                        # Combine prompt and response
                        full_sequence = mx.concatenate([prompt_array, response_array], axis=0)
                        
                        # Get model logits
                        logits = self.model(full_sequence[None, :])[0, :-1, :]  # Remove last token for prediction
                        
                        # Calculate log probabilities for response tokens
                        labels = full_sequence[1:]  # Shift by 1 for next token prediction
                        labels_expanded = mx.expand_dims(labels, axis=-1)
                        
                        # Get log probabilities
                        log_probs = nn.log_softmax(logits, axis=-1)
                        
                        # Gather the log probabilities for the actual tokens
                        gathered_log_probs = mx.take_along_axis(
                            log_probs, labels_expanded, axis=-1
                        ).squeeze(-1)
                        
                        # If configured, mask out prompt positions so loss only covers response tokens
                        if self.response_only_loss:
                            prompt_len = prompt_array.shape[0]
                            # gathered_log_probs has length (len(full_sequence)-1)
                            start_idx = max(0, int(prompt_len) - 1)
                            masked_log_probs = gathered_log_probs[start_idx:]
                            response_loss = -mx.sum(masked_log_probs) * reward
                        else:
                            # Policy gradient loss over entire sequence (prompt+response)
                            response_loss = -mx.sum(gathered_log_probs) * reward
                        total_loss += response_loss
                
                return total_loss
            
            # Compute gradients using MLX's value_and_grad
            loss_and_grad_fn = nn.value_and_grad(self.model, loss_fn)
            loss, gradients = loss_and_grad_fn()
            
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
        """Execute one training step using all pre-computed rollouts and rewards."""
        try:
            logger.info(f"Step {self.step_count + 1}: Processing all {len(self.tasks)} examples")
            
            # Use all pre-computed responses and rewards
            group_prompts = []
            group_responses = []
            group_rewards = []
            
            for task in self.tasks:
                prompt = task['prompt']
                response = task['response']  # Pre-computed response
                raw_reward = self.reward_function.calculate_reward(task, response)  # Pre-computed reward
                # Apply EMA baseline as advantage if enabled (do not update EMA inside batch)
                reward = raw_reward - (self._ema_value if self.ema_baseline_enabled else 0.0)
                
                group_prompts.append(prompt)
                group_responses.append([response])  # Single pre-computed response
                group_rewards.append([reward])  # Single pre-computed reward
                
                logger.debug(f"Task: {task.get('headline', 'unknown')[:30]}..., Reward: {raw_reward:.4f}, Advantage: {reward:.4f}")
            
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

    def train_step_single(self, step_index: int) -> float:
        """Execute one training step using a single pre-computed example."""
        try:
            if step_index >= len(self.tasks):
                logger.error(f"Step index {step_index} out of range for {len(self.tasks)} tasks")
                return 0.0
            
            task = self.tasks[step_index]
            prompt = task['prompt']
            response = task['response']  # Pre-computed response
            reward = self.reward_function.calculate_reward(task, response)  # Pre-computed reward
            # Advantage with EMA baseline if enabled
            if self.ema_baseline_enabled:
                advantage = reward - self._ema_value
            else:
                advantage = reward
            
            logger.debug(f"Step {self.step_count + 1}: Processing example {step_index + 1}/{len(self.tasks)}")
            logger.debug(f"Headline: {task.get('headline', 'unknown')[:50]}...")
            logger.debug(f"Reward: {reward:.4f}")
            logger.debug(f"Prompt type: {type(prompt)}, length: {len(prompt)}")
            logger.debug(f"Response type: {type(response)}, length: {len(response)}")
            logger.debug(f"Reward type: {type(reward)}, value: {reward}")
            
            # Compute policy gradients for single example
            gradients, loss = self.compute_policy_gradients([prompt], [[response]], [[advantage]])
            
            # Update model parameters
            self.update_model_parameters(gradients)
            
            # Update statistics
            self.step_count += 1
            self.total_rewards.append(reward)
            self.kl_history.append(loss)  # Using loss as a proxy for KL divergence
            
            # Update EMA baseline after step
            if self.ema_baseline_enabled:
                self._ema_value = (
                    self.ema_momentum * self._ema_value + (1.0 - self.ema_momentum) * reward
                )
            
            logger.info(f"Step {self.step_count}: Reward = {reward:.4f}, Advantage = {advantage:.4f}, Loss = {loss:.4f}")
            
            return reward
            
        except Exception as e:
            logger.error(f"Error in training step {step_index}: {e}")
            return 0.0
    
    def save_model(self, step: int, epoch: int = 1, custom_dir: str = None):
        """Save model weights and training state."""
        try:
            if custom_dir:
                checkpoint_dir = custom_dir
            else:
                checkpoint_dir = os.path.join(self.save_dir, f"gspo_step_{step}")
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            # Save MLX model weights
            model_path = os.path.join(checkpoint_dir, "model.safetensors.npz")
            self.model.save_weights(model_path)
            logger.info(f"Saved MLX model weights to {model_path}")
            
            # Save tokenizer
            self.tokenizer.save_pretrained(checkpoint_dir)
            logger.info(f"Saved tokenizer to {checkpoint_dir}")
            
            # Save model config (MLX models don't have config attribute)
            config_path = os.path.join(checkpoint_dir, "config.json")
            config_data = {
                "model_type": "qwen3",
                "vocab_size": self.tokenizer.vocab_size if hasattr(self.tokenizer, 'vocab_size') else 32000,
                "hidden_size": 1024,  # Default for Qwen3-0.6B
                "num_attention_heads": 16,
                "num_hidden_layers": 24
            }
            with open(config_path, 'w') as f:
                json.dump(config_data, f, indent=2)
            logger.info(f"Saved model config to {config_path}")
            
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
                
                step_value = state.get('step', 0)
                # Handle case where step is "final" (string)
                if isinstance(step_value, str):
                    self.step_count = 0  # Reset to 0 for new training
                else:
                    self.step_count = step_value
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
            
            avg_reward = total_reward / num_tasks if num_tasks > 0 else 0.0
            logger.info(f"Evaluation: {num_tasks} tasks, avg reward = {avg_reward:.4f}")
            
            return avg_reward
            
        except Exception as e:
            logger.error(f"Error in evaluation: {e}")
            return 0.0
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get training statistics."""
        return {
            'total_steps': self.step_count,
            'total_rewards': self.total_rewards,
            'kl_history': self.kl_history,
            'num_task_groups': len(self.task_groups),
            'total_examples': len(self.tasks)
        }

def make_sampler(temp=0.8, top_p=0.95, top_k=50):
    """Create MLX sampler for non-deterministic generation."""
    def sampler(logits):
        # Apply temperature
        logits = logits / temp
        
        # Apply top-k filtering
        if top_k > 0:
            top_k_logits = mx.topk(logits, top_k)
            logits = mx.where(logits < top_k_logits, -float('inf'), logits)
        
        # Apply top-p filtering
        if top_p < 1.0:
            sorted_logits = mx.sort(logits, axis=-1)
            cumulative_probs = mx.cumsum(mx.softmax(sorted_logits, axis=-1), axis=-1)
            sorted_indices = mx.argsort(logits, axis=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove = mx.concatenate([
                mx.zeros_like(sorted_indices_to_remove[..., :1]),
                sorted_indices_to_remove[..., :-1]
            ], axis=-1)
            indices_to_remove = mx.take_along_axis(sorted_indices_to_remove, sorted_indices, axis=-1)
            logits = mx.where(indices_to_remove, -float('inf'), logits)
        
        return logits
    
    return sampler
