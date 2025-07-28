#!/usr/bin/env python3
"""
Script to test the trained GSPO model
"""

import json
from mlx_lm import load, generate as mlx_generate
from mlx_lm.sample_utils import make_sampler
from gspo_trainer import MetaLearningReward

def test_model():
    # Load the base model (we'll load the weights separately)
    model, tokenizer = load("Qwen/Qwen3-1.7B")
    
    # Load the trained weights
    model.load_weights("./training/checkpoints/gspo/gspo_step_10/model.safetensors.npz")
    
    # Create sampler
    sampler = make_sampler(temp=0.8, top_p=0.95, top_k=50)
    
    # Test prompt
    test_prompt = """Headline: "AI Model Achieves Breakthrough in Protein Folding Prediction"

Your task is to predict the next major development related to this story.

Please structure your response as follows:
1. Goal Hypothesis (g): What is the primary objective or outcome I should predict?
2. Prior Knowledge (K): What relevant information or patterns can I draw from?
3. Reward Model (θ): What would constitute a "good" prediction? Define key criteria.
4. Exploration Plan (π_probe): What information would be most valuable to gather to improve my prediction?
5. Prediction: Based on the above, what is the most likely next development?"""

    print("Testing trained GSPO model...")
    print(f"Prompt: {test_prompt[:100]}...")
    print("\n" + "="*50 + "\n")
    
    # Generate response
    response = mlx_generate(
        model,
        tokenizer,
        test_prompt,
        max_tokens=512,
        sampler=sampler
    )
    
    print("Model Response:")
    print(response)
    
    # Calculate reward
    reward_function = MetaLearningReward()
    reward = reward_function.calculate_reward({"prompt": test_prompt}, response)
    print(f"\nReward: {reward:.4f}")

if __name__ == "__main__":
    test_model()