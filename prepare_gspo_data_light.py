import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Simple reward function based on response quality
def calculate_reward(response):
    """
    Calculate a reward for a response based on:
    1. Length (longer responses might be more detailed)
    2. Structure (presence of all required sections)
    3. Coherence (checking for repeated content)
    """
    # Check if all sections are present
    sections = [
        "Goal Hypothesis",
        "Prior Knowledge", 
        "Reward Model",
        "Exploration Plan",
        "Prediction"
    ]
    
    section_score = 0
    for section in sections:
        if section in response:
            section_score += 1
    
    # Length score (normalized)
    length_score = min(len(response) / 1000, 1.0)
    
    # Coherence score (penalize repetition)
    unique_words = len(set(response.split()))
    total_words = len(response.split())
    coherence_score = unique_words / total_words if total_words > 0 else 0
    
    # Combine scores
    reward = (section_score / len(sections)) * 0.4 + length_score * 0.3 + coherence_score * 0.3
    
    return reward

# Load the dataset with rewards
with open("data/meta_learning_dataset_with_rewards.json", "r") as f:
    dataset = json.load(f)

# Select top 3 high-reward items for GSPO training
high_reward = [item for item in dataset if item["reward"] >= 0.75][:3]

print(f"Selected {len(high_reward)} high-reward items for GSPO training")

# For GSPO, we'll create groups of responses for each prompt
# We'll use the high-reward items as examples and create minimal additional data

# Create a minimal GSPO dataset
gsPO_dataset = []

# Add original high-reward responses
for i, item in enumerate(high_reward):
    gsPO_dataset.append({
        "prompt": item["prompt"],
        "response": item["response"],
        "reward": item["reward"],
        "group_id": i
    })

# Add one additional response for each prompt (instead of two)
# We'll use a simple approach to create variations by truncating the original responses
for i, item in enumerate(high_reward):
    # Create a truncated version of the response
    truncated_response = "\n".join(item["response"].split("\n")[:10]) + "\n[Response truncated for diversity]"
    
    # Calculate reward for the truncated response
    reward = calculate_reward(truncated_response)
    
    # Add to dataset
    gsPO_dataset.append({
        "prompt": item["prompt"],
        "response": truncated_response,
        "reward": reward,
        "group_id": i
    })

# Save the GSPO dataset
with open("data/gsPO_dataset.json", "w") as f:
    json.dump(gsPO_dataset, f, indent=2)

print(f"Generated GSPO dataset with {len(gsPO_dataset)} items. Saved to data/gsPO_dataset.json")

# Also save a simplified version for easier inspection
simplified_dataset = []
for item in gsPO_dataset:
    simplified_dataset.append({
        "group_id": item["group_id"],
        "reward": item["reward"],
        "response_preview": item["response"][:100] + "..." if len(item["response"]) > 100 else item["response"]
    })

with open("data/gsPO_dataset_simplified.json", "w") as f:
    json.dump(simplified_dataset, f, indent=2)

print("Saved simplified version to data/gsPO_dataset_simplified.json")