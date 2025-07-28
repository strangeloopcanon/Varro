import json
import re

# Load the dataset
with open("data/meta_learning_dataset.json", "r") as f:
    dataset = json.load(f)

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

# Add rewards to the dataset
for item in dataset:
    item["reward"] = calculate_reward(item["response"])

# Sort by reward (highest first)
dataset.sort(key=lambda x: x["reward"], reverse=True)

# Save the updated dataset
with open("data/meta_learning_dataset_with_rewards.json", "w") as f:
    json.dump(dataset, f, indent=2)

print(f"Added rewards to {len(dataset)} items. Saved to data/meta_learning_dataset_with_rewards.json")
print("\nTop 3 items by reward:")
for i, item in enumerate(dataset[:3]):
    print(f"{i+1}. Headline: {item['headline'][:50]}... Reward: {item['reward']:.4f}")