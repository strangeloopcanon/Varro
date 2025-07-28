# Meta-Learning with Qwen using GSPO and MLX

## Project Overview

This project implements a meta-learning framework for the Qwen 1.7B language model using Group Sequence Policy Optimization (GSPO) and MLX for efficient training on Apple Silicon.

## Key Components Implemented

### 1. Data Collection and Preparation

- **RSS Feed Integration**: Created scripts to fetch real-world news headlines from major RSS feeds (CNN, BBC, Reuters)
- **Meta-Learning Prompts**: Designed structured prompts that encourage the model to think like an "inner scientist" with components:
  - Goal Hypothesis (g)
  - Prior Knowledge (K)
  - Reward Model (θ)
  - Exploration Plan (π_probe)
  - Prediction
- **Dataset Generation**: Generated responses from the base Qwen model to these prompts
- **Reward System**: Implemented a reward function based on:
  - Presence of required sections
  - Response length
  - Coherence (uniqueness of words)

### 2. GSPO Implementation with MLX

- **GSPO Trainer**: Implemented a custom trainer that:
  - Groups tasks by headline for proper GSPO training
  - Uses MLX's sampler for non-deterministic generation
  - Computes policy gradients based on grouped sequences and their rewards
  - Updates model parameters using Adam optimizer
- **MLX Integration**: Leveraged MLX for efficient training on Apple Silicon:
  - Faster computation on M4 Mac
  - Lower memory usage
  - Hardware-optimized operations

### 3. Training Process

- **Training Loop**: Executed training for 10 steps across 2 epochs
- **Checkpointing**: Saved model weights and tokenizer at regular intervals
- **Performance Tracking**: Monitored rewards and loss values during training
- **Results**: Observed varying rewards (0.32 to 0.81) and loss values, indicating the model is learning

### 4. Model Testing

- **Inference Script**: Created a script to test the trained model with new prompts
- **Evaluation**: Verified the model can generate structured meta-learning responses
- **Reward Calculation**: Implemented reward scoring for test examples

## Core Files

### Training Pipeline
1. `rss_reader.py` - Fetches news headlines from RSS feeds
2. `generate_prompt.py` - Creates meta-learning prompts from headlines
3. `generate_response.py` - Generates model responses to prompts
4. `calculate_rewards.py` - Calculates reward scores for responses
5. `prepare_gspo_data_light.py` - Prepares data for GSPO training
6. `gspo_trainer.py` - Main GSPO trainer implementation with MLX
7. `train_gspo.py` - Training script that orchestrates the process
8. `test_trained_model.py` - Script to test the trained model

### Data
- `data/` - Contains all datasets and training data
- `training/` - Contains model checkpoints and outputs

### Vestigial Files
- `vestigial/` - Contains older implementations and examples (see `vestigial/README.md` for details)

## Training Results

- Successfully trained for 10 steps with varying rewards (0.32 to 0.81)
- Model checkpoints saved with weights and tokenizer
- Observed changes in loss values, indicating parameter updates
- Test model generates structured meta-learning responses with reasonable reward scores

## Next Steps

### 1. Evaluation of Meta-Learning Capabilities

- Design specific tests to evaluate the model's meta-learning abilities
- Create a test set with unseen headlines for performance evaluation
- Compare trained model performance against the base model
- Implement quantitative metrics for meta-learning effectiveness

### 2. Improving the Training Process

- Increase training steps for better convergence
- Experiment with different hyperparameters (learning rate, temperature, etc.)
- Implement more sophisticated reward functions
- Add techniques like curriculum learning

### 3. Scaling Up

- Generate a larger and more diverse dataset
- Use more complex environments or tasks for training
- Implement distributed training if needed
- Explore fine-tuning techniques like LoRA for parameter efficiency

## Challenges and Learnings

1. **MLX Integration**: Properly leveraging MLX required understanding its sampler implementation and gradient computation
2. **GSPO Implementation**: Grouping sequences correctly and computing gradients based on relative rewards was non-trivial
3. **Reward Design**: Creating meaningful reward functions for meta-learning tasks required careful consideration
4. **Training Efficiency**: MLX significantly improved training speed on Apple Silicon

## Future Work

- Implement a more comprehensive evaluation framework
- Explore other meta-learning algorithms (MAML, RL^2)
- Investigate few-shot learning capabilities
- Apply the trained model to real-world reasoning tasks