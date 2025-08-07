# Varro: Financial Prediction GSPO System

A daily financial prediction system using Group Sequence Policy Optimization (GSPO) with MLX, collecting news headlines, generating 8 rollouts per headline, tracking outcomes, and continuously training the model.

## 🎯 System Overview

This system teaches a language model to "think like a trader" through meta-learning using GSPO. The model structures its reasoning with specific components:

- **Market Impact**: How will this affect markets broadly?
- **Specific Assets**: Which specific assets will be most affected?
- **Trade Recommendation**: What specific trade would you make?
- **Timeframe**: When do you expect this to play out?
- **Risk Factors**: What could go wrong with this prediction?
- **World View**: What world does this news imagine or create?

## 🔄 Daily Pipeline Sequence

### **Day T (Morning): Data Collection & Rollout Generation**
1. **Collect Headlines**: RSS feeds from financial news sources
2. **Generate 8 Rollouts**: For each headline, create 8 different trading predictions
3. **Store Predictions**: Save all rollouts for next-day evaluation

### **Day T+1 (Evening): Outcome Tracking & LLM Evaluation**
1. **Collect Next-Day Headlines**: What actually happened
2. **Match Predictions**: Compare previous day's predictions with outcomes
3. **LLM Evaluation**: Use LLM to score each prediction (0-10 scale)
4. **Store Evaluation Results**: Save scores for training

### **Day T+1 (Night): GSPO Training**
1. **Prepare Training Data**: Combine rollouts with evaluation scores
2. **GSPO Training**: Use evaluation scores as rewards to train model
3. **Update Model**: Save improved checkpoint for next day

## 🏗️ Architecture

### **Core Components**
- **Enhanced RSS Collector**: Collects headlines from multiple financial sources
- **Adaptive Rollout Generator**: Generates 8 diverse predictions per headline
- **LLM Outcome Evaluator**: Evaluates predictions against next-day headlines
- **GSPO Trainer**: Trains model using evaluation scores as rewards

### **Data Storage**
- **Timestamped Storage**: Daily data organization by date
- **Prediction Storage**: Stores rollouts for evaluation
- **Evaluation Storage**: Stores LLM evaluation scores
- **Training Data**: Combines rollouts with evaluation scores

## 📁 Project Structure

```
Varro/
├── data_collection/
│   ├── enhanced_rss_collector.py      # RSS headline collection
│   ├── historical_data_collector.py   # Historical dataset creation (future use)
│   └── timestamped_storage.py        # Daily data organization
├── prediction_generation/
│   ├── adaptive_rollout_generator.py  # 8-rollout generation
│   └── prediction_storage.py         # Prediction storage
├── outcome_tracking/
│   ├── outcome_tracker.py            # Match predictions with outcomes
│   ├── llm_outcome_evaluator.py     # LLM-based evaluation
│   └── evaluation_storage.py        # Evaluation score storage
├── training/
│   ├── gspo_training_20250802.json  # 272 examples (34 headlines × 8 rollouts)
│   ├── stats/                       # Training statistics
│   └── checkpoints/                 # Model checkpoints
├── config/
│   ├── rss_sources.json             # RSS feed configuration
│   ├── training_config.json         # Training parameters
│   └── prompt_templates.json        # Prompt templates
├── timestamped_storage/             # Daily data files
├── gspo_core.py                     # GSPO algorithm implementation
├── run_gspo_training.py             # Main training script
├── run_daily_pipeline.py            # Complete daily pipeline
├── manage_models.py                 # Model version management
└── README.md
```

## 🚀 Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Run GSPO Training
```bash
# Train with existing rollout data and evaluation scores (auto-detects data size)
python run_gspo_training.py --epochs 1

# Train with custom parameters
python run_gspo_training.py --learning_rate 5e-7 --epochs 1
```

### Run Daily Pipeline
```bash
# Run full daily cycle
python run_daily_pipeline.py --mode full

# Run morning pipeline only (collect headlines and generate predictions)
python run_daily_pipeline.py --mode morning

# Run evening pipeline only (evaluate previous day's predictions)
python run_daily_pipeline.py --mode evening

# Run night training only (train GSPO model)
python run_daily_pipeline.py --mode night
```

### Manage Model Versions
```bash
# List all model versions
python manage_models.py --list

# Get info about specific model
python manage_models.py --info final_model_20250806

# Archive models older than 7 days
python manage_models.py --archive 7
```

## 🎯 Key Features

### **GSPO Algorithm**
- **Group-based training**: Groups tasks by headline
- **Policy gradients**: Uses evaluation scores as rewards
- **Normalized rewards**: Within-group reward normalization
- **Model updates**: Updates Qwen3-0.6B parameters

### **Rollout Generation**
- **8 rollouts per headline**: Diverse trading predictions
- **MLX sampler**: Stochastic generation for diversity
- **Structured responses**: Consistent format across all rollouts

### **LLM Evaluation**
- **0-10 scoring**: Detailed evaluation of predictions
- **Next-day comparison**: Evaluates against actual outcomes
- **Normalized rewards**: Converts scores to 0-1 for training

## 📊 Data Flow

```
RSS Feeds → Headlines → 8 Rollouts → Predictions
                                    ↓
Next Day Headlines → Outcome Tracking → LLM Evaluation
                                    ↓
Training Data → GSPO Training → Updated Model
```

## 📈 Current Status

### **Completed Components**
- ✅ RSS headline collection
- ✅ 8-rollout generation per headline
- ✅ LLM-based evaluation system
- ✅ GSPO training algorithm
- ✅ Daily pipeline orchestration

### **Current Training Data**
- **133 examples**: 25 unique headlines × 8 rollouts each (August 3rd)
- **Evaluation scores**: 0-10 scale from LLM evaluation
- **Training completed**: GSPO training successful with step-by-step processing

### **Training Results (August 3rd)**
- **Total steps**: 133 (one per evaluated prediction)
- **Average reward**: 0.0721 (7.2% average score)
- **Reward range**: 0.0429 - 0.1000 (4.3% to 10%)
- **Checkpoints**: Saved every 10 steps (13 total checkpoints)
- **Model**: Qwen3-0.6B with MLX-LM 0.25.2
- **Evaluation success rate**: 48.4% (120/248 predictions evaluated)

### **Next Steps**
1. Process August 4th data (headlines available)
2. Generate predictions and evaluations for August 4th
3. Train model on August 4th data
4. Continue daily pipeline for continuous learning

## 📄 License

This project is licensed under the MIT License.