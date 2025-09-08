# Varro: Financial Prediction GSPO System

## 🔗 Quick Links
- Final Report (So‑What + Story): `reports/FINAL_REPORT.md`

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
│   ├── checkpoints/                  # Model checkpoints
│   └── stats/                       # Training statistics
├── config/
│   ├── rss_sources.json             # RSS feed configuration
│   ├── training_config.json         # Training parameters
│   └── prompt_templates.json        # Prompt templates
├── timestamped_storage/             # Daily data files
├── gspo_core.py                     # GSPO algorithm implementation
├── run_gspo_training.py             # Main training script
├── run_daily_pipeline.py            # Complete daily pipeline
├── scripts/                         # Helper utilities (moved from root)
│   ├── manage_models.py             # Model version management
│   ├── run_backfill_newrun.py       # Backfill driver
│   ├── run_baseline_comparisons.py  # Baselines (MLE/KTO)
│   ├── run_next_update.py           # Yesterday→today wrapper
│   └── run_override_update.py       # Override eval + retrain + morning
└── README.md
```

## 🧰 Scripts Index

- Backfill/continue a run over dates: `python scripts/run_backfill_newrun.py --start_date 20250804 --end_date 20250810 --run_suffix NEWCOMPOSITERUN --resume_from_last_model`
- Next-day update (yesterday→today): `python scripts/run_next_update.py --run_suffix NEWCOMPOSITERUN --resume_from_last_model`
- Override evaluation + retrain + morning: `python scripts/run_override_update.py --prediction_date 20250808 --headlines_date 20250810 --resume_from_last_model`
- Manage models: `python scripts/manage_models.py --list | --info final_model_20250806 | --archive 7`
- Baselines (MLE/KTO): `python scripts/run_baseline_comparisons.py`

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

# Train from checkpoint
python run_gspo_training.py --load_checkpoint training/checkpoints/gspo/final_model

# Optional training toggles
# Use response-only loss masking (recommended, default on)
python run_gspo_training.py --response_only_loss 1

# Enable EMA reward baseline (variance reduction, default off)
python run_gspo_training.py --ema_baseline 1 --ema_momentum 0.9
```

### Run Daily Pipeline
```bash
# Run full daily cycle
python run_daily_pipeline.py --mode full

# Run morning pipeline only (collect headlines and generate predictions)
python run_daily_pipeline.py --mode morning --date 20250807

# Run evening pipeline only (evaluate previous day's predictions)
python run_daily_pipeline.py --mode evening --date 20250807

# Run night training only (prepare GSPO training data)
python run_daily_pipeline.py --mode night --date 20250807

# Use trained model for predictions
python run_daily_pipeline.py --mode morning --trained-model training/checkpoints/gspo/final_model

# Optional reproducibility hint (best-effort)
python run_daily_pipeline.py --mode morning --seed 1234
```

### Manage Model Versions
```bash
# List all model versions
python scripts/manage_models.py --list

# Get info about specific model
python scripts/manage_models.py --info final_model_20250806

# Archive models older than 7 days
python scripts/manage_models.py --archive 7
```

## 🎯 Key Features

### **GSPO Algorithm**
- **Group-based training**: Groups tasks by headline
- **Policy gradients**: Uses evaluation scores as rewards
- **Normalized rewards**: Within-group reward normalization
- **Model updates**: Updates Qwen3-0.6B parameters
- **Step-by-step processing**: One prediction at a time for stability

### **Rollout Generation**
- **8 rollouts per headline**: Diverse trading predictions
- **MLX sampler**: Stochastic generation for diversity
- **Structured responses**: Consistent format across all rollouts
- **Trained model support**: Can use previous day's trained model

### **LLM Evaluation**
- **Stack ranking**: 8 predictions per headline are ranked; dense rewards are derived from rank
- **Next-day comparison**: Evaluates against actual outcomes
- **Robust extraction**: Regex-based extraction with a deterministic fallback to enforce a single valid letter
- **Iterative selection**: Dynamic letter mapping (A, B, C...) only for available options each round

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
- ✅ LLM-based evaluation system (65% success rate)
- ✅ GSPO training algorithm
- ✅ Daily pipeline orchestration
- ✅ Model versioning and management
- ✅ Continuous learning pipeline

### **Latest Training Data (August 6th)**
- **622 examples**: 77 unique headlines × 8 rollouts each
- **Evaluation success rate**: 65% (improved from 40%)
- **Training completed**: GSPO training successful with step-by-step processing
- **Model saved**: `training/checkpoints/gspo/final_model`

### Article‑Aware Validation Run (Aug 29 → Sep 7)
- New run suffix: `SEMANTICRUN_TIGHT_Q25_ARTICLES` (stored under `timestamped_storage_SEMANTICRUN_TIGHT_Q25_ARTICLES/` and `training/checkpoints/gspo_SEMANTICRUN_TIGHT_Q25_ARTICLES/`).
- Command used:
  - `python scripts/run_semantic_with_articles.py --start_date 20250829 --end_date 20250907 --resume_from_last_model`
- What changed: prompts include a cleaned article excerpt when available; articles are cleaned and attached in morning.
- Results summary (overall averages; see cross‑run CSV/plots):
  - Quality=0.527, Zeros=0.164, Very‑low<0.2=0.184, Leak=0.218, Words≈128.
  - Baseline (SEMANTICRUN_TIGHT_Q25 without articles): Quality=0.643, Zeros=0.013, Leak=0.200.
- Takeaway: The article‑aware setting completed end‑to‑end but underperformed the prior tight baseline on our paragraph‑quality metric. Excerpt coverage was ~66% and some prompts echoed scaffolding; improving excerpt matching and paragraph cleaning is the next lever.
- Reports: `reports/CROSS_RUN_DAILY_METRICS_*.csv`, `reports/cross_run_daily_quality.png`, `reports/cross_run_daily_leak.png`, `reports/CROSS_RUN_COMPARISON_*.md`.

### **August 7th Pipeline Status**
- ✅ **35 headlines** collected from RSS sources
- ✅ **280 predictions** generated (35 × 8 rollouts)
- ⏳ **Waiting for August 8th headlines** for evaluation
- **Model used**: Trained model from August 6th

### **Training Results (August 6th)**
- **Total steps**: 622 (one per evaluated prediction)
- **Checkpoints**: Saved every 50 steps
- **Model**: Qwen3-0.6B with MLX-LM 0.25.2
- **Evaluation improvements**: Dynamic letter mapping, enhanced prompts, robust extraction

### **Next Steps**
1. **August 8th**: Collect headlines and evaluate August 7th predictions
2. **August 8th**: Train model on new evaluation data
3. **Continue daily pipeline** for continuous learning

## 🔧 Recent Improvements

### **Evaluation System**
- **Dynamic letter mapping**: Sequential A, B, C... for available predictions
- **Enhanced prompts**: Explicit single-letter format with reasoning tags
- **Robust extraction**: Handles JSON, markdown, typos, and various formats; adds a deterministic fallback query
- **Lowered threshold**: 4 rankings minimum (down from 5)
- **Fallback ranking**: Unranked predictions get rank 7

Note: The older Ollama/Llama-based evaluator has been removed; the system uses a single MLX/Qwen-based evaluator.

### **Training System**
- **Step-by-step processing**: One prediction at a time for stability
- **Checkpoint management**: Saves every 50 steps, final model to dated directory
- **Model versioning**: Daily trained models with cleanup capabilities
- **Trained model integration**: Can use previous day's model for predictions

### **Pipeline Automation**
- **Complete daily cycle**: Morning, evening, night modes
- **Trained model support**: Automatic integration of trained models
- **Error handling**: Robust error recovery and logging
- **Data organization**: Clean timestamped storage structure

## 📦 Sample Data

A small CC‑BY sample dataset is now available in `data/sample/` to help you try the pipeline without live RSS feeds.

Copy or link the `data/sample` folder into your working timestamped storage, and run the morning/evening/night pipeline stages against it as a demonstration.

## 📄 License

This project is licensed under the MIT License.

## 🗂️ Reports Index

- Final report: `reports/FINAL_REPORT.md` — canonical summary of experiments, conclusions, defaults, risks, and next steps (easy to find).
- Synthesis (latest alias): `reports/ALL_RUNS_SYNTHESIS_SO_WHAT.md` — rolling latest synthesis; dated snapshot: `reports/ALL_RUNS_SYNTHESIS_SO_WHAT_YYYYMMDD.md`.
- Cross-run comparison: `reports/CROSS_RUN_COMPARISON_20250819.md` — links to `reports/CROSS_RUN_DAILY_METRICS_20250819.csv` and plots (`reports/cross_run_daily_*.png`).
- Defaults & rationale: `reports/DEFAULTS_AND_RATIONALE_20250819.md` — recommended configs and why.
- Run-specific reports: `reports/*_YYYYMMDD_REPORT.md` — per-run details (e.g., `SEMANTICRUN_TIGHT_Q25_20250819_REPORT.md`, `NEWCOMPOSITERUN_20250818_REPORT.md`).
- Incidents/status: e.g., `reports/Evaluation_Fallback_Incident_20250802_20250811.md`, `reports/TEMP_STATUS_NEWCOMPOSITERUN.md`.
- Data inventory: `paper/DATA_INVENTORY.md` — where artifacts live and how to navigate them.

Notes
- We keep both dated reports (immutable snapshots for provenance) and a stable “latest” alias for quick reference. The final report above is the go‑to document.
