#!/usr/bin/env python3
"""
Model Management Script
Organizes and manages trained models by date.
"""

import os
import json
import logging
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def list_model_versions():
    """List all available model versions."""
    checkpoint_dir = "training/checkpoints/gspo"
    
    if not os.path.exists(checkpoint_dir):
        logger.info("No models found")
        return
    
    models = []
    for item in os.listdir(checkpoint_dir):
        if item.startswith("final_model"):
            model_path = os.path.join(checkpoint_dir, item)
            if os.path.isdir(model_path):
                # Check if it's a valid model
                config_file = os.path.join(model_path, "config.json")
                if os.path.exists(config_file):
                    models.append(item)
    
    logger.info(f"Found {len(models)} model versions:")
    for model in sorted(models):
        logger.info(f"  - {model}")
    
    return models

def get_model_info(model_name):
    """Get information about a specific model."""
    model_path = f"training/checkpoints/gspo/{model_name}"
    
    if not os.path.exists(model_path):
        logger.error(f"Model {model_name} not found")
        return None
    
    # Check for training state
    state_file = os.path.join(model_path, "training_state.json")
    if os.path.exists(state_file):
        try:
            with open(state_file, 'r') as f:
                state = json.load(f)
            return {
                "model_name": model_name,
                "training_date": state.get("training_date", "unknown"),
                "total_steps": state.get("total_steps", "unknown"),
                "average_reward": state.get("average_reward", "unknown")
            }
        except:
            return {"model_name": model_name, "status": "corrupted"}
    
    return {"model_name": model_name, "status": "no_training_state"}

def archive_old_models(keep_days=7):
    """Archive models older than specified days."""
    checkpoint_dir = "training/checkpoints/gspo"
    archive_dir = "training/checkpoints/archive"
    
    if not os.path.exists(checkpoint_dir):
        logger.info("No models to archive")
        return
    
    # Create archive directory
    os.makedirs(archive_dir, exist_ok=True)
    
    models = list_model_versions()
    if not models:
        return
    
    current_date = datetime.now()
    archived_count = 0
    
    for model in models:
        if model == "final_model":  # Keep the latest
            continue
            
        # Extract date from model name
        if model.startswith("final_model_"):
            try:
                date_str = model.replace("final_model_", "")
                model_date = datetime.strptime(date_str, "%Y%m%d")
                days_old = (current_date - model_date).days
                
                if days_old > keep_days:
                    old_path = os.path.join(checkpoint_dir, model)
                    new_path = os.path.join(archive_dir, model)
                    
                    # Move to archive
                    os.rename(old_path, new_path)
                    logger.info(f"Archived {model} (age: {days_old} days)")
                    archived_count += 1
                    
            except ValueError:
                logger.warning(f"Could not parse date from {model}")
    
    logger.info(f"Archived {archived_count} old models")

def main():
    """Main function for model management."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Manage trained models")
    parser.add_argument("--list", action="store_true", help="List all model versions")
    parser.add_argument("--info", type=str, help="Get info about specific model")
    parser.add_argument("--archive", type=int, default=7, help="Archive models older than N days")
    
    args = parser.parse_args()
    
    if args.list:
        models = list_model_versions()
        if models:
            print("\nModel Information:")
            for model in models:
                info = get_model_info(model)
                if info:
                    print(f"  {info}")
    
    elif args.info:
        info = get_model_info(args.info)
        if info:
            print(json.dumps(info, indent=2))
    
    elif args.archive:
        archive_old_models(args.archive)
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 
