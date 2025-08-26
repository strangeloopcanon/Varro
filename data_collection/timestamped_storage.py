#!/usr/bin/env python3
"""
Timestamped Storage System
Organizes daily data with easy retrieval for training.
"""

import json
import logging
import os
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class TimestampedStorage:
    """Manages timestamped storage for daily data.

    Supports an alternate run storage directory controlled by the env var
    VARRO_RUN_DIR_SUFFIX (e.g., NEWRUN). When set, data is saved under
    `timestamped_storage_<SUFFIX>` while reads will fall back to the base
    `timestamped_storage` if a file isn't found in the run directory.
    """
    
    def __init__(self, storage_dir: str = "timestamped_storage"):
        # Base directory always exists for reading historical headlines
        self.base_storage_dir = Path("timestamped_storage")
        self.base_storage_dir.mkdir(exist_ok=True)

        # Optional alternate run directory (e.g., timestamped_storage_NEWRUN)
        run_suffix = os.environ.get("VARRO_RUN_DIR_SUFFIX", "").strip()
        if run_suffix:
            # Normalize suffix like "NEWRUN" → timestamped_storage_NEWRUN
            alt_dir_name = f"{storage_dir}_{run_suffix}"
            self.storage_dir = Path(alt_dir_name)
        else:
            self.storage_dir = Path(storage_dir)

        self.storage_dir.mkdir(exist_ok=True)
        
        # Data types we store
        # Note: adding "articles" does not affect existing consumers; it enables
        # listing/stats and consistent save/load behavior for scraped article data.
        self.data_types = ["headlines", "articles", "predictions", "evaluations", "accuracy"]
    
    def save_data(self, data: Dict[str, Any], data_type: str, date: str = None):
        """Save data with timestamp."""
        if date is None:
            date = datetime.now().strftime("%Y%m%d")
        
        filename = self.storage_dir / f"{date}_{data_type}.json"
        
        # Add metadata
        data_with_metadata = {
            "date": date,
            "data_type": data_type,
            "saved_at": datetime.now().isoformat(),
            **data
        }
        
        with open(filename, 'w') as f:
            json.dump(data_with_metadata, f, indent=2)
        
        logger.info(f"Saved {data_type} data to {filename}")
        return str(filename)
    
    def load_data(self, data_type: str, date: str) -> Optional[Dict[str, Any]]:
        """Load data for specific date and type.

        Tries the active run directory first, then falls back to the base
        storage directory. This lets us run alternate pipelines without
        duplicating headlines while still isolating predictions/evaluations.
        """
        primary = self.storage_dir / f"{date}_{data_type}.json"
        fallback = self.base_storage_dir / f"{date}_{data_type}.json"

        for candidate in (primary, fallback):
            if candidate.exists():
                try:
                    with open(candidate, 'r') as f:
                        data = json.load(f)
                    logger.info(f"Loaded {data_type} data from {candidate}")
                    return data
                except Exception as e:
                    logger.error(f"Error loading data from {candidate}: {e}")
                    return None

        logger.warning(f"Data file not found in run or base storage: {primary} | {fallback}")
        return None
    
    def list_available_dates(self, data_type: str = None) -> List[str]:
        """List all available dates for a data type."""
        dates = set()
        
        # Combine dates from both active run dir and base dir
        seen_files = set()
        for file_path in list(self.storage_dir.glob("*.json")) + list(self.base_storage_dir.glob("*.json")):
            # Avoid double counting if paths overlap
            if str(file_path) in seen_files:
                continue
            seen_files.add(str(file_path))
            filename = file_path.stem
            parts = filename.split("_")
            
            if len(parts) >= 2:
                date = parts[0]
                file_data_type = parts[1]
                
                if data_type is None or file_data_type == data_type:
                    dates.add(date)
        
        return sorted(list(dates))
    
    def get_date_range(self, start_date: str, end_date: str, data_type: str = None) -> List[Dict[str, Any]]:
        """Get data for a date range."""
        all_data = []
        
        start = datetime.strptime(start_date, "%Y%m%d")
        end = datetime.strptime(end_date, "%Y%m%d")
        
        current = start
        while current <= end:
            date_str = current.strftime("%Y%m%d")
            
            if data_type:
                data = self.load_data(data_type, date_str)
                if data:
                    all_data.append(data)
            else:
                # Load all data types for this date
                for dt in self.data_types:
                    data = self.load_data(dt, date_str)
                    if data:
                        all_data.append(data)
            
            current += timedelta(days=1)
        
        return all_data
    
    def check_data_consistency(self, date: str) -> Dict[str, bool]:
        """Check if all data types exist for a date."""
        consistency = {}
        
        for data_type in self.data_types:
            filename = self.storage_dir / f"{date}_{data_type}.json"
            consistency[data_type] = filename.exists()
        
        return consistency
    
    def get_latest_data(self, data_type: str) -> Optional[Dict[str, Any]]:
        """Get the most recent data for a type."""
        dates = self.list_available_dates(data_type)
        
        if not dates:
            return None
        
        latest_date = dates[-1]
        return self.load_data(data_type, latest_date)
    
    def cleanup_old_data(self, days_to_keep: int = 30):
        """Remove data older than specified days."""
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        for file_path in self.storage_dir.glob("*.json"):
            filename = file_path.stem
            parts = filename.split("_")
            
            if len(parts) >= 2:
                try:
                    file_date = datetime.strptime(parts[0], "%Y%m%d")
                    
                    if file_date < cutoff_date:
                        file_path.unlink()
                        logger.info(f"Removed old data file: {file_path}")
                        
                except ValueError:
                    logger.warning(f"Invalid date format in filename: {filename}")
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get statistics about stored data."""
        stats = {
            "total_files": 0,
            "data_types": {},
            "date_range": {"earliest": None, "latest": None}
        }
        
        dates = set()
        
        for file_path in self.storage_dir.glob("*.json"):
            stats["total_files"] += 1
            filename = file_path.stem
            parts = filename.split("_")
            
            if len(parts) >= 2:
                date = parts[0]
                data_type = parts[1]
                
                dates.add(date)
                
                if data_type not in stats["data_types"]:
                    stats["data_types"][data_type] = 0
                stats["data_types"][data_type] += 1
        
        if dates:
            sorted_dates = sorted(list(dates))
            stats["date_range"]["earliest"] = sorted_dates[0]
            stats["date_range"]["latest"] = sorted_dates[-1]
        
        return stats

def main():
    """Test the timestamped storage system."""
    storage = TimestampedStorage()
    
    # Test saving data
    test_data = {
        "headlines": [
            {"text": "Fed signals rate cut", "source": "Reuters"},
            {"text": "Stocks rally on earnings", "source": "CNN"}
        ]
    }
    
    storage.save_data(test_data, "headlines", "20240115")
    
    # Test loading data
    loaded_data = storage.load_data("headlines", "20240115")
    print(f"Loaded data: {loaded_data}")
    
    # Test listing dates
    dates = storage.list_available_dates("headlines")
    print(f"Available dates: {dates}")
    
    # Test stats
    stats = storage.get_storage_stats()
    print(f"Storage stats: {stats}")

if __name__ == "__main__":
    main() 
