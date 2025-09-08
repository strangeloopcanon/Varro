#!/usr/bin/env python3
"""
Clean and save article text for one or more dates.

Loads timestamped_storage[*]/<date>_articles.json, cleans text, and writes
<date>_articles_clean.json under the active run namespace (or base if none set).
"""

import argparse
import logging
import sys
from pathlib import Path

# Ensure repo root on sys.path for imports (so data_collection is importable)
try:
    REPO_ROOT = Path(__file__).resolve().parents[1]
    if str(REPO_ROOT) not in sys.path:
        sys.path.append(str(REPO_ROOT))
except Exception:
    pass

from data_collection.timestamped_storage import TimestampedStorage
from data_collection.article_cleaning import (
    clean_articles_recordset,
    save_cleaned_articles,
)

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def process_date(date: str, storage: TimestampedStorage) -> None:
    raw = storage.load_data("articles", date)
    if not raw or not isinstance(raw, dict):
        logger.warning(f"No raw articles for {date}")
        return
    cleaned = clean_articles_recordset(raw.get("articles", []) or [])
    path = save_cleaned_articles(date, cleaned, storage)
    logger.info(f"Saved cleaned articles: {path}")


def main() -> int:
    p = argparse.ArgumentParser(description="Clean article text and save as articles_clean")
    p.add_argument("--date", type=str, help="YYYYMMDD single date")
    p.add_argument("--start_date", type=str, default=None)
    p.add_argument("--end_date", type=str, default=None)
    args = p.parse_args()

    storage = TimestampedStorage()
    if args.date:
        process_date(args.date, storage)
        return 0
    if not (args.start_date and args.end_date):
        logger.error("Provide --date or --start_date and --end_date")
        return 1
    from datetime import datetime, timedelta
    cur = datetime.strptime(args.start_date, "%Y%m%d")
    end = datetime.strptime(args.end_date, "%Y%m%d")
    while cur <= end:
        process_date(cur.strftime("%Y%m%d"), storage)
        cur += timedelta(days=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
