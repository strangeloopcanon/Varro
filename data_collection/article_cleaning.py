#!/usr/bin/env python3
"""
Article Cleaning Utilities
Heuristic cleaners to remove boilerplate, HTML artifacts, and noise from scraped article text.

This module is intentionally dependency-light and robust to varying schemas in
the stored `articles` JSON files. It looks for common text fields and produces
cleaned text plus a concise excerpt for prompting.
"""

from __future__ import annotations

import html
import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from data_collection.timestamped_storage import TimestampedStorage

logger = logging.getLogger(__name__)


BOILERPLATE_PATTERNS = [
    r"cookie policy",
    r"privacy policy",
    r"terms of service",
    r"subscribe now",
    r"sign up",
    r"newsletter",
    r"advertisement",
    r"sponsored",
    r"all rights reserved",
    r"use of this site",
    r"accept our cookies",
    r"enable javascript",
]


def _strip_html(text: str) -> str:
    # Remove script/style blocks
    text = re.sub(r"<script[\s\S]*?</script>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<style[\s\S]*?</style>", " ", text, flags=re.IGNORECASE)
    # Drop all tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Decode HTML entities
    text = html.unescape(text)
    return text


def _normalize_whitespace(text: str) -> str:
    # Remove bracketed reference numbers and stray artifacts
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\[[0-9]+\]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _drop_boilerplate_lines(text: str) -> str:
    lines = [ln.strip() for ln in text.splitlines()]
    out: List[str] = []
    rx = re.compile("|".join(BOILERPLATE_PATTERNS), re.IGNORECASE)
    for ln in lines:
        if not ln:
            continue
        if rx.search(ln):
            continue
        # Drop obvious navigation/utility lines
        if len(ln) < 5 and not ln.isalpha():
            continue
        out.append(ln)
    # Deduplicate adjacent lines
    dedup: List[str] = []
    prev = None
    for ln in out:
        if ln != prev:
            dedup.append(ln)
        prev = ln
    return " ".join(dedup)


def clean_article_text(raw: str) -> str:
    if not raw:
        return ""
    t = raw
    # If HTML-y, strip tags first
    if "</" in t or "<p" in t or "&nbsp;" in t:
        t = _strip_html(t)
    # Remove URLs (keep tickers/percentages)
    t = re.sub(r"https?://\S+", " ", t)
    t = _drop_boilerplate_lines(t)
    t = _normalize_whitespace(t)
    # Trim extra long text to keep prompts light
    max_len = 4000
    if len(t) > max_len:
        t = t[:max_len].rsplit(" ", 1)[0]
    return t


def make_excerpt(text: str, max_chars: int = 900) -> str:
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    # Avoid cutting mid-sentence if possible
    last_period = cut.rfind(".")
    if last_period > 200:
        cut = cut[: last_period + 1]
    return cut.strip()


def _extract_text_fields(article: Dict[str, Any]) -> Tuple[str, Optional[str]]:
    """Return (raw_text, title) from an article dict by checking common fields."""
    title = None
    for k in ("title", "headline", "name"):
        if isinstance(article.get(k), str) and article[k].strip():
            title = article[k].strip()
            break
    for k in ("clean_text", "text", "content", "body", "summary", "description", "html"):
        v = article.get(k)
        if isinstance(v, str) and v.strip():
            return v, title
    return "", title


def clean_articles_recordset(articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for a in articles or []:
        link = a.get("link") or a.get("url") or a.get("href")
        raw_text, title = _extract_text_fields(a)
        clean = clean_article_text(raw_text)
        excerpt = make_excerpt(clean)
        if not clean:
            continue
        rec = {
            "link": link,
            **({"title": title} if title else {}),
            "clean_text": clean,
            "excerpt": excerpt,
        }
        # Carry source when present
        if isinstance(a.get("source"), str):
            rec["source"] = a["source"]
        out.append(rec)
    return out


def save_cleaned_articles(date: str, cleaned: List[Dict[str, Any]], storage: Optional[TimestampedStorage] = None) -> str:
    storage = storage or TimestampedStorage()
    data = {
        "articles": cleaned,
        "total_articles": len(cleaned),
        "cleaned_at": datetime.now().isoformat(),
    }
    return storage.save_data(data, "articles_clean", date)


def load_articles_any(date: str, storage: Optional[TimestampedStorage] = None) -> Optional[Dict[str, Any]]:
    storage = storage or TimestampedStorage()
    # Prefer already-cleaned; else raw
    j = storage.load_data("articles_clean", date)
    if j:
        return j
    return storage.load_data("articles", date)


def build_link_to_excerpt_map(date: str, storage: Optional[TimestampedStorage] = None) -> Dict[str, str]:
    storage = storage or TimestampedStorage()
    data = load_articles_any(date, storage)
    if not data:
        return {}
    link_map: Dict[str, str] = {}
    for a in data.get("articles", []) or []:
        link = a.get("link") or a.get("url") or a.get("href")
        if not link:
            continue
        excerpt = a.get("excerpt") or a.get("clean_text") or a.get("text")
        if isinstance(excerpt, str) and excerpt.strip():
            link_map[link] = make_excerpt(excerpt)
    return link_map


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Clean article text for one or more dates and save as articles_clean")
    parser.add_argument("--date", type=str, help="YYYYMMDD single date")
    parser.add_argument("--start_date", type=str, default=None)
    parser.add_argument("--end_date", type=str, default=None)
    args = parser.parse_args()

    storage = TimestampedStorage()

    def _process(d: str):
        raw = storage.load_data("articles", d)
        if not raw:
            logger.warning(f"No raw articles for {d}")
            return
        cleaned = clean_articles_recordset(raw.get("articles", []))
        path = save_cleaned_articles(d, cleaned, storage)
        print(path)

    if args.date:
        _process(args.date)
    else:
        if not (args.start_date and args.end_date):
            raise SystemExit("Provide --date or --start_date and --end_date")
        from datetime import datetime as _dt, timedelta as _td
        cur = _dt.strptime(args.start_date, "%Y%m%d")
        end = _dt.strptime(args.end_date, "%Y%m%d")
        while cur <= end:
            _process(cur.strftime("%Y%m%d"))
            cur += _td(days=1)


if __name__ == "__main__":
    main()

