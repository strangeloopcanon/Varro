#!/usr/bin/env python3
"""
Enhanced RSS Collector for Financial News
Collects headlines from multiple RSS feeds with error handling and timestamped storage.
"""

import json
import logging
import os
import requests
import feedparser
from datetime import datetime, timezone
from typing import List, Dict, Any
from urllib.parse import urlparse
import time

logger = logging.getLogger(__name__)

class EnhancedRSSCollector:
    """Collects headlines from multiple RSS feeds with error handling."""
    
    def __init__(self):
        # Load sources from config with sensible defaults
        self.rss_sources, self.source_name_map = self._load_sources_from_config()

        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        })

    def _load_sources_from_config(self):
        """Load RSS sources and optional source name mapping from config/rss_sources.json."""
        default_sources = {
            "general_news": [
                "https://feeds.bbci.co.uk/news/rss.xml",
                "https://feeds.reuters.com/reuters/topNews",
            ],
            "financial_news": [
                "https://feeds.marketwatch.com/marketwatch/topstories/",
                "https://seekingalpha.com/feed.xml",
            ],
            "market_specific": [
                "https://www.investing.com/rss/news_301.rss",
            ],
        }
        source_name_map = {}

        try:
            cfg_path = os.path.join("config", "rss_sources.json")
            with open(cfg_path, "r") as f:
                cfg = json.load(f)
            rss_sources = {
                k: v for k, v in cfg.items() if isinstance(v, list)
            } or default_sources
            source_name_map = cfg.get("source_names", {})
            logger.info("Loaded RSS sources from config/rss_sources.json")
            return rss_sources, source_name_map
        except Exception as e:
            logger.warning(f"Falling back to default RSS sources (error loading config): {e}")
            return default_sources, source_name_map
    
    def collect_headlines(self) -> List[Dict[str, Any]]:
        """Collect headlines from all RSS sources."""
        all_headlines = []
        
        for category, urls in self.rss_sources.items():
            logger.info(f"Collecting from {category} sources...")
            
            for url in urls:
                try:
                    headlines = self._collect_from_source(url, category)
                    all_headlines.extend(headlines)
                    logger.info(f"Collected {len(headlines)} headlines from {url}")
                    
                    # Rate limiting
                    time.sleep(1)
                    
                except Exception as e:
                    logger.error(f"Error collecting from {url}: {e}")
                    continue
        
        # Remove duplicates and sort by timestamp
        unique_headlines = self._deduplicate_headlines(all_headlines)
        sorted_headlines = sorted(unique_headlines, key=lambda x: x['timestamp'], reverse=True)
        
        logger.info(f"Total unique headlines collected: {len(sorted_headlines)}")
        return sorted_headlines
    
    def _collect_from_source(self, url: str, category: str) -> List[Dict[str, Any]]:
        """Collect headlines from a single RSS source."""
        headlines = []
        
        try:
            # Parse RSS feed
            feed = feedparser.parse(url)
            
            if feed.bozo:
                logger.warning(f"Feed parsing issues for {url}")
            
            for entry in feed.entries:
                headline = self._parse_entry(entry, url, category)
                if headline:
                    headlines.append(headline)
            
        except Exception as e:
            logger.error(f"Failed to parse RSS feed {url}: {e}")
            return []
        
        return headlines
    
    def _parse_entry(self, entry, source_url: str, category: str) -> Dict[str, Any]:
        """Parse a single RSS entry."""
        try:
            # Extract title
            title = entry.get('title', '').strip()
            if not title:
                return None
            
            # Extract link
            link = entry.get('link', '')
            
            # Extract timestamp
            timestamp = self._extract_timestamp(entry)
            
            # Extract source name from URL
            source_name = self._extract_source_name(source_url)
            
            # Basic filtering for financial relevance
            if self._is_financially_relevant(title):
                return {
                    "text": title,
                    "link": link,
                    "source": source_name,
                    "category": category,
                    "timestamp": timestamp.isoformat(),
                    "url": source_url
                }
            
        except Exception as e:
            logger.error(f"Error parsing entry: {e}")
            return None
        
        return None
    
    def _extract_timestamp(self, entry) -> datetime:
        """Extract timestamp from RSS entry."""
        # Try different timestamp fields
        timestamp_fields = ['published_parsed', 'updated_parsed', 'created_parsed']
        
        for field in timestamp_fields:
            if hasattr(entry, field) and getattr(entry, field):
                time_tuple = getattr(entry, field)
                return datetime(*time_tuple[:6], tzinfo=timezone.utc)
        
        # Fallback to current time
        return datetime.now(timezone.utc)
    
    def _extract_source_name(self, url: str) -> str:
        """Extract source name from URL, using config mapping when available."""
        domain = urlparse(url).netloc.replace('www.', '')
        if self.source_name_map and domain in self.source_name_map:
            return self.source_name_map[domain]
        # Fallback: simple domain-based name
        return domain.replace('.com', '').replace('.co.uk', '')
    
    def _is_financially_relevant(self, title: str) -> bool:
        """Check if headline is financially relevant."""
        financial_keywords = [
            'fed', 'federal reserve', 'interest rate', 'inflation', 'earnings',
            'stock', 'market', 'trading', 'investment', 'economy', 'gdp',
            'unemployment', 'jobs', 'oil', 'gold', 'dollar', 'euro', 'bond',
            'treasury', 'central bank', 'monetary', 'fiscal', 'trade',
            'tariff', 'tariffs', 'deficit', 'surplus', 'recession',
            'growth', 'economic', 'financial', 'bank', 'banking'
        ]
        
        title_lower = title.lower()
        return any(keyword in title_lower for keyword in financial_keywords)
    
    def _deduplicate_headlines(self, headlines: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate headlines based on text similarity."""
        seen_texts = set()
        unique_headlines = []
        
        for headline in headlines:
            # Normalize text for comparison
            normalized_text = headline['text'].lower().strip()
            
            if normalized_text not in seen_texts:
                seen_texts.add(normalized_text)
                unique_headlines.append(headline)
        
        return unique_headlines
    
    def save_headlines(self, headlines: List[Dict[str, Any]], date: str = None):
        """Save headlines to timestamped file."""
        if date is None:
            date = datetime.now().strftime("%Y%m%d")
        
        filename = f"timestamped_storage/{date}_headlines.json"
        os.makedirs("timestamped_storage", exist_ok=True)
        
        data = {
            "date": date,
            "collected_at": datetime.now().isoformat(),
            "total_headlines": len(headlines),
            "headlines": headlines
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved {len(headlines)} headlines to {filename}")
        return filename

def main():
    """Main function for testing."""
    collector = EnhancedRSSCollector()
    
    # Collect headlines
    headlines = collector.collect_headlines()
    
    # Save to timestamped storage
    filename = collector.save_headlines(headlines)
    
    print(f"Collected {len(headlines)} headlines")
    print(f"Saved to: {filename}")
    
    # Show sample headlines
    print("\nSample headlines:")
    for i, headline in enumerate(headlines[:5]):
        print(f"{i+1}. {headline['text']} ({headline['source']})")

if __name__ == "__main__":
    main() 
