#!/usr/bin/env python3
"""
Historical Data Collector
One-time historical dataset creation for initial model training.
"""

import json
import logging
import os
from datetime import datetime, timedelta
from typing import List, Dict, Any
from pathlib import Path
import time

from enhanced_rss_collector import EnhancedRSSCollector
from timestamped_storage import TimestampedStorage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HistoricalDataCollector:
    """Collects historical data for initial model training."""
    
    def __init__(self):
        self.rss_collector = EnhancedRSSCollector()
        self.storage = TimestampedStorage()
        
        # Historical date range (6 months)
        self.end_date = datetime.now()
        self.start_date = self.end_date - timedelta(days=180)
        
        # Historical data storage
        self.historical_dir = Path("training")
        self.historical_dir.mkdir(exist_ok=True)
    
    def collect_historical_data(self):
        """Collect historical headlines and create training dataset."""
        logger.info("Starting historical data collection...")
        logger.info(f"Date range: {self.start_date.date()} to {self.end_date.date()}")
        
        # Collect headlines for each day in the range
        all_historical_data = []
        current_date = self.start_date
        
        while current_date <= self.end_date:
            date_str = current_date.strftime("%Y%m%d")
            
            logger.info(f"Collecting data for {date_str}...")
            
            # Try to collect headlines for this date
            # Note: This is a simplified approach - in practice, you'd need
            # historical RSS feeds or archived news data
            historical_headlines = self._simulate_historical_headlines(date_str)
            
            if historical_headlines:
                # Create training examples for this date
                training_examples = self._create_training_examples(historical_headlines, date_str)
                all_historical_data.extend(training_examples)
                
                logger.info(f"Created {len(training_examples)} training examples for {date_str}")
            
            current_date += timedelta(days=1)
            
            # Rate limiting
            time.sleep(0.1)
        
        # Save historical dataset
        self._save_historical_dataset(all_historical_data)
        
        logger.info(f"Historical data collection complete. Total examples: {len(all_historical_data)}")
        return all_historical_data
    
    def _simulate_historical_headlines(self, date_str: str) -> List[Dict[str, Any]]:
        """Simulate historical headlines for a given date."""
        # This is a simplified simulation - in practice, you'd need real historical data
        # For now, we'll create some realistic historical headlines
        
        historical_headlines = []
        
        # Simulate different types of financial news
        headline_templates = [
            "Fed signals {action} in {month}",
            "{company} reports {result} earnings",
            "{market} {direction} on {news}",
            "{country} {action} interest rates",
            "{commodity} prices {direction} on {news}",
            "{sector} stocks {direction} after {news}",
            "{economic_indicator} {direction} to {value}",
            "{central_bank} {action} monetary policy"
        ]
        
        # Historical context for different dates
        date_obj = datetime.strptime(date_str, "%Y%m%d")
        
        # Create realistic headlines based on the date
        for i in range(10):  # 10 headlines per day
            template = headline_templates[i % len(headline_templates)]
            
            # Fill template based on historical context
            headline = self._fill_historical_template(template, date_obj)
            
            historical_headlines.append({
                "text": headline,
                "link": f"https://example.com/news/{date_str}_{i}",
                "source": "historical_simulation",
                "category": "financial_news",
                "timestamp": date_obj.isoformat(),
                "url": "historical_simulation"
            })
        
        return historical_headlines
    
    def _fill_historical_template(self, template: str, date_obj: datetime) -> str:
        """Fill a headline template with realistic historical data."""
        import random
        
        # Historical context based on date
        year = date_obj.year
        month = date_obj.month
        
        # Different contexts for different time periods
        if year == 2023:
            if month in [1, 2, 3]:
                context = {
                    "action": "rate hikes",
                    "month": "March",
                    "company": "Apple",
                    "result": "strong",
                    "market": "Tech stocks",
                    "direction": "rally",
                    "news": "AI breakthrough",
                    "country": "US",
                    "commodity": "Oil",
                    "economic_indicator": "Inflation",
                    "value": "6.5%",
                    "central_bank": "Federal Reserve"
                }
            elif month in [4, 5, 6]:
                context = {
                    "action": "pause rate hikes",
                    "month": "June",
                    "company": "Tesla",
                    "result": "mixed",
                    "market": "Bond market",
                    "direction": "decline",
                    "news": "debt ceiling concerns",
                    "country": "UK",
                    "commodity": "Gold",
                    "economic_indicator": "GDP",
                    "value": "2.1%",
                    "central_bank": "ECB"
                }
            else:
                context = {
                    "action": "consider rate cuts",
                    "month": "December",
                    "company": "Microsoft",
                    "result": "excellent",
                    "market": "Cryptocurrency",
                    "direction": "surge",
                    "news": "ETF approval",
                    "country": "Japan",
                    "commodity": "Silver",
                    "economic_indicator": "Unemployment",
                    "value": "3.7%",
                    "central_bank": "Bank of Japan"
                }
        else:
            # Default context
            context = {
                "action": "maintain rates",
                "month": "next month",
                "company": "Major Corp",
                "result": "solid",
                "market": "Global markets",
                "direction": "move",
                "news": "economic data",
                "country": "Global",
                "commodity": "Commodities",
                "economic_indicator": "Economic data",
                "value": "stable",
                "central_bank": "Central banks"
            }
        
        # Fill the template
        try:
            headline = template.format(**context)
        except KeyError:
            # Fallback if template has missing keys
            headline = f"Financial news on {date_obj.strftime('%Y-%m-%d')}"
        
        return headline
    
    def _create_training_examples(self, headlines: List[Dict[str, Any]], date_str: str) -> List[Dict[str, Any]]:
        """Create training examples from historical headlines."""
        training_examples = []
        
        for headline in headlines:
            # Create the prediction prompt
            prompt = self._create_prediction_prompt(headline["text"])
            
            # Simulate a historical prediction (this would be from a model)
            prediction = self._simulate_historical_prediction(headline["text"])
            
            # Simulate historical outcome (this would be real market data)
            outcome = self._simulate_historical_outcome(headline["text"], date_str)
            
            # Calculate accuracy score
            accuracy_score = self._calculate_historical_accuracy(prediction, outcome)
            
            training_example = {
                "headline": headline["text"],
                "prompt": prompt,
                "prediction": prediction,
                "outcome": outcome,
                "accuracy_score": accuracy_score,
                "date": date_str,
                "source": headline["source"]
            }
            
            training_examples.append(training_example)
        
        return training_examples
    
    def _create_prediction_prompt(self, headline: str) -> str:
        """Create prediction prompt for a headline."""
        try:
            with open("config/prompt_templates.json", 'r') as f:
                prompt_config = json.load(f)
            return prompt_config["basic_prompt"].format(headline=headline)
        except Exception as e:
            logger.error(f"Error loading prompt template: {e}")
            # Fallback to hardcoded prompt
            return f"""
Headline: "{headline}"

Based on this news, what do you think will happen and what's the trade?

Please structure your response as follows:
1. **Market Impact**: How will this affect markets broadly?
2. **Specific Assets**: Which specific assets will be most affected?
3. **Trade Recommendation**: What specific trade would you make?
4. **Timeframe**: When do you expect this to play out?
5. **Risk Factors**: What could go wrong with this prediction?
6. **World View**: What world does this news imagine or create? What narrative does it suggest?

Think like a trader - be specific about what to buy/sell and why.
"""
    
    def _simulate_historical_prediction(self, headline: str) -> str:
        """Simulate a historical prediction for a headline."""
        # This would be generated by a model - for now, simulate realistic predictions
        
        if "Fed" in headline and "rate" in headline.lower():
            return """
**Market Impact**: Fed actions will significantly impact bond yields and risk assets.
**Specific Assets**: Treasury bonds, bank stocks, and the dollar will be most affected.
**Trade Recommendation**: Buy TLT calls if dovish, short bank stocks if hawkish.
**Timeframe**: Next 1-3 days
**Risk Factors**: Fed could walk back comments or data could surprise.
**World View**: This suggests the Fed is either maintaining or changing its monetary stance, which creates a new narrative for market participants.
"""
        elif "earnings" in headline.lower():
            return """
**Market Impact**: Earnings results will drive individual stock and sector performance.
**Specific Assets**: The reporting company and its competitors will be most affected.
**Trade Recommendation**: Buy calls if earnings beat, puts if miss expectations.
**Timeframe**: Next 1-2 days
**Risk Factors**: Guidance could be more important than actual results.
**World View**: This creates a narrative about company/sector strength and market sentiment.
"""
        else:
            return """
**Market Impact**: This news will have moderate impact on specific sectors.
**Specific Assets**: Related stocks and commodities will be affected.
**Trade Recommendation**: Monitor for specific trading opportunities.
**Timeframe**: Next few days
**Risk Factors**: Market reaction could be muted or unexpected.
**World View**: This suggests ongoing trends in the market narrative.
"""
    
    def _simulate_historical_outcome(self, headline: str, date_str: str) -> str:
        """Simulate historical outcome for a headline."""
        # This would be real market data - for now, simulate realistic outcomes
        
        if "Fed" in headline:
            return "Fed followed through with expected action. Treasury yields moved as predicted. Market reaction was moderate."
        elif "earnings" in headline.lower():
            return "Company reported earnings in line with expectations. Stock moved 2% in response. Sector peers also moved."
        else:
            return "News had expected market impact. Related assets moved moderately in predicted direction."
    
    def _calculate_historical_accuracy(self, prediction: str, outcome: str) -> float:
        """Calculate accuracy score for historical prediction."""
        # Simplified accuracy calculation
        # In practice, this would be more sophisticated
        
        # Check if prediction mentions specific assets
        has_specific_assets = any(word in prediction.lower() for word in ["buy", "sell", "calls", "puts", "stock", "bond"])
        
        # Check if outcome mentions market movement
        has_market_movement = any(word in outcome.lower() for word in ["moved", "changed", "impact", "response"])
        
        # Base score
        score = 0.5
        
        if has_specific_assets:
            score += 0.2
        
        if has_market_movement:
            score += 0.3
        
        # Add some randomness to simulate real variation
        import random
        score += random.uniform(-0.1, 0.1)
        
        return max(0.0, min(1.0, score))
    
    def _save_historical_dataset(self, training_data: List[Dict[str, Any]]):
        """Save historical training dataset."""
        dataset_path = self.historical_dir / "historical_training_dataset.json"
        
        dataset = {
            "created_at": datetime.now().isoformat(),
            "total_examples": len(training_data),
            "date_range": {
                "start": self.start_date.strftime("%Y%m%d"),
                "end": self.end_date.strftime("%Y%m%d")
            },
            "training_examples": training_data
        }
        
        with open(dataset_path, 'w') as f:
            json.dump(dataset, f, indent=2)
        
        logger.info(f"Saved historical dataset to {dataset_path}")
        logger.info(f"Total training examples: {len(training_data)}")

def main():
    """Main function for historical data collection."""
    collector = HistoricalDataCollector()
    
    # Collect historical data
    historical_data = collector.collect_historical_data()
    
    print(f"Collected {len(historical_data)} historical training examples")
    print("Historical dataset saved to training/historical_training_dataset.json")

if __name__ == "__main__":
    main() 