import feedparser
from typing import List, Tuple

# List of RSS feeds
RSS_FEEDS = [
    "https://rss.cnn.com/rss/edition.rss",
    "https://feeds.bbci.co.uk/news/rss.xml",
    "https://feeds.reuters.com/reuters/topNews"
]

def fetch_headlines(feed_url: str, num_headlines: int = 5) -> List[Tuple[str, str]]:
    """
    Fetches headlines from a given RSS feed URL.
    
    Args:
        feed_url: The URL of the RSS feed.
        num_headlines: Number of headlines to fetch.
        
    Returns:
        A list of tuples containing (title, link) of the headlines.
    """
    try:
        feed = feedparser.parse(feed_url)
        headlines = [
            (entry.title, entry.link)
            for entry in feed.entries[:num_headlines]
        ]
        return headlines
    except Exception as e:
        print(f"Error fetching from {feed_url}: {e}")
        return []

def main():
    """Main function to fetch headlines from all RSS feeds."""
    all_headlines = []
    for feed_url in RSS_FEEDS:
        print(f"\n--- Fetching from {feed_url} ---")
        headlines = fetch_headlines(feed_url)
        for title, link in headlines:
            print(f"Title: {title}")
            print(f"Link: {link}\n")
            all_headlines.append((title, link))
    
    # Save headlines to a file
    with open("data/headlines.txt", "w") as f:
        for title, link in all_headlines:
            f.write(f"{title}\n{link}\n\n")
    
    print(f"Saved {len(all_headlines)} headlines to data/headlines.txt")

if __name__ == "__main__":
    main()