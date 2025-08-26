#!/usr/bin/env python3
"""
Article Scraper
Lightweight, best-effort article metadata and text extraction for RSS links.

Goals:
- Keep dependencies optional (bs4/readability used if available, otherwise fallback).
- Be resilient: never raise on a bad page; return a structured error record.
- Respectful defaults: short timeouts, simple per-domain throttling.
"""

from __future__ import annotations

import logging
import re
import time
import hashlib
from typing import List, Dict, Any, Optional
from datetime import datetime
from urllib.parse import urlparse

try:
    # Optional dependencies; used if present
    from bs4 import BeautifulSoup  # type: ignore
except Exception:  # pragma: no cover - optional
    BeautifulSoup = None  # type: ignore

logger = logging.getLogger(__name__)


class ArticleScraper:
    """Best‑effort article scraper using an existing requests.Session.

    Args:
        session: requests-compatible session with headers set by caller.
        timeout_secs: per-request timeout.
        max_chars: cap extracted text length to avoid huge payloads.
        per_domain_delay: seconds to wait between requests to the same domain.
    """

    def __init__(self, session, timeout_secs: int = 8, max_chars: int = 20000, per_domain_delay: float = 1.0):
        self.session = session
        self.timeout_secs = int(timeout_secs)
        self.max_chars = int(max_chars)
        self.per_domain_delay = float(per_domain_delay)
        self._last_hit: Dict[str, float] = {}

    # --- Public API ---
    def scrape_many(self, headlines: List[Dict[str, Any]], max_workers: int = 6) -> List[Dict[str, Any]]:
        """Scrape article info for a batch of headline dicts.

        Deduplicates by URL; returns one article record per unique URL.
        """
        urls = []
        seen = set()
        for h in headlines or []:
            url = (h.get("link") or "").strip()
            if not url:
                continue
            if url in seen:
                continue
            seen.add(url)
            urls.append((url, h))

        results: List[Dict[str, Any]] = []
        if not urls:
            return results

        # Lightweight parallelism; keep small to be nice to sources
        try:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            with ThreadPoolExecutor(max_workers=max(1, int(max_workers))) as ex:
                futures = [ex.submit(self.scrape_one, h) for _, h in urls]
                for fut in as_completed(futures):
                    try:
                        rec = fut.result()
                        if rec:
                            results.append(rec)
                    except Exception as e:  # pragma: no cover - resilience path
                        logger.warning(f"Article scrape task failed: {e}")
        except Exception as e:
            # Fallback to sequential
            logger.warning(f"Falling back to sequential article scraping: {e}")
            for _, h in urls:
                try:
                    rec = self.scrape_one(h)
                    if rec:
                        results.append(rec)
                except Exception:
                    continue

        return results

    def scrape_one(self, headline: Dict[str, Any]) -> Dict[str, Any]:
        """Scrape a single article given a headline record (expects 'link')."""
        url = (headline.get("link") or "").strip()
        if not url:
            return self._error_record(url, headline, error="missing_url")

        # Throttle per domain
        domain = urlparse(url).netloc
        self._throttle(domain)

        http_status = None
        try:
            resp = self.session.get(url, timeout=self.timeout_secs, allow_redirects=True)
            http_status = resp.status_code
            html = resp.text or ""

            if http_status and http_status >= 400:
                return self._error_record(url, headline, http_status=http_status, error=f"http_{http_status}")

            meta = self._extract_meta(html)
            text, word_count = self._extract_text(html)

            canonical_url = meta.get("canonical_url") or meta.get("og:url") or resp.url or url
            article_id = self._mk_id(canonical_url or url)

            record = {
                "article_id": article_id,
                "url": url,
                "canonical_url": canonical_url,
                "source": headline.get("source") or domain,
                "title": meta.get("title") or headline.get("text"),
                "authors": meta.get("authors"),
                "published_at": meta.get("published_time"),
                "top_image": meta.get("image"),
                "lang": meta.get("lang"),
                "summary": meta.get("description"),
                "text": text[: self.max_chars] if text else None,
                "word_count": word_count,
                "headline_text": headline.get("text"),
                "category": headline.get("category"),
                "scrape_status": "ok",
                "http_status": http_status,
                "scraped_at": datetime.now().isoformat(),
            }
            return record
        except Exception as e:
            return self._error_record(url, headline, http_status=http_status, error=str(e))

    # --- Helpers ---
    def _throttle(self, domain: str):
        if not domain:
            return
        now = time.time()
        last = self._last_hit.get(domain, 0.0)
        wait = self.per_domain_delay - (now - last)
        if wait > 0:
            time.sleep(min(wait, 2.0))
        self._last_hit[domain] = time.time()

    def _mk_id(self, url: str) -> str:
        try:
            return hashlib.sha1((url or "").encode("utf-8")).hexdigest()
        except Exception:
            return hashlib.sha1(b"unknown").hexdigest()

    def _extract_meta(self, html: str) -> Dict[str, Any]:
        meta: Dict[str, Any] = {}
        if not html:
            return meta

        # Prefer bs4 if available
        if BeautifulSoup is not None:
            try:
                soup = BeautifulSoup(html, "html.parser")
                # Title
                title_tag = soup.find("meta", property="og:title") or soup.find("meta", attrs={"name": "twitter:title"})
                if title_tag and title_tag.get("content"):
                    meta["title"] = title_tag.get("content").strip()
                elif soup.title and soup.title.string:
                    meta["title"] = soup.title.string.strip()
                # Description
                desc = soup.find("meta", property="og:description") or soup.find("meta", attrs={"name": "description"})
                if desc and desc.get("content"):
                    meta["description"] = desc.get("content").strip()
                # Canonical
                canon = soup.find("link", rel=lambda v: v and "canonical" in v)
                if canon and canon.get("href"):
                    meta["canonical_url"] = canon.get("href").strip()
                # Image
                img = soup.find("meta", property="og:image")
                if img and img.get("content"):
                    meta["image"] = img.get("content").strip()
                # Lang
                html_tag = soup.find("html")
                if html_tag and html_tag.get("lang"):
                    meta["lang"] = html_tag.get("lang").strip()
                # Authors
                auth = soup.find("meta", attrs={"name": "author"})
                if auth and auth.get("content"):
                    meta["authors"] = auth.get("content").strip()
                # Published time
                pub = soup.find("meta", property="article:published_time")
                if pub and pub.get("content"):
                    meta["published_time"] = pub.get("content").strip()
                # og:url
                ogu = soup.find("meta", property="og:url")
                if ogu and ogu.get("content"):
                    meta["og:url"] = ogu.get("content").strip()
                return meta
            except Exception:
                # Fall through to regex-based best effort
                pass

        # Regex fallbacks
        def _meta(name: str) -> Optional[str]:
            m = re.search(rf'<meta[^>]+(?:name|property)=["\']{re.escape(name)}["\'][^>]*content=["\']([^"\']+)["\']', html, re.I)
            return m.group(1).strip() if m else None

        def _link_rel(rel: str) -> Optional[str]:
            m = re.search(rf'<link[^>]+rel=["\'][^"\']*{re.escape(rel)}[^"\']*["\'][^>]*href=["\']([^"\']+)["\']', html, re.I)
            return m.group(1).strip() if m else None

        meta["title"] = _meta("og:title") or _meta("twitter:title")
        meta["description"] = _meta("og:description") or _meta("description")
        meta["image"] = _meta("og:image")
        meta["published_time"] = _meta("article:published_time")
        meta["og:url"] = _meta("og:url")
        meta["canonical_url"] = _link_rel("canonical")
        return meta

    def _extract_text(self, html: str) -> tuple[Optional[str], Optional[int]]:
        if not html:
            return None, None

        # Prefer bs4-based paragraph extraction if available
        if BeautifulSoup is not None:
            try:
                soup = BeautifulSoup(html, "html.parser")
                # Remove script/style/nav/footer
                for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside"]):
                    tag.decompose()
                paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
                text = "\n\n".join([p for p in paragraphs if p])
                text = text.strip()
                if text:
                    words = len(text.split())
                    return text[: self.max_chars], words
            except Exception:
                pass

        # Fallback: crude strip of tags for a rough body text
        try:
            cleaned = re.sub(r"<script[\s\S]*?</script>", " ", html, flags=re.I)
            cleaned = re.sub(r"<style[\s\S]*?</style>", " ", cleaned, flags=re.I)
            cleaned = re.sub(r"<[^>]+>", " ", cleaned)
            cleaned = re.sub(r"\s+", " ", cleaned).strip()
            words = len(cleaned.split()) if cleaned else 0
            return (cleaned[: self.max_chars] if cleaned else None), (words if cleaned else None)
        except Exception:
            return None, None

    def _error_record(self, url: str, headline: Dict[str, Any], http_status: Optional[int] = None, error: str = "unknown_error") -> Dict[str, Any]:
        domain = urlparse(url).netloc if url else None
        return {
            "article_id": self._mk_id(url or (headline.get("text") or "")),
            "url": url,
            "canonical_url": None,
            "source": headline.get("source") or domain,
            "title": headline.get("text"),
            "authors": None,
            "published_at": None,
            "top_image": None,
            "lang": None,
            "summary": None,
            "text": None,
            "word_count": None,
            "headline_text": headline.get("text"),
            "category": headline.get("category"),
            "scrape_status": "error",
            "http_status": http_status,
            "error": error,
            "scraped_at": datetime.now().isoformat(),
        }


def main():  # Simple manual test
    import json
    import requests
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) ArticleScraper/1.0'
    })
    scraper = ArticleScraper(session)
    sample = [{"text": "Test", "link": "https://example.com", "source": "example"}]
    out = scraper.scrape_many(sample)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

