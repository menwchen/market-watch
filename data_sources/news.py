"""GNews-based news source for market and financial headlines."""

from __future__ import annotations

from datetime import datetime, timezone

import requests

from config import Config

GNEWS_BASE_URL = "https://gnews.io/api/v4/search"

# Pre-defined queries for broad market coverage
_MARKET_QUERIES: list[str] = [
    "stock market",
    "oil price crude",
    "federal reserve interest rate",
    "inflation CPI",
    "economy GDP",
]


class NewsSource:
    """Fetch financial and market news from the GNews API.

    Requires a valid ``GNEWS_API_KEY`` in the environment / Config.
    All public methods degrade gracefully when the key is missing.
    """

    def __init__(self) -> None:
        self._api_key: str = Config.GNEWS_API_KEY
        if not self._api_key:
            print("[News] Warning: GNEWS_API_KEY not set. News data will be unavailable.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_available(self) -> bool:
        return bool(self._api_key)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_news(
        self,
        query: str,
        max_results: int = 10,
        language: str = "en",
    ) -> list[dict]:
        """Search for news articles matching *query*.

        Parameters
        ----------
        query : str
            Free-text search query.
        max_results : int
            Maximum number of articles to return (GNews caps at 100).
        language : str
            ISO 639-1 language code (default ``"en"``).

        Returns
        -------
        list[dict]
            Each dict has keys: title, description, url, published_at, source.
        """
        if not self._is_available():
            return []

        params: dict = {
            "q": query,
            "lang": language,
            "max": min(max_results, 100),
            "apikey": self._api_key,
        }

        try:
            resp = requests.get(GNEWS_BASE_URL, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as exc:
            print(f"[News] Request error for query '{query}': {exc}")
            return []
        except ValueError as exc:
            print(f"[News] JSON decode error for query '{query}': {exc}")
            return []

        articles = data.get("articles", [])
        results: list[dict] = []

        for article in articles:
            source_info = article.get("source", {})
            results.append(
                {
                    "title": article.get("title", ""),
                    "description": article.get("description", ""),
                    "url": article.get("url", ""),
                    "published_at": article.get("publishedAt", ""),
                    "source": source_info.get("name", ""),
                }
            )

        return results

    def fetch_market_news(self, max_per_query: int = 5) -> list[dict]:
        """Fetch news across several predefined market topics.

        Parameters
        ----------
        max_per_query : int
            Maximum articles per topic query.

        Returns
        -------
        list[dict]
            Aggregated and de-duplicated list of articles, sorted by
            publish date (newest first).
        """
        all_articles: list[dict] = []
        seen_urls: set[str] = set()

        for query in _MARKET_QUERIES:
            articles = self.fetch_news(query, max_results=max_per_query)
            for article in articles:
                url = article.get("url", "")
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    all_articles.append(article)

        # Sort newest first
        def _parse_dt(a: dict) -> datetime:
            try:
                return datetime.fromisoformat(
                    a.get("published_at", "").replace("Z", "+00:00")
                )
            except (ValueError, TypeError):
                return datetime.min.replace(tzinfo=timezone.utc)

        all_articles.sort(key=_parse_dt, reverse=True)
        return all_articles
