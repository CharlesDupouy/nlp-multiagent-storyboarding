"""
news_tool.py — Tool wrapper for news / web search.

This module wraps an external news or search API.
The API provider is TBD (candidates: NewsAPI, Serper, DuckDuckGo).

For the MVP, a mock implementation is provided so the pipeline
can run end-to-end before the real API is integrated.
"""

from __future__ import annotations

import os


def search_news(query: str, max_results: int = 5) -> list[dict[str, str]]:
    """
    Search for recent news articles matching the query.

    Args:
        query:       Search query string (derived from user interests).
        max_results: Maximum number of results to return.

    Returns:
        List of dicts: {"headline": str, "summary": str, "url": str}

    TODO: Replace the mock implementation with a real API call.
    Candidate APIs:
      - NewsAPI (newsapi.org) — free tier: 100 req/day
      - Serper (serper.dev)   — Google Search via API, cheap
      - DuckDuckGo Instant Answers — free but limited
    """
    # --- MOCK IMPLEMENTATION (for testing without a real API key) ---
    return _mock_search_news(query, max_results)

    # --- REAL IMPLEMENTATION (uncomment when API key is ready) ---
    # provider = os.getenv("NEWS_API_PROVIDER", "newsapi")
    # if provider == "newsapi":
    #     return _newsapi_search(query, max_results)
    # elif provider == "serper":
    #     return _serper_search(query, max_results)
    # else:
    #     raise ValueError(f"Unknown news API provider: {provider}")


def _mock_search_news(query: str, max_results: int) -> list[dict[str, str]]:
    """Return fake news snippets for testing the pipeline end-to-end."""
    return [
        {
            "headline": f"[MOCK] Latest developments in {query}",
            "summary": f"Researchers and practitioners in the field of {query} "
                       f"continue to make significant advances this week.",
            "url": f"https://example.com/news/{query.replace(' ', '-')}",
        }
        for _ in range(min(max_results, 2))
    ]


# ---------------------------------------------------------------------------
# Real API implementations (to be filled in)
# ---------------------------------------------------------------------------

def _newsapi_search(query: str, max_results: int) -> list[dict[str, str]]:
    """
    Fetch news from NewsAPI.org.
    Requires environment variable: NEWSAPI_KEY
    """
    # TODO: implement
    raise NotImplementedError("NewsAPI integration not yet implemented.")


def _serper_search(query: str, max_results: int) -> list[dict[str, str]]:
    """
    Fetch search results from Serper.dev (Google Search API).
    Requires environment variable: SERPER_API_KEY
    """
    # TODO: implement
    raise NotImplementedError("Serper integration not yet implemented.")
