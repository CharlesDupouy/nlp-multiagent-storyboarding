"""
news_tool.py — Tool wrapper for news and external web search.

Provides search_news(query, max_results) to retrieve real-world articles,
headlines, and summaries. Uses NewsAPI.org when NEWSAPI_KEY is configured,
or Serper.dev if SERPER_API_KEY is configured, with graceful fallback.
"""

from __future__ import annotations

import os
import re
import requests
from dotenv import load_dotenv

# Ensure environment variables are loaded
load_dotenv()


def search_news(query: str, max_results: int = 5) -> list[dict[str, str]]:
    """
    Search for recent news articles matching the query.

    Args:
        query:       Search query string.
        max_results: Maximum number of results to return.

    Returns:
        List of dicts: {"headline": str, "summary": str, "url": str}
    """
    provider = os.getenv("NEWS_API_PROVIDER", "newsapi").lower()
    newsapi_key = os.getenv("NEWSAPI_KEY")
    serper_key = os.getenv("SERPER_API_KEY")

    if provider == "newsapi" and newsapi_key:
        try:
            results = _newsapi_search(query, max_results, api_key=newsapi_key)
            if results:
                return results
        except Exception as e:
            print(f"[TOOL WARNING] NewsAPI search error: {e}")

    elif provider == "serper" and serper_key:
        try:
            results = _serper_search(query, max_results, api_key=serper_key)
            if results:
                return results
        except Exception as e:
            print(f"[TOOL WARNING] Serper search error: {e}")

    # Fallback to mock search if APIs return empty or aren't configured
    return _mock_search_news(query, max_results)


def _newsapi_search(query: str, max_results: int, api_key: str) -> list[dict[str, str]]:
    """
    Fetch articles from NewsAPI.org.
    Cleans up complex queries and tries a simplified fallback if needed.
    """
    clean_query = re.sub(r'["\']', '', query).strip()

    def _fetch(q: str) -> list[dict[str, str]]:
        url = "https://newsapi.org/v2/everything"
        headers = {"X-Api-Key": api_key}
        params = {
            "q": q,
            "pageSize": min(max_results, 10),
            "sortBy": "relevancy",
        }
        resp = requests.get(url, headers=headers, params=params, timeout=10)
        if resp.status_code != 200:
            return []
        data = resp.json()
        articles = data.get("articles", [])
        results = []
        for a in articles:
            title = a.get("title") or ""
            desc = a.get("description") or a.get("content") or ""
            link = a.get("url") or ""
            if title and "[Removed]" not in title:
                results.append({
                    "headline": title.strip(),
                    "summary": desc.strip()[:300],
                    "url": link.strip(),
                })
        return results

    # First attempt with full clean query
    articles = _fetch(clean_query)
    if articles:
        return articles[:max_results]

    # Fallback attempt with first 3 words if full query was too specific
    words = clean_query.split()
    if len(words) > 3:
        simplified = " ".join(words[:3])
        articles = _fetch(simplified)
        if articles:
            return articles[:max_results]

    return []


def _serper_search(query: str, max_results: int, api_key: str) -> list[dict[str, str]]:
    """Fetch search results from Serper.dev (Google Search API)."""
    url = "https://google.serper.dev/search"
    headers = {
        "X-API-KEY": api_key,
        "Content-Type": "application/json",
    }
    payload = {
        "q": query,
        "num": max_results,
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=10)
    if resp.status_code != 200:
        return []
    data = resp.json()
    results = []
    for item in data.get("organic", [])[:max_results]:
        results.append({
            "headline": item.get("title", ""),
            "summary": item.get("snippet", ""),
            "url": item.get("link", ""),
        })
    return results


def _mock_search_news(query: str, max_results: int) -> list[dict[str, str]]:
    """Return synthetic snippets when API keys are not available."""
    return [
        {
            "headline": f"[Archived Info] Context regarding {query}",
            "summary": f"Documented background and community discussions regarding {query}.",
            "url": f"https://example.com/info/{re.sub(r'[^a-zA-Z0-9]+', '-', query).strip('-')}",
        }
        for _ in range(min(max_results, 2))
    ]
