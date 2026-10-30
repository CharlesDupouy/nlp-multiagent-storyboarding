"""
image_tool.py — Tool wrapper for photographic image asset search.

Searches for high-resolution stock images by keyword without needing a vision model.
Uses Unsplash API (when UNSPLASH_ACCESS_KEY is set) or Pexels API (when PEXELS_API_KEY is set),
with graceful fallback to mock image placeholders.
"""

from __future__ import annotations

import os
import re
import requests
from dotenv import load_dotenv

# Ensure environment variables are loaded
load_dotenv()


def search_images(query: str, max_results: int = 3) -> list[dict[str, str]]:
    """
    Search for photographic images matching the query keyword.

    Args:
        query:       Keyword derived from storyboard scenes.
        max_results: Maximum number of image results to return.

    Returns:
        List of dicts: {"query": str, "url": str, "alt_text": str}
    """
    provider = os.getenv("IMAGE_API_PROVIDER", "unsplash").lower()
    unsplash_key = os.getenv("UNSPLASH_ACCESS_KEY")
    pexels_key = os.getenv("PEXELS_API_KEY")

    if provider == "unsplash" and unsplash_key:
        try:
            results = _unsplash_search(query, max_results, api_key=unsplash_key)
            if results:
                return results
        except Exception as e:
            print(f"[TOOL WARNING] Unsplash search error: {e}")

    elif provider == "pexels" and pexels_key:
        try:
            results = _pexels_search(query, max_results, api_key=pexels_key)
            if results:
                return results
        except Exception as e:
            print(f"[TOOL WARNING] Pexels search error: {e}")

    # Fallback to mock images if API returns empty or key is not provided
    return _mock_search_images(query, max_results)


def _unsplash_search(query: str, max_results: int, api_key: str) -> list[dict[str, str]]:
    """
    Fetch images from Unsplash API.
    Attempts query simplification if specific phrase yields 0 results.
    """
    clean_query = re.sub(r'["\']', '', query).strip()

    def _fetch(q: str) -> list[dict[str, str]]:
        url = "https://api.unsplash.com/search/photos"
        headers = {"Authorization": f"Client-ID {api_key}"}
        params = {
            "query": q,
            "per_page": min(max_results, 10),
        }
        resp = requests.get(url, headers=headers, params=params, timeout=10)
        if resp.status_code != 200:
            return []
        data = resp.json()
        photos = data.get("results", [])
        results = []
        for p in photos:
            img_url = p.get("urls", {}).get("regular") or p.get("urls", {}).get("small")
            alt = p.get("alt_description") or p.get("description") or f"Photo of {q}"
            photographer = p.get("user", {}).get("name")
            if img_url:
                alt_with_credit = f"{alt} (Photo by {photographer})" if photographer else alt
                results.append({
                    "query": query,
                    "url": img_url,
                    "alt_text": alt_with_credit,
                })
        return results

    # 1. Try full cleaned query
    photos = _fetch(clean_query)
    if photos:
        return photos[:max_results]

    # 2. Fallback to first 2-3 words if too specific
    words = clean_query.split()
    if len(words) > 3:
        simplified = " ".join(words[:3])
        photos = _fetch(simplified)
        if photos:
            return photos[:max_results]

    return []


def _pexels_search(query: str, max_results: int, api_key: str) -> list[dict[str, str]]:
    """Fetch images from Pexels API."""
    url = "https://api.pexels.com/v1/search"
    headers = {"Authorization": api_key}
    params = {"query": query, "per_page": max_results}
    resp = requests.get(url, headers=headers, params=params, timeout=10)
    if resp.status_code != 200:
        return []
    data = resp.json()
    results = []
    for p in data.get("photos", [])[:max_results]:
        src = p.get("src", {})
        img_url = src.get("large") or src.get("medium")
        alt = p.get("alt") or f"Photo of {query}"
        photographer = p.get("photographer")
        if img_url:
            alt_with_credit = f"{alt} (Photo by {photographer})" if photographer else alt
            results.append({
                "query": query,
                "url": img_url,
                "alt_text": alt_with_credit,
            })
    return results


def _mock_search_images(query: str, max_results: int) -> list[dict[str, str]]:
    """Return synthetic image placeholders when API keys are not available."""
    clean = re.sub(r'[^a-zA-Z0-9]+', '_', query).strip('_')
    return [
        {
            "query": query,
            "url": f"https://example.com/images/{clean}_{i}.jpg",
            "alt_text": f"Stock image related to {query}",
        }
        for i in range(min(max_results, 2))
    ]
