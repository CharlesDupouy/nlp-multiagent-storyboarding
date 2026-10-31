"""
image_tool.py — Tool wrapper for photographic image asset search.

Searches for high-resolution stock images by keyword without needing a vision model.
Uses Unsplash API (when UNSPLASH_ACCESS_KEY is set) or Pexels API (when PEXELS_API_KEY is set),
with intelligent stock query cleaning and candidate retrieval.
"""

from __future__ import annotations

import os
import re
from typing import Any
import requests
from dotenv import load_dotenv

# Ensure environment variables are loaded
load_dotenv()

# Words that ruin stock photo search by injecting cinematography instructions instead of physical objects
_BANNED_DIRECTIVES = re.compile(
    r'\b(macro|closeup|close-up|split screen|exploded view|cinematic|footage|4k|8k|'
    r'wide shot|establishing shot|drone shot|pov|2000s|vintage style|action shot)\b',
    re.IGNORECASE,
)


def clean_stock_query(query: str) -> str:
    """
    Transform editorial/cinematic prompts into clean, physical stock photography terms.
    E.g. 'teenage hand clicking Bic 4 color pen macro' -> 'hand clicking ballpoint pen'
    """
    q = re.sub(r'["\']', '', query).strip()
    # Remove cinematic/shot directions
    q = _BANNED_DIRECTIVES.sub('', q)
    # Replace brand names and numerical formats with generic stock terms
    q = re.sub(r'\bbic\b', 'ballpoint', q, flags=re.IGNORECASE)
    q = re.sub(r'\b4[- ]colors?\b', 'multicolor', q, flags=re.IGNORECASE)
    q = re.sub(r'\b4[- ]couleurs?\b', 'multicolor', q, flags=re.IGNORECASE)
    # Collapse whitespace
    q = re.sub(r'\s+', ' ', q).strip()
    return q


def search_image_candidates(
    query: str,
    fallback_query: str = "",
    max_results: int = 5,
) -> list[dict[str, Any]]:
    """
    Search for multiple candidate images with rich visual descriptions.
    Enables downstream agents to select the most representative image.

    Args:
        query:          Primary search query.
        fallback_query: Backup query if the primary query yields 0 results.
        max_results:    Number of candidates to retrieve.

    Returns:
        List of dicts: {"candidate_id": str, "url": str, "description": str, "alt_text": str}
    """
    provider = os.getenv("IMAGE_API_PROVIDER", "unsplash").lower()
    unsplash_key = os.getenv("UNSPLASH_ACCESS_KEY")
    pexels_key = os.getenv("PEXELS_API_KEY")

    cleaned_primary = clean_stock_query(query)
    cleaned_fallback = clean_stock_query(fallback_query) if fallback_query else ""

    if provider == "unsplash" and unsplash_key:
        try:
            candidates = _unsplash_fetch_candidates(cleaned_primary, max_results, unsplash_key)
            if candidates:
                return candidates

            # Try fallback query if primary returned 0
            if cleaned_fallback and cleaned_fallback != cleaned_primary:
                candidates = _unsplash_fetch_candidates(cleaned_fallback, max_results, unsplash_key)
                if candidates:
                    return candidates

            # Core noun fallback: take the last 2 words (typically the noun phrase)
            words = cleaned_primary.split()
            if len(words) > 2:
                core_nouns = " ".join(words[-2:])
                candidates = _unsplash_fetch_candidates(core_nouns, max_results, unsplash_key)
                if candidates:
                    return candidates
        except Exception as e:
            print(f"[TOOL WARNING] Unsplash candidate search error: {e}")

    elif provider == "pexels" and pexels_key:
        try:
            return _pexels_search(cleaned_primary, max_results, pexels_key)
        except Exception as e:
            print(f"[TOOL WARNING] Pexels candidate search error: {e}")

    # Fallback to mock images if API fails or key is missing
    return _mock_search_images(query, max_results)


def search_images(query: str, max_results: int = 3) -> list[dict[str, str]]:
    """Backwards-compatible image search returning candidate list."""
    candidates = search_image_candidates(query=query, max_results=max_results)
    return [
        {
            "query": query,
            "url": c.get("url", ""),
            "alt_text": c.get("alt_text", ""),
        }
        for c in candidates
    ]


def _unsplash_fetch_candidates(q: str, max_results: int, api_key: str) -> list[dict[str, Any]]:
    """Query Unsplash API and return candidate metadata."""
    if not q:
        return []
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
    results: list[dict[str, Any]] = []

    for idx, p in enumerate(photos):
        img_url = p.get("urls", {}).get("regular") or p.get("urls", {}).get("small")
        raw_desc = p.get("alt_description") or p.get("description") or f"Photo of {q}"
        photographer = p.get("user", {}).get("name")
        if img_url:
            alt_with_credit = f"{raw_desc} (Photo by {photographer})" if photographer else raw_desc
            results.append({
                "candidate_id": f"cand{idx+1}",
                "url": img_url,
                "description": raw_desc.strip(),
                "alt_text": alt_with_credit.strip(),
                "photographer": photographer or "Unsplash",
            })

    return results[:max_results]


def _pexels_search(query: str, max_results: int, api_key: str) -> list[dict[str, Any]]:
    """Fetch images from Pexels API."""
    url = "https://api.pexels.com/v1/search"
    headers = {"Authorization": api_key}
    params = {"query": query, "per_page": max_results}
    resp = requests.get(url, headers=headers, params=params, timeout=10)
    if resp.status_code != 200:
        return []
    data = resp.json()
    results = []
    for idx, p in enumerate(data.get("photos", [])[:max_results]):
        src = p.get("src", {})
        img_url = src.get("large") or src.get("medium")
        alt = p.get("alt") or f"Photo of {query}"
        photographer = p.get("photographer")
        if img_url:
            alt_with_credit = f"{alt} (Photo by {photographer})" if photographer else alt
            results.append({
                "candidate_id": f"cand{idx+1}",
                "url": img_url,
                "description": alt.strip(),
                "alt_text": alt_with_credit.strip(),
                "photographer": photographer or "Pexels",
            })
    return results


def _mock_search_images(query: str, max_results: int) -> list[dict[str, Any]]:
    """Return synthetic image placeholders when API keys are not available."""
    clean = re.sub(r'[^a-zA-Z0-9]+', '_', query).strip('_')
    return [
        {
            "candidate_id": f"cand{i+1}",
            "query": query,
            "url": f"https://example.com/images/{clean}_{i}.jpg",
            "description": f"Stock photography depiction of {query}",
            "alt_text": f"Stock image related to {query}",
        }
        for i in range(min(max_results, 2))
    ]
