"""
image_tool.py — Tool wrapper for image search.

Searches for relevant images by keyword only — NO vision model is used.
Images are selected based on textual query relevance (API-side).

The downstream video renderer (demo component) fetches these URLs
to compose the video sketch.
"""

from __future__ import annotations

import os


def search_images(query: str, max_results: int = 3) -> list[dict[str, str]]:
    """
    Search for images matching the query keyword.

    Args:
        query:       Keyword derived from user themes / storyboard scenes.
        max_results: Maximum number of image results to return.

    Returns:
        List of dicts: {"query": str, "url": str, "alt_text": str}

    TODO: Replace mock with a real API call.
    Candidate APIs:
      - Unsplash API  — free tier: 50 req/hour, high-quality photos
      - Pexels API    — free, no attribution required for basic use
      - Pixabay API   — free, large library
    """
    # --- MOCK IMPLEMENTATION ---
    return _mock_search_images(query, max_results)

    # --- REAL IMPLEMENTATION (uncomment when API key is ready) ---
    # provider = os.getenv("IMAGE_API_PROVIDER", "unsplash")
    # if provider == "unsplash":
    #     return _unsplash_search(query, max_results)
    # elif provider == "pexels":
    #     return _pexels_search(query, max_results)
    # else:
    #     raise ValueError(f"Unknown image API provider: {provider}")


def _mock_search_images(query: str, max_results: int) -> list[dict[str, str]]:
    """Return fake image entries for pipeline testing."""
    return [
        {
            "query": query,
            "url": f"https://example.com/images/{query.replace(' ', '_')}_{i}.jpg",
            "alt_text": f"Stock image related to {query}",
        }
        for i in range(min(max_results, 2))
    ]


# ---------------------------------------------------------------------------
# Real API implementations (to be filled in)
# ---------------------------------------------------------------------------

def _unsplash_search(query: str, max_results: int) -> list[dict[str, str]]:
    """
    Fetch images from Unsplash.
    Requires environment variable: UNSPLASH_ACCESS_KEY
    """
    # TODO: implement
    raise NotImplementedError("Unsplash integration not yet implemented.")


def _pexels_search(query: str, max_results: int) -> list[dict[str, str]]:
    """
    Fetch images from Pexels.
    Requires environment variable: PEXELS_API_KEY
    """
    # TODO: implement
    raise NotImplementedError("Pexels integration not yet implemented.")
