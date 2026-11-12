"""
image_tool.py — Tool wrapper for photographic and web image asset search.

Searches for high-resolution photographic/illustrative images by keyword without needing
a vision model. Primary provider is Serper.dev Google Image Search (when SERPER_API_KEY
is set or IMAGE_API_PROVIDER=serper): if Serper fails or finds no usable verified images,
it returns [] cleanly so affected scenes omit images instead of failing over to other
providers or mock placeholder URLs. Intelligent stock query cleaning, watermark/login-wall
filtering, and URL reachability checks are applied to every candidate.
"""

from __future__ import annotations

import os
import re
from typing import Any
from urllib.parse import urlparse
import requests
from dotenv import load_dotenv

# Ensure environment variables are loaded
load_dotenv()

# Words that ruin stock/image search by injecting cinematography instructions instead of physical objects
_BANNED_DIRECTIVES = re.compile(
    r'\b(macro|closeup|close-up|split screen|exploded view|cinematic|footage|4k|8k|'
    r'wide shot|establishing shot|drone shot|pov|2000s|vintage style|action shot)\b',
    re.IGNORECASE,
)

# Watermarked stock agencies, disguised stock CDNs, and bot-blocked social login walls
_BLOCKED_IMAGE_DOMAINS: tuple[str, ...] = (
    # Watermarked stock providers & proxy CDNs
    "shutterstock.com",
    "gettyimages.",
    "alamy.com",
    "istockphoto.com",
    "dreamstime.com",
    "123rf.com",
    "depositphotos.com",
    "vecteezy.com",
    "stock.adobe.com",
    "ftcdn.net",
    "abposters.com",
    "europosters.",
    "plus.unsplash.com",
    "classroomclipart.com",
    "pond5.com",
    "bigstockphoto.com",
    "canstockphoto.com",
    "dissolve.com",
    "agefotostock.com",
    "superstock.com",
    "featurepics.com",
    "colourbox.com",
    "megapixl.com",
    "panthermedia.net",
    "storyblocks.com",
    "pngtree.com",
    "clipartkey.com",
    "clipart-library.com",
    # Walled-garden social media domains (return HTML login walls or HTTP 403 to video renderer)
    "facebook.com",
    "fbcdn.net",
    "fbsbx.com",
    "instagram.com",
    "cdninstagram.com",
    "tiktok.com",
    "tiktokcdn.com",
    "twitter.com",
    "x.com",
    "twimg.com",
    "threads.net",
)

# Shopping/poster marketplaces demoted behind real photographic/editorial sources
_LOW_PRIORITY_DOMAINS: tuple[str, ...] = (
    "ebay.",
    "etsy.",
    "amazon.",
    "walmart.",
    "aliexpress.",
    "temu.",
    "poshmark.",
    "mercari.",
    "wayfair.",
    "target.com",
    "alibaba.com",
    "pinterest.",
)

_NON_RASTER_EXTS = {".svg", ".tif", ".tiff", ".gif", ".pdf"}
_DEFAULT_IMAGE_UA = (
    "HKUST-NLP-Storyboarding/1.0 "
    "(https://github.com/CharlesDupouy/nlp-multiagent-storyboarding)"
)


class UnsplashRateLimitError(RuntimeError):
    """Raised when Unsplash rejects a query with HTTP 403/429 rate-limit status."""


class SerperSearchError(RuntimeError):
    """Raised when Serper.dev rejects a query or fails over the network."""


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
    serper_key = os.getenv("SERPER_API_KEY")
    unsplash_key = os.getenv("UNSPLASH_ACCESS_KEY")
    pexels_key = os.getenv("PEXELS_API_KEY")
    default_provider = "unsplash" if (not serper_key and unsplash_key) else "serper"
    provider = (os.getenv("IMAGE_API_PROVIDER") or default_provider).strip().lower()

    cleaned_primary = clean_stock_query(query)
    cleaned_fallback = clean_stock_query(fallback_query) if fallback_query else ""

    if provider == "mock":
        return _mock_search_images(query, max_results)

    if provider == "serper":
        if not serper_key:
            print(f"[TOOL WARNING] SERPER_API_KEY is not set; returning no images for {query!r}.")
            return []
        try:
            candidates = _serper_fetch_candidates(cleaned_primary, max_results, serper_key)
            if candidates:
                return candidates

            if cleaned_fallback and cleaned_fallback != cleaned_primary:
                candidates = _serper_fetch_candidates(cleaned_fallback, max_results, serper_key)
                if candidates:
                    return candidates
        except SerperSearchError as e:
            print(f"[TOOL WARNING] Serper unavailable — skipping image and fallback providers: {e}")
        except Exception as e:
            print(f"[TOOL WARNING] Serper candidate search error — skipping image and fallback providers: {e}")
        return []

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
        except UnsplashRateLimitError as e:
            print(f"[TOOL WARNING] Unsplash rate limit reached — skipping fallback retries: {e}")
        except Exception as e:
            print(f"[TOOL WARNING] Unsplash candidate search error: {e}")

    if provider == "pexels" and pexels_key:
        try:
            candidates = _pexels_search(cleaned_primary, max_results, pexels_key)
            if candidates:
                return candidates
        except Exception as e:
            print(f"[TOOL WARNING] Pexels candidate search error: {e}")

    # Fallback to mock images only for non-Serper providers if API fails, rate-limits, or key is missing
    return _mock_search_images(query, max_results)


def search_images(query: str, max_results: int = 3, per_page: int | None = None) -> list[dict[str, str]]:
    """Backwards-compatible image search returning candidate list."""
    limit = per_page if per_page is not None else max_results
    candidates = search_image_candidates(query=query, max_results=limit)
    return [
        {
            "query": query,
            "url": c.get("url", ""),
            "alt_text": c.get("alt_text", ""),
            "description": c.get("description", c.get("alt_text", "")),
        }
        for c in candidates
    ]


def _is_blocked_image_source(url: str, domain: str = "", source: str = "") -> bool:
    """Return True for watermarked stock sites/CDNs and walled-garden social hosts."""
    host = urlparse(url or "").netloc.lower()
    haystack = " ".join(part.lower() for part in (host, domain or "", source or "", url or ""))
    return any(marker in haystack for marker in _BLOCKED_IMAGE_DOMAINS)


def _is_low_priority_domain(url: str, domain: str = "") -> bool:
    """Return True for e-commerce/shopping domains that should sort after editorial photos."""
    host = (domain or urlparse(url or "").netloc).lower()
    return any(marker in host for marker in _LOW_PRIORITY_DOMAINS)


def _is_fetchable_image_url(url: str, timeout: float = 3.5) -> bool:
    """Quick streaming check ensuring URL responds 200 OK with raster image content."""
    headers = {"User-Agent": os.getenv("WIKIMEDIA_USER_AGENT") or _DEFAULT_IMAGE_UA}
    try:
        with requests.get(url, headers=headers, timeout=timeout, stream=True, allow_redirects=True) as resp:
            if resp.status_code != 200:
                return False
            ctype = (resp.headers.get("Content-Type") or "").lower()
            if ctype.startswith("image/"):
                return not any(bad in ctype for bad in ("svg", "tiff", "x-icon"))
            # Fallback when CDN sends generic octet-stream or omits Content-Type
            if not ctype or "octet-stream" in ctype:
                head = next(resp.iter_content(chunk_size=16), b"")
                return head.startswith((b"\xff\xd8\xff", b"\x89PNG", b"RIFF"))
            return False
    except Exception:
        return False


def _serper_fetch_candidates(
    q: str,
    max_results: int,
    api_key: str,
    verify_urls: bool = True,
) -> list[dict[str, Any]]:
    """Query Serper.dev Google Images and return filtered, fetchable candidates."""
    if not q:
        return []

    url = "https://google.serper.dev/images"
    headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}
    num_fetch = min(max(max_results * 5, 16), 20)
    payload = {"q": q, "num": num_fetch, "gl": "us", "hl": "en"}

    resp = requests.post(url, headers=headers, json=payload, timeout=12)
    if resp.status_code != 200:
        detail = (resp.text or "").strip().replace("\n", " ")[:160]
        raise SerperSearchError(f"HTTP {resp.status_code} for {q!r}: {detail}")

    raw_images = resp.json().get("images", [])
    seen_urls: set[str] = set()
    normal_bucket: list[dict[str, Any]] = []
    ecommerce_bucket: list[dict[str, Any]] = []
    tiny_bucket: list[dict[str, Any]] = []

    for item in raw_images:
        img_url = (item.get("imageUrl") or "").strip()
        if not img_url.lower().startswith(("http://", "https://")) or img_url in seen_urls:
            continue
        seen_urls.add(img_url)

        path = urlparse(img_url).path.lower()
        if any(path.endswith(ext) for ext in _NON_RASTER_EXTS):
            continue

        domain = (item.get("domain") or urlparse(img_url).netloc).lower().strip()
        source = (item.get("source") or domain).strip()
        if _is_blocked_image_source(img_url, domain, source):
            continue

        title = re.sub(r"\s+", " ", (item.get("title") or "").strip())
        desc = title or f"Image of {q}"
        alt_text = f"{desc} (Source: {source}, {domain})".strip() if source else desc
        entry = {
            "url": img_url,
            "description": desc,
            "alt_text": alt_text,
            "photographer": source or "Web",
        }

        img_width = item.get("imageWidth")
        if isinstance(img_width, (int, float)) and 0 < img_width < 320:
            tiny_bucket.append(entry)
        elif _is_low_priority_domain(img_url, domain):
            ecommerce_bucket.append(entry)
        else:
            normal_bucket.append(entry)

    ordered = normal_bucket + ecommerce_bucket + tiny_bucket
    if not ordered:
        return []

    selected: list[dict[str, Any]] = []
    if verify_urls:
        max_checks = min(len(ordered), max(max_results * 3, 6))
        for entry in ordered[:max_checks]:
            if _is_fetchable_image_url(entry["url"]):
                selected.append(entry)
                if len(selected) >= max_results:
                    break
    else:
        selected = ordered[:max_results]

    return [
        {"candidate_id": f"cand{idx + 1}", **entry}
        for idx, entry in enumerate(selected)
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
        detail = (resp.text or "").strip().replace("\n", " ")[:160]
        remaining = resp.headers.get("X-Ratelimit-Remaining")
        rate_info = f", remaining={remaining}" if remaining is not None else ""
        msg = f"HTTP {resp.status_code}{rate_info}: {detail}"
        if resp.status_code in (403, 429):
            raise UnsplashRateLimitError(msg)
        print(f"[TOOL WARNING] Unsplash search error for {q!r}: {msg}")
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
        detail = (resp.text or "").strip().replace("\n", " ")[:160]
        print(f"[TOOL WARNING] Pexels search error for {query!r}: HTTP {resp.status_code}: {detail}")
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
