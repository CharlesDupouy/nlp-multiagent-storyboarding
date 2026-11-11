"""
youtube_tool.py — Tool for retrieving subtitles, transcripts, and metadata
from YouTube videos WITHOUT requiring any Google Cloud API key or OAuth.

Uses:
  - youtube-transcript-api for instant spoken captions/transcripts.
  - YouTube oEmbed + web player response for metadata, author, and description.
"""

from __future__ import annotations

import re
import requests
from typing import Any

try:
    from youtube_transcript_api import YouTubeTranscriptApi
    _TRANSCRIPT_AVAILABLE = True
except ImportError:
    _TRANSCRIPT_AVAILABLE = False


def extract_video_id(url_or_id: str) -> str:
    """Extract canonical YouTube 11-char video ID from a URL or raw ID."""
    match = re.search(r"(?:shorts/|v=|youtu\.be/)([a-zA-Z0-9_-]{11})", url_or_id)
    if match:
        return match.group(1)
    # Check if raw 11-char ID
    cleaned = url_or_id.strip()
    if len(cleaned) == 11 and re.match(r"^[a-zA-Z0-9_-]+$", cleaned):
        return cleaned
    return cleaned


_FR_HINT_WORDS = {
    "le", "la", "les", "un", "une", "des", "du", "de", "et", "est", "pour",
    "avec", "qui", "que", "ce", "cette", "dans", "sur", "pas", "plus", "pourquoi",
    "comment", "quand", "tout", "tous", "mon", "ton", "son", "mes", "tes", "ses",
    "faire", "fait", "humour", "drole", "drôle", "vrai", "vraiment", "jamais",
}
_EN_HINT_WORDS = {
    "the", "a", "an", "and", "is", "are", "was", "were", "this", "that",
    "with", "for", "to", "in", "on", "of", "you", "your", "it", "they",
    "how", "why", "what", "when", "who", "secret", "never", "always",
}


def detect_text_language(text: str, default: str = "en") -> str:
    """Detect whether a short title/history block is predominantly French ('fr') or English ('en')."""
    if not text:
        return default
    lowered = text.lower()
    words = set(re.findall(r"\b[a-z\u00c0-\u017f]+\b", lowered))
    fr_hits = len(words & _FR_HINT_WORDS) + len(re.findall(r"[éèêëàâùûîïôçœ]", lowered))
    en_hits = len(words & _EN_HINT_WORDS)
    if fr_hits > en_hits:
        return "fr"
    if en_hits > fr_hits:
        return "en"
    return default


def get_video_transcript(
    url_or_id: str,
    max_chars: int = 1500,
    languages: list[str] | tuple[str, ...] | None = None,
) -> str | None:
    """
    Fetch the spoken transcript/subtitles for a YouTube video.
    Tries preferred `languages` (e.g. ['fr', 'en'] for French titles) before falling back
    to any available transcript track, returning None if no captions exist.
    """
    if not _TRANSCRIPT_AVAILABLE:
        return None

    video_id = extract_video_id(url_or_id)
    pref_langs = list(languages) if languages else ["en", "fr", "es", "de", "it"]
    try:
        api = YouTubeTranscriptApi()
        records = None
        if hasattr(api, "fetch"):
            try:
                records = api.fetch(video_id, languages=pref_langs)
            except Exception:
                if hasattr(api, "list"):
                    tlist = api.list(video_id)
                    first_track = next(iter(tlist), None)
                    if first_track is not None:
                        records = first_track.fetch()
        else:
            records = YouTubeTranscriptApi.get_transcript(video_id, languages=pref_langs)

        if not records:
            return None
        lines = [r.text if hasattr(r, "text") else r.get("text", "") for r in records]
        full_text = " ".join(lines).strip()
        # Clean double spaces and linebreaks
        full_text = re.sub(r"\s+", " ", full_text)
        return full_text[:max_chars] if full_text else None
    except Exception:
        return None


def get_video_metadata(url_or_id: str) -> dict[str, Any]:
    """
    Fetch public metadata for a YouTube video: title, author/channel, and description.
    Zero API key required.
    """
    video_id = extract_video_id(url_or_id)
    canonical_url = f"https://www.youtube.com/shorts/{video_id}"

    metadata: dict[str, Any] = {
        "video_id": video_id,
        "url": canonical_url,
        "title": "",
        "author": "",
        "description": "",
    }

    # 1. Fetch from oEmbed
    try:
        r = requests.get(
            f"https://www.youtube.com/oembed?url={canonical_url}&format=json",
            timeout=4,
        )
        if r.status_code == 200:
            data = r.json()
            metadata["title"] = data.get("title", "")
            metadata["author"] = data.get("author_name", "")
    except Exception:
        pass

    return metadata


def get_enriched_video_summary(url_or_id: str) -> str:
    """Convenience helper returning a formatted summary with transcript if available."""
    video_id = extract_video_id(url_or_id)
    meta = get_video_metadata(video_id)
    transcript = get_video_transcript(video_id)

    parts = [f"Title: {meta.get('title') or video_id}", f"Channel: {meta.get('author') or 'Unknown'}"]
    if transcript:
        parts.append(f"Spoken Dialogue: \"{transcript[:500]}...\"")
    else:
        parts.append("Spoken Dialogue: (No subtitles available)")

    return " | ".join(parts)
