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


def get_video_transcript(url_or_id: str, max_chars: int = 1500) -> str | None:
    """
    Fetch the spoken transcript/subtitles for a YouTube video.
    Returns cleaned dialogue text, or None if no captions are available.
    """
    if not _TRANSCRIPT_AVAILABLE:
        return None

    video_id = extract_video_id(url_or_id)
    try:
        api = YouTubeTranscriptApi()
        # In youtube-transcript-api v1.2+, api.fetch() retrieves the transcript
        if hasattr(api, "fetch"):
            records = api.fetch(video_id)
        else:
            records = YouTubeTranscriptApi.get_transcript(video_id)

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
