"""
import_user_history.py — Ingestion and validation helper for participant data.

Usage:
    python3 src/tools/import_user_history.py --input /path/to/downloaded_shorts.json --user alice

This script:
  1. Validates that the input JSON has the required schema ([{"title": "...", "url": "..."}]).
  2. Cleans whitespace and deduplicates URLs.
  3. Saves the file to data/user_histories/<user>.json, ready for main.py.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import random
import re
import sys
from pathlib import Path
import requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest, validate, and enrich a participant's YouTube Shorts history."
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="Path to the JSON file exported by the participant.",
    )
    parser.add_argument(
        "--user",
        "-u",
        type=str,
        required=True,
        help="Identifier or pseudonym for the participant (e.g. 'alice', 'user_01').",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/user_histories",
        help="Directory to save the cleaned history JSON file.",
    )
    parser.add_argument(
        "--no-enrich",
        action="store_true",
        help="Skip automatic title resolution via YouTube public oEmbed API.",
    )
    parser.add_argument(
        "--force-enrich",
        action="store_true",
        help="Force re-fetching all titles from YouTube oEmbed API even if titles already exist.",
    )
    parser.add_argument(
        "--cap",
        "--max-videos",
        type=int,
        default=None,
        dest="cap",
        help="Maximum number of videos to retain (e.g. 90 for 3x30 split). Applied before splitting.",
    )
    parser.add_argument(
        "--split",
        "-s",
        type=int,
        default=None,
        help="Number of parts to split the history into (e.g. 2, 3) for generating multiple distinct video pairs.",
    )
    parser.add_argument(
        "--split-mode",
        choices=["random", "sequential"],
        default="random",
        help="Split method: 'random' (unbiased shuffle) or 'sequential' (chronological chunks). Default: random.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible shuffling when split-mode is random (default: 42).",
    )
    return parser.parse_args()


def sanitize_username(name: str) -> str:
    """Sanitize participant identifier for safe filenames."""
    cleaned = re.sub(r"[^\w\-_]", "_", name.strip().lower())
    return cleaned or "anonymous_user"


GENERIC_DATE_HEADERS = {
    "today", "yesterday", "tomorrow",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "aujourd'hui", "hier", "demain",
    "lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche",
}


def is_generic_title(title: str) -> bool:
    """Identify placeholder titles or date headers that need oEmbed enrichment."""
    if not title:
        return True
    t = title.strip().lower()
    if "youtube short" in t:
        return True
    if t in GENERIC_DATE_HEADERS:
        return True
    if re.match(r"^(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|\d+[\s\-/])", t):
        return True
    return False


def enrich_single_video(video: dict[str, str], force: bool = False) -> dict[str, str]:
    """Fetch accurate video title and channel name via YouTube's public oEmbed API."""
    url = video.get("url", "")
    title = video.get("title", "")

    # Only fetch if forced or title is generic/missing
    if force or is_generic_title(title):
        try:
            resp = requests.get(
                f"https://www.youtube.com/oembed?url={url}&format=json",
                timeout=4,
            )
            if resp.status_code == 200:
                data = resp.json()
                t = data.get("title")
                author = data.get("author_name")
                if t:
                    video["title"] = f"{t} (by {author})" if author else t
        except Exception:
            pass
    return video


def enrich_titles_in_parallel(videos: list[dict[str, str]], force: bool = False, max_workers: int = 15) -> list[dict[str, str]]:
    """Enrich all video titles concurrently."""
    needs_enrichment = sum(
        1 for v in videos
        if force or is_generic_title(v.get("title", ""))
    )
    if needs_enrichment == 0:
        return videos

    print(f"🔄 Resolving real titles for {needs_enrichment} shorts via YouTube public API...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        enriched = list(executor.map(lambda v: enrich_single_video(v, force=force), videos))

    resolved = sum(
        1 for v in enriched
        if not is_generic_title(v.get("title", ""))
    )
    print(f"✅ Resolved {resolved}/{len(videos)} real video titles!")
    return enriched


def validate_and_clean_history(raw_data: list) -> list[dict[str, str]]:
    if not isinstance(raw_data, list):
        raise ValueError("Invalid format: JSON root must be an array of video items.")

    cleaned_videos = []
    seen_urls = set()

    for idx, item in enumerate(raw_data):
        if not isinstance(item, dict):
            continue

        title = str(item.get("title", "")).strip()
        url = str(item.get("url", "")).strip()

        if not url:
            continue

        # Standardize shorts URL
        match = re.search(r"shorts/([a-zA-Z0-9_-]+)", url)
        if match:
            url = f"https://www.youtube.com/shorts/{match.group(1)}"

        if url in seen_urls:
            continue

        if not title:
            title = f"YouTube Short #{idx + 1}"

        seen_urls.add(url)
        cleaned_videos.append({"title": title, "url": url})

    if not cleaned_videos:
        raise ValueError("No valid video entries found with a valid URL.")

    return cleaned_videos


def split_history(
    videos: list[dict[str, str]],
    parts: int,
    mode: str = "random",
    seed: int = 42,
) -> list[list[dict[str, str]]]:
    """
    Split a list of videos into X balanced subsets.
    - 'random': Shuffles the list with a deterministic seed for unbiased cross-sectional sampling.
    - 'sequential': Splits the list in original chronological order to isolate browsing sessions.
    """
    if parts <= 1:
        return [list(videos)]

    parts = min(parts, len(videos))
    v_list = list(videos)

    if mode == "random":
        rng = random.Random(seed)
        rng.shuffle(v_list)

    slices: list[list[dict[str, str]]] = []
    n = len(v_list)
    for i in range(parts):
        start = (i * n) // parts
        end = ((i + 1) * n) // parts
        slices.append(v_list[start:end])
    return slices


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ Error: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    try:
        with open(input_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
    except Exception as e:
        print(f"❌ Error reading JSON: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        videos = validate_and_clean_history(raw_data)
        if not args.no_enrich:
            videos = enrich_titles_in_parallel(videos, force=args.force_enrich)
    except ValueError as e:
        print(f"❌ Validation failed: {e}", file=sys.stderr)
        sys.exit(1)

    if args.cap and len(videos) > args.cap:
        print(f"✂️  Capping history from {len(videos)} to {args.cap} most recent videos.")
        videos = videos[:args.cap]

    user_slug = sanitize_username(args.user)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{user_slug}.json"

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(videos, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Ingestion successful!")
    print(f"   Participant: {user_slug}")
    print(f"   Videos ingested: {len(videos)}")
    print(f"   Full history saved to: {out_file}")

    if args.split and args.split > 1:
        slices = split_history(videos, parts=args.split, mode=args.split_mode, seed=args.seed)
        print(f"\n✂️  Split into {len(slices)} {args.split_mode} parts (seed={args.seed}):")
        for idx, part in enumerate(slices, 1):
            slice_path = out_dir / f"{user_slug}_part{idx}.json"
            with open(slice_path, "w", encoding="utf-8") as f:
                json.dump(part, f, indent=2, ensure_ascii=False)
            print(f"   - Part {idx}: {len(part)} videos -> {slice_path}")

        print(f"\n🚀 Ready for paired multi-agent and baseline generation:")
        for idx in range(1, len(slices) + 1):
            part_path = out_dir / f"{user_slug}_part{idx}.json"
            print(f"   # Pair {idx}:")
            print(f"   python main.py --history {part_path}")
            print(f"   python main.py --history {part_path} --baseline")
    else:
        print(f"\n🚀 To generate a storyboard for this participant, run:")
        print(f"   python main.py --history {out_file}\n")


if __name__ == "__main__":
    main()
