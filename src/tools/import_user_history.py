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
import json
import re
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest and validate a participant's YouTube Shorts history."
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
    return parser.parse_args()


def sanitize_username(name: str) -> str:
    """Sanitize participant identifier for safe filenames."""
    cleaned = re.sub(r"[^\w\-_]", "_", name.strip().lower())
    return cleaned or "anonymous_user"


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
    except ValueError as e:
        print(f"❌ Validation failed: {e}", file=sys.stderr)
        sys.exit(1)

    user_slug = sanitize_username(args.user)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{user_slug}.json"

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(videos, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Ingestion successful!")
    print(f"   Participant: {user_slug}")
    print(f"   Videos ingested: {len(videos)}")
    print(f"   Saved to: {out_file}")
    print(f"\n🚀 To generate a storyboard for this participant, run:")
    print(f"   python main.py --history {out_file}\n")


if __name__ == "__main__":
    main()
