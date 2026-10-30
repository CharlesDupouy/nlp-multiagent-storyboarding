"""
main.py — Top-level entry point for the multi-agent pipeline.

Usage:
    python main.py --config configs/config.yaml --history data/user_histories/example.json

For the single-agent baseline:
    python main.py --config configs/config.yaml --history data/user_histories/example.json --baseline
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-Agent Storyboarding Pipeline (COMP 4221)"
    )
    parser.add_argument(
        "--config", type=str, default="configs/config.yaml",
        help="Path to the YAML config file."
    )
    parser.add_argument(
        "--history", type=str, required=True,
        help="Path to a JSON file containing the user's watch history."
    )
    parser.add_argument(
        "--baseline", action="store_true",
        help="Run the single-agent baseline instead of the full pipeline."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Load user history
    history_path = Path(args.history)
    if not history_path.exists():
        print(f"Error: history file not found: {history_path}", file=sys.stderr)
        sys.exit(1)

    with open(history_path, "r", encoding="utf-8") as f:
        videos = json.load(f)  # expects list of {"title": str, "url": str}

    if args.baseline:
        from baseline.single_agent_baseline import SingleAgentBaseline
        runner = SingleAgentBaseline.from_config(args.config)
        print("Running single-agent baseline...")
    else:
        from src.orchestrator import Orchestrator
        runner = Orchestrator.from_config(args.config)
        print("Running multi-agent pipeline...")

    state = runner.run(videos=videos)

    print(f"\n✅ Done! Storyboard: '{state.storyboard.title}'")
    print(f"   Scenes: {len(state.storyboard.scenes)}")
    print(f"   run_id: {state.run_id}")


if __name__ == "__main__":
    main()
