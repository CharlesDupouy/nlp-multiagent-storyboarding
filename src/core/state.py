"""
SharedState — the single in-memory blackboard passed through the pipeline.

Every agent reads from and writes to this object. The Orchestrator
initializes it with raw user input; each agent enriches it with its output.
At the end of the pipeline, SharedState contains everything needed by the
ScriptwriterAgent to produce the final storyboard.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class UserHistory:
    """Raw user input: a list of watched video titles and their URLs."""
    videos: list[dict[str, str]] = field(default_factory=list)
    # Each entry: {"title": "...", "url": "https://..."}


@dataclass
class UserProfile:
    """
    Output of ProfilerAgent.
    A structured summary of the user's inferred interests,
    preferred themes, and content style preferences.
    """
    interests: list[str] = field(default_factory=list)
    themes: list[str] = field(default_factory=list)
    preferred_style: str = ""          # e.g. "educational", "entertaining", "news"
    raw_summary: str = ""              # free-text profile written by the LLM


@dataclass
class NewsContext:
    """
    Output of NewsRetrieverAgent.
    A list of recent news snippets relevant to the user's profile.
    """
    snippets: list[dict[str, str]] = field(default_factory=list)
    # Each entry: {"headline": "...", "summary": "...", "url": "..."}


@dataclass
class ImageContext:
    """
    Output of ImageSearchAgent.
    A list of image results (URL + description) relevant to the storyboard theme.
    No vision model is used; images are selected by keyword relevance only.
    """
    images: list[dict[str, str]] = field(default_factory=list)
    # Each entry: {"query": "...", "url": "...", "alt_text": "..."}


@dataclass
class Storyboard:
    """
    Output of ScriptwriterAgent — the final deliverable.
    A structured video storyboard with scenes, subtitles, and audio cues.
    """
    title: str = ""
    target_duration_seconds: int = 60
    scenes: list[dict[str, Any]] = field(default_factory=list)
    # Each scene: {
    #   "scene_id": int,
    #   "description": str,       # visual description
    #   "subtitle": str,          # on-screen text / caption
    #   "audio_cue": str,         # voiceover or music note
    #   "image_url": str | None,  # sourced from ImageContext
    #   "duration_seconds": int,
    # }
    overall_narrative: str = ""


@dataclass
class SharedState:
    """
    The central blackboard shared across all agents in the pipeline.
    Initialized by the Orchestrator; progressively filled by each agent.
    """
    # --- INPUT ---
    user_history: UserHistory = field(default_factory=UserHistory)

    # --- INTERMEDIATE OUTPUTS (filled by agents in order) ---
    user_profile: UserProfile = field(default_factory=UserProfile)
    news_context: NewsContext = field(default_factory=NewsContext)
    image_context: ImageContext = field(default_factory=ImageContext)

    # --- FINAL OUTPUT ---
    storyboard: Storyboard = field(default_factory=Storyboard)

    # --- METADATA ---
    run_id: str = ""
    pipeline_log: list[str] = field(default_factory=list)  # human-readable trace

    def log(self, message: str) -> None:
        """Append a message to the pipeline trace log."""
        self.pipeline_log.append(message)
        print(f"[PIPELINE] {message}")

    def to_dict(self) -> dict:
        """Serialize the full state to a plain dict (for saving to JSON)."""
        import dataclasses
        return dataclasses.asdict(self)

    def save(self, path: str) -> None:
        """Save the full state as a JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
