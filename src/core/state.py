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
    A list of image results (ID, URL, query, description) relevant to storyboard scenes.
    Each image has a discrete ID (e.g. 'img1', 'img2') so the scriptwriter
    can reference visuals symbolically without hallucinating raw URLs.
    """
    images: list[dict[str, Any]] = field(default_factory=list)
    # Each entry: {"image_id": "img1", "query": "...", "url": "...", "alt_text": "..."}

    def get_image_map(self) -> dict[str, dict[str, Any]]:
        """Return a mapping from image_id -> image entry dict."""
        return {
            img.get("image_id", f"img{i+1}"): img
            for i, img in enumerate(self.images)
        }


@dataclass
class Storyboard:
    """
    Output of ScriptwriterAgent — the final deliverable.
    A structured video storyboard with scenes, subtitles, audio cues,
    and groundable image references.
    """
    title: str = ""
    target_duration_seconds: int = 60
    scenes: list[dict[str, Any]] = field(default_factory=list)
    # Each scene: {
    #   "scene_id": int,
    #   "description": str,       # visual description
    #   "subtitle": str,          # on-screen text / caption
    #   "audio_cue": str,         # voiceover or music note
    #   "image_ref": str | None,  # symbolic identifier, e.g. "img1"
    #   "image_url": str | None,  # programmatically resolved from image_ref
    #   "duration_seconds": int,
    # }
    overall_narrative: str = ""

    def resolve_image_urls(self, image_context: ImageContext) -> None:
        """
        Programmatically resolve image_ref ('img1', 'img2') to real image_url.
        Ensures the LLM never has to handle or hallucinate raw URLs.
        """
        img_map = image_context.get_image_map()
        for scene in self.scenes:
            ref = scene.get("image_ref")
            if ref and ref in img_map:
                scene["image_url"] = img_map[ref].get("url")
            elif not scene.get("image_url"):
                scene["image_url"] = None


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
