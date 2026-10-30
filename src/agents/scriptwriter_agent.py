"""
ScriptwriterAgent — Multi-pass creative scriptwriting agent.

Responsibilities:
  1. Draft Pass:
     - Consumes UserProfile and NewsContext.
     - Generates initial narrative arc and scene structure (descriptions, subtitles, audio cues).
  2. Refinement Pass:
     - Runs AFTER ImageSearchAgent.
     - Reviews draft scenes against retrieved visual assets (referenced symbolically as img1, img2...).
     - Assigns 'image_ref' to scenes and refines scene descriptions/subtitles to ground them in the visuals.
     - Never touches or hallucinates raw URLs. The orchestrator programmatically maps
       'image_ref' -> 'image_url'.

Output written to state:
  - state.storyboard  (Storyboard dataclass)
"""

from __future__ import annotations

import json
from typing import Any
from openai import OpenAI

from src.agents.base_agent import BaseAgent
from src.core.llm_client import chat_completion
from src.core.state import SharedState, Storyboard


class ScriptwriterAgent(BaseAgent):
    """
    Two-pass storyboard writer:
    - Drafts scenes from user profile and news.
    - Refines scenes based on retrieved image assets using symbolic IDs (img1, img2...).
    """

    def __init__(self, model: str, client: OpenAI, target_duration: int = 60) -> None:
        super().__init__(name="ScriptwriterAgent", model=model, client=client)
        self.target_duration = target_duration

    def draft(self, state: SharedState) -> SharedState:
        """Pass 1: Generate initial draft storyboard without image constraints."""
        self._log(state, "Drafting initial storyboard narrative and scenes.")

        user_prompt = _build_draft_prompt(state, self.target_duration)
        raw_response = chat_completion(
            client=self.client,
            model=self.model,
            system_prompt=_DRAFT_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=0.75,
            max_tokens=2500,
        )

        data = _parse_storyboard_response(raw_response)
        state.storyboard = Storyboard(
            title=data.get("title", "Untitled Draft"),
            target_duration_seconds=self.target_duration,
            scenes=data.get("scenes", []),
            overall_narrative=data.get("overall_narrative", ""),
        )
        self._log(
            state,
            f"Draft created: '{state.storyboard.title}' with {len(state.storyboard.scenes)} scenes.",
        )
        return state

    def refine(self, state: SharedState) -> SharedState:
        """Pass 2: Ground and refine draft scenes using available visual image assets."""
        self._log(state, "Refining storyboard against retrieved visual image assets.")

        if not state.storyboard.scenes:
            self._log(state, "No draft scenes found to refine. Running draft first.")
            return self.draft(state)

        user_prompt = _build_refine_prompt(state)
        raw_response = chat_completion(
            client=self.client,
            model=self.model,
            system_prompt=_REFINE_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=0.6,
            max_tokens=2500,
        )

        refined_data = _parse_storyboard_response(raw_response)
        if refined_data.get("scenes"):
            state.storyboard.scenes = refined_data["scenes"]
            if refined_data.get("title"):
                state.storyboard.title = refined_data["title"]
            if refined_data.get("overall_narrative"):
                state.storyboard.overall_narrative = refined_data["overall_narrative"]

        # Programmatically link image_ref -> real image_url
        state.storyboard.resolve_image_urls(state.image_context)
        self._log(
            state,
            f"Storyboard refined: '{state.storyboard.title}' ({len(state.storyboard.scenes)} scenes aligned with imagery).",
        )
        return state

    def run(self, state: SharedState) -> SharedState:
        """
        Adaptive run:
        - If no scenes exist: drafts initial storyboard.
        - If scenes already exist and images are available: refines storyboard.
        """
        if state.storyboard.scenes and state.image_context.images:
            return self.refine(state)
        return self.draft(state)


# ---------------------------------------------------------------------------
# Prompt Builders & Parsing Helpers
# ---------------------------------------------------------------------------

def _build_draft_prompt(state: SharedState, duration: int) -> str:
    profile = state.user_profile
    news = state.news_context

    news_block = "\n".join(
        f"- {s.get('headline', '')}: {s.get('summary', '')}"
        for s in news.snippets[:6]
    ) or "No news snippets available."

    return f"""\
USER PROFILE
============
Interests: {', '.join(profile.interests)}
Themes:    {', '.join(profile.themes)}
Style:     {profile.preferred_style}
Summary:   {profile.raw_summary}

RECENT NEWS / WEB TRENDS
========================
{news_block}

TASK
====
Draft a personalized short-video storyboard (target duration: {duration} seconds).
Create an engaging hook, smooth transitions, and distinct scenes.
"""


def _build_refine_prompt(state: SharedState) -> str:
    # Format available images as a symbolic reference catalog
    image_lines = []
    for img in state.image_context.images:
        img_id = img.get("image_id", "")
        desc = img.get("alt_text", "")
        query = img.get("query", "")
        target = img.get("target_scene_id")
        target_str = f" (matched for Scene {target})" if target else ""
        image_lines.append(f"- [{img_id}]: \"{desc}\" [Search query: '{query}']{target_str}")

    image_catalog = "\n".join(image_lines) or "No image assets available."

    scenes_json = json.dumps(state.storyboard.scenes, indent=2)

    return f"""\
CURRENT STORYBOARD DRAFT
========================
Title: {state.storyboard.title}
Overall Narrative: {state.storyboard.overall_narrative}

Draft Scenes:
{scenes_json}

AVAILABLE VISUAL ASSETS CATALOG
===============================
{image_catalog}

TASK
====
Refine the storyboard scenes to visually ground them with the available assets:
1. For each scene, assign the best fitting asset ID (e.g. "image_ref": "img1") or null if none fits.
2. Slightly tweak the visual description and subtitle if needed to match what is depicted in the chosen image asset.
3. Keep scene pacing and target duration consistent.
"""


def _parse_storyboard_response(raw: str) -> dict:
    try:
        cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return {"overall_narrative": raw, "scenes": [], "title": "Parse Error"}


_DRAFT_SYSTEM_PROMPT = """\
You are an expert short-form video creator (YouTube Shorts / TikTok).
Draft a compelling, personalized video storyboard based on the user's profile and recent news trends.

Return ONLY valid JSON with this structure:
{
  "title": "Catchy video title",
  "overall_narrative": "2-3 sentence summary of the video arc",
  "scenes": [
    {
      "scene_id": 1,
      "description": "Visual description of what appears on screen",
      "subtitle": "On-screen text or caption (punchy, max 12 words)",
      "audio_cue": "Voiceover line or sound effect",
      "duration_seconds": 10
    }
  ]
}

Guidelines:
- 4 to 7 scenes total, summing to the target duration.
- Each scene must have clear visual descriptions and punchy subtitles.
- Ground the video in the user's specific tastes.
"""


_REFINE_SYSTEM_PROMPT = """\
You are a senior video editor polishing a short-form video storyboard.
You have been provided with draft scenes and a catalog of available stock imagery (img1, img2, etc.).

Refine the storyboard scenes by grounding them into the available visual assets.
Return ONLY valid JSON with this structure:
{
  "title": "Refined video title",
  "overall_narrative": "Updated narrative summary",
  "scenes": [
    {
      "scene_id": 1,
      "description": "Refined visual description matching the chosen visual asset",
      "subtitle": "On-screen text or caption",
      "audio_cue": "Voiceover line or sound effect",
      "image_ref": "img1",
      "duration_seconds": 10
    }
  ]
}

CRITICAL RULES:
- In "image_ref", use ONLY the symbolic identifiers provided (e.g. "img1", "img2") or null. NEVER write a full URL.
- Align scene descriptions with the image asset selected so the video visually makes sense.
"""
