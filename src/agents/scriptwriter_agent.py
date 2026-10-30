"""
ScriptwriterAgent — Agent 4 (final agent) in the pipeline.

Responsibility:
  - Reads the complete SharedState (user profile, news snippets, images).
  - Uses an LLM to compose a full, structured video storyboard tailored
    to the user's profile and enriched by retrieved context.
  - The storyboard is the final deliverable of the multi-agent pipeline.

Output written to state:
  - state.storyboard  (Storyboard dataclass)
"""

from __future__ import annotations

import json

from openai import OpenAI

from src.agents.base_agent import BaseAgent
from src.core.llm_client import chat_completion
from src.core.state import SharedState, Storyboard


class ScriptwriterAgent(BaseAgent):
    """
    Synthesises all retrieved context into a personalized video storyboard.
    This agent uses the largest / best model since it does the most
    complex reasoning.
    """

    def __init__(self, model: str, client: OpenAI, target_duration: int = 60) -> None:
        super().__init__(name="ScriptwriterAgent", model=model, client=client)
        self.target_duration = target_duration

    def run(self, state: SharedState) -> SharedState:
        self._log(state, "Starting storyboard generation.")

        user_prompt = _build_scriptwriter_prompt(state, self.target_duration)
        system_prompt = _SCRIPTWRITER_SYSTEM_PROMPT

        self._log(state, f"Calling LLM ({self.model}) to write storyboard.")
        raw_response = chat_completion(
            client=self.client,
            model=self.model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.8,   # higher temperature for creative output
            max_tokens=3000,
        )

        # TODO: add retry / fallback if JSON is malformed
        storyboard_data = _parse_storyboard_response(raw_response)
        state.storyboard = Storyboard(
            title=storyboard_data.get("title", "Untitled"),
            target_duration_seconds=self.target_duration,
            scenes=storyboard_data.get("scenes", []),
            overall_narrative=storyboard_data.get("overall_narrative", ""),
        )

        self._log(
            state,
            f"Storyboard '{state.storyboard.title}' written with "
            f"{len(state.storyboard.scenes)} scenes.",
        )
        return state


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_scriptwriter_prompt(state: SharedState, duration: int) -> str:
    profile = state.user_profile
    news = state.news_context
    images = state.image_context

    # Format news snippets
    news_block = "\n".join(
        f"- {s.get('headline','')}: {s.get('summary','')}"
        for s in news.snippets[:5]
    ) or "No news snippets available."

    # Format available images
    images_block = "\n".join(
        f"- Query: '{img.get('query','')}' | URL: {img.get('url','')} | Alt: {img.get('alt_text','')}"
        for img in images.images
    ) or "No images available."

    return f"""\
USER PROFILE
============
Interests: {', '.join(profile.interests)}
Themes:    {', '.join(profile.themes)}
Style:     {profile.preferred_style}
Summary:   {profile.raw_summary}

RECENT NEWS / WEB CONTEXT
==========================
{news_block}

AVAILABLE IMAGES
================
{images_block}

TASK
====
Write a personalized short-video storyboard (target duration: {duration} seconds).
The video must feel tailor-made for this specific user.
"""


def _parse_storyboard_response(raw: str) -> dict:
    try:
        cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return {"overall_narrative": raw, "scenes": [], "title": "Parse Error"}


_SCRIPTWRITER_SYSTEM_PROMPT = """\
You are a creative video scriptwriter specializing in short-form content.
Given a user profile, recent news context, and available images, produce a
personalized short-video storyboard.

Return ONLY valid JSON with this structure:
{
  "title": "Catchy video title",
  "overall_narrative": "2-3 sentence arc of the video",
  "scenes": [
    {
      "scene_id": 1,
      "description": "Visual description of what appears on screen",
      "subtitle": "On-screen text or caption (max 15 words)",
      "audio_cue": "Voiceover text or music/sound note",
      "image_url": "URL from available images, or null",
      "duration_seconds": 10
    }
  ]
}

Guidelines:
- Aim for 4-7 scenes that sum to the target duration.
- Each subtitle must be punchy and readable in its allotted time.
- Ground the narrative in the user's actual interests — avoid generic content.
- Use available image URLs where they fit the scene.
"""
