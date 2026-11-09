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
from pathlib import Path
from typing import Any
from openai import OpenAI

from src.agents.base_agent import BaseAgent
from src.core.llm_client import chat_completion, extract_json
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
            max_tokens=4000,
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
            max_tokens=4000,
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
    pitch = state.selected_pitch
    director_notes = state.director_notes

    news_block = "\n".join(
        f"- {s.get('headline', '')}: {s.get('summary', '')}"
        for s in news.snippets[:6]
    ) or "No news snippets available."

    pitch_block = ""
    if pitch:
        beats_str = "\n".join(f"  * {b}" for b in pitch.get("core_beats", []))
        pitch_block = f"""\
COMMISSIONED PITCH (from Writers' Room)
======================================
Title: {pitch.get('title')}
Target Tone / Vibe: {pitch.get('target_tone')}
3-Second Hook: "{pitch.get('hook_3s')}"
Concept Premise: {pitch.get('concept')}
Core Beats:
{beats_str}

DIRECTORIAL NOTES (from Showrunner)
===================================
{director_notes or 'Execute with tight pacing and strong visual contrast.'}
"""

    return f"""\
{pitch_block}
USER PROFILE
============
Interests: {', '.join(profile.interests)}
Themes:    {', '.join(profile.themes)}
Preferred Style: {profile.preferred_style}
Summary:   {profile.raw_summary}

RETRIEVED FACTS & RECENT EVIDENCE
=================================
{news_block}

TASK
====
Draft a personalized short-video storyboard (target duration: {duration} seconds).
Faithfully bring the commissioned pitch to life following the Director's notes and integrating the retrieved facts.
Create an immediate 3-second hook, smooth escalating transitions, and distinct vivid scenes.
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
        data = extract_json(raw)
        if isinstance(data, dict):
            return data
        return {"overall_narrative": raw, "scenes": [], "title": "Parse Error"}
    except Exception:
        return {"overall_narrative": raw, "scenes": [], "title": "Parse Error"}


_PROMPTS_DIR = Path(__file__).resolve().parent.parent.parent / "prompts"


def _load_prompt(filename: str, default: str) -> str:
    path = _PROMPTS_DIR / filename
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    return default.strip()


_DRAFT_SYSTEM_PROMPT_DEFAULT = """\
You are a premier short-form video screenwriter and visual storyteller.
Your superpower is turning creative concepts and real-world facts into gripping, fast-paced 60-second video storyboards.

Core Philosophy:
- Faithfully execute the commissioned story pitch and directorial guidance.
- Adapt your voice and energy to match the commissioned tone (e.g., witty & comedic, mindbending curiosity, forensic mystery, chill craft, or high-stakes esports).
- Ground the script using the retrieved real-world facts and evidence—never rely on hollow generic fluff.
- Deliver an instant 3-second hook that halts the scroll.
- Ensure narrative progression: each scene must reveal new information, escalate stakes, or provide visual contrast.
- Subtitles: Punchy, memorable on-screen text (max 12 words per cut, formatted for mobile).
- Audio cues: Charismatic voiceover lines, dynamic sound effects, or musical beats.
- Language: Write the script in the primary language of the user's profile/history.

CRITICAL NEGATIVE CONSTRAINTS (NO PLACEHOLDERS):
- NEVER write placeholder, evasive, or meta descriptions in subtitles or audio cues.
  - FORBIDDEN: "A character says an innocent line...", "Zoom on the key quote...", "Explains the double meaning..."
  - REQUIRED: You MUST invent or quote the EXACT words verbatim (e.g., "Voix off: 'Maman, pourquoi le monsieur a une carotte dans la poche ?'").
- Every scene's `subtitle` and `audio_cue` must contain the actual spoken text, ready to be read by TTS or an actor.

Return ONLY valid JSON with this structure:
{
  "title": "Engaging video title",
  "overall_narrative": "2-3 sentence overview of the story arc and concept",
  "scenes": [
    {
      "scene_id": 1,
      "description": "Visual scene description: authentic action, dynamic camera movement, distinct visual elements",
      "subtitle": "Punchy on-screen caption (max 12 words)",
      "audio_cue": "Voiceover line or music/SFX cue",
      "duration_seconds": 10
    }
  ]
}

Total scenes: 4 to 6 scenes summing strictly to the target duration.
"""

_DRAFT_SYSTEM_PROMPT = _load_prompt("scriptwriter_draft.txt", _DRAFT_SYSTEM_PROMPT_DEFAULT)


_REFINE_SYSTEM_PROMPT_DEFAULT = """\
You are an expert video director and storyteller refining an entertaining short video against real-world retrieved imagery (img1, img2, etc.).

Refine the scenes to anchor your entertaining narrative in the real visual evidence:
1. Ground the story into the real-world objects, places, or people shown in the image assets.
2. Maintain high entertainment value: witty, fast-paced, and captivating.
3. In "image_ref", use ONLY the symbolic identifiers provided (e.g. "img1", "img2") or null. NEVER write a full URL.
4. Language: Keep the same language as the draft.

CRITICAL NEGATIVE CONSTRAINTS (NO PLACEHOLDERS):
- NEVER write placeholder, evasive, or meta descriptions in subtitles or audio cues.
  - FORBIDDEN: "A character says an innocent line...", "Zoom on the key quote...", "Explains the double meaning..."
  - REQUIRED: You MUST invent or quote the EXACT words verbatim (e.g., "Voix off: 'Maman, pourquoi le monsieur a une carotte dans la poche ?'").
- Every scene's `subtitle` and `audio_cue` must contain the actual spoken text, ready to be read by TTS or an actor.

Return ONLY valid JSON with this structure:
{
  "title": "Refined title",
  "overall_narrative": "Refined story overview",
  "scenes": [
    {
      "scene_id": 1,
      "description": "Visual scene description aligned with the real-world image asset",
      "subtitle": "Punchy on-screen caption",
      "audio_cue": "Charismatic voiceover line or sound effect",
      "image_ref": "img1",
      "duration_seconds": 10
    }
  ]
}
"""

_REFINE_SYSTEM_PROMPT = _load_prompt("scriptwriter_refine.txt", _REFINE_SYSTEM_PROMPT_DEFAULT)
