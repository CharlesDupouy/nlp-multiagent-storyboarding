"""
ScriptwriterAgent — Multi-pass creative scriptwriting agent.

Responsibilities:
  1. Draft Pass:
     - Consumes UserProfile and NewsContext.
     - Generates initial narrative arc and scene structure (visual descriptions, pure voiceover).
  2. Refinement Pass:
     - Runs AFTER ImageSearchAgent.
     - Reviews draft scenes against retrieved visual assets (referenced symbolically as img1, img2...).
     - Assigns 'image_ref' to scenes and refines scene descriptions and voiceover to ground them in the visuals.
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
from src.core.llm_client import chat_completion, extract_json, load_prompt
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
        state.storyboard.sanitize_scenes()
        total_dur = sum(int(s.get("duration_seconds", 0) or 0) for s in state.storyboard.scenes)
        self._log(
            state,
            f"Draft created: '{state.storyboard.title}' ({len(state.storyboard.scenes)} scenes, {total_dur}s).",
        )
        self._log_scene_outline(state, include_images=False)
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
        total_dur = sum(int(s.get("duration_seconds", 0) or 0) for s in state.storyboard.scenes)
        self._log(
            state,
            f"Storyboard refined: '{state.storyboard.title}' ({len(state.storyboard.scenes)} scenes, {total_dur}s aligned with imagery).",
        )
        self._log_scene_outline(state, include_images=True)
        return state

    def _log_scene_outline(self, state: SharedState, include_images: bool = False) -> None:
        """Print a compact per-scene summary showing progress through drafting/refinement."""
        for s in state.storyboard.scenes:
            sid = s.get("scene_id", "?")
            dur = s.get("duration_seconds", "?")
            preview = str(s.get("voiceover") or s.get("audio_cue") or s.get("description") or "")[:60]
            img_tag = f" | img={s.get('image_ref') or 'none'}" if include_images else ""
            self._log(state, f"  Scene {sid} ({dur}s{img_tag}): \"{preview}\"")

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
2. Slightly tweak the visual description and voiceover if needed to match what is depicted in the chosen image asset.
3. Keep scene pacing (~15-25 spoken words per scene, ~120-145 words total for 60s) and target duration consistent.
"""


def _parse_storyboard_response(raw: str) -> dict:
    try:
        data = extract_json(raw)
        if isinstance(data, dict):
            return data
        return {"overall_narrative": raw, "scenes": [], "title": "Parse Error"}
    except Exception:
        return {"overall_narrative": raw, "scenes": [], "title": "Parse Error"}


_DRAFT_SYSTEM_PROMPT_DEFAULT = """\
You are a premier short-form video screenwriter and visual storyteller.
Your superpower is turning creative concepts and real-world facts into gripping, fast-paced 60-second video storyboards.

Core Philosophy:
- Faithfully execute the commissioned story pitch and directorial guidance.
- Adapt your voice and energy to match the commissioned tone (e.g., witty & comedic, mindbending curiosity, forensic mystery, chill craft, or high-stakes esports).
- Ground the script using the retrieved real-world facts and evidence—never rely on hollow generic fluff.
- Deliver an instant 3-second hook that halts the scroll.
- Ensure narrative progression: each scene must reveal new information, escalate stakes, or provide visual contrast.
- Voiceover & Pacing: Pure spoken narrative dialogue or voiceover lines only (NO SFX, NO music cues, NO stage labels). Keep speech concise so Text-To-Speech fits within the target duration: aim for ~15 to 25 spoken words per scene (~120 to 145 spoken words total across a 60s video; roughly 2 to 2.5 words per second of scene duration). Do NOT include an on-screen "subtitle" field.
- Language: Write the script in the primary language of the user's profile/history.

CRITICAL NEGATIVE CONSTRAINTS (VOICEOVER ONLY, ZERO SFX, NO PLACEHOLDERS):
- STRICTLY NO SOUND EFFECTS (SFX) OR MUSIC: Sound effects are NOT supported in this project.
  - FORBIDDEN: NEVER write "SFX: ...", "FX: ...", "(whoosh)", "(dramatic chord)", "(camera click)", or any sound effect descriptions.
- STRICTLY NO STAGE LABELS OR PREFIXES:
  - FORBIDDEN: NEVER write "VO:", "VO (...):", "Voix off:", "Narrator:", "Host:", "Speaker:", etc.
- STRICTLY NO PLACEHOLDER OR EVASIVE DESCRIPTIONS:
  - FORBIDDEN: "A character says an innocent line...", "Zoom on the key quote...", "Explains the double meaning..."
- REQUIRED: The "voiceover" field must contain ONLY the pure verbatim spoken words to be read aloud by Text-To-Speech.
  - EXAMPLE OF CORRECT VOICEOVER: "That warning label isn't for you. It's for about one in twelve thousand people."
  - EXAMPLE OF FORBIDDEN VOICEOVER: "VO (flat): 'That warning label isn't for you.' SFX: turntable hum"

Return ONLY valid JSON with this structure:
{
  "title": "Engaging video title",
  "overall_narrative": "2-3 sentence overview of the story arc and concept",
  "scenes": [
    {
      "scene_id": 1,
      "description": "Visual scene description: authentic action, dynamic camera movement, distinct visual elements",
      "voiceover": "Pure spoken words to be read aloud by voiceover (~15-25 words, NO 'VO:', NO SFX, NO music cues)",
      "duration_seconds": 10
    }
  ]
}

Total scenes: 4 to 6 scenes summing strictly to the target duration.
"""

_DRAFT_SYSTEM_PROMPT = load_prompt("scriptwriter_draft.txt", _DRAFT_SYSTEM_PROMPT_DEFAULT)


_REFINE_SYSTEM_PROMPT_DEFAULT = """\
You are an expert video director and storyteller refining an entertaining short video against real-world retrieved imagery (img1, img2, etc.).

Refine the scenes to anchor your entertaining narrative in the real visual evidence:
1. Ground the story into the real-world objects, places, or people shown in the image assets.
2. Maintain high entertainment value: witty, fast-paced, and captivating, while keeping spoken voiceover crisp (~15 to 25 words per scene, ~120 to 145 spoken words total for 60s) so Text-To-Speech fits the target duration. Do NOT include a "subtitle" field.
3. In "image_ref", use ONLY the symbolic identifiers provided (e.g. "img1", "img2") or null. NEVER write a full URL.
4. Language: Keep the same language as the draft.

CRITICAL NEGATIVE CONSTRAINTS (VOICEOVER ONLY, ZERO SFX, NO PLACEHOLDERS):
- STRICTLY NO SOUND EFFECTS (SFX) OR MUSIC: Sound effects are NOT supported in this project.
  - FORBIDDEN: NEVER write "SFX: ...", "FX: ...", "(whoosh)", "(dramatic chord)", "(camera click)", or any sound effect descriptions.
- STRICTLY NO STAGE LABELS OR PREFIXES:
  - FORBIDDEN: NEVER write "VO:", "VO (...):", "Voix off:", "Narrator:", "Host:", "Speaker:", etc.
- STRICTLY NO PLACEHOLDER OR EVASIVE DESCRIPTIONS:
  - FORBIDDEN: "A character says an innocent line...", "Zoom on the key quote...", "Explains the double meaning..."
- REQUIRED: The "voiceover" field must contain ONLY the pure verbatim spoken words to be read aloud by Text-To-Speech.
  - EXAMPLE OF CORRECT VOICEOVER: "That warning label isn't for you. It's for about one in twelve thousand people."
  - EXAMPLE OF FORBIDDEN VOICEOVER: "VO (flat): 'That warning label isn't for you.' SFX: turntable hum"

Return ONLY valid JSON with this structure:
{
  "title": "Refined title",
  "overall_narrative": "Refined story overview",
  "scenes": [
    {
      "scene_id": 1,
      "description": "Visual scene description aligned with the real-world image asset",
      "voiceover": "Pure spoken words to be read aloud by voiceover (~15-25 words, NO 'VO:', NO SFX, NO music cues)",
      "image_ref": "img1",
      "duration_seconds": 10
    }
  ]
}
"""

_REFINE_SYSTEM_PROMPT = load_prompt("scriptwriter_refine.txt", _REFINE_SYSTEM_PROMPT_DEFAULT)
