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

        if not state.image_context.images:
            self._log(
                state,
                "No image assets retrieved; keeping draft storyboard scenes with image_url=None.",
            )
            state.storyboard.resolve_image_urls(state.image_context)
            return state

        draft_scenes = list(state.storyboard.scenes)
        expected_count = len(draft_scenes)

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
        refined_scenes = refined_data.get("scenes") or []

        # Guard against truncated reasoning outputs dropping trailing payoff scenes.
        if len(refined_scenes) < expected_count:
            self._log(
                state,
                f"⚠ Refine pass returned {len(refined_scenes)}/{expected_count} scenes (likely truncation); retrying once.",
            )
            retry_prompt = (
                user_prompt
                + f"\n\nCRITICAL: Your previous attempt returned only {len(refined_scenes)} of {expected_count} scenes. "
                f"Return valid JSON containing ALL {expected_count} scenes (scene_id 1 through {expected_count}) completely."
            )
            retry_raw = chat_completion(
                client=self.client,
                model=self.model,
                system_prompt=_REFINE_SYSTEM_PROMPT,
                user_prompt=retry_prompt,
                temperature=0.4,
                max_tokens=6000,
            )
            retry_data = _parse_storyboard_response(retry_raw)
            retry_scenes = retry_data.get("scenes") or []
            if len(retry_scenes) >= len(refined_scenes):
                refined_data = retry_data
                refined_scenes = retry_scenes

        if refined_scenes:
            if len(refined_scenes) < expected_count:
                self._log(
                    state,
                    f"⚠ Refine retry still returned {len(refined_scenes)}/{expected_count} scenes; preserving missing draft scenes with matched asset refs.",
                )
                merged = list(refined_scenes)
                # Preserve trailing draft scenes cleanly and link catalog assets by target_scene_id.
                target_to_img = {
                    img.get("target_scene_id"): img.get("image_id")
                    for img in state.image_context.images
                    if img.get("target_scene_id") is not None and img.get("image_id")
                }
                for ds in draft_scenes[len(refined_scenes):]:
                    fallback_scene = dict(ds)
                    sid = fallback_scene.get("scene_id")
                    if not fallback_scene.get("image_ref") and sid in target_to_img:
                        fallback_scene["image_ref"] = target_to_img[sid]
                    merged.append(fallback_scene)
                refined_scenes = merged

            state.storyboard.scenes = refined_scenes
            if refined_data.get("title"):
                state.storyboard.title = refined_data["title"]
            if refined_data.get("overall_narrative"):
                state.storyboard.overall_narrative = refined_data["overall_narrative"]

        # Normalize voiceovers/durations and programmatically link image_ref -> real image_url
        state.storyboard.sanitize_scenes()
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
    news = state.knowledge_context if state.knowledge_context.snippets else state.news_context
    pitch = state.selected_pitch
    director_notes = state.director_notes

    news_block = "\n".join(
        f"- {s.get('headline', '')}: {s.get('summary', '')}"
        for s in news.snippets[:9]
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
Faithfully bring the commissioned pitch to life following the Director's notes and integrating the retrieved facts directly into the spoken voiceover.
Ensure the narrative is 100% understandable on the very first watch: establish all rules/context cleanly up front, explain the causal "why" in plain words (no cryptic shorthand jumps), and write strictly as ONE external off-screen voiceover narrator (NO character roleplay, NO spectator dialogue).
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
1. For each scene, assign the best fitting asset ID (e.g. "image_ref": "img1") ONLY if it genuinely fits the scene, or null if no listed asset matches (do NOT force an unrelated image onto a scene).
2. Slightly tweak the visual description and voiceover if needed to match what is depicted in the chosen image asset.
3. Preserve (and clarify if any step feels rushed) complete first-watch setup/rules and causal explanation ("why" it works).
4. Enforce ONE consistent external off-screen narrator voiceover across all scenes (eliminate any character roleplay, spectator commands, or dialogue). Keep scene pacing (~15-25 spoken words per scene, ~125-150 words total for 60s) and target duration consistent.
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
- STANDALONE CLARITY & FIRST-WATCH UNDERSTANDABILITY (CRITICAL): Write a 100% self-contained story completely understandable on the VERY FIRST WATCH to someone who has zero prior knowledge of the topic, riddle, or watch history.
  - Setup & Rules FIRST: Always establish *who*, *what physical objects exist*, *the exact rules/constraints*, and *the stakes* in Scenes 1–2 BEFORE introducing twists, strategies, statistics, or academic names/dates (never open with unexplained paper citations before the premise makes sense).
  - Explain the "WHY" (Zero Cryptic Jumps): Never state a punchline or mathematical rule as magic shorthand. Explicitly state in plain everyday terms *why* the mechanism produces that outcome (e.g., explain *why* following numbers forms closed loops before stating loop length, or *why* interleaving halves preserves relative order inside each half).
  - Slideshow Format Awareness: Because the video renders as stock photos with a single AI voiceover (no live video acting or pausing), the spoken narration alone must carry every step smoothly from first word to last.
- SHOWABLE PHYSICAL SCENES & SPOKEN PAYOFF: Ground visuals in real physical places/objects that stock photography can show (NEVER rely on viewers reading fabricated phone chat bubbles or on-screen text overlays). Deliver every crucial fact and the final explanatory payoff aloud in "voiceover".
- Deliver an instant 3-second hook that halts the scroll.
- Ensure narrative progression: each scene must reveal new information, escalate stakes, or provide visual contrast.
- Voiceover & Pacing: Pure single-narrator off-screen voiceover narration only (NEVER character dialogue, NEVER roleplay, NO SFX, NO music cues, NO stage labels). Keep speech concise yet complete enough to stay crystal clear so Text-To-Speech fits within the target duration: use 5 to 6 scenes (prefer 6 scenes whenever establishing setup rules + a causal explanation), aiming for ~15 to 25 spoken words per scene (~125 to 150 spoken words total across a 60s video). Do NOT include an on-screen "subtitle" field.
- Language: Write the script in the primary language of the user's profile/history.

CRITICAL NEGATIVE CONSTRAINTS (VOICEOVER ONLY, ZERO ROLEPLAY, ZERO SFX, NO PLACEHOLDERS):
- STRICTLY SINGLE EXTERNAL VOICEOVER NARRATOR (NO CHARACTER ROLEPLAY OR DIALOGUE):
  - Write the entire script from ONE consistent external off-screen documentary/explanatory narrator voice speaking about the subject.
  - FORBIDDEN — IN-SCENE ROLEPLAY / SPECTATOR ADDRESS: NEVER write voiceover as a character inside the scene acting out props or talking to an imaginary spectator (e.g., FORBIDDEN: "D'abord je range les cartes... et je te les donne", "Maintenant mélange vraiment", "Retourne cette carte", "Tu seras pendu demain"). Instead narrate cleanly in third person (e.g., REQUIRED: "Le magicien prépare deux moitiés alternées, puis laisse un spectateur mélanger le jeu une seule fois.").
  - FORBIDDEN — CHARACTER-TO-CHARACTER DIALOGUE OR SHIFTING FIRST-PERSON LINES: NEVER switch into first-person lines spoken by characters ("Si je suis vivant jeudi, alors..."). Use smooth third-person indirect narration throughout.
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
2. FIRST-WATCH CLARITY & CAUSAL COMPLETENESS: Preserve (and clarify if needed) all essential setup rules, context, and plain-language causal explanations ("why" the trick/phenomenon works) so a viewer with zero prior knowledge understands the entire point on the first watch. Never compress away crucial setup or causal bridges.
3. Maintain high entertainment value: witty, fast-paced, and captivating, while keeping spoken voiceover crisp (~15 to 25 words per scene, ~125 to 150 spoken words total for 60s) so Text-To-Speech fits the target duration. Do NOT include a "subtitle" field.
4. In "image_ref", use ONLY the symbolic identifiers provided (e.g. "img1", "img2") that genuinely fit the scene, or null if no listed asset matches (never force an unrelated image onto a scene). NEVER write a full URL.
5. Language: Keep the same language as the draft.

CRITICAL NEGATIVE CONSTRAINTS (VOICEOVER ONLY, ZERO ROLEPLAY, ZERO SFX, NO PLACEHOLDERS):
- STRICTLY SINGLE EXTERNAL VOICEOVER NARRATOR (NO CHARACTER ROLEPLAY OR DIALOGUE):
  - Ensure every scene uses ONE consistent external off-screen narrator voice speaking about the events (NEVER roleplay as an in-scene magician/character talking to another character or giving live commands to a spectator like "je te les donne", "mélange vraiment", "retourne la carte"). Convert any first-person character lines or direct dialogue into smooth third-person narrator voiceover.
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
