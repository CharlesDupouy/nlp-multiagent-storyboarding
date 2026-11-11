"""
WritersRoomAgent — Phase 2 in the Multi-Agent Pipeline.

Responsibility:
  - Takes the rich UserProfile (including investigated transcript insights).
  - Acts as a collaborative creative writers' room.
  - Generates 3 distinct, high-potential candidate story pitches tailored to the
    user's natural style preference (witty banter, deep mystery, edutainment puzzle,
    chill craft, or gaming/pop-culture lore).
  - Does NOT lock the user into a rigid single format: adapts flexibly to their vibe.

Output written to state:
  - state.candidate_pitches: list[dict[str, Any]] (3 structured pitches)
"""

from __future__ import annotations

import json
from typing import Any
from openai import OpenAI

from src.agents.base_agent import BaseAgent
from src.core.llm_client import chat_completion, extract_json, load_prompt
from src.core.state import SharedState


class WritersRoomAgent(BaseAgent):
    """Generates 3 diverse candidate short-video pitches tailored to user tastes."""

    def __init__(self, model: str, client: OpenAI) -> None:
        super().__init__(name="WritersRoomAgent", model=model, client=client)

    def run(self, state: SharedState) -> SharedState:
        profile = state.user_profile
        self._log(state, " Convening Writers' Room to brainstorm 3 distinct candidate pitches.")

        user_prompt = _build_writers_room_prompt(profile)

        raw = chat_completion(
            client=self.client,
            model=self.model,
            system_prompt=_WRITERS_ROOM_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=0.8,
            max_tokens=2000,
        )

        pitches_data, parse_err = _parse_pitches_checked(raw)
        pitches = pitches_data.get("pitches", [])

        if not pitches:
            reason = parse_err or "empty 'pitches' array in JSON response"
            self._log(state, f"Writers' Room attempt 1 returned 0 pitches ({reason}); retrying with strict JSON repair...")
            retry_prompt = (
                f"{user_prompt}\n\n"
                "IMPORTANT RETRY INSTRUCTION: Your previous response could not be parsed as valid JSON "
                f"({reason}). Return ONLY one valid JSON object with top-level key \"pitches\" containing 3 pitch objects. "
                "Escape all double quotes inside string values as \\\" or use single quotes/guillemets."
            )
            raw_retry = chat_completion(
                client=self.client,
                model=self.model,
                system_prompt=_WRITERS_ROOM_SYSTEM_PROMPT,
                user_prompt=retry_prompt,
                temperature=0.4,
                max_tokens=2500,
            )
            pitches_data, parse_err = _parse_pitches_checked(raw_retry)
            pitches = pitches_data.get("pitches", [])
            if not pitches:
                self._log(state, f"[WARNING] Writers' Room retry still produced 0 pitches ({parse_err or 'empty list'}).")

        # Ensure fallback pitch IDs
        for i, p in enumerate(pitches):
            if "pitch_id" not in p:
                p["pitch_id"] = f"pitch{i+1}"

        state.candidate_pitches = pitches
        self._log(state, f"Developed {len(pitches)} candidate pitches:")
        for p in pitches:
            self._log(state, f"  [{p.get('pitch_id')}] '{p.get('title')}' (Tone: {p.get('target_tone')})")
            if p.get("hook_3s"):
                self._log(state, f"      3s Hook: \"{p.get('hook_3s')}\"")

        return state


def _build_writers_room_prompt(profile: Any) -> str:
    insights_str = "\n".join(f"- {ins}" for ins in profile.investigated_insights) if profile.investigated_insights else "None"

    return f"""\
USER PROFILE
============
Interests: {', '.join(profile.interests)}
Themes:    {', '.join(profile.themes)}
Preferred Style / Vibe: {profile.preferred_style or 'Engaging & tailored'}
Summary:   {profile.raw_summary}

DEEP-DIVE TRANSCRIPT INSIGHTS:
{insights_str}

TASK:
Brainstorm 3 DISTINCT, HIGH-POTENTIAL video pitches (each suitable for a 60-second Short).
Make sure each pitch explores a DIFFERENT topic or angle from the user's profile and matches their natural style preference.
"""


def _parse_pitches_checked(raw: str) -> tuple[dict, str | None]:
    """Parse candidate pitches JSON and return (parsed_dict, error_reason_or_None)."""
    try:
        data = extract_json(raw)
        if isinstance(data, dict):
            return data, None
        if isinstance(data, list):
            return {"pitches": data}, None
        return {"pitches": []}, f"unexpected JSON type {type(data).__name__}"
    except Exception as exc:
        return {"pitches": []}, f"{type(exc).__name__}: {exc}"


def _parse_pitches(raw: str) -> dict:
    data, _ = _parse_pitches_checked(raw)
    return data


_WRITERS_ROOM_SYSTEM_PROMPT_DEFAULT = """\
You are the lead showrunner of a premier short-form video writers' room.
Your mission is to pitch 3 DISTINCT, highly engaging 60-second video concepts tailored to the user's unique personality and watch history.

Crucial Guidelines:
1. Adapt to the user's natural style:
   - If they love humor, comedy sketches, or couple dynamics, pitch witty, relatable concepts.
   - If they love puzzles, math, or tech, pitch mindbending, counterintuitive curiosities.
   - If they love nostalgia, pop culture, or gaming, pitch deep-dive lore, hidden details, or controversies.
   - Do NOT force a single rigid format on every user.
2. DIVERSITY across the 3 pitches:
   - Pitch 1: Explores one major interest pillar from their profile.
   - Pitch 2: Explores a second distinct passion or transcript insight.
   - Pitch 3: A creative cross-over or surprising angle they wouldn't expect.
3. Every pitch MUST have an instant 3-second hook that stops the scroll.
4. NO REMAKES / ANTI-REGURGITATION (CRITICAL):
   - NEVER pitch a remake, recap, or direct variation of a specific video or skit already in the user's watch history or transcripts. Extend their underlying curiosity into a BRAND-NEW topic, story, or mechanism they haven't seen.
5. VISUAL & NARRATIVE FEASIBILITY (CRITICAL):
   - FIRST-WATCH UNDERSTANDABILITY: Every pitch MUST work as a self-contained 60s narrated story (clear rules/setup -> causal mechanism explaining *why* it works -> spoken payoff) understandable on the very first watch by someone with zero prior knowledge. Do NOT pick a puzzle or trick unless its setup AND the intuitive "why" can be explained cleanly in plain words within 60s.
   - SINGLE EXTERNAL VOICEOVER NARRATOR: Every pitch must be written for ONE external off-screen documentary/storyteller voiceover narrator (NO character-to-character dialogue, NO in-scene roleplay pretending to talk to an on-screen spectator like "I hand you the deck / now shuffle it").
   - NEVER pitch premises whose humor/narrative depends on reading tiny fabricated chat-UI bubbles on a phone screen — keep scenes grounded in concrete physical subjects that stock photography can show.
6. Language: Match the primary language of the user's profile/history (French if the history is primarily French, English if English). Do not switch language just because one video title or transcript is translated.

Return ONLY valid JSON with this structure:
{
  "pitches": [
    {
      "pitch_id": "pitch1",
      "title": "Punchy, irresistible video title",
      "target_tone": "e.g. Witty & ironic / Mindbending & curious / High-energy forensic",
      "hook_3s": "Exact visual action and opening spoken line (0-3s)",
      "concept": "2-3 sentences explaining the core premise and conflict/revelation",
      "core_beats": [
        "Beat 1: The setup/provocation",
        "Beat 2: The twist or escalation",
        "Beat 3: The climax or takeaway"
      ],
      "user_alignment_reason": "Why this specific user will love this concept based on their history"
    },
    {
      "pitch_id": "pitch2",
      "title": "...",
      "target_tone": "...",
      "hook_3s": "...",
      "concept": "...",
      "core_beats": ["..."],
      "user_alignment_reason": "..."
    },
    {
      "pitch_id": "pitch3",
      "title": "...",
      "target_tone": "...",
      "hook_3s": "...",
      "concept": "...",
      "core_beats": ["..."],
      "user_alignment_reason": "..."
    }
  ]
}
"""

_WRITERS_ROOM_SYSTEM_PROMPT = load_prompt("writers_room.txt", _WRITERS_ROOM_SYSTEM_PROMPT_DEFAULT)
