"""
DirectorAgent — Phase 3 in the Multi-Agent Pipeline.

Responsibility:
  - Acts as the Executive Creative Director / Showrunner.
  - Receives the 3 candidate pitches from the Writers' Room.
  - Critiques each pitch on Hook Strength, Personal Resonance, and Visual Feasibility.
  - Selects the winning pitch and issues authoritative Directorial Notes for the
    production team (NewsRetrieverAgent and ScriptwriterAgent).

Output written to state:
  - state.selected_pitch: dict[str, Any]
  - state.director_critique: list[dict[str, Any]]
  - state.director_notes: str
"""

from __future__ import annotations

import json
from typing import Any
from openai import OpenAI

from src.agents.base_agent import BaseAgent
from src.core.llm_client import chat_completion, extract_json, load_prompt
from src.core.state import SharedState


class DirectorAgent(BaseAgent):
    """Evaluates candidate pitches, commissions the winning story, and issues directorial guidance."""

    def __init__(self, model: str, client: OpenAI) -> None:
        super().__init__(name="DirectorAgent", model=model, client=client)

    def run(self, state: SharedState) -> SharedState:
        pitches = state.candidate_pitches
        if not pitches:
            self._log(state, "No candidate pitches found. Skipping Director deliberation.")
            return state

        self._log(state, f"Reviewing {len(pitches)} candidate pitches in the Director's chair.")

        user_prompt = _build_director_prompt(state.user_profile, pitches)

        raw = chat_completion(
            client=self.client,
            model=self.model,
            system_prompt=_DIRECTOR_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=0.5,
            max_tokens=1500,
        )

        decision_data = _parse_decision(raw)
        winning_id = decision_data.get("winning_pitch_id", "pitch1")

        # Find matching pitch dict
        selected = next((p for p in pitches if p.get("pitch_id") == winning_id), pitches[0])

        state.selected_pitch = selected
        state.director_critique = decision_data.get("critiques", [])
        state.director_notes = decision_data.get("director_notes", "")

        for c in state.director_critique:
            if isinstance(c, dict):
                pid = c.get("pitch_id", "?")
                verdict = c.get("verdict") or c.get("strengths") or c.get("critique", "")
                if verdict:
                    self._log(state, f"  Critique [{pid}]: {str(verdict)[:120]}")

        self._log(state, f"🏆 Commissioned winning pitch: [{selected.get('pitch_id')}] '{selected.get('title')}'")
        if state.director_notes:
            self._log(state, f"Directorial Notes: {state.director_notes[:180]}{'...' if len(state.director_notes) > 180 else ''}")

        return state


def _build_director_prompt(profile: Any, pitches: list[dict[str, Any]]) -> str:
    pitches_str = "\n\n".join(
        f"--- PITCH [{p.get('pitch_id')}] ---\n"
        f"Title: {p.get('title')}\n"
        f"Tone: {p.get('target_tone')}\n"
        f"3s Hook: \"{p.get('hook_3s')}\"\n"
        f"Concept: {p.get('concept')}\n"
        f"Core Beats: {', '.join(p.get('core_beats', []))}\n"
        f"Target Fit: {p.get('user_alignment_reason')}"
        for p in pitches
    )

    return f"""\
USER PROFILE
============
Interests: {', '.join(profile.interests)}
Preferred Style: {profile.preferred_style or 'Engaging'}
Summary: {profile.raw_summary}

CANDIDATE PITCHES:
==================
{pitches_str}

TASK:
Critique each pitch, select the SINGLE STRONGEST pitch for production, and write concrete Directorial Notes for the Researcher and Scriptwriter.
"""


def _parse_decision(raw: str) -> dict:
    try:
        data = extract_json(raw)
        if isinstance(data, dict):
            return data
        return {"winning_pitch_id": "pitch1", "critiques": [], "director_notes": "Focus on high energy and clean pacing."}
    except Exception:
        return {"winning_pitch_id": "pitch1", "critiques": [], "director_notes": "Focus on high energy and clean pacing."}


_DIRECTOR_SYSTEM_PROMPT_DEFAULT = """\
You are an award-winning Executive Creative Director for short-form video (YouTube Shorts, Reels).
Your job is to make tough editorial decisions: review 3 pitches, critique them, pick the winning pitch, and give clear production instructions.

Evaluation Rubric:
1. Hook & Retention Potential: Does the 3-second hook immediately stop the thumb scroll?
2. Personal Resonance: Does this authentically match what this viewer loves without being generic?
3. Visual & Dynamic Feasibility: Can this be brought alive visually across 6 scenes within 60s?

Return ONLY valid JSON with this structure:
{
  "critiques": [
    {
      "pitch_id": "pitch1",
      "strengths": "What works well",
      "weaknesses": "What falls flat or risks losing retention"
    },
    {
      "pitch_id": "pitch2",
      "strengths": "...",
      "weaknesses": "..."
    },
    {
      "pitch_id": "pitch3",
      "strengths": "...",
      "weaknesses": "..."
    }
  ],
  "winning_pitch_id": "pitch1",
  "selection_rationale": "Clear 2-sentence explanation of why this pitch beat the others",
  "director_notes": "Authoritative, clear guidance for the Fact Researcher and Scriptwriter: specify exact angles to research, tone/pacing to maintain, and what the climax must deliver."
}
"""

_DIRECTOR_SYSTEM_PROMPT = load_prompt("director.txt", _DIRECTOR_SYSTEM_PROMPT_DEFAULT)
