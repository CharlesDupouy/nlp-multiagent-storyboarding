"""
ProfilerAgent — Agent 1 in the pipeline.

Responsibility:
  - Receives the raw user watch history (video titles + URLs).
  - Has NO access to external tools (no web search, no news).
  - Uses an LLM to infer the user's interests, themes, and content
    style preferences solely from the titles and URLs provided.
  - Optionally: could be extended to fetch video captions/tags via
    a YouTube Data API tool call (future work beyond MVP).

Output written to state:
  - state.user_profile  (UserProfile dataclass)
"""

from __future__ import annotations

import json

from openai import OpenAI

from src.agents.base_agent import BaseAgent
from src.core.llm_client import chat_completion
from src.core.state import SharedState, UserProfile


class ProfilerAgent(BaseAgent):
    """Infers a structured user interest profile from watch history."""

    def __init__(self, model: str, client: OpenAI) -> None:
        super().__init__(name="ProfilerAgent", model=model, client=client)

    def run(self, state: SharedState) -> SharedState:
        self._log(state, "Starting user profiling from watch history.")

        # Format the history for the LLM prompt using structured IDs (vid1, vid2, ...)
        history_text = "\n".join(
            f"- [vid{i+1}] Title: \"{v.get('title', '')}\" | URL: {v.get('url', '')}"
            for i, v in enumerate(state.user_history.videos)
        )

        # Load system prompt from file (see prompts/profiler_system.txt)
        # TODO: load from prompts/ directory once prompts are finalised
        system_prompt = _PROFILER_SYSTEM_PROMPT
        user_prompt = f"Here is the user's watch history:\n\n{history_text}"

        self._log(state, f"Calling LLM ({self.model}) to build user profile.")
        raw_response = chat_completion(
            client=self.client,
            model=self.model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.4,   # lower temperature for factual extraction
            max_tokens=1024,
        )

        # Parse the structured JSON output from the LLM
        # TODO: add retry / fallback if JSON is malformed
        profile_data = _parse_profile_response(raw_response)
        state.user_profile = UserProfile(
            interests=profile_data.get("interests", []),
            themes=profile_data.get("themes", []),
            preferred_style=profile_data.get("preferred_style", ""),
            raw_summary=profile_data.get("raw_summary", raw_response),
        )

        self._log(
            state,
            f"Profile built. Interests: {state.user_profile.interests}. "
            f"Style: {state.user_profile.preferred_style}.",
        )
        return state


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_profile_response(raw: str) -> dict:
    """
    Extract JSON from the LLM response.
    Expects the LLM to return a JSON block; falls back to an empty dict.
    """
    try:
        # Strip markdown code fences if present
        cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Graceful degradation: store the raw text in raw_summary
        return {"raw_summary": raw}


_PROFILER_SYSTEM_PROMPT = """\
You are an expert analyst of intellectual curiosity and user interests.
Given a list of YouTube Shorts video titles and URLs, identify what this user genuinely wants to learn about, understand, or explore.

Focus on:
- Underlying subjects of curiosity (e.g. historical mechanisms, mathematics, technical skills, science, language, craft techniques, culture).
- The depth of concepts they engage with.
- An informative, curiosity-driven lens rather than superficial entertainment.

Return ONLY valid JSON with the following keys:
{
  "interests": ["list", "of", "specific", "intellectual", "or", "skill-based", "topics"],
  "themes":    ["broader", "educational", "or", "cultural", "themes"],
  "preferred_style": "informative",
  "raw_summary": "2-3 sentences synthesizing this user's curiosity profile and the kinds of insightful, informative breakdowns they would appreciate."
}

Be specific and substantive. Avoid superficial labels like 'comedy' or 'funny videos' when there is an underlying topic (e.g. language nuances, social observation, logic puzzles).
"""
