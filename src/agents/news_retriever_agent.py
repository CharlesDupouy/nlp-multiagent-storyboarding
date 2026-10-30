"""
NewsRetrieverAgent — Agent 2 in the pipeline.

Responsibility:
  - Takes the user profile produced by ProfilerAgent.
  - Uses an LLM to actively brainstorm timely, interesting news and trend angles
    that intersect the user's specific interests and themes.
  - Queries external search/news tools with these targeted queries.
  - Enriches SharedState with retrieved headlines, summaries, and URLs.

Tool used:
  - src/tools/news_tool.py  (wraps search / news API)

Output written to state:
  - state.news_context  (NewsContext dataclass)
"""

from __future__ import annotations

import json
from openai import OpenAI

from src.agents.base_agent import BaseAgent
from src.core.llm_client import chat_completion
from src.core.state import NewsContext, SharedState
from src.tools.news_tool import search_news


class NewsRetrieverAgent(BaseAgent):
    """
    LLM-driven news and web retrieval agent.
    Brainstorms creative search topics from user interests before executing tool calls.
    """

    def __init__(self, model: str, client: OpenAI, max_results: int = 3, num_queries: int = 3) -> None:
        super().__init__(name="NewsRetrieverAgent", model=model, client=client)
        self.max_results = max_results
        self.num_queries = num_queries

    def run(self, state: SharedState) -> SharedState:
        profile = state.user_profile
        pitch = state.selected_pitch
        director_notes = state.director_notes

        if pitch:
            self._log(state, f"Targeting news & fact retrieval for winning pitch: '{pitch.get('title')}'")
            user_prompt = f"""\
WINNING PITCH:
- Title: {pitch.get('title')}
- Tone: {pitch.get('target_tone')}
- Concept: {pitch.get('concept')}
- 3s Hook: "{pitch.get('hook_3s')}"

DIRECTORIAL NOTES:
{director_notes or 'Find compelling facts, verified details, and fresh angles.'}

TASK:
Formulate {self.num_queries} specific search queries to retrieve real-world facts, recent news, controversy details, or proof points to back up this specific story.
"""
            system_prompt = _TARGETED_RESEARCH_SYSTEM_PROMPT
        else:
            self._log(state, "Brainstorming news queries from user profile using LLM.")
            if not profile.interests and not profile.themes:
                self._log(state, "No interests or themes found in profile. Skipping news retrieval.")
                return state

            user_prompt = f"""\
USER PROFILE:
- Specific Interests: {', '.join(profile.interests)}
- Core Themes: {', '.join(profile.themes)}
- Preferred Style: {profile.preferred_style}
- Summary: {profile.raw_summary}

Brainstorm {self.num_queries} specific, compelling search queries for recent news, trending stories, or viral techniques that would interest this user.
"""
            system_prompt = _NEWS_BRAINSTORM_SYSTEM_PROMPT

        # 1. LLM Brainstorming Pass
        raw_response = chat_completion(
            client=self.client,
            model=self.model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.7,
            max_tokens=500,
        )

        queries = _parse_queries(raw_response)
        if not queries:
            if pitch:
                queries = [pitch.get("title", "viral story")]
            else:
                queries = profile.interests[: self.num_queries]

        self._log(state, f"Formulated research queries: {queries}")

        # 2. Tool Execution Pass
        all_snippets: list[dict[str, str]] = []
        for query in queries:
            self._log(state, f"Searching information for: '{query}'")
            results = search_news(query=query, max_results=self.max_results)
            all_snippets.extend(results)

        state.news_context = NewsContext(snippets=all_snippets)
        self._log(state, f"Retrieved {len(all_snippets)} news snippets across {len(queries)} topics.")
        return state


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_queries(raw: str) -> list[str]:
    """Parse JSON list of query strings from LLM response."""
    try:
        cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        data = json.loads(cleaned)
        if isinstance(data, list):
            return [str(q).strip() for q in data if str(q).strip()]
        if isinstance(data, dict):
            # Check common keys like "queries" or "search_queries"
            for k in ("queries", "search_queries", "topics"):
                if k in data and isinstance(data[k], list):
                    return [str(q).strip() for q in data[k] if str(q).strip()]
    except Exception:
        pass
    return []


_TARGETED_RESEARCH_SYSTEM_PROMPT = """\
You are an expert factual researcher on a high-performing video production team.
The Director has selected a specific winning story pitch and issued production notes.
Your task is to formulate 3 precise search queries to retrieve verifiable real-world facts, recent developments, specific controversy records, or technical details that back up this exact story concept.

Guidelines:
- Align directly with the Director's instructions and the winning concept.
- Target queries that uncover concrete details: exact dates, real names, statistics, quotes, or counterintuitive evidence.
- Avoid generic search terms; make queries specific and journalistic.

Return ONLY valid JSON in this format:
{
  "queries": [
    "query 1",
    "query 2",
    "query 3"
  ]
}
"""


_NEWS_BRAINSTORM_SYSTEM_PROMPT = """\
You are an investigative pop-culture and storytelling researcher gathering wild, true real-world stories and facts.
Your goal is to dig up mindblowing external information (bizarre historical anecdotes, shocking real-world controversies, counterintuitive discoveries, or incredible true stories) related to the user's interests.

Guidelines:
- Formulate 3 search queries designed to find fascinating, real-world ammunition for an entertaining story.
- Target queries like: "the insane true story behind [topic]", "the bizarre 2026 controversy with [topic]", "the unbelievable origin of [topic]", or "how [topic] actually happened in real life".
- Avoid dry academic search terms. Search for real stories and facts that would shock or captivate an audience in 60 seconds.

Return ONLY valid JSON in this format:
{
  "queries": [
    "query 1",
    "query 2",
    "query 3"
  ]
}
"""

