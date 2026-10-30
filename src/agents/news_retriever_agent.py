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
        self._log(state, "Brainstorming news queries from user profile using LLM.")

        profile = state.user_profile
        if not profile.interests and not profile.themes:
            self._log(state, "No interests or themes found in profile. Skipping news retrieval.")
            return state

        # 1. LLM Brainstorming Pass
        user_prompt = f"""\
USER PROFILE:
- Specific Interests: {', '.join(profile.interests)}
- Core Themes: {', '.join(profile.themes)}
- Preferred Style: {profile.preferred_style}
- Summary: {profile.raw_summary}

Brainstorm {self.num_queries} specific, compelling search queries for recent news, trending stories, or viral techniques that would interest this user.
"""
        raw_response = chat_completion(
            client=self.client,
            model=self.model,
            system_prompt=_NEWS_BRAINSTORM_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=0.7,
            max_tokens=500,
        )

        queries = _parse_queries(raw_response)
        if not queries:
            # Fallback to top interests if JSON parsing fails
            self._log(state, "Failed to parse brainstormed queries; falling back to top profile interests.")
            queries = profile.interests[: self.num_queries]

        self._log(state, f"Brainstormed queries: {queries}")

        # 2. Tool Execution Pass
        all_snippets: list[dict[str, str]] = []
        for query in queries:
            self._log(state, f"Searching news for: '{query}'")
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


_NEWS_BRAINSTORM_SYSTEM_PROMPT = """\
You are an investigative science and culture journalist formulating research queries for an informative short documentary.
Your goal is to uncover surprising, educational, and fact-rich stories matching the user's intellectual curiosity.

Given the user's profile:
1. Brainstorm search queries focused on real mechanisms, fascinating historical origins, scientific discoveries, or technical breakdowns.
2. Formulate questions like: "the real origin of...", "how X actually works scientifically", "the engineering behind Y", or "little-known historical facts about Z".
3. Avoid clickbait or shallow meme queries. Prioritize depth, educational value, and genuine insight.

Return ONLY valid JSON in this format:
{
  "queries": [
    "query 1",
    "query 2",
    "query 3"
  ]
}
"""
