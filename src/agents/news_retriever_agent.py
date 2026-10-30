"""
NewsRetrieverAgent — Agent 2 in the pipeline.

Responsibility:
  - Takes the user profile produced by ProfilerAgent.
  - Searches for recent news articles / web content relevant to the
    user's interests and themes.
  - Returns a list of headlines + summaries to enrich the storyboard.

Tool used:
  - src/tools/news_tool.py  (wraps a news/search API — TBD: NewsAPI, Serper, etc.)

Output written to state:
  - state.news_context  (NewsContext dataclass)
"""

from __future__ import annotations

from openai import OpenAI

from src.agents.base_agent import BaseAgent
from src.core.llm_client import chat_completion
from src.core.state import NewsContext, SharedState
from src.tools.news_tool import search_news  # noqa: F401  (to be implemented)


class NewsRetrieverAgent(BaseAgent):
    """Retrieves recent news relevant to the user's interest profile."""

    def __init__(self, model: str, client: OpenAI, max_results: int = 5) -> None:
        super().__init__(name="NewsRetrieverAgent", model=model, client=client)
        self.max_results = max_results

    def run(self, state: SharedState) -> SharedState:
        self._log(state, "Starting news retrieval based on user profile.")

        # Build search queries from the user profile
        queries = state.user_profile.interests[:3]  # use top-3 interests
        if not queries:
            self._log(state, "No interests found in profile. Skipping news retrieval.")
            return state

        all_snippets: list[dict] = []
        for query in queries:
            self._log(state, f"Searching news for: '{query}'")
            # TODO: implement search_news in src/tools/news_tool.py
            results = search_news(query=query, max_results=self.max_results)
            all_snippets.extend(results)

        # Optionally: use LLM to filter / summarise the fetched snippets
        # TODO: add LLM-based relevance filtering pass here

        state.news_context = NewsContext(snippets=all_snippets)
        self._log(state, f"Retrieved {len(all_snippets)} news snippets.")
        return state
