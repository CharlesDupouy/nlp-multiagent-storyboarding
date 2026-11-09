"""
KnowledgeRetrieverAgent — Fact, Lore, Science & News Retrieval Agent (Phase 4).

Responsibility:
  - Formulates targeted search queries across specific domains (news, encyclopedia, technical).
  - Uses multi-tiered retrieval (NewsAPI, Wikipedia, Tavily, and LLM Tier 3 Fact Synthesis).
  - Enriches SharedState with concrete, verified real-world facts, quotes, timestamps, and lore.

Tools used:
  - src/tools/knowledge_tool.py (multi-tiered news & knowledge retriever)

Output written to state:
  - state.knowledge_context / state.news_context (KnowledgeContext dataclass)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from openai import OpenAI

from src.agents.base_agent import BaseAgent
from src.core.llm_client import chat_completion, extract_json
from src.core.state import KnowledgeContext, NewsContext, SharedState
from src.tools.knowledge_tool import search_knowledge


class KnowledgeRetrieverAgent(BaseAgent):
    """
    Targeted fact, news, and lore retrieval agent.
    Formulates domain-aware queries (news, encyclopedia, technical) and executes multi-tiered lookup.
    """

    def __init__(self, model: str, client: OpenAI, max_results: int = 3, num_queries: int = 3) -> None:
        super().__init__(name="KnowledgeRetrieverAgent", model=model, client=client)
        self.max_results = max_results
        self.num_queries = num_queries

    def run(self, state: SharedState) -> SharedState:
        profile = state.user_profile
        pitch = state.selected_pitch
        director_notes = state.director_notes

        if pitch:
            self._log(state, f"Targeting knowledge & fact retrieval for winning pitch: '{pitch.get('title')}'")
            user_prompt = f"""\
WINNING PITCH:
- Title: {pitch.get('title')}
- Tone: {pitch.get('target_tone')}
- Concept: {pitch.get('concept')}
- 3s Hook: "{pitch.get('hook_3s')}"

DIRECTORIAL NOTES:
{director_notes or 'Find compelling facts, verified details, and fresh angles.'}

TASK:
Formulate {self.num_queries} specific search queries to retrieve verifiable real-world facts, dates, controversy details, or proof points to back up this specific story.
Choose the best target domain for each query: "news" for breaking/current events, "encyclopedia" for pop-culture lore/history, or "technical" for scientific/mathematical mechanisms.
"""
            system_prompt = _TARGETED_RESEARCH_SYSTEM_PROMPT
        else:
            self._log(state, "Brainstorming knowledge queries from user profile using LLM.")
            if not profile.interests and not profile.themes:
                self._log(state, "No interests or themes found in profile. Skipping knowledge retrieval.")
                return state

            user_prompt = f"""\
USER PROFILE:
- Specific Interests: {', '.join(profile.interests)}
- Core Themes: {', '.join(profile.themes)}
- Preferred Style: {profile.preferred_style}
- Summary: {profile.raw_summary}

Brainstorm {self.num_queries} specific, compelling search queries across news, lore, or science for real-world stories or viral techniques that would interest this user.
"""
            system_prompt = _NEWS_BRAINSTORM_SYSTEM_PROMPT

        # 1. LLM Brainstorming Pass
        raw_response = chat_completion(
            client=self.client,
            model=self.model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.7,
            max_tokens=600,
        )

        research_queries = _parse_research_queries(raw_response)
        if not research_queries:
            if pitch:
                research_queries = [{"query": pitch.get("title", "viral story"), "domain": "auto", "rationale": "pitch title"}]
            else:
                research_queries = [
                    {"query": interest, "domain": "auto", "rationale": "user interest"}
                    for interest in profile.interests[: self.num_queries]
                ]

        self._log(state, f"Formulated {len(research_queries)} domain-targeted research queries:")
        for rq in research_queries:
            self._log(state, f"  - [{rq.get('domain', 'auto')}] '{rq.get('query')}' ({rq.get('rationale', '')})")

        # 2. Multi-Tiered Tool Execution Pass
        all_snippets: list[dict[str, str]] = []
        for rq in research_queries:
            q = rq.get("query", "")
            domain = rq.get("domain", "auto")
            self._log(state, f"Searching knowledge ({domain}) for: '{q}'")
            results = search_knowledge(
                query=q,
                domain=domain,
                max_results=self.max_results,
                client=self.client,
                model=self.model,
            )
            all_snippets.extend(results)

        state.knowledge_context = KnowledgeContext(snippets=all_snippets)
        self._log(state, f"Retrieved {len(all_snippets)} verified knowledge/news snippets across {len(research_queries)} topics.")
        return state


# Backward-compatibility alias
NewsRetrieverAgent = KnowledgeRetrieverAgent


# ---------------------------------------------------------------------------
# Helpers & Prompt Loaders
# ---------------------------------------------------------------------------

_PROMPTS_DIR = Path(__file__).resolve().parent.parent.parent / "prompts"


def _load_prompt(filename: str, default: str) -> str:
    path = _PROMPTS_DIR / filename
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    return default.strip()


def _parse_research_queries(raw: str) -> list[dict[str, str]]:
    """
    Parse JSON queries from LLM response.
    Supports list of dicts (with query, domain, rationale) or list of strings.
    """
    try:
        data = extract_json(raw)
        items = data if isinstance(data, list) else data.get("queries", data.get("search_queries", []))
        parsed: list[dict[str, str]] = []
        for item in items:
            if isinstance(item, dict):
                q = str(item.get("query") or "").strip()
                d = str(item.get("domain") or "auto").strip().lower()
                r = str(item.get("rationale") or "").strip()
                if q:
                    parsed.append({"query": q, "domain": d, "rationale": r})
            elif isinstance(item, str) and item.strip():
                parsed.append({"query": item.strip(), "domain": "auto", "rationale": ""})
        return parsed
    except Exception:
        return []


def _parse_queries(raw: str) -> list[str]:
    """Backwards-compatible helper returning query strings."""
    parsed = _parse_research_queries(raw)
    return [p["query"] for p in parsed]


_TARGETED_RESEARCH_SYSTEM_PROMPT_DEFAULT = """\
You are an expert investigative researcher and fact-checker on a high-performing video production team.
The Director has selected a specific winning story pitch and issued production notes.
Your task is to formulate 3 precise, targeted search queries to retrieve verifiable real-world facts, quotes, controversy details, or technical mechanisms to back up this exact story concept.

Guidelines:
- Align directly with the Director's instructions and the winning concept.
- Select the best target domain for each query:
  * "news": for breaking events, current 2026/recent controversies, or active developments.
  * "encyclopedia": for pop-culture lore, cartoon history, banned episode records, origins, or cultural anecdotes.
  * "technical": for scientific mechanisms, math paradoxes, engineering data, or exact numbers.
- Formulate queries with concrete search keywords (e.g., "Dexter's Laboratory Rude Removal banned episode 1998", "paper folding exponential growth 42 times thickness").
- Target exact evidence: real names, broadcast dates, quotes, statistics, or counterintuitive mechanisms.

Return ONLY valid JSON in this format:
{
  "queries": [
    {
      "query": "precise search query",
      "domain": "news | encyclopedia | technical",
      "rationale": "why this verified fact or evidence is needed"
    }
  ]
}
"""

_TARGETED_RESEARCH_SYSTEM_PROMPT = _load_prompt("news_targeted_research.txt", _TARGETED_RESEARCH_SYSTEM_PROMPT_DEFAULT)


_NEWS_BRAINSTORM_SYSTEM_PROMPT = """\
You are an investigative pop-culture and storytelling researcher gathering wild, true real-world stories and facts.
Your goal is to dig up mindblowing external information (bizarre historical anecdotes, shocking real-world controversies, counterintuitive discoveries, or incredible true stories) related to the user's interests.

Guidelines:
- Formulate 3 search queries designed to find fascinating, real-world ammunition for an entertaining story.
- Choose domain: "news", "encyclopedia", or "technical".
- Target queries like: "the insane true story behind [topic]", "the bizarre 2026 controversy with [topic]", "the unbelievable origin of [topic]", or "how [topic] actually happened in real life".
- Avoid dry academic search terms. Search for real stories and facts that would shock or captivate an audience in 60 seconds.

Return ONLY valid JSON in this format:
{
  "queries": [
    {
      "query": "precise search query",
      "domain": "news | encyclopedia | technical",
      "rationale": "why this verified fact or evidence is needed"
    }
  ]
}
"""
