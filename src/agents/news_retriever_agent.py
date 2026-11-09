"""
news_retriever_agent.py — Backward-compatibility shim.

The canonical agent is now KnowledgeRetrieverAgent in `src/agents/knowledge_retriever_agent.py`,
reflecting its expanded multi-tiered capabilities across news, Wikipedia encyclopedia lore,
science/technical lookup, and LLM fact synthesis.
"""

from src.agents.knowledge_retriever_agent import (
    KnowledgeRetrieverAgent,
    NewsRetrieverAgent,
    _parse_research_queries,
    _parse_queries,
    _TARGETED_RESEARCH_SYSTEM_PROMPT,
    _NEWS_BRAINSTORM_SYSTEM_PROMPT,
)

__all__ = [
    "KnowledgeRetrieverAgent",
    "NewsRetrieverAgent",
    "_parse_research_queries",
    "_parse_queries",
    "_TARGETED_RESEARCH_SYSTEM_PROMPT",
    "_NEWS_BRAINSTORM_SYSTEM_PROMPT",
]
