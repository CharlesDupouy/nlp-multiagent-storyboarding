"""
news_tool.py — Tool wrapper for news and external knowledge retrieval.

Backwards-compatible interface forwarding to knowledge_tool.py.
Provides search_news(query, max_results) and search_knowledge(...)
with multi-tiered retrieval (NewsAPI, Wikipedia, Tavily, and LLM Tier 3 Synthesis).
"""

from __future__ import annotations

from src.tools.knowledge_tool import (
    search_news,
    search_knowledge,
    _newsapi_search,
    _serper_search,
    _wikipedia_search,
    _tavily_search,
    _llm_synthesize_facts,
    _mock_fallback_facts,
)

__all__ = [
    "search_news",
    "search_knowledge",
    "_newsapi_search",
    "_serper_search",
    "_wikipedia_search",
    "_tavily_search",
    "_llm_synthesize_facts",
    "_mock_fallback_facts",
]
