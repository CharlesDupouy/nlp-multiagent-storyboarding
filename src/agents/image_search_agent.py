"""
image_search_agent.py — Backward-compatibility shim.

The canonical agent is now ArtDirectorAgent in `src/agents/art_director_agent.py`,
reflecting its dual role in photographic query sanitization and multi-candidate
semantic re-ranking.
"""

from src.agents.art_director_agent import (
    ArtDirectorAgent,
    ImageSearchAgent,
    _build_art_director_prompt,
    _parse_scene_queries,
    _parse_art_director_selections,
    _IMAGE_QUERY_SYSTEM_PROMPT,
    _ART_DIRECTOR_SYSTEM_PROMPT,
)

__all__ = [
    "ArtDirectorAgent",
    "ImageSearchAgent",
    "_build_art_director_prompt",
    "_parse_scene_queries",
    "_parse_art_director_selections",
    "_IMAGE_QUERY_SYSTEM_PROMPT",
    "_ART_DIRECTOR_SYSTEM_PROMPT",
]
