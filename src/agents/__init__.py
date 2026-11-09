"""
Agents package for multi-agent personalized short-video storyboarding.

Canonical Agent Pipeline:
  - ProfilerAgent: Two-round watch history profiler with spoken transcripts.
  - WritersRoomAgent: Brainstorms 3 distinct candidate pitches adapted to user vibe.
  - DirectorAgent: Showrunner pitch critique, winning selection & production notes.
  - KnowledgeRetrieverAgent: Multi-tiered factual, lore, science & news retriever.
  - ScriptwriterAgent: 2-pass scriptwriter (draft pass + visual grounding refine pass).
  - ArtDirectorAgent: Physical stock query sanitization & 4-candidate semantic re-ranker.
"""

from src.agents.base_agent import BaseAgent
from src.agents.profiler_agent import ProfilerAgent
from src.agents.writers_room_agent import WritersRoomAgent
from src.agents.director_agent import DirectorAgent
from src.agents.knowledge_retriever_agent import KnowledgeRetrieverAgent, NewsRetrieverAgent
from src.agents.art_director_agent import ArtDirectorAgent, ImageSearchAgent
from src.agents.scriptwriter_agent import ScriptwriterAgent

# Backward-compatibility aliases
from src.agents.pitch_agent import WritersRoomAgent as PitchAgent

__all__ = [
    "BaseAgent",
    "ProfilerAgent",
    "WritersRoomAgent",
    "DirectorAgent",
    "KnowledgeRetrieverAgent",
    "NewsRetrieverAgent",
    "ArtDirectorAgent",
    "ImageSearchAgent",
    "ScriptwriterAgent",
    "PitchAgent",
]
