"""
pitch_agent.py — Backward-compatibility shim.

The canonical agent is now WritersRoomAgent in `src/agents/writers_room_agent.py`.
This module re-exports WritersRoomAgent and its helpers to preserve compatibility
with older scripts and tests.
"""

from src.agents.writers_room_agent import (
    WritersRoomAgent,
    _build_writers_room_prompt,
    _parse_pitches,
    _WRITERS_ROOM_SYSTEM_PROMPT,
)

__all__ = [
    "WritersRoomAgent",
    "_build_writers_room_prompt",
    "_parse_pitches",
    "_WRITERS_ROOM_SYSTEM_PROMPT",
]
