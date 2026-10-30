"""
BaseAgent — abstract base class for all agents in the pipeline.

Every concrete agent must implement `run(state: SharedState) -> SharedState`.
This enforces a uniform interface so the Orchestrator can call agents
interchangeably and evaluation can swap them in/out easily.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from openai import OpenAI

from src.core.state import SharedState


class BaseAgent(ABC):
    """
    Abstract base for all pipeline agents.

    Attributes:
        name:   Human-readable agent name (used in logs and reports).
        model:  LLM model string to use (e.g. "deepseek-chat").
        client: Initialised OpenAI-compatible client.
    """

    def __init__(self, name: str, model: str, client: OpenAI) -> None:
        self.name = name
        self.model = model
        self.client = client

    @abstractmethod
    def run(self, state: SharedState) -> SharedState:
        """
        Execute this agent's logic.

        Reads relevant fields from `state`, performs its task
        (LLM calls, tool calls, etc.), writes results back into `state`,
        and returns the updated state.
        """
        ...

    def _log(self, state: SharedState, message: str) -> None:
        """Convenience wrapper to write to the shared pipeline log."""
        state.log(f"[{self.name}] {message}")
