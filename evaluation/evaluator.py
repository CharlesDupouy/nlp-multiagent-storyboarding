"""
Evaluation module — LLM-as-a-judge scoring.

Scores generated storyboards on three dimensions:
  1. Relevance   — how well the storyboard matches the user's profile
  2. Completeness — whether all required fields are present and coherent
  3. Creativity  — originality and diversity of narrative choices

Owner: Max

TODO:
  - Implement score_storyboard()
  - Implement run_automated_eval() over the public dataset
  - Implement blind A/B comparison helper for human eval
"""

from __future__ import annotations

from src.core.state import SharedState


def score_storyboard(state: SharedState, judge_model: str, client) -> dict[str, float]:
    """
    Use an LLM judge to score a storyboard against the user profile.

    Args:
        state:       SharedState containing both user_profile and storyboard.
        judge_model: Model name to use as the judge (e.g. "deepseek-chat").
        client:      OpenAI-compatible client.

    Returns:
        Dict with keys: "relevance", "completeness", "creativity", "mean"
        Each score is in [0.0, 10.0].

    TODO: implement
    """
    raise NotImplementedError("LLM-as-a-judge evaluation not yet implemented.")


def run_automated_eval(
    dataset_path: str,
    system_runner,  # Orchestrator or SingleAgentBaseline instance
    judge_model: str,
    client,
    output_path: str,
) -> None:
    """
    Run end-to-end automated evaluation over a dataset of user histories.

    Args:
        dataset_path:  Path to a JSONL file of user histories.
        system_runner: Orchestrator or SingleAgentBaseline with a .run() method.
        judge_model:   LLM to use as the judge.
        client:        OpenAI-compatible client.
        output_path:   Where to save the scored results (CSV or JSONL).

    TODO: implement
    """
    raise NotImplementedError("Automated evaluation not yet implemented.")
