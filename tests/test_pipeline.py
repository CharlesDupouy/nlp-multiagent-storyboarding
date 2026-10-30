"""
Basic smoke tests for the pipeline with mock tools.
Run with: pytest tests/
"""

import pytest
from src.core.state import SharedState, UserHistory


def test_shared_state_initialization():
    state = SharedState(
        user_history=UserHistory(videos=[
            {"title": "Test video", "url": "https://youtube.com/shorts/test"}
        ]),
        run_id="test_run",
    )
    assert len(state.user_history.videos) == 1
    assert state.run_id == "test_run"
    assert state.storyboard.title == ""


def test_shared_state_save(tmp_path):
    state = SharedState(run_id="test")
    out = tmp_path / "state.json"
    state.save(str(out))
    assert out.exists()


def test_profiler_agent_mock():
    """
    Smoke test for ProfilerAgent using a mocked LLM response.
    TODO: expand with proper mocking once implementation is done.
    """
    pass  # placeholder


def test_pipeline_end_to_end_mock():
    """
    Full pipeline smoke test using mock tools and a mock LLM.
    TODO: implement with unittest.mock to avoid real API calls.
    """
    pass  # placeholder
