"""
Orchestrator — controls the multi-agent pipeline.

The Orchestrator is the single entry point for running the full pipeline.
It:
  1. Initialises a SharedState from raw user input.
  2. Calls each agent in sequence (pipeline pattern for MVP).
  3. Saves the resulting storyboard to the outputs/ directory.

Pipeline order (MVP):
  ProfilerAgent → NewsRetrieverAgent → ImageSearchAgent → ScriptwriterAgent

Design note:
  The pipeline pattern was chosen for the MVP because it is simple, cheap
  (linear API calls), and easy to debug. A debate/critique loop can be
  added later as an ablation experiment for the final report.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from pathlib import Path

from src.agents.image_search_agent import ImageSearchAgent
from src.agents.news_retriever_agent import NewsRetrieverAgent
from src.agents.profiler_agent import ProfilerAgent
from src.agents.scriptwriter_agent import ScriptwriterAgent
from src.core.llm_client import get_client
from src.core.state import SharedState, UserHistory


class Orchestrator:
    """
    Manages the sequential multi-agent storyboard generation pipeline.

    Usage:
        orchestrator = Orchestrator.from_config("configs/config.yaml")
        storyboard = orchestrator.run(videos=[{"title": "...", "url": "..."}])
    """

    def __init__(
        self,
        profiler_model: str,
        retriever_model: str,
        image_model: str,
        scriptwriter_model: str,
        base_url: str | None = None,
        api_key: str | None = None,
        output_dir: str = "outputs/storyboards",
    ) -> None:
        # Shared LLM client (all agents use the same DeepSeek-compatible client)
        client = get_client(base_url=base_url, api_key=api_key)

        # Instantiate all agents
        self.agents = [
            ProfilerAgent(model=profiler_model, client=client),
            NewsRetrieverAgent(model=retriever_model, client=client),
            ImageSearchAgent(model=image_model, client=client),
            ScriptwriterAgent(model=scriptwriter_model, client=client),
        ]
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_config(cls, config_path: str) -> "Orchestrator":
        """
        Instantiate an Orchestrator from a YAML config file.
        See configs/config.yaml for the expected format.
        """
        import yaml  # optional dependency, only needed here

        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)

        return cls(
            profiler_model=cfg["models"]["profiler"],
            retriever_model=cfg["models"]["retriever"],
            image_model=cfg["models"]["image_search"],
            scriptwriter_model=cfg["models"]["scriptwriter"],
            base_url=cfg.get("llm", {}).get("base_url"),
            api_key=cfg.get("llm", {}).get("api_key"),
            output_dir=cfg.get("output_dir", "outputs/storyboards"),
        )

    def run(self, videos: list[dict[str, str]]) -> SharedState:
        """
        Run the full pipeline for a given user watch history.

        Args:
            videos: List of {"title": str, "url": str} dicts.

        Returns:
            The fully populated SharedState (with storyboard).
        """
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + str(uuid.uuid4())[:8]

        # Initialise shared state
        state = SharedState(
            user_history=UserHistory(videos=videos),
            run_id=run_id,
        )
        state.log(f"Pipeline started. run_id={run_id}. Videos in history: {len(videos)}.")

        # Run each agent sequentially
        for agent in self.agents:
            state.log(f"--- Running {agent.name} ---")
            state = agent.run(state)

        state.log("Pipeline complete.")

        # Save output
        out_path = self.output_dir / f"storyboard_{run_id}.json"
        state.save(str(out_path))
        state.log(f"Output saved to {out_path}.")

        return state
