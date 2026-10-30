"""
Orchestrator — controls the multi-agent pipeline.

The Orchestrator is the single entry point for running the full pipeline.
It:
  1. Initialises a SharedState from raw user input.
  2. Executes agents in a deliberate feedback loop:
     - ProfilerAgent: extracts structured user interests
     - NewsRetrieverAgent: LLM brainstorms & fetches trending news topics
     - ScriptwriterAgent (Draft Pass): writes initial narrative & scene structure
     - ImageSearchAgent: formulates photographic queries per scene and indexes assets (img1, img2...)
     - ScriptwriterAgent (Refinement Pass): aligns scenes with visual assets using symbolic refs
     - Programmatic Resolution: maps symbolic image refs ('img1') to real image URLs
  3. Saves the resulting storyboard and execution trace to outputs/.

Design note:
  This architecture ensures images are fetched to serve specific storyboard scenes
  rather than vague themes, and prevents LLM URL hallucination by decoupling
  creative reference selection from actual asset URL resolution.
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
        client = get_client(base_url=base_url, api_key=api_key)

        self.profiler = ProfilerAgent(model=profiler_model, client=client)
        self.news_retriever = NewsRetrieverAgent(model=retriever_model, client=client)
        self.image_searcher = ImageSearchAgent(model=image_model, client=client)
        self.scriptwriter = ScriptwriterAgent(model=scriptwriter_model, client=client)

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_config(cls, config_path: str) -> "Orchestrator":
        """
        Instantiate an Orchestrator from a YAML config file.
        See configs/config.yaml for the expected format.
        """
        import yaml

        with open(config_path, "r", encoding="utf-8") as f:
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

        # Phase 1: User Profiling
        state.log("--- Phase 1: User Profiling (ProfilerAgent) ---")
        state = self.profiler.run(state)

        # Phase 2: News & Trends Brainstorming + Retrieval
        state.log("--- Phase 2: News Brainstorming & Retrieval (NewsRetrieverAgent) ---")
        state = self.news_retriever.run(state)

        # Phase 3: Initial Storyboard Draft
        state.log("--- Phase 3: Storyboard Drafting (ScriptwriterAgent - Draft Pass) ---")
        state = self.scriptwriter.draft(state)

        # Phase 4: Visual Search for Draft Scenes
        state.log("--- Phase 4: Scene Visual Search (ImageSearchAgent) ---")
        state = self.image_searcher.run(state)

        # Phase 5: Storyboard Refinement & Grounding
        state.log("--- Phase 5: Storyboard Grounding & Refinement (ScriptwriterAgent - Refine Pass) ---")
        state = self.scriptwriter.refine(state)

        state.log("Pipeline complete.")

        # Save output
        out_path = self.output_dir / f"storyboard_{run_id}.json"
        state.save(str(out_path))
        state.log(f"Output saved to {out_path}.")

        return state
