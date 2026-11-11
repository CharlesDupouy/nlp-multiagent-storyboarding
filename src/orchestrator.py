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

import time
from datetime import datetime
from pathlib import Path

from src.agents.art_director_agent import ArtDirectorAgent, ImageSearchAgent
from src.agents.director_agent import DirectorAgent
from src.agents.knowledge_retriever_agent import KnowledgeRetrieverAgent, NewsRetrieverAgent
from src.agents.profiler_agent import ProfilerAgent
from src.agents.scriptwriter_agent import ScriptwriterAgent
from src.agents.writers_room_agent import WritersRoomAgent
from src.core.llm_client import (
    get_client,
    reset_token_usage,
    get_token_usage,
    get_session_cost_usd,
    set_reasoning_effort,
    get_reasoning_effort,
)
from src.core.state import SharedState, UserHistory


class Orchestrator:
    """
    Manages the sequential multi-agent storyboard generation pipeline:
      Phase 1: ProfilerAgent (2-round with spoken transcripts)
      Phase 2: WritersRoomAgent (3 distinct candidate pitches adapting to user vibe)
      Phase 3: DirectorAgent (Showrunner critiques pitches & selects winner)
      Phase 4: KnowledgeRetrieverAgent (Targeted news, lore & fact retrieval for winning pitch)
      Phase 5: ScriptwriterAgent (Draft Pass based on winning pitch & director notes)
      Phase 6: ArtDirectorAgent (Physical query formulation & candidate semantic re-ranking)
      Phase 7: ScriptwriterAgent (Refine Pass grounded to curated visual assets)
    """

    def __init__(
        self,
        profiler_model: str,
        retriever_model: str,
        image_model: str,
        scriptwriter_model: str,
        writers_room_model: str | None = None,
        director_model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        reasoning_effort: str | None = "low",
        output_dir: str = "outputs",
        target_duration: int = 60,
        max_news_results: int = 5,
    ) -> None:
        self.reasoning_effort = reasoning_effort
        if reasoning_effort is not None:
            set_reasoning_effort(reasoning_effort)

        client = get_client(base_url=base_url, api_key=api_key)

        self.profiler = ProfilerAgent(model=profiler_model, client=client)
        self.writers_room = WritersRoomAgent(model=writers_room_model or profiler_model, client=client)
        self.director = DirectorAgent(model=director_model or scriptwriter_model, client=client)
        self.knowledge_retriever = KnowledgeRetrieverAgent(
            model=retriever_model,
            client=client,
            max_results=max_news_results,
        )
        self.news_retriever = self.knowledge_retriever  # Backward-compatibility alias
        self.art_director = ArtDirectorAgent(model=image_model, client=client)
        self.image_searcher = self.art_director         # Backward-compatibility alias
        self.scriptwriter = ScriptwriterAgent(
            model=scriptwriter_model,
            client=client,
            target_duration=target_duration,
        )

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

        models = cfg.get("models", {})
        llm_cfg = cfg.get("llm", {})
        pipeline_cfg = cfg.get("pipeline", {})
        return cls(
            profiler_model=models.get("profiler", "deepseek-chat"),
            writers_room_model=models.get("writers_room", "deepseek-chat"),
            director_model=models.get("director", "deepseek-chat"),
            retriever_model=models.get("knowledge_retriever") or models.get("retriever", "deepseek-chat"),
            image_model=models.get("art_director") or models.get("image_search", "deepseek-chat"),
            scriptwriter_model=models.get("scriptwriter", "deepseek-chat"),
            base_url=llm_cfg.get("base_url"),
            api_key=llm_cfg.get("api_key"),
            reasoning_effort=llm_cfg.get("reasoning_effort", "low"),
            output_dir=cfg.get("output_dir", "outputs"),
            target_duration=pipeline_cfg.get("target_duration_seconds", 60),
            max_news_results=pipeline_cfg.get("max_news_results", 5),
        )

    def run(self, videos: list[dict[str, str]], user_id: str | None = None) -> SharedState:
        """
        Run the full multi-agent pipeline for a given user watch history.

        Args:
            videos:  List of {"title": str, "url": str} dicts.
            user_id: Optional username or identifier for output grouping (default: 'default_user').

        Returns:
            The fully populated SharedState (with storyboard).
        """
        uid = (user_id or "default_user").strip()
        if self.reasoning_effort is not None:
            set_reasoning_effort(self.reasoning_effort)
        reset_token_usage()
        t_pipeline_start = time.time()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"agentic_{uid}_{timestamp}"

        # Initialise shared state
        state = SharedState(
            user_id=uid,
            user_history=UserHistory(videos=videos),
            run_id=run_id,
        )
        eff = get_reasoning_effort() or "default"
        state.log(
            f"Pipeline started. user={uid}, run_id={run_id}. "
            f"Videos in history: {len(videos)}. reasoning_effort={eff}."
        )

        phases = [
            ("User Profiling (ProfilerAgent)", self.profiler.model, self.profiler.run),
            ("Writers' Room Ideation (WritersRoomAgent)", self.writers_room.model, self.writers_room.run),
            ("Pitch Deliberation & Selection (DirectorAgent)", self.director.model, self.director.run),
            ("Targeted Knowledge & Fact Retrieval (KnowledgeRetrieverAgent)", self.knowledge_retriever.model, self.knowledge_retriever.run),
            ("Storyboard Drafting (ScriptwriterAgent - Draft Pass)", self.scriptwriter.model, self.scriptwriter.draft),
            ("Visual Grounding & Asset Curation (ArtDirectorAgent)", self.art_director.model, self.art_director.run),
            ("Storyboard Grounding & Refinement (ScriptwriterAgent - Refine Pass)", self.scriptwriter.model, self.scriptwriter.refine),
        ]

        total_phases = len(phases)
        for idx, (label, model_name, step_fn) in enumerate(phases, 1):
            print(f"\n--- [Phase {idx}/{total_phases}] {label} [{model_name}] ---", flush=True)
            state.log(f"--- Phase {idx}: {label} ---")
            t_phase = time.time()
            state = step_fn(state)
            print(f"--- ✓ Phase {idx}/{total_phases} done in {time.time() - t_phase:.1f}s ---", flush=True)

        total_elapsed = time.time() - t_pipeline_start

        # Audit token accounting
        usage = get_token_usage()
        cost = get_session_cost_usd(
            fallback_model=self.scriptwriter.model if usage.get("reasoning_tokens", 0) > 0 else None,
        )
        state.token_usage = usage
        state.estimated_cost_usd = cost
        r_tok = usage.get("reasoning_tokens", 0)
        r_str = f" (incl. {r_tok:,} reasoning)" if r_tok else ""
        state.log(
            f"Token audit: {usage.get('prompt_tokens', 0):,} prompt + "
            f"{usage.get('completion_tokens', 0):,} completion{r_str} across "
            f"{usage.get('total_calls', 0)} calls in {total_elapsed:.1f}s. Est. Cost: ${cost:.5f} USD."
        )
        state.log("Pipeline complete.")

        # Save output in outputs/<user>/agentic/agentic_storyboard_<user>_<datetime>.json
        base_dir = self.output_dir.parent if self.output_dir.name in ("storyboards", "agentic", "baseline") else self.output_dir
        agentic_dir = base_dir / uid / "agentic"
        out_path = agentic_dir / f"agentic_storyboard_{uid}_{timestamp}.json"
        state.save(str(out_path))
        setattr(state, "output_json_path", str(out_path))
        state.log(f"Output saved to {out_path}.")

        return state
