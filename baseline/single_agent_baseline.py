"""
Single-Agent Baseline — zero-shot storyboard generation.

This baseline replicates the exact input/output contract of the
multi-agent pipeline (SharedState in → SharedState out) so that
evaluation code can compare both systems identically.

Instead of a pipeline of specialized agents, a single LLM call
receives the raw watch history and produces the storyboard directly.
This isolates the contribution of multi-agent brainstorming and
retrieval augmentation.

Owner: Antoine
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path

from src.core.llm_client import chat_completion, get_client
from src.core.state import SharedState, Storyboard, UserHistory


class SingleAgentBaseline:
    """
    Zero-shot single-agent storyboard generator.

    Mirrors the Orchestrator interface so evaluation code treats
    both systems identically.
    """

    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
        output_dir: str = "outputs/storyboards",
        target_duration: int = 60,
    ) -> None:
        self.model = model
        self.client = get_client(base_url=base_url, api_key=api_key)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.target_duration = target_duration

    @classmethod
    def from_config(cls, config_path: str) -> "SingleAgentBaseline":
        import yaml

        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        return cls(
            model=cfg["baseline"]["model"],
            base_url=cfg.get("llm", {}).get("base_url"),
            api_key=cfg.get("llm", {}).get("api_key"),
            output_dir=cfg.get("output_dir", "outputs/storyboards"),
        )

    def run(self, videos: list[dict[str, str]]) -> SharedState:
        """
        Generate a storyboard from raw watch history in a single LLM call.

        Same signature as Orchestrator.run() for easy swapping in evaluation.
        """
        run_id = "baseline_" + datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + str(uuid.uuid4())[:8]
        state = SharedState(
            user_history=UserHistory(videos=videos),
            run_id=run_id,
        )
        state.log(f"[BASELINE] Single-agent run started. run_id={run_id}.")

        history_text = "\n".join(
            f"- [vid{i+1}] Title: \"{v.get('title', '')}\" | URL: {v.get('url', '')}"
            for i, v in enumerate(videos)
        )
        user_prompt = (
            f"Here is a user's YouTube Shorts watch history:\n\n{history_text}\n\n"
            f"Generate a personalized short-video storyboard (target: {self.target_duration}s) "
            f"tailored to this user's interests."
        )

        raw = chat_completion(
            client=self.client,
            model=self.model,
            system_prompt=_BASELINE_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=0.8,
            max_tokens=3000,
        )

        storyboard_data = _parse(raw)
        state.storyboard = Storyboard(
            title=storyboard_data.get("title", "Untitled"),
            target_duration_seconds=self.target_duration,
            scenes=storyboard_data.get("scenes", []),
            overall_narrative=storyboard_data.get("overall_narrative", ""),
        )

        state.log(f"[BASELINE] Storyboard generated: '{state.storyboard.title}'.")
        out_path = self.output_dir / f"storyboard_{run_id}.json"
        state.save(str(out_path))
        return state


def _parse(raw: str) -> dict:
    try:
        cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return {"overall_narrative": raw, "scenes": [], "title": "Parse Error"}


_BASELINE_SYSTEM_PROMPT = """\
You are a creative video scriptwriter specializing in short-form content.
Given a user's YouTube Shorts watch history, produce a personalized
short-video storyboard in a SINGLE pass.

Return ONLY valid JSON with this structure:
{
  "title": "Catchy video title",
  "overall_narrative": "2-3 sentence arc of the video",
  "scenes": [
    {
      "scene_id": 1,
      "description": "Visual description",
      "subtitle": "On-screen text (max 15 words)",
      "audio_cue": "Voiceover or music note",
      "image_url": null,
      "duration_seconds": 10
    }
  ]
}

Guidelines:
- Aim for 4-7 scenes summing to the target duration.
- Base the narrative on the user's actual watch history.
- Be creative and specific, not generic.
"""
