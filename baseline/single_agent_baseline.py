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
import time
import uuid
from datetime import datetime
from pathlib import Path

from src.core.llm_client import (
    chat_completion,
    get_client,
    reset_token_usage,
    get_token_usage,
    calculate_cost_usd,
    extract_json,
    set_reasoning_effort,
)
from src.core.state import SharedState, Storyboard, UserHistory
from src.tools.image_tool import search_images


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
        output_dir: str = "outputs",
        target_duration: int = 60,
        reasoning_effort: str | None = "low",
    ) -> None:
        self.model = model
        self.reasoning_effort = reasoning_effort
        if reasoning_effort is not None:
            set_reasoning_effort(reasoning_effort)
        self.client = get_client(base_url=base_url, api_key=api_key)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.target_duration = target_duration

    @classmethod
    def from_config(cls, config_path: str) -> "SingleAgentBaseline":
        import yaml

        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        llm_cfg = cfg.get("llm", {})
        return cls(
            model=cfg["baseline"]["model"],
            base_url=llm_cfg.get("base_url"),
            api_key=llm_cfg.get("api_key"),
            output_dir=cfg.get("output_dir", "outputs"),
            reasoning_effort=llm_cfg.get("reasoning_effort", "low"),
        )

    def run(self, videos: list[dict[str, str]], user_id: str | None = None) -> SharedState:
        """
        Generate a storyboard from raw watch history in a single LLM call.

        Same signature as Orchestrator.run() for easy swapping in evaluation.
        """
        uid = (user_id or "default_user").strip()
        if self.reasoning_effort is not None:
            set_reasoning_effort(self.reasoning_effort)
        reset_token_usage()
        t_start = time.time()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"baseline_{uid}_{timestamp}"
        state = SharedState(
            user_id=uid,
            user_history=UserHistory(videos=videos),
            run_id=run_id,
        )
        state.log(f"[BASELINE] Single-agent run started. user={uid}, run_id={run_id}. Videos={len(videos)}.")

        history_text = "\n".join(
            f"- [vid{i+1}] Title: \"{v.get('title', '')}\" | URL: {v.get('url', '')}"
            for i, v in enumerate(videos)
        )
        user_prompt = (
            f"Here is a user's YouTube Shorts watch history:\n\n{history_text}\n\n"
            f"Generate a personalized short-video storyboard (target: {self.target_duration}s) "
            f"tailored to this user's interests."
        )

        state.log(f"[BASELINE] Step 1/2: Generating zero-shot storyboard via {self.model}...")
        raw = chat_completion(
            client=self.client,
            model=self.model,
            system_prompt=_BASELINE_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=0.7,
            max_tokens=3000,
        )

        storyboard_data = _parse(raw)
        scenes = storyboard_data.get("scenes", [])
        state.log(f"[BASELINE] Parsed '{storyboard_data.get('title', 'Untitled')}' with {len(scenes)} scenes.")

        # Visual parity: resolve naive top-1 image search for each scene
        state.log(f"[BASELINE] Step 2/2: Fetching top-1 stock photos for {len(scenes)} scenes...")
        for s_idx, s in enumerate(scenes, 1):
            query = s.get("image_query") or s.get("description", "")
            if query:
                state.log(f"[BASELINE]   Scene {s_idx}/{len(scenes)}: searching '{query[:55]}'")
                try:
                    results = search_images(query, per_page=1)
                    if results:
                        s["image_url"] = results[0].get("url")
                        s["image_description"] = results[0].get("description") or results[0].get("alt_text", "")
                    else:
                        s["image_url"] = None
                        s["image_description"] = None
                except Exception as e:
                    state.log(f"[BASELINE] Warning: image retrieval failed for scene {s.get('scene_id')}: {e}")
                    s["image_url"] = None
                    s["image_description"] = None
            else:
                s["image_url"] = None
                s["image_description"] = None

        state.storyboard = Storyboard(
            title=storyboard_data.get("title", "Untitled"),
            target_duration_seconds=self.target_duration,
            scenes=scenes,
            overall_narrative=storyboard_data.get("overall_narrative", ""),
        )
        state.storyboard.sanitize_scenes()

        total_elapsed = time.time() - t_start

        # Audit token accounting
        usage = get_token_usage()
        cost = calculate_cost_usd(usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0))
        state.token_usage = usage
        state.estimated_cost_usd = cost
        state.log(f"[BASELINE] Token audit: {usage.get('prompt_tokens', 0):,} prompt + {usage.get('completion_tokens', 0):,} completion in {total_elapsed:.1f}s. Est. Cost: ${cost:.5f} USD.")
        state.log(f"[BASELINE] Storyboard generated: '{state.storyboard.title}'.")
        
        # Save output in outputs/<user>/baseline/baseline_storyboard_<user>_<datetime>.json
        base_dir = self.output_dir.parent if self.output_dir.name in ("storyboards", "agentic", "baseline") else self.output_dir
        baseline_dir = base_dir / uid / "baseline"
        out_path = baseline_dir / f"baseline_storyboard_{uid}_{timestamp}.json"
        state.save(str(out_path))
        setattr(state, "output_json_path", str(out_path))
        return state


def _parse(raw: str) -> dict:
    try:
        data = extract_json(raw)
        if isinstance(data, dict):
            return data
        return {"overall_narrative": raw, "scenes": [], "title": "Parse Error"}
    except Exception:
        return {"overall_narrative": raw, "scenes": [], "title": "Parse Error"}


_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


def _load_prompt(filename: str, default: str) -> str:
    path = _PROMPTS_DIR / filename
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    return default.strip()


_BASELINE_SYSTEM_PROMPT_DEFAULT = """\
You are an expert short-form video creator and storyteller.
Your task is to generate a high-quality, engaging, personalized 60-second short-video storyboard based on a user's YouTube Shorts watch history.

Guidelines:
- Analyze the user's watch history to understand their core interests, passions, and curiosity themes.
- Select a single, focused, and compelling concept that deeply resonates with those interests (do not create an unfocused mashup of random channels).
- Craft a cohesive 60-second narrative arc with an attention-grabbing hook, clear progression, and a memorable conclusion.
- Scenes: Provide vivid visual descriptions, punchy on-screen subtitles, and pure spoken voiceover dialogue (NO stage directions, NO SFX or music cues).
- Timing: Aim for 4 to 6 scenes summing strictly to the target duration (e.g., 60 seconds).
- Language: Write the script in the primary language of the user's history (e.g. French if the history is French, English if English).

CRITICAL NEGATIVE CONSTRAINTS (VOICEOVER ONLY, ZERO SFX, NO PLACEHOLDERS):
- STRICTLY NO SOUND EFFECTS (SFX) OR MUSIC: Sound effects are NOT supported in this project.
  - FORBIDDEN: NEVER write "SFX: ...", "FX: ...", "(whoosh)", "(dramatic chord)", "(camera click)", or any sound effect descriptions.
- STRICTLY NO STAGE LABELS OR PREFIXES:
  - FORBIDDEN: NEVER write "VO:", "VO (...):", "Voix off:", "Narrator:", "Host:", "Speaker:", etc.
- STRICTLY NO PLACEHOLDER OR EVASIVE DESCRIPTIONS:
  - FORBIDDEN: "A character says an innocent line...", "Zoom on the key quote...", "Explains the double meaning..."
- REQUIRED: The "voiceover" field must contain ONLY the pure verbatim spoken words to be read aloud by Text-To-Speech.
  - EXAMPLE OF CORRECT VOICEOVER: "That warning label isn't for you. It's for about one in twelve thousand people."
  - EXAMPLE OF FORBIDDEN VOICEOVER: "VO (flat): 'That warning label isn't for you.' SFX: turntable hum"

Return ONLY valid JSON with this structure:
{
  "title": "Compelling video title",
  "overall_narrative": "2-3 sentence overview of the video narrative and concept",
  "scenes": [
    {
      "scene_id": 1,
      "description": "Vivid visual scene description of what appears on screen",
      "image_query": "Concrete physical stock photo keywords (e.g., 'white paper folded on wooden desk')",
      "subtitle": "Punchy on-screen caption (max 12 words)",
      "voiceover": "Pure spoken words to be read aloud by voiceover (NO 'VO:', NO SFX, NO music cues)",
      "duration_seconds": 10
    }
  ]
}
"""

_BASELINE_SYSTEM_PROMPT = _load_prompt("baseline.txt", _BASELINE_SYSTEM_PROMPT_DEFAULT)
