"""
evaluation/evaluator.py — LLM-as-a-judge scoring and comparison module.

Scores storyboards across four standard NLP dimensions (0.0 to 10.0 scale):
  1. Relevance (Personalization): How accurately the storyboard matches the user's inferred interests.
  2. Completeness (Structure): Whether scene durations, subtitles, audio cues, and arcs are intact.
  3. Creativity (Storytelling): Originality, novelty, 3-second hook power, and narrative escalation.
  4. Visual-Narrative Alignment: How faithfully scene descriptions and imagery reflect the story.

Usage:
  # Score a single storyboard
  python3 evaluation/evaluator.py --storyboard outputs/storyboards/storyboard_xyz.json

  # Compare a Multi-Agent storyboard against a Baseline storyboard side-by-side
  python3 evaluation/evaluator.py --compare outputs/storyboards/storyboard_ma.json outputs/storyboards/storyboard_base.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Ensure project root is in sys.path when invoked directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from openai import OpenAI
from src.core.llm_client import get_client, chat_completion
from src.core.state import SharedState


def score_storyboard(
    state_or_dict: SharedState | dict[str, Any],
    judge_model: str = "deepseek-chat",
    client: OpenAI | None = None,
) -> dict[str, Any]:
    """
    Use an LLM judge to evaluate a storyboard based on a rigorous 4-criteria rubric.

    Args:
        state_or_dict: SharedState instance or dict deserialized from state.json.
        judge_model:   Model name to use as judge (default: deepseek-chat).
        client:        OpenAI-compatible client.

    Returns:
        Dict with keys:
          - "relevance": float [0.0 - 10.0]
          - "completeness": float [0.0 - 10.0]
          - "creativity": float [0.0 - 10.0]
          - "visual_alignment": float [0.0 - 10.0]
          - "mean": float [0.0 - 10.0]
          - "critique": str
    """
    if client is None:
        client = get_client()

    data = state_or_dict.to_dict() if isinstance(state_or_dict, SharedState) else state_or_dict

    history_videos = data.get("user_history", {}).get("videos", [])
    history_summary = "\n".join(
        f"- {v.get('title', '')}" for v in history_videos[:20]
    ) or "None provided"

    profile = data.get("user_profile", {})
    interests = ", ".join(profile.get("interests", [])) or "Inferred from watch history"

    storyboard = data.get("storyboard", {})
    title = storyboard.get("title", "Untitled")
    narrative = storyboard.get("overall_narrative", "")
    scenes = storyboard.get("scenes", [])

    scenes_text = "\n".join(
        f"Scene {s.get('scene_id', i+1)} ({s.get('duration_seconds', '?')}s):\n"
        f"  Visual: {s.get('description', '')}\n"
        f"  Subtitle: \"{s.get('subtitle', '')}\"\n"
        f"  Audio: {s.get('audio_cue', '')}\n"
        f"  Image: {s.get('image_url') or s.get('image_ref') or 'None'}"
        for i, s in enumerate(scenes)
    )

    user_prompt = f"""\
USER WATCH HISTORY SAMPLE:
{history_summary}

INFERRED INTERESTS:
{interests}

EVALUATED STORYBOARD:
Title: {title}
Overall Narrative: {narrative}
Scenes:
{scenes_text}

Score this storyboard on all 4 dimensions according to the rubric.
"""
    raw = chat_completion(
        client=client,
        model=judge_model,
        system_prompt=_JUDGE_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        temperature=0.2,
        max_tokens=800,
    )

    return _parse_judge_response(raw)


def compare_storyboards(
    storyboard_a_path: str | Path,
    storyboard_b_path: str | Path,
    judge_model: str = "deepseek-chat",
    client: OpenAI | None = None,
) -> dict[str, Any]:
    """
    Score two storyboards and compare their metrics side-by-side.
    """
    with open(storyboard_a_path, "r", encoding="utf-8") as f:
        data_a = json.load(f)
    with open(storyboard_b_path, "r", encoding="utf-8") as f:
        data_b = json.load(f)

    score_a = score_storyboard(data_a, judge_model=judge_model, client=client)
    score_b = score_storyboard(data_b, judge_model=judge_model, client=client)

    return {
        "storyboard_a": {"path": str(storyboard_a_path), "scores": score_a},
        "storyboard_b": {"path": str(storyboard_b_path), "scores": score_b},
        "winner": "A" if score_a.get("mean", 0) >= score_b.get("mean", 0) else "B",
    }


def _parse_judge_response(raw: str) -> dict[str, Any]:
    try:
        cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        data = json.loads(cleaned)
        rel = float(data.get("relevance", 0.0))
        comp = float(data.get("completeness", 0.0))
        creat = float(data.get("creativity", 0.0))
        vis = float(data.get("visual_alignment", 0.0))
        mean_score = round((rel + comp + creat + vis) / 4.0, 2)
        return {
            "relevance": rel,
            "completeness": comp,
            "creativity": creat,
            "visual_alignment": vis,
            "mean": mean_score,
            "critique": data.get("critique", ""),
        }
    except Exception:
        return {
            "relevance": 5.0,
            "completeness": 5.0,
            "creativity": 5.0,
            "visual_alignment": 5.0,
            "mean": 5.0,
            "critique": "Failed to parse judge JSON.",
        }


_JUDGE_SYSTEM_PROMPT = """\
You are an impartial academic NLP evaluation judge for personalized short-form video storyboarding (COMP 4221).
You will evaluate a generated 60-second video storyboard against the user's watch history and interests.

Evaluate strictly on these 4 dimensions (0.0 to 10.0 scale):
1. Relevance (Personalization):
   - 10: Deeply connects to specific niche curiosities, subcultures, or spoken details from history.
   - 5: Generic connection to broad keywords (e.g. general 'math' or 'cartoons').
   - 0: Completely irrelevant or off-topic.
2. Completeness (Format & Coherence):
   - 10: Seamless 60s structure, 4-6 distinct scenes, vivid visuals, punchy subtitles (≤12 words), audio cues.
   - 5: Missing audio cues or choppy pacing.
   - 0: Broken JSON or incoherent fragments.
3. Creativity & Hook:
   - 10: Gripping 3-second curiosity hook, surprising angle/revelation, high entertainment value.
   - 5: Predictable, standard textbook summary.
   - 0: Extremely boring, clichéd, or nonsensical mashup.
4. Visual-Narrative Alignment:
   - 10: Descriptions and image assets concretely reflect the narrative; visuals actually depict what is said.
   - 5: Abstract or loosely related images.
   - 0: No visuals or contradictory images.

Return ONLY valid JSON with this exact structure:
{
  "relevance": 8.5,
  "completeness": 9.0,
  "creativity": 8.0,
  "visual_alignment": 8.5,
  "critique": "Concise 2-sentence rationale explaining the strengths and weaknesses."
}
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate storyboards using LLM-as-a-judge.")
    parser.add_argument("--storyboard", type=str, help="Path to a single storyboard JSON file to score.")
    parser.add_argument("--compare", nargs=2, metavar=("A", "B"), help="Compare two storyboard JSON files.")
    parser.add_argument("--model", type=str, default="deepseek-chat", help="Judge model name.")
    args = parser.parse_args()

    client = get_client()

    if args.storyboard:
        path = Path(args.storyboard)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        scores = score_storyboard(data, judge_model=args.model, client=client)
        print("\n================ EVALUATION RESULTS ================")
        print(f"File: {path.name}")
        print(f"Relevance:           {scores['relevance']:.1f} / 10.0")
        print(f"Completeness:        {scores['completeness']:.1f} / 10.0")
        print(f"Creativity:          {scores['creativity']:.1f} / 10.0")
        print(f"Visual Alignment:    {scores['visual_alignment']:.1f} / 10.0")
        print(f"MEAN SCORE:          {scores['mean']:.2f} / 10.0")
        print(f"Critique:            {scores['critique']}")
        print("===================================================\n")

    elif args.compare:
        res = compare_storyboards(args.compare[0], args.compare[1], judge_model=args.model, client=client)
        sa = res["storyboard_a"]["scores"]
        sb = res["storyboard_b"]["scores"]
        print("\n================ SIDE-BY-SIDE COMPARISON ================")
        print(f"{'Dimension':<25} | {'Storyboard A':<15} | {'Storyboard B':<15}")
        print("-" * 62)
        print(f"{'Relevance':<25} | {sa['relevance']:<15.1f} | {sb['relevance']:<15.1f}")
        print(f"{'Completeness':<25} | {sa['completeness']:<15.1f} | {sb['completeness']:<15.1f}")
        print(f"{'Creativity':<25} | {sa['creativity']:<15.1f} | {sb['creativity']:<15.1f}")
        print(f"{'Visual Alignment':<25} | {sa['visual_alignment']:<15.1f} | {sb['visual_alignment']:<15.1f}")
        print("-" * 62)
        print(f"{'MEAN OVERALL':<25} | {sa['mean']:<15.2f} | {sb['mean']:<15.2f}")
        print(f"\nWINNER: Storyboard {res['winner']}")
        print(f"Critique A: {sa['critique']}")
        print(f"Critique B: {sb['critique']}")
        print("=========================================================\n")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
