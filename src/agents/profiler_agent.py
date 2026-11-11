"""
ProfilerAgent — Two-Round Deep User Profiling Agent.

Responsibilities:
  - Round 1 (Macro Scan & Hypothesis Generation):
    Scans the complete watch history (titles + channels).
    Identifies broad interest clusters and actively selects 3-5 specific videos
    with ambiguous or rich content to investigate deeper.
  - Tool Execution:
    Calls `get_video_transcript` and `get_video_metadata` to retrieve the actual spoken
    dialogue and creator metadata for the chosen videos (zero API key needed).
  - Round 2 (Fine-Grained Deep Profiling):
    Synthesizes the macro history WITH the actual spoken words and nuances from the
    transcripts into a fine-grained, concrete user profile.

Output written to state:
  - state.user_profile  (UserProfile dataclass with investigated_insights)
"""

from __future__ import annotations

import json
from typing import Any
from openai import OpenAI

from src.agents.base_agent import BaseAgent
from src.core.llm_client import chat_completion, extract_json, load_prompt
from src.core.state import SharedState, UserProfile
from src.tools.youtube_tool import get_video_transcript, get_video_metadata


class ProfilerAgent(BaseAgent):
    """
    Two-round profiling agent:
    1. Scans watch history and actively picks videos for deep transcript inspection.
    2. Uses spoken dialogue to build a fine-grained, nuanced user profile.
    """

    def __init__(self, model: str, client: OpenAI, max_investigate: int = 4) -> None:
        super().__init__(name="ProfilerAgent", model=model, client=client)
        self.max_investigate = max_investigate

    def run(self, state: SharedState) -> SharedState:
        videos = state.user_history.videos
        if not videos:
            self._log(state, "No videos in history. Skipping profiling.")
            return state

        # Create indexed map: vid1 -> video dict
        vid_map: dict[str, dict[str, str]] = {
            f"vid{i+1}": v for i, v in enumerate(videos)
        }

        # -------------------------------------------------------------------
        # ROUND 1: Macro Scan & Targeted Video Selection
        # -------------------------------------------------------------------
        self._log(state, f"Starting Round 1: Macro scan across {len(videos)} watch history items.")

        history_lines = [
            f"- [{vid_id}] Title: \"{v.get('title', '')}\""
            for vid_id, v in vid_map.items()
        ]
        history_text = "\n".join(history_lines)

        user_prompt_r1 = f"""\
WATCH HISTORY:
{history_text}

Analyze this user's overall tastes and select up to {self.max_investigate} specific video IDs to inspect their spoken transcripts for deeper understanding.
"""
        raw_r1 = chat_completion(
            client=self.client,
            model=self.model,
            system_prompt=_ROUND1_MACRO_SYSTEM_PROMPT,
            user_prompt=user_prompt_r1,
            temperature=0.4,
            max_tokens=1000,
        )

        r1_data = _parse_json(raw_r1)
        investigate_targets = r1_data.get("investigate_targets", [])
        if not isinstance(investigate_targets, list) or not investigate_targets:
            # Fallback: pick first few videos
            investigate_targets = [{"vid_id": f"vid{i+1}", "reason": "Representative video"} for i in range(min(3, len(videos)))]

        self._log(state, f"Round 1 completed. Selected {len(investigate_targets)} videos for deep transcript inspection.")
        if r1_data.get("preliminary_themes"):
            self._log(state, f"  Preliminary themes: {', '.join(str(t) for t in r1_data.get('preliminary_themes', [])[:5])}")

        # -------------------------------------------------------------------
        # TOOL EXECUTION: Fetch Spoken Transcripts & Metadata
        # -------------------------------------------------------------------
        transcript_evidence: list[dict[str, Any]] = []

        for idx_t, target in enumerate(investigate_targets[:self.max_investigate], 1):
            vid_id = target.get("vid_id", "")
            reason = target.get("reason", "")
            video_info = vid_map.get(vid_id)
            if not video_info:
                continue

            url = video_info.get("url", "")
            title = video_info.get("title", "")
            self._log(state, f"Investigating ({idx_t}/{len(investigate_targets[:self.max_investigate])}) [{vid_id}] '{title[:40]}...' -> Reason: {reason}")

            transcript = get_video_transcript(url, max_chars=1200)
            meta = get_video_metadata(url)
            channel = meta.get("author", "") or "Unknown"
            tr_status = f"✓ Transcript ({len(transcript)} chars)" if transcript else "✗ No subtitles"
            self._log(state, f"  -> {tr_status} | Creator: {channel}")

            transcript_evidence.append({
                "vid_id": vid_id,
                "title": title,
                "channel": meta.get("author", ""),
                "investigation_reason": reason,
                "spoken_transcript": transcript or "(No spoken subtitles detected for this short)",
            })

        self._log(state, f"Retrieved transcript evidence for {len(transcript_evidence)} videos.")

        # -------------------------------------------------------------------
        # ROUND 2: Fine-Grained Deep Profile Synthesis
        # -------------------------------------------------------------------
        self._log(state, "Starting Round 2: Synthesizing fine-grained profile using spoken transcripts.")

        evidence_blocks = []
        for ev in transcript_evidence:
            evidence_blocks.append(
                f"[{ev['vid_id']}] \"{ev['title']}\" (Channel: {ev['channel']})\n"
                f"Investigation Goal: {ev['investigation_reason']}\n"
                f"Actual Spoken Dialogue: \"{ev['spoken_transcript']}\"\n"
            )
        evidence_text = "\n".join(evidence_blocks)

        user_prompt_r2 = f"""\
ROUND 1 PRELIMINARY THEMES:
{json.dumps(r1_data.get('preliminary_themes', []), ensure_ascii=False)}

INVESTIGATED VIDEO SPOKEN TRANSCRIPTS:
{evidence_text}

FULL WATCH HISTORY SUMMARY:
User has watched {len(videos)} videos spanning topics like {', '.join(r1_data.get('preliminary_interests', [])[:6])}.

Synthesize a deeply fine-grained, nuanced UserProfile using the spoken words and specific details uncovered in the transcripts.
"""
        raw_r2 = chat_completion(
            client=self.client,
            model=self.model,
            system_prompt=_ROUND2_SYNTHESIS_SYSTEM_PROMPT,
            user_prompt=user_prompt_r2,
            temperature=0.4,
            max_tokens=1500,
        )

        r2_data = _parse_json(raw_r2)
        state.user_profile = UserProfile(
            interests=r2_data.get("interests", r1_data.get("preliminary_interests", [])),
            themes=r2_data.get("themes", r1_data.get("preliminary_themes", [])),
            preferred_style=r2_data.get("preferred_style", "infotainment"),
            raw_summary=r2_data.get("raw_summary", ""),
            investigated_insights=r2_data.get("investigated_insights", []),
        )

        self._log(
            state,
            f"Round 2 complete. Fine-grained profile established ({len(state.user_profile.interests)} niche interests, "
            f"{len(state.user_profile.investigated_insights)} transcript insights).",
        )
        self._log(state, f"  Preferred style: {state.user_profile.preferred_style}")
        self._log(state, f"  Top interests:   {', '.join(state.user_profile.interests[:6])}")
        if state.user_profile.raw_summary:
            self._log(state, f"  Summary:         {state.user_profile.raw_summary[:140]}...")
        return state


# ---------------------------------------------------------------------------
# Helpers & System Prompts
# ---------------------------------------------------------------------------

def _parse_json(raw: str) -> dict:
    try:
        data = extract_json(raw)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


_ROUND1_MACRO_SYSTEM_PROMPT_DEFAULT = """\
You are an expert analyst of user tastes conducting Round 1 (Macro Scan) of a user's YouTube Shorts history.

Your tasks:
1. Identify high-level interest clusters and preliminary themes.
2. Select 3 to 4 specific video IDs (e.g. "vid4", "vid12") that are ambiguous, surprising, or central, where inspecting the spoken transcript would reveal the exact subject being explored.
3. For each selected video, state the specific reason/hypothesis you want to verify.

Return ONLY valid JSON in this format:
{
  "preliminary_interests": ["broad", "interest", "tags"],
  "preliminary_themes": ["macro", "themes"],
  "investigate_targets": [
    {
      "vid_id": "vid4",
      "reason": "Verify what specific cartoon show and controversy was discussed in the video"
    }
  ]
}
"""

_ROUND1_MACRO_SYSTEM_PROMPT = load_prompt("profiler_r1_macro.txt", _ROUND1_MACRO_SYSTEM_PROMPT_DEFAULT)


_ROUND2_SYNTHESIS_SYSTEM_PROMPT_DEFAULT = """\
You are an expert profiler conducting Round 2 (Fine-Grained Deep Profiling) of a user.
You are provided with:
1. Round 1 preliminary themes.
2. The actual SPOKEN TRANSCRIPTS and dialogue from investigated videos.

Your goal is to transform generic categories into precise, fine-grained niche passions based on what was actually said:
- Use specific terminology, real controversies, creator perspectives, and cultural references revealed in the transcripts.
- Highlight 2-4 concrete insights discovered from the spoken dialogue.
- Remember: the user wants entertaining stories fueled by real facts and information (infotainment).

Return ONLY valid JSON in this format:
{
  "interests": ["fine-grained", "highly specific", "passions with concrete nuance"],
  "themes": ["core thematic pillars"],
  "preferred_style": "infotainment",
  "investigated_insights": [
    "From [vidX] transcript: user engages with [specific nuance/controversy/concept]",
    "From [vidY] transcript: user is fascinated by [specific mechanism]"
  ],
  "raw_summary": "2-3 sentences synthesizing their nuanced personality and what kind of real-world stories would hook them."
}
"""

_ROUND2_SYNTHESIS_SYSTEM_PROMPT = load_prompt("profiler_r2_synthesis.txt", _ROUND2_SYNTHESIS_SYSTEM_PROMPT_DEFAULT)
