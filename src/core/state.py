"""
SharedState — the single in-memory blackboard passed through the pipeline.

Every agent reads from and writes to this object. The Orchestrator
initializes it with raw user input; each agent enriches it with its output.
At the end of the pipeline, SharedState contains everything needed by the
ScriptwriterAgent to produce the final storyboard.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class UserHistory:
    """Raw user input: a list of watched video titles and their URLs."""
    videos: list[dict[str, str]] = field(default_factory=list)
    # Each entry: {"title": "...", "url": "https://..."}


@dataclass
class UserProfile:
    """
    Output of ProfilerAgent.
    A structured summary of the user's inferred interests,
    preferred themes, and content style preferences.
    """
    interests: list[str] = field(default_factory=list)
    themes: list[str] = field(default_factory=list)
    preferred_style: str = ""          # e.g. "educational", "entertaining", "infotainment"
    raw_summary: str = ""              # free-text profile written by the LLM
    investigated_insights: list[str] = field(default_factory=list)  # insights from video transcripts


@dataclass
class KnowledgeContext:
    """
    Output of KnowledgeRetrieverAgent (Phase 4).
    A list of verified news, encyclopedic lore, scientific mechanisms, and factual snippets
    grounding the story in real-world details.
    """
    snippets: list[dict[str, str]] = field(default_factory=list)
    # Each entry: {"headline": "...", "summary": "...", "url": "..."}


# Backward-compatibility alias
NewsContext = KnowledgeContext


@dataclass
class ImageContext:
    """
    Output of ImageSearchAgent.
    A list of image results (ID, URL, query, description) relevant to storyboard scenes.
    Each image has a discrete ID (e.g. 'img1', 'img2') so the scriptwriter
    can reference visuals symbolically without hallucinating raw URLs.
    """
    images: list[dict[str, Any]] = field(default_factory=list)
    # Each entry: {"image_id": "img1", "query": "...", "url": "...", "alt_text": "..."}

    def get_image_map(self) -> dict[str, dict[str, Any]]:
        """Return a mapping from image_id -> image entry dict."""
        return {
            img.get("image_id", f"img{i+1}"): img
            for i, img in enumerate(self.images)
        }


@dataclass
class Storyboard:
    """
    Output of ScriptwriterAgent — the final deliverable.
    A structured video storyboard with scenes, subtitles, audio cues,
    and groundable image references.
    """
    title: str = ""
    target_duration_seconds: int = 60
    scenes: list[dict[str, Any]] = field(default_factory=list)
    # Each scene: {
    #   "scene_id": int,
    #   "description": str,       # visual description
    #   "subtitle": str,          # on-screen text / caption
    #   "audio_cue": str,         # voiceover or music note
    #   "image_ref": str | None,  # symbolic identifier, e.g. "img1"
    #   "image_url": str | None,  # programmatically resolved from image_ref
    #   "duration_seconds": int,
    # }
    overall_narrative: str = ""

    def resolve_image_urls(self, image_context: ImageContext) -> None:
        """
        Programmatically resolve image_ref ('img1', 'img2') to real image_url.
        Ensures the LLM never has to handle or hallucinate raw URLs.
        """
        img_map = image_context.get_image_map()
        for scene in self.scenes:
            ref = scene.get("image_ref")
            if ref and ref in img_map:
                scene["image_url"] = img_map[ref].get("url")
            elif not scene.get("image_url"):
                scene["image_url"] = None


@dataclass
class SharedState:
    """
    The central blackboard shared across all agents in the pipeline.
    Initialized by the Orchestrator; progressively filled by each agent.
    """
    # --- INPUT ---
    user_history: UserHistory = field(default_factory=UserHistory)

    # --- INTERMEDIATE OUTPUTS (filled by agents in order) ---
    user_profile: UserProfile = field(default_factory=UserProfile)
    candidate_pitches: list[dict[str, Any]] = field(default_factory=list)  # Writers' Room pitches
    selected_pitch: dict[str, Any] = field(default_factory=dict)            # Winning pitch
    director_critique: list[dict[str, Any]] = field(default_factory=list)  # Director reviews of each pitch
    director_notes: str = ""                                               # Directorial instructions for production
    knowledge_context: KnowledgeContext = field(default_factory=KnowledgeContext)  # Verified facts, lore & news
    news_context: NewsContext = field(default_factory=NewsContext)            # Backward-compatibility mirror
    image_context: ImageContext = field(default_factory=ImageContext)

    # --- FINAL OUTPUT ---
    storyboard: Storyboard = field(default_factory=Storyboard)

    def __post_init__(self) -> None:
        # Keep knowledge_context and news_context synchronized
        if not self.knowledge_context.snippets and self.news_context.snippets:
            self.knowledge_context = self.news_context
        elif not self.news_context.snippets and self.knowledge_context.snippets:
            self.news_context = self.knowledge_context

    # --- METADATA ---
    user_id: str = "default_user"
    run_id: str = ""
    pipeline_log: list[str] = field(default_factory=list)  # human-readable trace
    token_usage: dict[str, int] = field(default_factory=dict)  # {"prompt_tokens": int, "completion_tokens": int, "total_calls": int}
    estimated_cost_usd: float = 0.0  # cost in USD based on provider pricing

    def log(self, message: str) -> None:
        """Append a message to the pipeline trace log."""
        self.pipeline_log.append(message)
        print(f"[PIPELINE] {message}")

    def to_dict(self) -> dict:
        """Serialize the full state to a plain dict (for saving to JSON)."""
        import dataclasses
        d = dataclasses.asdict(self)
        # Ensure synchronization in serialized dict
        d["news_context"] = d.get("knowledge_context", d.get("news_context", {}))
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "SharedState":
        """Deserialize SharedState from a dictionary (e.g., loaded from JSON)."""
        history_data = data.get("user_history", {})
        profile_data = data.get("user_profile", {})
        k_data = data.get("knowledge_context") or data.get("news_context", {})
        img_data = data.get("image_context", {})
        sb_data = data.get("storyboard", {})

        return cls(
            user_history=UserHistory(videos=history_data.get("videos", [])),
            user_profile=UserProfile(
                interests=profile_data.get("interests", []),
                themes=profile_data.get("themes", []),
                preferred_style=profile_data.get("preferred_style", ""),
                raw_summary=profile_data.get("raw_summary", ""),
                investigated_insights=profile_data.get("investigated_insights", []),
            ),
            candidate_pitches=data.get("candidate_pitches", []),
            selected_pitch=data.get("selected_pitch", {}),
            director_critique=data.get("director_critique", []),
            director_notes=data.get("director_notes", ""),
            knowledge_context=KnowledgeContext(snippets=k_data.get("snippets", [])),
            news_context=NewsContext(snippets=k_data.get("snippets", [])),
            image_context=ImageContext(images=img_data.get("images", [])),
            storyboard=Storyboard(
                title=sb_data.get("title", ""),
                target_duration_seconds=sb_data.get("target_duration_seconds", 60),
                scenes=sb_data.get("scenes", []),
                overall_narrative=sb_data.get("overall_narrative", ""),
            ),
            user_id=data.get("user_id", "default_user"),
            run_id=data.get("run_id", ""),
            pipeline_log=data.get("pipeline_log", []),
            token_usage=data.get("token_usage", {}),
            estimated_cost_usd=data.get("estimated_cost_usd", 0.0),
        )

    def to_markdown(self) -> str:
        """Convert the storyboard into a human-readable Markdown format."""
        sb = self.storyboard
        cost_str = f" | **Est. Cost:** `${self.estimated_cost_usd:.5f}`" if self.estimated_cost_usd else ""
        md_lines = [
            f"# {sb.title or 'Untitled Storyboard'}",
            f"**Target Duration:** {sb.target_duration_seconds}s | **Run ID:** `{self.run_id}`{cost_str}\n",
            f"## Narrative Arc",
            f"{sb.overall_narrative or 'No narrative summary provided.'}\n",
            f"## Scene Breakdown\n",
        ]

        for s in sb.scenes:
            scene_id = s.get("scene_id", "?")
            dur = s.get("duration_seconds", "?")
            desc = s.get("description", "")
            sub = s.get("subtitle", "")
            audio = s.get("audio_cue", "")
            img_ref = s.get("image_ref")
            img_url = s.get("image_url")

            md_lines.append(f"### Scene {scene_id} ({dur}s)")
            md_lines.append(f"- **Visual Description:** {desc}")
            md_lines.append(f"- **Subtitle (On-screen):** *\"{sub}\"*")
            md_lines.append(f"- **Audio / Voiceover:** {audio}")
            if img_url:
                md_lines.append(f"- **Image Asset:** [`{img_ref or 'asset'}`]({img_url})")
            elif img_ref:
                md_lines.append(f"- **Image Ref:** `{img_ref}`")
            md_lines.append("")

        if self.candidate_pitches:
            md_lines.append("## 🎬 Writers' Room: Pitches & Directorial Selection\n")
            for p in self.candidate_pitches:
                pid = p.get("pitch_id", "pitch")
                title = p.get("title", "")
                tone = p.get("target_tone", "")
                hook = p.get("hook_3s", "")
                concept = p.get("concept", "")
                is_selected = (self.selected_pitch.get("pitch_id") == pid)
                status = "🏆 **[SELECTED BY DIRECTOR]**" if is_selected else "*[Archived]*"
                md_lines.append(f"### {status} Pitch `{pid}`: {title}")
                md_lines.append(f"- **Tone/Vibe:** {tone}")
                md_lines.append(f"- **3-Second Hook:** *\"{hook}\"*")
                md_lines.append(f"- **Concept:** {concept}")
                if p.get("user_alignment_reason"):
                    md_lines.append(f"- **Audience Fit:** {p.get('user_alignment_reason')}")
                md_lines.append("")

            if self.director_notes:
                md_lines.append("### 📋 Directorial Notes & Production Guidance")
                md_lines.append(f"{self.director_notes}\n")

        snippets = self.knowledge_context.snippets or self.news_context.snippets
        if snippets:
            md_lines.append("## 📚 Grounded Knowledge & Factual Evidence\n")
            for snip in snippets:
                head = snip.get("headline", "Verified Fact")
                summ = snip.get("summary", "")
                url = snip.get("url", "")
                source_str = f" ([Source]({url}))" if url else ""
                md_lines.append(f"- **{head}**{source_str}: {summ}")
            md_lines.append("")

        if self.user_profile.interests:
            md_lines.append("## Inferred User Profile")
            md_lines.append(f"- **Interests:** {', '.join(self.user_profile.interests)}")
            md_lines.append(f"- **Themes:** {', '.join(self.user_profile.themes)}")
            md_lines.append(f"- **Style:** {self.user_profile.preferred_style}")
            if self.user_profile.investigated_insights:
                md_lines.append(f"- **Deep-Dive Spoken Insights:**")
                for ins in self.user_profile.investigated_insights:
                    md_lines.append(f"  - {ins}")
            md_lines.append(f"- **Summary:** {self.user_profile.raw_summary}\n")

        return "\n".join(md_lines)

    def save(self, path: str) -> None:
        """Save the full state as JSON and companion Markdown file."""
        json_path = Path(path)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

        md_path = json_path.with_suffix(".md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(self.to_markdown())
