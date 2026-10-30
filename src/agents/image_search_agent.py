"""
ImageSearchAgent — Agent in the pipeline illustrating the draft storyboard.

Responsibility:
  - Takes the drafted storyboard (scenes, visual descriptions).
  - Uses an LLM to analyze what visual stock photography would best illustrate each scene.
  - Searches for candidate images using external tools (Unsplash/Pexels) based on keywords.
  - Formats results with discrete identifiers (img1, img2, ...) so the downstream
    scriptwriter refinement phase can reference images cleanly without handling raw URLs.

Tool used:
  - src/tools/image_tool.py  (wraps keyword image search API)

Output written to state:
  - state.image_context  (ImageContext dataclass with image_id mapping)
"""

from __future__ import annotations

import json
from typing import Any
from openai import OpenAI

from src.agents.base_agent import BaseAgent
from src.core.llm_client import chat_completion
from src.core.state import ImageContext, SharedState
from src.tools.image_tool import search_images


class ImageSearchAgent(BaseAgent):
    """
    Finds visual imagery tailored specifically to scenes in a drafted storyboard.
    Assigns symbolic IDs (img1, img2, ...) to prevent URL hallucination.
    """

    def __init__(self, model: str, client: OpenAI, max_images_per_scene: int = 1) -> None:
        super().__init__(name="ImageSearchAgent", model=model, client=client)
        self.max_images_per_scene = max_images_per_scene

    def run(self, state: SharedState) -> SharedState:
        self._log(state, "Analyzing draft storyboard to formulate visual image queries.")

        scenes = state.storyboard.scenes
        if not scenes:
            self._log(state, "No scenes found in storyboard. Skipping image search.")
            return state

        # 1. Format draft scenes for LLM query formulation
        scene_descriptions = "\n".join(
            f"Scene {s.get('scene_id', idx + 1)}: {s.get('description', '')} (Subtitle: '{s.get('subtitle', '')}')"
            for idx, s in enumerate(scenes)
        )

        user_prompt = f"""\
STORYBOARD TITLE: {state.storyboard.title}
OVERALL NARRATIVE: {state.storyboard.overall_narrative}

DRAFT SCENES:
{scene_descriptions}

For each scene, formulate a clear, descriptive visual search query (1 query per scene) to find stock photography or background footage.
"""
        raw_response = chat_completion(
            client=self.client,
            model=self.model,
            system_prompt=_IMAGE_QUERY_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=0.5,
            max_tokens=600,
        )

        scene_queries = _parse_scene_queries(raw_response, len(scenes))
        self._log(state, f"Formulated {len(scene_queries)} visual scene search queries.")

        # 2. Search for images and assign discrete IDs (img1, img2, ...)
        all_images: list[dict[str, Any]] = []
        img_counter = 1

        for item in scene_queries:
            scene_id = item.get("scene_id")
            query = item.get("query", "")
            if not query:
                continue

            self._log(state, f"Searching imagery for Scene {scene_id}: '{query}'")
            results = search_images(query=query, max_results=self.max_images_per_scene)
            for res in results:
                image_id = f"img{img_counter}"
                all_images.append({
                    "image_id": image_id,
                    "target_scene_id": scene_id,
                    "query": query,
                    "url": res.get("url", ""),
                    "alt_text": res.get("alt_text", f"Image for {query}"),
                })
                img_counter += 1

        state.image_context = ImageContext(images=all_images)
        self._log(state, f"Retrieved {len(all_images)} candidate images with discrete IDs (img1 to img{len(all_images)}).")
        return state


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_scene_queries(raw: str, num_scenes: int) -> list[dict[str, Any]]:
    """Parse JSON mapping scenes to visual queries."""
    try:
        cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        data = json.loads(cleaned)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            for k in ("scene_queries", "queries", "scenes"):
                if k in data and isinstance(data[k], list):
                    return data[k]
    except Exception:
        pass

    # Fallback: create generic visual queries if parsing fails
    return [{"scene_id": i + 1, "query": f"scene visual {i + 1}"} for i in range(num_scenes)]


_IMAGE_QUERY_SYSTEM_PROMPT = """\
You are a visual director for entertaining, high-impact short videos.
Your task is to analyze storyboard scenes and formulate specific visual search queries
to find real-world photography, iconic imagery, authentic objects, or dramatic archival footage that brings the story to life.

Guidelines:
- Create 1 vivid, realistic search query for each scene.
- Target real places, authentic objects, dramatic real-world moments, or striking close-ups (e.g. "Wikipedia servers datacenter glowing", "vintage Daft Punk helmets studio neon", "ancient Greek pottery symposium").
- Avoid sterile generic stock photos or fake staged business handshakes.
- Keep each query concise (3 to 6 descriptive words).

Return ONLY valid JSON in this format:
{
  "scene_queries": [
    {
      "scene_id": 1,
      "query": "vintage daft punk helmets studio neon"
    },
    {
      "scene_id": 2,
      "query": "giant server room blinking lights"
    }
  ]
}
"""
