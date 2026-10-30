"""
ImageSearchAgent — Agent 3 in the pipeline.

Responsibility:
  - Takes the user profile and (optionally) news snippets.
  - Searches for relevant images by keyword — NO vision model is used.
  - Returns a list of image URLs with alt text / descriptions,
    which the ScriptwriterAgent embeds in scene metadata.
  - The downstream video renderer (Max's component) will fetch and use
    these URLs to compose the video sketch.

Tool used:
  - src/tools/image_tool.py  (wraps an image search API — TBD: Unsplash, Pexels, etc.)

Output written to state:
  - state.image_context  (ImageContext dataclass)
"""

from __future__ import annotations

from openai import OpenAI

from src.agents.base_agent import BaseAgent
from src.core.state import ImageContext, SharedState
from src.tools.image_tool import search_images  # noqa: F401  (to be implemented)


class ImageSearchAgent(BaseAgent):
    """Fetches relevant images by keyword for use in the storyboard scenes."""

    def __init__(self, model: str, client: OpenAI, images_per_theme: int = 2) -> None:
        super().__init__(name="ImageSearchAgent", model=model, client=client)
        self.images_per_theme = images_per_theme

    def run(self, state: SharedState) -> SharedState:
        self._log(state, "Starting image search based on user profile and themes.")

        themes = state.user_profile.themes[:4]  # use top-4 themes
        if not themes:
            self._log(state, "No themes found in profile. Skipping image search.")
            return state

        all_images: list[dict] = []
        for theme in themes:
            self._log(state, f"Searching images for theme: '{theme}'")
            # TODO: implement search_images in src/tools/image_tool.py
            results = search_images(query=theme, max_results=self.images_per_theme)
            all_images.extend(results)

        state.image_context = ImageContext(images=all_images)
        self._log(state, f"Found {len(all_images)} images.")
        return state
