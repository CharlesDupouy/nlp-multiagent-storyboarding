"""
ArtDirectorAgent — Visual Grounding & Asset Curation Agent (Phase 6).

Responsibility:
  - Takes the drafted storyboard (scenes, visual descriptions).
  - Phase 1: Formulates tangible, concrete physical queries (primary & fallback)
    avoiding cinematography jargon that breaks stock search.
  - Phase 2: Retrieves multiple candidate stock photos per scene from Unsplash/Pexels.
  - Phase 3 (Art Director Pass): Inspects candidate descriptions and selects the
    most visually faithful and representative image for each scene.
  - Formats results with discrete identifiers (img1, img2, ...) so the downstream
    scriptwriter refinement phase can reference images cleanly without handling raw URLs.

Tool used:
  - src/tools/image_tool.py (wraps keyword search & candidate retrieval)

Output written to state:
  - state.image_context (ImageContext dataclass with image_id mapping)
"""

from __future__ import annotations

import json
from typing import Any
from openai import OpenAI

from src.agents.base_agent import BaseAgent
from src.core.llm_client import chat_completion, extract_json
from src.core.state import ImageContext, SharedState
from src.tools.image_tool import search_image_candidates


class ArtDirectorAgent(BaseAgent):
    """
    Finds and curates visual imagery tailored specifically to scenes in a drafted storyboard.
    Uses candidate retrieval and an Art Director semantic selection pass
    to ensure chosen images are representative and relevant.
    """

    def __init__(self, model: str, client: OpenAI, candidates_per_scene: int = 4) -> None:
        super().__init__(name="ArtDirectorAgent", model=model, client=client)
        self.candidates_per_scene = candidates_per_scene

    def run(self, state: SharedState) -> SharedState:
        self._log(state, "Analyzing draft storyboard to formulate physical visual queries.")

        scenes = state.storyboard.scenes
        if not scenes:
            self._log(state, "No scenes found in storyboard. Skipping image curation.")
            return state

        # -------------------------------------------------------------------
        # 1. Query Formulation Pass: Concrete physical nouns only
        # -------------------------------------------------------------------
        scene_descriptions = "\n".join(
            f"Scene {s.get('scene_id', idx + 1)}: {s.get('description', '')} (Subtitle: '{s.get('subtitle', '')}')"
            for idx, s in enumerate(scenes)
        )

        user_prompt_queries = f"""\
STORYBOARD TITLE: {state.storyboard.title}
OVERALL NARRATIVE: {state.storyboard.overall_narrative}

DRAFT SCENES:
{scene_descriptions}

Formulate concrete visual search queries for stock photography libraries (Unsplash).
Remember: stock search engines match physical objects and places, NOT camera angles or editing techniques.
"""
        raw_queries = chat_completion(
            client=self.client,
            model=self.model,
            system_prompt=_IMAGE_QUERY_SYSTEM_PROMPT,
            user_prompt=user_prompt_queries,
            temperature=0.3,
            max_tokens=800,
        )

        scene_queries = _parse_scene_queries(raw_queries, len(scenes))
        self._log(state, f"Formulated {len(scene_queries)} physical visual queries.")

        # -------------------------------------------------------------------
        # 2. Tool Candidate Retrieval Pass: Fetch candidate photos per scene
        # -------------------------------------------------------------------
        scene_candidates_map: dict[int, list[dict[str, Any]]] = {}
        total_sq = len(scene_queries)
        for q_idx, item in enumerate(scene_queries, 1):
            scene_id = item.get("scene_id")
            primary_q = item.get("primary_query") or item.get("query", "")
            fallback_q = item.get("fallback_query", "")

            self._log(state, f"Fetching photo candidates ({q_idx}/{total_sq}) for Scene {scene_id}: '{primary_q}'")
            candidates = search_image_candidates(
                query=primary_q,
                fallback_query=fallback_q,
                max_results=self.candidates_per_scene,
            )
            scene_candidates_map[scene_id] = candidates
            self._log(
                state,
                f"  -> Retrieved {len(candidates)} photo candidate(s) for Scene {scene_id}."
            )

        # -------------------------------------------------------------------
        # 3. Art Director Selection Pass: Pick most representative candidate
        # -------------------------------------------------------------------
        self._log(state, "Art Director evaluating candidates to select the most representative photos.")
        art_director_prompt = _build_art_director_prompt(scenes, scene_candidates_map)

        raw_selection = chat_completion(
            client=self.client,
            model=self.model,
            system_prompt=_ART_DIRECTOR_SYSTEM_PROMPT,
            user_prompt=art_director_prompt,
            temperature=0.2,
            max_tokens=1000,
        )

        selections = _parse_art_director_selections(raw_selection)

        # -------------------------------------------------------------------
        # 4. Catalog final images with discrete symbolic IDs (img1, img2, ...)
        # -------------------------------------------------------------------
        all_images: list[dict[str, Any]] = []
        for idx, scene in enumerate(scenes):
            scene_id = scene.get("scene_id", idx + 1)
            candidates = scene_candidates_map.get(scene_id, [])
            if not candidates:
                continue

            # Find selected candidate ID or default to first
            chosen_cand_id = selections.get(scene_id, "cand1")
            chosen_cand = next((c for c in candidates if c.get("candidate_id") == chosen_cand_id), candidates[0])

            image_id = f"img{idx + 1}"
            all_images.append({
                "image_id": image_id,
                "target_scene_id": scene_id,
                "query": scene_queries[idx].get("primary_query", "") if idx < len(scene_queries) else "",
                "url": chosen_cand.get("url", ""),
                "alt_text": chosen_cand.get("alt_text", ""),
                "description": chosen_cand.get("description", ""),
            })
            self._log(
                state,
                f"  -> Assigned [{image_id}] for Scene {scene_id}: \"{chosen_cand.get('description')}\""
            )

        state.image_context = ImageContext(images=all_images)
        self._log(state, f"Curated {len(all_images)} representative images cataloged (img1 to img{len(all_images)}).")
        return state


# Backward-compatibility alias
ImageSearchAgent = ArtDirectorAgent


# ---------------------------------------------------------------------------
# Prompt Builders & Helpers
# ---------------------------------------------------------------------------

def _build_art_director_prompt(
    scenes: list[dict[str, Any]],
    scene_candidates: dict[int, list[dict[str, Any]]],
) -> str:
    blocks = []
    for s in scenes:
        sid = s.get("scene_id", 1)
        desc = s.get("description", "")
        cands = scene_candidates.get(sid, [])
        cands_text = "\n".join(
            f"  - [{c.get('candidate_id')}]: \"{c.get('description')}\""
            for c in cands
        ) or "  - No candidates available"

        blocks.append(
            f"SCENE {sid}:\n"
            f"Desired Visual: {desc}\n"
            f"Available Photo Candidates:\n{cands_text}"
        )
    return "\n\n".join(blocks)


def _parse_scene_queries(raw: str, num_scenes: int) -> list[dict[str, Any]]:
    """Parse JSON mapping scenes to visual queries."""
    try:
        data = extract_json(raw)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            for k in ("scene_queries", "queries", "scenes"):
                if k in data and isinstance(data[k], list):
                    return data[k]
    except Exception:
        pass
    return [
        {
            "scene_id": i + 1,
            "primary_query": f"subject {i + 1}",
            "fallback_query": "object",
        }
        for i in range(num_scenes)
    ]


def _parse_art_director_selections(raw: str) -> dict[int, str]:
    """Parse JSON mapping scene_id -> selected candidate_id."""
    selections: dict[int, str] = {}
    try:
        data = extract_json(raw)
        items = data if isinstance(data, list) else data.get("selections", [])
        for item in items:
            sid = item.get("scene_id")
            cid = item.get("selected_candidate_id")
            if sid and cid:
                selections[int(sid)] = str(cid)
    except Exception:
        pass
    return selections


_IMAGE_QUERY_SYSTEM_PROMPT = """\
You are an expert Art Director formulating stock photo search queries for an external image database (Unsplash).

CRITICAL SEARCH ENGINE PRINCIPLES:
1. Search engines match TANGIBLE PHYSICAL OBJECTS and REAL-WORLD SETTINGS (e.g., "multicolor ballpoint pen", "students in classroom", "vintage rotary phone", "newspaper on wooden table").
2. STRICTLY FORBIDDEN: NEVER include video production directions or shot modifiers in the query.
   - Do NOT include: "split screen", "exploded view", "macro", "closeup", "cinematic", "drone shot", "pov", "2000s", "vintage style".
   - Do NOT include brand names (e.g. use "ballpoint pen", not "Bic").
3. Each scene MUST have:
   - "primary_query": 2 to 4 tangible visual nouns/adjectives describing the single main physical object or setting.
   - "fallback_query": 1 to 2 word core physical noun (e.g. "ballpoint pen", "classroom", "telephone").

Return ONLY valid JSON in this format:
{
  "scene_queries": [
    {
      "scene_id": 1,
      "primary_query": "multicolor click ballpoint pen",
      "fallback_query": "ballpoint pen"
    },
    {
      "scene_id": 2,
      "primary_query": "school classroom children desks",
      "fallback_query": "classroom"
    }
  ]
}
"""


_ART_DIRECTOR_SYSTEM_PROMPT = """\
You are an Art Director reviewing photographic image candidates for short-video scenes.
For each scene, evaluate the candidate descriptions and select the SINGLE candidate that is most visually faithful and representative of what the scene describes.

Guidelines:
- Choose the candidate whose description actually depicts the requested physical subject (e.g. if the scene is about a pen, pick the pen, NOT a woman pointing or a basketball hoop).
- If multiple candidates match, pick the most descriptive and aesthetically fitting one.

Return ONLY valid JSON in this format:
{
  "selections": [
    {
      "scene_id": 1,
      "selected_candidate_id": "cand2",
      "reason": "Directly depicts a ballpoint click pen"
    }
  ]
}
"""
