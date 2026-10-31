# AGENT.md — Developer & AI Agent Operating Manual
**Project:** Multi-Agent Storyboarding for Personalized Short-Video Generation  
**Course:** COMP 4221 (Intro to NLP, HKUST — Fall 2026)  
**Living Documentation Tracker:** [`docs/project_tracker.md`](docs/project_tracker.md)  
**Live Data Collection Tool:** [`https://charlesdupouy.github.io/nlp-multiagent-storyboarding/`](https://charlesdupouy.github.io/nlp-multiagent-storyboarding/)

---

## 1. Project Mission & Core Research Thesis

This project investigates whether a **tool-augmented multi-agent framework with visual grounding** outperforms a **single-agent zero-shot LLM baseline** on generating personalized 60-second video storyboards from real YouTube Shorts watch histories.

### Team Roles & Component Ownership
- **Charles Dupouy (Lead Engineer):** Multi-agent pipeline architecture, prompt engineering, tool integrations (`youtube_tool`, `news_tool`, `image_tool`), orchestration, and baseline harmonization.
- **Antoine Lepin (Data & Baseline Owner):** Peer watch history collection via the GitHub Pages collector, dataset preprocessing/filtering, and baseline experimental execution.
- **Max Robertson (Evaluation & Demo Owner):** Automated LLM-as-a-judge scoring (`evaluation/evaluator.py`), human blind A/B preference testing, and video sketch renderer (TTS + images).

---

## 2. Non-Negotiable Invariants & Architectural Rules

Any AI agent or engineer editing this codebase **MUST strictly uphold the following rules**:

### Rule 1: Unified I/O Contract
- `Orchestrator.run(videos)` and `SingleAgentBaseline.run(videos)` **must accept the exact same input**:
  `videos: list[dict[str, str]]` where each dict contains `{"title": str, "url": str}`.
- Both systems **must return an instance of `SharedState`** (`src/core/state.py`).
- Never introduce breaking changes to `SharedState` without updating both pipeline and baseline serializers.

### Rule 2: Zero URL Hallucination (Symbolic Referencing)
- **NEVER** instruct the scriptwriting LLM to invent or copy/paste raw image URLs.
- The `ImageSearchAgent` catalogs candidate images as symbolic IDs (`img1`, `img2`, ...).
- The `ScriptwriterAgent` assigns visual assets strictly using `"image_ref": "img1"`.
- Real URLs are resolved programmatically in Python via `storyboard.resolve_image_urls(state.image_context)`.

### Rule 3: Strict Scientific Fairness (No Strawman Baseline)
- **Academic Integrity:** The single-agent baseline (`baseline/single_agent_baseline.py`) must remain **neutral, rigorous, and competitive**.
- Never alter the baseline prompt to steer it toward funny, shitty, or clip-show mashups.
- Both systems must be held to identical high standards of thematic focus, narrative arc, and visual descriptions.
- The multi-agent system's advantage must arise purely from its architectural capabilities (spoken transcript inspection, Writers' Room deliberation, external fact retrieval, and visual asset curation), not from artificially crippling the baseline.

### Rule 4: Data Privacy & Secret Hygiene
- **NEVER** commit `.env` or personal participant data (`data/user_histories/*.json` except `example.json`) to git.
- `.gitignore` is configured to allow only reference storyboards and `example.json`. Keep it that way.

### Rule 5: Zero-Auth YouTube Tools
- Video title resolution and transcript extraction (`src/tools/youtube_tool.py`) use **oEmbed** and **`youtube-transcript-api`**.
- Do **NOT** introduce official Google Cloud YouTube Data API v3 keys or OAuth requirements. The current tool is keyless, zero-quota, and instant.

---

## 3. Pipeline Architecture (The 7-Phase Execution Loop)

```mermaid
flowchart TD
    I["User Watch History\n(Titles + URLs)"] --> P1["Phase 1A: Profiler Round 1\nMacro scan across history; selects 3-4 target videos"]
    P1 --> T1["Tool: youtube_transcript_api\nExtracts actual spoken dialogue & oEmbed creator metadata"]
    T1 --> P2["Phase 1B: Profiler Round 2\nSynthesizes fine-grained profile with concrete spoken insights"]
    P2 --> WR["Phase 2: WritersRoomAgent\nBrainstorms 3 distinct candidate pitches matching user style"]
    WR --> DIR["Phase 3: DirectorAgent (Showrunner)\nCritiques all 3 pitches, selects winner & issues production notes"]
    DIR --> N["Phase 4: NewsRetrieverAgent (Targeted Research)\nFormulates search queries specific to the winning pitch via NewsAPI.org"]
    N --> S1["Phase 5: ScriptwriterAgent (Draft Pass)\nDrafts 60s storyboard executing winning pitch + director guidance"]
    S1 --> IMG["Phase 6: ImageSearchAgent (Art Director)\n1. Physical query sanitization (no cinematic fluff)\n2. Candidate retrieval (4 per scene via Unsplash API)\n3. Semantic re-ranking to select most representative photo"]
    IMG --> S2["Phase 7: ScriptwriterAgent (Refine Pass)\nGrounds scenes to visual assets using symbolic refs ('img1')"]
    S2 --> RES["Programmatic URL Resolution\nMaps 'image_ref' → validated Unsplash CDN URLs (no hallucination)"]
    RES --> O["Storyboard JSON & Markdown\n(Pitches, Director Decision, Scenes, Subtitles, Narration, Image URLs)"]
```

---

## 4. Key Files & Directory Layout

| File / Path | Responsibility |
|---|---|
| `main.py` | CLI entry point (`--history`, `--baseline`, `--config`). |
| `src/core/state.py` | The central `SharedState` blackboard (dataclasses, serialization, markdown output, token accounting). |
| `src/core/llm_client.py` | DeepSeek/OpenAI-compatible wrapper with global token tracking (`get_token_usage`) and cost calculation. |
| `src/orchestrator.py` | Sequences the 7 phases of the multi-agent pipeline. |
| `src/agents/profiler_agent.py` | Two-round profiling agent using `youtube_tool.py` for transcript evidence. |
| `src/agents/pitch_agent.py` | `WritersRoomAgent` producing 3 distinct candidate story pitches. |
| `src/agents/director_agent.py` | `DirectorAgent` critiquing candidate pitches and issuing directorial production notes. |
| `src/agents/news_retriever_agent.py` | Targeted external research agent querying NewsAPI.org on the winning pitch. |
| `src/agents/image_search_agent.py` | Art Director agent: physical query formulation + Unsplash candidate semantic selection. |
| `src/agents/scriptwriter_agent.py` | Two-pass screenwriter: initial draft from pitch + refine pass against image catalog. |
| `src/tools/youtube_tool.py` | Extracts subtitles via `youtube-transcript-api` and titles via YouTube oEmbed. |
| `src/tools/news_tool.py` | Wraps NewsAPI.org (with query simplification and mock fallback). |
| `src/tools/image_tool.py` | Wraps Unsplash API (with cinematic query sanitization and mock fallback). |
| `src/tools/import_user_history.py` | Ingests and auto-enriches participant watch histories. |
| `baseline/single_agent_baseline.py` | Competitive zero-shot baseline running the same prompt and model. |
| `evaluation/evaluator.py` | Automated LLM-as-a-judge scoring CLI (Relevance, Completeness, Creativity, Visual Alignment). |
| `prompts/` | Plain-text registry of all versioned system prompts. |
| `tests/test_pipeline.py` | Unit tests covering state, tools, query cleaning, and serialization. |
| `docs/project_tracker.md` | Living research log with cost tables, design decisions, and team status. |

---

## 5. Standard CLI Commands

```bash
# Run the multi-agent pipeline on an example history
python3 main.py --history data/user_histories/example.json

# Run the single-agent baseline
python3 main.py --history data/user_histories/example.json --baseline

# Ingest an exported history from the web collector
python3 src/tools/import_user_history.py /path/to/exported.json participant_name

# Score a single storyboard using the LLM judge
python3 evaluation/evaluator.py --storyboard outputs/storyboards/storyboard_xyz.json

# Side-by-side comparison between Multi-Agent and Baseline
python3 evaluation/evaluator.py --compare outputs/storyboards/storyboard_ma.json outputs/storyboards/storyboard_base.json

# Run all unit tests
python3 -m unittest discover tests
```

---

## 6. Coding Standards & Conventions

1. **Python Compatibility:** Python 3.10+ required. Always use `from __future__ import annotations`.
2. **Type Annotations:** All functions and methods must have complete type hints.
3. **Robust JSON Parsing:** Always use `_parse_json()` helpers that strip markdown code fences (````json ... ````) and handle edge cases gracefully with fallbacks.
4. **Tool Graceful Degradation:** External tools (`news_tool`, `image_tool`) must check for API keys and gracefully fall back to synthetic mocks if keys are absent or if network requests fail.
5. **No Blind Front-Truncation:** When simplifying search queries for external APIs, never blindly slice the first 3 words (`words[:3]`), because in English the critical subject noun is often at the end. Use noun-preserving sanitization (`clean_stock_query`).
6. **Network / Sandboxing:** API calls (DeepSeek, NewsAPI, Unsplash, GitHub push) require network access. If running inside a restricted sandbox, ensure network bypass permissions are active.
