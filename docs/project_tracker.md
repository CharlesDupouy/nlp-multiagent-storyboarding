# COMP 4221 — Project Tracker
## Multi-Agent Storyboarding for Personalized Short-Video Generation

> [!IMPORTANT]
> **This document is the living record of the project.** Update it after every significant implementation step. It will feed directly into the final 6–8 page report (due Dec 4) and the live presentation (Nov 24–26).

---

## Research Question (Practical Test)

> *On **personalized short-video storyboarding**, our change — **a tool-augmented multi-agent brainstorming pipeline with visual grounding** — should improve **content creativity, visual coherence, and alignment with niche user interests** relative to **a single-agent zero-shot generation baseline**, measured by **LLM-as-a-judge relevance/creativity scores and blind human preference testing** under a budget of **\$50 in API credits and local compute**.*

---

## Team & Ownership

| Component | Owner | Status |
|---|---|---|
| Data collection & public dataset filtering | Antoine | 🔲 In progress (Peer collector live on GitHub Pages; `--split` partitioning ready) |
| Single-agent baseline | Antoine | 🟢 Implemented, neutralized & tested on real user data (with Serper top-1 visual parity) |
| Multi-agent framework (core architecture) | Charles | 🟢 Writers' Room (R1) + Director (R1) + Knowledge Retriever (Multi-Tier) + Art Director (Re-ranking) complete & verified (v4.1) |
| Prompt engineering & refinement | Charles | 🟢 v3.1 active (Verbatim dialogue, zero placeholders, domain-targeted research, anti-regurgitation) |
| Real API integration (knowledge, images) | Charles | 🟢 100% Complete (NewsAPI + Wikipedia + Serper Google Images + YouTube transcripts integrated & live) |
| LLM-as-a-judge evaluator | Max | 🟢 Complete (`evaluation/llm_judge.py` with 5D pairwise verdicts, 1–5 Likert grades, and visual caption grounding) |
| Automated evaluation over dataset | Max | 🟢 Validated on DeepSeek pilot & dry-run (`outputs/judge/`); pending full batch |
| Demo sketch renderer (TTS + images) | Max | 🟢 Neural Edge-TTS + scene footer + visual slideshow (`storyboard_to_video.py`) |

---

## System Architecture

### Pipeline (Writers' Room & Director with Targeted Retrieval & Visual Grounding)

```mermaid
flowchart TD
    I["User Watch History\n(vid1, vid2, ... titles + URLs)"] --> P1["Phase 1A: Profiler Round 1\nMacro scan & active selection of 3-4 target videos"]
    P1 --> T1["Tool: youtube_transcript_api\nExtracts actual spoken dialogue & oEmbed channel metadata"]
    T1 --> P2["Phase 1B: Profiler Round 2\nSynthesizes fine-grained profile with concrete spoken insights"]
    P2 --> WR["Phase 2: WritersRoomAgent (DeepSeek R1)\nBrainstorms 3 distinct candidate pitches matching user style"]
    WR --> DIR["Phase 3: DirectorAgent (DeepSeek R1 Showrunner)\nCritiques all 3 pitches, selects winner & issues production notes"]
    DIR --> K["Phase 4: KnowledgeRetrieverAgent (Targeted Research)\nMulti-tiered search: NewsAPI + MediaWiki/Wikipedia + LLM Synthesis"]
    K --> S1["Phase 5: ScriptwriterAgent (Draft Pass, DeepSeek R1)\nDrafts 60s storyboard executing winning pitch + director guidance"]
    S1 --> ART["Phase 6: ArtDirectorAgent\n1. Physical query sanitization (no camera jargon)\n2. 4-candidate Serper Google Images retrieval\n3. Semantic re-ranking to select best photographic asset"]
    ART --> S2["Phase 7: ScriptwriterAgent (Refine Pass, DeepSeek R1)\nGrounds scenes to visual assets using symbolic refs ('img1')"]
    S2 --> RES["Programmatic URL Resolution\nMaps 'image_ref' → validated image URLs (or null on miss; zero hallucination)"]
    RES --> O["Storyboard JSON & Markdown\n(Pitches, Director Decision, Scenes, Subtitles, Narration, Image URLs)"]
```

> [!NOTE]
> **Why this Writers' Room architecture over a linear pipeline?**  
> 1. **User Style Adaptability:** Instead of forcing every user into an arbitrary documentary format, the Writers' Room proposes 3 distinct concepts matching their natural preferred style (witty humor, craft, pop culture, tech puzzle, or lore).
> 2. **Parallel Exploration (Exploration vs. Exploitation):** Parallel candidate pitches allow creative breadth before committing to production.
> 3. **Showrunner Deliberation (DeepSeek R1):** The Director evaluates pitches against 3s hook potential, personal resonance, and visual feasibility.
> 4. **Targeted Information Retrieval:** The Knowledge Retriever searches for concrete evidence across news, Wikipedia, and technical lore *tailored specifically to the winning pitch* and the Director's instructions.
> 5. **Art Director Photographic Grounding:** Sanitizes cinematic jargon and semantically re-ranks 4 candidates per scene. Symbolic keys (`img1`, `img2`, ...) eliminate URL hallucination.

### Agent Roles

| Agent | Canonical Class | Input | Output | Model | Tools |
|---|---|---|---|---|---|
| **Phase 1: Profiler** | `ProfilerAgent` | Watch history (`vid1`, `vid2`, ...) | `UserProfile` (interests, themes, style, spoken insights) | `deepseek-chat` | `youtube_transcript_api` + oEmbed metadata |
| **Phase 2: Writers' Room** | `WritersRoomAgent` | `UserProfile` + spoken insights | `candidate_pitches` (3 diverse story concepts) | `deepseek-reasoner` (R1) | Creative reasoning pass |
| **Phase 3: Director** | `DirectorAgent` | `UserProfile` + `candidate_pitches` | `selected_pitch` + critique + production notes | `deepseek-reasoner` (R1) | Critical showrunner evaluation |
| **Phase 4: Knowledge Retriever** | `KnowledgeRetrieverAgent` | `selected_pitch` + production notes | `KnowledgeContext` (verified headlines, lore & summaries) | `deepseek-chat` | `knowledge_tool.py` (NewsAPI + Wikipedia + Tavily + LLM Synthesis) |
| **Phase 5: Scriptwriter (Draft)** | `ScriptwriterAgent` | Pitch + notes + `KnowledgeContext` | Draft `Storyboard` (scenes, verbatim dialogue, cues) | `deepseek-reasoner` (R1) | Structured narrative synthesis |
| **Phase 6: Art Director** | `ArtDirectorAgent` | Draft scenes + query prompts | `ImageContext` (`img1`..`img6` catalog of validated photos) | `deepseek-chat` | `image_tool.py` (Serper Google Images + 4-candidate semantic re-ranking) |
| **Phase 7: Scriptwriter (Refine)** | `ScriptwriterAgent` | Draft scenes + `ImageContext` | Refined `Storyboard` (`image_ref: 'img1'`) | `deepseek-reasoner` (R1) | Programmatic CDN URL resolver |

### Controlled Comparison

```mermaid
flowchart LR
    H["Identical User Watch History\n(videos: list[dict[str, str]])"]
    H --> MA["Multi-Agent Pipeline (v4.1)\nProfile → Writers' Room → Director → Knowledge → Draft → Art Director → Refine"]
    H --> SA["Single-Agent Baseline (v3.5)\nZero-shot, same model, same watch history + Naive Top-1 Serper Search"]
    MA --> SB1["Storyboard A\n(Grounded with curated imagery & verified facts)"]
    SA --> SB2["Storyboard B\n(Zero-shot single pass with naive image retrieval)"]
    SB1 --> E["Automated Judge & Blind Human A/B\n5D Pairwise Rubric + 1-5 Likert Grades + Neural Video Sketches"]
    SB2 --> E
```

---

## Data

### Public Dataset
- **Candidate:** KuaiRec or keyword-adapted MovieLens
- **Split:** train / dev / test (to be documented once selected)
- **Status:** 🔲 TODO (Antoine)

### Real User Histories
- YouTube Shorts watch histories from peers
- Format: `[{"title": "...", "url": "https://..."}]`
- Represented in agent prompts cleanly as `[vid1] Title: "..." | URL: ...`
- **Collector Tool:** Client-side 1-click bookmarklet + tutorial web page (`tools/history_collector/index.html`) hosted on GitHub Pages (`https://charlesdupouy.github.io/nlp-multiagent-storyboarding/`).
- **Ingestion & Slicing Helper:** `src/tools/import_user_history.py` validates and saves participant files to `data/user_histories/<user>.json`.
  - Supports `--split <X>` (e.g. `--split 3`) to partition a single participant's watch history into $X$ balanced, disjoint slices (`<user>_part1.json`, `<user>_part2.json`, ...).
  - Supports `--split-mode` (`random` vs `sequential`) and deterministic `--seed` for reproducible, unbiased data partitioning, multiplying the blind A/B evaluation sample size (e.g., 10 participants $\times$ 3 slices = 30 evaluation pairs).
- **Status:** Tooling complete & verified; peer recruitment & data collection in progress (Antoine / Team).
- > [!CAUTION]
  > Must be anonymised before storage. Do not commit real user data to git.

---

## Models & API Configuration

| Parameter | Value | Rationale |
|---|---|---|
| LLM provider | DeepSeek API | OpenAI-compatible, high reasoning capability, very low latency & cost |
| Extraction agents (`profiler`, `knowledge_retriever`, `art_director`) | `deepseek-chat` | Fast classification, keyword extraction & semantic re-ranking |
| Reasoning agents (`writers_room`, `director`, `scriptwriter`) | `deepseek-reasoner` (DeepSeek R1) | High-order chain-of-thought reasoning for diverse pitch ideation, critical pitch selection, and sharp narrative drafting |
| Baseline model | `deepseek-chat` | Neutral zero-shot comparison with verbatim dialogue & naive top-1 image retrieval |
| Temperature (profiler) | 0.4 | Low — factual extraction |
| Temperature (knowledge brainstorm) | 0.7 | Medium — creative angle ideation |
| Temperature (script draft/refine) | N/A (`deepseek-reasoner`) | Managed internally by R1 deterministic reasoning |
| News & Knowledge API | NewsAPI.org + MediaWiki (Wikipedia REST) | Multi-tiered live API client with evergreen lore lookup and LLM Tier 3 fact synthesis fallback |
| Image API | Serper.dev Google Images (`50 QPS`, 2,500 free queries; no fallback providers) | Switched primary provider after empirical 10-query benchmark; hardened against watermarked stock CDNs, social login walls, and non-image links via live reachability check; omits scene images cleanly (`[]` / `null`) on error or miss instead of failing over |
| Speech Synthesis (TTS) | Microsoft Edge Neural TTS (`edge-tts`) | Broadcast-quality neural voices (`fr-FR-HenriNeural`, `en-US-ChristopherNeural`, etc.) with standalone `ffmpeg` duration probing; graceful offline fallback to `pyttsx3`/`espeak` |
| DeepSeek Account Balance | **$3.35 USD** (live) | Sufficient for ~615+ full 7-phase multi-agent runs or ~11,000 baseline runs |

> [!TIP]
> For the report's **Reproducibility Statement**, log exact model name, API access date, and decoding params for every experiment run. These are saved in each `storyboard_*.json` output file.

---

## Prompts (Versioned)

All system prompts are externalized in `prompts/*.txt` as a versioned plain-text registry to maintain clean git diffs and ensure scientific reproducibility.

| Agent / Component | Prompt File | Version | Role / Focus | Last Updated |
|---|---|---|---|---|
| ProfilerAgent (Round 1) | `prompts/profiler_r1_macro.txt` | v2.0 | Macro history scan & targeted video selection | 2026-10-31 |
| ProfilerAgent (Round 2) | `prompts/profiler_r2_synthesis.txt` | v2.0 | Fine-grained profile synthesis from spoken transcripts | 2026-10-31 |
| WritersRoomAgent | `prompts/writers_room.txt` | v3.0 | 3 distinct candidate pitches adapted to user vibe (R1) | 2026-10-31 |
| DirectorAgent | `prompts/director.txt` | v3.0 | Pitch critique, winning selection & production notes (R1) | 2026-10-31 |
| KnowledgeRetrieverAgent | `prompts/news_targeted_research.txt` | v3.1 | Domain-targeted factual search queries (news, encyclopedia, technical) | 2026-11-09 |
| ScriptwriterAgent (Draft) | `prompts/scriptwriter_draft.txt` | v3.1 | 60s storyboard narrative execution (verbatim dialogue, zero placeholders, R1) | 2026-11-09 |
| ArtDirectorAgent (Query) | `prompts/image_query.txt` | v3.2 | Concrete physical nouns (no camera jargon) | 2026-10-31 |
| ArtDirectorAgent (Curate) | `prompts/art_director_selection.txt` | v3.2 | Art director semantic re-ranking of photo candidates | 2026-10-31 |
| ScriptwriterAgent (Refine) | `prompts/scriptwriter_refine.txt` | v3.1 | Visual grounding with symbolic image refs (verbatim dialogue, zero placeholders, R1) | 2026-11-09 |
| Baseline | `prompts/baseline.txt` | v2.2 | Neutral, competitive zero-shot baseline (verbatim dialogue + naive top-1 image query schema) | 2026-11-09 |
| LLM Judge | `prompts/judge.txt` | v2.1 | 5-dimensional rubric (Relevance, Creativity, Completeness, Coherence, Visual Grounding) with blind photo captions & Anti-Regurgitation Penalty | 2026-11-09 |

---

## Evaluation Plan

### Automated (LLM-as-a-Judge)
- **Framework:** `evaluation/llm_judge.py` evaluates storyboards either pairwise (`--compare A B` or `--preset deepseek/gpt4o`) or individually, applying strict position debiasing ($A/B$ followed by $B/A$ order swapping).
- **5-Dimensional Multi-Criteria Rubric (`prompts/judge.txt`, v2.1):**
  - **1. Relevance:** Does the storyboard connect with ANY genuine interest cluster in the watch history (multi-facet directive)? Merely rehashing a watched video is penalized.
  - **2. Creativity:** Is the narrative fresh and original? Remakes/rehashes of watched videos are strictly **CAPPED at 2/5 (or ≤4.0/10)** under the Anti-Regurgitation Penalty.
  - **3. Completeness:** Are all storyboard fields populated, coherent, and producible (visual descriptions, subtitles, verbatim audio dialogue cues)?
  - **4. Coherence:** Narrative pacing, tension, hook payoff, and flow across the 60-second arc.
  - **5. Visual Grounding:** Unblinded photographic asset fidelity — how faithfully retrieved photo captions (`Visual Asset Photo: {image_desc}`) ground the narrative scenes (without exposing URLs or system identities).
- **Dual Evaluation Outputs:**
  - **1–5 Likert numerical grades** per dimension (with decimal averages).
  - **Explicit pairwise winner verdicts** for each dimension (`relevance_winner`, `creativity_winner`, `completeness_winner`, `coherence_winner`, `visual_grounding_winner`) plus `overall_winner` (`"A" | "B" | "tie"`).
- **Aggregations:** Dimension-level win rates, ties, and position consistency percentages exported to `summary.txt` and `summary.csv`.
- **Legacy Cleanup:** Obsolete, ungrounded `evaluation/evaluator.py` cleanly removed.
- **Judge models supported:** `deepseek-chat` (default budget preset), `deepseek-reasoner` (DeepSeek R1), and `gpt-4o` (gold standard benchmark).

### Human Evaluation (Blind A/B)
- Real users see two storyboards (or rendered video sketches) — one from each system
- Blind: users do not know which is multi-agent vs. baseline
- Metric: preference vote + optional free-text reason
- **Status:** 🔲 TODO (Max)

### Preliminary Benchmark Results (Pilot Pair on `example.json`)

Evaluated with `evaluation/llm_judge.py --compare` using `deepseek-chat` with the **Anti-Regurgitation, Multi-Facet, & Visual-Narrative Link Metadata Rubric (v3.9)**:

| Metric (1–5 Likert) | Multi-Agent Pipeline (`20261109_152317_88655d48`) | Single-Agent Baseline (`baseline_20261109_152556_0a6f118c`) | Delta ($\Delta$) | Judge Rationale (from Link Metadata & Rubric) |
|---|---|---|---|---|
| **Relevance** | **5.0** / 5 | 2.0 / 5 | +3.0 | Multi-agent extends Lyon food interest into forensic bouchon culture; baseline merely remakes watched 'How to make croissants' video (penalized under anti-regurgitation rule) |
| **Creativity** | **5.0** / 5 | 2.0 / 5 | +3.0 | **Baseline CAPPED at 2/5** under Anti-Regurgitation penalty for copying watched video premise; multi-agent invents an original "secret handshake" codebook narrative |
| **Completeness** | **5.0** / 5 | 5.0 / 5 | 0.0 | Both systems specify visual descriptions, punchy subtitles, and verbatim producible audio cues |
| **Coherence** | **5.0** / 5 | 4.0 / 5 | +1.0 | Multi-agent builds an engaging forensic arc with exam metaphor and payoff ritual |
| **Visual Grounding** | **3.0** / 5 | 2.0 / 5 | **+1.0** | **Art Director advantage:** Multi-agent semantic re-ranking selected accurate bistro/menu photos; baseline naive top-1 search returned badly mismatched photos (e.g. dumpling folding & chopsticks for croissant scenes) |
| **Pairwise Win Rate (Overall)** | **100.0% (2 / 2)** | 0.0% (0 / 2) | — | **Winner: Multi-Agent Pipeline in both AB and BA orders** (100% position consistency) |

#### Pairwise Multi-Criteria Win Rate & Grade Breakdown (both orders judged):
| Dimension | Agentic Grade (/5) | Baseline Grade (/5) | Grade Delta | Agentic Wins | Baseline Wins | Ties | Agentic Win Rate (excl. ties) | Position Consistency |
|---|---|---|---|---|---|---|---|---|
| **Relevance** | **4.50** | 3.00 | **+1.50** | **2** | 0 | 0 | **100.0%** | **100.0%** |
| **Creativity** | **5.00** | 1.50 | **+3.50** | **2** | 0 | 0 | **100.0%** | **100.0%** |
| **Completeness** | **5.00** | 3.50 | **+1.50** | **2** | 0 | 0 | **100.0%** | **100.0%** |
| **Coherence** | **5.00** | 3.50 | **+1.50** | **2** | 0 | 0 | **100.0%** | **100.0%** |
| **Visual Grounding** | **4.00** | 2.00 | **+2.00** | **2** | 0 | 0 | **100.0%** | **100.0%** |
| **OVERALL** | **4.70** | 2.70 | **+2.00** | **2** | 0 | 0 | **100.0%** | **100.0%** |

---

## Milestone Checklist

### October 29 MVP (Complete & Exceeded)
- [x] Project skeleton created
- [x] SharedState blackboard defined with markdown and JSON serialization
- [x] BaseAgent abstract class with unified `.run(state)` contract
- [x] ProfilerAgent with two-round deep profiling and spoken transcript tool integration
- [x] YouTube subtitle & metadata extractor (`src/tools/youtube_tool.py`) without API keys
- [x] Client-side YouTube Shorts history collector bookmarklet (`docs/collector.js` + `docs/index.html`) hosted on GitHub Pages
- [x] NewsRetrieverAgent with LLM query brainstorming focused on infotainment
- [x] ScriptwriterAgent with 2-pass Draft + Refine architecture & forensic storytelling
- [x] ImageSearchAgent with scene-guided queries & symbolic IDs (`img1`, `img2`, ...)
- [x] Programmatic image URL resolution (zero URL hallucination)
- [x] Orchestrator pipeline with 5-phase execution
- [x] Single-agent baseline aligned with identical I/O contract and tested
- [x] Config and .env setup (with automatic python-dotenv loading)
- [x] Unit tests passing (`python3 -m unittest discover tests` — 6 unit tests)
- [x] End-to-end pipeline run verified on real DeepSeek API and 94 real watch history items
- [x] Real API integration for news (NewsAPI.org live in `src/tools/news_tool.py`)
- [x] Real API integration for images (Unsplash API live in `src/tools/image_tool.py`)
- [x] Baseline tested and producing output on example and real user data

### Mid-November Milestones (v3.0 – v4.1 Architecture & Tooling Complete)
- [x] Writers' Room + Director showrunner architecture with 3 candidate pitches (`src/agents/writers_room_agent.py`, `src/agents/director_agent.py`)
- [x] Hybrid model routing with DeepSeek R1 reasoning for ideation, selection, and scriptwriting
- [x] Multi-tiered knowledge retriever (NewsAPI + Wikipedia + LLM Synthesis) preventing information starvation (`src/tools/knowledge_tool.py`)
- [x] Art Director 4-candidate semantic re-ranking pass for stock photo assets (`src/agents/art_director_agent.py`)
- [x] Single-agent baseline visual parity with naive top-1 image retrieval for fair comparative testing (`baseline/single_agent_baseline.py`)
- [x] Participant watch history slicer with `--split`, `--split-mode`, and deterministic `--seed` (`src/tools/import_user_history.py`)
- [x] Demo video sketch renderer with Microsoft Edge Neural TTS + dynamic subtitles (`storyboard_to_video.py`)
- [x] Automated 5-dimensional pairwise LLM judge with position debiasing and visual caption grounding (`evaluation/llm_judge.py`)
- [x] Redundant legacy `evaluation/evaluator.py` cleanly removed
- [x] Unit test suite expanded to 24 passing tests (`tests/test_pipeline.py`)
- [x] Authoritative agent and developer operating manual created (`GEMINI.md`)

### Final Report & Evaluation Phase (Target: Dec 4)
- [x] Automated evaluation implemented (`evaluation/llm_judge.py` with batch, `--compare`, and 5D criteria)
- [ ] Complete peer watch history collection (10–15 participants partitioned via `--split 3`)
- [ ] Run full automated benchmark matrix across collected histories (Max)
- [ ] Conduct blind human preference A/B testing on rendered neural video sketches (Max)
- [ ] Synthesize benchmark results tables, figures, and win-rate plots
- [ ] Expand living error analysis (CS-01 through CS-09) into final conference paper
- [ ] Draft 6–8 page conference-style report in LaTeX (`Assignment/main.tex`) (Charles lead)
- [ ] Deliver live project presentation & demo (Nov 24–26) (All)

---

## Error & Failure Case Studies (Living Error Analysis)

Documented failure modes encountered during development, their root causes, and their engineered mitigations (directly populating the final report's *Error Analysis* section):

### CS-01: Direct Image URL Hallucination by LLM
- **Component:** `ScriptwriterAgent` / Pipeline Output
- **Trigger / Symptom:** In early v0.1 prototypes, prompts asked the LLM to provide image URLs directly for each scene. The LLM generated plausible-looking Unsplash URLs (e.g., `https://images.unsplash.com/photo-151...`) which consistently returned HTTP 404 Not Found errors.
- **Root Cause:** Large language models cannot query the live web to check link validity during generation; they hallucinate structurally convincing but nonexistent URLs.
- **Mitigation & Fix:** Decoupled creative image selection from asset resolution. `ImageSearchAgent` indexes retrieved assets into `state.image_context` using discrete symbolic IDs (`img1`, `img2`, ...). The `ScriptwriterAgent` in Phase 7 references *only* these symbolic keys (`"image_ref": "img1"`). Downstream, `Storyboard.resolve_image_urls()` programmatically binds real, validated CDN URLs to each scene.
- **Paper Section:** *Methodology: Visual Grounding & Asset Decoupling; Error Analysis.*

### CS-02: Stock Photo Search Degradation from Cinematography Jargon
- **Component:** `ImageSearchAgent` / `src/tools/image_tool.py`
- **Trigger / Symptom:** When queries contained cinematic descriptions like `"split screen teenager macro exploded view 8k"`, the Unsplash API returned 0 results or completely irrelevant images (e.g. an outdoor basketball hoop for a scene about writing with a pen).
- **Root Cause:** Stock image databases index physical nouns and real-world subjects, not camera shot types, editing techniques, or resolution buzzwords.
- **Mitigation & Fix:** Implemented `clean_stock_query()` in `src/tools/image_tool.py` to strip camera terminology, introduced a core noun fallback, and added an Art Director semantic re-ranking pass (`_build_art_director_prompt`) that inspects candidate descriptions across 4 photos per scene to select the most representative asset.
- **Paper Section:** *Methodology: Tool-Augmented Visual Search; Error Analysis.*

### CS-03: Reasoning Model Token Exhaustion & Naive JSON Stripping
- **Component:** `src/core/llm_client.py` & Agent Parsers (`WritersRoomAgent`, `DirectorAgent`, `ScriptwriterAgent`)
- **Trigger / Symptom:** When switching to `deepseek-reasoner` (DeepSeek R1), agents with `max_tokens=1500` or `2000` occasionally returned empty `content` strings (`finish_reason: length`), or conversational thinking preambles/postambles broke naive `removeprefix("```json")` parsers, causing all pitches and scene data to drop to fallback defaults.
- **Root Cause:** In DeepSeek R1, internal chain-of-thought tokens (`reasoning_content`) count against the request's total completion token limit (`max_tokens`). If `max_tokens` is small, reasoning tokens consume the entire budget before the final answer can be emitted. Additionally, reasoning models frequently wrap responses in markdown or include conversational explanations outside code fences.
- **Mitigation & Fix:** (1) Implemented automatic token headroom expansion in `chat_completion()`: for reasoning models, `call_max_tokens = max(max_tokens, 4096)`. (2) Engineered a bulletproof multi-strategy parser `extract_json()` in `src/core/llm_client.py` using regex block extraction, outermost brace matching, and trailing comma cleaning, rolling it out to all agents.
- **Paper Section:** *Methodology: Hybrid Model Routing; Error Analysis.*

### CS-04: Evasive Placeholder Dialogue in Generated Scripts
- **Component:** `ScriptwriterAgent` & `SingleAgentBaseline`
- **Trigger / Symptom:** Storyboards occasionally contained meta-descriptions or evasive placeholders instead of concrete dialogue (e.g., `"A character says an innocent line..."`, `"Zoom on the key quote..."`, `"Explains the double meaning..."`). Downstream text-to-speech (TTS) in `storyboard_to_video.py` would read these meta-statements literally rather than speaking authentic character lines.
- **Root Cause:** In the absence of strict negative constraints, LLMs often default to descriptive stage directions or meta-summaries of what a character ought to say rather than writing out the verbatim script lines.
- **Mitigation & Fix:** Added explicit `CRITICAL NEGATIVE CONSTRAINTS (NO PLACEHOLDERS)` to `prompts/scriptwriter_draft.txt`, `prompts/scriptwriter_refine.txt`, and `prompts/baseline.txt`. Strict rules forbid evasive phrases, enforce 100% producible verbatim dialogue in both `subtitle` and `audio_cue`, and provide concrete positive examples.
- **Paper Section:** *Methodology: Prompt Engineering & Negative Constraints; Error Analysis.*

### CS-05: Information Starvation from 30-Day NewsAPI Window on Evergreen Lore & Science
- **Component:** `NewsRetrieverAgent` / `src/tools/knowledge_tool.py`
- **Trigger / Symptom:** When researching evergreen concepts (e.g. 1990s cartoon controversy, mathematical paradoxes, vintage inventions), NewsAPI returned 0 articles (limited to past 30 days of breaking news). Downstream scriptwriters received empty or generic `[Archived Info]` placeholders, starving the narrative of verifiable facts, dates, and quotes.
- **Root Cause:** Breaking news search APIs are poorly suited for historical, scientific, or pop-culture lore which spans decades rather than the last 30 days.
- **Mitigation & Fix:** Engineered `src/tools/knowledge_tool.py` with multi-tiered lookup: (1) Domain classification (`news`, `encyclopedia`, `technical`), (2) Tier 1 NewsAPI for current events, (3) Tier 2 MediaWiki Wikipedia API for evergreen lore/science, (4) Tier 3 LLM Knowledge Synthesis Fallback that extracts verifiable historical dates, episode names, creator quotes, and exact numbers when external APIs return 0 results.
- **Paper Section:** *Methodology: Multi-Tiered Knowledge Retrieval; Error Analysis.*

### CS-06: Superficial Keyword Regurgitation & Baseline Remake Bias
- **Component:** `evaluation/llm_judge.py` & `prompts/judge.txt`
- **Trigger / Symptom:** In initial evaluations, single-agent baseline storyboards that merely remade or copied watched videos verbatim (e.g. user watched "How to make croissants from scratch" and baseline produced "10-minute croissant hack") were awarded high relevance (8.5/10) and creativity (8.0/10) by the LLM judge due to surface title keyword matching.
- **Root Cause:** Standard judge prompts lacked negative constraints against regurgitation and failed to distinguish between genuine personalization (extending user taste to an unseen curiosity or fresh angle) and uninspired remakes/reruns.
- **Mitigation & Fix:** (1) Introduced explicit **Multi-Facet Relevance Directive**: a storyboard is relevant if it connects with *any* genuine interest cluster in the watch history (not just majority frequency). (2) Introduced strict **Anti-Regurgitation Penalty**: if a storyboard merely copies, rehashes, or retells an already-watched video, Creativity is **strictly capped at 2/5 (or ≤4.0/10)**, and relevance is penalized because users do not want reruns in their feeds. In comparative evaluations, baseline score dropped from 8.50 to 6.75, widening the multi-agent advantage to +2.37 Δ.
- **Paper Section:** *Methodology: Evaluation & Position-Debiased Rubrics; Error Analysis.*

### CS-07: Robotic Monotone Audio in Video Sketch Renderer Harming Human Blind Evaluations
- **Component:** `storyboard_to_video.py`
- **Trigger / Symptom:** Initial video sketches rendered with local offline TTS (`pyttsx3`/`espeak`) produced flat, robotic, metallic narration. In preliminary listening tests, listeners found it hard to judge narrative pacing and comedic timing, conflating low audio engine naturalness with script quality.
- **Root Cause:** Traditional offline concatenative/formant speech engines lack natural prosody, emotional inflection, and phoneme smoothing.
- **Mitigation & Fix:** Re-architected `storyboard_to_video.py` to use Microsoft Edge Neural TTS (`edge-tts`) as the default primary engine, supporting rich multilingual neural voices (e.g. `fr-FR-HenriNeural`, `en-US-ChristopherNeural`, `es-ES-AlvaroNeural`), while preserving local `pyttsx3`/`espeak` as graceful offline fallbacks. Video sketch audio now exhibits human-level cadence, natural breath pauses, and authentic pronunciation without cloud API subscription costs.
- **Paper Section:** *Methodology: Demo Sketch Rendering & Human Evaluation; Error Analysis.*

### CS-08: Host Environment Failure on Missing ffprobe Dependency
- **Component:** `storyboard_to_video.py`
- **Trigger / Symptom:** On Linux test environments without a full `ffmpeg` package from OS package managers, `storyboard_to_video.py` crashed during scene length calculation with `FileNotFoundError: [Errno 2] No such file or directory: 'ffprobe'`.
- **Root Cause:** Python video generation scripts commonly assume `ffprobe` is installed alongside `ffmpeg`. However, lightweight environments, conda environments, and minimal Docker containers frequently install only `ffmpeg` or `imageio-ffmpeg` without the standalone `ffprobe` executable.
- **Mitigation & Fix:** Rewrote `audio_duration()` to check `shutil.which("ffprobe")`. If missing, it falls back to running `ffmpeg -i <file> -f null -` and extracting the `Duration: HH:MM:SS.ss` timestamp via regex from ffmpeg's stderr output, plus a native Python `wave` module fallback for WAV files. Coupled with bundled `imageio-ffmpeg` auto-detection, rendering is now 100% self-contained on any environment.
- **Paper Section:** *Methodology: Robust Multimedia Tooling; Error Analysis.*

### CS-09: History Collector Web Tool 404 from GitHub Pages Visibility Policy
- **Component:** `tools/history_collector/` & GitHub Pages hosting
- **Trigger / Symptom:** Participants accessing the live client-side data collection URL (`https://charlesdupouy.github.io/nlp-multiagent-storyboarding/`) were greeted by an HTTP 404 error page.
- **Root Cause:** The GitHub repository visibility had been toggled to `private`. On standard personal/free GitHub accounts, GitHub Pages for private repositories is disabled by platform policy (`Page is disabled because current plan does not support private GitHub Pages (HTTP 422)`), causing GitHub to automatically suspend pages builds and unpublish the deployed site.
- **Mitigation & Fix:** Restored repository visibility to `public` via GitHub CLI (`gh repo edit --visibility public`), pushed current release commits to `origin/main`, and confirmed deployment workflow execution (`pages-build-deployment` finished with `success`). Site accessibility verified via HTTP 200 checks on the live endpoint.
- **Paper Section:** *Methodology: Data Collection & Ethics; Reproducibility.*

### CS-10: Section Header ("Today" / "Saturday") Scraping in Browser Collector & Automatic oEmbed Title Enrichment
- **Component:** `tools/history_collector/collector.js` & `src/tools/import_user_history.py`
- **Trigger / Symptom:** When participant watch histories were exported from the client-side collector, video titles were populated with date strings (e.g., all 26 titles in `max.json` were `"Today"` and all 104 titles in `charles.json` were `"Saturday"`).
- **Root Cause:** YouTube's desktop history feed groups items under `<ytd-item-section-renderer>` wrappers, where the section header element (`#title`, `h3`) displays the viewing date. Naive text extraction picked up the date header text instead of the inner video title inside `<ytd-reel-item-renderer>`.
- **Mitigation & Fix:** (1) Added `is_generic_title()` in `src/tools/import_user_history.py` to identify placeholder titles, date labels, and day headers. (2) Implemented automatic, concurrent resolution via YouTube's public oEmbed API (`enrich_titles_in_parallel()`) which pulls the exact video title and creator author name with 0 API keys required. (3) Added `--cap` to support exact subset sizing (e.g. 90 videos for $3 \times 30$ splits) and `--force-enrich` to refresh existing JSON exports without re-downloading.
- **Paper Section:** *Methodology: Data Collection & Curation; Error Analysis.*

### CS-11: Silent Phase 2/3 Collapse, Transcript-Blind Judge Asymmetry & Unsplash 50 req/hr Exhaustion
- **Component:** `WritersRoomAgent`, `DirectorAgent`, `src/tools/image_tool.py` & `evaluation/llm_judge.py`
- **Trigger / Symptom:** During the full v4.5 batch regeneration (`max` + `charles_part1..3`), three compounding failure modes were observed: (1) `charles_part2` agentic developed `0` candidate pitches in Phase 2, causing `DirectorAgent` (Phase 3) to skip deliberation (`11` LLM calls instead of `12`) and forcing Phase 4/5 to fall back from a commissioned pitch to raw-profile topics (`[vid18]` Mazatlán cave); (2) the title-only LLM judge penalized that `charles_part2` storyboard (`2/5` pairwise relevance) because `[vid18]`'s clickbait YouTube title (`ÉPISODE 592 : SPÉCIAL HALLOWEEN 2026`) gave no indication of its spoken Mazatlán content; and (3) from `charles_part1` baseline Scene 6 onward, `28/33` scenes silently received synthetic `https://example.com/images/...jpg` URLs and rendered solid-color placeholder MP4 frames (dropping judge `visual_grounding` to `2.50`/`3.12`).
- **Root Cause:** (1) `_parse_pitches()` silently caught JSON parse exceptions and returned `[]` with zero logging or retry; (2) our title-only judge evaluates relevance strictly against video titles (`data/user_histories/*.json`), creating a **Transcript-Blind Judge Asymmetry** whenever the Profiler discovers interests inside spoken captions of opaque titles; and (3) running v4.4 and v4.5 within one 60-minute window exhausted Unsplash Demo's `50 req/hour` quota (`HTTP 403: Rate Limit Exceeded`), while `_unsplash_fetch_candidates()` silently returned `[]` and repeated 2 fallback queries per scene before hitting `_mock_search_images()`.
- **Mitigation & Fix:** Added parse-error warning logging and automatic 1-shot strict-JSON retry in both `WritersRoomAgent` (`src/agents/writers_room_agent.py`) and `DirectorAgent` (`src/agents/director_agent.py`). Added `UnsplashRateLimitError` short-circuiting on HTTP `403`/`429` with clear `[TOOL WARNING]` logging and automatic `PEXELS_API_KEY` failover in `src/tools/image_tool.py` (while flagging the 50 req/hr Unsplash bottleneck ahead of the 15-user human study).
- **Paper Section:** *Error Analysis: Pipeline Resilience, Rate-Limit Handling & Evaluation Methodology.*

### CS-12: Opening 1–2s Audio Truncation on Playback from `0.3s` Intro Lead-In & Per-Clip AAC Concatenation
- **Component:** `storyboard_to_video.py`
- **Trigger / Symptom:** When playing back rendered MP4 video sketches, listeners consistently heard the opening 1–2 seconds clipped (first word cut in half before audio settled into normal playback).
- **Root Cause:** Quantitative waveform and container inspection across all 8 generated MP4s revealed two compounding issues: (1) shortening `DEFAULT_PADDING` from `1.0s` to `0.3s` placed the first spoken speech energy at `0.505s–0.528s` into the video, giving media players, OS audio sinks (PipeWire/PulseAudio idle wake-up), and Bluetooth devices insufficient startup buffer before speech began; and (2) encoding AAC (`-c:a aac`) independently on every intermediate `.mp4` clip and stitching with `-f concat -c copy` introduced AAC encoder priming / edit-list timing shifts (`Video stream start: 0.023203s` vs `Audio: 0.000000s`) without `faststart` container indexing.
- **Mitigation & Fix:** (1) Preserved `DEFAULT_PADDING = 0.3s` for crisp inter-scene cuts while introducing `DEFAULT_INTRO_PADDING = 1.5s` (`--intro-padding`) specifically for Scene 1 (`idx == 0`), pushing opening speech onset to `~1.68s` so hardware/software audio wake-up finishes silently; (2) switched intermediate scene clips to lossless PCM inside `.mkv` (`-c:a pcm_s16le`) and encoded a single continuous AAC stream at final concat (`-c:v copy -c:a aac -b:a 128k -movflags +faststart`), eliminating stream start offsets (`start: 0.000000s` on both video and audio).
- **Paper Section:** *Methodology: Demo Sketch Rendering & Human Evaluation Tooling.*

### CS-13: Missing First-Watch Causal Setup & In-Scene Character Roleplay in Voiceover Scripts
- **Component:** `WritersRoomAgent`, `DirectorAgent`, `ScriptwriterAgent`, & `SingleAgentBaseline` prompts
- **Trigger / Symptom:** Human inspection of rendered v4.5.1 videos surfaced two script comprehensibility failures: (1) `charles_part1` agentic (`agentic_charles_part1_20261111_155347`, *Les 100 prisonniers*) compressed complex riddle rules, front-loaded an unexplained citation (`"Miltersen et Gál, 2003 :"`), and omitted *why* following numbers forms closed loops — leaving a first-time viewer lost; (2) `charles_part3` agentic (`agentic_charles_part3_20261111_161105`, *Le piège de Gilbreath*) alternated between in-scene magician roleplay addressing an off-screen spectator (`"je te les donne... Maintenant, mélange vraiment... Retourne :"`), and 3rd-person commentary (`"Le tricheur ne mémorise rien"`), sounding incoherent in a single-TTS slideshow (similar mild character quote shifts appeared in `charles_part3` baseline).
- **Root Cause:** `prompts/scriptwriter_draft.txt` and `prompts/baseline.txt` previously permitted `"narrative dialogue"` / `"voiceover dialogue"`, while `prompts/scriptwriter_refine.txt` omitted standalone clarity and narrator-voice guardrails altogether; tight compression in 5 scenes also encouraged mathematical shorthand jumps over explicit step-by-step setup.
- **Mitigation & Fix:** (1) Removed `"dialogue"` wording and added explicit `STRICTLY SINGLE EXTERNAL VOICEOVER NARRATOR` negative constraints banning character roleplay, spectator commands, and character-to-character quotes across `writers_room.txt`, `director.txt`, `scriptwriter_draft.txt`, `scriptwriter_refine.txt`, and `baseline.txt` (plus `_DEFAULT` constants and `_build_refine_prompt()`). (2) Mandated `FIRST-WATCH UNDERSTANDABILITY` (setup/rules before twists or citations, explicit causal *"why"*, recommending 6 scenes / ~125–150 words). Regenerated `charles_part1` agentic (`*Le trou secret de votre stylo*`, `59.60s`), `charles_part3` agentic (`*Couper 100 fois un jeu de cartes*`, `58.66s`), and `charles_part3` baseline (`*Le paradoxe des anniversaires*`, `58.77s`), improving LLM-judge `coherence` (`5.00` abs; `71.4%` pairwise win rate) and overall preference (`62.5%` win rate, `4.25 vs 3.77`).
- **Paper Section:** *Prompt Engineering, Human-Evaluated Script Quality & Error Analysis.*

### CS-14: Profiler Outlier Tunnel Vision & Judge Pairwise Relevance Asymmetry
- **Component:** `ProfilerAgent`, `WritersRoomAgent`, `DirectorAgent` (`state.py`) & `evaluation/llm_judge.py`
- **Trigger / Symptom:** Under `v4.5.2`, `llm_judge.py` scored agentic `relevance` cleanly ahead in Absolute mode (`4.25 vs 3.50`, `+0.75`) yet gave agentic only a `28.6%` (`2-5-1`) pairwise `relevance` win rate (`3.62 vs 4.38`). Side-by-side inspection showed two distinct issues: (1) upstream **Profiler Cluster Blindness / Outlier Tunnel Vision** — on `max`, Round 1's transcript focus on 1 niche clip (`[vid4]` Chubbyemu diet soda) eclipsed the `8/26` Game of Thrones cluster, and on `charles_part1..2` unranked `interests[]` elevated 1-off items (4-colour pen `[vid14]`, dry shampoo) over recurring `#mathctober` (`7/30`) and French comedy (`8/30`) clusters; (2) downstream **Pairwise Relevance Asymmetry** in `build_pair_prompt()`, which enforced the Anti-Regurgitation `3/5` cap under `creativity` (`METRICS`) but omitted it under `1. relevance` in `PAIR_SYSTEM`/`build_pair_prompt()`, letting baseline remakes (`How Many Folds To Reach The Moon?`, `How Far Can the Blocks Lean?`) tie or beat agentic on pairwise relevance.
- **Root Cause:** `UserProfile` carried no frequency-weighted `dominant_clusters`, Round 2 saw only the 2–3 inspected transcript videos rather than the full title list, and `build_pair_prompt()` evaluated `1. relevance` solely by keyword overlap rather than novel extension.
- **Mitigation & Fix:** Added `dominant_clusters` + `format_dominant_clusters()` on `UserProfile` (`src/core/state.py`), required Round 1 to emit frequency-ranked `interest_clusters` and allocate ≥2 transcript picks to top clusters, passed `COMPLETE WATCH HISTORY` into Round 2 (`src/agents/profiler_agent.py`), mandated cluster-anchored Pitch #1/#2 diversity in `WritersRoomAgent`/`DirectorAgent`, and aligned `build_pair_prompt()` item `1. relevance` with `METRICS["relevance"]`. Across the regenerated `n=4` benchmark (`max`, `charles_part1..3`), pairwise `relevance` flipped from `28.6%` (`2-5-1`) to **`71.4%` (`5-2-1`, `4.25 vs 3.50`)** and Abs `relevance` reached **`4.75 vs 3.50` (`+1.25`)**.
- **Paper Section:** *Error Analysis: Interest Frequency Preservation & Evaluation Prompt Consistency.*

### CS-15: DeepSeek Reasoner 8,192-Token Ceiling Truncating Phase 7 Refine Scenes
- **Component:** `src/core/llm_client.py` & `ScriptwriterAgent.refine()` (`src/agents/scriptwriter_agent.py`)
- **Trigger / Symptom:** During `v4.6` regeneration of `charles_part1` (`agentic_charles_part1_20261111_170104`, Braess's paradox), Phase 5 drafted all `6` scenes (`60s`), and Phase 6 fetched all `6` Unsplash photos (`img1..img6`), yet Phase 7 emitted only `4` scenes (`37.1s` MP4) and cut off before the narrative payoff — dropping Abs `completeness`/`coherence` to `3/5`.
- **Root Cause:** Even at `reasoning_effort="low"`, `deepseek-reasoner` spent `7,405` thinking tokens in Phase 7, hitting `call_max_tokens = 8192` after Scene 4. `extract_json()`'s bracket-repair step (`src/core/llm_client.py`) parsed the truncated 4-scene prefix without raising an error, and `ScriptwriterAgent.refine()` overwrote the 6-scene draft without checking `len(refined_scenes) < len(draft_scenes)`.
- **Mitigation & Fix:** Raised the thinking headroom floor from `8192` to `16384` in `chat_completion()` (`src/core/llm_client.py`) and added a scene-count completeness retry + fallback draft-scene preservation guard (plus `sanitize_scenes()`) in `ScriptwriterAgent.refine()`. Cleanly regenerated `charles_part1` (`agentic_charles_part1_20261111_170818`, `6` scenes, `6/6` Unsplash, `58.48s` MP4, Abs `relevance=5`, `completeness=5`, `coherence=5`).
- **Paper Section:** *Methodology & Systems Engineering: Reasoning-Token Headroom & Output Integrity.*

### CS-16: Visual Benchmarking & Migration from 50 req/hr Unsplash Demo to Hardened Serper.dev Images
- **Component:** `src/tools/image_tool.py`, `prompts/image_query.txt`, `prompts/art_director_selection.txt`, `storyboard_to_video.py`, & `tools/benchmark_image_apis.py`
- **Trigger / Symptom:** With Pexels new API key issuance paused and Unsplash Demo capped at `50 req/hour`, batch generation for a ~15-participant study (~350–700 scene queries) frequently exhausted Unsplash quota (`HTTP 403`). Before switching providers, we built `tools/benchmark_image_apis.py` and evaluated **Openverse**, **Wikimedia Commons**, and **Serper.dev Google Images** across 10 representative multi-noun storyboard queries with real Pillow image download verification.
- **Root Cause / Empirical Findings:** Openverse (`3/10` hits, `7.99s` mean latency, `HTTP 504` timeout on Eiffel Tower, `6/9` hosts blocking bare `"Mozilla/5.0"`) and Wikimedia Commons (`3/10` hits on 5–7 word scene descriptions due to strict AND matching) failed on narrative scene queries. Serper.dev achieved **`10/10` query hits** (`1.61s` mean latency, `1,240 px` median width) and even surfaced un-throttled `images.unsplash.com` / `images.pexels.com` assets, *but* exposed three web-scraping hazards: (1) hidden watermarked CDNs (`abposters.com` serving Adobe Stock `as1.ftcdn.net`, `plus.unsplash.com`, `classroomclipart.com`) slipping past naive `shutterstock`/`getty` filters; (2) login-walled social hosts (`facebook.com`, `lookaside.fbsbx.com`, `instagram.com`, `tiktok.com`) causing `HTTP 403` / HTML `UnidentifiedImageError` (`0/4` on `q07`); and (3) shopping listings (`ebay.com`, `etsy.com`) appearing for physical objects (`q06`).
- **Mitigation & Fix:** Switched `IMAGE_API_PROVIDER` primary to `serper` (`20` candidates fetched per query, `1` credit) with Unsplash failover in `src/tools/image_tool.py`, adding: (1) expanded watermark-CDN & social-wall domain blocklists (`ftcdn.net`, `abposters.com`, `plus.unsplash.com`, `facebook/instagram/tiktok/fbsbx`); (2) soft demotion of shopping domains behind editorial/photo results plus an Art Director anti-e-commerce selection rule (`prompts/art_director_selection.txt`); (3) streaming HTTP `200` + raster `Content-Type` / magic-byte reachability verification before candidate indexing; and (4) compliant project `User-Agent` handling + `Mozilla/5.0` retry fallback in `storyboard_to_video.py:download_image()`.
- **Paper Section:** *Methodology: Visual Search Trade-offs; Error Analysis.*

---

## Cost & Compute Log

| Date | Run ID | Component | Model | Status / Output | Notes |
|---|---|---|---|---|---|
| 2026-10-30 | `20261030_203738_4988344d` | Multi-Agent Pipeline (v0.1) | `deepseek-chat` | Success (6 scenes) | Initial linear pipeline smoke test |
| 2026-10-30 | `20261030_211708_49b2c57a` | Multi-Agent Pipeline (v1.0) | `deepseek-chat` | Success (7 scenes) | 5-phase feedback loop with scene image search & refinement |
| 2026-10-30 | `20261030_233028_8f65d1f9` | Multi-Agent Pipeline (v2.0) | `deepseek-chat` | Success (6 scenes) | Two-round profiling with spoken transcript extraction; forensic infotainment storyboard on Teletoon |
| 2026-10-30 | `baseline_20261030_234731_1896cb67` | Single-Agent Baseline (Neutral) | `deepseek-chat` | Success (6 scenes) | Scientifically neutral zero-shot baseline on same 94 videos; focused math paradox storyboard |
| 2026-10-31 | `20261031_014151_46ff9e3d` | Multi-Agent Pipeline (v3.0 - Writers' Room) | `deepseek-chat` | Success (5 scenes) | Writers' Room (3 candidate pitches) + Director deliberation + Targeted News Retrieval; Bic 4-color pen storyboard |
| 2026-10-31 | `20261031_015924_4a5dc1c6` | Multi-Agent Pipeline (v3.1 - Live APIs) | `deepseek-chat` | Success (6 scenes) | Fully live tool-augmented run: real NewsAPI articles + real Unsplash photographic CDN assets |
| 2026-10-31 | `20261031_124354_0ec406a8` | Multi-Agent Pipeline (v3.2 - Art Director Curation) | `deepseek-chat` | Success (6 scenes) | Two-phase Art Director image curation: concrete physical queries + candidate description re-ranking |
| 2026-11-09 | `verification_v3.3_hybrid` | Hybrid Reasoning Integration (v3.3) | `deepseek-chat` + `deepseek-reasoner` | Success (20 unit tests) | Validated model assignments: `deepseek-reasoner` for Phases 2, 3, 5/7; `deepseek-chat` for 1, 4, 6. Headroom expansion active |
| 2026-11-09 | `20261109_152317_88655d48` | Multi-Agent Pipeline (v3.6 - Hybrid R1 + Multi-Tier Knowledge) | `deepseek-chat` + `deepseek-reasoner` | Success (6 scenes with images) | End-to-end run on `example.json`. Multi-tier Wikipedia extraction (2012 charter, tablier de sapeur, pot de vin), R1 verbatim dialogue, Unsplash photographic grounding. 11,836 prompt + 13,463 completion across 12 calls. Est. Cost: $0.00543 USD. Judge Score: 9.38 / 10.0 |
| 2026-11-09 | `baseline_20261109_152556_0a6f118c` | Single-Agent Baseline (v3.5 - Visual Parity) | `deepseek-chat` | Success (6 scenes with images) | Zero-shot baseline on `example.json` with naive Top-1 Unsplash image retrieval and verbatim dialogue. All 6 scenes populated with real Unsplash CDN URLs. 714 prompt + 826 completion. Est. Cost: $0.00033 USD. Judge Score: 8.50 / 10.0 |
| 2026-11-10 | `agentic_max_20261110_095334` | Multi-Agent Pipeline (v4.2) | `deepseek-chat` + `deepseek-reasoner` | Success (5 scenes + video) | Storyboard on Chubbyemu/PKU diet-soda medical history. Neural video sketch rendered (76.8s). |
| 2026-11-10 | `baseline_max_20261110_100659` | Single-Agent Baseline (v4.2) | `deepseek-chat` | Success (6 scenes + video) | Zero-shot baseline on Omnipotence Paradox. Neural video sketch rendered (58.4s). |
| 2026-11-11 | `agentic_charles_part1_20261111_103251` | Multi-Agent Pipeline (`reasoning_effort=low`) | `deepseek-chat` + `deepseek-reasoner` | Success (6 scenes + video) | `charles_part1` (30 videos): "Le procès de la Gen Z : « Votre Honneur, la prison a-t-elle la fibre ? »". 13,899 prompt + 14,190 completion (5,222 reasoning) across 12 calls in 114.4s. Est. Cost: $0.03872 USD. Neural video sketch rendered (67.9s). |
| 2026-11-11 | `baseline_charles_part1_20261111_103307` | Single-Agent Baseline (VO-only) | `deepseek-chat` | Success (6 scenes + video) | `charles_part1` (30 videos): "Le stylo interdit et la magie des maths". 2,174 prompt + 822 completion in 10.0s. Est. Cost: $0.00053 USD. Neural video sketch rendered (47.8s). |
| 2026-11-11 | `agentic_charles_part2_20261111_103410` | Multi-Agent Pipeline (`reasoning_effort=low`) | `deepseek-chat` + `deepseek-reasoner` | Success (6 scenes + video) | `charles_part2` (30 videos): "Maeva, l'IA « 100 % humaine », affronte le puzzle des blocs". 14,527 prompt + 12,548 completion (4,073 reasoning) across 12 calls in 100.8s. Est. Cost: $0.03547 USD. Neural video sketch rendered (79.6s). |
| 2026-11-11 | `baseline_charles_part2_20261111_103318` | Single-Agent Baseline (VO-only) | `deepseek-chat` | Success (6 scenes + video) | `charles_part2` (30 videos): "Le paradoxe du dernier-né : pourquoi il a droit à tout ?". 2,298 prompt + 872 completion in 11.1s. Est. Cost: $0.00057 USD. Neural video sketch rendered (53.0s). |
| 2026-11-11 | `agentic_charles_part3_20261111_103551` | Multi-Agent Pipeline (`reasoning_effort=low`) | `deepseek-chat` + `deepseek-reasoner` | Success (6 scenes + video) | `charles_part3` (30 videos): "Les oreilles trop brillantes de Joe Musgrove". 12,732 prompt + 13,251 completion (5,438 reasoning) across 12 calls in 102.8s. Est. Cost: $0.03602 USD. Neural video sketch rendered (69.9s). |
| 2026-11-11 | `baseline_charles_part3_20261111_103330` | Single-Agent Baseline (VO-only) | `deepseek-chat` | Success (6 scenes + video) | `charles_part3` (30 videos): "Le paradoxe de Wikipédia : pourquoi vous ne devriez pas donner un centime". 2,207 prompt + 969 completion in 13.0s. Est. Cost: $0.00058 USD. Neural video sketch rendered (64.3s; Scenes 4-5 used placeholder fallback after HTTP 404 on mock URLs). |
| 2026-11-11 | `baseline_max_20261111_145554` | Single-Agent Baseline (v4.4 Reasoner `low`) | `deepseek-reasoner` (`low`) | Success (5 scenes + video) | `max` (26 videos): 2,150 prompt + 2,006 completion (1,281 reasoning) in 16.84s; Est. Cost: $0.00558 USD. Video rendered in 14.14s (57.44s MP4). |
| 2026-11-11 | `agentic_max_20261111_145625` | Multi-Agent Pipeline (v4.4) | `deepseek-chat` + `deepseek-reasoner` | Success (6 scenes + video) | `max` (26 videos): 13,516 prompt + 14,558 completion (5,599 reasoning) in 111.54s (12 calls); Est. Cost: $0.02821 USD. Video rendered in 18.63s (66.33s MP4). |
| 2026-11-11 | `baseline_charles_part1_20261111_145835` | Single-Agent Baseline (v4.4 Reasoner `low`) | `deepseek-reasoner` (`low`) | Success (5 scenes + video) | `charles_part1` (30 videos): 2,321 prompt + 5,233 completion (4,474 reasoning) in 28.55s; Est. Cost: $0.01274 USD. Video rendered in 13.19s (55.90s MP4). |
| 2026-11-11 | `agentic_charles_part1_20261111_145917` | Multi-Agent Pipeline (v4.4) | `deepseek-chat` + `deepseek-reasoner` | Success (6 scenes + video) | `charles_part1` (30 videos): 12,896 prompt + 13,576 completion (5,628 reasoning) in 107.60s (12 calls); Est. Cost: $0.02658 USD. Video rendered in 11.40s (53.98s MP4). |
| 2026-11-11 | `baseline_charles_part2_20261111_150116` | Single-Agent Baseline (v4.4 Reasoner `low`) | `deepseek-reasoner` (`low`) | Success (6 scenes + video) | `charles_part2` (30 videos): 2,445 prompt + 1,956 completion (955 reasoning) in 16.80s; Est. Cost: $0.00563 USD. Video rendered in 13.27s (48.60s MP4). |
| 2026-11-11 | `agentic_charles_part2_20261111_150146` | Multi-Agent Pipeline (v4.4) | `deepseek-chat` + `deepseek-reasoner` | Success (6 scenes + video) | `charles_part2` (30 videos): 12,341 prompt + 13,780 completion (6,717 reasoning) in 107.86s (11 calls); Est. Cost: $0.02816 USD. Video rendered in 13.91s (53.13s MP4). |
| 2026-11-11 | `baseline_charles_part3_20261111_150348` | Single-Agent Baseline (v4.4 Reasoner `low`) | `deepseek-reasoner` (`low`) | Success (5 scenes + video) | `charles_part3` (30 videos): 2,354 prompt + 2,636 completion (1,837 reasoning) in 19.05s; Est. Cost: $0.00707 USD. Video rendered in 10.76s (48.20s MP4; 0 image failures). |
| 2026-11-11 | `agentic_charles_part3_20261111_150418` | Multi-Agent Pipeline (v4.4) | `deepseek-chat` + `deepseek-reasoner` | Success (6 scenes + video) | `charles_part3` (30 videos): 13,378 prompt + 10,593 completion (3,538 reasoning) in 90.00s (10 calls); Est. Cost: $0.02336 USD. Video rendered in 16.68s (54.42s MP4). |
| 2026-11-11 | `judge_deepseek_v4_4` | Automated Pairwise + Absolute Judge | `deepseek-chat` | Success (4 pairs, 16 calls) | Regenerated benchmark on `max` + `charles_part1..3`: 49,526 in / 5,426 out in 9.27s (~$0.0085 USD). Agentic overall win rate 87.5% (`4.42` vs `3.35`, delta `+1.07`). |
| 2026-11-11 | `baseline_max_20261111_152246` | Single-Agent Baseline (v4.5) | `deepseek-reasoner` (`low`) | Success (6 scenes + video) | `max` (26 videos): `'The Murder Trial Decided by a Sword Fight (1819)'`; 2,205 prompt + 1,572 completion (779 reasoning) in 16.24s; Est. Cost: $0.00466 USD. Video rendered in 18.36s (62.92s MP4, 2.04MB, 0 image failures). |
| 2026-11-11 | `agentic_max_20261111_152321` | Multi-Agent Pipeline (v4.5) | `deepseek-chat` + `deepseek-reasoner` | Success (6 scenes + video) | `max` (26 videos): `'The Six Nuclear Bombs That Disappeared Before Anyone Could Count Them'`; 14,701 prompt + 16,292 completion (8,671 reasoning) across 11 calls in 134.54s; Est. Cost: $0.03424 USD. Video rendered in 21.53s (63.90s MP4, 1.52MB, 0 image failures). |
| 2026-11-11 | `baseline_charles_part1_20261111_152557` | Single-Agent Baseline (v4.5) | `deepseek-reasoner` (`low`) | Partial images (6 scenes + video) | `charles_part1` (30 videos): `'42 plis et la Lune : le calcul qui défie vos mains'`; 2,376 prompt + 2,572 completion (1,758 reasoning) in 20.59s; Est. Cost: $0.00694 USD. Video rendered in 12.15s (51.10s MP4, 1.05MB; Scene 6 hit Unsplash 50/hr 403 limit, `bad_image_downloads=1`). |
| 2026-11-11 | `agentic_charles_part1_20261111_152630` | Multi-Agent Pipeline (v4.5) | `deepseek-chat` + `deepseek-reasoner` | Mock images (`403`, 5 scenes + video) | `charles_part1` (30 videos): `'Le coq qui a gagné son procès (et fait trembler la France)'`; 14,934 prompt + 17,754 completion (9,274 reasoning) across 12 calls in 127.70s; Est. Cost: $0.03602 USD. Video rendered in 8.27s (48.24s MP4, 0.89MB; `bad_image_downloads=5` due to Unsplash 50/hr 403 limit). |
| 2026-11-11 | `baseline_charles_part2_20261111_152846` | Single-Agent Baseline (v4.5) | `deepseek-reasoner` (`low`) | Mock images (`403`, 5 scenes + video) | `charles_part2` (30 videos): `"Pourquoi l'eau rend le poulet croustillant"`; 2,500 prompt + 2,001 completion (1,262 reasoning) in 15.23s; Est. Cost: $0.00576 USD. Video rendered in 7.66s (39.04s MP4, 0.63MB; `bad_image_downloads=5`). |
| 2026-11-11 | `agentic_charles_part2_20261111_152909` | Multi-Agent Pipeline (v4.5) | `deepseek-chat` + `deepseek-reasoner` | Degraded (Phase 2/3 skipped + `403`) | `charles_part2` (30 videos): `'La Porte Rouge de Mazatlán : La Vérité sur la Cueva del Diablo'`; Phase 2 JSON parse drop skipped Director (`0 pitches`), falling back to `[vid18]` transcript topic; 12,700 prompt + 15,096 completion (7,234 reasoning) across 11 calls in 111.76s; Est. Cost: $0.02868 USD. Video rendered in 8.91s (49.81s MP4, 0.85MB; `bad_image_downloads=5`). |
| 2026-11-11 | `baseline_charles_part3_20261111_153109` | Single-Agent Baseline (v4.5) | `deepseek-reasoner` (`low`) | Mock images (`403`, 6 scenes + video) | `charles_part3` (30 videos): `'La règle cachée qui transforme un pion en reine'`; 2,409 prompt + 5,312 completion (4,399 reasoning) in 38.69s; Est. Cost: $0.01296 USD. Video rendered in 10.48s (53.84s MP4, 0.89MB; `bad_image_downloads=6`). |
| 2026-11-11 | `agentic_charles_part3_20261111_153159` | Multi-Agent Pipeline (v4.5) | `deepseek-chat` + `deepseek-reasoner` | Mock images (`403`, 6 scenes + video) | `charles_part3` (30 videos): `"Le rouge qui bavait sur ta télé de 2003 — et c'était de la physique, pas ta vue"`; 17,110 prompt + 21,970 completion (12,832 reasoning) across 11 calls in 161.23s; Est. Cost: $0.04704 USD. Video rendered in 9.83s (52.02s MP4, 0.93MB; `bad_image_downloads=6`). |
| 2026-11-11 | `judge_deepseek_v4_5` | Automated Pairwise + Absolute Judge | `deepseek-chat` | Success (4 pairs, 16 calls) | v4.5 batch on `max` + `charles_part1..3` (723.38s total regen wall time): 49,661 in / 5,383 out in 9.83s (~$0.0085 USD). Creativity `100%` agentic pairwise win rate (`4.00` vs `2.88`), Overall `3.88` vs `3.75` (`visual_grounding` degraded to `2.50`/`3.12` due to Unsplash 403 placeholders on `charles_part1..3`). |
| 2026-11-11 | `baseline_charles_part1_20261111_155321` | Baseline (`deepseek-reasoner` low) | `deepseek-reasoner` | Success (5 scenes, 120 VO words) | Clean regen (new Unsplash key, `5/5` Unsplash, `0` failures): 2,376 in / 1,707 out (889 reasoning, 1 call), sb `14.94s`, vid `10.88s` (`52.82s` MP4, `first_sound=1.692s`), ~$0.0050 USD. |
| 2026-11-11 | `agentic_charles_part1_20261111_155347` | Multi-Agent (v4.5) | `deepseek-chat` + `deepseek-reasoner` | Success (5 scenes, 126 VO words) | Clean regen (new Unsplash key, `3` pitches, `5/5` Unsplash, `0` failures): 15,784 in / 18,017 out (10,264 reasoning, 11 calls), sb `138.70s`, ~$0.0388 USD. |
| 2026-11-11 | `baseline_charles_part2_20261111_155645` | Baseline (`deepseek-reasoner` low) | `deepseek-reasoner` | Success (5 scenes, 134 VO words) | Clean regen (new Unsplash key, `5/5` Unsplash, `0` failures): 2,500 in / 1,743 out (954 reasoning, 1 call), sb `14.80s`, ~$0.0052 USD. |
| 2026-11-11 | `agentic_charles_part2_20261111_160733` | Multi-Agent (v4.5.1) | `deepseek-chat` + `deepseek-reasoner` | Success (6 scenes, 142 VO words) | Clean regen (`2ba550e` timeout fix, `3` pitches, 11 Wikipedia facts, `6/6` Unsplash, `0` failures): 16,102 in / 23,767 out (14,754 reasoning, 12 calls), sb `167.47s`, vid `13.21s` (`48.74s` MP4, `first_sound=1.705s`), ~$0.0495 USD. |
| 2026-11-11 | `baseline_charles_part3_20261111_161034` | Baseline (`deepseek-reasoner` low) | `deepseek-reasoner` | Success (5 scenes, 125 VO words) | Clean regen (new Unsplash key, `5/5` Unsplash, `0` failures): 2,409 in / 3,082 out (2,301 reasoning, 1 call), sb `21.60s`, vid `9.47s` (`49.85s` MP4, `first_sound=1.700s`), ~$0.0081 USD. |
| 2026-11-11 | `agentic_charles_part3_20261111_161105` | Multi-Agent (v4.5.1) | `deepseek-chat` + `deepseek-reasoner` | Success (5 scenes, 124 VO words) | Clean regen (`2ba550e` timeout fix, `3` pitches, `5/5` Unsplash, `0` failures): 15,810 in / 18,214 out (9,615 reasoning, 12 calls), sb `145.30s`, vid `13.02s` (`54.38s` MP4, `first_sound=1.693s`), ~$0.0371 USD. |
| 2026-11-11 | `judge_deepseek_v4_5_clean` | Automated Pairwise + Absolute Judge | `deepseek-chat` | Success (4 pairs, 16 calls) | Clean v4.5.1 batch (`max` + `charles_part1..3`, 100% real Unsplash photos): 48,440 in / 5,352 out (~$0.0083 USD). Pairwise `visual_grounding` flipped to `100.0%` agentic win rate (`8-0-0`, `4.00` vs `2.88`), `creativity` `100.0%` (`4-0-4`, `4.12` vs `3.12`), `OVERALL` `4.28` vs `3.95`. |
| 2026-11-11 | `agentic_charles_part1_20261111_162644` | Multi-Agent (v4.5.2) | `deepseek-chat` + `deepseek-reasoner` | Success (6 scenes, 146 VO words) | `'Le trou secret de votre stylo (et pourquoi il sauve des vies)'`; `6/6` Unsplash, `0` placeholders, 17,886 in / 20,022 out (11,463 reasoning, 11 calls), sb `153.67s`, vid `13.18s` (`59.60s` MP4, `first_sound=1.700s`), ~$0.0445 USD. |
| 2026-11-11 | `agentic_charles_part3_20261111_162931` | Multi-Agent (v4.5.2) | `deepseek-chat` + `deepseek-reasoner` | Success (6 scenes, 153 VO words) | `'Couper 100 fois un jeu de cartes ne le mélange pas'`; `6/6` Unsplash, `0` placeholders, 17,614 in / 14,990 out (6,010 reasoning, 11 calls), sb `113.17s`, vid `14.91s` (`58.66s` MP4, `first_sound=1.697s`), ~$0.0325 USD. |
| 2026-11-11 | `baseline_charles_part3_20261111_163140` | Baseline (v4.5.2 parity) | `deepseek-reasoner` (`low`) | Success (6 scenes, 131 VO words) | `'Le paradoxe des anniversaires : pourquoi 23 personnes suffisent pour 50% de chances'`; `6/6` Unsplash, `0` placeholders, 2,545 in / 2,659 out (1,906 reasoning, 1 call), sb `19.17s`, vid `16.65s` (`58.77s` MP4, `first_sound=1.692s`), ~$0.0072 USD. |
| 2026-11-11 | `judge_deepseek_v4_5_2` | Automated Pairwise + Absolute Judge | `deepseek-chat` | Success (4 pairs, 16 calls, 7 new) | v4.5.2 batch (`max` + `charles_part1..3`, cached `max`/`part2`): 22,064 in / 2,391 out (~$0.0038 USD). Abs coherence `5.00` vs `4.50`; Pairwise coherence `71.4%` (`4.75` vs `4.38`), creativity `100.0%` (`4.00` vs `2.75`), visual_grounding `100.0%` (`4.00` vs `2.88`), OVERALL `4.25` vs `3.77` (`62.5%` win rate). |
| 2026-11-11 | `agentic_max_20261111_164719` | Multi-Agent (v4.6) | `deepseek-chat` + `deepseek-reasoner` | Success (6 scenes, 141 VO words) | Cluster-weighted GoT/GRRM (#1 cluster `8/26 = 31%`): `'Why Game of Thrones Kept Getting Delayed: GRRM\'s Gardening Problem'`; `6/6` Unsplash, `0` placeholders, 17,199 in / 16,904 out (8,609 reasoning, 11 calls), sb `130.87s`, vid `26.62s` (`67.79s` MP4, `first_sound=1.701s`), `$0.03473 USD`. |
| 2026-11-11 | `agentic_charles_part2_20261111_170343` | Multi-Agent (v4.6) | `deepseek-chat` + `deepseek-reasoner` | Success (6 scenes, 161 VO words) | Cluster-weighted French comedy (#1 cluster `8/30 = 27%`): `'« Je m\'en occupe » : la phrase qui fait exploser les couples'`; `6/6` Unsplash, `0` placeholders, 19,145 in / 18,974 out (8,727 reasoning, 12 calls), sb `135.73s`, vid `15.00s` (`58.68s` MP4, `first_sound=1.704s`), `$0.03791 USD` (after archiving Unsplash-403 partial attempt `$0.03575`). |
| 2026-11-11 | `agentic_charles_part1_20261111_170818` | Multi-Agent (v4.6.1) | `deepseek-chat` + `deepseek-reasoner` | Success (6 scenes, 156 VO words) | Cluster-weighted math/logic (#1 cluster `7/30 = 23%`) + `16384` headroom fix: `'Vos amis ont plus d\'amis que vous (ce n\'est pas une insulte, c\'est un théorème)'`; `6/6` Unsplash, `0` placeholders, 21,344 in / 20,266 out (10,163 reasoning, 10 calls), sb `135.04s`, vid `15.28s` (`58.48s` MP4, `first_sound=1.696s`), `$0.04538 USD` (after archiving rate-limited `$0.03933` and 4-scene truncated `$0.04583` attempts). |
| 2026-11-11 | `judge_deepseek_v4_6` | Automated Pairwise + Absolute Judge | `deepseek-chat` | Success (4 pairs, 16 calls) | v4.6.1 final batch (`max`, `charles_part1..3`, 100% real Unsplash across all 8 videos; 3 judge passes totaling ~$0.0109 USD): Abs relevance **`4.75` vs `3.50`**, creativity `4.00` vs `2.50`, coherence `5.00` vs `4.50`; Pairwise relevance **`71.4%`** (`4.25` vs `3.50`, up from `28.6%`), creativity `71.4%` (`4.12` vs `3.25`), visual_grounding **`100.0%`** (`8-0-0`, `4.00` vs `2.75`), OVERALL **`4.38` vs `3.65`** (`62.5%` win rate). |

**Budget remaining: ~$48.54 / $50.00**

---

## Design Decisions Log

| Date | Decision | Rationale | Alternatives considered |
|---|---|---|---|
| 2026-11-12 | Simplify History Collector UI (`docs/index.html`, `docs/collector.js`, `tools/history_collector/`) | Removed the **Developer Console** tab, the **Verify Exported JSON** paste/drop field, and the bookmarklet's **Copy JSON to Clipboard** button to eliminate participant confusion. The collector page and bookmarklet overlay now focus strictly on the Bookmarks Bar drag-and-drop flow and a single **Download JSON** action saving `youtube_shorts_history.json`, with updated EN/FR guides in `docs/human_eval/`. | Keeping dual Bookmarklet/Console tabs, manual clipboard copy, and post-export JSON paste box |
| 2026-11-12 | Disable Multi-Provider & Mock Image Failover in Serper Mode (`image_tool.py`, `ArtDirectorAgent`, `ScriptwriterAgent`) | After adopting Serper.dev Google Images, failing over on Serper errors/misses into Unsplash Demo (`50 req/hr`), Pexels (paused), or fake `example.com` mock URLs re-introduced rate-limit failures and invalid placeholder URLs. When `IMAGE_API_PROVIDER=serper` fails or yields 0 verified candidates, `search_image_candidates()` / `search_images()` now returns `[]` directly; `ArtDirectorAgent` and `ScriptwriterAgent.refine()` skip wasted selection/refine LLM passes when no images exist, and affected scenes retain `image_ref = null` / `image_url = null` cleanly across storyboards and `storyboard_to_video.py`. | Failing over silently to Unsplash Demo / Pexels / fake `example.com` placeholder URLs |
| 2026-11-11 | Switch Primary Image Search to Hardened Serper.dev Google Images with Unsplash Failover (`image_tool.py`) | Unsplash Demo (`50 req/hr`) repeatedly stalled human-study batch generation and Pexels new-key issuance is paused. Our 10-query empirical benchmark (`tools/benchmark_image_apis.py`) showed Openverse (`3/10` hits, `8.0s`, HTTP `504`) and Wikimedia (`3/10` hits) fail on multi-word scene descriptions, whereas Serper.dev hit `10/10` queries (`1.6s`, `2,500` free credits at `50 QPS`) and returned clean Unsplash/Pexels/editorial photos. Hardened with watermark-CDN & social-wall blocking, shopping demotion, and live `200 + image/*` reachability checks. | Staying on Unsplash Demo `50/hr` (rate-limit bottleneck); Openverse / Wikimedia Commons standalone (30% recall on narrative scene prompts) |
| 2026-11-11 | Blind Human Evaluation Packager (`evaluation/prepare_human_eval.py`), Google Form Blueprint & Bilingual EN/FR Manuals (`docs/human_eval/`) | Standardized the ~15-participant human study around a **15–20 minute** single-pass pair-by-pair flow (`~6 min` watching + `~10 min` grading) combining 1–5 absolute ratings (**Personal Relevance**, **Creativity**, **Engagement**, **Overall Preference**) with per-pair forced/tie preference (`Video A` / `Video B` / `No preference`). `prepare_human_eval.py` automates balanced per-pair `AB`/`BA` blinding, optional pre-filled Google Form links, `.zip` bundles, and CSV unblinding/Wilcoxon analysis reusing `evaluation.llm_judge` stats. Bilingual EN/FR participant guides live in `docs/human_eval/`. | Separate absolute and pairwise watching passes (~25–30 min fatigue); Excel workbooks per participant (poor mobile UX); local HTML grader (`file://` browser friction) |
| 2026-11-11 | `16384`-Token Reasoner Headroom Floor & `ScriptwriterAgent.refine()` Scene-Count Integrity Guard (`v4.6.1`) | DeepSeek Reasoner spent `7,405 / 8,192` completion tokens thinking on `charles_part1` Phase 7 (`agentic_charles_part1_20261111_170104`), silently truncating a 6-scene draft to 4 scenes after `extract_json()` bracket repair. Doubled thinking headroom floor in `llm_client.py` (`8192 -> 16384`) and added a 1-shot retry + trailing draft-scene preservation guard (plus `sanitize_scenes()`) in `ScriptwriterAgent.refine()`. | Increasing `max_tokens` only in `ScriptwriterAgent` (leaves Draft/Baseline/Director vulnerable to thinking exhaustion) |
| 2026-11-11 | Cluster-Weighted Profiling (`dominant_clusters`) & Pairwise Relevance Rubric Alignment (`v4.6`) | Eliminated 1-video outlier tunnel vision (`[vid4]` Chubbyemu eclipse of `8/26` GoT on `max`; 1-off props eclipsing `7/30` `#mathctober` on `charles_part1`). Preserves frequency-ranked `dominant_clusters` + full watch-history titles across Round 1 -> Round 2 -> `WritersRoomAgent` -> `DirectorAgent` (`CLUSTER-ANCHORED DIVERSITY` forcing Pitch 1 on cluster #1 and Pitch 2 on cluster #2), and aligns `evaluation/llm_judge.py:build_pair_prompt()` item `1. relevance` with `METRICS["relevance"]` (anti-regurgitation `3/5` cap for direct remakes). | Tuning only Round 1 transcript count without changing Round 2 / Writers' Room / Pairwise judge prompt |
| 2026-11-11 | First-Watch Causal Setup & Single External Narrator Constraints across All Storyboard Prompts (v4.5.2) | Inspection of `charles_part1` and `charles_part3` revealed missing rule/mechanism bridges on first watch and confusing in-scene magician/spectator roleplay (`"je te les donne..."`). Enforced explicit rules-before-twist + causal *"why"* explanation (`FIRST-WATCH UNDERSTANDABILITY`) and `STRICTLY SINGLE EXTERNAL VOICEOVER NARRATOR` across `writers_room.txt`, `director.txt`, `scriptwriter_draft.txt`, `scriptwriter_refine.txt`, and `baseline.txt` (removed `"dialogue"` ambiguity, synced Pass 2 refinement constraints). | Post-hoc rewriting only in Pass 2 (still allows roleplay-driven pitches to win Phase 3) |
| 2026-11-11 | Bounded `120s` HTTP Timeout & Explicit 1-Retry Loop (`llm_client.py`) | OpenAI SDK's default `600s` timeout caused `WritersRoomAgent` to hang past `500s` on a stalled DeepSeek TCP stream during `charles_part2`. Enforced `LLM_TIMEOUT_SECONDS=120` (`timeout=120.0`, `max_retries=0`) in `get_client()` with an explicit, logged 1-retry handler on timeout, connection, or HTTP 5xx errors in `chat_completion()`. | Relying on OpenAI SDK default `600s` timeout (masks hung sockets for 10 minutes) |
| 2026-11-11 | Intro Audio Warmup Padding (`1.5s`) & Single-Pass AAC Final Mux (`storyboard_to_video.py`) | Reducing `DEFAULT_PADDING` to `0.3s` alongside per-clip AAC `.mp4` concatenation caused the opening `0.5–1.5s` of rendered videos to drop on players/audio devices and introduced a `0.023s` H.264/AAC start offset. Added `--intro-padding` (`1.5s` default) on Scene 1 while keeping `0.3s` inter-scene padding, and switched intermediate clips to `.mkv` (`pcm_s16le`) muxed to final `.mp4` in one pass with `-movflags +faststart`. | Increasing inter-scene padding globally to `1.5s` (makes scene transitions sluggish) |
| 2026-11-11 | Strict-JSON Retry Safety Net (`WritersRoomAgent`, `DirectorAgent`) & Unsplash Rate-Limit Fast-Bailout (v4.5.1) | Observed in `charles_part2` (`agentic_charles_part2_20261111_152909`) that a single unlogged JSON parse failure in `WritersRoomAgent` caused `DirectorAgent` to skip Phase 3 entirely. Added parse-failure warning logs + 1 automatic strict-JSON retry in both creative deliberation agents. Simultaneously added `UnsplashRateLimitError` short-circuiting (`HTTP 403/429`), warning diagnostics, and Pexels failover in `src/tools/image_tool.py` after Unsplash Demo's `50 req/hr` cap silently produced `example.com` mock URLs across 28/33 scenes on `charles_part1..3`. | Silent empty-list / empty-dict catches masking transient LLM formatting failures and HTTP 403 rate limits |
| 2026-11-11 | Alias Sync Fix, Anti-Regurgitation Prompts, Native-Language Transcripts & Round-Robin Fact Interleaving (v4.5) | Post-generation inspection and unit testing on `charles_part2` uncovered four root causes: (1) `KnowledgeRetrieverAgent` updated `state.knowledge_context` without updating `state.news_context`, while `ScriptwriterAgent._build_draft_prompt()` read `state.news_context` — silently passing `"No news snippets available."` to Phase 5; synchronized both fields + round-robin interleaved query buckets (`[:9]`). (2) `youtube_tool.get_video_transcript()` defaulted to `en`-only tracks, dropping French transcripts and feeding English auto-translations into Round 2; fixed via `detect_text_language()` + per-video `languages` preference. (3) `WritersRoomAgent` / `DirectorAgent` lacked anti-regurgitation and stock-photo feasibility rules, allowing direct skits/remakes of `[vid7]` reliant on fabricated chat UI text overlays; added explicit No-Remake, Self-Contained Causal Clarity, and Spoken-Payoff rules across prompts. | Relying solely on post-hoc judge penalties (penalized remakes after generation rather than preventing them upstream) |
| 2026-11-11 | Subtitle Removal, Tight VO Word Budget, English Image Queries, Reasoner Baseline Parity & Exact Per-Model Cost Audit (v4.4) | (1) Upgraded `baseline.model` in `configs/config.yaml` to `deepseek-reasoner` (`reasoning_effort="low"`) for strict backbone parity with multi-agent creative agents. (2) Mandated English-only Unsplash keywords in `prompts/image_query.txt` and `prompts/baseline.txt` after French stock queries produced irrelevant or missing photos. (3) Removed redundant `"subtitle"` generation across Scriptwriter/Baseline prompts and judge completeness rubric, tightened VO pacing (~15–25 words/scene, ~120–145 words total for 60s), and reduced `storyboard_to_video.py` default `--padding` from `1.0s` to `0.3s` so sketches stay dynamic near ~60s. (4) Strengthened `clean_voiceover()` to strip mid-string speaker cues (e.g., `CLINICIAN 1 (...):`) and reused it cleanly in `storyboard_to_video.py`. (5) Tracked per-model token usage (`_USAGE_BY_MODEL` / `get_session_cost_usd()`) so mixed `deepseek-chat` and `deepseek-reasoner` costs are computed accurately, wired `pipeline.*` config keys into both pipelines, and centralized `load_prompt()` for every agent (`prompts/news_brainstorm.txt` added). | Leaving `baseline` on non-reasoning `deepseek-chat` (unfair capability gap); retaining unused `"subtitle"` fields; flat reasoner-rate cost overestimation across mixed-model runs |
| 2026-11-11 | Visual Description Footer Below Image (`storyboard_to_video.py`) | Replaces on-screen `subtitle` overlays burned over scene photos with a dedicated footer panel below the unobstructed scene image displaying the full `visual_description` (`description`) with adaptive font scaling (`render_scene_frame()`). Ensures blind human evaluators can see what the AI intended each scene to depict alongside the retrieved photo without reading redundant subtitle captions over the image. | Burning subtitles on top of photos (obscures imagery and hides intended visual direction from human judges); dual text overlays |
| 2026-11-11 | Configurable Reasoning Effort (`reasoning_effort: "low"`) & Real-Time Pipeline Progress Prints | DeepSeek reasoning models default to `high` thinking effort, generating excessive hidden reasoning tokens (~2.4k/call) and multi-minute latency across 4 reasoner calls. Exposes `llm.reasoning_effort` in `configs/config.yaml` and `chat_completion()` (default `"low"`, overridable via `LLM_REASONING_EFFORT`), cutting Writers' Room latency ~44% and reasoning tokens ~62% while preserving CoT reasoning. Adds flushed `[Phase X/7]` banners, 5s LLM waiting heartbeat, per-call token/timing logs, and intermediate content previews (profile, 3s hooks, knowledge hits, scene outlines). | Disabling thinking completely via `deepseek-chat` (loses CoT deliberation in Writers' Room / Director / Scriptwriter); silent API waits |
| 2026-11-10 | Voiceover-Only Dialogue & Zero SFX Enforcement | Replaces mixed `audio_cue` (containing `VO:` stage prefixes and `SFX:` descriptions) with pure `voiceover` dialogue across both multi-agent and baseline pipelines. Adds strict negative constraints forbidding sound effects or actor prefixes in prompts, implements `clean_voiceover()` in `state.py` and `storyboard_to_video.py`, standardizes JSON to `"voiceover"` (preserving `"audio_cue"` as alias), and updates Markdown to `- **Voiceover:** <spoken text>`. Prevents TTS engines from reading out sound effect directions (e.g., 'Deep rumble SFX'). | Post-hoc regex cleaning alone (prompts would still encourage models to generate SFX); relying on complex audio mixing libraries |
| 2026-11-10 | Automatic oEmbed Title Enrichment & Sample Capping (`--cap`, `--force-enrich`) | Solves date-header title scraping from YouTube DOM ("Today", "Saturday"). Integrates automated parallel YouTube oEmbed metadata resolution directly into ingestion, guaranteeing genuine video titles and channel creators for downstream Profiler analysis. Adds `--cap` parameter allowing exact subset sizing (e.g. 90 videos for 3x30 splits) to ensure balanced, equal-sized partitions across comparative runs. | Manual title renaming; relying on generic fallback title numbering |
| 2026-11-09 | Per-User Output Partitioning & Universal Naming Convention (v4.2) | Replaces monolithic flat `outputs/` directories with organized per-user subdirectories (`outputs/<user>/baseline/`, `outputs/<user>/agentic/`, `outputs/<user>/judge/`). Enforces universal sortable ISO timestamp naming: `baseline_storyboard_<user>_<YYYYMMDD_HHMMSS>.json`, `agentic_storyboard_<user>_<YYYYMMDD_HHMMSS>.json`, `baseline_video_<user>_<YYYYMMDD_HHMMSS>.mp4`, `agentic_video_<user>_<YYYYMMDD_HHMMSS>.mp4`, and `judge_comparison_<user>_<YYYYMMDD_HHMMSS>.json`. Supports automatic video path derivation in `storyboard_to_video.py` and recursive file discovery in `evaluation/llm_judge.py`. | Flat single-folder dumping with hash-based IDs (unreadable when testing multiple users and runs) |
| 2026-11-09 | Neural Edge-TTS Speech Synthesis in Demo Sketch Renderer (v4.1) | Upgrades `storyboard_to_video.py` from robotic/offline engines (`pyttsx3`/`espeak`) to Microsoft Edge neural speech synthesis (`edge-tts`) as default primary engine, with graceful offline fallback. Adds `--voice` CLI option with multilingual neural defaults (e.g., `fr-FR-HenriNeural`, `en-US-ChristopherNeural`), enables standalone `ffmpeg` duration probing when `ffprobe` is absent, and adds `imageio-ffmpeg` dependency. Produces broadcast-quality vocal narration for blind human A/B evaluation and live presentations with zero cloud subscription fees. | Paid elevenlabs/Google TTS APIs (expensive/rate-limited); robotic local `espeak` (monotone robotic voice harms human blind preference tests) |
| 2026-11-09 | Participant Watch History Slicing & Partitioning (`--split`) (v4.0) | Adds `--split <X>`, `--split-mode` (random vs sequential), and deterministic `--seed` to `src/tools/import_user_history.py`. Allows dividing a participant's watch history into X balanced, disjoint slices (`<user>_part1.json`, `<user>_part2.json`, ...). Multiplies blind evaluation sample size (e.g. 10 users x 3 slices = 30 evaluation pairs) with zero subjective clustering bias, while allowing each participant to judge multiple distinct videos across their interests. | Manual thematic categorization (subjective and tedious); single video pair per user (insufficient sample size) |
| 2026-11-09 | Multi-Criteria Pairwise Scoring & Visual Grounding in Judge (v3.9) | Upgrades `evaluation/llm_judge.py` to: (1) Render blind visual asset photo descriptions (`Visual Asset Photo: {image_desc}`) directly into `render_storyboard()`, giving the judge exact visibility into asset quality without leaking system identity or URLs; (2) Decompose pairwise preference into individual verdicts for all 5 dimensions (`relevance`, `creativity`, `completeness`, `coherence`, `visual_grounding`) plus `overall_winner`; (3) Aggregate dimension-level win rates, ties, and position consistency across runs into both `summary.txt` and `summary.csv`. Eliminates single-metric ambiguity in pairwise evaluation. | Monolithic overall-only winner without dimension-level breakdown; passing image URLs or system tags to the judge (breaking blinding) |
| 2026-11-09 | Anti-Regurgitation & Multi-Facet Relevance Judge Rubric (v3.8) | Overhauls `evaluation/llm_judge.py` and `prompts/judge.txt` (v2.0) to penalize zero-shot baseline regurgitation. (1) Multi-Facet Relevance Directive rewards connecting with ANY genuine interest cluster rather than majority keyword counting; (2) Anti-Regurgitation Penalty strictly caps Creativity at 2/5 (or ≤4.0/10) if a storyboard merely copies or rehashes an already-watched video (e.g. croissant tutorial -> croissant hack). Ensures remakes cannot win on surface relevance alone. | Permissive keyword-matching judge without anti-regurgitation constraints (rewarded uninspired reruns) |
| 2026-11-09 | Purpose-Driven Architectural Renaming & Decoupled Aliases (v3.7) | Aligns component filenames and class identifiers with their exact operational purposes across the 7 phases: (1) `WritersRoomAgent` in `src/agents/writers_room_agent.py` replaces legacy `pitch_agent.py`; (2) `KnowledgeRetrieverAgent` in `src/agents/knowledge_retriever_agent.py` replaces `news_retriever_agent.py`; (3) `ArtDirectorAgent` in `src/agents/art_director_agent.py` replaces `image_search_agent.py` (reflecting both photographic query sanitization and 4-candidate semantic re-ranking); (4) `KnowledgeContext` in `src/core/state.py` becomes primary with `to_markdown()` factual evidence rendering and `SharedState.from_dict()` deserializer. All legacy modules and class names are preserved as 100% backward-compatible shims. | Leaving mismatched file/class names (`pitch_agent.py` defining `WritersRoomAgent`); hard breaking changes without backward shims |
| 2026-11-09 | Multi-Tiered Knowledge Retrieval (NewsAPI + Wikipedia + Tavily + LLM Synthesis) (v3.6) | Solves information starvation for evergreen topics (lore, science, history) where 30-day NewsAPI returns 0 articles. Introduces domain-aware queries (`news`, `encyclopedia`, `technical`), zero-auth Wikipedia search, and a Tier 3 LLM Knowledge Synthesis Fallback that extracts verifiable dates, quotes, and trivia instead of empty `[Archived Info]` placeholders. | Restricting searches strictly to breaking news; scraping Google with fragile web scrapers |
| 2026-11-09 | Baseline Visual Parity via Naive Top-1 Image Retrieval (v3.5) | Enables direct end-to-end video sketch rendering (`storyboard_to_video.py`) and fair visual-narrative alignment scoring for both Multi-Agent and Baseline storyboards. The baseline emits `image_query` zero-shot and resolves the top-1 stock photo, scientifically contrasting with the multi-agent pipeline's 4-candidate Art Director semantic re-ranking pass. | Leaving baseline image-less (prevented baseline video rendering and broke visual-narrative alignment comparison) |
| 2026-11-09 | Strict Negative Constraints for Verbatim Dialogue & Zero Placeholders (v3.4) | Eliminates evasive placeholder descriptions (`"A character says an innocent line..."`, `"Zoom on the quote..."`) in `scriptwriter_draft`, `scriptwriter_refine`, and `baseline`. Enforces 100% producible, word-for-word spoken dialogue and punchlines in both `subtitle` and `audio_cue`, directly enabling seamless TTS synthesis in `storyboard_to_video.py`. | Permissive general prompts; post-hoc script regex cleaning |
| 2026-11-09 | Hybrid Model Routing with DeepSeek R1 (`deepseek-reasoner`) for Creative & Deliberative Phases | Routes high-order reasoning (`deepseek-reasoner`) specifically to creative ideation (`writers_room`), critical pitch selection (`director`), and sharp dialogue drafting (`scriptwriter`), while keeping fast extraction tasks on low-latency `deepseek-chat`. Accompanied by multi-strategy JSON extraction (`extract_json`) and auto headroom expansion (`max_tokens >= 4096`) to prevent reasoning tokens from choking completion outputs. | Uniform model across all agents (either slow/expensive everywhere or underpowered for creative ideation) |
| 2026-11-09 | Comprehensive Agent Operating Manual (`GEMINI.md`) | Establishes the canonical Antigravity/Gemini equivalent of `CLAUDE.md`. Provides any blank, no-context AI agent or engineer with full architectural orientation, standard CLI commands, directory map ("Where to look for what"), 6 core invariants, and developer workflow recipes. | Ad-hoc tribal knowledge or fragmented markdown files |
| 2026-10-31 | Two-Phase Art Director Image Curation (v3.2) | Eliminates unrepresentative stock photos. Prohibits cinematography jargon (macro, split screen) in queries; retrieves 4 candidates per scene and uses an Art Director LLM pass to select the most visually faithful candidate | Naive top-1 search with cinematic prompts (produced irrelevant photos like basketball courts for pens) |
| 2026-10-31 | Writers' Room + Director + Targeted News Retrieval (v3.0) | Adapts flexibly to any user style (witty humor, craft, puzzle, lore) without forcing a rigid format; parallel candidate pitches enable exploration before commitment; NewsRetriever targets the winning pitch with concrete evidence | Single rigid documentary prompt; blind retrieval before ideation |
| 2026-10-30 | Strictly Neutral, Competitive Baseline Prompting | Academic integrity: avoids creating a "strawman" baseline. Baseline is held to the identical high standards of thematic focus, narrative arc, and storytelling without biased jokes/skit prompts | Biased/comedic baseline prompts (disingenuous comparison) |
| 2026-10-30 | Two-Pass Feedback Loop (`Draft -> Image Search -> Refine`) | Ensures image queries match specific scenes rather than generic themes; lets scriptwriter ground descriptions in actual visual assets | Naive linear pipeline (`Image -> Script`) |
| 2026-10-30 | Two-Round Profiler with Spoken Transcript Extraction | Video titles are often clickbait or vague; Round 1 identifies candidate videos, tool fetches spoken dialogue via `youtube-transcript-api`, Round 2 discovers deep niche interests | Single-pass title-only profiling |
| 2026-10-30 | Infotainment & Forensic Storytelling over Cheesy AI Comedy | LLMs struggle with authentic humor, but excel at engaging narrative journalism and surprising real-world curiosities | Pure comedy/meme skits (results in cringe dialogue) |
| 2026-10-30 | LLM Query Brainstorming in NewsRetrieverAgent | Enables creative, niche cross-topic search angles rather than dumb keyword passthrough | Direct string passthrough of raw profile interests |
| 2026-10-30 | Symbolic Visual Identifiers (`img1`, `img2`...) + Code Resolution | Eliminates URL hallucination, shortens prompts, ensures robust data integrity | Trusting LLM to copy/paste long image URLs |
| 2026-10-30 | Pure Python orchestration (no heavy framework) | Full control, complete transparency for report, easy debugging | LangGraph, AutoGen, CrewAI |
| 2026-10-30 | DeepSeek API for all agents | OpenAI-compatible, extremely low cost, strong reasoning | GPT-4o, local quantized LLMs |
| 2026-10-30 | Image search by keyword only (no vision model) | Keeps compute/API cost low; vision model not required for asset retrieval | Multimodal vision models (expensive) |

---

## Known Issues / TODOs

- [x] `news_tool.py` and `image_tool.py` live API keys integrated (NewsAPI + Unsplash) with graceful mock fallback
- [x] Move system prompts to dedicated `prompts/` text files for cleaner reproducibility archiving
- [x] Implement token usage accounting in `state.py` and `llm_client.py` to auto-calculate exact costs
- [x] Create comprehensive developer & agent operating manual (`GEMINI.md`)
- [x] Multi-strategy JSON extraction (`extract_json()`) and auto-headroom expansion for reasoning models (`CS-03`)
- [x] Multi-tiered knowledge retrieval (Wikipedia + NewsAPI + LLM Synthesis) preventing information starvation (`CS-05`)
- [x] Watch history partitioning tool with `--split`, `--split-mode`, and `--seed` (`src/tools/import_user_history.py`)
- [x] Neural speech synthesis (Edge-TTS) + dynamic subtitles in video sketch renderer (`storyboard_to_video.py`)
- [x] Multi-criteria 5D pairwise evaluation + 1–5 Likert grading + position debiasing (`evaluation/llm_judge.py`)
- [x] Legacy ungrounded `evaluator.py` removed
- [ ] Ingest 10-15 peer watch histories from web collector (Antoine)
- [ ] Run automated batch LLM judge evaluation (`evaluation/llm_judge.py`) across all histories (Max)
- [ ] Conduct human blind A/B evaluation using rendered neural sketches (Max)
- [ ] Draft 6-8 page final paper using living tracker logs (Charles)

---

## References

- Cheng et al. (2026). *Recommendation as Generation: Unifying Personalized Video Generation and Recommendation at Industrial Scale.*
- AutoGen: microsoft/autogen on GitHub
- ChatDev: Relevant multi-agent collaboration framework
- KuaiRec dataset: public short-video recommendation dataset
- DeepSeek API: platform.deepseek.com
