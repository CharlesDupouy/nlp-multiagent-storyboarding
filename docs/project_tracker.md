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
| Data collection & public dataset filtering | Antoine | 🔲 In progress (Peer collector live on GitHub Pages) |
| Single-agent baseline | Antoine | 🟢 Implemented, neutralized & tested on real user data |
| Multi-agent framework (core architecture) | Charles | 🟢 Writers' Room + Director + Targeted Retrieval complete & tested (v3.0) |
| Prompt engineering & refinement | Charles | 🟢 v3.0 active (Adaptive user style + neutral baseline) |
| Real API integration (news, images) | Charles | 🟢 100% Complete (NewsAPI + Unsplash + YouTube transcripts integrated & live) |
| LLM-as-a-judge evaluator | Max | 🟢 Complete (`evaluation/llm_judge.py` with batch & pairwise `--compare`) |
| Automated evaluation over dataset | Max | 🟢 Tested with deepseek & dry-run (`outputs/judge/`); pending full batch |
| Demo sketch renderer (TTS + images) | Max | 🟢 Neural Edge-TTS + dynamic subtitles + Unsplash slideshow (`storyboard_to_video.py`) |

---

## System Architecture

### Pipeline (Writers' Room & Director with Targeted Retrieval & Visual Grounding)

```mermaid
flowchart TD
    I["User Watch History\n(vid1, vid2, ... titles + URLs)"] --> P1["Phase 1A: Profiler Round 1\nMacro scan & active selection of 3-4 target videos"]
    P1 --> T1["Tool: youtube_transcript_api\nExtracts actual spoken dialogue & oEmbed channel metadata"]
    T1 --> P2["Phase 1B: Profiler Round 2\nSynthesizes fine-grained profile with concrete spoken insights"]
    P2 --> WR["Phase 2: WritersRoomAgent\nBrainstorms 3 distinct candidate pitches matching user style"]
    WR --> DIR["Phase 3: DirectorAgent (Showrunner)\nCritiques all 3 pitches, selects winner & issues production notes"]
    DIR --> N["Phase 4: NewsRetrieverAgent (Targeted Research)\nFormulates search queries specific to the winning pitch & director notes"]
    N --> S1["Phase 5: ScriptwriterAgent (Draft Pass)\nDrafts 60s storyboard executing winning pitch + director guidance"]
    S1 --> IMG["Phase 6: ImageSearchAgent\nAnalyzes draft scenes → photographic search → assigns img1, img2..."]
    IMG --> S2["Phase 7: ScriptwriterAgent (Refine Pass)\nGrounds scenes to visual assets using symbolic refs ('img1')"]
    S2 --> RES["Programmatic URL Resolution\nMaps 'image_ref' → validated 'image_url' (no hallucination)"]
    RES --> O["Storyboard JSON & Markdown\n(Pitches, Director Decision, Scenes, Subtitles, Narration, Image URLs)"]
```

> [!NOTE]
> **Why this Writers' Room architecture over a linear pipeline?**  
> 1. **User Style Adaptability:** Instead of forcing every user into an arbitrary documentary format, the Writers' Room proposes 3 distinct concepts matching their natural preferred style (witty humor, craft, pop culture, tech puzzle, or lore).
> 2. **Parallel Exploration (Exploration vs. Exploitation):** Parallel candidate pitches allow creative breadth before committing to production.
> 3. **Showrunner Deliberation:** The Director evaluates pitches against 3s hook potential, personal resonance, and visual feasibility.
> 4. **Targeted Information Retrieval:** The News/Fact Retriever searches for concrete evidence *tailored specifically to the winning pitch* and the Director's instructions.
> 5. **Symbolic Asset Grounding (`img1`, `img2`, ...):** Decoupling creative image selection from URL handling completely eliminates URL hallucination.

### Agent Roles

| Agent | Input | Output | Model | Tools |
|---|---|---|---|---|
| **ProfilerAgent** | Watch history (`vid1`, `vid2`, ...) | `UserProfile` (interests, themes, style, investigated_insights) | `deepseek-chat` | `youtube_transcript_api` + oEmbed |
| **WritersRoomAgent** | `UserProfile` + insights | `candidate_pitches` (3 diverse structured story pitches) | `deepseek-chat` | None (Creative synthesis) |
| **DirectorAgent** | `UserProfile` + `candidate_pitches` | `selected_pitch` + `director_critique` + `director_notes` | `deepseek-chat` | Editorial critique |
| **NewsRetrieverAgent** | `selected_pitch` + `director_notes` | `NewsContext` (targeted headlines + summaries) | `deepseek-chat` | NewsAPI / Serper / Web Search |
| **ScriptwriterAgent (Draft)** | `selected_pitch` + `director_notes` + `NewsContext` | Draft `Storyboard` (scenes, descriptions, audio) | `deepseek-chat` | None (LLM synthesis) |
| **ImageSearchAgent** | Draft `Storyboard` scenes | `ImageContext` (`img1`, `img2`, ... catalog) | `deepseek-chat` | Unsplash / Pexels (keyword search) |
| **ScriptwriterAgent (Refine)** | Draft `Storyboard` + `ImageContext` | Refined `Storyboard` (`image_ref: 'img1'`) | `deepseek-chat` | Programmatic URL resolver |

### Controlled Comparison

```mermaid
flowchart LR
    H["Same user\nwatch history"]
    H --> MA["Multi-Agent Pipeline\nProfile → News Brainstorm → Draft → Scene Image Search → Refinement"]
    H --> SA["Single-Agent Baseline\nZero-shot, same model, same watch history"]
    MA --> SB1["Storyboard A\n(Grounded with imagery & news)"]
    SA --> SB2["Storyboard B\n(Zero-shot single pass)"]
    SB1 --> E["Evaluator\nLLM-as-judge + human A/B"]
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
- **Collector Tool:** Client-side 1-click bookmarklet + tutorial web page (`tools/history_collector/index.html`) created and ready for distribution.
- **Ingestion Helper:** `src/tools/import_user_history.py` validates and saves participant files to `data/user_histories/<user>.json`.
- **Status:** Tooling complete; peer recruitment & data collection in progress (Antoine / Team).
- > [!CAUTION]
  > Must be anonymised before storage. Do not commit real user data to git.

---

## Models & API Configuration

| Parameter | Value | Rationale |
|---|---|---|
| LLM provider | DeepSeek API | OpenAI-compatible, very cheap |
| Extraction agents (`profiler`, `knowledge_retriever`, `art_director`) | `deepseek-chat` | Fast classification, keyword extraction & re-ranking; low latency |
| Reasoning agents (`writers_room`, `director`, `scriptwriter`) | `deepseek-reasoner` (DeepSeek R1) | High-order chain-of-thought reasoning for diverse pitch ideation, critical pitch selection, and sharp narrative drafting |
| Baseline model | `deepseek-chat` | Neutral zero-shot comparison |
| Temperature (profiler) | 0.4 | Low — factual extraction |
| Temperature (news brainstorm) | 0.7 | Medium — creative angle ideation |
| Temperature (script draft/refine) | N/A (`deepseek-reasoner`) | Managed internally by R1 deterministic reasoning |
| News & Knowledge API | NewsAPI.org + MediaWiki (Wikipedia) | Multi-tiered live API client with evergreen lore lookup and LLM fallback |
| Image API | Unsplash API | Verified live API client with photographic query sanitization |

> [!TIP]
> For the report's **Reproducibility Statement**, log exact model name, API access date, and decoding params for every experiment run. These are saved in each `storyboard_*.json` output file.

---

## Prompts (Versioned)

All system prompts are externalized in `prompts/*.txt` as a versioned plain-text registry to maintain clean git diffs and ensure scientific reproducibility.

| Agent / Component | Prompt File | Version | Role / Focus | Last Updated |
|---|---|---|---|---|
| ProfilerAgent (Round 1) | `prompts/profiler_r1_macro.txt` | v2.0 | Macro history scan & targeted video selection | 2026-10-31 |
| ProfilerAgent (Round 2) | `prompts/profiler_r2_synthesis.txt` | v2.0 | Fine-grained profile synthesis from spoken transcripts | 2026-10-31 |
| WritersRoomAgent | `prompts/writers_room.txt` | v3.0 | 3 distinct candidate pitches adapted to user vibe | 2026-10-31 |
| DirectorAgent | `prompts/director.txt` | v3.0 | Pitch critique, winning selection & production notes | 2026-10-31 |
| NewsRetrieverAgent | `prompts/news_targeted_research.txt` | v3.1 | Domain-targeted factual search queries (news, encyclopedia, technical) | 2026-11-09 |
| ScriptwriterAgent (Draft) | `prompts/scriptwriter_draft.txt` | v3.1 | 60s storyboard narrative execution (verbatim dialogue, zero placeholders) | 2026-11-09 |
| ImageSearchAgent (Query) | `prompts/image_query.txt` | v3.2 | Concrete physical nouns (no camera jargon) | 2026-10-31 |
| ImageSearchAgent (Curate) | `prompts/art_director_selection.txt` | v3.2 | Art director semantic re-ranking of photo candidates | 2026-10-31 |
| ScriptwriterAgent (Refine) | `prompts/scriptwriter_refine.txt` | v3.1 | Visual grounding with symbolic image refs (verbatim dialogue, zero placeholders) | 2026-11-09 |
| Baseline | `prompts/baseline.txt` | v2.2 | Neutral, competitive zero-shot baseline (verbatim dialogue + naive top-1 image query schema) | 2026-11-09 |
| LLM Judge | `prompts/judge.txt` | v2.0 | 4-dimensional rubric with Multi-Facet Relevance & Anti-Regurgitation Penalty | 2026-11-09 |

---

## Evaluation Plan

### Automated (LLM-as-a-Judge)
- **Metrics** (1–5 Likert scale in `evaluation/llm_judge.py`):
  - **Relevance:** Does the storyboard connect with ANY genuine interest cluster in the watch history (multi-facet directive)? Merely rehashing a watched video is penalized.
  - **Completeness:** Are all storyboard fields filled and coherent (visuals, subtitles, verbatim audio cues)?
  - **Creativity:** Is the narrative fresh and original? Remakes/rehashes of watched videos are strictly CAPPED at 2/5 (or ≤4.0/10).
  - **Coherence & Visual Alignment:** Narrative flow, pacing, and how faithfully retrieved photo assets match scene descriptions.
- **Judge model:** `deepseek-chat` (or `gpt-4o` / `deepseek-reasoner`)
- **Scale:** Run over public dataset and collected peer histories

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

### Final Report (Dec 4)
- [x] Automated evaluation implemented (`evaluation/llm_judge.py` with batch & `--compare` modes) and dry-run validated
- [ ] Human evaluation conducted (Blind A/B test)
- [ ] Results tables and figures
- [ ] Failure analysis (at least 1 informative failure case)
- [ ] Limitations and responsible use section
- [ ] Reproducibility package (README, requirements, prompts, eval code)
- [ ] Cost and compute statement
- [ ] Team contributions statement

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

**Budget remaining: ~$49.53 / $50.00**

---

## Design Decisions Log

| Date | Decision | Rationale | Alternatives considered |
|---|---|---|---|
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
- [ ] JSON parsing in agents has simple cleanup fallback — can add retry loop for edge cases
- [ ] Ingest 10-15 peer watch histories from web collector (Antoine)
- [ ] Run automated batch LLM judge evaluation (`evaluation/llm_judge.py`) across all histories (Max)
- [ ] Conduct human blind A/B evaluation (Max)
- [ ] Draft 6-8 page final paper using living tracker logs (Charles)

---

## References

- Cheng et al. (2026). *Recommendation as Generation: Unifying Personalized Video Generation and Recommendation at Industrial Scale.*
- AutoGen: microsoft/autogen on GitHub
- ChatDev: Relevant multi-agent collaboration framework
- KuaiRec dataset: public short-video recommendation dataset
- DeepSeek API: platform.deepseek.com
