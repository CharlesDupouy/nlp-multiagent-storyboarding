# Multi-Agent Storyboarding for Personalized Short-Video Generation
**COMP 4221: Intro to NLP — Fall 2026**  
Team: Antoine Lepin · Max Robertson · Charles Dupouy

---

## Quick Start

```bash
# 1. Clone & install
pip install -r requirements.txt

# 2. Set your API key
cp .env.example .env
# Edit .env and fill in DEEPSEEK_API_KEY

# 3. Run the multi-agent pipeline
python main.py --config configs/config.yaml --history data/user_histories/example.json

# 4. Run the single-agent baseline
python main.py --config configs/config.yaml --history data/user_histories/example.json --baseline

# 5. Run tests
python3 -m unittest discover tests/
```

## Collecting Real User Histories for Evaluation

To collect watch histories from study participants without API keys or credentials:
1. Share `tools/history_collector/index.html` with your participants (or host it on GitHub Pages).
2. The participant drags the **📌 Export Shorts History** bookmarklet to their browser bar.
3. They visit [youtube.com/feed/history](https://www.youtube.com/feed/history), click the bookmark, and download `youtube_shorts_history.json`.
4. Ingest their exported file into the pipeline:
   ```bash
   python3 src/tools/import_user_history.py --input /path/to/downloaded.json --user participant_alice
   ```
5. Run the storyboarding generator:
   ```bash
   python3 main.py --history data/user_histories/participant_alice.json
   ```

## Project Structure

```
.
├── main.py                        # Entry point (pipeline & baseline)
├── configs/config.yaml            # Model and API configuration
├── src/
│   ├── orchestrator.py            # Pipeline controller
│   ├── core/
│   │   ├── state.py               # SharedState blackboard (all data structures)
│   │   └── llm_client.py          # OpenAI-compatible LLM wrapper (DeepSeek)
│   ├── agents/
│   │   ├── base_agent.py          # Abstract base class for all agents
│   │   ├── profiler_agent.py      # Agent 1: user interest profiling
│   │   ├── news_retriever_agent.py # Agent 2: news/web retrieval
│   │   ├── image_search_agent.py  # Agent 3: image URL retrieval
│   │   └── scriptwriter_agent.py  # Agent 4: storyboard generation
│   └── tools/
│       ├── news_tool.py           # News/search API wrapper (TODO: real impl)
│       └── image_tool.py          # Image search API wrapper (TODO: real impl)
├── baseline/
│   └── single_agent_baseline.py  # Zero-shot single-agent baseline (Antoine)
├── evaluation/
│   └── evaluator.py              # LLM-as-a-judge scoring (Max)
├── data/
│   ├── raw/                      # Raw dataset files
│   ├── processed/                # Filtered/preprocessed data
│   └── user_histories/           # User watch histories (JSON)
├── outputs/storyboards/          # Generated storyboard JSON files
├── prompts/                      # Versioned prompt files
├── tests/                        # Unit and integration tests
└── docs/                         # Project documentation and dev log
```

## Environment Variables

| Variable | Description |
|---|---|
| `DEEPSEEK_API_KEY` | DeepSeek API key (required) |
| `LLM_BASE_URL` | LLM API base URL (default: `https://api.deepseek.com`) |
| `NEWSAPI_KEY` | NewsAPI.org key (for NewsRetrieverAgent) |
| `SERPER_API_KEY` | Serper.dev key (alternative to NewsAPI) |
| `UNSPLASH_ACCESS_KEY` | Unsplash key (for ImageSearchAgent) |
| `PEXELS_API_KEY` | Pexels key (alternative to Unsplash) |

## Reproducibility

See `docs/dev_log.md` for a full log of model versions, API providers,
prompt versions, and results at each stage of development.
