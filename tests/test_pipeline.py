"""
Basic smoke tests for the pipeline data structures and tools.
Run with: python3 -m unittest discover tests
"""

import unittest
from pathlib import Path
import tempfile
from src.core.state import SharedState, UserHistory, UserProfile, ImageContext, Storyboard
from src.tools.youtube_tool import extract_video_id


class TestPipeline(unittest.TestCase):
    def test_shared_state_initialization(self):
        state = SharedState(
            user_history=UserHistory(videos=[
                {"title": "Test video", "url": "https://youtube.com/shorts/test"}
            ]),
            run_id="test_run",
        )
        self.assertEqual(len(state.user_history.videos), 1)
        self.assertEqual(state.run_id, "test_run")
        self.assertEqual(state.storyboard.title, "")

    def test_shared_state_save(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            state = SharedState(run_id="test")
            out = Path(tmp_dir) / "state.json"
            state.save(str(out))
            self.assertTrue(out.exists())

    def test_storyboard_resolve_image_urls(self):
        img_ctx = ImageContext(images=[
            {"image_id": "img1", "url": "https://example.com/croissant.jpg", "alt_text": "Flaky croissant"},
            {"image_id": "img2", "url": "https://example.com/cafe.jpg", "alt_text": "Hong Kong cafe"},
        ])
        storyboard = Storyboard(
            title="Test Storyboard",
            scenes=[
                {"scene_id": 1, "description": "Scene 1", "image_ref": "img1"},
                {"scene_id": 2, "description": "Scene 2", "image_ref": "img2"},
                {"scene_id": 3, "description": "Scene 3", "image_ref": None},
            ],
        )
        storyboard.resolve_image_urls(img_ctx)
        self.assertEqual(storyboard.scenes[0]["image_url"], "https://example.com/croissant.jpg")
        self.assertEqual(storyboard.scenes[1]["image_url"], "https://example.com/cafe.jpg")
        self.assertIsNone(storyboard.scenes[2]["image_url"])

    def test_user_profile_markdown_with_insights(self):
        profile = UserProfile(
            interests=["teletoon", "puzzles"],
            themes=["nostalgia"],
            preferred_style="infotainment",
            raw_summary="Summary of tastes.",
            investigated_insights=["From vid1: controversy about banned episodes"],
        )
        state = SharedState(user_profile=profile, run_id="test_insights")
        md = state.to_markdown()
        self.assertIn("Deep-Dive Spoken Insights", md)
        self.assertIn("controversy about banned episodes", md)

    def test_extract_video_id(self):
        self.assertEqual(extract_video_id("https://www.youtube.com/shorts/dQw4w9WgXcQ"), "dQw4w9WgXcQ")
        self.assertEqual(extract_video_id("https://youtu.be/dQw4w9WgXcQ"), "dQw4w9WgXcQ")
        self.assertEqual(extract_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ"), "dQw4w9WgXcQ")
        self.assertEqual(extract_video_id("dQw4w9WgXcQ"), "dQw4w9WgXcQ")



    def test_writers_room_and_director_markdown(self):
        state = SharedState(
            candidate_pitches=[
                {
                    "pitch_id": "pitch1",
                    "title": "Banned Cartoons",
                    "target_tone": "Witty forensic",
                    "hook_3s": "Watch this cartoon clip...",
                    "concept": "The real reasons Teletoon episodes were banned.",
                    "user_alignment_reason": "Matches nostalgia interest",
                },
                {
                    "pitch_id": "pitch2",
                    "title": "Paper to the Moon",
                    "target_tone": "Curious puzzle",
                    "hook_3s": "Fold this paper 42 times...",
                    "concept": "Exponential folding math.",
                    "user_alignment_reason": "Matches math interest",
                },
            ],
            selected_pitch={
                "pitch_id": "pitch1",
                "title": "Banned Cartoons",
            },
            director_notes="Make scene 1 fast-paced and highlight censorship documents.",
            run_id="test_writers_room",
        )
        md = state.to_markdown()
        self.assertIn("Writers' Room: Pitches & Directorial Selection", md)
        self.assertIn("SELECTED BY DIRECTOR", md)
        self.assertIn("Banned Cartoons", md)
        self.assertIn("Directorial Notes & Production Guidance", md)
        self.assertIn("highlight censorship documents", md)


    def test_clean_stock_query(self):
        from src.tools.image_tool import clean_stock_query
        q = "teenage hand clicking Bic 4 color pen macro split screen"
        cleaned = clean_stock_query(q)
        self.assertNotIn("macro", cleaned)
        self.assertNotIn("split screen", cleaned)
        self.assertNotIn("Bic", cleaned)
        self.assertIn("ballpoint", cleaned)
        self.assertIn("pen", cleaned)

    def test_parse_art_director_selections(self):
        from src.agents.image_search_agent import _parse_art_director_selections
        raw_json = '{"selections": [{"scene_id": 1, "selected_candidate_id": "cand2"}, {"scene_id": 2, "selected_candidate_id": "cand1"}]}'
        res = _parse_art_director_selections(raw_json)
        self.assertEqual(res.get(1), "cand2")
        self.assertEqual(res.get(2), "cand1")

    def test_extract_json_edge_cases(self):
        from src.core.llm_client import extract_json

        # 1. Clean JSON
        self.assertEqual(extract_json('{"status": "ok"}'), {"status": "ok"})

        # 2. Markdown fenced with ```json
        fenced_json = '```json\n{"status": "ok"}\n```'
        self.assertEqual(extract_json(fenced_json), {"status": "ok"})

        # 3. Preamble and postamble thinking from reasoning models
        reasoning_resp = (
            "Thinking Process:\n1. The user requested pitches.\n2. Formulating JSON.\n\n"
            "```json\n"
            "{\n"
            '  "pitches": [\n'
            '    {"pitch_id": "pitch1", "title": "Test Pitch",}\n'  # Note trailing comma
            "  ]\n"
            "}\n"
            "```\n"
            "Hope this helps the director make a decision!"
        )
        res = extract_json(reasoning_resp)
        self.assertIn("pitches", res)
        self.assertEqual(res["pitches"][0]["pitch_id"], "pitch1")

        # 4. Raw JSON surrounded by text without markdown code blocks
        raw_surrounded = 'Preamble text here {"scene_id": 1, "title": "Draft"} final thoughts.'
        self.assertEqual(extract_json(raw_surrounded), {"scene_id": 1, "title": "Draft"})

    def test_orchestrator_config_models(self):
        from src.orchestrator import Orchestrator
        orch = Orchestrator.from_config("configs/config.yaml")
        self.assertEqual(orch.profiler.model, "deepseek-chat")
        self.assertEqual(orch.writers_room.model, "deepseek-reasoner")
        self.assertEqual(orch.director.model, "deepseek-reasoner")
        self.assertEqual(orch.knowledge_retriever.model, "deepseek-chat")
        self.assertEqual(orch.news_retriever.model, "deepseek-chat")
        self.assertEqual(orch.art_director.model, "deepseek-chat")
        self.assertEqual(orch.image_searcher.model, "deepseek-chat")
        self.assertEqual(orch.scriptwriter.model, "deepseek-reasoner")
        self.assertIs(orch.knowledge_retriever, orch.news_retriever)
        self.assertIs(orch.art_director, orch.image_searcher)

    def test_reasoning_parsers(self):
        from src.agents.pitch_agent import _parse_pitches
        from src.agents.director_agent import _parse_decision
        from src.agents.scriptwriter_agent import _parse_storyboard_response

        # Writers room parsing with reasoning commentary
        pitch_raw = "I've analyzed the profile.\n```json\n{\"pitches\": [{\"pitch_id\": \"pitch1\", \"title\": \"Secret Lore\"}]}\n```\nAll done."
        pitches = _parse_pitches(pitch_raw)
        self.assertEqual(len(pitches.get("pitches", [])), 1)
        self.assertEqual(pitches["pitches"][0]["title"], "Secret Lore")

        # Director decision parsing with reasoning commentary
        director_raw = "Evaluating candidates...\n```json\n{\"winning_pitch_id\": \"pitch2\", \"critiques\": [], \"director_notes\": \"High tempo.\"}\n```"
        decision = _parse_decision(director_raw)
        self.assertEqual(decision.get("winning_pitch_id"), "pitch2")
        self.assertEqual(decision.get("director_notes"), "High tempo.")

        # Scriptwriter parsing with reasoning commentary
        script_raw = "Drafting script...\n```json\n{\"title\": \"Epic Short\", \"overall_narrative\": \"Narrative.\", \"scenes\": [{\"scene_id\": 1, \"duration_seconds\": 10}]}\n```"
        sb = _parse_storyboard_response(script_raw)
        self.assertEqual(sb.get("title"), "Epic Short")
        self.assertEqual(len(sb.get("scenes", [])), 1)

    def test_calculate_cost_usd_reasoner(self):
        from src.core.llm_client import calculate_cost_usd
        # Chat model pricing: $0.14 / 1M prompt, $0.28 / 1M completion
        chat_cost = calculate_cost_usd(1_000_000, 1_000_000, model="deepseek-chat")
        self.assertAlmostEqual(chat_cost, 0.42, places=4)

        # Reasoner model pricing: $0.55 / 1M prompt, $2.19 / 1M completion
        reasoner_cost = calculate_cost_usd(1_000_000, 1_000_000, model="deepseek-reasoner")
        self.assertAlmostEqual(reasoner_cost, 2.74, places=4)

    def test_dialogue_negative_constraints(self):
        from src.agents.scriptwriter_agent import _DRAFT_SYSTEM_PROMPT, _REFINE_SYSTEM_PROMPT
        from baseline.single_agent_baseline import _BASELINE_SYSTEM_PROMPT

        required_phrase = "CRITICAL NEGATIVE CONSTRAINTS"
        forbidden_example = "A character says an innocent line"

        for prompt_name, prompt_text in [
            ("draft", _DRAFT_SYSTEM_PROMPT),
            ("refine", _REFINE_SYSTEM_PROMPT),
            ("baseline", _BASELINE_SYSTEM_PROMPT),
        ]:
            self.assertIn(required_phrase, prompt_text, f"{prompt_name} missing negative constraints header")
            self.assertIn("NO PLACEHOLDERS", prompt_text, f"{prompt_name} missing no placeholders clause")
            self.assertIn(forbidden_example, prompt_text, f"{prompt_name} missing forbidden example")

    def test_search_images_per_page(self):
        from src.tools.image_tool import search_images
        res = search_images("origami paper", per_page=1)
        self.assertEqual(len(res), 1)
        self.assertIn("url", res[0])
        self.assertIn("description", res[0])
        self.assertTrue(bool(res[0]["url"]))

    def test_baseline_image_query_in_schema(self):
        from pathlib import Path
        from baseline.single_agent_baseline import _BASELINE_SYSTEM_PROMPT

        prompt_file = Path("prompts/baseline.txt").read_text(encoding="utf-8")
        self.assertIn('"image_query"', prompt_file)
        self.assertIn('"image_query"', _BASELINE_SYSTEM_PROMPT)

    def test_baseline_visual_resolution(self):
        from unittest.mock import patch, MagicMock
        from baseline.single_agent_baseline import SingleAgentBaseline

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = (
            '{"title": "Fold the Universe", "overall_narrative": "Math of folding.", '
            '"scenes": [{"scene_id": 1, "description": "Paper on desk", '
            '"image_query": "folded white paper", "subtitle": "Fold it 42 times.", '
            '"audio_cue": "Voice: Look at this paper.", "duration_seconds": 10}]}'
        )
        mock_response.usage.prompt_tokens = 50
        mock_response.usage.completion_tokens = 50
        mock_client.chat.completions.create.return_value = mock_response

        with tempfile.TemporaryDirectory() as tmp_dir:
            baseline = SingleAgentBaseline(model="deepseek-chat", output_dir=tmp_dir)
            baseline.client = mock_client
            state = baseline.run([{"title": "Folding paper", "url": "https://youtube.com/shorts/test"}])

            self.assertEqual(len(state.storyboard.scenes), 1)
            scene = state.storyboard.scenes[0]
            self.assertIsNotNone(scene.get("image_url"))
            self.assertIsNotNone(scene.get("image_description"))
            self.assertTrue(scene["image_url"].startswith("http"))

    def test_knowledge_context_alias(self):
        from src.core.state import SharedState, NewsContext
        state = SharedState(news_context=NewsContext(snippets=[{"headline": "Test Fact", "summary": "Fact summary", "url": "https://example.com"}]))
        self.assertEqual(len(state.knowledge_context.snippets), 1)
        self.assertEqual(state.knowledge_context.snippets[0]["headline"], "Test Fact")

    def test_parse_research_queries(self):
        from src.agents.news_retriever_agent import _parse_research_queries

        # Dict format with queries list of dicts
        raw_json = (
            '{"queries": ['
            '{"query": "Dexter Rude Removal 1998", "domain": "encyclopedia", "rationale": "Lore background"},'
            '{"query": "Paper folding moon 42", "domain": "technical", "rationale": "Math proof"}'
            ']}'
        )
        parsed = _parse_research_queries(raw_json)
        self.assertEqual(len(parsed), 2)
        self.assertEqual(parsed[0]["domain"], "encyclopedia")
        self.assertEqual(parsed[1]["domain"], "technical")

        # Flat string list format
        raw_strings = '{"queries": ["Topic 1", "Topic 2"]}'
        parsed2 = _parse_research_queries(raw_strings)
        self.assertEqual(len(parsed2), 2)
        self.assertEqual(parsed2[0]["domain"], "auto")

    def test_wikipedia_evergreen_search(self):
        from src.tools.knowledge_tool import _wikipedia_search
        results = _wikipedia_search("origami mathematics", max_results=1)
        self.assertTrue(len(results) >= 1)
        self.assertIn("Wikipedia:", results[0]["headline"])
        self.assertTrue(bool(results[0]["summary"]))
        self.assertTrue(results[0]["url"].startswith("https://en.wikipedia.org/wiki/"))

    def test_tier3_llm_synthesize_facts_mock(self):
        from unittest.mock import MagicMock
        from src.tools.knowledge_tool import _llm_synthesize_facts

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = (
            '[{"headline": "Rude Removal Uncensored in 1998", "summary": "Produced by Tartakovsky with 81 cuts.", '
            '"url": "https://en.wikipedia.org/wiki/Dexter%27s_Laboratory"}]'
        )
        mock_response.usage.prompt_tokens = 20
        mock_response.usage.completion_tokens = 20
        mock_client.chat.completions.create.return_value = mock_response

        facts = _llm_synthesize_facts("Dexter Rude Removal", max_results=1, client=mock_client)
        self.assertEqual(len(facts), 1)
        self.assertEqual(facts[0]["headline"], "Rude Removal Uncensored in 1998")
        self.assertIn("Tartakovsky", facts[0]["summary"])

    def test_shared_state_from_dict_and_knowledge_markdown(self):
        from src.core.state import SharedState, KnowledgeContext
        state = SharedState(
            run_id="test_from_dict",
            knowledge_context=KnowledgeContext(snippets=[
                {"headline": "Bouchon Charter 2012", "summary": "Official label by Lyon CCI", "url": "https://example.com/bouchon"}
            ]),
        )
        # Test markdown rendering of grounded knowledge
        md = state.to_markdown()
        self.assertIn("📚 Grounded Knowledge & Factual Evidence", md)
        self.assertIn("Bouchon Charter 2012", md)
        self.assertIn("Official label by Lyon CCI", md)

        # Test to_dict and from_dict roundtrip
        d = state.to_dict()
        self.assertIn("knowledge_context", d)
        self.assertIn("news_context", d)

        restored = SharedState.from_dict(d)
        self.assertEqual(restored.run_id, "test_from_dict")
        self.assertEqual(len(restored.knowledge_context.snippets), 1)
        self.assertEqual(restored.knowledge_context.snippets[0]["headline"], "Bouchon Charter 2012")
        self.assertEqual(restored.news_context.snippets[0]["headline"], "Bouchon Charter 2012")

    def test_canonical_agent_exports(self):
        from src.agents.writers_room_agent import WritersRoomAgent
        from src.agents.knowledge_retriever_agent import KnowledgeRetrieverAgent
        from src.agents.art_director_agent import ArtDirectorAgent
        from src.agents.pitch_agent import WritersRoomAgent as WritersRoomAgentShim
        from src.agents.news_retriever_agent import NewsRetrieverAgent as NewsRetrieverShim
        from src.agents.image_search_agent import ImageSearchAgent as ImageSearchShim
        import src.agents as agents

        self.assertIs(WritersRoomAgent, WritersRoomAgentShim)
        self.assertIs(KnowledgeRetrieverAgent, NewsRetrieverShim)
        self.assertIs(ArtDirectorAgent, ImageSearchShim)
        self.assertTrue(hasattr(agents, "WritersRoomAgent"))
        self.assertTrue(hasattr(agents, "KnowledgeRetrieverAgent"))
        self.assertTrue(hasattr(agents, "ArtDirectorAgent"))


    def test_render_storyboard_blinding_and_captions(self):
        from evaluation.llm_judge import render_storyboard

        # Agentic storyboard with image_ref and image_context
        agentic_data = {
            "storyboard": {
                "title": "Agentic Short",
                "overall_narrative": "A deep dive into French bouchons.",
                "scenes": [
                    {
                        "scene_id": 1,
                        "description": "Macro shot of a wine chalkboard menu.",
                        "subtitle": "This menu is a test.",
                        "audio_cue": "VO: Listen closely.",
                        "image_ref": "img1",
                        "image_url": "https://images.unsplash.com/secret_url.jpg",
                        "duration_seconds": 5,
                    }
                ],
            },
            "image_context": {
                "images": [
                    {
                        "image_id": "img1",
                        "description": "Handwritten wine menu board behind glass",
                        "alt_text": "Wine menu board behind glass (Photo by John Doe)",
                        "url": "https://images.unsplash.com/secret_url.jpg",
                    }
                ]
            },
        }

        rendered = render_storyboard(agentic_data)
        self.assertIn("Visual: Macro shot of a wine chalkboard menu.", rendered)
        self.assertIn("Visual Asset Photo: Handwritten wine menu board behind glass", rendered)
        self.assertIn("Subtitle: This menu is a test.", rendered)
        self.assertIn("Voiceover: VO: Listen closely.", rendered)
        self.assertNotIn("https://", rendered)
        self.assertNotIn("secret_url", rendered)
        self.assertNotIn("John Doe", rendered)  # photographer attribution stripped

        # Baseline storyboard with inline image_description
        baseline_data = {
            "storyboard": {
                "title": "Baseline Short",
                "overall_narrative": "A quick croissant recipe.",
                "scenes": [
                    {
                        "scene_id": 1,
                        "description": "Hands folding croissant dough.",
                        "image_description": "Person folding pastry dough on marble surface",
                        "subtitle": "Fold it three times.",
                        "audio_cue": "VO: Fold precisely.",
                        "image_url": "https://images.unsplash.com/baseline_image.jpg",
                        "duration_seconds": 6,
                    }
                ],
            }
        }

        rendered_bl = render_storyboard(baseline_data)
        self.assertIn("Visual: Hands folding croissant dough.", rendered_bl)
        self.assertIn("Visual Asset Photo: Person folding pastry dough on marble surface", rendered_bl)
        self.assertIn("Subtitle: Fold it three times.", rendered_bl)
        self.assertIn("Voiceover: VO: Fold precisely.", rendered_bl)
        self.assertNotIn("https://", rendered_bl)
        self.assertNotIn("baseline_image", rendered_bl)

    def test_split_history_random_and_sequential(self):
        from src.tools.import_user_history import split_history

        sample_videos = [{"title": f"Video {i}", "url": f"https://youtube.com/shorts/{i}"} for i in range(10)]

        # Test random split into 3 parts
        parts_random = split_history(sample_videos, parts=3, mode="random", seed=42)
        self.assertEqual(len(parts_random), 3)
        # Check balanced sizes (3 + 3 + 4 = 10)
        total_items = sum(len(p) for p in parts_random)
        self.assertEqual(total_items, 10)
        all_urls = [v["url"] for p in parts_random for v in p]
        self.assertEqual(len(set(all_urls)), 10)  # No duplicates

        # Test reproducibility with same seed
        parts_random_seed = split_history(sample_videos, parts=3, mode="random", seed=42)
        self.assertEqual(parts_random, parts_random_seed)

        # Test sequential split preserves original order
        parts_seq = split_history(sample_videos, parts=2, mode="sequential")
        self.assertEqual(len(parts_seq), 2)
        self.assertEqual([v["title"] for v in parts_seq[0]], [f"Video {i}" for i in range(5)])
        self.assertEqual([v["title"] for v in parts_seq[1]], [f"Video {i}" for i in range(5, 10)])

        # Test single part
        single = split_history(sample_videos, parts=1)
        self.assertEqual(len(single), 1)
        self.assertEqual(len(single[0]), 10)

    def test_user_output_directory_structure_and_naming(self):
        """Test per-user subfolder structure and universal naming convention."""
        import tempfile, re
        from unittest.mock import MagicMock
        from baseline.single_agent_baseline import SingleAgentBaseline

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = (
            '{"title": "Paper Art", "overall_narrative": "Paper folds.", '
            '"scenes": [{"scene_id": 1, "description": "Paper desk", '
            '"image_query": "origami paper", "subtitle": "Fold here.", '
            '"audio_cue": "Voice: Look at this paper.", "duration_seconds": 10}]}'
        )
        mock_response.usage.prompt_tokens = 40
        mock_response.usage.completion_tokens = 40
        mock_client.chat.completions.create.return_value = mock_response

        with tempfile.TemporaryDirectory() as tmp_dir:
            baseline = SingleAgentBaseline(model="deepseek-chat", output_dir=tmp_dir)
            baseline.client = mock_client
            state = baseline.run(
                [{"title": "Paper Magic", "url": "https://youtube.com/shorts/test"}],
                user_id="charles"
            )

            self.assertEqual(state.user_id, "charles")
            self.assertTrue(re.match(r"^baseline_charles_\d{8}_\d{6}$", state.run_id))

            user_baseline_dir = Path(tmp_dir) / "charles" / "baseline"
            self.assertTrue(user_baseline_dir.exists(), "User baseline directory should exist")

            json_files = list(user_baseline_dir.glob("baseline_storyboard_charles_*.json"))
            md_files = list(user_baseline_dir.glob("baseline_storyboard_charles_*.md"))
            self.assertEqual(len(json_files), 1, "Should create exactly 1 baseline storyboard JSON")
            self.assertEqual(len(md_files), 1, "Should create exactly 1 companion Markdown file")

            filename = json_files[0].name
            self.assertTrue(
                re.match(r"^baseline_storyboard_charles_\d{8}_\d{6}\.json$", filename),
                f"Filename '{filename}' must follow universal naming convention"
            )

    def test_clean_voiceover_and_markdown(self):
        from src.core.state import clean_voiceover, SharedState, Storyboard

        # 1. Test clean_voiceover strips VO labels, SFX, and outer quotes
        raw1 = 'VO (flat, dead-direct, no music): "That warning label isn\'t for you. It\'s for about one in twelve thousand people." SFX: low turntable hum, a single sharp camera-shutter click on the punch-in.'
        self.assertEqual(
            clean_voiceover(raw1),
            "That warning label isn't for you. It's for about one in twelve thousand people."
        )

        raw2 = "Voiceover: 'Can an all-powerful God create a stone so heavy that even He cannot lift it?' Deep rumble SFX."
        self.assertEqual(
            clean_voiceover(raw2),
            "Can an all-powerful God create a stone so heavy that even He cannot lift it?"
        )

        raw3 = "Voix off : « Regardez ce stylo, c'est une légende. » SFX : clic métallique"
        self.assertEqual(
            clean_voiceover(raw3),
            "Regardez ce stylo, c'est une légende."
        )

        # 2. Test Storyboard.sanitize_scenes populates voiceover and audio_cue
        sb = Storyboard(
            title="Test Voiceover Storyboard",
            scenes=[
                {
                    "scene_id": 1,
                    "description": "A close up on a soda can",
                    "subtitle": "Warning label",
                    "voiceover": raw1,
                    "duration_seconds": 10,
                }
            ]
        )
        sb.sanitize_scenes()
        self.assertEqual(
            sb.scenes[0]["voiceover"],
            "That warning label isn't for you. It's for about one in twelve thousand people."
        )
        self.assertEqual(
            sb.scenes[0]["audio_cue"],
            "That warning label isn't for you. It's for about one in twelve thousand people."
        )

        # 3. Test markdown output format uses - **Voiceover:**
        state = SharedState(storyboard=sb, run_id="test_vo_md")
        md = state.to_markdown()
        self.assertIn("- **Voiceover:** That warning label isn't for you.", md)
        self.assertNotIn("- **Audio / Voiceover:**", md)
        self.assertNotIn("SFX:", md)
        self.assertNotIn("VO:", md)

    def test_reasoning_effort_config_and_llm_client(self):
        import os
        from unittest.mock import MagicMock
        from src.orchestrator import Orchestrator
        from src.core.llm_client import (
            chat_completion,
            get_reasoning_effort,
            set_reasoning_effort,
            reset_token_usage,
            get_token_usage,
        )

        prev_effort = get_reasoning_effort()
        prev_env = os.environ.get("LLM_REASONING_EFFORT")
        try:
            # 1. Config loads reasoning_effort="low"
            orch = Orchestrator.from_config("configs/config.yaml")
            self.assertEqual(orch.reasoning_effort, "low")
            self.assertEqual(get_reasoning_effort(), "low")

            mock_client = MagicMock()
            mock_resp = MagicMock()
            mock_resp.choices = [MagicMock()]
            mock_resp.choices[0].message.content = "ok"
            mock_resp.usage.prompt_tokens = 100
            mock_resp.usage.completion_tokens = 250
            mock_resp.usage.completion_tokens_details.reasoning_tokens = 180
            mock_client.chat.completions.create.return_value = mock_resp

            reset_token_usage()

            # 2. Reasoner receives reasoning_effort="low", max_tokens>=8192, no temperature
            chat_completion(mock_client, "deepseek-reasoner", "sys", "usr", max_tokens=1500)
            kwargs_reasoner = mock_client.chat.completions.create.call_args.kwargs
            self.assertEqual(kwargs_reasoner.get("reasoning_effort"), "low")
            self.assertEqual(kwargs_reasoner.get("max_tokens"), 8192)
            self.assertNotIn("temperature", kwargs_reasoner)

            usage = get_token_usage()
            self.assertEqual(usage["reasoning_tokens"], 180)
            self.assertEqual(usage["completion_tokens"], 250)

            # 3. Non-reasoner omits reasoning_effort and passes temperature
            chat_completion(mock_client, "deepseek-chat", "sys", "usr", temperature=0.4, max_tokens=1000)
            kwargs_chat = mock_client.chat.completions.create.call_args.kwargs
            self.assertNotIn("reasoning_effort", kwargs_chat)
            self.assertEqual(kwargs_chat.get("temperature"), 0.4)
            self.assertEqual(kwargs_chat.get("max_tokens"), 1000)

            # 4. reasoning_effort="none" disables thinking constraints on reasoner
            os.environ["LLM_REASONING_EFFORT"] = "none"
            chat_completion(mock_client, "deepseek-reasoner", "sys", "usr", temperature=0.5, max_tokens=1200)
            kwargs_none = mock_client.chat.completions.create.call_args.kwargs
            self.assertEqual(kwargs_none.get("reasoning_effort"), "none")
            self.assertEqual(kwargs_none.get("max_tokens"), 1200)
            self.assertEqual(kwargs_none.get("temperature"), 0.5)
        finally:
            if prev_env is None:
                os.environ.pop("LLM_REASONING_EFFORT", None)
            else:
                os.environ["LLM_REASONING_EFFORT"] = prev_env
            set_reasoning_effort(prev_effort)

    def test_orchestrator_end_to_end_progress_logs(self):
        import io
        import tempfile
        from contextlib import redirect_stdout
        from pathlib import Path
        from unittest.mock import MagicMock, patch
        from src.orchestrator import Orchestrator

        # Ordered mock responses for the 8 LLM calls across Phases 1..7:
        # 1. Profiler R1, 2. Profiler R2, 3. WritersRoom, 4. Director,
        # 5. KnowledgeRetriever, 6. Scriptwriter Draft,
        # 7. ArtDirector Queries, 8. ArtDirector Selection, 9. Scriptwriter Refine
        responses = [
            '{"preliminary_interests": ["origami", "math"], "preliminary_themes": ["paper folding"], "investigate_targets": [{"vid_id": "vid1", "reason": "Check math angle"}]}',
            '{"interests": ["origami math"], "themes": ["exponential growth"], "preferred_style": "curiosity", "raw_summary": "Loves paper math puzzles.", "investigated_insights": ["Folding 42 times"]}',
            '{"pitches": [{"pitch_id": "pitch1", "title": "Fold to the Moon", "target_tone": "Curious", "hook_3s": "Grab one sheet of paper.", "concept": "42 folds math", "core_beats": ["Hook", "Math", "Payoff"]}]}',
            '{"winning_pitch_id": "pitch1", "critiques": [{"pitch_id": "pitch1", "verdict": "Sharp visual hook."}], "director_notes": "Keep it visual and crisp."}',
            '{"queries": [{"query": "exponential paper folding math", "domain": "technical", "rationale": "Verify thickness math"}]}',
            '{"title": "Fold to the Moon", "overall_narrative": "Math puzzle.", "scenes": [{"scene_id": 1, "description": "White paper on wood desk", "subtitle": "Forty-two folds.", "voiceover": "One sheet of paper can reach the Moon.", "duration_seconds": 10}]}',
            '{"queries": [{"scene_id": 1, "primary_query": "white paper desk", "fallback_query": "paper"}]}',
            '{"selections": [{"scene_id": 1, "selected_candidate_id": "cand1"}]}',
            '{"title": "Fold to the Moon", "overall_narrative": "Math puzzle.", "scenes": [{"scene_id": 1, "description": "White paper on wood desk", "subtitle": "Forty-two folds.", "voiceover": "One sheet of paper can reach the Moon.", "image_ref": "img1", "duration_seconds": 10}]}',
        ]

        mock_client = MagicMock()
        mock_msgs = []
        for body in responses:
            r = MagicMock()
            r.choices = [MagicMock()]
            r.choices[0].message.content = body
            r.usage.prompt_tokens = 50
            r.usage.completion_tokens = 80
            r.usage.completion_tokens_details.reasoning_tokens = 20
            mock_msgs.append(r)
        mock_client.chat.completions.create.side_effect = mock_msgs

        with tempfile.TemporaryDirectory() as tmp_dir, \
             patch("src.orchestrator.get_client", return_value=mock_client), \
             patch("src.agents.profiler_agent.get_video_transcript", return_value="Fold a sheet 42 times."), \
             patch("src.agents.profiler_agent.get_video_metadata", return_value={"author": "MathChannel"}), \
             patch("src.agents.knowledge_retriever_agent.search_knowledge", return_value=[{"headline": "Exponential Paper Folding", "summary": "0.1mm * 2^42 > 384,400km", "url": "https://example.com"}]), \
             patch("src.agents.art_director_agent.search_image_candidates", return_value=[{"candidate_id": "cand1", "url": "https://images.unsplash.com/photo-test", "alt_text": "Paper", "description": "White paper on desk"}]):

            orch = Orchestrator(
                profiler_model="deepseek-chat",
                retriever_model="deepseek-chat",
                image_model="deepseek-chat",
                scriptwriter_model="deepseek-reasoner",
                writers_room_model="deepseek-reasoner",
                director_model="deepseek-reasoner",
                reasoning_effort="low",
                output_dir=tmp_dir,
            )

            buf = io.StringIO()
            with redirect_stdout(buf):
                state = orch.run([{"title": "Folding Paper", "url": "https://youtube.com/shorts/abc"}], user_id="test_prog")

            out = buf.getvalue()
            for phase_idx in range(1, 8):
                self.assertIn(f"[Phase {phase_idx}/7]", out)
                self.assertIn(f"Phase {phase_idx}/7 done", out)
            self.assertIn("reasoning_effort=low", out)
            self.assertIn("3s Hook:", out)
            self.assertIn("Draft created:", out)
            self.assertIn("img=img1", out)
            self.assertTrue(Path(state.output_json_path).exists())

    def test_video_frame_shows_visual_description_below_image(self):
        from PIL import Image
        import storyboard_to_video as s2v

        src_img = Image.new("RGB", (600, 400), (220, 40, 40))
        scene = {
            "scene_id": 2,
            "description": "Gros plan sur un marteau de bois posé sur une table sombre.",
            "subtitle": "NE DOIT PAS APPARAITRE",
            "voiceover": "Dix-huit mois ferme.",
        }
        frame = s2v.render_scene_frame(src_img, scene, (1280, 720), lang="fr")
        self.assertEqual(frame.size, (1280, 720))

        # Top area contains the fitted image (red pixel near upper center)
        top_center = frame.getpixel((640, 260))
        self.assertGreater(top_center[0], 180)
        self.assertLess(top_center[1], 80)

        # Bottom description panel (y >= 519) is dark slate (#141824) with bright text pixels, not red photo
        bottom_bg_vals = [frame.getpixel((10, 620))]
        self.assertLess(bottom_bg_vals[0][0], 40)

        panel_crop = frame.crop((0, 520, 1280, 720))
        bright_pixels = sum(1 for r, g, b in panel_crop.getdata() if r > 150 or b > 180)
        self.assertGreater(bright_pixels, 500)

        # Changing subtitle must not alter rendered frame pixels (subtitles omitted from video)
        scene_alt_sub = dict(scene, subtitle="COMPLETELY DIFFERENT SUBTITLE TEXT")
        frame_alt_sub = s2v.render_scene_frame(src_img, scene_alt_sub, (1280, 720), lang="fr")
        self.assertEqual(frame.tobytes(), frame_alt_sub.tobytes())

        # Missing image renders clean placeholder + visual description frame without crashing
        ph_frame = s2v.render_scene_frame(None, scene, (1280, 720), lang="en")
        self.assertEqual(ph_frame.size, (1280, 720))

    def test_config_baseline_reasoner_and_pipeline_wiring(self):
        import yaml
        from src.orchestrator import Orchestrator
        from baseline.single_agent_baseline import SingleAgentBaseline

        repo_root = Path(__file__).resolve().parent.parent
        cfg_path = repo_root / "configs" / "config.yaml"
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        self.assertEqual(cfg["baseline"]["model"], "deepseek-reasoner")
        self.assertEqual(cfg["llm"]["reasoning_effort"], "low")

        orch = Orchestrator.from_config(str(cfg_path))
        base = SingleAgentBaseline.from_config(str(cfg_path))
        self.assertEqual(orch.scriptwriter.target_duration, 60)
        self.assertEqual(orch.knowledge_retriever.max_results, 5)
        self.assertEqual(base.model, "deepseek-reasoner")
        self.assertEqual(base.target_duration, 60)

        custom_cfg = {
            "llm": {"base_url": "https://api.deepseek.com", "reasoning_effort": "low"},
            "models": {"profiler": "deepseek-chat", "scriptwriter": "deepseek-reasoner"},
            "baseline": {"model": "deepseek-reasoner"},
            "pipeline": {"target_duration_seconds": 45, "max_news_results": 2},
            "output_dir": "outputs",
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            custom_path = Path(tmp_dir) / "custom.yaml"
            custom_path.write_text(yaml.safe_dump(custom_cfg), encoding="utf-8")
            custom_orch = Orchestrator.from_config(str(custom_path))
            custom_base = SingleAgentBaseline.from_config(str(custom_path))
            self.assertEqual(custom_orch.scriptwriter.target_duration, 45)
            self.assertEqual(custom_orch.knowledge_retriever.max_results, 2)
            self.assertEqual(custom_base.target_duration, 45)

    def test_prompts_subtitle_free_english_image_queries_and_agent_loaders(self):
        from baseline.single_agent_baseline import _BASELINE_SYSTEM_PROMPT
        from src.agents.scriptwriter_agent import _DRAFT_SYSTEM_PROMPT, _REFINE_SYSTEM_PROMPT
        from src.agents.profiler_agent import _ROUND1_MACRO_SYSTEM_PROMPT, _ROUND2_SYNTHESIS_SYSTEM_PROMPT
        from src.agents.knowledge_retriever_agent import _NEWS_BRAINSTORM_SYSTEM_PROMPT
        from src.agents.art_director_agent import _IMAGE_QUERY_SYSTEM_PROMPT, _ART_DIRECTOR_SYSTEM_PROMPT

        repo_root = Path(__file__).resolve().parent.parent

        # Subtitle schema field removed & voiceover pacing tightened
        for prompt_text in (_BASELINE_SYSTEM_PROMPT, _DRAFT_SYSTEM_PROMPT, _REFINE_SYSTEM_PROMPT):
            self.assertNotIn('"subtitle":', prompt_text)
            self.assertIn("15 to 25", prompt_text)

        # English-only stock photo search rule enforced
        self.assertIn("ENGLISH", _IMAGE_QUERY_SYSTEM_PROMPT)
        self.assertIn("ENGLISH", _BASELINE_SYSTEM_PROMPT)

        # All agents load from prompts/*.txt cleanly
        self.assertEqual(_ROUND1_MACRO_SYSTEM_PROMPT, (repo_root / "prompts" / "profiler_r1_macro.txt").read_text(encoding="utf-8").strip())
        self.assertEqual(_ROUND2_SYNTHESIS_SYSTEM_PROMPT, (repo_root / "prompts" / "profiler_r2_synthesis.txt").read_text(encoding="utf-8").strip())
        self.assertEqual(_NEWS_BRAINSTORM_SYSTEM_PROMPT, (repo_root / "prompts" / "news_brainstorm.txt").read_text(encoding="utf-8").strip())
        self.assertEqual(_IMAGE_QUERY_SYSTEM_PROMPT, (repo_root / "prompts" / "image_query.txt").read_text(encoding="utf-8").strip())
        self.assertEqual(_ART_DIRECTOR_SYSTEM_PROMPT, (repo_root / "prompts" / "art_director_selection.txt").read_text(encoding="utf-8").strip())

    def test_per_model_mixed_cost_accounting(self):
        from unittest.mock import MagicMock
        from src.core.llm_client import (
            chat_completion,
            reset_token_usage,
            get_token_usage,
            get_token_usage_by_model,
            get_session_cost_usd,
        )

        reset_token_usage()
        mock_client = MagicMock()
        resp_chat = MagicMock()
        resp_chat.choices = [MagicMock()]
        resp_chat.choices[0].message.content = "ok"
        resp_chat.usage.prompt_tokens = 1_000_000
        resp_chat.usage.completion_tokens = 1_000_000
        resp_chat.usage.completion_tokens_details.reasoning_tokens = 0

        resp_reason = MagicMock()
        resp_reason.choices = [MagicMock()]
        resp_reason.choices[0].message.content = "ok"
        resp_reason.usage.prompt_tokens = 1_000_000
        resp_reason.usage.completion_tokens = 1_000_000
        resp_reason.usage.completion_tokens_details.reasoning_tokens = 200_000

        mock_client.chat.completions.create.side_effect = [resp_chat, resp_reason]
        chat_completion(mock_client, "deepseek-chat", "sys", "usr")
        chat_completion(mock_client, "deepseek-reasoner", "sys", "usr")

        totals = get_token_usage()
        self.assertEqual(totals["prompt_tokens"], 2_000_000)
        self.assertEqual(totals["completion_tokens"], 2_000_000)
        by_model = get_token_usage_by_model()
        self.assertIn("deepseek-chat", by_model)
        self.assertIn("deepseek-reasoner", by_model)
        # Expected: (0.14 + 0.28) + (0.55 + 2.19) = 3.16 USD (not 5.48 USD)
        self.assertAlmostEqual(get_session_cost_usd(provider="deepseek"), 3.16, places=4)
        reset_token_usage()

    def test_clean_voiceover_midstring_speaker_cues_and_subtitle_free_views(self):
        import storyboard_to_video as s2v
        from evaluation.llm_judge import render_storyboard
        from src.core.state import clean_voiceover, SharedState, Storyboard

        self.assertEqual(s2v.DEFAULT_PADDING, 0.3)

        raw = 'The door bursts open. CLINICIAN 1 (calling down the hall): "Get the crash cart!"'
        cleaned = clean_voiceover(raw)
        self.assertNotIn("CLINICIAN", cleaned)
        self.assertNotIn("calling down the hall", cleaned)
        self.assertEqual(cleaned, "The door bursts open. Get the crash cart!")
        self.assertEqual(s2v.clean_audio_cue(raw), cleaned)

        self.assertEqual(clean_voiceover("CLINICIAN 1: Check her vitals."), "Check her vitals.")
        self.assertEqual(clean_voiceover('VO: Hello. MAN IN SUIT (whispering): "Run!"'), "Hello. Run!")

        # Subtitle-free scenes omit Subtitle lines from Markdown and blind judge view
        st = SharedState(
            storyboard=Storyboard(
                title="Clean Video",
                scenes=[{"scene_id": 1, "description": "Wooden desk", "voiceover": "Spoken line.", "duration_seconds": 10}],
            )
        )
        md = st.to_markdown()
        self.assertNotIn("Subtitle", md)
        judged_view = render_storyboard({"storyboard": {"title": "Clean Video", "overall_narrative": "Arc", "scenes": st.storyboard.scenes}})
        self.assertNotIn("Subtitle:", judged_view)


if __name__ == "__main__":
    unittest.main()

