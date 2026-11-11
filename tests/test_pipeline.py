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

            # 2. Reasoner receives reasoning_effort="low", max_tokens>=16384, no temperature
            chat_completion(mock_client, "deepseek-reasoner", "sys", "usr", max_tokens=1500)
            kwargs_reasoner = mock_client.chat.completions.create.call_args.kwargs
            self.assertEqual(kwargs_reasoner.get("reasoning_effort"), "low")
            self.assertEqual(kwargs_reasoner.get("max_tokens"), 16384)
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

    def test_language_aware_transcript_and_round_robin_snippets(self):
        from unittest.mock import MagicMock, patch
        from src.tools.youtube_tool import detect_text_language
        from src.agents.writers_room_agent import _WRITERS_ROOM_SYSTEM_PROMPT
        from src.agents.director_agent import _DIRECTOR_SYSTEM_PROMPT
        from src.agents.scriptwriter_agent import _DRAFT_SYSTEM_PROMPT, _build_draft_prompt
        from src.agents.knowledge_retriever_agent import KnowledgeRetrieverAgent
        from src.core.state import KnowledgeContext, SharedState, UserProfile

        # Language detection distinguishes French vs English watch history items cleanly
        self.assertEqual(detect_text_language("POV : l'IA du site veut absolument te faire croire que c'est un humain"), "fr")
        self.assertEqual(detect_text_language("The Secret To Crispy Chicken Is Water?"), "en")

        # Anti-regurgitation & standalone clarity rules present in ideation/drafting prompts
        self.assertIn("ANTI-REGURGITATION", _WRITERS_ROOM_SYSTEM_PROMPT)
        self.assertIn("Anti-Regurgitation", _DIRECTOR_SYSTEM_PROMPT)
        self.assertIn("STANDALONE CLARITY", _DRAFT_SYSTEM_PROMPT)

        # Round-robin snippet interleaving gives query 2 & 3 equal priority in top-6/9 slices
        mock_client = MagicMock()
        r_queries = MagicMock()
        r_queries.choices = [MagicMock()]
        r_queries.choices[0].message.content = '{"queries": [{"query": "q1", "domain": "news"}, {"query": "q2", "domain": "technical"}, {"query": "q3", "domain": "encyclopedia"}]}'
        r_queries.usage = None
        mock_client.chat.completions.create.return_value = r_queries

        st = SharedState(
            user_profile=UserProfile(interests=["math"], themes=["logic"], raw_summary="Test"),
            selected_pitch={"title": "Test Pitch", "target_tone": "curious", "concept": "c", "hook_3s": "h"},
        )
        agent = KnowledgeRetrieverAgent(model="deepseek-chat", client=mock_client, max_results=3, num_queries=3)

        def fake_search(query, domain, max_results, client, model):
            return [{"headline": f"{query}_{i}", "summary": f"fact {query} {i}", "url": "https://ex.com"} for i in range(3)]

        with patch("src.agents.knowledge_retriever_agent.search_knowledge", side_effect=fake_search):
            st = agent.run(st)

        top_headlines = [s["headline"] for s in st.knowledge_context.snippets[:3]]
        self.assertEqual(top_headlines, ["q1_0", "q2_0", "q3_0"])
        draft_prompt = _build_draft_prompt(st, 60)
        for h in top_headlines:
            self.assertIn(h, draft_prompt)

    def test_writers_room_and_director_retry_on_bad_json_and_rate_limit_bailout(self):
        import os
        from unittest.mock import MagicMock, patch
        from src.agents.writers_room_agent import WritersRoomAgent
        from src.agents.director_agent import DirectorAgent
        from src.core.state import SharedState, UserProfile
        from src.tools.image_tool import search_image_candidates

        # 1. WritersRoomAgent retries once when attempt 1 returns unparseable JSON
        mock_client = MagicMock()
        bad_resp = MagicMock()
        bad_resp.choices = [MagicMock()]
        bad_resp.choices[0].message.content = "Not valid JSON {"
        bad_resp.usage = None

        good_wr = MagicMock()
        good_wr.choices = [MagicMock()]
        good_wr.choices[0].message.content = (
            '{"pitches": [{"pitch_id": "pitch1", "title": "Bread Fridge Physics", '
            '"target_tone": "Curieux", "hook_3s": "Stop!", "concept": "Starch retrogradation.", '
            '"core_beats": ["b1"], "user_alignment_reason": "Food science"}]}'
        )
        good_wr.usage = None

        good_dir = MagicMock()
        good_dir.choices = [MagicMock()]
        good_dir.choices[0].message.content = (
            '{"critiques": [{"pitch_id": "pitch1", "strengths": "Clear", "weaknesses": "None"}], '
            '"winning_pitch_id": "pitch1", "selection_rationale": "Strong hook.", '
            '"director_notes": "Keep concrete visuals."}'
        )
        good_dir.usage = None

        mock_client.chat.completions.create.side_effect = [bad_resp, good_wr, bad_resp, good_dir]

        st = SharedState(user_profile=UserProfile(interests=["food science"], themes=["myth busting"], raw_summary="French"))
        st = WritersRoomAgent(model="deepseek-chat", client=mock_client).run(st)
        self.assertEqual(len(st.candidate_pitches), 1)
        self.assertEqual(st.candidate_pitches[0]["title"], "Bread Fridge Physics")

        st = DirectorAgent(model="deepseek-chat", client=mock_client).run(st)
        self.assertEqual(st.selected_pitch["pitch_id"], "pitch1")
        self.assertEqual(st.director_notes, "Keep concrete visuals.")
        self.assertEqual(mock_client.chat.completions.create.call_count, 4)

        # 2. Unsplash HTTP 403 short-circuits immediately (only 1 Unsplash HTTP call, no fallback waste)
        resp_403 = MagicMock()
        resp_403.status_code = 403
        resp_403.text = "Rate Limit Exceeded"
        resp_403.headers = {"X-Ratelimit-Remaining": "0"}

        with patch.dict(os.environ, {"IMAGE_API_PROVIDER": "unsplash", "UNSPLASH_ACCESS_KEY": "fake_u", "PEXELS_API_KEY": ""}, clear=False), \
             patch("src.tools.image_tool.requests.get", return_value=resp_403) as mock_get:
            res = search_image_candidates("ancient carved stone wall", fallback_query="stone wall", max_results=4)

        self.assertEqual(mock_get.call_count, 1)
        self.assertTrue(res[0]["url"].startswith("https://example.com/images/"))

    def test_video_intro_padding_and_pcm_mkv_intermediates(self):
        import types
        import tempfile
        from unittest.mock import patch
        import storyboard_to_video as s2v

        self.assertEqual(s2v.DEFAULT_PADDING, 0.3)
        self.assertEqual(s2v.DEFAULT_INTRO_PADDING, 1.5)

        captured_cmds = []
        fake_tts = types.SimpleNamespace(synth=lambda text, wav: Path(wav).write_bytes(b"RIFF"))
        args = types.SimpleNamespace(
            padding=0.3,
            intro_padding=1.5,
            json_durations=False,
            raw_audio=False,
        )

        with tempfile.TemporaryDirectory() as d, \
             patch("storyboard_to_video.audio_duration", return_value=5.0), \
             patch("storyboard_to_video.run", side_effect=lambda cmd: captured_cmds.append(cmd)):
            tmp = Path(d)
            c0 = s2v.build_scene_clip(
                {"scene_id": 1, "description": "Opening desk shot", "voiceover": "Hello world.", "duration_seconds": 6},
                0, tmp, fake_tts, (640, 360), args, lang="en",
            )
            c1 = s2v.build_scene_clip(
                {"scene_id": 2, "description": "Second shot", "voiceover": "Next sentence.", "duration_seconds": 6},
                1, tmp, fake_tts, (640, 360), args, lang="en",
            )

        self.assertEqual(c0.suffix, ".mkv")
        self.assertEqual(c1.suffix, ".mkv")
        cmd0_str = " ".join(captured_cmds[0])
        cmd1_str = " ".join(captured_cmds[1])
        self.assertIn("adelay=1500|1500,apad=whole_dur=6.8", cmd0_str)
        self.assertIn("adelay=300|300,apad=whole_dur=5.6", cmd1_str)
        self.assertIn("pcm_s16le", cmd0_str)

    def test_llm_client_bounded_timeout_and_retry_on_timeout(self):
        import os
        from unittest.mock import MagicMock, patch
        from src.core.llm_client import chat_completion, get_client

        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test_key", "LLM_TIMEOUT_SECONDS": "95"}, clear=False):
            client = get_client()
        self.assertEqual(client.timeout, 95.0)
        self.assertEqual(client.max_retries, 0)

        mock_client = MagicMock()
        good_resp = MagicMock()
        good_resp.choices = [MagicMock()]
        good_resp.choices[0].message.content = "ok after retry"
        good_resp.usage = None
        mock_client.chat.completions.create.side_effect = [TimeoutError("stalled"), good_resp]

        out = chat_completion(mock_client, "deepseek-chat", "sys", "usr")
        self.assertEqual(out, "ok after retry")
        self.assertEqual(mock_client.chat.completions.create.call_count, 2)

    def test_first_watch_and_single_narrator_prompts(self):
        from baseline.single_agent_baseline import _BASELINE_SYSTEM_PROMPT, _BASELINE_SYSTEM_PROMPT_DEFAULT
        from src.agents.director_agent import _DIRECTOR_SYSTEM_PROMPT, _DIRECTOR_SYSTEM_PROMPT_DEFAULT
        from src.agents.scriptwriter_agent import (
            _DRAFT_SYSTEM_PROMPT,
            _DRAFT_SYSTEM_PROMPT_DEFAULT,
            _REFINE_SYSTEM_PROMPT,
            _REFINE_SYSTEM_PROMPT_DEFAULT,
            _build_draft_prompt,
            _build_refine_prompt,
        )
        from src.agents.writers_room_agent import _WRITERS_ROOM_SYSTEM_PROMPT, _WRITERS_ROOM_SYSTEM_PROMPT_DEFAULT
        from src.core.state import ImageContext, SharedState, Storyboard, UserProfile

        pairs = [
            ("writers_room.txt", _WRITERS_ROOM_SYSTEM_PROMPT, _WRITERS_ROOM_SYSTEM_PROMPT_DEFAULT, "SINGLE EXTERNAL VOICEOVER NARRATOR"),
            ("director.txt", _DIRECTOR_SYSTEM_PROMPT, _DIRECTOR_SYSTEM_PROMPT_DEFAULT, "single external off-screen"),
            ("scriptwriter_draft.txt", _DRAFT_SYSTEM_PROMPT, _DRAFT_SYSTEM_PROMPT_DEFAULT, "STRICTLY SINGLE EXTERNAL VOICEOVER NARRATOR"),
            ("scriptwriter_refine.txt", _REFINE_SYSTEM_PROMPT, _REFINE_SYSTEM_PROMPT_DEFAULT, "STRICTLY SINGLE EXTERNAL VOICEOVER NARRATOR"),
            ("baseline.txt", _BASELINE_SYSTEM_PROMPT, _BASELINE_SYSTEM_PROMPT_DEFAULT, "STRICTLY SINGLE EXTERNAL VOICEOVER NARRATOR"),
        ]
        repo_root = Path(__file__).resolve().parent.parent
        for fname, loaded, default, marker in pairs:
            self.assertEqual(loaded, (repo_root / "prompts" / fname).read_text(encoding="utf-8").strip())
            for variant in (loaded, default):
                self.assertIn(marker, variant)
                self.assertNotIn("narrative dialogue", variant)
                self.assertNotIn("voiceover dialogue", variant)

        for p in (_WRITERS_ROOM_SYSTEM_PROMPT, _DRAFT_SYSTEM_PROMPT, _REFINE_SYSTEM_PROMPT, _BASELINE_SYSTEM_PROMPT):
            self.assertIn("FIRST WATCH", p.upper().replace("-", " "))

        st = SharedState(
            user_profile=UserProfile(interests=["math"], themes=["logic"], raw_summary="Test"),
            storyboard=Storyboard(title="T", scenes=[{"scene_id": 1, "description": "d", "voiceover": "v", "duration_seconds": 10}]),
            image_context=ImageContext(images=[{"image_id": "img1", "alt_text": "paper", "query": "paper"}]),
        )
        self.assertIn("first watch", _build_draft_prompt(st, 60))
        self.assertIn("external off-screen narrator", _build_refine_prompt(st))

    def test_cluster_weighted_profiling_and_judge_relevance_rubric(self):
        from unittest.mock import MagicMock, patch
        import evaluation.llm_judge as judge_mod
        from src.agents.director_agent import _DIRECTOR_SYSTEM_PROMPT, _DIRECTOR_SYSTEM_PROMPT_DEFAULT, _build_director_prompt
        from src.agents.profiler_agent import (
            ProfilerAgent,
            _ROUND1_MACRO_SYSTEM_PROMPT,
            _ROUND1_MACRO_SYSTEM_PROMPT_DEFAULT,
            _ROUND2_SYNTHESIS_SYSTEM_PROMPT,
            _ROUND2_SYNTHESIS_SYSTEM_PROMPT_DEFAULT,
        )
        from src.agents.writers_room_agent import _WRITERS_ROOM_SYSTEM_PROMPT, _build_writers_room_prompt
        from src.core.state import SharedState, UserHistory, UserProfile

        self.assertEqual(_ROUND1_MACRO_SYSTEM_PROMPT, _ROUND1_MACRO_SYSTEM_PROMPT_DEFAULT.strip())
        self.assertEqual(_ROUND2_SYNTHESIS_SYSTEM_PROMPT, _ROUND2_SYNTHESIS_SYSTEM_PROMPT_DEFAULT.strip())
        self.assertIn("dominant_clusters", _ROUND2_SYNTHESIS_SYSTEM_PROMPT)
        self.assertIn("CLUSTER-ANCHORED", _WRITERS_ROOM_SYSTEM_PROMPT)
        self.assertIn("DOMINANT multi-video watch-history clusters", _DIRECTOR_SYSTEM_PROMPT_DEFAULT)
        self.assertIn("DOMINANT multi-video watch-history clusters", _DIRECTOR_SYSTEM_PROMPT)

        pair_prompt = judge_mod.build_pair_prompt("- vid1\n- vid2", "Title: A", "Title: B")
        rel_line = next(l for l in pair_prompt.splitlines() if l.startswith("1. relevance:"))
        self.assertIn("Anti-Regurgitation", rel_line)
        self.assertIn("3/5", rel_line)

        mock_client = MagicMock()
        r1 = MagicMock()
        r1.choices = [MagicMock()]
        r1.choices[0].message.content = (
            '{"interest_clusters": [{"cluster": "GoT Lore", "video_count": 7, "example_titles": ["Aerion Trial"]}], '
            '"preliminary_interests": ["GoT"], "preliminary_themes": ["Medieval"], '
            '"investigate_targets": [{"vid_id": "vid1", "reason": "Top cluster"}]}'
        )
        r1.usage = None
        r2 = MagicMock()
        r2.choices = [MagicMock()]
        r2.choices[0].message.content = (
            '{"dominant_clusters": [{"cluster": "GoT Lore", "video_count": 7, "share_pct": 70, "example_titles": ["Aerion Trial"]}], '
            '"interests": ["GoT trial by combat"], "themes": ["Medieval Law"], "preferred_style": "infotainment", '
            '"investigated_insights": ["From vid1: trial by combat"], "raw_summary": "English GoT lore fan."}'
        )
        r2.usage = None
        mock_client.chat.completions.create.side_effect = [r1, r2]

        st = SharedState(
            user_history=UserHistory(videos=[
                {"title": "Aerion Refuses Trial By Combat", "url": "https://youtube.com/shorts/a1"},
                {"title": "Rare Sanity Check Title", "url": "https://youtube.com/shorts/a2"},
            ])
        )
        agent = ProfilerAgent(model="deepseek-chat", client=mock_client, max_investigate=1)
        with patch("src.agents.profiler_agent.get_video_transcript", return_value="Trial by combat lore."), \
             patch("src.agents.profiler_agent.get_video_metadata", return_value={"author": "LoreVault"}):
            st = agent.run(st)

        # Round 2 prompt received COMPLETE WATCH HISTORY including a2 even though only vid1 transcript was fetched
        r2_user_prompt = mock_client.chat.completions.create.call_args_list[1].kwargs["messages"][1]["content"]
        self.assertIn("COMPLETE WATCH HISTORY", r2_user_prompt)
        self.assertIn("Rare Sanity Check Title", r2_user_prompt)
        self.assertIn("GoT Lore (7 videos", r2_user_prompt)

        # Dominant clusters round-trip through SharedState serialization & feed WritersRoom / Director
        restored = SharedState.from_dict(st.to_dict())
        self.assertEqual(restored.user_profile.dominant_clusters[0]["cluster"], "GoT Lore")
        self.assertIn("GoT Lore (7 videos, ~70%)", _build_writers_room_prompt(restored.user_profile))
        self.assertIn("GoT Lore (7 videos, ~70%)", _build_director_prompt(restored.user_profile, []))

    def test_scriptwriter_refine_truncation_retry_and_fallback_preservation(self):
        """ScriptwriterAgent.refine must retry on truncated scene counts and preserve missing draft scenes."""
        import json
        from unittest.mock import MagicMock
        from src.agents.scriptwriter_agent import ScriptwriterAgent
        from src.core.state import ImageContext, SharedState, Storyboard

        st = SharedState(
            storyboard=Storyboard(
                title="Braess Paradox",
                target_duration_seconds=60,
                overall_narrative="Full arc setup + payoff.",
                scenes=[
                    {"scene_id": i, "description": f"Desc {i}", "voiceover": f"Voice {i}", "duration_seconds": 10}
                    for i in range(1, 7)
                ],
            ),
            image_context=ImageContext(
                images=[
                    {"image_id": f"img{i}", "target_scene_id": i, "url": f"https://images.unsplash.com/photo-{i}"}
                    for i in range(1, 7)
                ]
            ),
        )

        trunc = MagicMock()
        trunc.choices = [MagicMock()]
        trunc.choices[0].message.content = json.dumps({
            "title": "Braess Paradox",
            "overall_narrative": "Full arc.",
            "scenes": [
                {"scene_id": i, "description": f"Ref {i}", "voiceover": f"RefVO {i}", "image_ref": f"img{i}", "duration_seconds": 10}
                for i in range(1, 5)  # truncated after 4 of 6 scenes!
            ],
        })
        trunc.usage = None

        # Even if retry returns 5/6 scenes, trailing Scene 6 from Draft must be preserved & linked to img6
        partial_retry = MagicMock()
        partial_retry.choices = [MagicMock()]
        partial_retry.choices[0].message.content = json.dumps({
            "title": "Braess Paradox",
            "overall_narrative": "Full arc.",
            "scenes": [
                {"scene_id": i, "description": f"Retry {i}", "voiceover": f"RetryVO {i}", "image_ref": f"img{i}", "duration_seconds": 10}
                for i in range(1, 6)
            ],
        })
        partial_retry.usage = None

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = [trunc, partial_retry]

        agent = ScriptwriterAgent(model="deepseek-reasoner", client=mock_client)
        res = agent.refine(st)

        self.assertEqual(mock_client.chat.completions.create.call_count, 2)
        self.assertEqual(len(res.storyboard.scenes), 6)
        self.assertEqual(res.storyboard.scenes[5]["voiceover"], "Voice 6")
        self.assertEqual(res.storyboard.scenes[5]["image_ref"], "img6")
        self.assertEqual(res.storyboard.scenes[5]["image_url"], "https://images.unsplash.com/photo-6")

    def test_prepare_human_eval_packaging_and_csv_analysis(self):
        """prepare_human_eval must create balanced blind packages and unblind Google Form CSV rows accurately."""
        import csv
        import tempfile
        import zipfile
        from pathlib import Path
        from evaluation.prepare_human_eval import analyze_form_responses, package_human_eval

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            out_src = tmp_path / "outputs"
            for part in ("part1", "part2", "part3"):
                for mode in ("agentic", "baseline"):
                    d = out_src / f"alice_{part}" / mode
                    d.mkdir(parents=True)
                    (d / f"{mode}_video_alice_{part}_20261111_120000.mp4").write_bytes(f"{mode}:{part}".encode())

            pkg_dir = tmp_path / "packages"
            manifest = package_human_eval(
                outputs_dir=out_src,
                out_dir=pkg_dir,
                seed=7,
                form_url="https://docs.google.com/forms/d/e/FAKE/viewform",
                participant_entry="entry.999",
                zip_packages=True,
            )

            self.assertIn("alice", manifest["participants"])
            pairs = manifest["participants"]["alice"]["pairs"]
            self.assertEqual(len(pairs), 3)
            orders = {p["order"] for p in pairs}
            # 3 pairs must contain both AB and BA orders (no all-same position bias)
            self.assertEqual(orders, {"AB", "BA"})

            alice_dir = pkg_dir / "alice"
            guide_text = (alice_dir / "INSTRUCTIONS_EN_FR.md").read_text(encoding="utf-8")
            self.assertIn("entry.999=alice", guide_text)
            self.assertTrue((pkg_dir / "alice.zip").is_file())
            with zipfile.ZipFile(pkg_dir / "alice.zip") as zf:
                self.assertIn("alice/Pair1_VideoA.mp4", zf.namelist())
                self.assertIn("alice/Pair3_VideoB.mp4", zf.namelist())

            # Simulate Form CSV where alice picks Video A whenever A=agentic and Video B whenever B=agentic on Pairs 1-2, and Tie on Pair 3
            csv_path = tmp_path / "responses.csv"
            headers = ["Timestamp", "Participant ID"]
            row = {"Timestamp": "2026-11-11 18:00:00", "Participant ID": "alice"}
            for p in pairs:
                k = p["pair_num"]
                ag_slot = "A" if p["video_a_system"] == "agentic" else "B"
                bl_slot = "B" if ag_slot == "A" else "A"
                for slot, scores in ((ag_slot, (5, 5, 4, 5)), (bl_slot, (3, 2, 3, 2))):
                    cols = [
                        (f"Pair {k} — Video {slot}: Personal Relevance (1–5)", scores[0]),
                        (f"Pair {k} — Video {slot}: Creativity (1–5)", scores[1]),
                        (f"Pair {k} — Video {slot}: Engagement (1–5)", scores[2]),
                        (f"Pair {k} — Video {slot}: Overall Preference (1–5)", scores[3]),
                    ]
                    for h, val in cols:
                        headers.append(h)
                        row[h] = str(val)
                w_col = f"Pair {k} — Which video would you rather watch?"
                headers.append(w_col)
                row[w_col] = f"Video {ag_slot}" if k in (1, 2) else "No preference / Tie"

            with csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writeheader()
                writer.writerow(row)

            analysis = analyze_form_responses(csv_path=csv_path, key_path=pkg_dir / "blinding_key.json")
            self.assertEqual(analysis["n_users"], 1)
            self.assertEqual(analysis["n_pairs"], 3)
            self.assertAlmostEqual(analysis["metrics"]["relevance"]["agentic_mean"], 5.0)
            self.assertAlmostEqual(analysis["metrics"]["relevance"]["baseline_mean"], 3.0)
            self.assertEqual(analysis["pairwise_votes"], {"agentic": 2, "baseline": 0, "tie": 1})
            self.assertAlmostEqual(analysis["win_rate_excl_ties"], 1.0)


if __name__ == "__main__":
    unittest.main()

