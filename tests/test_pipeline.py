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


if __name__ == "__main__":
    unittest.main()

