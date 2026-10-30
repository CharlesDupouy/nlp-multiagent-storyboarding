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



if __name__ == "__main__":
    unittest.main()

