"""
Basic smoke tests for the pipeline data structures and tools.
Run with: python3 -m unittest discover tests
"""

import unittest
from pathlib import Path
import tempfile
from src.core.state import SharedState, UserHistory, ImageContext, Storyboard


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


if __name__ == "__main__":
    unittest.main()

