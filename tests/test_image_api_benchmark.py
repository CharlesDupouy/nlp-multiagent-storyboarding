"""
Offline unit tests for tools/benchmark_image_apis.py parsers and summary helpers.
Run with: python3 -m unittest discover tests
"""

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from tools.benchmark_image_apis import (
    compute_provider_summary,
    is_watermarked_domain,
    parse_openverse_response,
    parse_serper_response,
    parse_wikimedia_response,
    render_html_report,
    strip_html,
)


class TestImageApiBenchmark(unittest.TestCase):
    def test_strip_html_and_watermark_detection(self):
        raw = '<p>Eiffel Tower at <b>sunset</b> &amp; mist</p>'
        self.assertEqual(strip_html(raw), "Eiffel Tower at sunset & mist")

        self.assertTrue(is_watermarked_domain("https://www.shutterstock.com/image-photo/123.jpg"))
        self.assertTrue(is_watermarked_domain("https://media.gettyimages.com/photos/test.jpg"))
        self.assertFalse(is_watermarked_domain("https://upload.wikimedia.org/wikipedia/commons/a/a1/Tower.jpg"))

    def test_parse_openverse_skips_svg_and_formats_tags(self):
        payload = {
            "results": [
                {
                    "title": "Vector diagram",
                    "url": "https://example.org/diagram.svg",
                    "filetype": "svg",
                    "tags": [{"name": "vector"}],
                },
                {
                    "title": "Folding paper",
                    "url": "https://live.staticflickr.com/1/paper.jpg",
                    "thumbnail": "https://api.openverse.org/v1/images/1/thumb/",
                    "filetype": "jpg",
                    "license": "cc0",
                    "source": "flickr",
                    "width": 1024,
                    "height": 768,
                    "tags": [{"name": "origami"}, {"name": "paper"}],
                },
            ]
        }
        cands, meta = parse_openverse_response(payload, max_candidates=4)
        self.assertEqual(len(cands), 1)
        self.assertEqual(meta["skipped_non_raster"], 1)
        self.assertEqual(cands[0]["candidate_id"], "cand1")
        self.assertIn("origami, paper", cands[0]["alt_text"])

    def test_parse_serper_filters_watermarks_and_tracks_raw_counts(self):
        payload = {
            "images": [
                {"title": "Stock 1", "imageUrl": "https://image.shutterstock.com/x.jpg", "source": "Shutterstock", "position": 1},
                {"title": "Clean Kitchen Sink", "imageUrl": "https://example.com/sink.jpg", "source": "Blog", "domain": "example.com", "position": 2},
                {"title": "Stock 2", "imageUrl": "https://media.istockphoto.com/y.jpg", "source": "iStock", "position": 3},
                {"title": "Dishware Photo", "imageUrl": "https://cdn.site.org/dishes.png", "source": "Site", "position": 4},
            ]
        }
        cands, meta = parse_serper_response(payload, max_candidates=4, filter_watermarks=True)
        self.assertEqual(meta["raw_top_k_watermarked"], 2)
        self.assertEqual(len(cands), 2)
        self.assertEqual(cands[0]["url"], "https://example.com/sink.jpg")
        self.assertEqual(cands[1]["url"], "https://cdn.site.org/dishes.png")

    def test_parse_wikimedia_sorts_by_index_and_filters_mime(self):
        payload = {
            "query": {
                "pages": {
                    "200": {
                        "index": 2,
                        "title": "File:Second_photo.jpg",
                        "imageinfo": [{
                            "mime": "image/jpeg",
                            "thumburl": "https://thumb.wikimedia.org/second_1280.jpg",
                            "url": "https://upload.wikimedia.org/second.jpg",
                            "thumbwidth": 1280,
                            "thumbheight": 853,
                            "extmetadata": {
                                "ImageDescription": {"value": "<div>Second description</div>"},
                                "LicenseShortName": {"value": "CC BY 4.0"},
                            },
                        }],
                    },
                    "100": {
                        "index": 1,
                        "title": "File:Vector_icon.svg",
                        "imageinfo": [{
                            "mime": "image/svg+xml",
                            "url": "https://upload.wikimedia.org/icon.svg",
                        }],
                    },
                    "300": {
                        "index": 1,
                        "title": "File:First_photo.png",
                        "imageinfo": [{
                            "mime": "image/png",
                            "url": "https://upload.wikimedia.org/first.png",
                            "width": 900,
                            "height": 600,
                            "extmetadata": {"ObjectName": {"value": "First Object"}},
                        }],
                    },
                }
            }
        }
        cands, meta = parse_wikimedia_response(payload, max_candidates=4)
        self.assertEqual(meta["skipped_non_bitmap"], 1)
        self.assertEqual(len(cands), 2)
        self.assertEqual(cands[0]["url"], "https://upload.wikimedia.org/first.png")
        self.assertEqual(cands[0]["description"], "First Object")
        self.assertEqual(cands[1]["url"], "https://thumb.wikimedia.org/second_1280.jpg")
        self.assertEqual(cands[1]["description"], "Second description")

    def test_summary_and_html_report_escaping(self):
        records = [{
            "id": "q01",
            "category": "Test",
            "query": "test <script>",
            "providers": {
                "openverse": {
                    "meta": {"latency_ms": 120.0, "raw_count": 1, "error": None},
                    "candidates": [{
                        "candidate_id": "cand1",
                        "url": "https://example.com/a.jpg",
                        "description": "<b>desc</b>",
                        "alt_text": "alt text value",
                        "source": "flickr",
                        "domain": "example.com",
                        "license": "CC0",
                        "is_watermarked": False,
                        "download": {"download_ok": True, "decoded_width": 800, "decoded_height": 600, "local_path": None},
                    }],
                }
            },
        }]
        summary = compute_provider_summary(records, ["openverse"], max_candidates=4)
        self.assertEqual(summary["openverse"]["queries_with_hits"], 1)
        self.assertEqual(summary["openverse"]["download_success_pct"], 100.0)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            img_path = tmp_dir / "sample.jpg"
            Image.new("RGB", (64, 48), color=(30, 120, 200)).save(img_path, format="JPEG")
            records[0]["providers"]["openverse"]["candidates"][0]["download"]["local_path"] = str(img_path)

            run_data = {
                "generated_at": "2026-11-11T18:25:00",
                "providers": ["openverse"],
                "max_candidates": 4,
                "skip_download": False,
                "summary": summary,
                "queries": records,
            }
            html_path = render_html_report(run_data, tmp_dir)
            text = html_path.read_text(encoding="utf-8")
            self.assertIn("test &lt;script&gt;", text)
            self.assertNotIn("test <script>", text)
            self.assertIn("data:image/jpeg;base64,", text)


if __name__ == "__main__":
    unittest.main()
