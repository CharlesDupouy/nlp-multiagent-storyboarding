#!/usr/bin/env python3
"""
Benchmark and visual comparison suite for stock/web image APIs:
  - Openverse (api.openverse.org)
  - Serper.dev Google Image Search (google.serper.dev/images)
  - Wikimedia Commons Action API (commons.wikimedia.org)

Evaluates each provider against realistic storyboard image queries on:
  1. Hit rate & candidate availability (up to 4 per query)
  2. API latency
  3. Actual image download + Pillow RGB decode reliability (simulating storyboard_to_video.py)
  4. Text metadata richness (description / alt_text visible to ArtDirectorAgent)
  5. Stock-watermark exposure (raw vs domain-filtered on Serper.dev)

Outputs:
  - outputs/image_api_benchmark/results.json
  - outputs/image_api_benchmark/report.html   (side-by-side visual evaluation sheet)
  - outputs/image_api_benchmark/images/...    (cached downloaded thumbnails/images)
"""

from __future__ import annotations

import argparse
import base64
import html
import io
import json
import os
import re
import statistics
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests
from dotenv import load_dotenv
from PIL import Image


DEFAULT_USER_AGENT = (
    "HKUST-NLP-Storyboarding/1.0 "
    "(https://github.com/CharlesDupouy/nlp-multiagent-storyboarding; "
    "charlesdupouy2006@gmail.com)"
)
LEGACY_MOZILLA_UA = "Mozilla/5.0"

DEFAULT_QUERIES: list[dict[str, str]] = [
    {"id": "q01", "category": "UI / Tech", "query": "hand holding black android smartphone contact list"},
    {"id": "q02", "category": "Abstract / Diagram", "query": "network graph nodes connections diagram white background"},
    {"id": "q03", "category": "Physical Action", "query": "hands folding white paper sheet wooden table"},
    {"id": "q04", "category": "Human Emotion", "query": "couple sitting apart on living room sofa"},
    {"id": "q05", "category": "Household Stock", "query": "kitchen sink full of dirty dishes"},
    {"id": "q06", "category": "Physics Demo", "query": "stack of books balanced edge wooden table"},
    {"id": "q07", "category": "Niche Hobby", "query": "magician hands shuffling playing cards poker mat"},
    {"id": "q08", "category": "Indoor Event", "query": "birthday cake lit candles classroom"},
    {"id": "q09", "category": "Famous Landmark", "query": "Eiffel Tower sunset Paris"},
    {"id": "q10", "category": "Named Science", "query": "James Webb Space Telescope deep field nebula"},
]

STOCK_WATERMARK_DOMAINS: tuple[str, ...] = (
    "shutterstock.com",
    "gettyimages.",
    "alamy.com",
    "istockphoto.com",
    "dreamstime.com",
    "123rf.com",
    "depositphotos.com",
    "vecteezy.com",
    "stock.adobe.com",
    "pond5.com",
    "bigstockphoto.com",
    "canstockphoto.com",
    "dissolve.com",
    "agefotostock.com",
    "superstock.com",
    "featurepics.com",
)

ALLOWED_RASTER_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
ALLOWED_WIKI_MIMES = {"image/jpeg", "image/png", "image/webp"}
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def strip_html(text: str | None) -> str:
    """Remove HTML tags and collapse whitespace from metadata strings."""
    if not text:
        return ""
    cleaned = _TAG_RE.sub(" ", html.unescape(str(text)))
    return _WS_RE.sub(" ", cleaned).strip()


def is_watermarked_domain(url: str, domain: str = "", source: str = "") -> bool:
    """Return True when the URL/domain/source belongs to a known watermarked stock vendor."""
    parsed_host = urlparse(url or "").netloc.lower()
    haystack = " ".join(part.lower() for part in (parsed_host, domain or "", source or "", url or ""))
    return any(marker in haystack for marker in STOCK_WATERMARK_DOMAINS)


def _url_extension(url: str) -> str:
    path = urlparse(url or "").path.lower()
    if "." not in path:
        return ""
    return "." + path.rsplit(".", 1)[-1]


def _is_wikimedia_host(url: str) -> bool:
    host = urlparse(url or "").netloc.lower()
    return host.endswith("wikimedia.org") or host.endswith("wikipedia.org")


# --------------------------------------------------------------------------- #
# Provider adapters
# --------------------------------------------------------------------------- #
def parse_openverse_response(
    payload: dict[str, Any],
    max_candidates: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Normalize Openverse JSON into unified candidate dicts."""
    raw_results = payload.get("results") or []
    candidates: list[dict[str, Any]] = []
    skipped_format = 0

    for item in raw_results:
        url = (item.get("url") or "").strip()
        if not url:
            continue
        filetype = (item.get("filetype") or "").lower().strip()
        ext = _url_extension(url)
        if filetype and filetype not in ("jpg", "jpeg", "png", "webp"):
            skipped_format += 1
            continue
        if ext and ext not in ALLOWED_RASTER_EXTS and filetype not in ("jpg", "jpeg", "png", "webp"):
            skipped_format += 1
            continue

        title = strip_html(item.get("title") or "")
        tags = [
            t["name"].strip()
            for t in (item.get("tags") or [])
            if isinstance(t, dict) and isinstance(t.get("name"), str) and t["name"].strip()
        ]
        tag_str = ", ".join(tags[:12])
        source = (item.get("source") or item.get("provider") or "openverse").strip()
        lic = (item.get("license") or "").upper()
        desc = title or (f"Open-licensed image tagged: {tag_str}" if tag_str else "Openverse image")
        alt_parts = [p for p in (title, f"Tags: {tag_str}" if tag_str else "", f"Source: {source} ({lic})" if lic else f"Source: {source}") if p]

        candidates.append({
            "candidate_id": f"cand{len(candidates) + 1}",
            "url": url,
            "preview_url": item.get("thumbnail") or url,
            "description": desc,
            "alt_text": " | ".join(alt_parts),
            "source": source,
            "domain": urlparse(url).netloc.lower(),
            "license": lic or "CC",
            "width": item.get("width"),
            "height": item.get("height"),
            "is_watermarked": is_watermarked_domain(url, source=source),
            "raw_tag_count": len(tags),
        })
        if len(candidates) >= max_candidates:
            break

    meta = {
        "raw_count": len(raw_results),
        "returned_count": len(candidates),
        "skipped_non_raster": skipped_format,
    }
    return candidates, meta


def fetch_openverse(
    query: str,
    max_candidates: int = 4,
    user_agent: str = DEFAULT_USER_AGENT,
    timeout: int = 20,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Query Openverse anonymous API with polite retry handling."""
    url = "https://api.openverse.org/v1/images/"
    params = {"q": query, "page_size": max(12, max_candidates * 3), "mature": "false"}
    headers = {"User-Agent": user_agent}

    t0 = time.perf_counter()
    for attempt in range(3):
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=timeout)
            if resp.status_code == 429 and attempt < 2:
                wait_s = min(int(resp.headers.get("Retry-After", 4)), 10)
                time.sleep(wait_s)
                continue
            latency_ms = round((time.perf_counter() - t0) * 1000, 1)
            if not resp.ok:
                return [], {
                    "latency_ms": latency_ms,
                    "http_status": resp.status_code,
                    "error": f"HTTP {resp.status_code}: {resp.text[:160]}",
                    "raw_count": 0,
                    "returned_count": 0,
                }
            candidates, meta = parse_openverse_response(resp.json(), max_candidates)
            meta.update({"latency_ms": latency_ms, "http_status": resp.status_code, "error": None})
            return candidates, meta
        except Exception as exc:
            if attempt == 2:
                return [], {
                    "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
                    "http_status": None,
                    "error": f"{type(exc).__name__}: {exc}",
                    "raw_count": 0,
                    "returned_count": 0,
                }
            time.sleep(1.5)
    return [], {"latency_ms": 0.0, "http_status": None, "error": "Unknown error", "raw_count": 0, "returned_count": 0}


def parse_serper_response(
    payload: dict[str, Any],
    max_candidates: int,
    filter_watermarks: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Normalize Serper.dev /images JSON and measure raw vs filtered watermark exposure."""
    raw_images = payload.get("images") or []
    raw_top4_watermarked = 0
    raw_total_watermarked = 0

    for idx, item in enumerate(raw_images):
        u = (item.get("imageUrl") or "").strip()
        wm = is_watermarked_domain(u, item.get("domain", ""), item.get("source", ""))
        if wm:
            raw_total_watermarked += 1
            if idx < max_candidates:
                raw_top4_watermarked += 1

    candidates: list[dict[str, Any]] = []
    for item in raw_images:
        url = (item.get("imageUrl") or "").strip()
        if not url:
            continue
        domain = (item.get("domain") or urlparse(url).netloc).lower()
        source = (item.get("source") or domain).strip()
        wm = is_watermarked_domain(url, domain, source)
        if filter_watermarks and wm:
            continue

        title = strip_html(item.get("title") or "")
        alt_text = f"{title} | Source: {source} ({domain})" if title else f"Web image from {source} ({domain})"
        candidates.append({
            "candidate_id": f"cand{len(candidates) + 1}",
            "url": url,
            "preview_url": item.get("thumbnailUrl") or url,
            "description": title or f"Image result from {source}",
            "alt_text": alt_text,
            "source": source,
            "domain": domain,
            "license": "Web result (check copyright)",
            "width": item.get("imageWidth"),
            "height": item.get("imageHeight"),
            "is_watermarked": wm,
            "position": item.get("position"),
        })
        if len(candidates) >= max_candidates:
            break

    meta = {
        "raw_count": len(raw_images),
        "returned_count": len(candidates),
        "raw_top_k_watermarked": raw_top4_watermarked,
        "raw_total_watermarked": raw_total_watermarked,
        "watermark_filter_enabled": filter_watermarks,
    }
    return candidates, meta


def fetch_serper(
    query: str,
    api_key: str,
    max_candidates: int = 4,
    filter_watermarks: bool = True,
    timeout: int = 20,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Query Serper.dev Google Images endpoint."""
    if not api_key:
        return [], {
            "latency_ms": 0.0,
            "http_status": None,
            "error": "Missing SERPER_API_KEY",
            "raw_count": 0,
            "returned_count": 0,
        }

    url = "https://google.serper.dev/images"
    headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}
    body = {"q": query, "num": 10, "gl": "us", "hl": "en"}

    t0 = time.perf_counter()
    try:
        resp = requests.post(url, headers=headers, json=body, timeout=timeout)
        latency_ms = round((time.perf_counter() - t0) * 1000, 1)
        if not resp.ok:
            return [], {
                "latency_ms": latency_ms,
                "http_status": resp.status_code,
                "error": f"HTTP {resp.status_code}: {resp.text[:160]}",
                "raw_count": 0,
                "returned_count": 0,
            }
        candidates, meta = parse_serper_response(resp.json(), max_candidates, filter_watermarks=filter_watermarks)
        meta.update({"latency_ms": latency_ms, "http_status": resp.status_code, "error": None})
        return candidates, meta
    except Exception as exc:
        return [], {
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
            "http_status": None,
            "error": f"{type(exc).__name__}: {exc}",
            "raw_count": 0,
            "returned_count": 0,
        }


def parse_wikimedia_response(
    payload: dict[str, Any],
    max_candidates: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Normalize Wikimedia Commons Action API response sorted by search relevance index."""
    pages_dict = (payload.get("query") or {}).get("pages") or {}
    pages = sorted(pages_dict.values(), key=lambda p: p.get("index", 999999))
    candidates: list[dict[str, Any]] = []
    skipped_mime = 0

    for page in pages:
        imageinfo = page.get("imageinfo") or []
        if not imageinfo:
            continue
        ii = imageinfo[0]
        mime = (ii.get("mime") or "").lower()
        if mime not in ALLOWED_WIKI_MIMES:
            skipped_mime += 1
            continue

        chosen_url = (ii.get("thumburl") or ii.get("url") or "").strip()
        if not chosen_url:
            continue

        ext = ii.get("extmetadata") or {}
        img_desc = strip_html((ext.get("ImageDescription") or {}).get("value", ""))
        obj_name = strip_html((ext.get("ObjectName") or {}).get("value", ""))
        raw_title = re.sub(r"^File:", "", page.get("title") or "")
        clean_title = raw_title.rsplit(".", 1)[0].replace("_", " ").strip()
        license_short = strip_html((ext.get("LicenseShortName") or {}).get("value", "")) or "Wikimedia"
        artist = strip_html((ext.get("Artist") or {}).get("value", ""))

        description = img_desc or obj_name or clean_title
        alt_parts = [description]
        if clean_title and clean_title.lower() not in description.lower():
            alt_parts.append(f"Title: {clean_title}")
        if license_short:
            alt_parts.append(f"License: {license_short}")
        if artist and len(artist) < 60:
            alt_parts.append(f"Author: {artist}")

        candidates.append({
            "candidate_id": f"cand{len(candidates) + 1}",
            "url": chosen_url,
            "orig_url": ii.get("url"),
            "preview_url": chosen_url,
            "description": description[:400],
            "alt_text": " | ".join(alt_parts)[:550],
            "source": "commons.wikimedia.org",
            "domain": urlparse(chosen_url).netloc.lower(),
            "license": license_short,
            "mime": mime,
            "width": ii.get("thumbwidth") or ii.get("width"),
            "height": ii.get("thumbheight") or ii.get("height"),
            "is_watermarked": False,
        })
        if len(candidates) >= max_candidates:
            break

    meta = {
        "raw_count": len(pages),
        "returned_count": len(candidates),
        "skipped_non_bitmap": skipped_mime,
    }
    return candidates, meta


def fetch_wikimedia(
    query: str,
    max_candidates: int = 4,
    user_agent: str = DEFAULT_USER_AGENT,
    timeout: int = 20,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Query Wikimedia Commons Action API using bitmap filter and polite User-Agent."""
    url = "https://commons.wikimedia.org/w/api.php"
    params = {
        "action": "query",
        "generator": "search",
        "gsrsearch": f"{query} filetype:bitmap -filetype:svg -filetype:tiff",
        "gsrnamespace": 6,
        "gsrlimit": max(12, max_candidates * 3),
        "prop": "imageinfo",
        "iiprop": "url|mime|size|extmetadata",
        "iiurlwidth": 1280,
        "format": "json",
    }
    headers = {"User-Agent": user_agent}

    t0 = time.perf_counter()
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=timeout)
        latency_ms = round((time.perf_counter() - t0) * 1000, 1)
        if not resp.ok:
            return [], {
                "latency_ms": latency_ms,
                "http_status": resp.status_code,
                "error": f"HTTP {resp.status_code}: {resp.text[:160]}",
                "raw_count": 0,
                "returned_count": 0,
            }
        candidates, meta = parse_wikimedia_response(resp.json(), max_candidates)
        meta.update({"latency_ms": latency_ms, "http_status": resp.status_code, "error": None})
        return candidates, meta
    except Exception as exc:
        return [], {
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
            "http_status": None,
            "error": f"{type(exc).__name__}: {exc}",
            "raw_count": 0,
            "returned_count": 0,
        }


# --------------------------------------------------------------------------- #
# Download & PIL verification (mirrors storyboard_to_video.py behavior)
# --------------------------------------------------------------------------- #
def verify_and_cache_image(
    candidate: dict[str, Any],
    save_path: Path,
    project_ua: str = DEFAULT_USER_AGENT,
    timeout: int = 15,
) -> dict[str, Any]:
    """
    Download candidate['url'], validate it decodes cleanly via Pillow RGB,
    and cache a local JPEG copy for the HTML report.

    Wikimedia hosts receive `project_ua` directly (per Wikimedia User-Agent policy).
    Other hosts are tested first with storyboard_to_video.py's current `Mozilla/5.0`
    header and retried with `project_ua` if blocked.
    """
    url = candidate.get("url") or ""
    if not url:
        return {"download_ok": False, "error": "Empty URL", "used_ua": None}

    ua_sequence = (
        [("project_ua", project_ua)]
        if _is_wikimedia_host(url)
        else [("mozilla", LEGACY_MOZILLA_UA), ("project_ua", project_ua)]
    )

    last_error = "Unknown download error"
    last_status = None
    first_ua_error = None

    for ua_label, ua_value in ua_sequence:
        try:
            resp = requests.get(
                url,
                headers={"User-Agent": ua_value},
                timeout=timeout,
                allow_redirects=True,
            )
            last_status = resp.status_code
            if resp.status_code != 200:
                last_error = f"HTTP {resp.status_code}"
                if first_ua_error is None:
                    first_ua_error = f"{ua_label}: HTTP {resp.status_code}"
                continue

            img = Image.open(io.BytesIO(resp.content)).convert("RGB")
            w, h = img.size
            save_path.parent.mkdir(parents=True, exist_ok=True)
            img.save(save_path, format="JPEG", quality=88)

            return {
                "download_ok": True,
                "http_status": resp.status_code,
                "used_ua": ua_label,
                "first_ua_error": first_ua_error,
                "decoded_width": w,
                "decoded_height": h,
                "bytes": len(resp.content),
                "local_path": str(save_path),
                "error": None,
            }
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {str(exc)[:120]}"
            if first_ua_error is None:
                first_ua_error = f"{ua_label}: {last_error}"

    return {
        "download_ok": False,
        "http_status": last_status,
        "used_ua": None,
        "first_ua_error": first_ua_error,
        "decoded_width": None,
        "decoded_height": None,
        "bytes": 0,
        "local_path": None,
        "error": last_error,
    }


# --------------------------------------------------------------------------- #
# Summary stats & HTML visual report
# --------------------------------------------------------------------------- #
def compute_provider_summary(
    query_records: list[dict[str, Any]],
    providers: list[str],
    max_candidates: int,
) -> dict[str, dict[str, Any]]:
    """Aggregate accuracy/reliability/latency/metadata metrics per provider."""
    summary: dict[str, dict[str, Any]] = {}
    total_q = len(query_records)

    for prov in providers:
        latencies: list[float] = []
        hit_queries = 0
        full_queries = 0
        total_cands = 0
        dl_tested = 0
        dl_ok = 0
        mozilla_fail_project_ok = 0
        alt_lengths: list[int] = []
        widths: list[int] = []
        wm_raw_top_k = 0
        wm_raw_total = 0
        raw_total = 0
        errors = 0

        for rec in query_records:
            pdata = rec["providers"].get(prov, {})
            meta = pdata.get("meta") or {}
            cands = pdata.get("candidates") or []
            if meta.get("error"):
                errors += 1
            if isinstance(meta.get("latency_ms"), (int, float)) and meta["latency_ms"] > 0:
                latencies.append(float(meta["latency_ms"]))
            if len(cands) > 0:
                hit_queries += 1
            if len(cands) >= max_candidates:
                full_queries += 1

            raw_total += int(meta.get("raw_count") or 0)
            wm_raw_top_k += int(meta.get("raw_top_k_watermarked") or 0)
            wm_raw_total += int(meta.get("raw_total_watermarked") or 0)

            for c in cands:
                total_cands += 1
                alt_lengths.append(len(c.get("alt_text") or ""))
                dl = c.get("download")
                if dl and not dl.get("skipped"):
                    dl_tested += 1
                    if dl.get("download_ok"):
                        dl_ok += 1
                        if dl.get("first_ua_error") and dl.get("used_ua") == "project_ua":
                            mozilla_fail_project_ok += 1
                        if dl.get("decoded_width"):
                            widths.append(int(dl["decoded_width"]))

        summary[prov] = {
            "total_queries": total_q,
            "queries_with_hits": hit_queries,
            "queries_with_full_candidates": full_queries,
            "api_errors": errors,
            "total_candidates": total_cands,
            "downloads_tested": dl_tested,
            "downloads_ok": dl_ok,
            "download_success_pct": round((dl_ok / dl_tested) * 100, 1) if dl_tested else None,
            "mozilla_blocked_but_project_ua_ok": mozilla_fail_project_ok,
            "mean_latency_ms": round(statistics.mean(latencies), 1) if latencies else None,
            "avg_alt_text_chars": round(statistics.mean(alt_lengths), 1) if alt_lengths else 0.0,
            "median_width_px": int(statistics.median(widths)) if widths else None,
            "raw_results_total": raw_total,
            "raw_top_k_watermarked": wm_raw_top_k,
            "raw_total_watermarked": wm_raw_total,
        }

    return summary


_PROVIDER_LABELS = {
    "openverse": "Openverse",
    "serper": "Serper.dev (Google Images)",
    "wikimedia": "Wikimedia Commons",
}


def _image_data_uri(local_path: str | Path, max_size: tuple[int, int] = (480, 320)) -> str | None:
    """Encode a local verified JPEG into a compact base64 data URI for standalone HTML viewing."""
    path = Path(local_path)
    if not path.is_file():
        return None
    try:
        with Image.open(path) as im:
            thumb = im.convert("RGB")
            thumb.thumbnail(max_size, Image.LANCZOS)
            buf = io.BytesIO()
            thumb.save(buf, format="JPEG", quality=78)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{b64}"
    except Exception:
        return None


def render_html_report(
    run_data: dict[str, Any],
    out_dir: Path,
) -> Path:
    """Render a side-by-side dark HTML visual comparison sheet in out_dir/report.html."""
    providers: list[str] = run_data["providers"]
    summary: dict[str, dict[str, Any]] = run_data["summary"]
    records: list[dict[str, Any]] = run_data["queries"]

    nav_links = " ".join(
        f'<a class="nav-pill" href="#{html.escape(r["id"])}">{html.escape(r["id"].upper())}: {html.escape(r["category"])}</a>'
        for r in records
    )

    summary_rows = []
    for prov in providers:
        s = summary[prov]
        dl_str = (
            f'{s["downloads_ok"]}/{s["downloads_tested"]} ({s["download_success_pct"]}%)'
            if s["downloads_tested"]
            else "skipped"
        )
        wm_str = (
            f'{s["raw_total_watermarked"]}/{s["raw_results_total"]} raw'
            if prov == "serper" and s["raw_results_total"]
            else "0"
        )
        lat_str = f'{s["mean_latency_ms"]} ms' if s["mean_latency_ms"] is not None else "n/a"
        med_w = f'{s["median_width_px"]} px' if s["median_width_px"] else "n/a"
        summary_rows.append(
            f"<tr>"
            f"<td><strong>{html.escape(_PROVIDER_LABELS.get(prov, prov))}</strong></td>"
            f"<td>{s['queries_with_hits']}/{s['total_queries']}</td>"
            f"<td>{s['queries_with_full_candidates']}/{s['total_queries']}</td>"
            f"<td>{dl_str}</td>"
            f"<td>{lat_str}</td>"
            f"<td>{s['avg_alt_text_chars']} chars</td>"
            f"<td>{med_w}</td>"
            f"<td>{wm_str}</td>"
            f"</tr>"
        )

    query_sections = []
    for rec in records:
        qid = rec["id"]
        cols = []
        for prov in providers:
            pdata = rec["providers"].get(prov, {})
            meta = pdata.get("meta") or {}
            cands = pdata.get("candidates") or []

            meta_line_parts = [f"latency: {meta.get('latency_ms', '?')} ms", f"raw: {meta.get('raw_count', 0)}"]
            if prov == "serper":
                meta_line_parts.append(f"raw top-4 watermarked: {meta.get('raw_top_k_watermarked', 0)}")
            if meta.get("error"):
                meta_line_parts.append(f"ERROR: {meta['error']}")

            cards_html = []
            if not cands:
                cards_html.append('<div class="empty-card">No valid bitmap candidates returned.</div>')

            for cand in cands:
                dl = cand.get("download") or {}
                rel_img = None
                if dl.get("local_path"):
                    try:
                        rel_img = str(Path(dl["local_path"]).relative_to(out_dir))
                    except ValueError:
                        rel_img = dl["local_path"]

                if dl.get("skipped"):
                    alt_attr = html.escape((cand.get("description") or cand["candidate_id"])[:80])
                    media_html = f'<a href="{html.escape(cand["url"])}" target="_blank" rel="noopener"><img src="{html.escape(cand.get("preview_url") or cand["url"])}" alt="{alt_attr}" /></a>'
                    status_badge = '<span class="badge warn">Download check skipped</span>'
                elif dl.get("download_ok") and dl.get("local_path"):
                    img_src = _image_data_uri(dl["local_path"]) or rel_img or cand["url"]
                    alt_attr = html.escape((cand.get("description") or cand["candidate_id"])[:80])
                    media_html = f'<a href="{html.escape(cand["url"])}" target="_blank" rel="noopener"><img src="{html.escape(img_src)}" alt="{alt_attr}" /></a>'
                    dims = f'{dl.get("decoded_width")}x{dl.get("decoded_height")}'
                    ua_note = " (needed project UA)" if dl.get("first_ua_error") else ""
                    low_res = dl.get("decoded_width") and dl["decoded_width"] < 500
                    cls = "warn" if low_res else "ok"
                    status_badge = f'<span class="badge {cls}">PIL OK {dims}{html.escape(ua_note)}</span>'
                else:
                    err_txt = dl.get("error") or "Failed"
                    media_html = f'<div class="broken-img">Download / PIL Decode Failed<br/><code>{html.escape(err_txt)}</code></div>'
                    status_badge = f'<span class="badge err">{html.escape(err_txt)}</span>'

                wm_badge = ' <span class="badge warn">Stock Watermark Domain</span>' if cand.get("is_watermarked") else ""
                domain_badge = f'<span class="badge neutral">{html.escape(cand.get("domain") or cand.get("source") or "")}</span>'
                lic_badge = f'<span class="badge neutral">{html.escape(cand.get("license") or "")}</span>'

                cards_html.append(
                    f'<div class="cand-card">'
                    f'  <div class="cand-media">{media_html}</div>'
                    f'  <div class="cand-body">'
                    f'    <div class="badges">{status_badge}{wm_badge} {domain_badge} {lic_badge}</div>'
                    f'    <div class="cand-title"><strong>{html.escape(cand["candidate_id"])}</strong> — {html.escape(cand.get("description") or "")}</div>'
                    f'    <details><summary>ArtDirector alt_text ({len(cand.get("alt_text") or "")} chars)</summary>'
                    f'      <p class="alt-raw">{html.escape(cand.get("alt_text") or "")}</p>'
                    f'      <p class="url-raw"><a href="{html.escape(cand["url"])}" target="_blank" rel="noopener">{html.escape(cand["url"])}</a></p>'
                    f'    </details>'
                    f'  </div>'
                    f'</div>'
                )

            cols.append(
                f'<div class="prov-col">'
                f'  <div class="prov-head">'
                f'    <h3>{html.escape(_PROVIDER_LABELS.get(prov, prov))}</h3>'
                f'    <div class="prov-meta">{html.escape(" | ".join(meta_line_parts))}</div>'
                f'  </div>'
                f'  <div class="cand-grid">{"".join(cards_html)}</div>'
                f'</div>'
            )

        query_sections.append(
            f'<section class="query-block" id="{html.escape(qid)}">'
            f'  <div class="query-head">'
            f'    <span class="q-cat">{html.escape(rec["category"])}</span>'
            f'    <h2>{html.escape(qid.upper())}: <code>{html.escape(rec["query"])}</code></h2>'
            f'  </div>'
            f'  <div class="prov-row">{"".join(cols)}</div>'
            f'</section>'
        )

    doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>Image API Benchmark — Openverse vs Serper vs Wikimedia</title>
<style>
  :root {{ --bg:#0f121a; --card:#171c28; --border:#273043; --text:#e8ecf4; --muted:#9aa6be; --ok:#1e7e4f; --warn:#996515; --err:#9b2c2c; --accent:#5b8cff; }}
  * {{ box-sizing: border-box; }}
  body {{ margin:0; padding:24px; background:var(--bg); color:var(--text); font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif; line-height:1.45; }}
  h1,h2,h3 {{ margin:0 0 8px 0; }}
  code {{ background:#1f2636; padding:2px 6px; border-radius:4px; color:#b8d0ff; font-size:0.9em; }}
  a {{ color:var(--accent); text-decoration:none; }}
  a:hover {{ text-decoration:underline; }}
  .header {{ margin-bottom:20px; }}
  .subtitle {{ color:var(--muted); margin-bottom:14px; font-size:0.95rem; }}
  table {{ width:100%; border-collapse:collapse; background:var(--card); border:1px solid var(--border); border-radius:8px; overflow:hidden; margin-bottom:18px; }}
  th,td {{ padding:10px 12px; border-bottom:1px solid var(--border); text-align:left; font-size:0.9rem; }}
  th {{ background:#1d2433; color:#c8d3ea; font-weight:600; }}
  .nav {{ display:flex; flex-wrap:wrap; gap:8px; margin-bottom:24px; }}
  .nav-pill {{ background:var(--card); border:1px solid var(--border); padding:5px 10px; border-radius:999px; font-size:0.8rem; color:var(--text); }}
  .query-block {{ border:1px solid var(--border); border-radius:10px; padding:16px; margin-bottom:26px; background:#131722; }}
  .query-head {{ display:flex; align-items:center; gap:10px; margin-bottom:14px; flex-wrap:wrap; }}
  .q-cat {{ background:#222c42; color:#9fc0ff; padding:3px 9px; border-radius:6px; font-size:0.78rem; font-weight:600; text-transform:uppercase; }}
  .prov-row {{ display:grid; grid-template-columns:repeat({max(1, len(providers))}, minmax(0, 1fr)); gap:14px; }}
  @media (max-width: 1100px) {{ .prov-row {{ grid-template-columns:1fr; }} }}
  .prov-col {{ background:var(--card); border:1px solid var(--border); border-radius:8px; padding:12px; }}
  .prov-head {{ border-bottom:1px solid var(--border); padding-bottom:8px; margin-bottom:10px; }}
  .prov-meta {{ color:var(--muted); font-size:0.8rem; }}
  .cand-grid {{ display:grid; grid-template-columns:1fr; gap:10px; }}
  .cand-card {{ border:1px solid var(--border); border-radius:8px; overflow:hidden; background:#121620; }}
  .cand-media {{ height:185px; background:#090b10; display:flex; align-items:center; justify-content:center; overflow:hidden; }}
  .cand-media a {{ display:flex; align-items:center; justify-content:center; width:100%; height:100%; }}
  .cand-media img {{ width:100%; height:100%; max-width:100%; max-height:185px; object-fit:contain; display:block; }}
  .broken-img {{ color:#fca5a5; padding:14px; text-align:center; font-size:0.82rem; }}
  .empty-card {{ color:var(--muted); padding:20px; text-align:center; border:1px dashed var(--border); border-radius:8px; }}
  .cand-body {{ padding:9px 10px; }}
  .badges {{ display:flex; flex-wrap:wrap; gap:5px; margin-bottom:6px; }}
  .badge {{ font-size:0.72rem; padding:2px 6px; border-radius:4px; background:#232b3d; color:#d2dcf0; }}
  .badge.ok {{ background:var(--ok); color:#fff; }}
  .badge.warn {{ background:var(--warn); color:#fff; }}
  .badge.err {{ background:var(--err); color:#fff; }}
  .cand-title {{ font-size:0.84rem; margin-bottom:5px; }}
  details summary {{ cursor:pointer; color:var(--muted); font-size:0.78rem; }}
  .alt-raw, .url-raw {{ font-size:0.76rem; color:var(--muted); margin:4px 0; word-break:break-word; }}
</style>
</head>
<body>
  <div class="header">
    <h1>Image Search API Visual Benchmark</h1>
    <div class="subtitle">Generated {html.escape(run_data["generated_at"])} · {len(records)} queries · top {run_data["max_candidates"]} candidates per API · real PIL download verification enabled={not run_data["skip_download"]}</div>
    <table>
      <thead>
        <tr>
          <th>Provider</th>
          <th>Queries w/ ≥1 hit</th>
          <th>Queries w/ full {run_data["max_candidates"]}</th>
          <th>Download + PIL OK</th>
          <th>Mean Latency</th>
          <th>Avg Text Metadata</th>
          <th>Median Width</th>
          <th>Watermark Stock Hits</th>
        </tr>
      </thead>
      <tbody>{"".join(summary_rows)}</tbody>
    </table>
    <div class="nav">{nav_links}</div>
  </div>
  {"".join(query_sections)}
</body>
</html>
"""
    report_path = out_dir / "report.html"
    report_path.write_text(doc, encoding="utf-8")
    return report_path


# --------------------------------------------------------------------------- #
# CLI entry point
# --------------------------------------------------------------------------- #
def run_benchmark(
    queries: list[dict[str, str]],
    providers: list[str],
    out_dir: Path,
    max_candidates: int = 4,
    skip_download: bool = False,
    filter_watermarks: bool = True,
    user_agent: str = DEFAULT_USER_AGENT,
) -> dict[str, Any]:
    """Execute the benchmark suite across the requested providers and write JSON + HTML."""
    load_dotenv()
    serper_key = os.getenv("SERPER_API_KEY", "").strip()
    out_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    for idx, qitem in enumerate(queries, start=1):
        qid = qitem["id"]
        query = qitem["query"]
        print(f"[{idx}/{len(queries)}] {qid}: {query}")
        prov_results: dict[str, Any] = {}

        for prov in providers:
            if prov == "openverse":
                cands, meta = fetch_openverse(query, max_candidates=max_candidates, user_agent=user_agent)
                time.sleep(1.5)  # Respect anonymous 20 req/min burst ceiling politely
            elif prov == "serper":
                cands, meta = fetch_serper(
                    query,
                    api_key=serper_key,
                    max_candidates=max_candidates,
                    filter_watermarks=filter_watermarks,
                )
                time.sleep(0.1)
            elif prov == "wikimedia":
                cands, meta = fetch_wikimedia(query, max_candidates=max_candidates, user_agent=user_agent)
                time.sleep(0.3)
            else:
                raise ValueError(f"Unsupported provider: {prov}")

            dl_ok_count = 0
            for c_idx, cand in enumerate(cands, start=1):
                if skip_download:
                    cand["download"] = {"skipped": True}
                else:
                    save_path = out_dir / "images" / qid / f"{prov}_{cand['candidate_id']}.jpg"
                    dl = verify_and_cache_image(cand, save_path, project_ua=user_agent)
                    cand["download"] = dl
                    if dl.get("download_ok"):
                        dl_ok_count += 1

            status_str = f"hits={len(cands)}/{max_candidates}, raw={meta.get('raw_count', 0)}"
            if not skip_download:
                status_str += f", dl_ok={dl_ok_count}/{len(cands)}"
            if meta.get("error"):
                status_str += f", err={meta['error']}"
            print(f"    - {prov:<10} {status_str} ({meta.get('latency_ms', 0)} ms)")
            prov_results[prov] = {"candidates": cands, "meta": meta}

        records.append({
            "id": qid,
            "category": qitem["category"],
            "query": query,
            "providers": prov_results,
        })

    summary = compute_provider_summary(records, providers, max_candidates=max_candidates)
    run_data = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "providers": providers,
        "max_candidates": max_candidates,
        "skip_download": skip_download,
        "filter_watermarks": filter_watermarks,
        "summary": summary,
        "queries": records,
    }
    json_path = out_dir / "results.json"
    json_path.write_text(json.dumps(run_data, indent=2, ensure_ascii=False), encoding="utf-8")
    report_path = render_html_report(run_data, out_dir)
    print(f"\nSaved JSON : {json_path}")
    print(f"Saved HTML : {report_path}")
    return run_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Openverse, Serper.dev, and Wikimedia Commons image APIs.")
    parser.add_argument(
        "--providers",
        default="openverse,serper,wikimedia",
        help="Comma-separated providers (openverse,serper,wikimedia)",
    )
    parser.add_argument("--max-candidates", type=int, default=4, help="Candidates per query per provider (default: 4)")
    parser.add_argument(
        "--out-dir",
        default="outputs/image_api_benchmark",
        help="Output folder for JSON, HTML report, and cached images",
    )
    parser.add_argument("--skip-download", action="store_true", help="Skip live image download + Pillow check")
    parser.add_argument(
        "--no-watermark-filter",
        action="store_true",
        help="Keep watermarked stock domains in Serper.dev top candidates",
    )
    parser.add_argument(
        "--user-agent",
        default=os.getenv("WIKIMEDIA_USER_AGENT") or DEFAULT_USER_AGENT,
        help="Identifying User-Agent string for Wikimedia & Openverse",
    )
    parser.add_argument("--queries", nargs="*", help="Optional custom queries to override the default 10-query suite")
    parser.add_argument(
        "--from-json",
        help="Re-render HTML report offline from an existing results.json file without calling APIs",
    )
    args = parser.parse_args()

    if args.from_json:
        json_path = Path(args.from_json)
        run_data = json.loads(json_path.read_text(encoding="utf-8"))
        report_path = render_html_report(run_data, json_path.parent)
        print(f"Re-rendered HTML : {report_path}")
        return

    providers = [p.strip().lower() for p in args.providers.split(",") if p.strip()]
    queries = (
        [{"id": f"q{i:02d}", "category": "Custom", "query": q} for i, q in enumerate(args.queries, start=1)]
        if args.queries
        else DEFAULT_QUERIES
    )
    run_benchmark(
        queries=queries,
        providers=providers,
        out_dir=Path(args.out_dir),
        max_candidates=args.max_candidates,
        skip_download=args.skip_download,
        filter_watermarks=not args.no_watermark_filter,
        user_agent=args.user_agent,
    )


if __name__ == "__main__":
    main()
