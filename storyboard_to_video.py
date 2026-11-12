#!/usr/bin/env python3
"""
storyboard_to_video.py
Turn a storyboard JSON file into a slideshow video sketch with the AI visual
description displayed below each scene image and neural Edge-TTS voiceover.

Usage:
    python storyboard_to_video.py storyboard.json -o out.mp4
    python storyboard_to_video.py storyboard.json --lang fr --voice fr-FR-HenriNeural

Requirements:
    - ffmpeg on PATH (or pip install imageio-ffmpeg)
    - pip install edge-tts pillow
"""

import argparse
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

from PIL import Image, ImageDraw, ImageFont

FPS = 25
SAMPLE_RATE = 44100
DEFAULT_PADDING = 0.3
DEFAULT_INTRO_PADDING = 1.5


# --------------------------------------------------------------------------- #
# Storyboard loading / audio-cue cleaning
# --------------------------------------------------------------------------- #
def load_scenes(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    sb = data.get("storyboard", data)
    scenes = sb["scenes"] if isinstance(sb, dict) else sb
    return sb.get("title", "storyboard") if isinstance(sb, dict) else "storyboard", scenes


FRENCH_STOPWORDS = {
    "le", "la", "les", "un", "une", "des", "du", "de", "d'", "l'", "et", "est", "dans",
    "pour", "avec", "qui", "que", "ce", "cette", "ces", "c'est", "pas", "sur", "au",
    "aux", "en", "nous", "vous", "ils", "elles", "mais", "ou", "donc", "or", "ni", "car",
    "tu", "il", "elle", "on", "mon", "ton", "son", "mes", "tes", "ses", "faire", "plus"
}

ENGLISH_STOPWORDS = {
    "the", "a", "an", "and", "is", "are", "was", "were", "this", "that", "these", "those",
    "with", "for", "to", "in", "on", "at", "by", "from", "of", "you", "your", "it", "its",
    "they", "them", "their", "we", "us", "our", "he", "him", "his", "she", "her", "here",
    "there", "what", "which", "who", "when", "where", "why", "how", "all", "any", "not", "so"
}


def detect_language(scenes):
    """Auto-detect primary language of scenes (English vs French)."""
    all_text = " ".join(
        (str(s.get("voiceover") or s.get("audio_cue", "")) + " " + s.get("subtitle", "") + " " + s.get("visual_description", "") + " " + s.get("description", ""))
        for s in scenes
    ).lower()
    words = set(re.findall(r"\b[a-zA-Z\u00C0-\u017F']+\b", all_text))
    fr_count = len(words.intersection(FRENCH_STOPWORDS))
    en_count = len(words.intersection(ENGLISH_STOPWORDS))
    return "fr" if fr_count > en_count else "en"


def clean_audio_cue(cue, raw=False):
    """
    Cleans audio cues or voiceover into pure spoken script dialogue via clean_voiceover(),
    stripping SFX, narrator prefixes, and mid-string speaker/stage cues.
    """
    cue = (cue or "").strip()
    if raw or not cue:
        return cue
    try:
        from src.core.state import clean_voiceover
    except ImportError:
        repo_root = str(Path(__file__).resolve().parent)
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        from src.core.state import clean_voiceover
    return clean_voiceover(cue) or cue


# --------------------------------------------------------------------------- #
# Images and frame rendering
# --------------------------------------------------------------------------- #
_PROJECT_USER_AGENT = (
    "HKUST-NLP-Storyboarding/1.0 "
    "(https://github.com/CharlesDupouy/nlp-multiagent-storyboarding)"
)


def download_image(url, timeout=15):
    try:
        if not url.lower().startswith(("http://", "https://")):   # local file path
            return Image.open(url.replace("file://", "")).convert("RGB")
    except Exception as e:
        print(f"  ! could not fetch image ({type(e).__name__}: {e}); using placeholder")
        return None

    project_ua = os.getenv("WIKIMEDIA_USER_AGENT") or _PROJECT_USER_AGENT
    host = urlparse(url).netloc.lower()
    ua_candidates = (
        [project_ua]
        if host.endswith(("wikimedia.org", "wikipedia.org"))
        else ["Mozilla/5.0", project_ua]
    )

    last_exc = None
    for ua in ua_candidates:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": ua})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return Image.open(io.BytesIO(r.read())).convert("RGB")
        except Exception as e:
            last_exc = e
    print(f"  ! could not fetch image ({type(last_exc).__name__}: {last_exc}); using placeholder")
    return None


def get_font(size, bold=True):
    if bold:
        candidates = (
            "DejaVuSans-Bold.ttf", "Arial Bold.ttf", "arialbd.ttf", "Arial.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        )
    else:
        candidates = (
            "DejaVuSans.ttf", "Arial.ttf", "arial.ttf", "DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/System/Library/Fonts/Supplemental/Arial.ttf",
        )
    for name in candidates:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size)


def wrap_text(draw, text, font, max_width):
    lines, line = [], ""
    for word in text.split():
        trial = f"{line} {word}".strip()
        if draw.textlength(trial, font=font) <= max_width or not line:
            line = trial
        else:
            lines.append(line)
            line = word
    if line:
        lines.append(line)
    return lines


def get_visual_description(scene):
    """Return the scene's AI visual description text."""
    return (scene.get("description") or scene.get("visual_description") or "").strip()


def make_placeholder(size, scene, lang="fr"):
    """Used inside the upper image region when image_url is missing or unreachable."""
    w, h = size
    img = Image.new("RGB", size, (24, 28, 42))
    d = ImageDraw.Draw(img)
    sid = scene.get("scene_id", "?")
    is_fr = str(lang).lower().startswith("fr")
    title = f"Scène {sid}" if is_fr else f"Scene {sid}"
    subtitle = "Image indisponible" if is_fr else "Image unavailable"
    title_font = get_font(max(20, h // 10), bold=True)
    sub_font = get_font(max(14, h // 18), bold=False)
    d.rectangle((0, 0, w - 1, h - 1), outline=(45, 53, 72), width=2)
    d.text((w // 2, int(h * 0.44)), title, font=title_font, fill=(230, 234, 244), anchor="mm")
    d.text((w // 2, int(h * 0.58)), subtitle, font=sub_font, fill=(148, 158, 180), anchor="mm")
    return img


def fit_image(img, size, bg=(11, 13, 19)):
    """Scale to fit inside size (no cropping) centered on a background color."""
    w, h = size
    img = img.copy()
    img.thumbnail((w, h), Image.LANCZOS)
    canvas = Image.new("RGB", size, bg)
    canvas.paste(img, ((w - img.width) // 2, (h - img.height) // 2))
    return canvas


def render_scene_frame(img, scene, size, lang="fr"):
    """
    Compose a scene frame with the scene image on top and the AI visual
    description cleanly displayed in a dedicated panel underneath it.
    """
    w, h = size
    is_fr = str(lang).lower().startswith("fr")
    sid = scene.get("scene_id", "?")
    desc = get_visual_description(scene)

    panel_h = max(150, int(h * 0.28))
    img_area_h = h - panel_h
    inner_pad = max(10, min(w, img_area_h) // 36)
    box_w = max(64, w - 2 * inner_pad)
    box_h = max(64, img_area_h - 2 * inner_pad)

    canvas = Image.new("RGB", size, (11, 13, 19))

    if img is not None:
        fitted = img.copy()
        fitted.thumbnail((box_w, box_h), Image.LANCZOS)
        px = (w - fitted.width) // 2
        py = (img_area_h - fitted.height) // 2
        canvas.paste(fitted, (px, py))
        d_top = ImageDraw.Draw(canvas)
        d_top.rectangle(
            (px - 1, py - 1, px + fitted.width, py + fitted.height),
            outline=(36, 41, 56),
            width=1,
        )
    else:
        placeholder = make_placeholder((box_w, box_h), scene, lang=lang)
        canvas.paste(placeholder, (inner_pad, inner_pad))

    # Dedicated visual-description panel beneath the image area
    d = ImageDraw.Draw(canvas)
    d.rectangle((0, img_area_h, w, h), fill=(20, 24, 36))
    d.line((0, img_area_h, w, img_area_h), fill=(45, 53, 72), width=2)

    pad_x = max(28, int(w * 0.04))
    pad_top = max(12, int(panel_h * 0.08))
    pad_bot = max(10, int(panel_h * 0.07))
    text_w = max(120, w - 2 * pad_x)

    hdr_label = (
        f"SCÈNE {sid}  •  DESCRIPTION VISUELLE"
        if is_fr
        else f"SCENE {sid}  •  VISUAL DESCRIPTION"
    )
    hdr_size = max(13, min(22, int(panel_h * 0.085)))
    hdr_font = get_font(hdr_size, bold=True)
    hdr_h = int(hdr_size * 1.4)
    d.text((pad_x, img_area_h + pad_top), hdr_label, font=hdr_font, fill=(122, 162, 247))

    body_text = desc or ("Aucune description visuelle." if is_fr else "No visual description provided.")
    body_y0 = img_area_h + pad_top + hdr_h + max(4, int(panel_h * 0.03))
    avail_body_h = max(24, h - pad_bot - body_y0)

    max_font_sz = max(18, min(30, int(panel_h * 0.135)))
    min_font_sz = 12

    chosen_font = get_font(min_font_sz, bold=False)
    chosen_lines = wrap_text(d, body_text, chosen_font, text_w)
    chosen_line_h = int(min_font_sz * 1.32)

    for font_sz in range(max_font_sz, min_font_sz - 1, -1):
        cand_font = get_font(font_sz, bold=False)
        cand_lines = wrap_text(d, body_text, cand_font, text_w)
        cand_line_h = int(font_sz * 1.32)
        if len(cand_lines) * cand_line_h <= avail_body_h:
            chosen_font = cand_font
            chosen_lines = cand_lines
            chosen_line_h = cand_line_h
            break

    max_lines = max(1, avail_body_h // max(1, chosen_line_h))
    if len(chosen_lines) > max_lines:
        chosen_lines = chosen_lines[:max_lines]
        if chosen_lines[-1] and not chosen_lines[-1].endswith("…"):
            chosen_lines[-1] = chosen_lines[-1].rstrip(" .") + "…"

    for i, line in enumerate(chosen_lines):
        d.text(
            (pad_x, body_y0 + i * chosen_line_h),
            line,
            font=chosen_font,
            fill=(235, 239, 248),
        )

    return canvas


# --------------------------------------------------------------------------- #
# Text-to-speech (robotic, offline)
# --------------------------------------------------------------------------- #
def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(map(str, cmd))}\n{r.stderr[-1500:]}")
    return r


def make_silence(path, seconds):
    run(["ffmpeg", "-y", "-f", "lavfi", "-i", f"anullsrc=r={SAMPLE_RATE}:cl=stereo",
         "-t", str(seconds), str(path)])


def audio_duration(path):
    if shutil.which("ffprobe"):
        try:
            r = run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                     "-of", "default=nw=1:nokey=1", str(path)])
            return float(r.stdout.strip())
        except Exception:
            pass

    # Fallback directly via ffmpeg without ffprobe
    r = subprocess.run(["ffmpeg", "-i", str(path), "-f", "null", "-"], capture_output=True, text=True)
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", r.stderr)
    if m:
        h, mn, s = m.groups()
        return int(h) * 3600 + int(mn) * 60 + float(s)

    if str(path).lower().endswith(".wav"):
        try:
            import wave
            with wave.open(str(path), "rb") as wf:
                return wf.getnframes() / float(wf.getframerate())
        except Exception:
            pass

    raise RuntimeError(f"Could not determine audio duration for: {path}")


# Default neural Edge-TTS voices by language
DEFAULT_EDGE_VOICES = {
    "fr": "fr-FR-HenriNeural",           # Rich, expressive French male voice
    "en": "en-US-ChristopherNeural",     # Engaging, conversational English voice
    "es": "es-ES-AlvaroNeural",          # Spanish
    "de": "de-DE-ConradNeural",          # German
    "it": "it-IT-DiegoNeural",           # Italian
}


# Run in a separate process for every utterance: pyttsx3's SAPI5 engine on Windows
# can hang forever after a few save_to_file()/runAndWait() calls on one engine.
_PYTTSX3_WORKER = r"""
import sys, pyttsx3
text, out, rate, lang = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4].lower()
e = pyttsx3.init()
e.setProperty("rate", rate)
for v in e.getProperty("voices"):
    langs = [str(l).lower() for l in (getattr(v, "languages", None) or [])]
    blob = " ".join(langs) + " " + (v.id or "").lower() + " " + (v.name or "").lower()
    if lang in blob or ("french" in blob and lang == "fr"):
        e.setProperty("voice", v.id)
        break
e.save_to_file(text, out)
e.runAndWait()
e.stop()
"""

_POWERSHELL_TTS = (
    "Add-Type -AssemblyName System.Speech; "
    "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
    "try { $s.SelectVoiceByHints('NotSet','NotSet',0,[Globalization.CultureInfo]$env:TTS_LANG) } catch {}; "
    "$s.SetOutputToWaveFile($env:TTS_OUT); $s.Speak($env:TTS_TEXT); $s.Dispose()"
)

_LOCALES = {"fr": "fr-FR", "en": "en-US", "es": "es-ES", "de": "de-DE", "it": "it-IT"}


class TTS:
    def __init__(self, lang="fr", rate=170, backend="auto", voice=None, timeout=40):
        self.lang, self.rate, self.voice, self.timeout = lang, rate, voice, timeout
        self.exe = shutil.which("espeak-ng") or shutil.which("espeak")
        self.powershell = shutil.which("powershell") if sys.platform == "win32" else None

        if backend == "auto":
            try:
                import edge_tts  # noqa: F401
                backend = "edge"
            except ImportError:
                try:
                    import pyttsx3  # noqa: F401
                    backend = "pyttsx3"
                except ImportError:
                    backend = "espeak" if self.exe else ("powershell" if self.powershell else None)

        if backend in ("edge", "edge-tts", "edgetts"):
            backend = "edge"
        elif backend == "pyttsx3":
            backend = "pyttsx3"
        elif backend == "espeak":
            backend = "espeak"
        elif backend == "powershell":
            backend = "powershell"
        elif backend == "silent":
            backend = "silent"
        else:
            raise RuntimeError(f"Unknown or unavailable TTS backend '{backend}'")

        self.backend = backend

    def _edge(self, text, raw):
        import asyncio
        import edge_tts

        voice = self.voice or DEFAULT_EDGE_VOICES.get(self.lang[:2].lower(), "fr-FR-HenriNeural")
        # Translate speaking rate (words/min, baseline ~160) to edge-tts percent format e.g. "+5%"
        pct = round((self.rate - 160) / 160 * 100)
        rate_str = f"{pct:+d}%"

        async def _synth():
            comm = edge_tts.Communicate(text, voice=voice, rate=rate_str)
            await comm.save(str(raw))

        asyncio.run(_synth())

    def _pyttsx3(self, text, raw):
        subprocess.run([sys.executable, "-c", _PYTTSX3_WORKER, text, str(raw), str(self.rate), self.lang],
                       check=True, capture_output=True, timeout=self.timeout)

    def _espeak(self, text, raw):
        run([self.exe, "-v", self.lang, "-s", str(self.rate), "-w", str(raw), text])

    def _powershell(self, text, raw):
        import os
        env = dict(os.environ, TTS_TEXT=text, TTS_OUT=str(raw),
                   TTS_LANG=_LOCALES.get(self.lang.lower(), self.lang))
        subprocess.run([self.powershell, "-NoProfile", "-Command", _POWERSHELL_TTS],
                       check=True, capture_output=True, env=env, timeout=self.timeout)

    def synth(self, text, out_wav):
        """Write speech for `text` to out_wav (WAV, 44.1 kHz stereo)."""
        if not text.strip() or self.backend == "silent":
            make_silence(out_wav, 0.1)
            return

        raw = Path(out_wav).with_suffix(".raw.audio")
        attempts = [getattr(self, "_" + self.backend)]

        # Graceful fallbacks if primary backend encounters network/local error
        if self.backend == "edge":
            try:
                import pyttsx3  # noqa: F401
                attempts.append(self._pyttsx3)
            except ImportError:
                if self.exe:
                    attempts.append(self._espeak)
                elif self.powershell:
                    attempts.append(self._powershell)
        elif self.powershell and self.backend != "powershell":
            attempts.append(self._powershell)       # Windows fallback

        last_err = None
        for fn in attempts:
            raw.unlink(missing_ok=True)
            try:
                fn(text, raw)
                if raw.exists() and raw.stat().st_size > 0:
                    break
                last_err = "engine produced no audio"
            except subprocess.TimeoutExpired:
                last_err = f"{fn.__name__} timed out after {self.timeout}s"
            except Exception as e:
                last_err = f"{fn.__name__} failed: {e}"
            print(f"  ! {last_err}; trying fallback" if fn is not attempts[-1] else "")
        else:
            raise RuntimeError(last_err)

        run(["ffmpeg", "-y", "-i", str(raw), "-ar", str(SAMPLE_RATE), "-ac", "2", str(out_wav)])
        raw.unlink(missing_ok=True)


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #
def build_scene_clip(scene, idx, tmp, tts, size, args, lang="fr"):
    sid = scene.get("scene_id", idx + 1)
    desc = get_visual_description(scene)
    preview = (desc[:80] + "...") if len(desc) > 80 else desc
    print(f"Scene {sid}: {preview!r}")

    # Image + visual description panel underneath
    img = download_image(scene["image_url"]) if scene.get("image_url") else None
    frame = tmp / f"frame_{idx:03d}.png"
    render_scene_frame(img, scene, size, lang=lang).save(frame)

    # Speech
    cue = scene.get("voiceover") or scene.get("audio_cue", "")
    speech = clean_audio_cue(cue, raw=args.raw_audio)
    print(f"  speech: {speech!r}")
    wav = tmp / f"speech_{idx:03d}.wav"
    try:
        tts.synth(speech, wav)
    except Exception as e:
        print(f"  ! TTS failed ({e}); scene will be silent")
        make_silence(wav, 0.1)

    pad = args.padding
    intro_pad = max(pad, args.intro_padding) if idx == 0 else pad
    json_duration = float(scene.get("duration_seconds", 5))
    if args.json_durations or not speech.strip():
        # Old behaviour (or nothing to say): trust the storyboard's duration.
        duration, lead = json_duration, (intro_pad if speech.strip() else 0.0)
    else:
        speech_len = audio_duration(wav)
        lead = intro_pad
        duration = lead + speech_len + pad
        print(f"  speech {speech_len:.1f}s -> scene {duration:.1f}s (lead {lead:.1f}s, json said {json_duration:.0f}s)")

    # Write intermediate clips in MKV with uncompressed PCM audio so multi-clip
    # concatenation has zero per-clip AAC priming drift or MP4 edit-list offsets.
    clip = tmp / f"clip_{idx:03d}.mkv"
    delay_ms = int(round(lead * 1000))
    run(["ffmpeg", "-y",
         "-loop", "1", "-framerate", str(FPS), "-i", str(frame),
         "-i", str(wav),
         # lead-in silence, then speech, then silence up to the scene length
         "-af", f"adelay={delay_ms}|{delay_ms},apad=whole_dur={duration}",
         "-t", f"{duration}",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", str(FPS),
         "-c:a", "pcm_s16le", "-ar", str(SAMPLE_RATE), "-ac", "2",
         str(clip)])
    return clip


def main():
    ap = argparse.ArgumentParser(description="Storyboard JSON -> slideshow video with visual description panel & TTS")
    ap.add_argument("storyboard", help="path to the storyboard JSON")
    ap.add_argument("-o", "--output", default=None,
                    help="Output MP4 path. If omitted, automatically placed alongside the storyboard.")
    ap.add_argument("--lang", default="auto", help="TTS language code ('auto', 'en', 'fr', 'es', 'de', 'it'; default: auto)")
    ap.add_argument("--voice", default=None,
                    help="Specific TTS voice (e.g. 'fr-FR-HenriNeural', 'en-US-ChristopherNeural'). Auto-selected if omitted.")
    ap.add_argument("--rate", type=int, default=170, help="TTS speaking rate (words/min)")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720, help="use 1920 x 1080 for vertical shorts, etc.")
    ap.add_argument("--padding", type=float, default=DEFAULT_PADDING,
                    help=f"seconds of silence between/after scenes (default: {DEFAULT_PADDING})")
    ap.add_argument("--intro-padding", type=float, default=DEFAULT_INTRO_PADDING,
                    help=f"initial lead-in silence in seconds before Scene 1 speech so players/devices never cut opening words (default: {DEFAULT_INTRO_PADDING})")
    ap.add_argument("--json-durations", action="store_true",
                    help="use duration_seconds from the storyboard instead of the speech length")
    ap.add_argument("--raw-audio", action="store_true",
                    help="read audio_cue verbatim (including 'VOIX :' / 'SFX :' text)")
    ap.add_argument("--tts-backend", choices=["auto", "edge", "pyttsx3", "espeak", "powershell", "silent"], default="auto")
    ap.add_argument("--keep-temp", action="store_true")
    args = ap.parse_args()

    # Ensure ffmpeg binary is available (check PATH, then imageio-ffmpeg)
    if not shutil.which("ffmpeg"):
        try:
            import imageio_ffmpeg
            ffmpeg_dir = str(Path(imageio_ffmpeg.get_ffmpeg_exe()).parent)
            import os
            os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")
        except Exception:
            pass

    if not shutil.which("ffmpeg"):
        sys.exit("ffmpeg not found on PATH. Install it via apt/brew/conda or run: pip install imageio-ffmpeg")

    title, scenes = load_scenes(args.storyboard)
    print(f"Building '{title}' ({len(scenes)} scenes)")

    lang = args.lang
    if lang == "auto":
        lang = detect_language(scenes)
        print(f"Detected storyboard language: '{lang}'")

    size = (args.width, args.height)
    tts = TTS(lang, args.rate, args.tts_backend, args.voice)

    if args.output:
        out_path = Path(args.output)
    else:
        sb_path = Path(args.storyboard)
        stem = sb_path.stem
        if "storyboard" in stem:
            video_name = stem.replace("storyboard", "video") + ".mp4"
        else:
            video_name = f"video_{stem}.mp4"
        out_path = sb_path.parent / video_name

    out_path.parent.mkdir(parents=True, exist_ok=True)

    tmp = Path(tempfile.mkdtemp(prefix="storyboard_"))
    try:
        clips = [build_scene_clip(s, i, tmp, tts, size, args, lang=lang) for i, s in enumerate(scenes)]

        listing = tmp / "clips.txt"
        listing.write_text("".join(f"file '{c.as_posix()}'\n" for c in clips), encoding="utf-8")
        # Encode AAC once across the full concatenated PCM timeline with faststart for clean t=0 playback.
        run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(listing),
             "-c:v", "copy", "-c:a", "aac", "-b:a", "128k",
             "-movflags", "+faststart", str(out_path)])
        print(f"\nDone -> {out_path} ({audio_duration(out_path):.1f}s)")
    finally:
        if args.keep_temp:
            print(f"Temp files kept in {tmp}")
        else:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
