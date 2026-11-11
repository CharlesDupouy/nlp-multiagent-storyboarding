#!/usr/bin/env python3
"""
storyboard_to_video.py
Turn a storyboard JSON file into a slideshow video sketch with burned-in
subtitles and neural Edge-TTS voiceover.

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
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

FPS = 25
SAMPLE_RATE = 44100


# --------------------------------------------------------------------------- #
# Storyboard loading / audio-cue cleaning
# --------------------------------------------------------------------------- #
def load_scenes(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    sb = data.get("storyboard", data)
    scenes = sb["scenes"] if isinstance(sb, dict) else sb
    return sb.get("title", "storyboard") if isinstance(sb, dict) else "storyboard", scenes


_VOICE_LABELS = r"(?:VOIX(?:\s*OFF)?|VOICE(?:\s*OVER)?|VOICEOVER|VO|NARRATEUR|NARRATRICE|NARRATOR)"
_SFX_LABELS = r"(?:SFX|FX|BRUITAGE|MUSIQUE|MUSIC|SOUND|SON)"
_QUOTES = "'\"\u2018\u2019\u201c\u201d\u00ab\u00bb` "

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
    Cleans audio cues or voiceover into pure spoken script dialogue:
    - Strips bracketed or parenthesized SFX tags (e.g. '[SFX: ...]', '(SFX: ...)')
    - Strips leading SFX blocks (e.g. 'SFX: bell rings. VO: Hello')
    - Strips voice labels and descriptors (e.g. 'VO (flat, dead-direct):', 'Voix off :', 'Voiceover:')
    - Strips trailing SFX blocks (e.g. 'SFX: low hum...', 'SFX : bruit...')
    - Strips trailing inline SFX mentions (e.g. 'Deep rumble SFX.', 'Grunt SFX, rock crumbling.')
    - Strips trailing sound effect annotations (e.g. '+ beat de révélation (whoosh)')
    - Extracts dialogue from explicit quotes (« ... », " ... ", ' ... ')
    - Normalizes whitespace
    """
    cue = (cue or "").strip()
    if raw or not cue:
        return cue

    s = cue

    # 1. Strip bracketed or parenthesized SFX tags anywhere
    s = re.sub(r"\[(?:SFX|FX|SOUND|BRUITAGE)[^\]]*\]", "", s, flags=re.I)
    s = re.sub(r"\((?:SFX|FX|SOUND|BRUITAGE)[^)]*\)", "", s, flags=re.I)

    # 2. If SFX is at the very beginning, strip the leading SFX block
    s = re.sub(r"^(?:SFX|FX|BRUITAGE|MUSIQUE|MUSIC|SOUND)\s*[:\-][^.!?\n]*[.!?]?\s*", "", s, flags=re.I).strip()

    # 3. Strip leading voice/narrator labels with optional parenthetical descriptors
    voice_prefix = r"^(?:VO|VOIX(?:\s*OFF)?|VOICE(?:\s*OVER)?|VOICEOVER|NARRATEUR|NARRATRICE|NARRATOR|SPEAKER)(?:\s*\([^)]*\))*\s*[:\-]\s*"
    s = re.sub(voice_prefix, "", s, flags=re.I).strip()

    # 4. Strip trailing text after closing quote (e.g. '...omnipotent.' Frustrated sigh.)
    s = re.sub(r"([»'\"])\s+(?:[A-Z][a-z0-9\s,–—-]+[.!?]?\s*)+$", r"\1", s)

    # 5. Strip trailing SFX blocks
    s = re.sub(r"\b(?:SFX|FX|BRUITAGE|MUSIQUE|MUSIC|SOUND)\s*[:\-].*$", "", s, flags=re.I | re.S)

    # 6. Strip trailing inline SFX mentions
    s = re.sub(r"\s*['\"]?\s*[^'\"\n]*\b(?:SFX|FX)\b[^'\"\n]*\.?\s*$", "", s, flags=re.I)

    # 7. Strip trailing stage direction annotations with '+'
    s = re.sub(r"\s*[»'\"]?\s*\+\s*.*$", "", s, flags=re.S)

    # 8. Strip lingering outer quotation marks and whitespace
    s = s.strip(_QUOTES + " \n")
    s = re.sub(r"\s+", " ", s).strip()
    return s or cue


# --------------------------------------------------------------------------- #
# Images and frame rendering
# --------------------------------------------------------------------------- #
def download_image(url, timeout=15):
    try:
        if not url.lower().startswith(("http://", "https://")):   # local file path
            return Image.open(url.replace("file://", "")).convert("RGB")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return Image.open(io.BytesIO(r.read())).convert("RGB")
    except Exception as e:
        print(f"  ! could not fetch image ({type(e).__name__}: {e}); using placeholder")
        return None


def get_font(size):
    for name in ("DejaVuSans-Bold.ttf", "Arial Bold.ttf", "arialbd.ttf", "Arial.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                 "/System/Library/Fonts/Supplemental/Arial Bold.ttf"):
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


def make_placeholder(size, scene):
    """Used when the image URL is missing or unreachable."""
    w, h = size
    img = Image.new("RGB", size, (28, 32, 48))
    d = ImageDraw.Draw(img)
    title_font, body_font = get_font(h // 12), get_font(h // 26)
    d.text((w // 2, h * 0.28), f"Scène {scene.get('scene_id', '?')}", font=title_font,
           fill=(230, 230, 240), anchor="mm")
    desc = scene.get("description", "")[:220]
    for i, line in enumerate(wrap_text(d, desc, body_font, w * 0.75)[:5]):
        d.text((w // 2, h * 0.42 + i * h / 20), line, font=body_font,
               fill=(160, 165, 185), anchor="mm")
    return img


def fit_image(img, size):
    """Scale to fit inside size (no cropping) on a black background."""
    w, h = size
    img = img.copy()
    img.thumbnail((w, h), Image.LANCZOS)
    canvas = Image.new("RGB", size, (0, 0, 0))
    canvas.paste(img, ((w - img.width) // 2, (h - img.height) // 2))
    return canvas


def burn_subtitle(img, text):
    if not text:
        return img
    w, h = img.size
    font = get_font(h // 18)
    probe = ImageDraw.Draw(img)
    lines = wrap_text(probe, text, font, w * 0.86)
    line_h = int(font.size * 1.3)
    box_h = line_h * len(lines) + 24
    y0 = h - box_h - int(h * 0.05)

    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    od.rounded_rectangle((w * 0.05, y0, w * 0.95, y0 + box_h), radius=12, fill=(0, 0, 0, 175))
    img = Image.alpha_composite(img.convert("RGBA"), overlay)
    d = ImageDraw.Draw(img)
    for i, line in enumerate(lines):
        d.text((w / 2, y0 + 12 + i * line_h + line_h / 2), line, font=font,
               fill=(255, 255, 255), anchor="mm")
    return img.convert("RGB")


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
        raw.unlink(missing_ok=True)


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #
def build_scene_clip(scene, idx, tmp, tts, size, args):
    sid = scene.get("scene_id", idx + 1)
    print(f"Scene {sid}: {scene.get('subtitle', '')!r}")

    # Image
    img = download_image(scene["image_url"]) if scene.get("image_url") else None
    img = fit_image(img, size) if img else make_placeholder(size, scene)
    frame = tmp / f"frame_{idx:03d}.png"
    burn_subtitle(img, scene.get("subtitle", "")).save(frame)

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
    json_duration = float(scene.get("duration_seconds", 5))
    if args.json_durations or not speech.strip():
        # Old behaviour (or nothing to say): trust the storyboard's duration.
        duration, lead = json_duration, (pad if speech.strip() else 0.0)
    else:
        speech_len = audio_duration(wav)
        duration, lead = pad + speech_len + pad, pad
        print(f"  speech {speech_len:.1f}s -> scene {duration:.1f}s (json said {json_duration:.0f}s)")

    clip = tmp / f"clip_{idx:03d}.mp4"
    delay_ms = int(lead * 1000)
    run(["ffmpeg", "-y",
         "-loop", "1", "-framerate", str(FPS), "-i", str(frame),
         "-i", str(wav),
         # lead-in silence, then speech, then silence up to the scene length
         "-af", f"adelay={delay_ms}|{delay_ms},apad=whole_dur={duration}",
         "-t", f"{duration}",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", str(FPS),
         "-c:a", "aac", "-ar", str(SAMPLE_RATE), "-ac", "2",
         str(clip)])
    return clip


def main():
    ap = argparse.ArgumentParser(description="Storyboard JSON -> slideshow video with TTS")
    ap.add_argument("storyboard", help="path to the storyboard JSON")
    ap.add_argument("-o", "--output", default=None,
                    help="Output MP4 path. If omitted, automatically placed alongside the storyboard.")
    ap.add_argument("--lang", default="auto", help="TTS language code ('auto', 'en', 'fr', 'es', 'de', 'it'; default: auto)")
    ap.add_argument("--voice", default=None,
                    help="Specific TTS voice (e.g. 'fr-FR-HenriNeural', 'en-US-ChristopherNeural'). Auto-selected if omitted.")
    ap.add_argument("--rate", type=int, default=170, help="TTS speaking rate (words/min)")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720, help="use 1920 x 1080 for vertical shorts, etc.")
    ap.add_argument("--padding", type=float, default=1.0,
                    help="seconds of silence before and after the speech in each scene (default: 1.0)")
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
        clips = [build_scene_clip(s, i, tmp, tts, size, args) for i, s in enumerate(scenes)]

        listing = tmp / "clips.txt"
        listing.write_text("".join(f"file '{c.as_posix()}'\n" for c in clips), encoding="utf-8")
        run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(listing),
             "-c", "copy", str(out_path)])
        print(f"\nDone -> {out_path} ({audio_duration(out_path):.1f}s)")
    finally:
        if args.keep_temp:
            print(f"Temp files kept in {tmp}")
        else:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
