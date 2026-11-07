#!/usr/bin/env python3
"""
storyboard_to_video.py
Turn a storyboard JSON file into a simple slideshow video with burned-in
subtitles and robotic (offline) text-to-speech.

Usage:
    python storyboard_to_video.py storyboard.json -o out.mp4
    python storyboard_to_video.py storyboard.json --lang fr --keep-temp

Requirements:
    - ffmpeg on PATH
    - pip install pillow pyttsx3
    - Linux only: sudo apt install espeak-ng   (pyttsx3 uses it as its engine)
      (Windows uses SAPI5, macOS uses NSSpeechSynthesizer: nothing extra to install)
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


_VOICE_LABELS = r"(?:VOIX(?:\s*OFF)?|VOICE(?:\s*OVER)?|VO|NARRATEUR|NARRATRICE|NARRATOR)"
_SFX_LABELS = r"(?:SFX|FX|BRUITAGE|MUSIQUE|MUSIC|SOUND|SON)"
_QUOTES = "'\"\u2018\u2019\u201c\u201d\u00ab\u00bb` "


def clean_audio_cue(cue, raw=False):
    """
    The storyboard agent often writes cues like:
        VOIX : 'Tu te souviens ?' SFX : télé qui grésille
    We treat the cue as speech, but drop the labels and the SFX descriptions so
    the TTS doesn't read "S F X" out loud. Use raw=True to read the cue verbatim.
    """
    cue = (cue or "").strip()
    if raw or not cue:
        return cue

    # 1) Keep only the segments introduced by a voice label, stop at an SFX label.
    pattern = rf"\b{_VOICE_LABELS}\s*:\s*(.*?)(?=\b{_SFX_LABELS}\b\s*:|\b{_VOICE_LABELS}\s*:|$)"
    parts = re.findall(pattern, cue, flags=re.I | re.S)
    if parts:
        text = " ".join(p.strip(_QUOTES + " \n") for p in parts if p.strip(_QUOTES))
    else:
        # 2) No voice label: just cut out any "SFX : ..." segments.
        text = re.sub(rf"\b{_SFX_LABELS}\b\s*(?:of|de|:)?[^.!?]*[.!?]?", "", cue, flags=re.I)
        text = text.strip(_QUOTES + " \n")
    return re.sub(r"\s+", " ", text).strip() or cue


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
    r = run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nokey=1", str(path)])
    return float(r.stdout.strip())


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
    def __init__(self, lang="fr", rate=170, backend="auto", timeout=40):
        self.lang, self.rate, self.timeout = lang, rate, timeout
        self.exe = shutil.which("espeak-ng") or shutil.which("espeak")
        self.powershell = shutil.which("powershell") if sys.platform == "win32" else None
        if backend == "auto":
            try:
                import pyttsx3  # noqa: F401
                backend = "pyttsx3"
            except ImportError:
                backend = "espeak" if self.exe else ("powershell" if self.powershell else None)
        if backend is None:
            raise RuntimeError("No TTS engine found. Run: pip install pyttsx3")
        if backend == "espeak" and not self.exe:
            raise RuntimeError("espeak-ng not found on PATH")
        self.backend = backend

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
        raw = Path(out_wav).with_suffix(".raw.wav")
        attempts = [getattr(self, "_" + self.backend)]
        if self.powershell and self.backend != "powershell":
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
def build_scene_clip(scene, idx, tmp, tts, size, args):
    sid = scene.get("scene_id", idx + 1)
    print(f"Scene {sid}: {scene.get('subtitle', '')!r}")

    # Image
    img = download_image(scene["image_url"]) if scene.get("image_url") else None
    img = fit_image(img, size) if img else make_placeholder(size, scene)
    frame = tmp / f"frame_{idx:03d}.png"
    burn_subtitle(img, scene.get("subtitle", "")).save(frame)

    # Speech
    speech = clean_audio_cue(scene.get("audio_cue", ""), raw=args.raw_audio)
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
    ap.add_argument("-o", "--output", default="output.mp4")
    ap.add_argument("--lang", default="fr", help="TTS language code (default: fr)")
    ap.add_argument("--rate", type=int, default=170, help="TTS speaking rate (words/min)")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720, help="use 1920 x 1080 for vertical shorts, etc.")
    ap.add_argument("--padding", type=float, default=1.0,
                    help="seconds of silence before and after the speech in each scene (default: 1.0)")
    ap.add_argument("--json-durations", action="store_true",
                    help="use duration_seconds from the storyboard instead of the speech length")
    ap.add_argument("--raw-audio", action="store_true",
                    help="read audio_cue verbatim (including 'VOIX :' / 'SFX :' text)")
    ap.add_argument("--tts-backend", choices=["auto", "pyttsx3", "espeak", "powershell", "silent"], default="auto")
    ap.add_argument("--keep-temp", action="store_true")
    args = ap.parse_args()

    for tool in ("ffmpeg", "ffprobe"):
        if not shutil.which(tool):
            sys.exit(f"{tool} not found on PATH")

    title, scenes = load_scenes(args.storyboard)
    print(f"Building '{title}' ({len(scenes)} scenes)")
    size = (args.width, args.height)
    tts = TTS(args.lang, args.rate, args.tts_backend)

    tmp = Path(tempfile.mkdtemp(prefix="storyboard_"))
    try:
        clips = [build_scene_clip(s, i, tmp, tts, size, args) for i, s in enumerate(scenes)]

        listing = tmp / "clips.txt"
        listing.write_text("".join(f"file '{c.as_posix()}'\n" for c in clips), encoding="utf-8")
        run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(listing),
             "-c", "copy", str(args.output)])
        print(f"\nDone -> {args.output} ({audio_duration(args.output):.1f}s)")
    finally:
        if args.keep_temp:
            print(f"Temp files kept in {tmp}")
        else:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
