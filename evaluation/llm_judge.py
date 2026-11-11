#!/usr/bin/env python3
"""
evaluation/llm_judge.py - LLM-as-a-judge for the multi-agent vs one-shot storyboard study.

Reads the storyboard JSONs the pipeline already writes (outputs/storyboards/storyboard_*.json),
pairs the agentic and baseline storyboard generated from the SAME watch history, and judges:

  1. Absolute scores  - each storyboard scored 1-5 on relevance, creativity, completeness,
                        coherence (reasoning first, anchored rubric, JSON output).
  2. Pairwise choice  - "which would this user rather watch?", judged in BOTH A/B orders to
                        cancel position bias.

Blinding: the judge only sees the user's watch-history titles and the storyboard text
(title, narrative, per-scene visual/subtitle/audio). It never sees the agentic-only fields
(pitches, director notes, inferred profile, news, image URLs, pipeline log), nor which system
produced the storyboard.

Quick start:
    python evaluation/llm_judge.py --dry-run                      # no API calls, tests plumbing
    python evaluation/llm_judge.py --preset deepseek --limit 5    # cheap pilot
    python evaluation/llm_judge.py --preset gpt4o                 # stronger judge later
    python evaluation/llm_judge.py --preset sonnet                # Claude judge

Presets (override any piece with --model/--provider/--base-url/--api-key-env):
    deepseek -> deepseek-chat   via https://api.deepseek.com   (DEEPSEEK_API_KEY)
    gpt4o    -> gpt-4o          via OpenAI                     (OPENAI_API_KEY)
    sonnet   -> claude-sonnet-5-5 via Anthropic                (ANTHROPIC_API_KEY)

Results go to <out>/<model>/ (results.jsonl, summary.txt, summary_absolute.csv, run_config.json).
Calls are cached in <out>/cache.jsonl keyed by the full prompt + model + temperature, so
changing the rubric or model never reuses stale scores, and reruns resume for free.
"""
import argparse
import csv
import hashlib
import json
import random
import re
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

try:  # repo convention: keys live in .env
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

# --------------------------------------------------------------------------
# Rubric / prompts  (snapshotted into run_config.json for reproducibility)
# --------------------------------------------------------------------------
CRITERIA = ["relevance", "creativity", "completeness", "coherence", "visual_grounding"]

METRICS = {
    "relevance": (
        "How well the storyboard fits THIS user's specific interests, as evidenced by their "
        "watch history.\n"
        "MULTI-FACET RULE: The user's watch history contains multiple distinct topic clusters "
        "(e.g., Math Paradoxes, 90s/2000s Animation Nostalgia, French Comedy). A storyboard is "
        "considered RELEVANT (4–5/5) if it connects with ANY interest cluster the user seems "
        "genuinely interested in. It does not need to be the most frequent topic in the list.\n"
        "ANTI-REGURGITATION RULE: Merely copying or rehashing a watched video's exact premise is NOT "
        "true personalization; true relevance requires extending the user's taste to a fresh angle.\n"
        "1 = unrelated to the user's interests.\n"
        "2 = loosely related to a broad interest category only.\n"
        "3 = touches an interest evident in the history but in a generic way any fan could get, "
        "or merely rehashes an already-watched video.\n"
        "4 = clearly built on specific interests visible in the history with a fresh angle.\n"
        "5 = builds on specific niche interests and recurring patterns in the history in a way "
        "a generic video would not, offering an unseen curiosity."
    ),
    "creativity": (
        "Novelty and freshness of the concept, angle, and execution, while still suiting the user.\n"
        "ANTI-REGURGITATION PENALTY: Check if the storyboard merely copies, rehashes, or retells "
        "the exact topic/premise of a specific video already in the user's history (e.g., user watched "
        "'How Many Folds To Reach The Moon?' and the storyboard is 'Paper folding to the Moon'; or user "
        "watched 'How to make croissants from scratch' and the storyboard is '10-minute croissant hack'). "
        "If so:\n"
        "- CAP Creativity at 2/5 (it is a remake, not an original creation).\n"
        "- Do NOT reward this as high personalization; true relevance requires extending the user's "
        "taste to a FRESH angle or an UNSEEN curiosity.\n"
        "1 = generic template; the most obvious video on the topic, or direct copy of a watched video.\n"
        "2 = mostly conventional with minor variation, or a remake/rehash of a watched video (CAPPED at 2/5).\n"
        "3 = some original touches but a predictable structure.\n"
        "4 = a distinctive angle or structure that feels fresh.\n"
        "5 = surprising, memorable concept with high novelty that still clearly fits the user."
    ),
    "completeness": (
        "Whether the storyboard fully specifies a producible short video: scene descriptions, "
        "subtitles, AND voiceover.\n"
        "1 = major components missing or only a vague outline.\n"
        "2 = several scenes lack subtitles or voiceover, or key details are missing.\n"
        "3 = all components present but thin or uneven.\n"
        "4 = complete and specific, minor gaps.\n"
        "5 = fully specified for every scene (visuals, subtitles, voiceover), ready to produce."
    ),
    "coherence": (
        "Narrative flow and consistency: do scenes, subtitles and voiceover fit together as one "
        "watchable short with a hook and a payoff?\n"
        "1 = disjointed or contradictory.\n"
        "2 = weak flow; scenes feel unconnected.\n"
        "3 = understandable but with a flat or unclear arc.\n"
        "4 = good flow with a clear hook and ending.\n"
        "5 = tight, engaging arc; every element supports the whole."
    ),
    "visual_grounding": (
        "How faithfully the retrieved photo asset ('Visual Asset Photo' description/alt-text "
        "from the photo link) matches the scene's desired visual description.\n"
        "1 = completely contradictory or irrelevant photo asset (e.g., photo of a basketball court "
        "for a scene describing writing with a pen).\n"
        "2 = loosely or tangentially related only (e.g., generic street view when a specific food dish is described).\n"
        "3 = acceptable topical match, but misses key specific subjects or actions depicted in the scene.\n"
        "4 = strong visual match: the photo asset directly depicts the primary subject and setting.\n"
        "5 = exceptional visual-narrative alignment: the photo asset precisely captures the exact subject, "
        "atmosphere, and physical focus described in the storyboard."
    ),
}

ABS_SYSTEM = (
    "You are a strict, impartial evaluator of short-video storyboards. Use the full range of "
    "the scales and avoid defaulting to middle or high scores. Do NOT reward length or "
    "verbosity; judge quality only. Do not guess which system produced the storyboard.\n\n"
    "MULTI-FACET RELEVANCE DIRECTIVE:\n"
    "The user's watch history contains multiple distinct topic clusters (e.g., Math Paradoxes, "
    "90s/2000s Animation Nostalgia, French Comedy). A storyboard is considered RELEVANT (4–5/5) "
    "if it connects with ANY interest cluster the user seems genuinely interested in. It does not "
    "need to be the most frequent topic in the list.\n\n"
    "ANTI-REGURGITATION PENALTY:\n"
    "Check if the storyboard merely copies, rehashes, or retells the exact topic/premise of a "
    "specific video already in the user's history (e.g., user watched 'How Many Folds To Reach The Moon?' "
    "and the storyboard is 'Paper folding to the Moon'; or user watched 'How to make croissants from scratch' "
    "and the storyboard is '10-minute croissant hack'). If so:\n"
    "- CAP Creativity at 2/5 (it is a remake, not an original creation).\n"
    "- Do NOT reward this as high personalization; true relevance requires extending the user's "
    "taste to a FRESH angle or an UNSEEN curiosity.\n\n"
    "VISUAL-NARRATIVE GROUNDING EVALUATION:\n"
    "For each scene, check the 'Visual Asset Photo' (which contains the description and alt-text "
    "retrieved from the photo link) against the scene's desired 'Visual' description. "
    "Score high (4–5/5) if the retrieved photo accurately illustrates the specific physical subject "
    "and setting; score low (1–2/5) if the photo is generic, tangential, or contradicts the scene."
)

PAIR_SYSTEM = (
    "You are a strict, impartial evaluator comparing two short-video storyboards written for "
    "the same user. Evaluate each dimension independently. Do NOT reward length or verbosity. "
    "The order in which the storyboards are shown carries no information.\n\n"
    "MULTI-FACET RELEVANCE DIRECTIVE:\n"
    "The user's watch history contains multiple distinct topic clusters (e.g., Math Paradoxes, "
    "90s/2000s Animation Nostalgia, French Comedy). A storyboard is considered RELEVANT if it "
    "connects with ANY interest cluster the user seems genuinely interested in. It does not need "
    "to be the most frequent topic in the list.\n\n"
    "ANTI-REGURGITATION PENALTY:\n"
    "Check if a storyboard merely copies, rehashes, or retells the exact topic/premise of a "
    "specific video already in the user's history (e.g., user watched 'How Many Folds To Reach The Moon?' "
    "and the storyboard is 'Paper folding to the Moon'; or user watched 'How to make croissants from scratch' "
    "and the storyboard is '10-minute croissant hack'). If so, heavily penalize it: it is a remake/rerun, "
    "not an original creation. A user does NOT want to be shown a video they have already watched; "
    "true recommendation and personalization requires introducing a fresh angle, unseen curiosity, "
    "or novel extension of their taste.\n\n"
    "VISUAL-NARRATIVE GROUNDING EVALUATION:\n"
    "For each scene, evaluate how well the 'Visual Asset Photo' (which provides the real description "
    "and alt-text retrieved from the photo link) matches the scene's desired 'Visual' description. "
    "Favor the storyboard whose retrieved photo assets faithfully depict the specific physical scenes "
    "over one with uncurated, generic, or mismatched stock photos."
)


def build_abs_prompt(history: str, storyboard: str) -> str:
    rubric = "\n\n".join(f"### {k}\n{v}" for k, v in METRICS.items())
    schema = ", ".join(f'"{k}": {{"reasoning": "...", "score": <1-5 integer>}}' for k in METRICS)
    return f"""Evaluate the storyboard below for the given user.

<user_watch_history>
Titles of YouTube Shorts this user recently watched:
{history}
</user_watch_history>

<storyboard>
{storyboard}
</storyboard>

Score the storyboard on each criterion from 1 to 5 using these anchored scales:

{rubric}

For each criterion, write 1-2 sentences of reasoning FIRST, then give the score.
Return ONLY valid JSON, no markdown fences, in this form:
{{{schema}}}"""


def build_pair_prompt(history: str, sb_a: str, sb_b: str) -> str:
    return f"""Two short-video storyboards were generated for the same user.

<user_watch_history>
Titles of YouTube Shorts this user recently watched:
{history}
</user_watch_history>

<storyboard_A>
{sb_a}
</storyboard_A>

<storyboard_B>
{sb_b}
</storyboard_B>

Evaluate and compare the two storyboards across each of the 5 criteria below, plus pick an overall winner.
For each criterion, assign a grade on 5 (1 to 5 integer scale) to Storyboard A and Storyboard B, and declare the winner ("A", "B", or "tie"):

1. relevance: How well the storyboard connects to ANY genuine interest cluster in the watch history (Multi-Facet Relevance).
2. creativity: Concept novelty, distinctive hook/angle. Penalize direct remakes/rehashes of watched videos (Anti-Regurgitation Penalty: CAP at 2/5).
3. completeness: Full specification of visual descriptions, punchy subtitles, and voiceover for every scene (1-5).
4. coherence: Narrative flow, progression, hook-to-payoff arc (1-5).
5. visual_grounding: How faithfully the retrieved visual assets ('Visual Asset Photo') illustrate each scene's desired visual description (1-5).
Overall: Which video would this specific user prefer to watch in their feed?

Choose "tie" for any dimension only if they are genuinely indistinguishable.

Write 2-4 sentences of comparative reasoning FIRST, then declare scores on 5 and winners.
Return ONLY valid JSON, no markdown fences, in this exact format:
{{
  "scores_A": {{
    "relevance": <1-5 integer>,
    "creativity": <1-5 integer>,
    "completeness": <1-5 integer>,
    "coherence": <1-5 integer>,
    "visual_grounding": <1-5 integer>
  }},
  "scores_B": {{
    "relevance": <1-5 integer>,
    "creativity": <1-5 integer>,
    "completeness": <1-5 integer>,
    "coherence": <1-5 integer>,
    "visual_grounding": <1-5 integer>
  }},
  "relevance_winner": "A" | "B" | "tie",
  "creativity_winner": "A" | "B" | "tie",
  "completeness_winner": "A" | "B" | "tie",
  "coherence_winner": "A" | "B" | "tie",
  "visual_grounding_winner": "A" | "B" | "tie",
  "overall_winner": "A" | "B" | "tie",
  "reasoning": "..."
}}"""


# --------------------------------------------------------------------------
# Loading storyboards produced by the pipeline (Orchestrator / SingleAgentBaseline)
# --------------------------------------------------------------------------
def history_key(videos) -> str:
    urls = sorted(v.get("url", "").strip() for v in videos)
    return hashlib.sha1("\n".join(urls).encode()).hexdigest()


def render_storyboard(data: dict) -> str:
    """Neutral text view: renders title, narrative, scenes, and visual assets."""
    sb = data.get("storyboard", {})
    img_ctx = data.get("image_context", {})
    img_map = {img.get("image_id"): img for img in img_ctx.get("images", [])} if isinstance(img_ctx, dict) else {}

    lines = [f"Title: {sb.get('title', '')}", f"Concept: {sb.get('overall_narrative', '')}", "", "Scenes:"]
    for i, s in enumerate(sb.get("scenes", []), 1):
        idx = s.get("scene_id", i)
        dur = s.get("duration_seconds", "?")
        desc = s.get("description", "")
        subtitle = s.get("subtitle", "")
        voiceover = s.get("voiceover") or s.get("audio_cue", "")

        # Extract metadata retrieved strictly from the image link / source website
        ref = s.get("image_ref")
        photo_desc = s.get("image_description")
        photo_alt = s.get("image_alt_text")
        if ref and ref in img_map:
            if not photo_desc:
                photo_desc = img_map[ref].get("description")
            if not photo_alt:
                photo_alt = img_map[ref].get("alt_text")

        parts = []
        if photo_desc and str(photo_desc).strip():
            parts.append(str(photo_desc).strip())
        if photo_alt and str(photo_alt).strip():
            cleaned_alt = re.sub(r"\s*\(Photo by [^)]+\)", "", str(photo_alt).strip())
            if cleaned_alt and (not photo_desc or cleaned_alt.lower() != str(photo_desc).strip().lower()):
                parts.append(cleaned_alt)
        image_desc = " - ".join(parts) if parts else None

        lines.append(f"Scene {idx} ({dur}s):")
        lines.append(f"  Visual: {desc}")
        if image_desc:
            lines.append(f"  Visual Asset Photo: {image_desc}")
        lines.append(f"  Subtitle: {subtitle}")
        lines.append(f"  Voiceover: {voiceover}")
    return "\n".join(lines)


def render_history(videos, max_items: int) -> str:
    return "\n".join(f"{i}. {v.get('title', '').strip()}" for i, v in enumerate(videos[:max_items], 1))


def load_items(sb_dir: Path, hist_dir, run_ids, max_history):
    """Return {user_id: {"agentic": (run_id, data), "baseline": (run_id, data), "history": str}}."""
    names = {}
    if hist_dir and hist_dir.exists():
        for p in hist_dir.glob("*.json"):
            try:
                names[history_key(json.loads(p.read_text(encoding="utf-8")))] = p.stem
            except Exception:
                pass

    wanted = None
    if run_ids:
        wanted = {r.replace("storyboard_", "").replace("agentic_storyboard_", "").replace("baseline_storyboard_", "").replace(".json", "") for r in run_ids}

    # Find storyboard JSONs recursively under sb_dir
    json_paths = set()
    if sb_dir.exists():
        for pat in ("**/agentic_storyboard_*.json", "**/baseline_storyboard_*.json", "**/storyboard_*.json", "*.json"):
            for p in sb_dir.glob(pat):
                if p.is_file() and not p.name.startswith(("run_config", "cache", "result", "judge_")):
                    json_paths.add(p)

    cands = {}  # (hkey, system) -> list[(sortkey, run_id, data)]
    for p in sorted(json_paths):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[warn] unreadable {p.name}: {e}", file=sys.stderr)
            continue
        run_id = data.get("run_id") or p.stem
        if wanted and run_id not in wanted and p.stem not in wanted:
            continue
        sb = data.get("storyboard", {})
        videos = data.get("user_history", {}).get("videos", [])
        if not sb.get("scenes") or not videos:
            continue

        # Detect system
        is_base = (
            run_id.startswith("baseline_")
            or p.name.startswith("baseline_")
            or "baseline" in str(p.parent).lower()
        )
        system = "baseline" if is_base else "agentic"
        sortkey = run_id
        cands.setdefault((history_key(videos), system), []).append((sortkey, run_id, data))

    items, hkeys = {}, sorted({k for k, _ in cands})
    for hk in hkeys:
        if (hk, "agentic") not in cands or (hk, "baseline") not in cands:
            print(f"[warn] history {names.get(hk, hk[:8])} has only one system; skipped.", file=sys.stderr)
            continue
        ag_data = cands[(hk, "agentic")][-1][2]
        bl_data = cands[(hk, "baseline")][-1][2]
        uid = (
            ag_data.get("user_id")
            or bl_data.get("user_id")
            or names.get(hk, f"user_{hk[:8]}")
        )
        if uid == "default_user":
            uid = names.get(hk, f"user_{hk[:8]}")

        entry = {}
        for system in ("agentic", "baseline"):
            runs = sorted(cands[(hk, system)])
            if len(runs) > 1:
                print(f"[info] {uid}/{system}: {len(runs)} runs found, using latest "
                      f"({runs[-1][1]}). Use --run-ids to choose explicitly.")
            entry[system] = (runs[-1][1], runs[-1][2])
        entry["history"] = render_history(entry["agentic"][1]["user_history"]["videos"], max_history)
        items[uid] = entry
    return items


# --------------------------------------------------------------------------
# Judge clients
# --------------------------------------------------------------------------
USAGE = {"in": 0, "out": 0, "calls": 0}
_USAGE_LOCK = threading.Lock()


def _add_usage(i, o):
    with _USAGE_LOCK:
        USAGE["in"] += i or 0
        USAGE["out"] += o or 0
        USAGE["calls"] += 1


def _backoff(attempt, e):
    wait = 2 ** attempt + random.random()
    print(f"[warn] API error ({e}); retry in {wait:.1f}s", file=sys.stderr)
    time.sleep(wait)


def _need_key(env):
    import os

    key = os.environ.get(env)
    if not key:
        sys.exit(f"Environment variable {env} is not set (put it in .env or export it).")
    return key


class OpenAICompatJudge:
    """OpenAI, DeepSeek, Gemini (OpenAI-compat endpoint), vLLM... anything OpenAI-compatible."""

    def __init__(self, model, temperature, base_url, api_key_env, json_mode=True, max_tokens=1200):
        from openai import OpenAI

        self.client = OpenAI(api_key=_need_key(api_key_env), base_url=base_url)
        self.model, self.temperature, self.max_tokens, self.json_mode = model, temperature, max_tokens, json_mode

    def __call__(self, system, prompt, retries=5):
        extra = {"response_format": {"type": "json_object"}} if self.json_mode else {}
        for attempt in range(retries):
            try:
                r = self.client.chat.completions.create(
                    model=self.model, max_tokens=self.max_tokens, temperature=self.temperature,
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
                    **extra)
                u = getattr(r, "usage", None)
                _add_usage(getattr(u, "prompt_tokens", 0), getattr(u, "completion_tokens", 0))
                return r.choices[0].message.content or ""
            except Exception as e:
                _backoff(attempt, e)
        raise RuntimeError("API failed after retries")


class AnthropicJudge:
    def __init__(self, model, temperature, api_key_env, max_tokens=1200):
        import anthropic

        self.client = anthropic.Anthropic(api_key=_need_key(api_key_env))
        self.model, self.temperature, self.max_tokens = model, temperature, max_tokens

    def __call__(self, system, prompt, retries=5):
        for attempt in range(retries):
            try:
                r = self.client.messages.create(
                    model=self.model, max_tokens=self.max_tokens, temperature=self.temperature,
                    system=system, messages=[{"role": "user", "content": prompt}])
                _add_usage(r.usage.input_tokens, r.usage.output_tokens)
                return "".join(b.text for b in r.content if b.type == "text")
            except Exception as e:
                _backoff(attempt, e)
        raise RuntimeError("API failed after retries")


class StubJudge:
    """--dry-run: deterministic fake verdicts, zero cost. Only for testing the plumbing."""

    model = "dry-run"

    def __call__(self, system, prompt, retries=0):
        h = int(hashlib.sha1(prompt.encode()).hexdigest(), 16)
        if "<storyboard_A>" in prompt:
            choices = ["A", "B", "tie"]
            res = {f"{c}_winner": choices[(h >> (i * 2)) % 3] for i, c in enumerate(CRITERIA)}
            res["overall_winner"] = choices[h % 3]
            res["scores_A"] = {c: 1 + (h >> (i * 2)) % 5 for i, c in enumerate(CRITERIA)}
            res["scores_B"] = {c: 1 + ((h >> (i * 2 + 1)) ^ 3) % 5 for i, c in enumerate(CRITERIA)}
            res["reasoning"] = "stub pairwise reasoning"
            return json.dumps(res)
        return json.dumps({m: {"reasoning": "stub", "score": 1 + (h >> (i * 3)) % 5}
                           for i, m in enumerate(METRICS)})


PRESETS = {
    "deepseek": dict(provider="openai", model="deepseek-chat", base_url="https://api.deepseek.com",
                     api_key_env="DEEPSEEK_API_KEY"),
    "gpt4o": dict(provider="openai", model="gpt-4o", base_url=None, api_key_env="OPENAI_API_KEY"),
    "sonnet": dict(provider="anthropic", model="claude-sonnet-5-5", base_url=None,
                   api_key_env="ANTHROPIC_API_KEY"),
}


def make_judge(a):
    if a.dry_run:
        return StubJudge()
    if a.provider == "anthropic":
        return AnthropicJudge(a.model, a.temperature, a.api_key_env)
    return OpenAICompatJudge(a.model, a.temperature, a.base_url, a.api_key_env, not a.no_json_mode)


# --------------------------------------------------------------------------
# Cache + parsing
# --------------------------------------------------------------------------
class Cache:
    def __init__(self, path: Path):
        self.path, self.lock, self.data = path, threading.Lock(), {}
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rec = json.loads(line)
                    self.data[rec["key"]] = rec["value"]

    def get(self, key):
        return self.data.get(key)

    def put(self, key, value):
        with self.lock:
            self.data[key] = value
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"key": key, "value": value}, ensure_ascii=False) + "\n")


def make_key(*parts) -> str:
    return hashlib.sha256("||".join(map(str, parts)).encode()).hexdigest()


def extract_json(raw: str) -> dict:
    raw = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    s, e = raw.find("{"), raw.rfind("}")
    if s == -1 or e == -1:
        raise ValueError("no JSON object found")
    return json.loads(raw[s:e + 1])


def call_parsed(judge, system, prompt, parser, attempts=3):
    """Call the judge; re-ask if the output is malformed (cheap models do this occasionally)."""
    last = None
    for _ in range(attempts):
        try:
            return parser(extract_json(judge(system, prompt)))
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as e:
            last = e
    raise RuntimeError(f"unparseable judge output after {attempts} tries: {last}")


# --------------------------------------------------------------------------
# Tasks
# --------------------------------------------------------------------------
def run_absolute(judge, cache, model, temp, uid, system_name, history, sb_text, run):
    prompt = build_abs_prompt(history, sb_text)
    key = make_key("abs", model, temp, ABS_SYSTEM, prompt, run)
    rec = cache.get(key)
    if rec:
        return {**rec, "user": uid, "system": system_name, "run": run}

    def parse(d):
        out = {}
        for m in METRICS:
            sc = int(d[m]["score"])
            if not 1 <= sc <= 5:
                raise ValueError(f"{m} score {sc} out of range")
            out[m] = {"score": sc, "reasoning": d[m].get("reasoning", "")}
        return out

    scores = call_parsed(judge, ABS_SYSTEM, prompt, parse)
    cache.put(key, {"type": "abs", "scores": scores})
    return {"type": "abs", "user": uid, "system": system_name, "run": run, "scores": scores}


def run_pairwise(judge, cache, model, temp, uid, history, sb_ag, sb_bl, run, order):
    a, b = (sb_ag, sb_bl) if order == "AB" else (sb_bl, sb_ag)
    prompt = build_pair_prompt(history, a, b)
    key = make_key("pair_v4", model, temp, PAIR_SYSTEM, prompt, run)
    rec = cache.get(key)

    def parse(d):
        out = {}
        scores_a, scores_b = {}, {}
        raw_sa = d.get("scores_A", {}) if isinstance(d.get("scores_A"), dict) else {}
        raw_sb = d.get("scores_B", {}) if isinstance(d.get("scores_B"), dict) else {}

        def parse_score(val):
            try:
                sc = round(float(val))
                return max(1, min(5, sc))
            except (ValueError, TypeError):
                return None

        for c in CRITERIA:
            sc_a = parse_score(raw_sa.get(c))
            sc_b = parse_score(raw_sb.get(c))
            scores_a[c] = sc_a
            scores_b[c] = sc_b

            k = f"{c}_winner"
            val = str(d.get(k, "")).strip().lower()
            if val not in ("a", "b", "tie"):
                if sc_a is not None and sc_b is not None:
                    val = "a" if sc_a > sc_b else ("b" if sc_b > sc_a else "tie")
                else:
                    val = str(d.get("winner", "tie")).strip().lower()
                    if val not in ("a", "b", "tie"):
                        val = "tie"
            out[k] = val

        overall = str(d.get("overall_winner", d.get("winner", "tie"))).strip().lower()
        if overall not in ("a", "b", "tie"):
            overall = "tie"
        out["overall_winner"] = overall
        out["scores_A"] = scores_a
        out["scores_B"] = scores_b
        out["reasoning"] = d.get("reasoning", "")
        return out

    if not rec:
        rec = call_parsed(judge, PAIR_SYSTEM, prompt, parse)
        cache.put(key, rec)

    def map_choice(w: str) -> str:
        if w == "tie":
            return "tie"
        return "agentic" if (w == "a") == (order == "AB") else "baseline"

    criteria_winners = {c: map_choice(rec.get(f"{c}_winner", "tie")) for c in CRITERIA}
    overall_winner = map_choice(rec.get("overall_winner", "tie"))

    if order == "AB":
        agentic_scores = rec.get("scores_A", {})
        baseline_scores = rec.get("scores_B", {})
    else:
        agentic_scores = rec.get("scores_B", {})
        baseline_scores = rec.get("scores_A", {})

    return {
        "type": "pair",
        "user": uid,
        "run": run,
        "order": order,
        "winner": overall_winner,
        "overall_winner": overall_winner,
        "criteria_winners": criteria_winners,
        "agentic_scores": agentic_scores,
        "baseline_scores": baseline_scores,
        "raw_choices": {f"{c}_winner": rec.get(f"{c}_winner") for c in CRITERIA} | {"overall_winner": rec.get("overall_winner")},
        "reasoning": rec.get("reasoning", ""),
    }


# --------------------------------------------------------------------------
# Statistics / report
# --------------------------------------------------------------------------
def bootstrap_ci(diffs, n=5000, seed=0):
    rng = random.Random(seed)
    means = sorted(statistics.mean(rng.choices(diffs, k=len(diffs))) for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n)]


def wilcoxon_p(a, b):
    try:
        from scipy.stats import wilcoxon

        if all(x == y for x, y in zip(a, b)):
            return 1.0
        return float(wilcoxon(a, b).pvalue)
    except Exception:
        return None


def summarize(abs_recs, pair_recs, users, texts, out_dir, human_csv):
    lines = []

    def log(s=""):
        print(s)
        lines.append(s)

    if len(users) < 10:
        log(f"!! Only {len(users)} user(s): CIs and p-values are not meaningful below ~10 users. "
            "Treat these numbers as a pilot / qualitative signal.\n")

    per = {}
    for r in abs_recs:
        for m, v in r["scores"].items():
            per.setdefault((r["system"], m), {}).setdefault(r["user"], []).append(v["score"])
    per = {k: {u: statistics.mean(v) for u, v in d.items()} for k, d in per.items()}

    log("=" * 78)
    log(f"ABSOLUTE SCORES (1-5), n_users = {len(users)}")
    log("=" * 78)
    log(f"{'metric':<18}{'agentic':>9}{'baseline':>10}{'delta':>8}{'95% CI (boot)':>20}{'wilcoxon p':>12}")
    rows = []
    for m in METRICS:
        ag, bl = per.get(("agentic", m), {}), per.get(("baseline", m), {})
        common = sorted(set(ag) & set(bl))
        if not common:
            continue
        a, b = [ag[u] for u in common], [bl[u] for u in common]
        diffs = [x - y for x, y in zip(a, b)]
        lo, hi = bootstrap_ci(diffs) if len(diffs) > 1 else (float("nan"),) * 2
        p = wilcoxon_p(a, b) if len(diffs) > 1 else None
        log(f"{m:<18}{statistics.mean(a):>9.2f}{statistics.mean(b):>10.2f}{statistics.mean(diffs):>+8.2f}"
            f"{f'[{lo:+.2f}, {hi:+.2f}]':>20}{(f'{p:.4f}' if p is not None else 'n/a'):>12}")
        rows.append({"metric": m, "agentic_mean": statistics.mean(a), "baseline_mean": statistics.mean(b),
                     "delta": statistics.mean(diffs), "ci_lo": lo, "ci_hi": hi, "wilcoxon_p": p, "n": len(common)})

    log()
    log("=" * 98)
    log("PAIRWISE EVALUATION & PREFERENCE BREAKDOWN (grades on 5 + win rates, both orders judged)")
    log("=" * 98)
    log(f"{'dimension':<18}{'ag_grade':>9}{'bl_grade':>9}{'delta':>8}{'agentic':>8}{'baseline':>9}{'tie':>6}{'win_rate (excl)':>18}{'consistency':>13}")
    log("-" * 98)

    by_user = {}
    for r in pair_recs:
        by_user.setdefault(r["user"], []).append(r)

    dims_all = list(CRITERIA) + ["overall"]
    pairwise_rows = []

    for dim in dims_all:
        dim_votes = {"agentic": 0, "baseline": 0, "tie": 0}
        ag_scores_list, bl_scores_list = [], []
        n_pairs, consistent = 0, 0
        for u, recs in by_user.items():
            by_run = {}
            for r in recs:
                w = r.get("overall_winner", r.get("winner", "tie")) if dim == "overall" else r.get("criteria_winners", {}).get(dim, "tie")
                dim_votes[w] += 1
                by_run.setdefault(r["run"], {})[r["order"]] = w

                # Collect grades on 5
                if dim == "overall":
                    scs_ag = [v for v in r.get("agentic_scores", {}).values() if v is not None]
                    scs_bl = [v for v in r.get("baseline_scores", {}).values() if v is not None]
                    if scs_ag:
                        ag_scores_list.append(statistics.mean(scs_ag))
                    if scs_bl:
                        bl_scores_list.append(statistics.mean(scs_bl))
                else:
                    s_ag = r.get("agentic_scores", {}).get(dim)
                    s_bl = r.get("baseline_scores", {}).get(dim)
                    if s_ag is not None:
                        ag_scores_list.append(s_ag)
                    if s_bl is not None:
                        bl_scores_list.append(s_bl)

            for d in by_run.values():
                if len(d) == 2:
                    n_pairs += 1
                    consistent += (d["AB"] == d["BA"])

        total = dim_votes["agentic"] + dim_votes["baseline"] + dim_votes["tie"]
        dec = dim_votes["agentic"] + dim_votes["baseline"]
        wr = (dim_votes["agentic"] / dec) if dec > 0 else 0.0
        cons_rate = (consistent / n_pairs) if n_pairs > 0 else 1.0

        mean_ag_score = statistics.mean(ag_scores_list) if ag_scores_list else None
        mean_bl_score = statistics.mean(bl_scores_list) if bl_scores_list else None
        delta_score = (mean_ag_score - mean_bl_score) if (mean_ag_score is not None and mean_bl_score is not None) else None

        p_val = None
        if dec > 0:
            try:
                from scipy.stats import binomtest
                p_val = float(binomtest(dim_votes["agentic"], dec, 0.5).pvalue)
            except Exception:
                pass

        disp_dim = "OVERALL" if dim == "overall" else dim
        ag_sc_str = f"{mean_ag_score:.2f}" if mean_ag_score is not None else "n/a"
        bl_sc_str = f"{mean_bl_score:.2f}" if mean_bl_score is not None else "n/a"
        delta_str = f"{delta_score:+.2f}" if delta_score is not None else "n/a"

        log(f"{disp_dim:<18}{ag_sc_str:>9}{bl_sc_str:>9}{delta_str:>8}"
            f"{dim_votes['agentic']:>8}{dim_votes['baseline']:>9}{dim_votes['tie']:>6}"
            f"{(f'{wr:.1%}' if dec > 0 else 'n/a'):>18}{cons_rate:>13.1%}")

        pairwise_rows.append({
            "dimension": dim,
            "agentic_grade_on_5": ag_sc_str,
            "baseline_grade_on_5": bl_sc_str,
            "grade_delta": delta_str,
            "agentic_wins": dim_votes["agentic"],
            "baseline_wins": dim_votes["baseline"],
            "ties": dim_votes["tie"],
            "total_calls": total,
            "agentic_win_rate": f"{wr:.3f}" if dec > 0 else "n/a",
            "position_consistency": f"{cons_rate:.3f}" if n_pairs > 0 else "1.000",
            "binom_p_value": f"{p_val:.4f}" if p_val is not None else "n/a",
        })

    log("=" * 98)

    user_winner = {}
    for u, recs in by_user.items():
        sc = statistics.mean(
            1.0 if r.get("overall_winner", r.get("winner")) == "agentic"
            else 0.5 if r.get("overall_winner", r.get("winner")) == "tie"
            else 0.0
            for r in recs
        )
        user_winner[u] = "agentic" if sc > 0.5 else "baseline" if sc < 0.5 else "tie"
    uw = list(user_winner.values())
    log(f"Per-user majority (overall): agentic {uw.count('agentic')} | baseline {uw.count('baseline')} | tie {uw.count('tie')}")

    log()
    log("=" * 78)
    log("LENGTH CHECK (verbosity-bias diagnostic)")
    log("=" * 78)
    for s in ("agentic", "baseline"):
        log(f"{s:<10} mean words in judged text: {statistics.mean(len(texts[(u, s)].split()) for u in users):.0f}")

    if human_csv:
        log()
        log("=" * 78)
        log("JUDGE vs HUMAN AGREEMENT")
        log("=" * 78)
        with open(human_csv, newline="", encoding="utf-8") as f:
            human = {r["user_id"].strip(): r["winner"].strip().lower() for r in csv.DictReader(f)}
        pairs = [(user_winner[u], h) for u, h in human.items()
                 if user_winner.get(u) in ("agentic", "baseline") and h in ("agentic", "baseline")]
        log(f"Agreement on {len(pairs)} overlapping users: "
            f"{sum(j == h for j, h in pairs) / len(pairs):.1%}" if pairs else "No overlapping decisive users.")

    (out_dir / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
    if pairwise_rows:
        with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(pairwise_rows[0].keys()))
            w.writeheader()
            w.writerows(pairwise_rows)
    if rows:
        with (out_dir / "summary_absolute.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--storyboards", type=Path, default=Path("outputs"))
    ap.add_argument("--history-dir", type=Path, default=Path("data/user_histories"),
                    help="Only used to name users by their history file (e.g. participant_alice).")
    ap.add_argument("--run-ids", nargs="*", default=None,
                    help="Restrict to these run ids (default: latest agentic + latest baseline per user).")
    ap.add_argument("--out", type=Path, default=Path("outputs/judge"))
    ap.add_argument("--preset", choices=list(PRESETS), default="deepseek")
    ap.add_argument("--provider", choices=["openai", "anthropic"], default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--api-key-env", default=None)
    ap.add_argument("--no-json-mode", action="store_true", help="If the provider rejects response_format.")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--runs", type=int, default=1, help="Repeats per item (needs temperature > 0 to differ).")
    ap.add_argument("--max-history", type=int, default=150, help="Max watch-history titles shown to the judge.")
    ap.add_argument("--limit", type=int, default=None, help="Only the first N users (pilot).")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--skip-absolute", action="store_true")
    ap.add_argument("--skip-pairwise", action="store_true")
    ap.add_argument("--human-votes", type=Path, default=None, help="CSV: user_id,winner(agentic|baseline).")
    ap.add_argument("--price-in", type=float, default=None, help="USD per 1M input tokens (cost estimate).")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"), type=Path, default=None,
                    help="Compare two specific storyboard JSON files side-by-side.")
    ap.add_argument("--price-out", type=float, default=None, help="USD per 1M output tokens.")
    ap.add_argument("--dry-run", action="store_true", help="Fake judge, no API calls.")
    args = ap.parse_args()

    preset = PRESETS[args.preset]
    for k in ("provider", "model", "base_url", "api_key_env"):
        if getattr(args, k) is None:
            setattr(args, k, preset[k])
    if args.dry_run:
        args.model = "dry-run"

    compare_user_id = None
    if args.compare:
        path_a, path_b = args.compare
        with open(path_a, encoding="utf-8") as f:
            data_a = json.load(f)
        with open(path_b, encoding="utf-8") as f:
            data_b = json.load(f)

        # Detect which is baseline vs agentic
        is_a_base = "baseline" in str(path_a).lower() or "baseline" in str(data_a.get("run_id", "")).lower()
        is_b_base = "baseline" in str(path_b).lower() or "baseline" in str(data_b.get("run_id", "")).lower()

        if is_a_base and not is_b_base:
            agentic_data, baseline_data = data_b, data_a
            agentic_id, baseline_id = path_b.stem, path_a.stem
        else:
            agentic_data, baseline_data = data_a, data_b
            agentic_id, baseline_id = path_a.stem, path_b.stem

        # Detect user identifier
        compare_user_id = agentic_data.get("user_id") or baseline_data.get("user_id")
        if not compare_user_id or compare_user_id == "default_user":
            m = re.search(r"outputs[/\\]([^/\\]+)", str(path_a)) or re.search(r"outputs[/\\]([^/\\]+)", str(path_b))
            if m and m.group(1) not in ("storyboards", "judge", "agentic", "baseline"):
                compare_user_id = m.group(1)
            else:
                compare_user_id = "comparison"

        history_videos = (agentic_data.get("user_history", {}).get("videos")
                          or baseline_data.get("user_history", {}).get("videos")
                          or [])
        items = {
            compare_user_id: {
                "agentic": (agentic_id, agentic_data),
                "baseline": (baseline_id, baseline_data),
                "history": render_history(history_videos, args.max_history),
            }
        }
    else:
        items = load_items(args.storyboards, args.history_dir, args.run_ids, args.max_history)
    users = sorted(items)[: args.limit] if args.limit else sorted(items)
    if not users:
        sys.exit("No user with BOTH an agentic and a baseline storyboard on the same watch history.")
    print(f"Judging {len(users)} user(s) with {args.model} (runs={args.runs})")

    args.out.mkdir(parents=True, exist_ok=True)
    run_dir = args.out / re.sub(r"[^A-Za-z0-9._-]", "_", args.model)
    run_dir.mkdir(parents=True, exist_ok=True)
    judge = make_judge(args)
    cache = Cache(args.out / ("cache_dryrun.jsonl" if args.dry_run else "cache.jsonl"))

    texts = {}
    for u in users:
        for s in ("agentic", "baseline"):
            texts[(u, s)] = render_storyboard(items[u][s][1])

    jobs, abs_recs, pair_recs, failures = [], [], [], 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for u in users:
            for run in range(args.runs):
                if not args.skip_absolute:
                    for s in ("agentic", "baseline"):
                        jobs.append(ex.submit(run_absolute, judge, cache, args.model, args.temperature, u, s,
                                              items[u]["history"], texts[(u, s)], run))
                if not args.skip_pairwise:
                    for order in ("AB", "BA"):
                        jobs.append(ex.submit(run_pairwise, judge, cache, args.model, args.temperature, u,
                                              items[u]["history"], texts[(u, "agentic")],
                                              texts[(u, "baseline")], run, order))
        for i, fut in enumerate(as_completed(jobs), 1):
            try:
                rec = fut.result()
                (abs_recs if rec["type"] == "abs" else pair_recs).append(rec)
            except Exception as e:
                failures += 1
                print(f"[error] {e}", file=sys.stderr)
            if i % 20 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} calls done")
    if failures:
        print(f"[warn] {failures} call(s) failed; rerun to retry them (successes are cached).")

    with (run_dir / "results.jsonl").open("w", encoding="utf-8") as f:
        for r in abs_recs + pair_recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    cost = None
    if args.price_in is not None and args.price_out is not None:
        cost = (USAGE["in"] * args.price_in + USAGE["out"] * args.price_out) / 1e6
    cfg = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "provider": args.provider, "model": args.model, "base_url": args.base_url,
        "temperature": args.temperature, "runs": args.runs, "max_history": args.max_history,
        "users": {u: {"agentic_run": items[u]["agentic"][0], "baseline_run": items[u]["baseline"][0]} for u in users},
        "token_usage_this_invocation": USAGE, "estimated_cost_usd": cost, "failures": failures,
        "prompts": {"abs_system": ABS_SYSTEM, "pair_system": PAIR_SYSTEM, "rubric": METRICS,
                    "abs_template": build_abs_prompt("{history}", "{storyboard}"),
                    "pair_template": build_pair_prompt("{history}", "{storyboard_A}", "{storyboard_B}")},
    }
    (run_dir / "run_config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")

    summarize(abs_recs, pair_recs, users, texts, run_dir, args.human_votes)
    print(f"\nTokens this invocation: {USAGE['in']:,} in / {USAGE['out']:,} out over {USAGE['calls']} calls"
          + (f" (~${cost:.4f})" if cost is not None else "  (pass --price-in/--price-out for a $ estimate)"))
    print(f"Wrote results to {run_dir}/  -> log tokens/cost in docs/project_tracker.md")

    if args.compare:
        print("\n================ SIDE-BY-SIDE COMPARISON ================")
        print(f"Agentic ({agentic_id}) vs Baseline ({baseline_id})\n")
        print("[ABSOLUTE GRADES (1-5)]")
        for rec in abs_recs:
            print(f"[{rec['system'].upper()}] Scores (1-5):")
            for m in METRICS:
                entry = rec['scores'].get(m, {})
                print(f"  - {m.title()}: {entry.get('score', '?')}/5 | {entry.get('reasoning', '')}")
            print()
        if pair_recs:
            print("Pairwise Verdicts & Grades on 5 (both A/B orders judged for position debiasing):")
            for p in pair_recs:
                print(f"  - Order {p.get('order')}: Overall Winner = {p.get('overall_winner', p.get('winner')).upper()}")
                cw = p.get("criteria_winners", {})
                ag_sc = p.get("agentic_scores", {})
                bl_sc = p.get("baseline_scores", {})
                if cw:
                    for d in CRITERIA:
                        w = cw.get(d, "tie").upper()
                        s_ag = ag_sc.get(d)
                        s_bl = bl_sc.get(d)
                        score_info = f" [Agentic: {s_ag}/5 vs Baseline: {s_bl}/5]" if s_ag is not None and s_bl is not None else ""
                        print(f"      * {d:<18}: {w:<8}{score_info}")
                print(f"    Reasoning: {p.get('reasoning')}")
        print("=========================================================\n")

        # Persist structured comparison to outputs/<user>/judge/
        user_target = compare_user_id or "comparison"
        user_judge_dir = Path("outputs") / user_target / "judge"
        user_judge_dir.mkdir(parents=True, exist_ok=True)
        cmp_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        cmp_json = user_judge_dir / f"judge_comparison_{user_target}_{cmp_ts}.json"
        cmp_json.write_text(json.dumps({
            "user_id": user_target,
            "timestamp": cmp_ts,
            "agentic_id": agentic_id,
            "baseline_id": baseline_id,
            "absolute_evaluations": abs_recs,
            "pairwise_evaluations": pair_recs,
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Saved comparison to: {cmp_json}")


if __name__ == "__main__":
    main()