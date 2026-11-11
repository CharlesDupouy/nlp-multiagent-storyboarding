"""
LLM Client — thin wrapper around the OpenAI-compatible API.

DeepSeek's API is fully OpenAI-compatible, so we simply point the
openai client at DeepSeek's base URL. To switch to OpenAI GPT-4o,
change the base_url and model in config.yaml.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from typing import Any

from openai import OpenAI

# Load .env file automatically if present (never committed to git)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Default reasoning effort for reasoning models ("none", "low", "high", "max")
_DEFAULT_REASONING_EFFORT: str | None = "low"

# Global token accounting for reproducibility and cost auditing
_TOKEN_USAGE = {
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "reasoning_tokens": 0,
    "total_calls": 0,
}


def set_reasoning_effort(effort: str | None) -> None:
    """Set the default reasoning effort level ('none', 'low', 'high', 'max', or None)."""
    global _DEFAULT_REASONING_EFFORT
    _DEFAULT_REASONING_EFFORT = (
        effort.strip().lower() if isinstance(effort, str) and effort.strip() else None
    )


def get_reasoning_effort() -> str | None:
    """Return effective default reasoning effort (LLM_REASONING_EFFORT env overrides config)."""
    env_val = os.getenv("LLM_REASONING_EFFORT")
    if env_val and env_val.strip():
        return env_val.strip().lower()
    return _DEFAULT_REASONING_EFFORT


def get_token_usage() -> dict[str, int]:
    """Return a copy of the current session's token usage."""
    return dict(_TOKEN_USAGE)


def reset_token_usage() -> None:
    """Reset the session token usage counter to zero."""
    _TOKEN_USAGE["prompt_tokens"] = 0
    _TOKEN_USAGE["completion_tokens"] = 0
    _TOKEN_USAGE["reasoning_tokens"] = 0
    _TOKEN_USAGE["total_calls"] = 0


def calculate_cost_usd(
    prompt_tokens: int,
    completion_tokens: int,
    provider: str = "deepseek",
    model: str | None = None,
) -> float:
    """
    Calculate estimated cost in USD based on provider pricing.
    DeepSeek pricing (as of 2026):
      - deepseek-chat:     $0.14 / 1M prompt, $0.28 / 1M completion
      - deepseek-reasoner: $0.55 / 1M prompt, $2.19 / 1M completion
    """
    if provider == "deepseek":
        if model and "reasoner" in model.lower():
            cost = (prompt_tokens * 0.55 + completion_tokens * 2.19) / 1_000_000.0
        else:
            cost = (prompt_tokens * 0.14 + completion_tokens * 0.28) / 1_000_000.0
        return round(cost, 6)
    # Default fallback: approx GPT-4o-mini pricing ($0.15 / 1M in, $0.60 / 1M out)
    return round((prompt_tokens * 0.15 + completion_tokens * 0.60) / 1_000_000.0, 6)


def extract_json(raw: str) -> Any:
    """
    Extract and parse JSON from an LLM response string robustly.

    Handles:
      - Clean JSON strings
      - Markdown code blocks (```json ... ``` or ``` ... ```)
      - Preamble/postamble commentary text (common with reasoning models)
      - Trailing commas before closing braces/brackets
    """
    if not raw or not raw.strip():
        raise ValueError("Empty response cannot be parsed as JSON")

    text = raw.strip()

    # 1. Direct JSON parse attempt
    try:
        return json.loads(text)
    except Exception:
        pass

    # 2. Markdown fenced code block search (```json ... ``` or ``` ... ```)
    fence_pattern = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.IGNORECASE)
    matches = fence_pattern.findall(text)
    for block in matches:
        b_clean = block.strip()
        try:
            return json.loads(b_clean)
        except Exception:
            cleaned_block = re.sub(r",\s*([\]}])", r"\1", b_clean)
            try:
                return json.loads(cleaned_block)
            except Exception:
                pass

    # 3. Outer bracket search { ... } or [ ... ]
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    first_bracket = text.find("[")
    last_bracket = text.rfind("]")

    candidates = []
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        candidates.append(text[first_brace:last_brace + 1])
    if first_bracket != -1 and last_bracket != -1 and last_bracket > first_bracket:
        candidates.append(text[first_bracket:last_bracket + 1])

    for cand in candidates:
        try:
            return json.loads(cand)
        except Exception:
            cleaned = re.sub(r",\s*([\]}])", r"\1", cand)
            try:
                return json.loads(cleaned)
            except Exception:
                pass

    # 4. Truncation repair: attempt balancing unclosed braces/brackets
    if first_brace != -1:
        sub = text[first_brace:]
        for last_close in [sub.rfind("}"), sub.rfind("]"), sub.rfind('"')]:
            if last_close != -1:
                prefix = sub[:last_close + 1]
                open_braces = prefix.count("{") - prefix.count("}")
                open_brackets = prefix.count("[") - prefix.count("]")
                if open_braces >= 0 and open_brackets >= 0:
                    repair = prefix + ("]" * open_brackets) + ("}" * open_braces)
                    repair = re.sub(r",\s*([\]}])", r"\1", repair)
                    try:
                        return json.loads(repair)
                    except Exception:
                        pass

    raise json.JSONDecodeError("No valid JSON found in response", text, 0)


def get_client(base_url: str | None = None, api_key: str | None = None) -> OpenAI:
    """
    Return an OpenAI-compatible client.

    Priority for credentials:
      1. Explicit arguments (useful for tests)
      2. Environment variables DEEPSEEK_API_KEY / OPENAI_API_KEY
      3. Raise a clear error
    """
    resolved_key = api_key or os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not resolved_key:
        raise EnvironmentError(
            "No API key found. Set DEEPSEEK_API_KEY (or OPENAI_API_KEY) "
            "as an environment variable."
        )
    resolved_url = base_url or os.getenv(
        "LLM_BASE_URL", "https://api.deepseek.com"
    )
    return OpenAI(api_key=resolved_key, base_url=resolved_url)


def chat_completion(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.7,
    max_tokens: int = 2048,
    reasoning_effort: str | None = None,
    **kwargs: Any,
) -> str:
    """
    Send a single chat completion request and return the response text.

    All agents call this helper so that LLM parameters are centralised
    and easy to log / reproduce (important for the final report).
    """
    # Allow global environment override for testing if LLM_MODEL_OVERRIDE is set
    resolved_model = os.getenv("LLM_MODEL_OVERRIDE") or model
    model_lower = resolved_model.lower()
    is_reasoner = (
        any(tok in model_lower for tok in ("reasoner", "v4-pro"))
        or model_lower.startswith(("o1", "o3"))
    )

    effective_effort = reasoning_effort or get_reasoning_effort()
    thinking_active = is_reasoner and effective_effort != "none"

    # For reasoning models, reasoning tokens count against completion limit.
    # Ensure sufficient token headroom (minimum 8192) to prevent output truncation.
    call_max_tokens = max(max_tokens, 8192) if thinking_active else max_tokens

    call_kwargs: dict[str, Any] = {
        "model": resolved_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": call_max_tokens,
        **kwargs,
    }

    if is_reasoner and effective_effort:
        call_kwargs["reasoning_effort"] = effective_effort

    # Only pass temperature if thinking mode is inactive (reasoning models ignore/disallow it)
    if not thinking_active:
        call_kwargs["temperature"] = temperature

    effort_label = f", effort={effective_effort}" if (is_reasoner and effective_effort) else ""
    print(f"   ⏳ [LLM] Calling {resolved_model}{effort_label}...", flush=True)

    t0 = time.time()
    stop_ticker = threading.Event()

    def _heartbeat() -> None:
        while not stop_ticker.wait(5.0):
            elapsed = int(time.time() - t0)
            print(f"   ⏱️  [LLM] Still working on {resolved_model} ({elapsed}s elapsed)...", flush=True)

    ticker = threading.Thread(target=_heartbeat, daemon=True)
    ticker.start()
    try:
        response = client.chat.completions.create(**call_kwargs)
    finally:
        stop_ticker.set()
        ticker.join(timeout=0.2)

    elapsed_s = time.time() - t0

    # Accumulate token usage if returned by the API (guard against MagicMock in tests)
    p_tok = c_tok = r_tok = 0
    if hasattr(response, "usage") and response.usage:
        raw_p = getattr(response.usage, "prompt_tokens", 0)
        raw_c = getattr(response.usage, "completion_tokens", 0)
        details = getattr(response.usage, "completion_tokens_details", None)
        raw_r = getattr(details, "reasoning_tokens", 0) if details is not None else 0

        p_tok = raw_p if isinstance(raw_p, int) else 0
        c_tok = raw_c if isinstance(raw_c, int) else 0
        r_tok = raw_r if isinstance(raw_r, int) else 0

        _TOKEN_USAGE["prompt_tokens"] += p_tok
        _TOKEN_USAGE["completion_tokens"] += c_tok
        _TOKEN_USAGE["reasoning_tokens"] += r_tok
        _TOKEN_USAGE["total_calls"] += 1

    r_info = f", {r_tok:,} reasoning" if r_tok > 0 else ""
    print(f"   ✓ [LLM] {resolved_model} finished in {elapsed_s:.1f}s ({p_tok:,} in / {c_tok:,} out{r_info})", flush=True)

    return response.choices[0].message.content or ""
