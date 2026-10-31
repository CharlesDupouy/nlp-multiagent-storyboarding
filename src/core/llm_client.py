"""
LLM Client — thin wrapper around the OpenAI-compatible API.

DeepSeek's API is fully OpenAI-compatible, so we simply point the
openai client at DeepSeek's base URL. To switch to OpenAI GPT-4o,
change the base_url and model in config.yaml.
"""

from __future__ import annotations

import os
from typing import Any

from openai import OpenAI

# Load .env file automatically if present (never committed to git)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Global token accounting for reproducibility and cost auditing
_TOKEN_USAGE = {
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_calls": 0,
}


def get_token_usage() -> dict[str, int]:
    """Return a copy of the current session's token usage."""
    return dict(_TOKEN_USAGE)


def reset_token_usage() -> None:
    """Reset the session token usage counter to zero."""
    _TOKEN_USAGE["prompt_tokens"] = 0
    _TOKEN_USAGE["completion_tokens"] = 0
    _TOKEN_USAGE["total_calls"] = 0


def calculate_cost_usd(prompt_tokens: int, completion_tokens: int, provider: str = "deepseek") -> float:
    """
    Calculate estimated cost in USD based on provider pricing.
    DeepSeek pricing (as of 2026):
      - Cache miss input: $0.14 / 1M tokens
      - Output tokens:     $0.28 / 1M tokens
    """
    if provider == "deepseek":
        cost = (prompt_tokens * 0.14 + completion_tokens * 0.28) / 1_000_000.0
        return round(cost, 6)
    # Default fallback: approx GPT-4o-mini pricing ($0.15 / 1M in, $0.60 / 1M out)
    return round((prompt_tokens * 0.15 + completion_tokens * 0.60) / 1_000_000.0, 6)


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
    **kwargs: Any,
) -> str:
    """
    Send a single chat completion request and return the response text.

    All agents call this helper so that LLM parameters are centralised
    and easy to log / reproduce (important for the final report).
    """
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs,
    )

    # Accumulate token usage if returned by the API
    if hasattr(response, "usage") and response.usage:
        _TOKEN_USAGE["prompt_tokens"] += getattr(response.usage, "prompt_tokens", 0) or 0
        _TOKEN_USAGE["completion_tokens"] += getattr(response.usage, "completion_tokens", 0) or 0
        _TOKEN_USAGE["total_calls"] += 1

    return response.choices[0].message.content or ""
