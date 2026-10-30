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
    return response.choices[0].message.content or ""
