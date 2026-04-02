"""LLM Client placeholder.

Drop-in integration point for any LLM provider.  The ``call_llm``
function accepts a prompt and returns a completion string.  When no
API key is configured it returns the raw prompt back — perfect for
testing the rest of the pipeline without incurring API costs.
"""

from __future__ import annotations

import os

from app.core.logger import logger


# Set to a real key via environment variable when ready.
_LLM_API_KEY: str | None = os.getenv("LLM_API_KEY")


def call_llm(prompt: str, *, max_tokens: int = 256) -> str:
    """Send *prompt* to the configured LLM and return the response.

    Falls back to echoing the prompt when no API key is set so the
    demo still works end-to-end.
    """
    if _LLM_API_KEY is None:
        logger.warning(
            "LLM_API_KEY not set — returning prompt as placeholder response"
        )
        return f"[LLM placeholder] {prompt}"

    # ── Future: real API call goes here ────────────────────────────
    # Example with OpenAI:
    #   import openai
    #   openai.api_key = _LLM_API_KEY
    #   response = openai.ChatCompletion.create(
    #       model="gpt-3.5-turbo",
    #       messages=[{"role": "user", "content": prompt}],
    #       max_tokens=max_tokens,
    #   )
    #   return response.choices[0].message.content

    logger.info("LLM call with %d-char prompt (max_tokens=%d)", len(prompt), max_tokens)
    return f"[LLM placeholder] {prompt}"
