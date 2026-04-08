"""LangChain-compatible chat model backed by the ChatGPT Codex API.

Uses ChatGPT Plus/Pro OAuth tokens instead of an API key. The model
forwards requests to ``https://chatgpt.com/backend-api/codex/responses``
with the required ``Authorization`` and ``ChatGPT-Account-Id`` headers.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "gpt-5.3-codex"


def _build_chatcodex(**kwargs: Any):  # -> ChatOpenAI
    """Instantiate a ``ChatOpenAI`` wired to the Codex endpoint with OAuth tokens.

    Loads (and refreshes if needed) stored OAuth tokens, then constructs a
    ``ChatOpenAI`` instance whose ``base_url`` points to the Codex backend so
    that all requests are routed to
    ``https://chatgpt.com/backend-api/codex/responses``.

    The model is returned as a plain ``ChatOpenAI``; callers do not need to
    import this module's internal types.

    Args:
        **kwargs: Extra keyword arguments forwarded to ``ChatOpenAI``
            (e.g. ``temperature``, ``max_tokens``, ``streaming``).  ``model``,
            ``api_key``, ``base_url``, and ``default_headers`` are set
            internally and should not be passed.

    Returns:
        A ``ChatOpenAI`` instance ready to make Codex API calls.

    Raises:
        ValueError: If no stored tokens are found (user has not logged in).
    """
    from deepagents._chatgpt_auth import (
        CODEX_API_BASE,
        load_tokens,
        refresh_if_needed,
    )
    from langchain_openai import ChatOpenAI

    tokens = load_tokens()
    if tokens is None:
        import sys

        from deepagents._chatgpt_auth import login_browser, login_device

        if sys.stdin.isatty() and sys.stdout.isatty():
            print("Not logged in to ChatGPT. Starting login flow...")
            try:
                tokens = login_browser()
            except Exception:
                tokens = login_device()
        else:
            msg = "Not logged in to ChatGPT. Run: deep-agents login openai"
            raise ValueError(msg)

    tokens = refresh_if_needed(tokens)

    model_name: str = kwargs.pop("model", _DEFAULT_MODEL)
    default_headers: dict[str, str] = {"originator": "deepagents"}
    account_id = tokens.get("account_id")
    if account_id:
        default_headers["ChatGPT-Account-Id"] = account_id

    return ChatOpenAI(
        model=model_name,
        api_key=tokens["access_token"],  # type: ignore[arg-type]
        base_url=CODEX_API_BASE,
        default_headers=default_headers,
        **kwargs,
    )
