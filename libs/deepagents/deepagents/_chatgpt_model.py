"""LangChain-compatible chat model backed by the ChatGPT Codex API.

Uses ChatGPT Plus/Pro OAuth tokens instead of an API key. The model
forwards requests to ``https://chatgpt.com/backend-api/codex/responses``
with the required ``Authorization`` and ``ChatGPT-Account-Id`` headers.
"""

from __future__ import annotations

import logging
from typing import Any

from collections.abc import AsyncIterator

from langchain_core.language_models.chat_models import AsyncCallbackManagerForLLMRun
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "gpt-5.3-codex"


def _flatten_content(content: Any) -> str:
    """Collapse a Responses API content-block list into a plain string.

    The Codex Responses API returns content as a list of typed blocks, e.g.
    ``[{"type": "text", "text": "Hello", ...}]``.  This helper extracts and
    joins all text parts so callers receive a normal string.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") in ("text", "output_text"):
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    return str(content)


class ChatCodex(ChatOpenAI):
    """``ChatOpenAI`` subclass tailored for the ChatGPT Codex Responses API.

    Differences from vanilla ``ChatOpenAI``:

    * Moves system/developer messages out of ``input`` into the required
      ``instructions`` top-level field.
    * Normalises the response so ``AIMessage.content`` is always a plain
      string instead of a list of Responses API content blocks.
    """

    def _get_request_payload(
        self,
        input_: Any,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> dict:
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        # The Codex Responses API requires the ``instructions`` top-level
        # field.  Upstream ``ChatOpenAI`` places system messages into the
        # ``input`` array instead.  Extract them and set ``instructions``.
        if isinstance(payload.get("input"), list):
            system_texts: list[str] = []
            remaining: list[dict] = []
            for item in payload["input"]:
                if isinstance(item, dict) and item.get("role") in (
                    "system",
                    "developer",
                ):
                    content = item.get("content", "")
                    if isinstance(content, str):
                        system_texts.append(content)
                    elif isinstance(content, list):
                        # Content-block format: extract text parts.
                        for block in content:
                            if isinstance(block, dict) and block.get("type") in (
                                "text",
                                "input_text",
                            ):
                                system_texts.append(
                                    block.get("text", "")
                                )
                            elif isinstance(block, str):
                                system_texts.append(block)
                else:
                    remaining.append(item)
            if system_texts:
                payload["instructions"] = "\n\n".join(system_texts)
                payload["input"] = remaining
        return payload

    async def _agenerate(
        self,
        messages: list,
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        result = await super()._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)
        return _normalize_result(result)

    async def _astream(
        self,
        messages: list,
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        async for chunk in super()._astream(messages, stop=stop, run_manager=run_manager, **kwargs):
            if isinstance(chunk.message, AIMessageChunk):
                chunk.message.content = _flatten_content(chunk.message.content)
            yield chunk


def _normalize_result(result: ChatResult) -> ChatResult:
    """Flatten Responses API content-block lists to plain strings."""
    for gen in result.generations:
        if isinstance(gen, ChatGeneration) and isinstance(gen.message, AIMessage):
            gen.message.content = _flatten_content(gen.message.content)
    return result


def _build_chatcodex(**kwargs: Any):  # -> ChatCodex
    """Instantiate a ``ChatCodex`` wired to the Codex endpoint with OAuth tokens.

    Loads (and refreshes if needed) stored OAuth tokens, then constructs a
    ``ChatCodex`` instance whose ``base_url`` points to the Codex backend so
    that all requests are routed to
    ``https://chatgpt.com/backend-api/codex/responses``.

    Args:
        **kwargs: Extra keyword arguments forwarded to ``ChatCodex``
            (e.g. ``temperature``, ``max_tokens``, ``streaming``).  ``model``,
            ``api_key``, ``base_url``, and ``default_headers`` are set
            internally and should not be passed.

    Returns:
        A ``ChatCodex`` instance ready to make Codex API calls.

    Raises:
        ValueError: If no stored tokens are found (user has not logged in).
    """
    from deepagents._chatgpt_auth import (
        CODEX_API_BASE,
        load_tokens,
        refresh_if_needed,
    )

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

    return ChatCodex(
        model=model_name,
        api_key=tokens["access_token"],  # type: ignore[arg-type]
        base_url=CODEX_API_BASE,
        default_headers=default_headers,
        store=False,
        streaming=True,
        **kwargs,
    )
