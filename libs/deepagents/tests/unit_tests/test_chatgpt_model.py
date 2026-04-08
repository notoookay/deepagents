"""Unit tests for _build_chatcodex auto-login behavior and ChatCodex."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from deepagents._chatgpt_auth import TokenData
from deepagents._chatgpt_model import ChatCodex, _build_chatcodex


def _make_tokens(**overrides) -> TokenData:
    base = TokenData(
        access_token="test-access-token",
        refresh_token="test-refresh-token",
        expires_at=time.time() + 3600,
        account_id="org-test",
    )
    base.update(overrides)
    return base


# Patch target for the ChatCodex constructor (imported inside _build_chatcodex)
_CHAT_CODEX = "deepagents._chatgpt_model.ChatCodex"


# ---------------------------------------------------------------------------
# Happy path: tokens already present
# ---------------------------------------------------------------------------


class TestBuildChatCodexWithTokens:
    def test_returns_chat_codex_instance(self) -> None:
        tokens = _make_tokens()
        fake_instance = MagicMock()
        with (
            patch("deepagents._chatgpt_auth.load_tokens", return_value=tokens),
            patch("deepagents._chatgpt_auth.refresh_if_needed", return_value=tokens),
            patch(_CHAT_CODEX, return_value=fake_instance) as MockChatCodex,
        ):
            result = _build_chatcodex()

        assert result is fake_instance
        MockChatCodex.assert_called_once()

    def test_passes_account_id_header(self) -> None:
        tokens = _make_tokens(account_id="org-abc")
        with (
            patch("deepagents._chatgpt_auth.load_tokens", return_value=tokens),
            patch("deepagents._chatgpt_auth.refresh_if_needed", return_value=tokens),
            patch(_CHAT_CODEX) as MockChatCodex,
        ):
            _build_chatcodex()

        _, kwargs = MockChatCodex.call_args
        assert kwargs["default_headers"]["ChatGPT-Account-Id"] == "org-abc"

    def test_no_account_id_header_when_absent(self) -> None:
        tokens = _make_tokens(account_id=None)
        with (
            patch("deepagents._chatgpt_auth.load_tokens", return_value=tokens),
            patch("deepagents._chatgpt_auth.refresh_if_needed", return_value=tokens),
            patch(_CHAT_CODEX) as MockChatCodex,
        ):
            _build_chatcodex()

        _, kwargs = MockChatCodex.call_args
        assert "ChatGPT-Account-Id" not in kwargs["default_headers"]

    def test_custom_model_kwarg_forwarded(self) -> None:
        tokens = _make_tokens()
        with (
            patch("deepagents._chatgpt_auth.load_tokens", return_value=tokens),
            patch("deepagents._chatgpt_auth.refresh_if_needed", return_value=tokens),
            patch(_CHAT_CODEX) as MockChatCodex,
        ):
            _build_chatcodex(model="gpt-5.1-codex")

        _, kwargs = MockChatCodex.call_args
        assert kwargs["model"] == "gpt-5.1-codex"

    def test_extra_kwargs_forwarded(self) -> None:
        tokens = _make_tokens()
        with (
            patch("deepagents._chatgpt_auth.load_tokens", return_value=tokens),
            patch("deepagents._chatgpt_auth.refresh_if_needed", return_value=tokens),
            patch(_CHAT_CODEX) as MockChatCodex,
        ):
            _build_chatcodex(temperature=0.5, max_tokens=256)

        _, kwargs = MockChatCodex.call_args
        assert kwargs["temperature"] == 0.5
        assert kwargs["max_tokens"] == 256

    def test_originator_header_always_set(self) -> None:
        tokens = _make_tokens()
        with (
            patch("deepagents._chatgpt_auth.load_tokens", return_value=tokens),
            patch("deepagents._chatgpt_auth.refresh_if_needed", return_value=tokens),
            patch(_CHAT_CODEX) as MockChatCodex,
        ):
            _build_chatcodex()

        _, kwargs = MockChatCodex.call_args
        assert kwargs["default_headers"]["originator"] == "deepagents"


# ---------------------------------------------------------------------------
# No tokens — non-interactive (raises ValueError)
# ---------------------------------------------------------------------------


class TestBuildChatCodexNotLoggedIn:
    def test_raises_when_not_tty(self) -> None:
        with (
            patch("deepagents._chatgpt_auth.load_tokens", return_value=None),
            patch("sys.stdin.isatty", return_value=False),
            patch("sys.stdout.isatty", return_value=False),
        ):
            with pytest.raises(ValueError, match="Not logged in"):
                _build_chatcodex()

    def test_raises_when_only_stdin_is_tty(self) -> None:
        """Both stdin AND stdout must be TTYs to attempt login."""
        with (
            patch("deepagents._chatgpt_auth.load_tokens", return_value=None),
            patch("sys.stdin.isatty", return_value=True),
            patch("sys.stdout.isatty", return_value=False),
        ):
            with pytest.raises(ValueError, match="Not logged in"):
                _build_chatcodex()


# ---------------------------------------------------------------------------
# No tokens — interactive TTY: triggers browser login
# ---------------------------------------------------------------------------


class TestBuildChatCodexAutoLogin:
    def test_browser_login_triggered_on_tty(self) -> None:
        tokens = _make_tokens()
        with (
            patch("deepagents._chatgpt_auth.load_tokens", return_value=None),
            patch("deepagents._chatgpt_auth.refresh_if_needed", return_value=tokens),
            patch("sys.stdin.isatty", return_value=True),
            patch("sys.stdout.isatty", return_value=True),
            patch("deepagents._chatgpt_auth.login_browser", return_value=tokens) as mock_browser,
            patch("deepagents._chatgpt_auth.login_device") as mock_device,
            patch(_CHAT_CODEX),
        ):
            _build_chatcodex()

        mock_browser.assert_called_once()
        mock_device.assert_not_called()

    def test_device_login_fallback_when_browser_fails(self) -> None:
        tokens = _make_tokens()
        with (
            patch("deepagents._chatgpt_auth.load_tokens", return_value=None),
            patch("deepagents._chatgpt_auth.refresh_if_needed", return_value=tokens),
            patch("sys.stdin.isatty", return_value=True),
            patch("sys.stdout.isatty", return_value=True),
            patch("deepagents._chatgpt_auth.login_browser", side_effect=RuntimeError("no browser")),
            patch("deepagents._chatgpt_auth.login_device", return_value=tokens) as mock_device,
            patch(_CHAT_CODEX),
        ):
            _build_chatcodex()

        mock_device.assert_called_once()

    def test_returns_model_after_auto_login(self) -> None:
        tokens = _make_tokens()
        fake_instance = MagicMock()
        with (
            patch("deepagents._chatgpt_auth.load_tokens", return_value=None),
            patch("deepagents._chatgpt_auth.refresh_if_needed", return_value=tokens),
            patch("sys.stdin.isatty", return_value=True),
            patch("sys.stdout.isatty", return_value=True),
            patch("deepagents._chatgpt_auth.login_browser", return_value=tokens),
            patch(_CHAT_CODEX, return_value=fake_instance),
        ):
            result = _build_chatcodex()

        assert result is fake_instance


# ---------------------------------------------------------------------------
# ChatCodex: system → developer role conversion
# ---------------------------------------------------------------------------


class TestChatCodexInstructions:
    """Verify that ChatCodex extracts system messages into ``instructions``."""

    def _make_model(self) -> ChatCodex:
        return ChatCodex(
            model="gpt-5.3-codex",
            api_key="fake-key",  # type: ignore[arg-type]
        )

    def test_system_message_extracted_to_instructions(self) -> None:
        model = self._make_model()
        messages = [
            SystemMessage(content="You are helpful."),
            HumanMessage(content="Hello"),
        ]
        payload = model._get_request_payload(messages)
        assert payload["instructions"] == "You are helpful."
        # system/developer items should be removed from input
        for item in payload["input"]:
            if isinstance(item, dict):
                assert item.get("role") not in ("system", "developer")

    def test_user_role_unchanged(self) -> None:
        model = self._make_model()
        messages = [
            SystemMessage(content="You are helpful."),
            HumanMessage(content="Hello"),
        ]
        payload = model._get_request_payload(messages)
        user_items = [
            item for item in payload["input"]
            if isinstance(item, dict) and item.get("role") == "user"
        ]
        assert len(user_items) == 1

    def test_no_system_message_still_works(self) -> None:
        model = self._make_model()
        messages = [HumanMessage(content="Hello")]
        payload = model._get_request_payload(messages)
        assert "input" in payload
        assert "instructions" not in payload

    def test_multiple_system_messages_joined(self) -> None:
        model = self._make_model()
        messages = [
            SystemMessage(content="First instruction."),
            SystemMessage(content="Second instruction."),
            HumanMessage(content="Hello"),
        ]
        payload = model._get_request_payload(messages)
        assert "First instruction." in payload["instructions"]
        assert "Second instruction." in payload["instructions"]
