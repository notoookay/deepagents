"""Unit tests for _build_chatcodex auto-login behavior."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from deepagents._chatgpt_auth import TokenData
from deepagents._chatgpt_model import _build_chatcodex


def _make_tokens(**overrides) -> TokenData:
    base = TokenData(
        access_token="test-access-token",
        refresh_token="test-refresh-token",
        expires_at=time.time() + 3600,
        account_id="org-test",
    )
    base.update(overrides)
    return base


# Patch target for the ChatOpenAI constructor (imported inside _build_chatcodex)
_CHAT_OPENAI = "langchain_openai.ChatOpenAI"


# ---------------------------------------------------------------------------
# Happy path: tokens already present
# ---------------------------------------------------------------------------


class TestBuildChatCodexWithTokens:
    def test_returns_chat_openai_instance(self) -> None:
        tokens = _make_tokens()
        fake_instance = MagicMock()
        with (
            patch("deepagents._chatgpt_auth.load_tokens", return_value=tokens),
            patch("deepagents._chatgpt_auth.refresh_if_needed", return_value=tokens),
            patch(_CHAT_OPENAI, return_value=fake_instance) as MockChatOpenAI,
        ):
            result = _build_chatcodex()

        assert result is fake_instance
        MockChatOpenAI.assert_called_once()

    def test_passes_account_id_header(self) -> None:
        tokens = _make_tokens(account_id="org-abc")
        with (
            patch("deepagents._chatgpt_auth.load_tokens", return_value=tokens),
            patch("deepagents._chatgpt_auth.refresh_if_needed", return_value=tokens),
            patch(_CHAT_OPENAI) as MockChatOpenAI,
        ):
            _build_chatcodex()

        _, kwargs = MockChatOpenAI.call_args
        assert kwargs["default_headers"]["ChatGPT-Account-Id"] == "org-abc"

    def test_no_account_id_header_when_absent(self) -> None:
        tokens = _make_tokens(account_id=None)
        with (
            patch("deepagents._chatgpt_auth.load_tokens", return_value=tokens),
            patch("deepagents._chatgpt_auth.refresh_if_needed", return_value=tokens),
            patch(_CHAT_OPENAI) as MockChatOpenAI,
        ):
            _build_chatcodex()

        _, kwargs = MockChatOpenAI.call_args
        assert "ChatGPT-Account-Id" not in kwargs["default_headers"]

    def test_custom_model_kwarg_forwarded(self) -> None:
        tokens = _make_tokens()
        with (
            patch("deepagents._chatgpt_auth.load_tokens", return_value=tokens),
            patch("deepagents._chatgpt_auth.refresh_if_needed", return_value=tokens),
            patch(_CHAT_OPENAI) as MockChatOpenAI,
        ):
            _build_chatcodex(model="gpt-5.1-codex")

        _, kwargs = MockChatOpenAI.call_args
        assert kwargs["model"] == "gpt-5.1-codex"

    def test_extra_kwargs_forwarded(self) -> None:
        tokens = _make_tokens()
        with (
            patch("deepagents._chatgpt_auth.load_tokens", return_value=tokens),
            patch("deepagents._chatgpt_auth.refresh_if_needed", return_value=tokens),
            patch(_CHAT_OPENAI) as MockChatOpenAI,
        ):
            _build_chatcodex(temperature=0.5, max_tokens=256)

        _, kwargs = MockChatOpenAI.call_args
        assert kwargs["temperature"] == 0.5
        assert kwargs["max_tokens"] == 256

    def test_originator_header_always_set(self) -> None:
        tokens = _make_tokens()
        with (
            patch("deepagents._chatgpt_auth.load_tokens", return_value=tokens),
            patch("deepagents._chatgpt_auth.refresh_if_needed", return_value=tokens),
            patch(_CHAT_OPENAI) as MockChatOpenAI,
        ):
            _build_chatcodex()

        _, kwargs = MockChatOpenAI.call_args
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
            patch(_CHAT_OPENAI),
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
            patch(_CHAT_OPENAI),
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
            patch(_CHAT_OPENAI, return_value=fake_instance),
        ):
            result = _build_chatcodex()

        assert result is fake_instance
