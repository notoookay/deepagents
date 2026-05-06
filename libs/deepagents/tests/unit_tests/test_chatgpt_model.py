"""Unit tests for _build_chatcodex auto-login behavior and ChatCodex."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from deepagents._chatgpt_auth import TokenData
from deepagents._chatgpt_model import (
    CHATGPT_MODELS,
    DEFAULT_CHATGPT_MODEL,
    ChatCodex,
    _build_chatcodex,
    _is_deprecated_chatgpt_model,
)


def _make_tokens(**overrides: object) -> TokenData:
    base = TokenData(
        access_token="test-access-token",  # noqa: S106
        refresh_token="test-refresh-token",  # noqa: S106
        expires_at=time.time() + 3600,
        account_id="org-test",
    )
    base.update(overrides)  # type: ignore[typeddict-item]
    return base


# Patch targets — functions are imported into _chatgpt_model at module load,
# so patches must target the name in that namespace.
_CHAT_CODEX = "deepagents._chatgpt_model.ChatCodex"
_LOAD_TOKENS = "deepagents._chatgpt_model.load_tokens"
_REFRESH = "deepagents._chatgpt_model.refresh_if_needed"
_LOGIN_BROWSER = "deepagents._chatgpt_model.login_browser"
_LOGIN_DEVICE = "deepagents._chatgpt_model.login_device"


# ---------------------------------------------------------------------------
# Happy path: tokens already present
# ---------------------------------------------------------------------------


class TestBuildChatCodexWithTokens:
    def test_returns_chat_codex_instance(self) -> None:
        tokens = _make_tokens()
        fake_instance = MagicMock()
        with (
            patch(_LOAD_TOKENS, return_value=tokens),
            patch(_REFRESH, return_value=tokens),
            patch(_CHAT_CODEX, return_value=fake_instance) as mock_chat_codex,
        ):
            result = _build_chatcodex()

        assert result is fake_instance
        mock_chat_codex.assert_called_once()

    def test_passes_account_id_header(self) -> None:
        tokens = _make_tokens(account_id="org-abc")
        with (
            patch(_LOAD_TOKENS, return_value=tokens),
            patch(_REFRESH, return_value=tokens),
            patch(_CHAT_CODEX) as mock_chat_codex,
        ):
            _build_chatcodex()

        _, kwargs = mock_chat_codex.call_args
        assert kwargs["default_headers"]["ChatGPT-Account-Id"] == "org-abc"

    def test_no_account_id_header_when_absent(self) -> None:
        tokens = _make_tokens(account_id=None)
        with (
            patch(_LOAD_TOKENS, return_value=tokens),
            patch(_REFRESH, return_value=tokens),
            patch(_CHAT_CODEX) as mock_chat_codex,
        ):
            _build_chatcodex()

        _, kwargs = mock_chat_codex.call_args
        assert "ChatGPT-Account-Id" not in kwargs["default_headers"]

    def test_custom_model_kwarg_forwarded(self) -> None:
        tokens = _make_tokens()
        with (
            patch(_LOAD_TOKENS, return_value=tokens),
            patch(_REFRESH, return_value=tokens),
            patch(_CHAT_CODEX) as mock_chat_codex,
        ):
            _build_chatcodex(model="gpt-5.4-mini")

        _, kwargs = mock_chat_codex.call_args
        assert kwargs["model"] == "gpt-5.4-mini"

    def test_default_model_used_when_omitted(self) -> None:
        tokens = _make_tokens()
        with (
            patch(_LOAD_TOKENS, return_value=tokens),
            patch(_REFRESH, return_value=tokens),
            patch(_CHAT_CODEX) as mock_chat_codex,
        ):
            _build_chatcodex()

        _, kwargs = mock_chat_codex.call_args
        assert kwargs["model"] == DEFAULT_CHATGPT_MODEL

    def test_extra_kwargs_forwarded(self) -> None:
        tokens = _make_tokens()
        with (
            patch(_LOAD_TOKENS, return_value=tokens),
            patch(_REFRESH, return_value=tokens),
            patch(_CHAT_CODEX) as mock_chat_codex,
        ):
            _build_chatcodex(temperature=0.5, max_tokens=256)

        _, kwargs = mock_chat_codex.call_args
        assert kwargs["temperature"] == 0.5
        assert kwargs["max_tokens"] == 256

    def test_originator_header_always_set(self) -> None:
        tokens = _make_tokens()
        with (
            patch(_LOAD_TOKENS, return_value=tokens),
            patch(_REFRESH, return_value=tokens),
            patch(_CHAT_CODEX) as mock_chat_codex,
        ):
            _build_chatcodex()

        _, kwargs = mock_chat_codex.call_args
        assert kwargs["default_headers"]["originator"] == "deepagents"


# ---------------------------------------------------------------------------
# No tokens — non-interactive (raises ValueError)
# ---------------------------------------------------------------------------


class TestBuildChatCodexNotLoggedIn:
    def test_raises_when_not_tty(self) -> None:
        with (
            patch(_LOAD_TOKENS, return_value=None),
            patch("sys.stdin.isatty", return_value=False),
            patch("sys.stdout.isatty", return_value=False),
            pytest.raises(ValueError, match="Not logged in"),
        ):
            _build_chatcodex()

    def test_raises_when_only_stdin_is_tty(self) -> None:
        """Both stdin AND stdout must be TTYs to attempt login."""
        with (
            patch(_LOAD_TOKENS, return_value=None),
            patch("sys.stdin.isatty", return_value=True),
            patch("sys.stdout.isatty", return_value=False),
            pytest.raises(ValueError, match="Not logged in"),
        ):
            _build_chatcodex()


# ---------------------------------------------------------------------------
# No tokens — interactive TTY: triggers browser login
# ---------------------------------------------------------------------------


class TestBuildChatCodexAutoLogin:
    def test_browser_login_triggered_on_tty(self) -> None:
        tokens = _make_tokens()
        with (
            patch(_LOAD_TOKENS, return_value=None),
            patch(_REFRESH, return_value=tokens),
            patch("sys.stdin.isatty", return_value=True),
            patch("sys.stdout.isatty", return_value=True),
            patch(_LOGIN_BROWSER, return_value=tokens) as mock_browser,
            patch(_LOGIN_DEVICE) as mock_device,
            patch(_CHAT_CODEX),
        ):
            _build_chatcodex()

        mock_browser.assert_called_once()
        mock_device.assert_not_called()

    def test_device_login_fallback_when_browser_fails(self) -> None:
        tokens = _make_tokens()
        with (
            patch(_LOAD_TOKENS, return_value=None),
            patch(_REFRESH, return_value=tokens),
            patch("sys.stdin.isatty", return_value=True),
            patch("sys.stdout.isatty", return_value=True),
            patch(_LOGIN_BROWSER, side_effect=RuntimeError("no browser")),
            patch(_LOGIN_DEVICE, return_value=tokens) as mock_device,
            patch(_CHAT_CODEX),
        ):
            _build_chatcodex()

        mock_device.assert_called_once()

    def test_returns_model_after_auto_login(self) -> None:
        tokens = _make_tokens()
        fake_instance = MagicMock()
        with (
            patch(_LOAD_TOKENS, return_value=None),
            patch(_REFRESH, return_value=tokens),
            patch("sys.stdin.isatty", return_value=True),
            patch("sys.stdout.isatty", return_value=True),
            patch(_LOGIN_BROWSER, return_value=tokens),
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
        user_items = [item for item in payload["input"] if isinstance(item, dict) and item.get("role") == "user"]
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


# ---------------------------------------------------------------------------
# Supported model list + deprecation gate
# ---------------------------------------------------------------------------


class TestChatGPTModelList:
    """Verify the hand-maintained supported-model list."""

    def test_default_is_in_supported_list(self) -> None:
        assert DEFAULT_CHATGPT_MODEL in CHATGPT_MODELS

    def test_known_models_present(self) -> None:
        # Spot-check the models expected from the upstream Codex catalog.
        for expected in ("gpt-5.4", "gpt-5.4-mini", "gpt-5.3-codex"):
            assert expected in CHATGPT_MODELS

    def test_deprecated_old_gpt5_families(self) -> None:
        for deprecated in (
            "gpt-5.2",
            "gpt-5.2-codex",
            "gpt-5.1-codex",
            "gpt-5.0",
            "gpt-5-mini",
            "gpt-4o",
        ):
            assert _is_deprecated_chatgpt_model(deprecated), f"{deprecated} should be flagged as deprecated"

    def test_supported_models_not_deprecated(self) -> None:
        for model in CHATGPT_MODELS:
            assert not _is_deprecated_chatgpt_model(model)

    def test_unknown_forward_compat_not_deprecated(self) -> None:
        # Unknown but plausible future IDs should pass the deprecation gate
        # (they'll trigger a warning, not an error).
        assert not _is_deprecated_chatgpt_model("gpt-5.5")
        assert not _is_deprecated_chatgpt_model("gpt-6")


class TestChatCodexRefreshCredentials:
    """``ChatCodex`` must refresh its OAuth token before each request."""

    def _make_model(self) -> ChatCodex:
        return ChatCodex(
            model="gpt-5.3-codex",
            api_key="stale-token",  # type: ignore[arg-type]
        )

    def test_generate_refreshes_token_on_underlying_clients(self) -> None:
        fresh = _make_tokens(access_token="fresh-token")  # noqa: S106
        model = self._make_model()
        with (
            patch(_LOAD_TOKENS, return_value=fresh),
            patch(_REFRESH, return_value=fresh) as mock_refresh,
            patch.object(ChatCodex, "_get_request_payload"),
            patch(
                "langchain_openai.ChatOpenAI._generate",
                return_value=MagicMock(generations=[]),
            ),
        ):
            model._generate([HumanMessage(content="hi")])

        mock_refresh.assert_called_once()
        assert model.root_client.api_key == "fresh-token"
        assert model.root_async_client.api_key == "fresh-token"

    def test_refresh_noop_when_no_tokens_on_disk(self) -> None:
        """If tokens can't be loaded, skip refresh rather than crash.

        This keeps ``ChatCodex`` usable in tests that construct it with a
        fake api_key without a tokens file on disk.
        """
        model = self._make_model()
        original_key = model.root_client.api_key
        with (
            patch(_LOAD_TOKENS, return_value=None),
            patch(_REFRESH) as mock_refresh,
        ):
            model._refresh_credentials()

        mock_refresh.assert_not_called()
        assert model.root_client.api_key == original_key

    def test_refresh_propagates_between_successive_calls(self) -> None:
        """A second call after ``expires_at`` passes should trigger a refresh."""
        expired = _make_tokens(access_token="old", expires_at=time.time() - 1)  # noqa: S106
        fresh = _make_tokens(access_token="new")  # noqa: S106
        model = self._make_model()
        with (
            patch(_LOAD_TOKENS, return_value=expired),
            patch(_REFRESH, return_value=fresh) as mock_refresh,
        ):
            model._refresh_credentials()
            model._refresh_credentials()

        assert mock_refresh.call_count == 2
        assert model.root_client.api_key == "new"


class TestBuildChatCodexDeprecationGate:
    """``_build_chatcodex`` must reject deprecated models up-front."""

    def test_raises_on_deprecated_model(self) -> None:
        tokens = _make_tokens()
        with (
            patch(_LOAD_TOKENS, return_value=tokens),
            patch(_REFRESH, return_value=tokens),
            patch(_CHAT_CODEX),
            pytest.raises(ValueError, match="deprecated"),
        ):
            _build_chatcodex(model="gpt-5.2-codex")

    def test_deprecated_model_error_lists_alternatives(self) -> None:
        tokens = _make_tokens()
        with (
            patch(_LOAD_TOKENS, return_value=tokens),
            patch(_REFRESH, return_value=tokens),
            patch(_CHAT_CODEX),
            pytest.raises(ValueError, match="deprecated") as excinfo,
        ):
            _build_chatcodex(model="gpt-5.1-codex")
        # Error message should surface the supported alternatives.
        for model in CHATGPT_MODELS:
            assert model in str(excinfo.value)

    def test_unknown_model_warns_but_proceeds(self, caplog) -> None:
        tokens = _make_tokens()
        with (
            patch(_LOAD_TOKENS, return_value=tokens),
            patch(_REFRESH, return_value=tokens),
            patch(_CHAT_CODEX) as mock_chat_codex,
            caplog.at_level("WARNING", logger="deepagents._chatgpt_model"),
        ):
            _build_chatcodex(model="gpt-6-codex")

        mock_chat_codex.assert_called_once()
        assert any("forward-compat" in rec.message for rec in caplog.records)
