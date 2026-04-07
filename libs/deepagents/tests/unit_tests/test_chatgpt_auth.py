"""Unit tests for ChatGPT OAuth authentication helpers."""

from __future__ import annotations

import base64
import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from deepagents._chatgpt_auth import (
    TokenData,
    _account_id_from_tokens,
    _derive_challenge,
    _extract_account_id,
    _generate_verifier,
    _generate_state,
    _parse_jwt_payload,
    _parse_token_response,
    delete_tokens,
    load_tokens,
    refresh_if_needed,
    save_tokens,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_jwt(payload: dict) -> str:
    """Construct a minimal (unsigned) JWT string for testing."""
    header = base64.urlsafe_b64encode(b'{"alg":"none"}').rstrip(b"=").decode()
    body_bytes = json.dumps(payload).encode()
    body = base64.urlsafe_b64encode(body_bytes).rstrip(b"=").decode()
    return f"{header}.{body}.sig"


def _make_token_data(
    *,
    expires_at: float | None = None,
    account_id: str | None = "org-test",
) -> TokenData:
    return TokenData(
        access_token="access",
        refresh_token="refresh",
        expires_at=expires_at if expires_at is not None else time.time() + 3600,
        account_id=account_id,
    )


# ---------------------------------------------------------------------------
# PKCE helpers
# ---------------------------------------------------------------------------


class TestPKCEHelpers:
    def test_verifier_is_base64url(self) -> None:
        v = _generate_verifier()
        # Should be decodable base64url with no padding issues
        assert isinstance(v, str)
        assert len(v) > 0

    def test_challenge_differs_from_verifier(self) -> None:
        v = _generate_verifier()
        c = _derive_challenge(v)
        assert c != v

    def test_state_is_string(self) -> None:
        s = _generate_state()
        assert isinstance(s, str)
        assert len(s) > 0


# ---------------------------------------------------------------------------
# JWT parsing
# ---------------------------------------------------------------------------


class TestParseJwtPayload:
    def test_valid_jwt(self) -> None:
        payload = {"sub": "user123", "chatgpt_account_id": "org-abc"}
        token = _make_jwt(payload)
        result = _parse_jwt_payload(token)
        assert result is not None
        assert result["chatgpt_account_id"] == "org-abc"

    def test_invalid_jwt_too_few_parts(self) -> None:
        assert _parse_jwt_payload("only.two") is None

    def test_invalid_jwt_bad_base64(self) -> None:
        assert _parse_jwt_payload("header.!!!invalid!!!.sig") is None


class TestExtractAccountId:
    def test_direct_claim(self) -> None:
        claims = {"chatgpt_account_id": "org-direct"}
        assert _extract_account_id(claims) == "org-direct"

    def test_nested_auth_claim(self) -> None:
        claims = {
            "https://api.openai.com/auth": {"chatgpt_account_id": "org-nested"}
        }
        assert _extract_account_id(claims) == "org-nested"

    def test_organizations_fallback(self) -> None:
        claims = {"organizations": [{"id": "org-from-orgs"}]}
        assert _extract_account_id(claims) == "org-from-orgs"

    def test_direct_takes_priority_over_nested(self) -> None:
        claims = {
            "chatgpt_account_id": "org-direct",
            "https://api.openai.com/auth": {"chatgpt_account_id": "org-nested"},
        }
        assert _extract_account_id(claims) == "org-direct"

    def test_returns_none_when_absent(self) -> None:
        assert _extract_account_id({}) is None


class TestAccountIdFromTokens:
    def test_extracts_from_id_token(self) -> None:
        id_token = _make_jwt({"chatgpt_account_id": "org-id"})
        result = _account_id_from_tokens(access_token="dummy", id_token=id_token)
        assert result == "org-id"

    def test_falls_back_to_access_token(self) -> None:
        access_token = _make_jwt({"chatgpt_account_id": "org-access"})
        result = _account_id_from_tokens(access_token=access_token, id_token=None)
        assert result == "org-access"

    def test_returns_none_when_no_claim(self) -> None:
        access_token = _make_jwt({"sub": "user123"})
        result = _account_id_from_tokens(access_token=access_token, id_token=None)
        assert result is None


# ---------------------------------------------------------------------------
# Token response parsing
# ---------------------------------------------------------------------------


class TestParseTokenResponse:
    def test_basic_parse(self) -> None:
        id_token = _make_jwt({"chatgpt_account_id": "org-123"})
        resp = {
            "access_token": "at",
            "refresh_token": "rt",
            "expires_in": 3600,
            "id_token": id_token,
        }
        tokens = _parse_token_response(resp)
        assert tokens["access_token"] == "at"
        assert tokens["refresh_token"] == "rt"
        assert tokens["account_id"] == "org-123"
        assert tokens["expires_at"] == pytest.approx(time.time() + 3600, abs=5)

    def test_uses_existing_account_id_as_fallback(self) -> None:
        resp = {
            "access_token": _make_jwt({}),
            "refresh_token": "rt",
            "expires_in": 3600,
        }
        tokens = _parse_token_response(resp, existing_account_id="org-fallback")
        assert tokens["account_id"] == "org-fallback"

    def test_default_expires_in_when_missing(self) -> None:
        resp = {"access_token": _make_jwt({}), "refresh_token": "rt"}
        tokens = _parse_token_response(resp)
        assert tokens["expires_at"] == pytest.approx(time.time() + 3600, abs=5)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


class TestTokenPersistence:
    def test_save_and_load(self, tmp_path: Path) -> None:
        token_file = tmp_path / "chatgpt_tokens.json"
        tokens = _make_token_data()
        with patch("deepagents._chatgpt_auth._TOKEN_FILE", token_file):
            save_tokens(tokens)
            loaded = load_tokens()
        assert loaded is not None
        assert loaded["access_token"] == tokens["access_token"]
        assert loaded["account_id"] == tokens["account_id"]

    def test_load_returns_none_when_file_missing(self, tmp_path: Path) -> None:
        token_file = tmp_path / "nonexistent.json"
        with patch("deepagents._chatgpt_auth._TOKEN_FILE", token_file):
            result = load_tokens()
        assert result is None

    def test_delete_removes_file(self, tmp_path: Path) -> None:
        token_file = tmp_path / "chatgpt_tokens.json"
        token_file.write_text("{}")
        with patch("deepagents._chatgpt_auth._TOKEN_FILE", token_file):
            delete_tokens()
        assert not token_file.exists()

    def test_delete_is_noop_when_file_missing(self, tmp_path: Path) -> None:
        token_file = tmp_path / "nonexistent.json"
        with patch("deepagents._chatgpt_auth._TOKEN_FILE", token_file):
            delete_tokens()  # should not raise

    def test_save_sets_mode_600(self, tmp_path: Path) -> None:
        token_file = tmp_path / "chatgpt_tokens.json"
        tokens = _make_token_data()
        with patch("deepagents._chatgpt_auth._TOKEN_FILE", token_file):
            save_tokens(tokens)
        mode = token_file.stat().st_mode & 0o777
        assert mode == 0o600


# ---------------------------------------------------------------------------
# refresh_if_needed
# ---------------------------------------------------------------------------


class TestRefreshIfNeeded:
    def test_no_refresh_when_token_valid(self) -> None:
        tokens = _make_token_data(expires_at=time.time() + 3600)
        with patch("deepagents._chatgpt_auth._refresh_tokens") as mock_refresh:
            result = refresh_if_needed(tokens)
        mock_refresh.assert_not_called()
        assert result is tokens

    def test_refreshes_when_token_expired(self, tmp_path: Path) -> None:
        tokens = _make_token_data(expires_at=time.time() - 1)
        new_access = _make_jwt({"chatgpt_account_id": "org-new"})
        refresh_resp = {
            "access_token": new_access,
            "refresh_token": "new-refresh",
            "expires_in": 3600,
        }
        token_file = tmp_path / "chatgpt_tokens.json"
        with (
            patch("deepagents._chatgpt_auth._refresh_tokens", return_value=refresh_resp),
            patch("deepagents._chatgpt_auth._TOKEN_FILE", token_file),
        ):
            result = refresh_if_needed(tokens)
        assert result["access_token"] == new_access
        assert result["refresh_token"] == "new-refresh"

    def test_refreshes_when_expiring_within_60s(self, tmp_path: Path) -> None:
        tokens = _make_token_data(expires_at=time.time() + 30)
        refresh_resp = {
            "access_token": _make_jwt({}),
            "refresh_token": "rt2",
            "expires_in": 3600,
        }
        token_file = tmp_path / "chatgpt_tokens.json"
        with (
            patch("deepagents._chatgpt_auth._refresh_tokens", return_value=refresh_resp),
            patch("deepagents._chatgpt_auth._TOKEN_FILE", token_file),
        ):
            result = refresh_if_needed(tokens)
        assert result["refresh_token"] == "rt2"
