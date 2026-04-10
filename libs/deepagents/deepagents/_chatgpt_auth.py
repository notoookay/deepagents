"""OpenAI OAuth PKCE authentication for ChatGPT subscription (Plus/Pro).

Provides browser-redirect and device-code (headless) flows, plus token
persistence to ``~/.deepagents/chatgpt_tokens.json``.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import secrets
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, TypedDict

logger = logging.getLogger(__name__)

CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
ISSUER = "https://auth.openai.com"
CODEX_API_BASE = "https://chatgpt.com/backend-api/codex"
OAUTH_PORT = 1455

_TOKEN_FILE = Path.home() / ".deepagents" / "chatgpt_tokens.json"


class TokenData(TypedDict):
    access_token: str
    refresh_token: str
    expires_at: float  # Unix timestamp
    account_id: str | None


# ---------------------------------------------------------------------------
# PKCE helpers
# ---------------------------------------------------------------------------


def _generate_verifier() -> str:
    return base64.urlsafe_b64encode(secrets.token_bytes(32)).rstrip(b"=").decode()


def _derive_challenge(verifier: str) -> str:
    digest = hashlib.sha256(verifier.encode()).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode()


def _generate_state() -> str:
    return base64.urlsafe_b64encode(secrets.token_bytes(24)).rstrip(b"=").decode()


# ---------------------------------------------------------------------------
# JWT helpers
# ---------------------------------------------------------------------------


def _parse_jwt_payload(token: str) -> dict | None:
    parts = token.split(".")
    if len(parts) != 3:  # noqa: PLR2004
        return None
    try:
        padded = parts[1] + "=" * (-len(parts[1]) % 4)
        return json.loads(base64.urlsafe_b64decode(padded))
    except (ValueError, json.JSONDecodeError):
        return None


def _extract_account_id(claims: dict) -> str | None:
    return (
        claims.get("chatgpt_account_id")
        or (claims.get("https://api.openai.com/auth") or {}).get("chatgpt_account_id")
        or ((claims.get("organizations") or [{}])[0].get("id"))
    )


def _account_id_from_tokens(access_token: str, id_token: str | None) -> str | None:
    if id_token:
        claims = _parse_jwt_payload(id_token)
        if claims:
            account_id = _extract_account_id(claims)
            if account_id:
                return account_id
    claims = _parse_jwt_payload(access_token)
    return _extract_account_id(claims) if claims else None


# ---------------------------------------------------------------------------
# Token exchange / refresh (urllib only — no extra deps)
# ---------------------------------------------------------------------------


def _post_form(url: str, data: dict[str, str]) -> dict:
    if not url.startswith("https://"):
        msg = f"Refusing non-HTTPS URL: {url}"
        raise ValueError(msg)
    body = urllib.parse.urlencode(data).encode()
    req = urllib.request.Request(  # noqa: S310
        url,
        data=body,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:  # noqa: S310
        return json.loads(resp.read())


def _post_json(url: str, data: dict, extra_headers: dict[str, str] | None = None) -> dict:
    if not url.startswith("https://"):
        msg = f"Refusing non-HTTPS URL: {url}"
        raise ValueError(msg)
    body = json.dumps(data).encode()
    headers = {"Content-Type": "application/json"}
    if extra_headers:
        headers.update(extra_headers)
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")  # noqa: S310
    with urllib.request.urlopen(req) as resp:  # noqa: S310
        return json.loads(resp.read())


def _exchange_code(
    code: str,
    redirect_uri: str,
    verifier: str,
) -> dict:
    return _post_form(
        f"{ISSUER}/oauth/token",
        {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": redirect_uri,
            "client_id": CLIENT_ID,
            "code_verifier": verifier,
        },
    )


def _refresh_tokens(refresh_token: str) -> dict:
    return _post_form(
        f"{ISSUER}/oauth/token",
        {
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": CLIENT_ID,
        },
    )


def _parse_token_response(resp: dict, existing_account_id: str | None = None) -> TokenData:
    access_token = resp["access_token"]
    refresh_token = resp["refresh_token"]
    expires_in = int(resp.get("expires_in", 3600))
    id_token = resp.get("id_token")
    account_id = _account_id_from_tokens(access_token, id_token) or existing_account_id
    return TokenData(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_at=time.time() + expires_in,
        account_id=account_id,
    )


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def load_tokens() -> TokenData | None:
    """Load saved tokens from ``~/.deepagents/chatgpt_tokens.json``."""
    if not _TOKEN_FILE.exists():
        return None
    try:
        data = json.loads(_TOKEN_FILE.read_text())
        return TokenData(
            access_token=data["access_token"],
            refresh_token=data["refresh_token"],
            expires_at=float(data["expires_at"]),
            account_id=data.get("account_id"),
        )
    except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
        logger.debug("Could not load ChatGPT tokens: %s", exc)
        return None


def save_tokens(tokens: TokenData) -> None:
    """Persist tokens to ``~/.deepagents/chatgpt_tokens.json``."""
    _TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
    _TOKEN_FILE.write_text(json.dumps(dict(tokens), indent=2))
    _TOKEN_FILE.chmod(0o600)


def delete_tokens() -> None:
    """Remove saved tokens."""
    if _TOKEN_FILE.exists():
        _TOKEN_FILE.unlink()


def refresh_if_needed(tokens: TokenData) -> TokenData:
    """Return tokens, refreshing if the access token has expired (or will in 60s)."""
    if time.time() < tokens["expires_at"] - 60:
        return tokens
    logger.debug("ChatGPT access token expired, refreshing")
    resp = _refresh_tokens(tokens["refresh_token"])
    refreshed = _parse_token_response(resp, existing_account_id=tokens.get("account_id"))
    save_tokens(refreshed)
    return refreshed


# ---------------------------------------------------------------------------
# Browser (redirect) flow
# ---------------------------------------------------------------------------


_SUCCESS_HTML = b"""<!doctype html><html><head><title>Deep Agents - Login Successful</title>
<style>body{font-family:system-ui,sans-serif;display:flex;justify-content:center;
align-items:center;height:100vh;margin:0;background:#131010;color:#f1ecec;}
.container{text-align:center;padding:2rem;}h1{margin-bottom:1rem;}</style></head>
<body><div class="container"><h1>Login Successful</h1>
<p>You can close this window and return to Deep Agents.</p></div>
<script>setTimeout(()=>window.close(),2000)</script></body></html>"""

_OAUTH_TIMEOUT_SECONDS = 300


def _make_callback_handler(
    expected_state: str,
    result: dict,
    error_holder: list[str],
    done: threading.Event,
) -> type[BaseHTTPRequestHandler]:
    """Build an ``HTTPServer`` handler class bound to a single OAuth callback."""

    class _Handler(BaseHTTPRequestHandler):
        def log_message(self, *args: Any, **kwargs: Any) -> None:
            """Silence stdout access logs."""

        def do_GET(self) -> None:
            parsed = urllib.parse.urlparse(self.path)
            qs = urllib.parse.parse_qs(parsed.query)

            if parsed.path != "/auth/callback":
                self._respond(404, b"Not found")
                return

            code = (qs.get("code") or [None])[0]
            returned_state = (qs.get("state") or [None])[0]
            err = (qs.get("error") or [None])[0]

            if err:
                error_holder.append(qs.get("error_description", [err])[0])
                self._respond(400, b"Authorization failed")
            elif not code:
                error_holder.append("Missing authorization code")
                self._respond(400, b"Missing code")
            elif returned_state != expected_state:
                error_holder.append("State mismatch (possible CSRF)")
                self._respond(400, b"State mismatch")
            else:
                result["code"] = code
                self._respond(200, _SUCCESS_HTML, content_type="text/html")
            done.set()

        def _respond(self, status: int, body: bytes, content_type: str = "text/plain") -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.end_headers()
            self.wfile.write(body)

    return _Handler


def login_browser() -> TokenData:
    """Interactive browser PKCE flow.

    Starts a local HTTP server on port 1455, opens the browser at the
    OpenAI authorization URL, waits for the callback, exchanges the code
    for tokens, and returns them.

    Raises:
        RuntimeError: If authorization fails or times out.
    """
    verifier = _generate_verifier()
    challenge = _derive_challenge(verifier)
    state = _generate_state()
    redirect_uri = f"http://localhost:{OAUTH_PORT}/auth/callback"

    params = urllib.parse.urlencode(
        {
            "response_type": "code",
            "client_id": CLIENT_ID,
            "redirect_uri": redirect_uri,
            "scope": "openid profile email offline_access",
            "code_challenge": challenge,
            "code_challenge_method": "S256",
            "id_token_add_organizations": "true",
            "codex_cli_simplified_flow": "true",
            "state": state,
            "originator": "deepagents",
        }
    )
    auth_url = f"{ISSUER}/oauth/authorize?{params}"

    result: dict = {}
    error_holder: list[str] = []
    done = threading.Event()

    handler_cls = _make_callback_handler(state, result, error_holder, done)
    server = HTTPServer(("127.0.0.1", OAUTH_PORT), handler_cls)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        logger.info("Opening browser for ChatGPT authorization...")
        logger.info("If the browser did not open, visit: %s", auth_url)
        webbrowser.open(auth_url)

        if not done.wait(timeout=_OAUTH_TIMEOUT_SECONDS):
            msg = "OAuth callback timed out after 5 minutes"
            raise RuntimeError(msg)
    finally:
        server.shutdown()

    if error_holder:
        msg = f"Authorization failed: {error_holder[0]}"
        raise RuntimeError(msg)

    code = result.get("code")
    if not code:
        msg = "No authorization code received"
        raise RuntimeError(msg)

    resp = _exchange_code(code, redirect_uri, verifier)
    tokens = _parse_token_response(resp)
    save_tokens(tokens)
    return tokens


# ---------------------------------------------------------------------------
# Device-code (headless) flow
# ---------------------------------------------------------------------------


_DEVICE_POLL_SAFETY_MARGIN_SECONDS = 3
_DEVICE_POLL_PENDING_STATUS = (403, 404)


def login_device() -> TokenData:
    """Headless device-code flow.

    Requests a user code, prints instructions, then polls until the user
    completes authorization in their browser.

    Raises:
        RuntimeError: If device authorization fails.
    """
    ua = f"deepagents-cli/{_deepagents_version()}"
    device_resp = _post_json(
        f"{ISSUER}/api/accounts/deviceauth/usercode",
        {"client_id": CLIENT_ID},
        extra_headers={"User-Agent": ua},
    )
    device_auth_id = device_resp["device_auth_id"]
    user_code = device_resp["user_code"]
    interval_s = max(int(device_resp.get("interval", 5)), 1)

    logger.info("Open %s/codex/device in your browser and enter code: %s", ISSUER, user_code)

    while True:
        time.sleep(interval_s + _DEVICE_POLL_SAFETY_MARGIN_SECONDS)
        try:
            poll_resp = _post_json(
                f"{ISSUER}/api/accounts/deviceauth/token",
                {"device_auth_id": device_auth_id, "user_code": user_code},
                extra_headers={"User-Agent": ua},
            )
        except urllib.error.HTTPError as exc:
            if exc.code in _DEVICE_POLL_PENDING_STATUS:
                continue  # still pending
            msg = f"Device auth polling failed: {exc}"
            raise RuntimeError(msg) from exc

        # Exchange the authorization_code returned by device auth
        token_resp = _post_form(
            f"{ISSUER}/oauth/token",
            {
                "grant_type": "authorization_code",
                "code": poll_resp["authorization_code"],
                "redirect_uri": f"{ISSUER}/deviceauth/callback",
                "client_id": CLIENT_ID,
                "code_verifier": poll_resp["code_verifier"],
            },
        )
        tokens = _parse_token_response(token_resp)
        save_tokens(tokens)
        return tokens


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _deepagents_version() -> str:
    try:
        return version("deepagents-cli")
    except PackageNotFoundError:
        return "0.0.0"
