# ChatGPT Subscription Support

Add ChatGPT Plus/Pro subscription as a provider option, allowing users to authenticate
via OpenAI OAuth instead of requiring an API key.

Reference implementation: `~/repos/opencode/packages/opencode/src/plugin/codex.ts`

## Tasks

### 1. OAuth Authentication
- [x] Implement OpenAI OAuth PKCE flow (issuer: `https://auth.openai.com`, client ID: `app_EMoamEEZ73f0CkXaXp7hrann`)
- [x] Support browser redirect and device-code (headless) auth methods
- [x] Token exchange, refresh, and secure persistence (`~/.deepagents/chatgpt_tokens.json`, mode 600)
- [x] Extract `chatgpt_account_id` from JWT claims

### 2. Codex API Provider
- [x] Create a LangChain-compatible chat model that routes to `https://chatgpt.com/backend-api/codex/responses`
- [x] Set required headers: `Authorization: Bearer <access_token>`, `ChatGPT-Account-Id`
- [x] Handle token refresh on expiry before requests
- [x] Filter to allowed codex models (gpt-5.x-codex variants)

### 3. Wire into `_models.py` and CLI
- [x] Add `chatgpt:` prefix handling in `resolve_model()` (`libs/deepagents/deepagents/_models.py`)
- [x] Add `chatgpt` provider detection in CLI config (`libs/cli/deepagents_cli/config.py`)
- [x] Add `deep-agents login openai` / `deep-agents logout openai` CLI commands

### 4. Tests
- [x] Unit tests for OAuth token parsing and refresh (`test_chatgpt_auth.py`)
- [x] Unit tests for model resolution with `chatgpt:` prefix (`test_models.py`)
- [ ] Integration test for end-to-end auth flow (mocked)

## Implementation notes

- Auth module: `libs/deepagents/deepagents/_chatgpt_auth.py`
- Model module: `libs/deepagents/deepagents/_chatgpt_model.py`
- Token file: `~/.deepagents/chatgpt_tokens.json` (chmod 600)
- CLI commands: `deep-agents login openai [--headless]`, `deep-agents logout openai`
- Model spec: `chatgpt:gpt-5.1-codex` (or any allowed Codex model name)
- `has_chatgpt` property on `Settings` checks for stored tokens
- ChatGPT tokens are preferred over OpenAI API key in default model selection
