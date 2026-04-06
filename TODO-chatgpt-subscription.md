# ChatGPT Subscription Support

Add ChatGPT Plus/Pro subscription as a provider option, allowing users to authenticate
via OpenAI OAuth instead of requiring an API key.

Reference implementation: `~/repos/opencode/packages/opencode/src/plugin/codex.ts`

## Tasks

### 1. OAuth Authentication
- [ ] Implement OpenAI OAuth PKCE flow (issuer: `https://auth.openai.com`, client ID: `app_EMoamEEZ73f0CkXaXp7hrann`)
- [ ] Support browser redirect and device-code (headless) auth methods
- [ ] Token exchange, refresh, and secure persistence
- [ ] Extract `chatgpt_account_id` from JWT claims

### 2. Codex API Provider
- [ ] Create a LangChain-compatible chat model that routes to `https://chatgpt.com/backend-api/codex/responses`
- [ ] Set required headers: `Authorization: Bearer <access_token>`, `ChatGPT-Account-Id`
- [ ] Handle token refresh on expiry before requests
- [ ] Filter to allowed codex models (gpt-5.x-codex variants)

### 3. Wire into `_models.py` and CLI
- [ ] Add `chatgpt:` prefix handling in `resolve_model()` (`libs/deepagents/deepagents/_models.py`)
- [ ] Add `chatgpt` provider detection in CLI config (`libs/cli/deepagents_cli/config.py`)
- [ ] Add `deep-agents login openai` or similar CLI command for OAuth flow

### 4. Tests
- [ ] Unit tests for OAuth token parsing and refresh
- [ ] Unit tests for model resolution with `chatgpt:` prefix
- [ ] Integration test for end-to-end auth flow (mocked)
