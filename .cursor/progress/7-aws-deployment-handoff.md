---
github_issue: 10
---
# Handoff: AWS Deployment Infrastructure

## Working directory

`~/Desktop/friendly-advice-columnist`

## Contents

**Date:** 2026-02-14 (updated)
**Branch:** `feature/aws-deployment`
**Version:** 1.2.0
**Claude Code conversation title/ID:** "Deploy Application To AWS" (CFS features/13)

## What Was Done

### 1. Containerization
- Created `Dockerfile` (Python 3.11-slim, web-only deps, no PyTorch/training code)
- Created `.dockerignore` to exclude training data, checkpoints, dev files
- Created `requirements-web.txt` with slim dependency list for the Docker image

### 2. Bring Your Own API Key (BYOAK) — addresses CFS security/1
- Users enter their OpenAI API key in a settings modal (stored in localStorage, sent per-request via `X-OpenAI-API-Key` header)
- Backend compares a SHA-256 hash of the provided key against `OWNER_API_KEY_HASH` env var
- **Owner's key** → full two-stage pipeline (base model draft + fine-tuned model revision)
- **Other keys** → single-stage pipeline (base model only, with extended columnist system prompt)
- No API keys are stored server-side; the owner's key is only stored as a hash
- Frontend: `app/static/js/api-key.js` handles the modal, localStorage, and HTMX header injection
- Backend: `app/routes/advice.py` has `is_owner_key()`, `hash_api_key()`, and updated route handlers

### 3. Extended System Prompt for Non-Owner Keys (NEW — 2026-02-14)
- Added `ADVICE_COLUMNIST_SYSTEM_PROMPT_EXTENDED` in `models/prompts.py`
- Extends the base system prompt with additional voice/style guidance (calm but morally decisive, concrete scripts/boundaries, anti-LLM-filler, no em dashes, substantive development)
- Non-owner keys use this extended prompt via a `system_prompt` parameter on `generate_answer()`
- Owner keys use the standard prompt (fine-tuned model handles voice in stage 2)
- 2 new tests: `test_non_owner_uses_extended_prompt`, `test_owner_uses_default_prompt`

### 4. DynamoDB Conversation Persistence
- `app/dynamodb.py` — full CRUD operations keyed by `user_id` (hashed API key prefix) + `session_id`
- `app/routes/conversations.py` — REST API endpoints: `POST/GET/DELETE /api/conversations`
- `app/static/js/sessions.js` — updated with background DynamoDB sync (fire-and-forget on save/delete)
- localStorage remains the primary store; DynamoDB provides cross-device persistence
- 90-day TTL for automatic cleanup
- **Default region: us-east-2**

### 5. Security Hardening
- Rate limiting: 10 requests/minute per IP on advice endpoints (slowapi)
- Security headers middleware: X-Content-Type-Options, X-Frame-Options, X-XSS-Protection, Referrer-Policy, Permissions-Policy
- XSS protection: all response content is `html.escape()`d before rendering in HTML endpoint
- CORS: configurable via `ALLOWED_ORIGINS` env var
- Input validation unchanged (max length, question relevance screening)

### 6. Deployment Script
- `deploy.sh` — one-command deployment with `--dry-run` and `--build-only` flags
- Uses `ECR_REGISTRY` env var (set in `.env`) instead of constructing ECR URI
- Builds with `--no-cache` by default to avoid stale Docker layers
- Creates ECR repo, builds/pushes Docker image, creates DynamoDB table with TTL, creates IAM roles (least-privilege), creates/updates App Runner service
- Custom domain instructions in docs

### 7. Docker Startup Fixes (NEW — 2026-02-14)
- Made `BigramLanguageModel` import conditional in `models/__init__.py` (PyTorch not available in Docker image)
- Made module-level OpenAI client creation conditional in `models/openai_backend.py` (OPENAI_API_KEY not set in BYOAK mode)
- Fixed `run_cmd` piping bug in `deploy.sh` that corrupted ECR login token (echo output was piped into docker login)

### 8. Tests and Documentation
- 51 tests passing (29 in test_advice_api.py, 22 in test_session_ui.py)
- New test classes: `TestSecurityHeaders`, `TestApiKeyRequirement`, `TestApiKeyRouting`, `TestAdviceHTMLEndpoint.test_advice_html_xss_prevention`
- `docs/DEPLOYMENT.md` — full deployment guide with architecture, prerequisites, env vars, cost estimate
- Updated `README.md` and `CLAUDE.md` with deployment commands

### 9. Lint Cleanup
- Fixed all 14 pre-existing ruff lint errors across the repo
- Added `per-file-ignores` in `pyproject.toml` for legitimate E402 patterns
- Pre-commit hook now passes cleanly — no need for `--no-verify`

## Commits on This Branch

```
328942e fix: Resolve Docker startup errors for web-only image
d3cd121 refactor: Use ECR_REGISTRY env var instead of constructing ECR URI
6d494f9 feat: Add extended system prompt for non-owner single-stage responses
4b49f78 chore: CFS doc
e9b3933 chore: Create handoff from last session.
931d1d1 fix: Resolve all pre-existing ruff lint errors
e9acd96 Bump version: 1.1.1 → 1.2.0
7402001 chore: Close CFS features/11 and fix double-prefix naming
d338535 feat: Add AWS deployment infrastructure with BYOAK model and DynamoDB persistence
```

**Note:** There are uncommitted changes in `deploy.sh` (ECR login pipe fix and `--no-cache` default). Commit these before continuing.

## Key Files to Know

| File | Purpose |
|---|---|
| `deploy.sh` | Main deployment script |
| `Dockerfile` | Container definition |
| `requirements-web.txt` | Docker-only dependencies |
| `app/routes/advice.py` | BYOAK routing logic, rate limiting, extended prompt |
| `app/dynamodb.py` | DynamoDB CRUD operations |
| `app/routes/conversations.py` | Conversation persistence API |
| `app/static/js/api-key.js` | Frontend API key management |
| `models/prompts.py` | All system prompts including extended prompt |
| `models/openai_backend.py` | OpenAI API calls with system_prompt override |
| `docs/DEPLOYMENT.md` | Full deployment guide |

## What's Left To Do (Next Steps)

### BLOCKER: App Runner Health Check Failure

The Docker image works perfectly locally (app starts, serves requests, health check responds). However, App Runner deployment fails with "Health check failed" after ~6 minutes. Two deployments have failed with the same error.

**Immediate debugging steps:**
1. **Check the application logs in CloudWatch** (not the event logs shown in the App Runner console). Go to CloudWatch > Log groups > look for `/aws/apprunner/friendly-advice-columnist/.../application`. This will show the actual Python traceback during startup.
2. **Possible causes to investigate:**
   - The `conversations_router` import triggers boto3/DynamoDB initialization at startup. Locally this logs a `NoCredentialsError` and continues, but App Runner may handle this differently or timeout during initialization. Check if the instance role (`friendly-advice-columnist-instance-role`) is properly attached and has the right trust policy for `tasks.apprunner.amazonaws.com`.
   - The health check timeout may be too aggressive. Current config: 5s timeout, 10s interval, 5 unhealthy threshold. Consider increasing timeout to 10s and unhealthy threshold to 10.
   - Verify the App Runner service has the correct instance role ARN attached (needed for DynamoDB access, which boto3 tries to use at import time via `app/dynamodb.py`).

### After resolving health check
3. **Set up custom domain** — follow instructions in `docs/DEPLOYMENT.md` (DNS CNAME records, certificate validation)
4. **Update `ALLOWED_ORIGINS`** env var in App Runner to include the custom domain
5. **Test the deployed app** — verify BYOAK works, conversation persistence works, rate limiting works
6. **Consider AWS WAF** — App Runner supports WAF for additional DDoS/bot protection
7. **Close CFS features/13** once deployment is verified: `cfs i features complete 13 --force`
8. **Close CFS security/1** once BYOAK is verified in production: `cfs i security complete 1 --force`
9. **Merge branch to master** and create a PR

## Open CFS Issues Related to This Work
- **features/13** — Deploy Application To AWS (leave open until deployed)
- **security/1** — Handle fine-tuned model vs base model with API keys (implemented but not yet deployed)

## Notes for the Next Agent
- The pre-commit hook runs ruff on the entire repo. All lint errors are now resolved and the hook passes cleanly.
- The `models/__init__.py` import of `BigramLanguageModel` is now wrapped in a try/except for the Docker image (which doesn't include PyTorch).
- The module-level `OpenAI` client in `openai_backend.py` is only created when `OPENAI_API_KEY` is set. In the web app (BYOAK mode), all calls use per-request clients.
- `deploy.sh` requires `ECR_REGISTRY` env var (set in `.env`). It derives `ECR_HOST` and `ECR_REPO_NAME` from it.
- `deploy.sh` builds with `--no-cache` by default.
- You **cannot** use `update-service` on a service in `CREATE_FAILED` state. You must delete the failed service first, then rerun `deploy.sh`.
- bump2version is configured with `commit = true` and `tag = true`, so it tries to commit and tag automatically. If the pre-commit hook blocks it, you may need `--allow-dirty` or to do the version bump manually.
- The owner's API key hash is `27a10673e7ae6af3bf01ef74cc43ff346919bb0ce617e0b37a2b345b5a5a935b`.

## Acceptance criteria
