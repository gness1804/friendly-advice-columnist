# Handoff: AWS Deployment Infrastructure

**Date:** 2026-02-12
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
- **Other keys** → single-stage pipeline (base model only, with the columnist system prompt)
- No API keys are stored server-side; the owner's key is only stored as a hash
- Frontend: `app/static/js/api-key.js` handles the modal, localStorage, and HTMX header injection
- Backend: `app/routes/advice.py` has `is_owner_key()`, `hash_api_key()`, and updated route handlers

### 3. DynamoDB Conversation Persistence
- `app/dynamodb.py` — full CRUD operations keyed by `user_id` (hashed API key prefix) + `session_id`
- `app/routes/conversations.py` — REST API endpoints: `POST/GET/DELETE /api/conversations`
- `app/static/js/sessions.js` — updated with background DynamoDB sync (fire-and-forget on save/delete)
- localStorage remains the primary store; DynamoDB provides cross-device persistence
- 90-day TTL for automatic cleanup
- **Default region: us-east-2**

### 4. Security Hardening
- Rate limiting: 10 requests/minute per IP on advice endpoints (slowapi)
- Security headers middleware: X-Content-Type-Options, X-Frame-Options, X-XSS-Protection, Referrer-Policy, Permissions-Policy
- XSS protection: all response content is `html.escape()`d before rendering in HTML endpoint
- CORS: configurable via `ALLOWED_ORIGINS` env var
- Input validation unchanged (max length, question relevance screening)

### 5. Deployment Script
- `deploy.sh` — one-command deployment with `--dry-run` and `--build-only` flags
- Creates ECR repo, builds/pushes Docker image, creates DynamoDB table with TTL, creates IAM roles (least-privilege), creates/updates App Runner service
- Custom domain instructions in docs

### 6. Tests and Documentation
- 49 tests passing (27 in test_advice_api.py, 22 in test_session_ui.py)
- New test classes: `TestSecurityHeaders`, `TestApiKeyRequirement`, `TestApiKeyRouting`, `TestAdviceHTMLEndpoint.test_advice_html_xss_prevention`
- `docs/DEPLOYMENT.md` — full deployment guide with architecture, prerequisites, env vars, cost estimate
- Updated `README.md` and `CLAUDE.md` with deployment commands

### 7. Lint Cleanup (bonus)
- Fixed all 14 pre-existing ruff lint errors across the repo
- Added `per-file-ignores` in `pyproject.toml` for legitimate E402 patterns
- Pre-commit hook now passes cleanly — no need for `--no-verify`

## Commits on This Branch

```
931d1d1 fix: Resolve all pre-existing ruff lint errors
e9acd96 Bump version: 1.1.1 → 1.2.0
7402001 chore: Close CFS features/11 and fix double-prefix naming
d338535 feat: Add AWS deployment infrastructure with BYOAK model and DynamoDB persistence
```

## Key Files to Know

| File | Purpose |
|---|---|
| `deploy.sh` | Main deployment script |
| `Dockerfile` | Container definition |
| `requirements-web.txt` | Docker-only dependencies |
| `.env.example` | All environment variables documented |
| `app/routes/advice.py` | BYOAK routing logic, rate limiting |
| `app/dynamodb.py` | DynamoDB CRUD operations |
| `app/routes/conversations.py` | Conversation persistence API |
| `app/static/js/api-key.js` | Frontend API key management |
| `docs/DEPLOYMENT.md` | Full deployment guide |

## What's Left To Do (Next Steps)

### Before deploying
1. **Run `./deploy.sh --dry-run`** to verify AWS CLI is configured and review what will be created
2. **Generate the owner API key hash:** `python -c "import hashlib; print(hashlib.sha256(b'sk-your-actual-key').hexdigest())"`
3. **Set `OWNER_API_KEY_HASH`** as an environment variable before deploying
4. **Test the Docker image locally** before pushing: `docker build -t friendly-advice-columnist . && docker run -p 8000:8000 -e OWNER_API_KEY_HASH=<hash> friendly-advice-columnist`

### During deployment
5. **Run `./deploy.sh`** for the full deployment
6. **Set up custom domain** — follow instructions in `docs/DEPLOYMENT.md` (DNS CNAME records, certificate validation)
7. **Update `ALLOWED_ORIGINS`** env var in App Runner to include the custom domain

### After deployment
8. **Test the deployed app** — verify BYOAK works, conversation persistence works, rate limiting works
9. **Consider AWS WAF** — App Runner supports WAF for additional DDoS/bot protection (configured at AWS level, not in code)
10. **Close CFS features/13** once deployment is verified: `cfs i features complete 13 --force`
11. **Close CFS security/1** once BYOAK is verified in production: `cfs i security complete 1 --force`
12. **Merge branch to master** and create a PR

## Open CFS Issues Related to This Work
- **features/13** — Deploy Application To AWS (leave open until deployed)
- **security/1** — Handle fine-tuned model vs base model with API keys (implemented but not yet deployed)

## Notes for the Next Agent
- The pre-commit hook runs ruff on the entire repo. All lint errors are now resolved and the hook passes cleanly.
- The `models/__init__.py` imports `BigramLanguageModel` which requires PyTorch. This works fine locally but is excluded from the Docker image (which only needs `openai_backend.py` and `prompts.py`). The Dockerfile copies only the specific model files needed.
- bump2version is configured with `commit = true` and `tag = true`, so it tries to commit and tag automatically. If the pre-commit hook blocks it, you may need `--allow-dirty` or to do the version bump manually.
