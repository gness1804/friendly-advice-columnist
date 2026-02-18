"""
Server-side session management for secure API key storage.

Uses Fernet symmetric encryption to store the user's OpenAI API key
in an httpOnly cookie, keeping it inaccessible to client-side JavaScript.
"""

import os

from cryptography.fernet import Fernet, InvalidToken
from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel, Field

router = APIRouter(tags=["session"])

# Session cookie configuration
COOKIE_NAME = "fac_session"
COOKIE_MAX_AGE = 60 * 60 * 24 * 7  # 7 days

# Encryption key: derived from SESSION_SECRET env var or auto-generated.
# In production, SESSION_SECRET should be set to a stable Fernet key so
# sessions survive container restarts.
_SESSION_SECRET = os.environ.get("SESSION_SECRET", "")
if _SESSION_SECRET:
    _fernet = Fernet(_SESSION_SECRET.encode())
else:
    # Auto-generate for development; sessions won't survive restarts
    _fernet = Fernet(Fernet.generate_key())


def _is_secure_request(request: Request) -> bool:
    """Check if the request was made over HTTPS (or via a trusted proxy)."""
    if request.url.scheme == "https":
        return True
    # App Runner and other reverse proxies set X-Forwarded-Proto
    forwarded_proto = request.headers.get("x-forwarded-proto", "")
    return forwarded_proto == "https"


def encrypt_api_key(api_key: str) -> str:
    """Encrypt an API key for cookie storage."""
    return _fernet.encrypt(api_key.encode()).decode()


def decrypt_api_key(token: str) -> str | None:
    """Decrypt an API key from a cookie value. Returns None if invalid."""
    try:
        return _fernet.decrypt(token.encode()).decode()
    except (InvalidToken, Exception):
        return None


def get_api_key_from_request(request: Request) -> str | None:
    """Extract the API key from the session cookie."""
    token = request.cookies.get(COOKIE_NAME)
    if not token:
        return None
    return decrypt_api_key(token)


def require_api_key(request: Request) -> str:
    """Extract the API key from the session cookie, or raise 401."""
    api_key = get_api_key_from_request(request)
    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="No active session. Please enter your API key in the settings.",
        )
    return api_key


# --- API Endpoints ---


class SessionCreateRequest(BaseModel):
    """Request body for creating a session."""

    api_key: str = Field(..., min_length=3, description="OpenAI API key")


@router.post("/session")
async def create_session(
    body: SessionCreateRequest, request: Request, response: Response
):
    """Store the API key in an encrypted httpOnly cookie."""
    encrypted = encrypt_api_key(body.api_key)
    secure = _is_secure_request(request)

    response.set_cookie(
        key=COOKIE_NAME,
        value=encrypted,
        max_age=COOKIE_MAX_AGE,
        httponly=True,
        secure=secure,
        samesite="strict",
    )
    return {"status": "ok"}


@router.get("/session/status")
async def session_status(request: Request):
    """Check whether the user has an active session (without exposing the key)."""
    api_key = get_api_key_from_request(request)
    has_key = api_key is not None
    # Return a masked hint so the user can verify which key is stored
    masked = ""
    if api_key and len(api_key) > 7:
        masked = api_key[:7] + "..." + api_key[-4:]
    return {"has_key": has_key, "masked_key": masked}


@router.delete("/session")
async def delete_session(response: Response):
    """Clear the session cookie (logout)."""
    response.delete_cookie(key=COOKIE_NAME)
    return {"status": "ok"}
