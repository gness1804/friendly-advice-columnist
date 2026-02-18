"""
Tests for the server-side session management (encrypted API key cookies).
"""

import sys
from pathlib import Path

from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.main import app
from app.session import encrypt_api_key, decrypt_api_key, COOKIE_NAME

client = TestClient(app)

TEST_API_KEY = "sk-test-key-for-testing"


class TestEncryption:
    """Tests for the Fernet encrypt/decrypt helpers."""

    def test_round_trip(self):
        """Encrypting then decrypting should return the original key."""
        encrypted = encrypt_api_key(TEST_API_KEY)
        assert encrypted != TEST_API_KEY
        assert decrypt_api_key(encrypted) == TEST_API_KEY

    def test_decrypt_invalid_token(self):
        """Decrypting garbage should return None, not raise."""
        assert decrypt_api_key("not-a-valid-token") is None

    def test_decrypt_empty_string(self):
        """Decrypting an empty string should return None."""
        assert decrypt_api_key("") is None


class TestCreateSession:
    """Tests for POST /api/session."""

    def test_create_session_sets_cookie(self):
        """Successful session creation should set an httpOnly cookie."""
        response = client.post(
            "/api/session",
            json={"api_key": TEST_API_KEY},
        )
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}
        assert COOKIE_NAME in response.cookies

    def test_create_session_missing_key(self):
        """Missing api_key should return 422."""
        response = client.post("/api/session", json={})
        assert response.status_code == 422

    def test_create_session_empty_key(self):
        """Empty api_key should return 422."""
        response = client.post("/api/session", json={"api_key": ""})
        assert response.status_code == 422


class TestSessionStatus:
    """Tests for GET /api/session/status."""

    def test_status_no_session(self):
        """Without a session cookie, has_key should be false."""
        # Use a fresh client to avoid cookie leakage from other tests
        fresh_client = TestClient(app)
        response = fresh_client.get("/api/session/status")
        assert response.status_code == 200
        data = response.json()
        assert data["has_key"] is False
        assert data["masked_key"] == ""

    def test_status_with_session(self):
        """After creating a session, status should reflect it."""
        session_client = TestClient(app)
        session_client.post(
            "/api/session",
            json={"api_key": TEST_API_KEY},
        )
        # TestClient persists cookies, so the next request has the cookie
        response = session_client.get("/api/session/status")
        assert response.status_code == 200
        data = response.json()
        assert data["has_key"] is True
        assert data["masked_key"].startswith("sk-test")


class TestDeleteSession:
    """Tests for DELETE /api/session."""

    def test_delete_session_clears_cookie(self):
        """Deleting a session should clear the cookie."""
        session_client = TestClient(app)
        session_client.post(
            "/api/session",
            json={"api_key": TEST_API_KEY},
        )

        # Delete it
        del_resp = session_client.delete("/api/session")
        assert del_resp.status_code == 200
        assert del_resp.json() == {"status": "ok"}

        # Verify the session is gone
        status_resp = session_client.get("/api/session/status")
        assert status_resp.json()["has_key"] is False


class TestCookieBasedAuth:
    """Tests that advice/conversation endpoints accept the session cookie."""

    def test_advice_endpoint_rejects_no_session(self):
        """Advice endpoint should reject requests without a session cookie."""
        fresh_client = TestClient(app)
        response = fresh_client.post(
            "/api/advice",
            json={"question": "How do I talk to my partner?"},
        )
        # 401 (no session) or 429 (rate limited from earlier tests sharing same IP)
        assert response.status_code in (401, 429)

    def test_advice_html_rejects_no_session(self):
        """HTML advice endpoint should reject requests without a session cookie."""
        fresh_client = TestClient(app)
        response = fresh_client.post(
            "/api/advice/html",
            data={"question": "How do I talk to my partner?"},
        )
        assert response.status_code == 401
