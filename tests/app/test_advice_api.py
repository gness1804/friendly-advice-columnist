"""
Tests for the advice API endpoints.
"""

import sys
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.main import app
from app.session import COOKIE_NAME, encrypt_api_key

client = TestClient(app)

# Fake API key and corresponding encrypted cookie for tests
FAKE_API_KEY = "sk-test-key-for-testing"
SESSION_COOKIES = {COOKIE_NAME: encrypt_api_key(FAKE_API_KEY)}

# Mock responses for testing
MOCK_DRAFT_RESPONSE = "This is a draft response about your situation."
MOCK_V3_RESPONSE = """SCORE
7

STRENGTHS
- Good empathy
- Clear advice

WEAKNESSES
- Could be more specific

REVISED_RESPONSE
Thank you for sharing your situation. Here's my thoughtful advice on how to handle this interpersonal challenge. Communication is key in any relationship."""


class TestHealthEndpoint:
    """Tests for the health check endpoint."""

    def test_health_check(self):
        """Health endpoint should return healthy status."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "healthy"}


class TestIndexPage:
    """Tests for the main index page."""

    def test_index_page_loads(self):
        """Index page should load successfully."""
        response = client.get("/")
        assert response.status_code == 200
        assert "Friendly Advice" in response.text
        assert "Columnist" in response.text

    def test_index_page_contains_form(self):
        """Index page should contain the advice form."""
        response = client.get("/")
        assert response.status_code == 200
        assert 'id="advice-form"' in response.text
        assert 'name="question"' in response.text

    def test_index_page_contains_htmx(self):
        """Index page should include HTMX."""
        response = client.get("/")
        assert response.status_code == 200
        assert "htmx.org" in response.text

    def test_index_page_contains_api_key_ui(self):
        """Index page should include API key settings UI."""
        response = client.get("/")
        assert response.status_code == 200
        assert "api-key-modal" in response.text
        assert "api-key.js" in response.text


class TestSecurityHeaders:
    """Tests for security headers."""

    def test_security_headers_present(self):
        """All security headers should be set on responses."""
        response = client.get("/health")
        assert response.headers["X-Content-Type-Options"] == "nosniff"
        assert response.headers["X-Frame-Options"] == "DENY"
        assert response.headers["X-XSS-Protection"] == "1; mode=block"
        assert "strict-origin" in response.headers["Referrer-Policy"]
        assert "camera=()" in response.headers["Permissions-Policy"]


class TestApiKeyRequirement:
    """Tests for API key requirement (via session cookie)."""

    def test_advice_endpoint_requires_session(self):
        """Advice endpoint should reject requests without a session cookie."""
        response = client.post(
            "/api/advice",
            json={"question": "How do I talk to my partner?"},
        )
        assert response.status_code == 401
        assert "session" in response.json()["detail"].lower()

    def test_advice_html_requires_session(self):
        """HTML advice endpoint should reject requests without a session cookie."""
        response = client.post(
            "/api/advice/html",
            data={"question": "How do I talk to my partner?"},
        )
        assert response.status_code == 401


class TestAdviceAPIEndpoint:
    """Tests for the JSON advice API endpoint."""

    @patch("app.routes.advice.is_owner_key", return_value=False)
    @patch("app.routes.advice.validate_question_relevance")
    @patch("app.routes.advice.call_base_model")
    def test_advice_endpoint_non_owner(self, mock_base, mock_validate, mock_is_owner):
        """Non-owner should get base model response directly."""
        mock_validate.return_value = True
        mock_base.return_value = MOCK_DRAFT_RESPONSE

        response = client.post(
            "/api/advice",
            json={"question": "How do I talk to my partner about finances?"},
            cookies=SESSION_COOKIES,
        )

        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "elapsed_time" in data
        assert data["used_fine_tuned"] is False

    @patch("app.routes.advice.is_owner_key", return_value=True)
    @patch("app.routes.advice.validate_question_relevance")
    @patch("app.routes.advice.call_base_model")
    @patch("app.routes.advice.call_fine_tuned_model")
    def test_advice_endpoint_owner(
        self, mock_fine_tuned, mock_base, mock_validate, mock_is_owner
    ):
        """Owner should get fine-tuned model response."""
        mock_validate.return_value = True
        mock_base.return_value = MOCK_DRAFT_RESPONSE
        mock_fine_tuned.return_value = MOCK_V3_RESPONSE

        response = client.post(
            "/api/advice",
            json={"question": "How do I talk to my partner about finances?"},
            cookies=SESSION_COOKIES,
        )

        assert response.status_code == 200
        data = response.json()
        assert data["used_fine_tuned"] is True
        assert "thoughtful advice" in data["answer"].lower()

    def test_advice_endpoint_empty_question(self):
        """Advice endpoint should reject empty questions."""
        response = client.post(
            "/api/advice",
            json={"question": ""},
            cookies=SESSION_COOKIES,
        )
        assert response.status_code == 422  # Validation error

    def test_advice_endpoint_missing_question(self):
        """Advice endpoint should reject requests without question."""
        response = client.post(
            "/api/advice",
            json={},
            cookies=SESSION_COOKIES,
        )
        assert response.status_code == 422

    def test_advice_endpoint_question_too_long(self):
        """Advice endpoint should reject questions exceeding max length."""
        long_question = "x" * 4001
        response = client.post(
            "/api/advice",
            json={"question": long_question},
            cookies=SESSION_COOKIES,
        )
        assert response.status_code == 422


class TestAdviceHTMLEndpoint:
    """Tests for the HTML advice endpoint (HTMX)."""

    @patch("app.routes.advice.is_owner_key", return_value=False)
    @patch("app.routes.advice.validate_question_relevance")
    @patch("app.routes.advice.call_base_model")
    def test_advice_html_endpoint_success(
        self, mock_base, mock_validate, mock_is_owner
    ):
        """HTML endpoint should return HTML fragment for valid questions."""
        mock_validate.return_value = True
        mock_base.return_value = MOCK_DRAFT_RESPONSE

        response = client.post(
            "/api/advice/html",
            data={"question": "How do I set boundaries with family?"},
            cookies=SESSION_COOKIES,
        )

        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "Advice" in response.text

    def test_advice_html_endpoint_empty_question(self):
        """HTML endpoint should return error HTML for empty questions."""
        response = client.post(
            "/api/advice/html",
            data={"question": ""},
            cookies=SESSION_COOKIES,
        )
        # FastAPI returns 422 for validation errors even with HTML
        assert response.status_code == 422

    @patch("app.routes.advice.is_owner_key", return_value=False)
    @patch("app.routes.advice.validate_question_relevance")
    @patch("app.routes.advice.call_base_model")
    def test_advice_html_endpoint_api_error(
        self, mock_base, mock_validate, mock_is_owner
    ):
        """HTML endpoint should return error HTML when API fails."""
        mock_validate.return_value = True
        mock_base.side_effect = RuntimeError("API connection failed")

        response = client.post(
            "/api/advice/html",
            data={"question": "How do I handle conflict?"},
            cookies=SESSION_COOKIES,
        )

        assert response.status_code == 503
        assert "Error" in response.text

    @patch("app.routes.advice.is_owner_key", return_value=False)
    @patch("app.routes.advice.validate_question_relevance")
    @patch("app.routes.advice.call_base_model")
    def test_advice_html_xss_prevention(self, mock_base, mock_validate, mock_is_owner):
        """HTML endpoint should escape HTML in responses to prevent XSS."""
        mock_validate.return_value = True
        mock_base.return_value = '<script>alert("xss")</script>Some advice'

        response = client.post(
            "/api/advice/html",
            data={"question": "How do I handle a difficult coworker?"},
            cookies=SESSION_COOKIES,
        )

        assert response.status_code == 200
        # The script tag should be escaped, not rendered
        assert "<script>" not in response.text
        assert "&lt;script&gt;" in response.text


class TestInputValidation:
    """Tests for input validation."""

    @patch("app.routes.advice.is_owner_key", return_value=False)
    @patch("app.routes.advice.validate_question_relevance")
    @patch("app.routes.advice.call_base_model")
    def test_question_with_special_characters(
        self, mock_base, mock_validate, mock_is_owner
    ):
        """Questions with special characters should be handled."""
        mock_validate.return_value = True
        mock_base.return_value = MOCK_DRAFT_RESPONSE

        response = client.post(
            "/api/advice",
            json={
                "question": "My partner & I aren't communicating well. What should I do?"
            },
            cookies=SESSION_COOKIES,
        )

        assert response.status_code == 200

    @patch("app.routes.advice.is_owner_key", return_value=False)
    @patch("app.routes.advice.validate_question_relevance")
    @patch("app.routes.advice.call_base_model")
    def test_question_with_newlines(self, mock_base, mock_validate, mock_is_owner):
        """Questions with newlines should be handled."""
        mock_validate.return_value = True
        mock_base.return_value = MOCK_DRAFT_RESPONSE

        response = client.post(
            "/api/advice",
            json={"question": "Line 1\nLine 2\nLine 3"},
            cookies=SESSION_COOKIES,
        )

        assert response.status_code == 200

    def test_question_at_max_length(self):
        """Questions at exactly max length should be accepted."""
        max_question = "x" * 4000

        from app.routes.advice import AdviceRequest

        request = AdviceRequest(question=max_question)
        assert len(request.question) == 4000


class TestQuestionScreening:
    """Tests for question relevance screening."""

    @patch("app.routes.advice.validate_question_relevance")
    def test_off_topic_question_rejected(self, mock_validate):
        """Off-topic questions should be rejected with a helpful error."""
        mock_validate.return_value = False

        response = client.post(
            "/api/advice",
            json={"question": "How do I fix my car's transmission?"},
            cookies=SESSION_COOKIES,
        )

        assert response.status_code == 400
        assert "interpersonal" in response.json()["detail"].lower()

    @patch("app.routes.advice.validate_question_relevance")
    def test_off_topic_question_html_endpoint(self, mock_validate):
        """Off-topic questions via HTML endpoint should show error."""
        mock_validate.return_value = False

        response = client.post(
            "/api/advice/html",
            data={"question": "What is the capital of France?"},
            cookies=SESSION_COOKIES,
        )

        assert response.status_code == 400
        assert "Error" in response.text
        assert "interpersonal" in response.text.lower()

    @patch("app.routes.advice.is_owner_key", return_value=False)
    @patch("app.routes.advice.validate_question_relevance")
    @patch("app.routes.advice.call_base_model")
    def test_relevant_question_processed(self, mock_base, mock_validate, mock_is_owner):
        """Relevant questions should be processed normally."""
        mock_validate.return_value = True
        mock_base.return_value = MOCK_DRAFT_RESPONSE

        response = client.post(
            "/api/advice",
            json={
                "question": "My mother-in-law keeps criticizing my parenting. What should I do?"
            },
            cookies=SESSION_COOKIES,
        )

        assert response.status_code == 200
        mock_validate.assert_called_once()
        mock_base.assert_called_once()

    @patch("app.routes.advice.validate_question_relevance")
    def test_validation_error_handled(self, mock_validate):
        """Validation errors should be handled gracefully."""
        mock_validate.side_effect = RuntimeError("API error during validation")

        response = client.post(
            "/api/advice",
            json={"question": "How do I talk to my sister?"},
            cookies=SESSION_COOKIES,
        )

        assert response.status_code == 503
        assert "error" in response.json()["detail"].lower()


class TestApiKeyRouting:
    """Tests for API key-based model routing (owner vs non-owner)."""

    @patch("app.routes.advice.OWNER_KEY_HASH", "abc123")
    def test_hash_api_key(self):
        """hash_api_key should produce consistent SHA-256 hashes."""
        from app.routes.advice import hash_api_key

        h1 = hash_api_key("sk-test-key")
        h2 = hash_api_key("sk-test-key")
        h3 = hash_api_key("sk-different-key")
        assert h1 == h2
        assert h1 != h3

    @patch("app.routes.advice.OWNER_KEY_HASH")
    def test_is_owner_key_match(self, mock_hash):
        """is_owner_key should return True when hashes match."""
        from app.routes.advice import hash_api_key, is_owner_key

        mock_hash.__eq__ = lambda self, other: other == hash_api_key("sk-owner-key")
        # Directly test the logic
        import app.routes.advice as advice_module

        advice_module.OWNER_KEY_HASH = hash_api_key("sk-owner-key")
        assert is_owner_key("sk-owner-key") is True
        assert is_owner_key("sk-other-key") is False

    def test_is_owner_key_no_hash_configured(self):
        """is_owner_key should return False when no owner hash is configured."""
        import app.routes.advice as advice_module

        original = advice_module.OWNER_KEY_HASH
        advice_module.OWNER_KEY_HASH = ""
        try:
            from app.routes.advice import is_owner_key

            assert is_owner_key("sk-any-key") is False
        finally:
            advice_module.OWNER_KEY_HASH = original

    @patch("app.routes.advice.is_owner_key", return_value=False)
    @patch("app.routes.advice.validate_question_relevance", return_value=True)
    @patch("app.routes.advice.call_base_model")
    def test_non_owner_uses_extended_prompt(
        self, mock_base, mock_validate, mock_is_owner
    ):
        """Non-owner requests should pass the extended system prompt to call_base_model."""
        from models.prompts import ADVICE_COLUMNIST_SYSTEM_PROMPT_EXTENDED

        mock_base.return_value = MOCK_DRAFT_RESPONSE

        client.post(
            "/api/advice",
            json={"question": "How do I talk to my partner?"},
            cookies=SESSION_COOKIES,
        )

        mock_base.assert_called_once()
        _, kwargs = mock_base.call_args
        assert kwargs.get("system_prompt") == ADVICE_COLUMNIST_SYSTEM_PROMPT_EXTENDED

    @patch("app.routes.advice.is_owner_key", return_value=True)
    @patch("app.routes.advice.validate_question_relevance", return_value=True)
    @patch("app.routes.advice.call_base_model")
    @patch("app.routes.advice.call_fine_tuned_model")
    def test_owner_uses_default_prompt(
        self, mock_fine_tuned, mock_base, mock_validate, mock_is_owner
    ):
        """Owner requests should not pass an extended system prompt to call_base_model."""
        mock_base.return_value = MOCK_DRAFT_RESPONSE
        mock_fine_tuned.return_value = MOCK_V3_RESPONSE

        client.post(
            "/api/advice",
            json={"question": "How do I talk to my partner?"},
            cookies=SESSION_COOKIES,
        )

        mock_base.assert_called_once()
        _, kwargs = mock_base.call_args
        assert kwargs.get("system_prompt") is None
