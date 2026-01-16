"""
Tests for the advice API endpoints.
"""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.main import app

client = TestClient(app)


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


class TestAdviceAPIEndpoint:
    """Tests for the JSON advice API endpoint."""

    @patch("app.routes.advice.validate_question_relevance")
    @patch("app.routes.advice.call_base_model")
    @patch("app.routes.advice.call_fine_tuned_model")
    def test_advice_endpoint_success(self, mock_fine_tuned, mock_base, mock_validate):
        """Advice endpoint should return a response for valid questions."""
        mock_validate.return_value = True
        mock_base.return_value = MOCK_DRAFT_RESPONSE
        mock_fine_tuned.return_value = MOCK_V3_RESPONSE

        response = client.post(
            "/api/advice",
            json={"question": "How do I talk to my partner about finances?"},
        )

        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "elapsed_time" in data
        assert "thoughtful advice" in data["answer"].lower()

    def test_advice_endpoint_empty_question(self):
        """Advice endpoint should reject empty questions."""
        response = client.post("/api/advice", json={"question": ""})
        assert response.status_code == 422  # Validation error

    def test_advice_endpoint_missing_question(self):
        """Advice endpoint should reject requests without question."""
        response = client.post("/api/advice", json={})
        assert response.status_code == 422

    def test_advice_endpoint_question_too_long(self):
        """Advice endpoint should reject questions exceeding max length."""
        long_question = "x" * 4001
        response = client.post("/api/advice", json={"question": long_question})
        assert response.status_code == 422


class TestAdviceHTMLEndpoint:
    """Tests for the HTML advice endpoint (HTMX)."""

    @patch("app.routes.advice.validate_question_relevance")
    @patch("app.routes.advice.call_base_model")
    @patch("app.routes.advice.call_fine_tuned_model")
    def test_advice_html_endpoint_success(self, mock_fine_tuned, mock_base, mock_validate):
        """HTML endpoint should return HTML fragment for valid questions."""
        mock_validate.return_value = True
        mock_base.return_value = MOCK_DRAFT_RESPONSE
        mock_fine_tuned.return_value = MOCK_V3_RESPONSE

        response = client.post(
            "/api/advice/html",
            data={"question": "How do I set boundaries with family?"},
        )

        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "Advice" in response.text
        assert "thoughtful advice" in response.text.lower()

    def test_advice_html_endpoint_empty_question(self):
        """HTML endpoint should return error HTML for empty questions."""
        response = client.post("/api/advice/html", data={"question": ""})
        # FastAPI returns 422 for validation errors even with HTML
        assert response.status_code == 422

    @patch("app.routes.advice.validate_question_relevance")
    @patch("app.routes.advice.call_base_model")
    def test_advice_html_endpoint_api_error(self, mock_base, mock_validate):
        """HTML endpoint should return error HTML when API fails."""
        mock_validate.return_value = True
        mock_base.side_effect = RuntimeError("API connection failed")

        response = client.post(
            "/api/advice/html",
            data={"question": "How do I handle conflict?"},
        )

        assert response.status_code == 503
        assert "Error" in response.text


class TestInputValidation:
    """Tests for input validation."""

    @patch("app.routes.advice.validate_question_relevance")
    @patch("app.routes.advice.call_base_model")
    @patch("app.routes.advice.call_fine_tuned_model")
    def test_question_with_special_characters(self, mock_fine_tuned, mock_base, mock_validate):
        """Questions with special characters should be handled."""
        mock_validate.return_value = True
        mock_base.return_value = MOCK_DRAFT_RESPONSE
        mock_fine_tuned.return_value = MOCK_V3_RESPONSE

        response = client.post(
            "/api/advice",
            json={"question": "My partner & I aren't communicating well. What should I do?"},
        )

        assert response.status_code == 200

    @patch("app.routes.advice.validate_question_relevance")
    @patch("app.routes.advice.call_base_model")
    @patch("app.routes.advice.call_fine_tuned_model")
    def test_question_with_newlines(self, mock_fine_tuned, mock_base, mock_validate):
        """Questions with newlines should be handled."""
        mock_validate.return_value = True
        mock_base.return_value = MOCK_DRAFT_RESPONSE
        mock_fine_tuned.return_value = MOCK_V3_RESPONSE

        response = client.post(
            "/api/advice",
            json={"question": "Line 1\nLine 2\nLine 3"},
        )

        assert response.status_code == 200

    def test_question_at_max_length(self):
        """Questions at exactly max length should be accepted."""
        # This test verifies the boundary condition
        max_question = "x" * 4000

        # We just test that validation passes - actual API call would need mocking
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
        )

        assert response.status_code == 400
        assert "Error" in response.text
        assert "interpersonal" in response.text.lower()

    @patch("app.routes.advice.validate_question_relevance")
    @patch("app.routes.advice.call_base_model")
    @patch("app.routes.advice.call_fine_tuned_model")
    def test_relevant_question_processed(self, mock_fine_tuned, mock_base, mock_validate):
        """Relevant questions should be processed normally."""
        mock_validate.return_value = True
        mock_base.return_value = MOCK_DRAFT_RESPONSE
        mock_fine_tuned.return_value = MOCK_V3_RESPONSE

        response = client.post(
            "/api/advice",
            json={"question": "My mother-in-law keeps criticizing my parenting. What should I do?"},
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
        )

        assert response.status_code == 503
        assert "error" in response.json()["detail"].lower()
