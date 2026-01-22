"""
Tests for the session history UI elements.

These tests verify that the HTML templates contain all necessary
elements for the session management feature. The actual JavaScript
functionality uses localStorage and would require browser-based
testing (e.g., Playwright) for full coverage.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


class TestSidebarElements:
    """Tests for sidebar UI elements in the templates."""

    def test_desktop_sidebar_present(self):
        """Index page should contain the desktop sidebar."""
        response = client.get("/")
        assert response.status_code == 200
        assert 'id="sidebar"' in response.text
        assert 'id="history-list"' in response.text

    def test_mobile_sidebar_present(self):
        """Index page should contain the mobile sidebar."""
        response = client.get("/")
        assert response.status_code == 200
        assert 'id="mobile-sidebar"' in response.text
        assert 'id="mobile-history-list"' in response.text

    def test_sidebar_toggle_button_present(self):
        """Index page should contain the mobile sidebar toggle button."""
        response = client.get("/")
        assert response.status_code == 200
        assert 'id="sidebar-toggle"' in response.text

    def test_sidebar_overlay_present(self):
        """Index page should contain the sidebar overlay for mobile."""
        response = client.get("/")
        assert response.status_code == 200
        assert 'id="sidebar-overlay"' in response.text

    def test_history_title_present(self):
        """Sidebar should have a History title."""
        response = client.get("/")
        assert response.status_code == 200
        assert "History" in response.text


class TestNewConversationButton:
    """Tests for the new conversation button."""

    def test_new_conversation_button_present(self):
        """Index page should contain the new conversation button."""
        response = client.get("/")
        assert response.status_code == 200
        assert 'id="new-conversation-btn"' in response.text
        assert "New Conversation" in response.text

    def test_mobile_new_conversation_button_present(self):
        """Index page should contain the mobile new conversation button."""
        response = client.get("/")
        assert response.status_code == 200
        assert 'id="mobile-new-conversation-btn"' in response.text


class TestClearHistoryButton:
    """Tests for the clear history button and modal."""

    def test_clear_history_button_present(self):
        """Index page should contain the clear history button."""
        response = client.get("/")
        assert response.status_code == 200
        assert 'id="clear-history-btn"' in response.text
        assert "Clear History" in response.text

    def test_mobile_clear_history_button_present(self):
        """Index page should contain the mobile clear history button."""
        response = client.get("/")
        assert response.status_code == 200
        assert 'id="mobile-clear-history-btn"' in response.text

    def test_clear_confirmation_modal_present(self):
        """Index page should contain the clear confirmation modal."""
        response = client.get("/")
        assert response.status_code == 200
        assert 'id="clear-confirm-modal"' in response.text
        assert "Clear History?" in response.text
        assert "permanently delete" in response.text

    def test_modal_buttons_present(self):
        """Clear confirmation modal should have cancel and confirm buttons."""
        response = client.get("/")
        assert response.status_code == 200
        assert 'id="cancel-clear-btn"' in response.text
        assert 'id="confirm-clear-btn"' in response.text
        assert "Cancel" in response.text
        assert "Clear All" in response.text


class TestSessionsScript:
    """Tests for the sessions JavaScript inclusion."""

    def test_sessions_script_included(self):
        """Index page should include the sessions.js script."""
        response = client.get("/")
        assert response.status_code == 200
        assert "/static/js/sessions.js" in response.text

    def test_sessions_script_loads(self):
        """Sessions.js should be accessible as a static file."""
        response = client.get("/static/js/sessions.js")
        assert response.status_code == 200
        assert "SessionManager" in response.text
        assert "UIManager" in response.text


class TestSessionsScriptContent:
    """Tests for the sessions.js script content."""

    def test_script_has_local_storage_key(self):
        """Sessions script should define localStorage keys."""
        response = client.get("/static/js/sessions.js")
        assert response.status_code == 200
        assert "friendly_advice_sessions" in response.text
        assert "friendly_advice_current_session" in response.text

    def test_script_has_crud_methods(self):
        """Sessions script should have CRUD methods."""
        response = client.get("/static/js/sessions.js")
        assert response.status_code == 200
        assert "getAllSessions" in response.text
        assert "createSession" in response.text
        assert "updateSession" in response.text
        assert "deleteSession" in response.text
        assert "clearAllSessions" in response.text

    def test_script_has_ui_methods(self):
        """Sessions script should have UI management methods."""
        response = client.get("/static/js/sessions.js")
        assert response.status_code == 200
        assert "renderHistoryList" in response.text
        assert "loadSession" in response.text
        assert "startNewConversation" in response.text
        assert "saveConversation" in response.text

    def test_script_has_modal_methods(self):
        """Sessions script should have modal control methods."""
        response = client.get("/static/js/sessions.js")
        assert response.status_code == 200
        assert "showClearConfirmation" in response.text
        assert "hideClearConfirmation" in response.text
        assert "confirmClearHistory" in response.text

    def test_script_has_mobile_sidebar_methods(self):
        """Sessions script should have mobile sidebar methods."""
        response = client.get("/static/js/sessions.js")
        assert response.status_code == 200
        assert "openMobileSidebar" in response.text
        assert "closeMobileSidebar" in response.text

    def test_script_escapes_html(self):
        """Sessions script should have XSS protection."""
        response = client.get("/static/js/sessions.js")
        assert response.status_code == 200
        assert "escapeHTML" in response.text


class TestLayoutStructure:
    """Tests for the overall layout structure."""

    def test_flex_layout_present(self):
        """Page should use flex layout for sidebar and main content."""
        response = client.get("/")
        assert response.status_code == 200
        # Check for flex container
        assert 'class="flex min-h-screen"' in response.text

    def test_main_content_area_present(self):
        """Page should have main content area alongside sidebar."""
        response = client.get("/")
        assert response.status_code == 200
        # The form and response area should still be present
        assert 'id="advice-form"' in response.text
        assert 'id="response-area"' in response.text

    def test_response_area_still_functional(self):
        """Response area should still have HTMX attributes."""
        response = client.get("/")
        assert response.status_code == 200
        assert 'hx-target="#response-area"' in response.text
        assert 'hx-swap="innerHTML"' in response.text
