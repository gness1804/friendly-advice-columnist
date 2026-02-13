/**
 * Session management for Friendly Advice Columnist
 * Handles saving, loading, and managing conversation history.
 * Uses localStorage as primary store and syncs to DynamoDB when an API key is available.
 */

const SessionManager = {
    STORAGE_KEY: 'friendly_advice_sessions',
    CURRENT_SESSION_KEY: 'friendly_advice_current_session',

    /**
     * Generate a unique session ID
     */
    generateId() {
        return Date.now().toString(36) + Math.random().toString(36).substr(2);
    },

    /**
     * Get all sessions from localStorage
     */
    getAllSessions() {
        try {
            const data = localStorage.getItem(this.STORAGE_KEY);
            return data ? JSON.parse(data) : [];
        } catch (e) {
            console.error('Error reading sessions from localStorage:', e);
            return [];
        }
    },

    /**
     * Save all sessions to localStorage
     */
    saveSessions(sessions) {
        try {
            localStorage.setItem(this.STORAGE_KEY, JSON.stringify(sessions));
        } catch (e) {
            console.error('Error saving sessions to localStorage:', e);
        }
    },

    /**
     * Get the current session ID
     */
    getCurrentSessionId() {
        return localStorage.getItem(this.CURRENT_SESSION_KEY);
    },

    /**
     * Set the current session ID
     */
    setCurrentSessionId(sessionId) {
        if (sessionId) {
            localStorage.setItem(this.CURRENT_SESSION_KEY, sessionId);
        } else {
            localStorage.removeItem(this.CURRENT_SESSION_KEY);
        }
    },

    /**
     * Get a specific session by ID
     */
    getSession(sessionId) {
        const sessions = this.getAllSessions();
        return sessions.find(s => s.id === sessionId);
    },

    /**
     * Create a new session
     */
    createSession(question, response) {
        const session = {
            id: this.generateId(),
            question: question,
            response: response,
            preview: this.createPreview(question),
            createdAt: new Date().toISOString(),
            updatedAt: new Date().toISOString()
        };

        const sessions = this.getAllSessions();
        sessions.unshift(session); // Add to beginning
        this.saveSessions(sessions);
        this.setCurrentSessionId(session.id);

        // Sync to DynamoDB in background
        this._syncSave(session);

        return session;
    },

    /**
     * Update an existing session
     */
    updateSession(sessionId, question, response) {
        const sessions = this.getAllSessions();
        const index = sessions.findIndex(s => s.id === sessionId);

        if (index !== -1) {
            sessions[index].question = question;
            sessions[index].response = response;
            sessions[index].preview = this.createPreview(question);
            sessions[index].updatedAt = new Date().toISOString();
            this.saveSessions(sessions);

            // Sync to DynamoDB in background
            this._syncSave(sessions[index]);

            return sessions[index];
        }

        return null;
    },

    /**
     * Delete a specific session
     */
    deleteSession(sessionId) {
        const sessions = this.getAllSessions();
        const filtered = sessions.filter(s => s.id !== sessionId);
        this.saveSessions(filtered);

        // Clear current session if it was deleted
        if (this.getCurrentSessionId() === sessionId) {
            this.setCurrentSessionId(null);
        }

        // Sync deletion to DynamoDB in background
        this._syncDelete(sessionId);
    },

    /**
     * Clear all sessions
     */
    clearAllSessions() {
        this.saveSessions([]);
        this.setCurrentSessionId(null);

        // Sync to DynamoDB in background
        this._syncDeleteAll();
    },

    /**
     * Create a preview string from the question (first 50 chars)
     */
    createPreview(question) {
        const maxLength = 50;
        if (question.length <= maxLength) {
            return question;
        }
        return question.substring(0, maxLength).trim() + '...';
    },

    /**
     * Format a date for display
     */
    formatDate(isoString) {
        const date = new Date(isoString);
        const now = new Date();
        const diffMs = now - date;
        const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24));

        if (diffDays === 0) {
            return 'Today';
        } else if (diffDays === 1) {
            return 'Yesterday';
        } else if (diffDays < 7) {
            return `${diffDays} days ago`;
        } else {
            return date.toLocaleDateString();
        }
    },

    /**
     * Load conversations from DynamoDB and merge with localStorage.
     * Called on init when an API key is available.
     */
    async loadFromServer() {
        if (typeof ApiKeyManager === 'undefined' || !ApiKeyManager.hasKey()) {
            return;
        }

        try {
            const response = await fetch('/api/conversations', {
                headers: { 'X-OpenAI-API-Key': ApiKeyManager.getKey() }
            });

            if (!response.ok) return;

            const data = await response.json();
            const serverSessions = (data.conversations || []).map(item => ({
                id: item.session_id,
                question: item.question,
                response: item.response,
                preview: item.preview,
                createdAt: item.updated_at,
                updatedAt: item.updated_at
            }));

            // Merge: server sessions that aren't in localStorage get added
            const localSessions = this.getAllSessions();
            const localIds = new Set(localSessions.map(s => s.id));

            for (const session of serverSessions) {
                if (!localIds.has(session.id)) {
                    localSessions.push(session);
                }
            }

            // Sort by updatedAt descending
            localSessions.sort((a, b) => new Date(b.updatedAt) - new Date(a.updatedAt));
            this.saveSessions(localSessions);
        } catch (e) {
            console.error('Failed to load conversations from server:', e);
        }
    },

    // --- DynamoDB sync helpers (fire-and-forget) ---

    _getHeaders() {
        const headers = { 'Content-Type': 'application/json' };
        if (typeof ApiKeyManager !== 'undefined' && ApiKeyManager.hasKey()) {
            headers['X-OpenAI-API-Key'] = ApiKeyManager.getKey();
        }
        return headers;
    },

    _syncSave(session) {
        if (typeof ApiKeyManager === 'undefined' || !ApiKeyManager.hasKey()) return;

        fetch('/api/conversations', {
            method: 'POST',
            headers: this._getHeaders(),
            body: JSON.stringify({
                session_id: session.id,
                question: session.question,
                response: session.response,
                preview: session.preview
            })
        }).catch(e => console.error('Failed to sync save:', e));
    },

    _syncDelete(sessionId) {
        if (typeof ApiKeyManager === 'undefined' || !ApiKeyManager.hasKey()) return;

        fetch(`/api/conversations/${sessionId}`, {
            method: 'DELETE',
            headers: this._getHeaders()
        }).catch(e => console.error('Failed to sync delete:', e));
    },

    _syncDeleteAll() {
        if (typeof ApiKeyManager === 'undefined' || !ApiKeyManager.hasKey()) return;

        fetch('/api/conversations', {
            method: 'DELETE',
            headers: this._getHeaders()
        }).catch(e => console.error('Failed to sync delete all:', e));
    }
};

/**
 * UI Manager for handling sidebar and history display
 */
const UIManager = {
    /**
     * Initialize the UI
     */
    init() {
        this.bindEvents();
        this.renderHistoryList();
        this.setupMobileSidebar();

        // Load server-side conversations and re-render
        SessionManager.loadFromServer().then(() => {
            this.renderHistoryList();
        });
    },

    /**
     * Bind event listeners
     */
    bindEvents() {
        // New conversation buttons
        const newConvBtn = document.getElementById('new-conversation-btn');
        const mobileNewConvBtn = document.getElementById('mobile-new-conversation-btn');

        if (newConvBtn) {
            newConvBtn.addEventListener('click', () => this.startNewConversation());
        }
        if (mobileNewConvBtn) {
            mobileNewConvBtn.addEventListener('click', () => {
                this.startNewConversation();
                this.closeMobileSidebar();
            });
        }

        // Clear history buttons
        const clearBtn = document.getElementById('clear-history-btn');
        const mobileClearBtn = document.getElementById('mobile-clear-history-btn');

        if (clearBtn) {
            clearBtn.addEventListener('click', () => this.showClearConfirmation());
        }
        if (mobileClearBtn) {
            mobileClearBtn.addEventListener('click', () => this.showClearConfirmation());
        }

        // Modal buttons
        const cancelClearBtn = document.getElementById('cancel-clear-btn');
        const confirmClearBtn = document.getElementById('confirm-clear-btn');
        const modalBackdrop = document.getElementById('modal-backdrop');

        if (cancelClearBtn) {
            cancelClearBtn.addEventListener('click', () => this.hideClearConfirmation());
        }
        if (confirmClearBtn) {
            confirmClearBtn.addEventListener('click', () => this.confirmClearHistory());
        }
        if (modalBackdrop) {
            modalBackdrop.addEventListener('click', () => this.hideClearConfirmation());
        }

        // Escape key to close modal
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                this.hideClearConfirmation();
                this.closeMobileSidebar();
            }
        });
    },

    /**
     * Setup mobile sidebar toggle
     */
    setupMobileSidebar() {
        const toggleBtn = document.getElementById('sidebar-toggle');
        const closeBtn = document.getElementById('close-sidebar-btn');
        const overlay = document.getElementById('sidebar-overlay');

        if (toggleBtn) {
            toggleBtn.addEventListener('click', () => this.openMobileSidebar());
        }
        if (closeBtn) {
            closeBtn.addEventListener('click', () => this.closeMobileSidebar());
        }
        if (overlay) {
            overlay.addEventListener('click', () => this.closeMobileSidebar());
        }
    },

    /**
     * Open mobile sidebar
     */
    openMobileSidebar() {
        const sidebar = document.getElementById('mobile-sidebar');
        const overlay = document.getElementById('sidebar-overlay');

        if (sidebar && overlay) {
            sidebar.classList.remove('hidden');
            sidebar.classList.add('flex');
            overlay.classList.remove('hidden');
            // Trigger animation
            requestAnimationFrame(() => {
                sidebar.classList.remove('-translate-x-full');
            });
        }
    },

    /**
     * Close mobile sidebar
     */
    closeMobileSidebar() {
        const sidebar = document.getElementById('mobile-sidebar');
        const overlay = document.getElementById('sidebar-overlay');

        if (sidebar && overlay) {
            sidebar.classList.add('-translate-x-full');
            setTimeout(() => {
                sidebar.classList.add('hidden');
                sidebar.classList.remove('flex');
                overlay.classList.add('hidden');
            }, 300);
        }
    },

    /**
     * Render the history list in both desktop and mobile sidebars
     */
    renderHistoryList() {
        const sessions = SessionManager.getAllSessions();
        const currentSessionId = SessionManager.getCurrentSessionId();

        const historyList = document.getElementById('history-list');
        const mobileHistoryList = document.getElementById('mobile-history-list');

        const html = this.generateHistoryHTML(sessions, currentSessionId);

        if (historyList) {
            historyList.innerHTML = html;
            this.bindHistoryItemEvents(historyList);
        }
        if (mobileHistoryList) {
            mobileHistoryList.innerHTML = html;
            this.bindHistoryItemEvents(mobileHistoryList);
        }
    },

    /**
     * Generate HTML for history items
     */
    generateHistoryHTML(sessions, currentSessionId) {
        if (sessions.length === 0) {
            return `
                <div class="text-center py-8 text-text-dark text-sm">
                    <p>No conversations yet</p>
                    <p class="mt-1">Ask a question to get started</p>
                </div>
            `;
        }

        return sessions.map(session => {
            const isActive = session.id === currentSessionId;
            const activeClass = isActive ? 'bg-surface-lighter border-primary' : 'border-transparent hover:bg-surface';

            return `
                <div class="history-item p-3 mb-1 rounded-md cursor-pointer border ${activeClass} transition-colors group"
                     data-session-id="${session.id}">
                    <div class="flex justify-between items-start">
                        <p class="text-sm text-text truncate flex-1 pr-2">${this.escapeHTML(session.preview)}</p>
                        <button class="delete-session-btn opacity-0 group-hover:opacity-100 p-1 hover:bg-surface-lighter rounded transition-opacity"
                                data-session-id="${session.id}"
                                title="Delete conversation">
                            <svg class="w-4 h-4 text-text-dark hover:text-primary" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"></path>
                            </svg>
                        </button>
                    </div>
                    <p class="text-xs text-text-dark mt-1">${SessionManager.formatDate(session.createdAt)}</p>
                </div>
            `;
        }).join('');
    },

    /**
     * Bind click events to history items
     */
    bindHistoryItemEvents(container) {
        // Click on history item to load it
        container.querySelectorAll('.history-item').forEach(item => {
            item.addEventListener('click', (e) => {
                // Don't trigger if clicking the delete button
                if (e.target.closest('.delete-session-btn')) {
                    return;
                }
                const sessionId = item.dataset.sessionId;
                this.loadSession(sessionId);
                this.closeMobileSidebar();
            });
        });

        // Delete session button
        container.querySelectorAll('.delete-session-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                e.stopPropagation();
                const sessionId = btn.dataset.sessionId;
                this.deleteSession(sessionId);
            });
        });
    },

    /**
     * Load a session into the main view
     */
    loadSession(sessionId) {
        const session = SessionManager.getSession(sessionId);
        if (!session) return;

        SessionManager.setCurrentSessionId(sessionId);

        // Update the textarea with the question
        const textarea = document.getElementById('question');
        if (textarea) {
            textarea.value = session.question;
            // Trigger character count update
            if (typeof updateCharCount === 'function') {
                updateCharCount();
            }
        }

        // Update the response area
        const responseArea = document.getElementById('response-area');
        if (responseArea && session.response) {
            responseArea.innerHTML = session.response;
        }

        // Re-render to update active state
        this.renderHistoryList();
    },

    /**
     * Start a new conversation
     */
    startNewConversation() {
        SessionManager.setCurrentSessionId(null);

        // Clear the form
        const textarea = document.getElementById('question');
        if (textarea) {
            textarea.value = '';
            if (typeof updateCharCount === 'function') {
                updateCharCount();
            }
        }

        // Clear the response area
        const responseArea = document.getElementById('response-area');
        if (responseArea) {
            responseArea.innerHTML = '';
        }

        // Re-render to update active state
        this.renderHistoryList();
    },

    /**
     * Delete a specific session
     */
    deleteSession(sessionId) {
        const currentId = SessionManager.getCurrentSessionId();
        SessionManager.deleteSession(sessionId);

        // If we deleted the current session, start a new one
        if (currentId === sessionId) {
            this.startNewConversation();
        }

        this.renderHistoryList();
    },

    /**
     * Show the clear history confirmation modal
     */
    showClearConfirmation() {
        const modal = document.getElementById('clear-confirm-modal');
        if (modal) {
            modal.classList.remove('hidden');
        }
    },

    /**
     * Hide the clear history confirmation modal
     */
    hideClearConfirmation() {
        const modal = document.getElementById('clear-confirm-modal');
        if (modal) {
            modal.classList.add('hidden');
        }
    },

    /**
     * Confirm and execute clear history
     */
    confirmClearHistory() {
        SessionManager.clearAllSessions();
        this.startNewConversation();
        this.renderHistoryList();
        this.hideClearConfirmation();
        this.closeMobileSidebar();
    },

    /**
     * Save a conversation (called after receiving a response)
     */
    saveConversation(question, responseHTML) {
        const currentSessionId = SessionManager.getCurrentSessionId();

        if (currentSessionId) {
            // Update existing session
            SessionManager.updateSession(currentSessionId, question, responseHTML);
        } else {
            // Create new session
            SessionManager.createSession(question, responseHTML);
        }

        this.renderHistoryList();
    },

    /**
     * Escape HTML to prevent XSS
     */
    escapeHTML(str) {
        const div = document.createElement('div');
        div.textContent = str;
        return div.innerHTML;
    }
};

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    UIManager.init();
});

// Export for use in other scripts
window.SessionManager = SessionManager;
window.UIManager = UIManager;
