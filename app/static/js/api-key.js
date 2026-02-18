/**
 * API Key management for Friendly Advice Columnist.
 *
 * Stores the API key server-side in an encrypted httpOnly cookie via
 * POST /api/session.  The key is never held in localStorage or accessible
 * to client-side JavaScript after the initial submission.
 */

const ApiKeyManager = {
    /** Whether the server has confirmed an active session. */
    _hasSession: false,
    /** Masked key hint returned by the server (e.g. "sk-abc...wxyz"). */
    _maskedKey: '',

    /**
     * Check session status with the server on page load.
     * Returns a promise that resolves to true/false.
     */
    async checkStatus() {
        try {
            const res = await fetch('/api/session/status', { credentials: 'same-origin' });
            if (!res.ok) {
                this._hasSession = false;
                this._maskedKey = '';
                return false;
            }
            const data = await res.json();
            this._hasSession = data.has_key;
            this._maskedKey = data.masked_key || '';
            return this._hasSession;
        } catch {
            this._hasSession = false;
            this._maskedKey = '';
            return false;
        }
    },

    /**
     * Send the API key to the server for secure storage.
     * Returns a promise resolving to { ok: true } or { ok: false, error: string }.
     */
    async saveKey(key) {
        try {
            const res = await fetch('/api/session', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                credentials: 'same-origin',
                body: JSON.stringify({ api_key: key }),
            });
            if (!res.ok) {
                const data = await res.json().catch(() => ({}));
                return { ok: false, error: data.detail || 'Failed to save API key.' };
            }
            this._hasSession = true;
            this._maskedKey = key.length > 7 ? key.substring(0, 7) + '...' + key.slice(-4) : '';
            return { ok: true };
        } catch {
            return { ok: false, error: 'Network error. Please try again.' };
        }
    },

    /**
     * Clear the server-side session (logout).
     */
    async clearKey() {
        try {
            await fetch('/api/session', {
                method: 'DELETE',
                credentials: 'same-origin',
            });
        } catch { /* ignore */ }
        this._hasSession = false;
        this._maskedKey = '';
    },

    hasKey() {
        return this._hasSession;
    },

    getMaskedKey() {
        return this._maskedKey;
    }
};

/**
 * API Key UI Manager
 */
const ApiKeyUI = {
    async init() {
        this.bindEvents();

        // Clean up legacy localStorage key from pre-session versions
        localStorage.removeItem('friendly_advice_api_key');

        // Check server-side session status (async)
        const hasKey = await ApiKeyManager.checkStatus();
        this.updateSettingsIndicator();

        // Notify the page that the session status is now known
        document.dispatchEvent(new Event('apiKeyStatusReady'));

        // Show modal on first visit if no session exists
        if (!hasKey) {
            this.showModal();
        }
    },

    bindEvents() {
        const settingsBtn = document.getElementById('settings-btn');
        const cancelBtn = document.getElementById('cancel-api-key-btn');
        const saveBtn = document.getElementById('save-api-key-btn');
        const logoutBtn = document.getElementById('logout-api-key-btn');
        const backdrop = document.getElementById('api-key-modal-backdrop');
        const input = document.getElementById('api-key-input');

        if (settingsBtn) {
            settingsBtn.addEventListener('click', () => this.showModal());
        }
        if (cancelBtn) {
            cancelBtn.addEventListener('click', () => this.hideModal());
        }
        if (saveBtn) {
            saveBtn.addEventListener('click', () => this.saveKey());
        }
        if (logoutBtn) {
            logoutBtn.addEventListener('click', () => this.logout());
        }
        if (backdrop) {
            backdrop.addEventListener('click', () => this.hideModal());
        }
        if (input) {
            input.addEventListener('keydown', (e) => {
                if (e.key === 'Enter') {
                    this.saveKey();
                }
            });
        }
    },

    showModal() {
        const modal = document.getElementById('api-key-modal');
        const input = document.getElementById('api-key-input');
        const status = document.getElementById('api-key-status');
        const logoutBtn = document.getElementById('logout-api-key-btn');

        if (modal) {
            modal.classList.remove('hidden');
            if (input) {
                // Don't pre-fill — the key is no longer accessible to JS
                input.value = '';
                input.placeholder = ApiKeyManager.hasKey()
                    ? ApiKeyManager.getMaskedKey() || 'sk-... (key stored on server)'
                    : 'sk-...';
                input.focus();
            }
            if (status) {
                status.classList.add('hidden');
            }
            // Show/hide logout button based on session state
            if (logoutBtn) {
                if (ApiKeyManager.hasKey()) {
                    logoutBtn.classList.remove('hidden');
                } else {
                    logoutBtn.classList.add('hidden');
                }
            }
        }
    },

    hideModal() {
        const modal = document.getElementById('api-key-modal');
        if (modal) {
            modal.classList.add('hidden');
        }
    },

    async saveKey() {
        const input = document.getElementById('api-key-input');
        if (!input) return;

        const key = input.value.trim();

        if (!key) {
            this.showStatus('Please enter an API key.', 'error');
            return;
        }

        if (!key.startsWith('sk-')) {
            this.showStatus('API key should start with "sk-".', 'error');
            return;
        }

        this.showStatus('Saving...', 'info');
        const result = await ApiKeyManager.saveKey(key);

        if (!result.ok) {
            this.showStatus(result.error, 'error');
            return;
        }

        // Clear the input immediately so the key isn't sitting in the DOM
        input.value = '';
        this.updateSettingsIndicator();
        this.showStatus('API key saved securely.', 'success');

        setTimeout(() => {
            this.hideModal();
        }, 1000);
    },

    async logout() {
        await ApiKeyManager.clearKey();
        this.updateSettingsIndicator();
        this.showStatus('API key removed.', 'success');

        setTimeout(() => {
            this.hideModal();
            // Re-check the prompt banner
            if (typeof checkApiKeyPrompt === 'function') {
                checkApiKeyPrompt();
            }
        }, 1000);
    },

    showStatus(message, type) {
        const status = document.getElementById('api-key-status');
        if (!status) return;

        status.classList.remove('hidden');
        status.textContent = message;
        status.className = 'mb-4 text-sm';

        if (type === 'error') {
            status.classList.add('text-primary');
        } else {
            status.classList.add('text-green-400');
        }
    },

    updateSettingsIndicator() {
        const btn = document.getElementById('settings-btn');
        if (!btn) return;

        if (ApiKeyManager.hasKey()) {
            btn.title = 'API Key Settings (key saved)';
        } else {
            btn.title = 'API Key Settings (no key set)';
        }
    }
};

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    ApiKeyUI.init();
});

// Export for use in other scripts
window.ApiKeyManager = ApiKeyManager;
window.ApiKeyUI = ApiKeyUI;
