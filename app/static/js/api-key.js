/**
 * API Key management for Friendly Advice Columnist.
 * Handles storing, retrieving, and injecting the user's OpenAI API key.
 */

const ApiKeyManager = {
    STORAGE_KEY: 'friendly_advice_api_key',

    /**
     * Get the stored API key from localStorage.
     */
    getKey() {
        return localStorage.getItem(this.STORAGE_KEY) || '';
    },

    /**
     * Save an API key to localStorage.
     */
    saveKey(key) {
        if (key) {
            localStorage.setItem(this.STORAGE_KEY, key);
        } else {
            localStorage.removeItem(this.STORAGE_KEY);
        }
    },

    /**
     * Check if an API key is stored.
     */
    hasKey() {
        return !!this.getKey();
    },

    /**
     * Clear the stored API key.
     */
    clearKey() {
        localStorage.removeItem(this.STORAGE_KEY);
    }
};

/**
 * API Key UI Manager
 */
const ApiKeyUI = {
    init() {
        this.bindEvents();
        this.updateSettingsIndicator();
        this.setupHtmxHeaders();

        // Show API key modal on first visit if no key stored
        if (!ApiKeyManager.hasKey()) {
            this.showModal();
        }
    },

    bindEvents() {
        const settingsBtn = document.getElementById('settings-btn');
        const cancelBtn = document.getElementById('cancel-api-key-btn');
        const saveBtn = document.getElementById('save-api-key-btn');
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

        if (modal) {
            modal.classList.remove('hidden');
            if (input) {
                input.value = ApiKeyManager.getKey();
                input.focus();
            }
            if (status) {
                status.classList.add('hidden');
            }
        }
    },

    hideModal() {
        const modal = document.getElementById('api-key-modal');
        if (modal) {
            modal.classList.add('hidden');
        }
    },

    saveKey() {
        const input = document.getElementById('api-key-input');
        const status = document.getElementById('api-key-status');

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

        ApiKeyManager.saveKey(key);
        this.updateSettingsIndicator();
        this.showStatus('API key saved successfully.', 'success');

        setTimeout(() => {
            this.hideModal();
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
    },

    /**
     * Configure HTMX to send the API key header with every request.
     */
    setupHtmxHeaders() {
        document.body.addEventListener('htmx:configRequest', function(evt) {
            const key = ApiKeyManager.getKey();
            if (key) {
                evt.detail.headers['X-OpenAI-API-Key'] = key;
            }
        });
    }
};

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    ApiKeyUI.init();
});

// Export for use in other scripts
window.ApiKeyManager = ApiKeyManager;
window.ApiKeyUI = ApiKeyUI;
