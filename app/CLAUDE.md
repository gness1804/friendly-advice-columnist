# CLAUDE.md - App Directory

This file documents the front-end web application architecture.

## Architecture Overview

The front-end uses a **server-side rendered architecture** with:
- **FastAPI** as the backend framework
- **Jinja2** for HTML templating
- **HTMX** for dynamic updates without page reloads
- **Tailwind CSS** for styling

## Directory Structure

```
app/
├── main.py                 # FastAPI app entry point, static/template mounting
├── CLAUDE.md               # This file
├── routes/
│   ├── __init__.py
│   └── advice.py           # API routes (/api/advice, /api/advice/html)
├── templates/
│   ├── base.html           # Base layout with sidebar and modal
│   └── index.html          # Main page with form and response area
└── static/
    ├── css/
    │   ├── input.css       # Tailwind source (directives + custom components)
    │   └── styles.css      # Compiled Tailwind CSS
    └── js/
        └── sessions.js     # Session/history management (localStorage)
```

## Key Files

### `main.py`
- FastAPI app initialization
- Mounts static files at `/static`
- Configures Jinja2 templates
- Includes advice router

### `routes/advice.py`
- `POST /api/advice` - JSON API endpoint
- `POST /api/advice/html` - HTML fragment endpoint (used by HTMX)
- Handles question validation and LLM pipeline calls

### `templates/base.html`
- Base layout with two-column structure (sidebar + main content)
- Desktop sidebar (`#sidebar`) and mobile sidebar (`#mobile-sidebar`)
- History list containers (`#history-list`, `#mobile-history-list`)
- Clear history confirmation modal (`#clear-confirm-modal`)
- Includes HTMX library and sessions.js

### `templates/index.html`
- Extends base.html
- Form with HTMX attributes for async submission
- Textarea with character counter
- Loading indicator and response area
- JavaScript for form handling (char count, submit state, keyboard shortcuts)

### `static/js/sessions.js`
- `SessionManager`: localStorage operations (CRUD for sessions)
- `UIManager`: UI interactions (sidebar, history list, modals)
- Sessions stored with: id, question, response, preview, createdAt, updatedAt

## Styling

### Tailwind Configuration (`tailwind.config.js`)
Custom color palette:
- `primary`: Red (#DC2626) with hover/light variants
- `surface`: Near-black (#0A0A0A) with light/lighter variants
- `text`: White/silver (#F5F5F5) with muted/dark variants

### Custom CSS Classes (`input.css`)
- `.btn-primary` - Red button with hover/disabled states
- `.input-field` - Styled textarea with focus ring
- `.card` - Container with border and padding
- `.loading-spinner` - CSS spin animation

## HTMX Usage

The form uses HTMX for dynamic submission:
```html
<form hx-post="/api/advice/html"
      hx-target="#response-area"
      hx-swap="innerHTML"
      hx-indicator="#loading-indicator">
```

## Session Storage

Conversations are persisted in localStorage:
- `friendly_advice_sessions`: Array of session objects
- `friendly_advice_current_session`: ID of active session

## Running the App

```bash
# Development server
uvicorn app.main:app --reload

# Rebuild Tailwind CSS (if modifying styles)
npx tailwindcss -i ./app/static/css/input.css -o ./app/static/css/styles.css --watch
```
