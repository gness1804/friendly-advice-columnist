"""
FastAPI application for the Friendly Advice Columnist.

Run with: uvicorn app.main:app --reload
"""

import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.routes.advice import router as advice_router


def get_version() -> str:
    """Read version from pyproject.toml."""
    pyproject_path = PROJECT_ROOT / "pyproject.toml"
    with open(pyproject_path, "rb") as f:
        data = tomllib.load(f)
    return data["project"]["version"]


app = FastAPI(
    title="Friendly Advice Columnist",
    description="AI-powered advice columnist for interpersonal questions",
    version=get_version(),
)

# Mount static files
app.mount(
    "/static",
    StaticFiles(directory=Path(__file__).parent / "static"),
    name="static",
)

# Set up templates
templates = Jinja2Templates(directory=Path(__file__).parent / "templates")

# Include API routes
app.include_router(advice_router, prefix="/api")


@app.get("/")
async def index(request: Request):
    """Render the main page."""
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "max_chars": 4000,
        },
    )


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}
