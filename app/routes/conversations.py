"""
API routes for conversation persistence (DynamoDB-backed).
"""

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field

from app.dynamodb import (
    save_conversation,
    get_conversations,
    delete_conversation,
    delete_all_conversations,
)

router = APIRouter(tags=["conversations"])


class SaveConversationRequest(BaseModel):
    """Request model for saving a conversation."""

    session_id: str = Field(..., min_length=1, max_length=100)
    question: str = Field(..., min_length=1, max_length=10000)
    response: str = Field(..., min_length=1, max_length=50000)
    preview: str = Field(..., min_length=1, max_length=200)


def _require_api_key(header_key: str | None) -> str:
    """Extract and validate the API key from the request header."""
    if not header_key:
        raise HTTPException(status_code=401, detail="API key required.")
    return header_key


@router.post("/conversations")
async def save(
    request: SaveConversationRequest,
    x_openai_api_key: str | None = Header(None),
):
    """Save or update a conversation."""
    api_key = _require_api_key(x_openai_api_key)
    result = save_conversation(
        api_key=api_key,
        session_id=request.session_id,
        question=request.question,
        response=request.response,
        preview=request.preview,
    )
    if result is None:
        raise HTTPException(status_code=500, detail="Failed to save conversation.")
    return {"status": "ok"}


@router.get("/conversations")
async def list_conversations(
    x_openai_api_key: str | None = Header(None),
):
    """Get all conversations for the current user."""
    api_key = _require_api_key(x_openai_api_key)
    items = get_conversations(api_key)
    return {"conversations": items}


@router.delete("/conversations/{session_id}")
async def delete(
    session_id: str,
    x_openai_api_key: str | None = Header(None),
):
    """Delete a specific conversation."""
    api_key = _require_api_key(x_openai_api_key)
    success = delete_conversation(api_key, session_id)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to delete conversation.")
    return {"status": "ok"}


@router.delete("/conversations")
async def delete_all(
    x_openai_api_key: str | None = Header(None),
):
    """Delete all conversations for the current user."""
    api_key = _require_api_key(x_openai_api_key)
    success = delete_all_conversations(api_key)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to delete conversations.")
    return {"status": "ok"}
