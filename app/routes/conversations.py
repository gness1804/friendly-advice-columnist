"""
API routes for conversation persistence (DynamoDB-backed).
"""

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from app.dynamodb import (
    save_conversation,
    get_conversations,
    delete_conversation,
    delete_all_conversations,
)
from app.session import require_api_key

router = APIRouter(tags=["conversations"])


class SaveConversationRequest(BaseModel):
    """Request model for saving a conversation."""

    session_id: str = Field(..., min_length=1, max_length=100)
    question: str = Field(..., min_length=1, max_length=10000)
    response: str = Field(..., min_length=1, max_length=50000)
    preview: str = Field(..., min_length=1, max_length=200)


@router.post("/conversations")
async def save(
    body: SaveConversationRequest,
    request: Request,
):
    """Save or update a conversation."""
    api_key = require_api_key(request)
    result = save_conversation(
        api_key=api_key,
        session_id=body.session_id,
        question=body.question,
        response=body.response,
        preview=body.preview,
    )
    if result is None:
        raise HTTPException(status_code=500, detail="Failed to save conversation.")
    return {"status": "ok"}


@router.get("/conversations")
async def list_conversations(
    request: Request,
):
    """Get all conversations for the current user."""
    api_key = require_api_key(request)
    items = get_conversations(api_key)
    return {"conversations": items}


@router.delete("/conversations/{session_id}")
async def delete(
    session_id: str,
    request: Request,
):
    """Delete a specific conversation."""
    api_key = require_api_key(request)
    success = delete_conversation(api_key, session_id)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to delete conversation.")
    return {"status": "ok"}


@router.delete("/conversations")
async def delete_all(
    request: Request,
):
    """Delete all conversations for the current user."""
    api_key = require_api_key(request)
    success = delete_all_conversations(api_key)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to delete conversations.")
    return {"status": "ok"}
