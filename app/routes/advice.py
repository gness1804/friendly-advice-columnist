"""
API routes for the advice columnist.
"""

import hashlib
import html
import os
import time
from pydantic import BaseModel, Field
from fastapi import APIRouter, HTTPException, Form, Request
from fastapi.responses import HTMLResponse
from openai import OpenAI, AuthenticationError
from slowapi import Limiter
from slowapi.util import get_remote_address

from models.openai_backend import generate_answer, BASE_MODEL, FINE_TUNED_MODEL
from models.prompts import (
    QUESTION_SCREENING_PROMPT,
    ADVICE_COLUMNIST_SYSTEM_PROMPT_EXTENDED,
)
from qa.mvp_utils import extract_revised_response
from app.session import require_api_key

limiter = Limiter(key_func=get_remote_address)

router = APIRouter(tags=["advice"])

MAX_QUESTION_LENGTH = 4000
REQUEST_TIMEOUT = 45  # seconds

# Hash of the owner's API key for comparison (set via env var)
OWNER_KEY_HASH = os.environ.get("OWNER_API_KEY_HASH", "")

# Error message for off-topic questions
OFF_TOPIC_ERROR = (
    "This question doesn't appear to be about interpersonal or relationship matters. "
    "Please submit a question about relationships, family, friends, workplace dynamics, "
    "or other interpersonal topics."
)


def hash_api_key(api_key: str) -> str:
    """Create a SHA-256 hash of an API key for comparison."""
    return hashlib.sha256(api_key.encode()).hexdigest()


def is_owner_key(api_key: str) -> bool:
    """Check if the provided API key belongs to the app owner."""
    if not OWNER_KEY_HASH:
        return False
    return hash_api_key(api_key) == OWNER_KEY_HASH


def validate_api_key(api_key: str) -> None:
    """
    Validate that an OpenAI API key is functional by making a lightweight call.

    Raises:
        HTTPException: If the key is invalid or the API is unreachable
    """
    try:
        client = OpenAI(api_key=api_key)
        client.models.list()
    except AuthenticationError:
        raise HTTPException(
            status_code=401,
            detail="Invalid OpenAI API key. Please check your key and try again.",
        )
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Could not validate API key: {e}",
        )


class AdviceRequest(BaseModel):
    """Request model for advice endpoint."""

    question: str = Field(
        ...,
        min_length=1,
        max_length=MAX_QUESTION_LENGTH,
        description="The interpersonal question to get advice on",
    )


class AdviceResponse(BaseModel):
    """Response model for advice endpoint."""

    answer: str = Field(..., description="The advice response")
    elapsed_time: float = Field(
        ..., description="Time taken to generate response in seconds"
    )
    used_fine_tuned: bool = Field(
        False, description="Whether the fine-tuned model was used"
    )


def format_question_for_v1(question: str) -> str:
    """Add QUESTION: prefix to user input for v1 model."""
    question = question.strip()
    if question.startswith("QUESTION:"):
        return question
    return f"QUESTION: {question}"


def format_prompt_for_v3(question: str, draft_response: str) -> str:
    """Format question and draft response for v3 model."""
    question = question.strip()
    draft_response = draft_response.strip()

    if question.startswith("QUESTION:"):
        question = question[9:].strip()

    return f"QUESTION: {question}\n\nDRAFT_RESPONSE: {draft_response}"


def validate_question_relevance(question: str, api_key: str) -> bool:
    """
    Check if the question is about interpersonal/relationship matters.

    Args:
        question: The user's question
        api_key: The user's OpenAI API key

    Returns:
        True if the question is relevant, False otherwise

    Raises:
        RuntimeError: If the screening call fails
    """
    client = OpenAI(api_key=api_key)

    try:
        response = client.chat.completions.create(
            model=BASE_MODEL,
            messages=[
                {"role": "system", "content": QUESTION_SCREENING_PROMPT},
                {"role": "user", "content": question},
            ],
            temperature=0.0,  # Deterministic for classification
            max_tokens=10,  # Only need one word
        )
        result = response.choices[0].message.content.strip().upper()
        return result == "RELEVANT"
    except Exception as e:
        raise RuntimeError(f"Failed to validate question: {e}")


def call_base_model(
    question: str, api_key: str, max_retries: int = 2, system_prompt: str = None
) -> str:
    """
    Call the base model to generate a draft response.

    Args:
        question: User's question (will have QUESTION: prefix added)
        api_key: The user's OpenAI API key
        max_retries: Maximum number of retry attempts
        system_prompt: Optional system prompt override

    Returns:
        Draft response from base model

    Raises:
        RuntimeError: If all retry attempts fail
    """
    formatted_question = format_question_for_v1(question)

    last_error = None
    for attempt in range(max_retries + 1):
        try:
            response = generate_answer(
                formatted_question,
                version="v1",
                model=BASE_MODEL,
                api_key=api_key,
                system_prompt=system_prompt,
            )
            return response
        except Exception as e:
            last_error = e
            if attempt < max_retries:
                wait_time = 2**attempt
                time.sleep(wait_time)

    raise RuntimeError(
        f"Failed to get response from base model after {max_retries + 1} attempts: {last_error}"
    )


def call_fine_tuned_model(question: str, draft_response: str, api_key: str) -> str:
    """
    Call the fine-tuned model to revise the draft response.

    Args:
        question: Original user question
        draft_response: Draft response from base model
        api_key: The user's OpenAI API key

    Returns:
        Full response from fine-tuned model

    Raises:
        RuntimeError: If the call fails
    """
    formatted_prompt = format_prompt_for_v3(question, draft_response)

    try:
        response = generate_answer(
            formatted_prompt, version="v3", model=FINE_TUNED_MODEL, api_key=api_key
        )
        return response
    except Exception as e:
        raise RuntimeError(f"Failed to get response from fine-tuned model: {e}")


@router.post("/advice", response_model=AdviceResponse)
@limiter.limit("10/minute")
async def get_advice(
    request: Request,
    body: AdviceRequest,
) -> AdviceResponse:
    """
    Get advice for an interpersonal question.

    This endpoint:
    1. Validates the user's API key
    2. Validates the question is about interpersonal matters
    3. Base model generates a draft response
    4. If using the owner's key, fine-tuned model revises the response
    5. Otherwise, returns the base model response directly
    """
    api_key = require_api_key(request)
    question = body.question.strip()

    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    # Validate question is about interpersonal matters (before timing starts)
    try:
        is_relevant = validate_question_relevance(question, api_key)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    if not is_relevant:
        raise HTTPException(status_code=400, detail=OFF_TOPIC_ERROR)

    start_time = time.time()
    owner = is_owner_key(api_key)

    try:
        if owner:
            # Owner: two-stage pipeline (base model draft + fine-tuned revision)
            draft_response = call_base_model(question, api_key)
            full_v3_response = call_fine_tuned_model(question, draft_response, api_key)
            revised_response = extract_revised_response(full_v3_response)
        else:
            # Non-owner: single-stage with extended prompt for richer voice
            draft_response = call_base_model(
                question, api_key, system_prompt=ADVICE_COLUMNIST_SYSTEM_PROMPT_EXTENDED
            )
            revised_response = draft_response
            # Strip "ANSWER: " prefix if present
            if revised_response.startswith("ANSWER:"):
                revised_response = revised_response[7:].strip()

        elapsed_time = time.time() - start_time

        return AdviceResponse(
            answer=revised_response, elapsed_time=elapsed_time, used_fine_tuned=owner
        )

    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"An unexpected error occurred: {e}"
        )


@router.post("/advice/html", response_class=HTMLResponse)
@limiter.limit("10/minute")
async def get_advice_html(
    request: Request,
    question: str = Form(...),
) -> HTMLResponse:
    """
    Get advice as HTML fragment (for HTMX).

    Returns the response formatted as HTML for direct insertion into the page.
    Accepts form data for easy HTMX integration.
    """
    try:
        advice_body = AdviceRequest(question=question)
        response = await get_advice(request, advice_body)
        # Escape HTML to prevent XSS, then convert newlines to <br>
        safe_answer = html.escape(response.answer).replace("\n", "<br>")
        model_note = " (fine-tuned)" if response.used_fine_tuned else ""
        html_content = f"""
        <div class="card">
            <h2 class="text-xl font-semibold text-primary mb-4">Advice</h2>
            <div class="text-text leading-relaxed">
                <p>{safe_answer}</p>
            </div>
            <p class="text-sm text-text-dark mt-4">
                Response generated in {response.elapsed_time:.1f} seconds{model_note}
            </p>
        </div>
        """
        return HTMLResponse(content=html_content)
    except HTTPException as e:
        safe_detail = html.escape(str(e.detail))
        error_html = f"""
        <div class="card border-primary">
            <p class="font-semibold text-primary">Error</p>
            <p class="text-text-muted mt-2">{safe_detail}</p>
        </div>
        """
        return HTMLResponse(content=error_html, status_code=e.status_code)
