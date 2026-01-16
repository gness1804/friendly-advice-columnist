"""
API routes for the advice columnist.
"""

import time
from pydantic import BaseModel, Field
from fastapi import APIRouter, HTTPException, Form
from fastapi.responses import HTMLResponse

from models.openai_backend import generate_answer, BASE_MODEL, FINE_TUNED_MODEL
from qa.mvp_utils import extract_revised_response

router = APIRouter(tags=["advice"])

MAX_QUESTION_LENGTH = 4000
REQUEST_TIMEOUT = 45  # seconds


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
    elapsed_time: float = Field(..., description="Time taken to generate response in seconds")


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


def call_base_model(question: str, max_retries: int = 2) -> str:
    """
    Call the base model to generate a draft response.

    Args:
        question: User's question (will have QUESTION: prefix added)
        max_retries: Maximum number of retry attempts

    Returns:
        Draft response from base model

    Raises:
        RuntimeError: If all retry attempts fail
    """
    formatted_question = format_question_for_v1(question)

    last_error = None
    for attempt in range(max_retries + 1):
        try:
            response = generate_answer(formatted_question, version="v1", model=BASE_MODEL)
            return response
        except Exception as e:
            last_error = e
            if attempt < max_retries:
                wait_time = 2**attempt
                time.sleep(wait_time)

    raise RuntimeError(
        f"Failed to get response from base model after {max_retries + 1} attempts: {last_error}"
    )


def call_fine_tuned_model(question: str, draft_response: str) -> str:
    """
    Call the fine-tuned model to revise the draft response.

    Args:
        question: Original user question
        draft_response: Draft response from base model

    Returns:
        Full response from fine-tuned model

    Raises:
        RuntimeError: If the call fails
    """
    formatted_prompt = format_prompt_for_v3(question, draft_response)

    try:
        response = generate_answer(formatted_prompt, version="v3", model=FINE_TUNED_MODEL)
        return response
    except Exception as e:
        raise RuntimeError(f"Failed to get response from fine-tuned model: {e}")


@router.post("/advice", response_model=AdviceResponse)
async def get_advice(request: AdviceRequest) -> AdviceResponse:
    """
    Get advice for an interpersonal question.

    This endpoint chains two LLM calls:
    1. Base model generates a draft response
    2. Fine-tuned model revises and improves the response
    """
    question = request.question.strip()

    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    start_time = time.time()

    try:
        # Step 1: Call base model for draft
        draft_response = call_base_model(question)

        # Step 2: Call fine-tuned model to revise
        full_v3_response = call_fine_tuned_model(question, draft_response)

        # Step 3: Extract the revised response
        revised_response = extract_revised_response(full_v3_response)

        elapsed_time = time.time() - start_time

        return AdviceResponse(answer=revised_response, elapsed_time=elapsed_time)

    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")


@router.post("/advice/html", response_class=HTMLResponse)
async def get_advice_html(question: str = Form(...)) -> HTMLResponse:
    """
    Get advice as HTML fragment (for HTMX).

    Returns the response formatted as HTML for direct insertion into the page.
    Accepts form data for easy HTMX integration.
    """
    try:
        request = AdviceRequest(question=question)
        response = await get_advice(request)
        # Convert newlines to <br> and wrap in a div
        formatted_answer = response.answer.replace("\n", "<br>")
        html = f"""
        <div class="card">
            <h2 class="text-xl font-semibold text-primary mb-4">Advice</h2>
            <div class="text-text leading-relaxed">
                <p>{formatted_answer}</p>
            </div>
            <p class="text-sm text-text-dark mt-4">
                Response generated in {response.elapsed_time:.1f} seconds
            </p>
        </div>
        """
        return HTMLResponse(content=html)
    except HTTPException as e:
        error_html = f"""
        <div class="card border-primary">
            <p class="font-semibold text-primary">Error</p>
            <p class="text-text-muted mt-2">{e.detail}</p>
        </div>
        """
        return HTMLResponse(content=error_html, status_code=e.status_code)
