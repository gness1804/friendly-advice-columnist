#!/usr/bin/env python3
"""
MVP Version 2: Advice Column AI Application

This script chains two LLM calls:
1. Base model (GPT-4.1-mini) generates a draft response
2. Fine-tuned model revises and improves the response

Usage:
    python qa/advice_mvp.py --question "My partner and I are having communication issues"
    python qa/advice_mvp.py --question "How do I set boundaries?" --verbose
    python qa/advice_mvp.py --question "Question here" --no-save-output
"""

import os
import sys
import argparse
import time
from pathlib import Path
from datetime import datetime

# Add project root to path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.openai_backend import generate_answer, BASE_MODEL, FINE_TUNED_MODEL
from qa.mvp_utils import parse_v3_response, extract_revised_response


def validate_environment() -> None:
    """Validate that required environment variables are set."""
    errors = []
    
    if not os.environ.get("OPENAI_API_KEY"):
        errors.append("OPENAI_API_KEY is required")
    
    # Models have defaults, so we just inform if they're not explicitly set
    if not os.environ.get("BASE_MODEL") and not os.environ.get("OPENAI_MODEL"):
        print("💡 Note: BASE_MODEL not set, using default: gpt-4.1-mini")
    
    if not os.environ.get("FINE_TUNED_MODEL") and not os.environ.get("OPENAI_MODEL"):
        print("💡 Note: FINE_TUNED_MODEL not set, using default fine-tuned model")
    
    if errors:
        print("❌ Error: Missing required environment variables:")
        for error in errors:
            print(f"   - {error}")
        print("\n💡 Tip: Set these in your .env file or export them before running.")
        sys.exit(1)


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
    
    # Remove QUESTION: prefix if present (we'll add it back)
    if question.startswith("QUESTION:"):
        question = question[9:].strip()
    
    return f"QUESTION: {question}\n\nDRAFT_RESPONSE: {draft_response}"


def call_base_model(question: str, max_retries: int = 2, verbose: bool = False) -> str:
    """
    Call the base model to generate a draft response.
    
    Args:
        question: User's question (will have QUESTION: prefix added)
        max_retries: Maximum number of retry attempts
        verbose: Print debug information
        
    Returns:
        Draft response from base model
        
    Raises:
        RuntimeError: If all retry attempts fail
    """
    formatted_question = format_question_for_v1(question)
    
    if verbose:
        print(f"\n📝 Calling base model ({BASE_MODEL})...")
        print(f"   Question: {question[:100]}...")
    
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            response = generate_answer(formatted_question, version="v1", model=BASE_MODEL)
            if verbose:
                print(f"✅ Base model response received (attempt {attempt + 1})")
            return response
        except Exception as e:
            last_error = e
            if attempt < max_retries:
                wait_time = 2 ** attempt  # Exponential backoff
                if verbose:
                    print(f"⚠️  Attempt {attempt + 1} failed: {e}")
                    print(f"   Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                if verbose:
                    print(f"❌ All {max_retries + 1} attempts failed")
    
    raise RuntimeError(f"Failed to get response from base model after {max_retries + 1} attempts: {last_error}")


def call_fine_tuned_model(question: str, draft_response: str, verbose: bool = False) -> str:
    """
    Call the fine-tuned model to revise the draft response.
    
    Args:
        question: Original user question
        draft_response: Draft response from base model
        verbose: Print debug information
        
    Returns:
        Full response from fine-tuned model (includes SCORE, STRENGTHS, WEAKNESSES, REVISED_RESPONSE)
        
    Raises:
        RuntimeError: If the call fails
    """
    formatted_prompt = format_prompt_for_v3(question, draft_response)
    
    if verbose:
        print(f"\n🔧 Calling fine-tuned model ({FINE_TUNED_MODEL})...")
    
    try:
        response = generate_answer(formatted_prompt, version="v3", model=FINE_TUNED_MODEL)
        if verbose:
            print("✅ Fine-tuned model response received")
        return response
    except Exception as e:
        raise RuntimeError(f"Failed to get response from fine-tuned model: {e}")


def format_elapsed_time(seconds: float) -> str:
    """
    Format elapsed time in a human-readable format.
    
    Args:
        seconds: Elapsed time in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.2f} seconds"
    else:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes} minute{'s' if minutes != 1 else ''} {secs:.2f} seconds"


def save_output(
    question: str,
    draft_response: str,
    full_v3_response: str,
    revised_response: str,
    output_dir: str,
    elapsed_time: float,
    verbose: bool = False
) -> str:
    """
    Save the complete output to a file.
    
    Args:
        question: Original user question
        draft_response: Draft response from base model
        full_v3_response: Full response from fine-tuned model
        revised_response: Extracted revised response
        output_dir: Directory to save output
        verbose: Print debug information
        
    Returns:
        Path to saved file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"mvp_output_{timestamp}.txt"
    output_path = os.path.join(output_dir, output_filename)
    
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("MVP ADVICE COLUMN OUTPUT\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total execution time: {format_elapsed_time(elapsed_time)}\n")
            f.write(f"Base Model: {BASE_MODEL}\n")
            f.write(f"Fine-tuned Model: {FINE_TUNED_MODEL}\n\n")
            
            f.write("=" * 80 + "\n")
            f.write("ORIGINAL QUESTION\n")
            f.write("=" * 80 + "\n")
            f.write(f"{question}\n\n")
            
            f.write("=" * 80 + "\n")
            f.write("BASE MODEL DRAFT RESPONSE\n")
            f.write("=" * 80 + "\n")
            f.write(f"{draft_response}\n\n")
            
            f.write("=" * 80 + "\n")
            f.write("FINE-TUNED MODEL FULL RESPONSE\n")
            f.write("=" * 80 + "\n")
            f.write(f"{full_v3_response}\n\n")
            
            f.write("=" * 80 + "\n")
            f.write("EXTRACTED REVISED RESPONSE (USER OUTPUT)\n")
            f.write("=" * 80 + "\n")
            f.write(f"{revised_response}\n")
            f.write("=" * 80 + "\n")
        
        if verbose:
            print(f"\n💾 Output saved to: {output_path}")
        else:
            print(f"\n💾 Output saved to: {output_path}")
        
        return output_path
    except Exception as e:
        print(f"\n⚠️  Error saving output: {e}")
        return ""


def main():
    parser = argparse.ArgumentParser(
        description='MVP Advice Column AI - Chains base and fine-tuned models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python qa/advice_mvp.py --question "My partner and I are having communication issues"
  python qa/advice_mvp.py --question "How do I set boundaries?" --verbose
  python qa/advice_mvp.py --question "Question here" --no-save-output
  python qa/advice_mvp.py --question "Family conflict" --output-dir outputs/custom
        """
    )
    
    parser.add_argument(
        '--question',
        '--prompt',
        type=str,
        required=True,
        help='User question (no trigger words needed - will be added automatically)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs/inference/mvp',
        help='Directory to save output (default: outputs/inference/mvp)'
    )
    
    parser.add_argument(
        '--save-output',
        action='store_true',
        default=True,
        help='Save output to file (default: True)'
    )
    
    parser.add_argument(
        '--no-save-output',
        dest='save_output',
        action='store_false',
        help='Disable saving output to file'
    )
    
    parser.add_argument(
        '--max-retries',
        type=int,
        default=2,
        help='Maximum number of retries for base model calls (default: 2)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Show debug information including scores, strengths, and weaknesses'
    )
    
    args = parser.parse_args()
    
    # Validate environment variables
    validate_environment()
    
    question = args.question.strip()
    
    if not question:
        print("❌ Error: Question cannot be empty")
        sys.exit(1)
    
    print("\n" + "=" * 80)
    print("MVP ADVICE COLUMN AI")
    print("=" * 80)
    print(f"\n📋 Question: {question[:100]}{'...' if len(question) > 100 else ''}")
    
    # Start timing
    start_time = time.time()
    
    try:
        # Step 1: Call base model
        draft_response = call_base_model(question, max_retries=args.max_retries, verbose=args.verbose)
        
        if args.verbose:
            print("\n📄 Draft Response Preview (first 150 chars):")
            print(f"   {draft_response[:150]}...")
        
        # Step 2: Call fine-tuned model
        full_v3_response = call_fine_tuned_model(question, draft_response, verbose=args.verbose)
        
        # Step 3: Parse and extract revised response
        parsed = parse_v3_response(full_v3_response)
        revised_response = extract_revised_response(full_v3_response)
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        
        # Step 4: Display output
        print("\n" + "=" * 80)
        print("REVISED RESPONSE")
        print("=" * 80)
        print(revised_response)
        print("=" * 80)
        
        # Display timing information
        print(f"\n⏱️  Total execution time: {format_elapsed_time(elapsed_time)}")
        
        # Step 5: Show debug info if verbose
        if args.verbose:
            print("\n" + "=" * 80)
            print("DEBUG INFORMATION")
            print("=" * 80)
            if parsed['score']:
                print(f"\n📊 Score: {parsed['score']}")
            if parsed['strengths']:
                print(f"\n✅ Strengths:\n{parsed['strengths']}")
            if parsed['weaknesses']:
                print(f"\n⚠️  Weaknesses:\n{parsed['weaknesses']}")
            print("=" * 80)
        
        # Step 6: Save to file if requested
        if args.save_output:
            save_output(
                question=question,
                draft_response=draft_response,
                full_v3_response=full_v3_response,
                revised_response=revised_response,
                output_dir=args.output_dir,
                elapsed_time=elapsed_time,
                verbose=args.verbose
            )
        
        print("\n✅ Done!")
        
    except RuntimeError as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
