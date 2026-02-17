#!/usr/bin/env python3
"""
Convert markdown dataset to JSONL format for fine-tuning.

This script parses a markdown document structured with QUESTION, DRAFT_RESPONSE,
SCORE, STRENGTHS, WEAKNESSES, and REVISED_RESPONSE sections and converts them
to JSONL format suitable for fine-tuning an LLM.
"""

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

# Import the system prompt
sys.path.insert(0, str(Path(__file__).parent.parent))
from models.prompts import SYSTEM_PROMPT_V3


def normalize_text(text: str) -> str:
    """Strip trailing spaces and normalize newlines."""
    # Strip trailing spaces from each line
    lines = [line.rstrip() for line in text.split('\n')]
    # Join with single newline and strip leading/trailing whitespace
    return '\n'.join(lines).strip()


def round_score(score: float) -> float:
    """Round score to nearest 0.5 increment (1.0, 1.5, 2.0, ..., 10.0)."""
    if score < 1.0:
        return 1.0
    if score > 10.0:
        return 10.0
    # Round to nearest 0.5
    return round(score * 2) / 2


def remove_links(text: str) -> str:
    """Remove markdown and HTML links from text."""
    # Remove markdown links [text](url)
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    # Remove HTML links <a href="...">text</a>
    text = re.sub(r'<a[^>]*>([^<]+)</a>', r'\1', text)
    # Remove bare URLs (http:// or https://)
    text = re.sub(r'https?://[^\s]+', '', text)
    return text


def replace_em_dashes(text: str) -> str:
    """
    Replace em dashes (—) with appropriate punctuation.
    Uses context to determine whether to use comma, period, or semicolon.
    """
    # Replace em dash followed by capital letter (likely start of new sentence) with period
    text = re.sub(r'—\s*([A-Z])', r'. \1', text)
    # Replace em dash at end of line/paragraph with period
    text = re.sub(r'—\s*$', '.', text, flags=re.MULTILINE)
    # Replace em dash followed by lowercase or punctuation (likely continuation) with comma
    text = re.sub(r'—\s*([a-z])', r', \1', text)
    # Replace any remaining em dashes with comma
    text = re.sub(r'—', ',', text)
    return text


def parse_markdown_document(content: str) -> List[Dict[str, str]]:
    """
    Parse markdown document into structured entries.
    
    Expected structure:
    QUESTION: <text>
    DRAFT_RESPONSE: <text>
    SCORE: <number>
    STRENGTHS:
    - <bullet points>
    WEAKNESSES:
    - <bullet points>
    REVISED_RESPONSE: <text>
    <END_OF_SET>
    """
    entries = []
    
    # Split by <END_OF_SET> markers
    sections = re.split(r'<END_OF_SET>', content)
    
    for section_idx, section in enumerate(sections):
        section = section.strip()
        if not section:
            continue
        
        # Use a more flexible pattern that matches the label and captures content
        # Pattern: LABEL: optional space/newline, then content until next LABEL: or end
        
        # Extract QUESTION (from QUESTION: until DRAFT_RESPONSE:)
        question_match = re.search(r'^QUESTION:\s*\n?(.*?)(?=\nDRAFT_RESPONSE:)', section, re.DOTALL | re.MULTILINE)
        if not question_match:
            # Try without requiring newline before DRAFT_RESPONSE
            question_match = re.search(r'^QUESTION:\s*\n?(.*?)(?=DRAFT_RESPONSE:)', section, re.DOTALL | re.MULTILINE)
        if not question_match:
            raise ValueError(f"Missing or malformed QUESTION section in entry {section_idx + 1}")
        question = normalize_text(question_match.group(1))
        
        # Extract DRAFT_RESPONSE (from DRAFT_RESPONSE: until SCORE:)
        # First check if SCORE exists
        if 'SCORE:' not in section:
            raise ValueError(f"Missing SCORE section in entry {section_idx + 1}")
        
        draft_match = re.search(r'DRAFT_RESPONSE:\s*\n?(.*?)(?=\nSCORE:)', section, re.DOTALL | re.MULTILINE)
        if not draft_match:
            # Try without requiring newline before SCORE
            draft_match = re.search(r'DRAFT_RESPONSE:\s*\n?(.*?)(?=SCORE:)', section, re.DOTALL | re.MULTILINE)
        if not draft_match:
            raise ValueError(f"Missing or malformed DRAFT_RESPONSE section in entry {section_idx + 1}")
        draft_response = normalize_text(draft_match.group(1))
        
        # Extract SCORE
        score_match = re.search(r'SCORE:\s*([\d.]+)', section, re.MULTILINE)
        if not score_match:
            raise ValueError(f"Missing or malformed SCORE section in entry {section_idx + 1}")
        try:
            score = round_score(float(score_match.group(1)))
        except ValueError:
            raise ValueError(f"Invalid score value in entry {section_idx + 1}: {score_match.group(1)}")
        
        # Extract STRENGTHS (from STRENGTHS: until WEAKNESSES:)
        strengths_match = re.search(r'STRENGTHS:\s*\n?(.*?)(?=\nWEAKNESSES:)', section, re.DOTALL | re.MULTILINE)
        if not strengths_match:
            # Try without requiring newline before WEAKNESSES
            strengths_match = re.search(r'STRENGTHS:\s*\n?(.*?)(?=WEAKNESSES:)', section, re.DOTALL | re.MULTILINE)
        if not strengths_match:
            raise ValueError(f"Missing or malformed STRENGTHS section in entry {section_idx + 1}")
        strengths = normalize_text(strengths_match.group(1))
        
        # Extract WEAKNESSES (from WEAKNESSES: until REVISED_RESPONSE:)
        weaknesses_match = re.search(r'WEAKNESSES:\s*\n?(.*?)(?=\nREVISED_RESPONSE:)', section, re.DOTALL | re.MULTILINE)
        if not weaknesses_match:
            # Try without requiring newline before REVISED_RESPONSE
            weaknesses_match = re.search(r'WEAKNESSES:\s*\n?(.*?)(?=REVISED_RESPONSE:)', section, re.DOTALL | re.MULTILINE)
        if not weaknesses_match:
            raise ValueError(f"Missing or malformed WEAKNESSES section in entry {section_idx + 1}")
        weaknesses = normalize_text(weaknesses_match.group(1))
        
        # Extract REVISED_RESPONSE (from REVISED_RESPONSE: until end of section)
        # Use \Z to match absolute end of string (not end of line), and greedy match to get all content
        revised_match = re.search(r'REVISED_RESPONSE:\s*\n?(.*)\Z', section, re.DOTALL)
        if not revised_match:
            raise ValueError(f"Missing or malformed REVISED_RESPONSE section in entry {section_idx + 1}")
        revised_response = normalize_text(revised_match.group(1))
        
        entries.append({
            'question': question,
            'draft_response': draft_response,
            'score': score,
            'strengths': strengths,
            'weaknesses': weaknesses,
            'revised_response': revised_response
        })
    
    return entries


def convert_to_jsonl(entries: List[Dict[str, str]], system_prompt: str) -> List[Dict]:
    """Convert parsed entries to JSONL format."""
    jsonl_entries = []
    
    for entry in entries:
        # Remove links from all text fields
        question = remove_links(entry['question'])
        draft_response = remove_links(entry['draft_response'])
        strengths = remove_links(entry['strengths'])
        weaknesses = remove_links(entry['weaknesses'])
        revised_response = remove_links(entry['revised_response'])
        
        # Remove em dashes ONLY from REVISED_RESPONSE
        revised_response = replace_em_dashes(revised_response)
        
        # Construct user message: QUESTION + DRAFT_RESPONSE
        user_message = f"{question}\n\nDRAFT_RESPONSE:\n{draft_response}"
        
        # Construct assistant message: SCORE + STRENGTHS + WEAKNESSES + REVISED_RESPONSE
        assistant_message = f"SCORE: {entry['score']}\n\nSTRENGTHS:\n{strengths}\n\nWEAKNESSES:\n{weaknesses}\n\nREVISED_RESPONSE:\n{revised_response}"
        
        jsonl_entry = {
            'messages': [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_message},
                {'role': 'assistant', 'content': assistant_message}
            ]
        }
        
        jsonl_entries.append(jsonl_entry)
    
    return jsonl_entries


def main():
    parser = argparse.ArgumentParser(
        description='Convert markdown dataset to JSONL format for fine-tuning'
    )
    parser.add_argument(
        '--input',
        type=str,
        default='sources/v3/dataset_source.md',
        help='Input markdown file (default: sources/v3/dataset_source.md)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output JSONL file (default: sources/v3/<input_name>_<timestamp>.jsonl)'
    )
    
    args = parser.parse_args()
    
    # Resolve input path
    script_dir = Path(__file__).parent.parent
    input_path = script_dir / args.input
    
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = script_dir / output_path
    else:
        # Default: sources/v3/<original_name>_<timestamp>.jsonl
        input_name = input_path.stem
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = script_dir / 'sources' / 'v3' / f"{input_name}_{timestamp}.jsonl"
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Read input file
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading input file: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Parse document
    try:
        entries = parse_markdown_document(content)
        print(f"Parsed {len(entries)} entries from {input_path}", file=sys.stderr)
    except ValueError as e:
        print(f"Error parsing document: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Convert to JSONL
    jsonl_entries = convert_to_jsonl(entries, SYSTEM_PROMPT_V3)
    
    # Write output
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            for entry in jsonl_entries:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        print(f"Wrote {len(jsonl_entries)} entries to {output_path}", file=sys.stderr)
    except Exception as e:
        print(f"Error writing output file: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
