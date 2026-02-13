#!/usr/bin/env python3
"""
Script to normalize the dataset_source.md file to match the canonical format.

Canonical format:
- QUESTION:
- DRAFT_RESPONSE:
- SCORE: X.X
- STRENGTHS: (bullet points)
- WEAKNESSES: (bullet points)
- REVISED_RESPONSE:
- <END_OF_SET>
"""

import re
import os
import argparse
import sys

def normalize_dataset(input_file, output_file):
    """Normalize the dataset file to match the canonical format."""
    
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split by <END_OF_SET> to process each example separately
    examples = content.split('<END_OF_SET>')
    normalized_examples = []
    
    for i, example in enumerate(examples):
        if not example.strip():
            continue
        
        # Check if example is already in correct format (has SCORE: not Score: or EVALUATION:)
        # If it's already formatted, skip processing
        is_already_formatted = (
            'SCORE:' in example and 
            'STRENGTHS:' in example and 
            'WEAKNESSES:' in example and 
            'REVISED_RESPONSE:' in example and
            'EVALUATION:' not in example and
            'OUTPUT:' not in example
        )
        
        if is_already_formatted:
            normalized_examples.append(example)
            if i < len(examples) - 1:  # Not the last one
                normalized_examples.append('<END_OF_SET>')
            continue
        
        # Process the example
        lines = example.split('\n')
        normalized_lines = []
        in_strengths = False
        in_weaknesses = False
        skip_next_line = False
        
        j = 0
        while j < len(lines):
            if skip_next_line:
                skip_next_line = False
                j += 1
                continue
                
            line = lines[j]
            stripped = line.strip()
            
            # Remove INPUT: header if present
            if stripped.startswith('INPUT:'):
                j += 1
                continue
            
            # Handle EVALUATION: section - remove it and process the next line
            if stripped.startswith('EVALUATION:'):
                j += 1
                # Look for "Score: X.X" in the next line
                if j < len(lines):
                    score_line = lines[j].strip()
                    # Match "Score: X.X" pattern (case insensitive)
                    score_match = re.match(r'^Score:\s*([0-9]+\.?[05]?)$', score_line, re.IGNORECASE)
                    if score_match:
                        score = score_match.group(1)
                        # Ensure it's properly formatted
                        try:
                            score_float = float(score)
                            if score_float % 0.5 != 0:
                                # Round to nearest 0.5
                                score_float = round(score_float * 2) / 2
                            score = f"{score_float:.1f}"
                        except (ValueError, ArithmeticError):
                            pass
                        normalized_lines.append('')
                        normalized_lines.append(f'SCORE: {score}')
                        j += 1
                        continue
                # If no score found, just skip the EVALUATION line
                continue
            
            # Handle standalone "Score: X.X" line (if not already processed)
            if re.match(r'^Score:\s*', stripped, re.IGNORECASE) and not re.match(r'^SCORE:\s*', stripped):
                score_match = re.search(r'([0-9]+\.?[05]?)', stripped)
                if score_match:
                    score = score_match.group(1)
                    try:
                        score_float = float(score)
                        if score_float % 0.5 != 0:
                            score_float = round(score_float * 2) / 2
                        score = f"{score_float:.1f}"
                    except (ValueError, ArithmeticError):
                        pass
                    normalized_lines.append('')
                    normalized_lines.append(f'SCORE: {score}')
                j += 1
                continue
            
            # Convert OUTPUT: to REVISED_RESPONSE:
            if stripped.startswith('OUTPUT:') or stripped.startswith('REVISED_RESPONSE:'):
                # Reset section flags
                in_strengths = False
                in_weaknesses = False
                normalized_lines.append('')
                normalized_lines.append('REVISED_RESPONSE: ')
                j += 1
                continue
            
            # Handle STRENGTHS: section (case-insensitive)
            if re.match(r'^Strengths:', stripped, re.IGNORECASE) and not stripped.startswith('STRENGTHS:'):
                normalized_lines.append('')
                normalized_lines.append('STRENGTHS:')
                in_strengths = True
                in_weaknesses = False
                j += 1
                continue
            
            # Handle WEAKNESSES: section (case-insensitive)
            if re.match(r'^Weaknesses:', stripped, re.IGNORECASE) and not stripped.startswith('WEAKNESSES:'):
                normalized_lines.append('')
                normalized_lines.append('WEAKNESSES:')
                in_strengths = False
                in_weaknesses = True
                j += 1
                continue
            
            # Process content within STRENGTHS or WEAKNESSES sections
            if in_strengths or in_weaknesses:
                if stripped:
                    # Check if this is a section header (means we're leaving this section)
                    if (stripped.startswith('WEAKNESSES:') or stripped.startswith('Weaknesses:') or
                        stripped.startswith('OUTPUT:') or stripped.startswith('REVISED_RESPONSE:') or
                        stripped.startswith('QUESTION:') or stripped.startswith('EVALUATION:') or
                        stripped.startswith('STRENGTHS:') or stripped.startswith('Strengths:')):
                        # We've hit a new section, exit this one
                        in_strengths = False
                        in_weaknesses = False
                        # Don't increment j, let the next iteration handle this line
                        continue
                    
                    # Ensure it's a bullet point
                    if not stripped.startswith('-'):
                        # Check if it's a numbered list item
                        if re.match(r'^\d+\.', stripped):
                            # Remove number and add bullet
                            stripped = re.sub(r'^\d+\.\s*', '- ', stripped)
                        else:
                            # Add bullet if it's not empty
                            stripped = '- ' + stripped
                        normalized_lines.append(stripped)
                    else:
                        # Already a bullet point, keep as is
                        normalized_lines.append(line)
                else:
                    # Empty line - check if we're moving to next section
                    if j + 1 < len(lines):
                        next_stripped = lines[j + 1].strip()
                        # If next line is a section header, don't add empty line
                        if (next_stripped.startswith('WEAKNESSES:') or next_stripped.startswith('Weaknesses:') or
                            next_stripped.startswith('OUTPUT:') or next_stripped.startswith('REVISED_RESPONSE:') or
                            next_stripped.startswith('QUESTION:') or next_stripped.startswith('EVALUATION:') or
                            next_stripped.startswith('STRENGTHS:') or next_stripped.startswith('Strengths:')):
                            # Don't add empty line, let next section handle it
                            pass
                        elif next_stripped:
                            # Next line is content (another strength/weakness item), skip the empty line
                            pass
                        else:
                            # Multiple empty lines, skip them
                            pass
                j += 1
                continue
            
            # Regular content line
            normalized_lines.append(line)
            j += 1
        
        # Join the normalized lines
        normalized_example = '\n'.join(normalized_lines)
        # Clean up multiple consecutive empty lines (max 1)
        normalized_example = re.sub(r'\n{3,}', '\n\n', normalized_example)
        # Remove trailing blank lines
        normalized_example = normalized_example.rstrip()
        normalized_examples.append(normalized_example)
        
        if i < len(examples) - 1:  # Not the last one
            normalized_examples.append('<END_OF_SET>')
    
    # Join all examples
    normalized_content = '\n'.join(normalized_examples)
    
    # Final cleanup: ensure proper spacing around headers (one blank line before each)
    normalized_content = re.sub(r'\n(QUESTION:)\n+', r'\n\n\1\n', normalized_content)
    normalized_content = re.sub(r'\n(DRAFT_RESPONSE:)\n+', r'\n\n\1\n', normalized_content)
    normalized_content = re.sub(r'\n(SCORE:)\s*', r'\n\n\1 ', normalized_content)
    normalized_content = re.sub(r'\n(STRENGTHS:)\n+', r'\n\n\1\n', normalized_content)
    normalized_content = re.sub(r'\n(WEAKNESSES:)\n+', r'\n\n\1\n', normalized_content)
    normalized_content = re.sub(r'\n(REVISED_RESPONSE:)\s*', r'\n\n\1\n', normalized_content)
    
    # Remove any remaining INPUT: or EVALUATION: headers
    normalized_content = re.sub(r'^INPUT:\s*$', '', normalized_content, flags=re.MULTILINE)
    normalized_content = re.sub(r'^EVALUATION:\s*$', '', normalized_content, flags=re.MULTILINE)
    
    # Final comprehensive cleanup: ensure never more than one blank line in a row anywhere
    # Replace any sequence of 3+ newlines (which means 2+ blank lines) with exactly 2 newlines (1 blank line)
    # Do this multiple times to catch all cases
    while re.search(r'\n{3,}', normalized_content):
        normalized_content = re.sub(r'\n{3,}', '\n\n', normalized_content)
    
    # Write output
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(normalized_content)
    
    print(f"Normalization complete. Output written to {output_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Normalize dataset_source.md file to match canonical format'
    )
    parser.add_argument(
        'input_file',
        nargs='?',
        default=os.path.expanduser('~/Desktop/build_llm_karpathy/sources/v3/dataset_source.md'),
        help='Input file to normalize (default: ~/Desktop/build_llm_karpathy/sources/v3/dataset_source.md)'
    )
    parser.add_argument(
        '-o', '--output',
        help='Output file path (default: input filename with "_normalized" before extension)'
    )
    
    args = parser.parse_args()
    
    # Expand user path if it contains ~
    input_file = os.path.expanduser(args.input_file)
    
    # Generate output filename if not provided
    if args.output:
        output_file = os.path.expanduser(args.output)
    else:
        # Add "_normalized" before the file extension
        base, ext = os.path.splitext(input_file)
        output_file = f"{base}_normalized{ext}"
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file not found: {input_file}", file=sys.stderr)
        sys.exit(1)
    
    normalize_dataset(input_file, output_file)
    print("\nPlease review the normalized file before replacing the original.")
