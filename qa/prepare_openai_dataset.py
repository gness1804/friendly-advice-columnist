import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.prompts import ADVICE_COLUMNIST_SYSTEM_PROMPT

DEFAULT_OUTPUT_PATH = f"sources/v2/v_2_1/canonical/openai_finetune_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"


def main():
    parser = argparse.ArgumentParser(
        description='Prepare OpenAI fine-tuning dataset from Q&A markdown file. Creates the JSONL file to be used in fine-tuning.'
    )
    parser.add_argument(
        'source',
        type=str,
        help='Path to the source markdown file containing Q&A pairs'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=DEFAULT_OUTPUT_PATH,
        help='Path to the output JSONL file'
    )
    
    args = parser.parse_args()
    
    # Resolve source file path
    source_path = Path(args.source)
    if not source_path.exists():
        print(f"Error: Source file not found: {source_path}", file=sys.stderr)
        sys.exit(1)
    
    # Read source file
    try:
        text = source_path.read_text(encoding='utf-8')
    except Exception as e:
        print(f"Error reading source file: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Split on your <END_OF_SET> token
    chunks = [c.strip() for c in text.split("<END_OF_SET>") if c.strip()]

    # Write output file
    try:
        with Path(args.output).open("w", encoding='utf-8') as f:
            for chunk in chunks:
                q_match = re.search(r"QUESTION:\s*(.+?)\nANSWER:", chunk, re.S)
                a_match = re.search(r"ANSWER:\s*(.+)$", chunk, re.S)
                if not q_match or not a_match:
                    continue

                question = q_match.group(1).strip()
                answer = a_match.group(1).strip()

                example = {
                    "messages": [
                        {"role": "system", "content": ADVICE_COLUMNIST_SYSTEM_PROMPT},
                        {"role": "user", "content": f"QUESTION: {question}"},
                        {"role": "assistant", "content": f"ANSWER: {answer}"},
                    ]
                }
                f.write(json.dumps(example) + "\n")
        
        print(f"Successfully created dataset: {args.output}")
        print(f"Processed {len(chunks)} Q&A pairs")
    except Exception as e:
        print(f"Error writing output file: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
