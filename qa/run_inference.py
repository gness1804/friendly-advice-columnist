#!/usr/bin/env python3
"""
Script to run inference on prompts from test_prompts_v1.md or direct prompt text

Usage:
    # v1: Using shorthand reference from test_prompts_v1.md (QUESTION: prefix added automatically)
    qa/run_inference.py --prompt 'parent overstepping' --checkpoint 'path/to/checkpoint.pt' --version v1
    qa/run_inference.py --prompt 'simple romantic miscommunication' --checkpoint 'path/to/checkpoint1.pt' --version v1
    
    # v1: Using direct prompt text (must include QUESTION: prefix)
    qa/run_inference.py --prompt 'QUESTION: My husband left me. He left me for a much younger woman.' --model_type openai_backend --version v1
    
    # v1: Explicitly force direct prompt mode (must include QUESTION: prefix)
    qa/run_inference.py --prompt 'QUESTION: Short text' --direct-prompt --model_type openai_backend --version v1
    
    # v3: Always uses direct prompt text from command line (must include QUESTION: and DRAFT_RESPONSE:)
    qa/run_inference.py --prompt 'QUESTION: ...\n\nDRAFT_RESPONSE: ...' --model_type openai_backend --version v3
"""

import os
import re
import sys
import argparse
import subprocess
from pathlib import Path

# Get the script directory and project root
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent


def load_env_file(env_path):
    """
    Load environment variables from a .env file.
    
    Args:
        env_path: Path to the .env file
        
    Returns:
        dict: Dictionary of environment variables
    """
    env_vars = {}
    if not env_path.exists():
        print(f"⚠️  Warning: .env file not found at {env_path}")
        return env_vars
    
    with open(env_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue
            # Parse KEY=VALUE format
            if '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()
                # Remove quotes if present
                if value.startswith('"') and value.endswith('"'):
                    value = value[1:-1]
                elif value.startswith("'") and value.endswith("'"):
                    value = value[1:-1]
                # Strip trailing backslashes and whitespace (common in shell script exports)
                value = value.rstrip(' \\').strip()
                env_vars[key] = value
    
    return env_vars


def parse_prompts_file(prompts_path):
    """
    Parse test_prompts_v1.md to extract prompts and stems, creating a mapping.
    Uses explicit SHORTHAND values from the file instead of generating from titles.
    
    Args:
        prompts_path: Path to test_prompts_v1.md
        
    Returns:
        dict: Dictionary mapping shorthand names to dicts with 'prompt' and 'stem' keys
    """
    prompts = {}
    
    if not prompts_path.exists():
        print(f"❌ Error: test_prompts_v1.md not found at {prompts_path}")
        sys.exit(1)
    
    with open(prompts_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Pattern to match prompt sections with explicit SHORTHAND
    # Matches: ### Easy 1 – Title
    #         SHORTHAND: explicit shorthand
    #         PROMPT:
    #         <prompt text>
    #         STEM:
    #         <stem text>
    pattern = r'###\s+(?:Easy|Medium|Hard)\s+\d+\s*[–-]\s*.+?\n\nSHORTHAND:\s*(.+?)\n\nPROMPT:\n(.+?)\n\nSTEM:\n(.+?)(?=\n\nWhat to look for:)'
    
    matches = re.finditer(pattern, content, re.DOTALL)
    
    for match in matches:
        shorthand_raw = match.group(1).strip()
        prompt_text = match.group(2).strip()
        stem_text = match.group(3).strip()
        
        # Normalize shorthand to lowercase and clean up whitespace
        shorthand = shorthand_raw.lower().strip()
        shorthand = re.sub(r'\s+', ' ', shorthand)
        
        prompts[shorthand] = {
            'prompt': prompt_text,
            'stem': stem_text
        }
    
    # Handle special case: "Hard 4" which doesn't have "PROMPT:" label
    # Format: SHORTHAND: ...\n\n<prompt text>\n\nSTEM: ...
    hard4_pattern = r'###\s+Hard\s+4\s*[–-]\s*.+?\n\nSHORTHAND:\s*(.+?)\n\n(.+?)\n\nSTEM:\n(.+?)(?=\n\nWhat to look for:)'
    hard4_match = re.search(hard4_pattern, content, re.DOTALL)
    if hard4_match:
        shorthand_raw = hard4_match.group(1).strip()
        prompt_text = hard4_match.group(2).strip()
        stem_text = hard4_match.group(3).strip()
        
        # Normalize shorthand to lowercase and clean up whitespace
        shorthand = shorthand_raw.lower().strip()
        shorthand = re.sub(r'\s+', ' ', shorthand)
        
        prompts[shorthand] = {
            'prompt': prompt_text,
            'stem': stem_text
        }
    
    return prompts


def find_prompt_by_shorthand(shorthand, prompts, strict=False):
    """
    Find a prompt by shorthand name (fuzzy matching).
    
    Args:
        shorthand: The shorthand name to search for
        prompts: Dictionary of prompts (with 'prompt' and 'stem' keys)
        strict: If True, only do exact matching (no fuzzy matching)
        
    Returns:
        tuple: (matched_key, prompt_dict) or (None, None) if not found
    """
    shorthand_lower = shorthand.lower().strip()
    
    # Exact match
    if shorthand_lower in prompts:
        return shorthand_lower, prompts[shorthand_lower]
    
    # If strict mode, only do exact match
    if strict:
        return None, None
    
    # For non-strict mode, only do substring matching if the input is short
    # (to avoid matching long prompts that happen to contain shorthand words)
    if len(shorthand_lower) <= 100:
        # Partial match - check if shorthand is contained in any key (but only for short inputs)
        for key, prompt_dict in prompts.items():
            if shorthand_lower in key or key in shorthand_lower:
                return key, prompt_dict
        
        # Fuzzy match - check if any words match (only for short inputs)
        shorthand_words = set(shorthand_lower.split())
        best_match = None
        best_score = 0
        
        for key, prompt_dict in prompts.items():
            key_words = set(key.split())
            common_words = shorthand_words & key_words
            if common_words and len(common_words) > best_score:
                best_score = len(common_words)
                best_match = (key, prompt_dict)
        
        return best_match if best_match else (None, None)
    
    # If input is long, don't do fuzzy matching
    return None, None


def list_available_prompts(prompts):
    """Print all available prompts with their explicit shorthands."""
    print("\nAvailable prompts (use these shorthands with --prompt):")
    print("=" * 80)
    for shorthand in sorted(prompts.keys()):
        # Show first 60 chars of prompt
        prompt_text = prompts[shorthand]['prompt']
        preview = prompt_text[:60].replace('\n', ' ')
        print(f"  Shorthand: {shorthand:50} | {preview}...")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description='Run inference on prompts from test_prompts_v1.md',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # v1: Using shorthand references from test_prompts_v1.md (QUESTION: added automatically)
  qa/run_inference.py --prompt 'parent overstepping' --checkpoint 'path/to/checkpoint.pt' --version v1
  qa/run_inference.py --prompt 'simple romantic miscommunication' --checkpoint 'checkpoints/model.pt' --version v1
  qa/run_inference.py --prompt 'value difference around ambition' --checkpoint 'checkpoints/model.pt' --version v1
  
  # v1: Using direct prompt text (must include QUESTION: prefix)
  qa/run_inference.py --prompt 'QUESTION: My husband left me. He left me for a much younger woman.' --model_type openai_backend --version v1
  qa/run_inference.py --prompt 'QUESTION: What are my options?' --checkpoint 'checkpoints/model.pt' --version v1
  
  # v1: Explicitly force direct prompt mode (must include QUESTION: prefix)
  qa/run_inference.py --prompt 'QUESTION: Short text' --direct-prompt --model_type openai_backend --version v1
  
  # v3: Always uses direct prompt text from command line (must include QUESTION: and DRAFT_RESPONSE:)
  qa/run_inference.py --prompt 'QUESTION: My husband left me.\n\nDRAFT_RESPONSE: Some draft response...' --model_type openai_backend --version v3
  
  qa/run_inference.py --list  # List all available prompts (v1 only)
        """
    )
    parser.add_argument(
        '--prompt',
        type=str,
        help='For v1: Shorthand name from test_prompts_v1.md (e.g., "parent overstepping") or full prompt text starting with "QUESTION:". Long prompts (>200 chars) or prompts with newlines are automatically treated as direct prompts. For v3: Full prompt text containing both "QUESTION:" and "DRAFT_RESPONSE:" (no shorthand lookup).'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=False,
        help='Path to the checkpoint file'
    )
    parser.add_argument(
        '--model_type',
        type=str,
        choices=["gpt2", "openai_backend", "from_scratch"], 
        default="gpt2"
    )
    parser.add_argument(
        '--version',
        type=str,
        choices=["v1", "v3"], 
        default="v1"
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='List all available prompts and exit'
    )
    parser.add_argument(
        '--use-stem',
        action='store_true',
        help='Include the STEM (first sentence of ideal answer) after ANSWER: to nudge the model'
    )
    parser.add_argument(
        '--direct-prompt',
        action='store_true',
        help='Treat --prompt as direct prompt text, bypassing shorthand lookup'
    )
    parser.add_argument(
        '--instruction',
        '--instructions',
        type=str,
        help='Optional extra instruction text to send along with the prompt (e.g., a system-level directive)'
    )
    parser.add_argument(
        '--save-output',
        action='store_true',
        help='Save the output to a file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs/inference',
        help='Directory to save the output to'
    )
    
    args = parser.parse_args()
    instruction_text = (args.instruction or '').strip()
    
    # Load prompts
    if args.version == "v3":
        prompts_path = SCRIPT_DIR / 'test_prompts_v3.md'
    else:
        prompts_path = SCRIPT_DIR / 'test_prompts_v1.md'
    prompts = parse_prompts_file(prompts_path)
    
    if args.list:
        list_available_prompts(prompts)
        sys.exit(0)
    
    if not args.prompt:
        print("❌ Error: --prompt is required (use --list to see available prompts)")
        sys.exit(1)
    
    if not args.checkpoint and args.model_type not in ["gpt2", "openai_backend"]:
        print("❌ Error: --checkpoint is required when not using a built-in model type")
        sys.exit(1)
    
    # Determine if we should treat this as a direct prompt
    # v3 always uses direct prompts (no shorthand lookup)
    # For v1, use shorthand lookup unless --direct-prompt is set or prompt is very long/contains newlines
    if args.version == "v3":
        # v3 always uses direct prompt text from command line (no shorthand)
        is_direct_prompt = True
    else:
        # For v1, check if we should use direct prompt
        is_direct_prompt = args.direct_prompt
        if not is_direct_prompt:
            # Auto-detect: long prompts (> 200 chars) or prompts with newlines are likely direct prompts
            if len(args.prompt) > 200 or '\n' in args.prompt:
                is_direct_prompt = True
                print("💡 Detected long prompt or newlines - treating as direct prompt text")
    
    # Try to find the prompt as a shorthand reference (only for v1, and only if not direct)
    if is_direct_prompt:
        # Treat as direct prompt text
        prompt_text = args.prompt
        stem_text = ''
        if args.version == "v3":
            print("✅ Using direct prompt text (v3 always uses command-line prompt)")
        else:
            print("✅ Using direct prompt text")
    else:
        # Only v1 can use shorthand lookup
        # Try shorthand lookup with strict matching for longer inputs
        strict_mode = len(args.prompt) > 50  # Use strict mode for inputs longer than 50 chars
        matched_key, prompt_dict = find_prompt_by_shorthand(args.prompt, prompts, strict=strict_mode)
        
        if matched_key:
            # Found as shorthand reference (v1 only)
            # Add "QUESTION:" prefix since file format doesn't include it
            raw_prompt = prompt_dict['prompt']
            prompt_text = f"QUESTION: {raw_prompt}"
            stem_text = prompt_dict.get('stem', '')
            print(f"✅ Found prompt: {matched_key}")
        else:
            # Not found in shorthand, treat as direct prompt text
            prompt_text = args.prompt
            stem_text = ''
            print("✅ Using direct prompt text (not found in shorthand references)")
    if args.checkpoint:
        print(f"   Using checkpoint: {args.checkpoint}")
    print(f"   Using model type: {args.model_type.upper()}")
    print(f"   Using version: {args.version}")
    
    if args.use_stem and args.version == "v1":
        if stem_text:
            print(f"   Using STEM: {stem_text[:60]}...")
        else:
            print("   ⚠️  Warning: --use-stem specified but no STEM found for this prompt")
    
    # Handle --use-stem for v1 (append stem as ANSWER: if available)
    if args.use_stem and args.version == "v1" and stem_text:
        stem_clean = stem_text.strip()
        prompt_text = f"{prompt_text}\n\nANSWER: {stem_clean}"
    
    # Clean the prompt text (strip any extra whitespace)
    prompt_text_clean = prompt_text.strip()
    
    # Validate required trigger words based on version
    if args.version == "v1":
        if not prompt_text_clean.startswith("QUESTION:"):
            print("❌ Error: For v1, the prompt must start with 'QUESTION:'")
            print(f"   Your prompt starts with: {prompt_text_clean[:50]}...")
            sys.exit(1)
    elif args.version == "v3":
        if "QUESTION:" not in prompt_text_clean:
            print("❌ Error: For v3, the prompt must contain 'QUESTION:'")
            print(f"   Your prompt starts with: {prompt_text_clean[:50]}...")
            sys.exit(1)
        if "DRAFT_RESPONSE:" not in prompt_text_clean:
            print("❌ Error: For v3, the prompt must contain 'DRAFT_RESPONSE:'")
            print(f"   Your prompt: {prompt_text_clean[:100]}...")
            sys.exit(1)
    
    # Check if checkpoint exists (only if provided)
    checkpoint_path = None
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.is_absolute():
            checkpoint_path = PROJECT_ROOT / checkpoint_path
        
        if not checkpoint_path.exists():
            print(f"❌ Error: Checkpoint not found at {checkpoint_path}")
            sys.exit(1)
    
    # Load .env file
    env_path = SCRIPT_DIR / '.env'
    env_vars = load_env_file(env_path)
    
    # Prepare environment variables
    env = os.environ.copy()
    
    # Load variables from .env file
    for key, value in env_vars.items():
        env[key] = value
    
    # Use prompt text directly (user must provide trigger words)
    # For v1: user must provide "QUESTION:" prefix
    # For v3: user must provide "QUESTION:" and "DRAFT_RESPONSE:" prefixes
    base_prompt = prompt_text_clean

    if instruction_text:
        formatted_prompt = f"INSTRUCTION: {instruction_text}\n\n{base_prompt}"
    else:
        formatted_prompt = base_prompt
    
    # Override with command-line arguments
    if checkpoint_path:
        env['CHECKPOINT_PATH'] = str(checkpoint_path)
    env['PROMPT'] = formatted_prompt
    env['MODE'] = 'inference'  # Ensure MODE is set
    env['MODEL_TYPE'] = args.model_type
    env['VERSION'] = args.version
    if args.save_output:
        env['SAVE_OUTPUT'] = 'True'
    if args.output_dir:
        env['OUTPUT_DIR'] = str(args.output_dir)
    if instruction_text:
        env['INSTRUCTION'] = instruction_text
    
    # Print what we're running
    if instruction_text:
        instr_preview = instruction_text.replace('\n', ' ')
        print("\n🧭 Instruction preview (first 120 chars):")
        print(f"   {instr_preview[:120]}...")
    print("\n📏 Prompt preview (first 100 chars):")
    print(f"   {prompt_text_clean[:100]}...")
    print("\n📋 Full formatted prompt (first 150 chars):")
    print(f"   {formatted_prompt[:150]}...")
    print("\n🚀 Running inference...")
    if checkpoint_path:
        print(f"   Checkpoint: {checkpoint_path}")
    print(f"   Environment variables from .env: {', '.join(env_vars.keys())}")
    
    # Run the inference script
    inference_script = PROJECT_ROOT / 'scripts' / 'load_checkpoint.py'
    
    if not inference_script.exists():
        print(f"❌ Error: Inference script not found at {inference_script}")
        sys.exit(1)
    
    try:
        # Run the script with the prepared environment
        result = subprocess.run(
            [sys.executable, str(inference_script)],
            env=env,
            cwd=str(PROJECT_ROOT),
            check=False
        )
        
        sys.exit(result.returncode)
        
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error running inference: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()

