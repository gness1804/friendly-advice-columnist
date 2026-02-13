# ruff: noqa: E731

"""
Load and use saved model checkpoints for inference or resuming training
"""

import torch
import os
import sys
from pathlib import Path

# Ensure project root is on sys.path so we can import top-level modules like `models`
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Configuration
VERSION = os.environ.get("VERSION", "v1")
CHECKPOINT_PATH = os.environ.get("CHECKPOINT_PATH", None)
MODE = os.environ.get("MODE", "inference").lower()  # "inference" or "resume"
DEVICE = os.environ.get("DEVICE", "auto")  # "auto", "cpu", "cuda", "mps"
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "300"))
PROMPT = os.environ.get("PROMPT", "")  # For inference mode
RESUME_TRAINING_STEPS = int(os.environ.get("RESUME_TRAINING_STEPS", "5000"))
RESUME_LEARNING_RATE = float(os.environ.get("RESUME_LEARNING_RATE", "3e-4"))
SAVE_OUTPUT = os.environ.get("SAVE_OUTPUT", "False").lower() in ["true", "1", "yes", "True"]
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "outputs/inference")  # For inference mode
MODEL_TYPE = os.environ.get("MODEL_TYPE", "gpt2").lower()

# ============================================================================
# OPENAI BACKEND FAST-PATH (no local checkpoint required)
# ============================================================================

if MODEL_TYPE == "openai_backend":
    try:
        from models.openai_backend import generate_answer as openai_generate_answer
    except ImportError as e:
        print(f"❌ Error: Failed to import OpenAI backend: {e}")
        sys.exit(1)

    if MODE == "inference":
        if not PROMPT:
            print("❌ Error: PROMPT must be set for OpenAI backend inference")
            sys.exit(1)

        print("\n" + "=" * 50)
        print("INFERENCE MODE (OpenAI backend)")
        print("=" * 50)
        print(f"\nPrompt: {PROMPT}")
        print(f"Version: {VERSION}")

        generated_text = openai_generate_answer(PROMPT, version=VERSION)

        print("\n" + "=" * 50)
        print("GENERATED TEXT")
        print("=" * 50)
        print(generated_text)
        print("=" * 50)

        if SAVE_OUTPUT:
            from datetime import datetime

            os.makedirs(OUTPUT_DIR, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"inference_output_openai_{timestamp}.txt"
            output_path = os.path.join(OUTPUT_DIR, output_filename)
            try:
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write("Backend: openai_backend\n")
                    f.write(f"Model: {os.environ.get('OPENAI_MODEL', '')}\n")
                    f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"Temperature: {os.environ.get('TEMPERATURE', 0.3)}\n")
                    f.write(f"Top P: {os.environ.get('TOP_P', 0.9)}\n")
                    f.write(f"Max new tokens: {os.environ.get('MAX_NEW_TOKENS', 700)}\n")
                    f.write("\nPROMPT:\n")
                    f.write(PROMPT)
                    f.write("\n\n" + "=" * 50 + "\n")
                    f.write("GENERATED TEXT\n")
                    f.write("=" * 50 + "\n")
                    f.write(generated_text)
                    f.write("\n" + "=" * 50 + "\n")
                print(f"\n✅ Output written to: {output_path}")
            except Exception as e:
                print(f"\n⚠️  Error saving output: {e}")

        sys.exit(0)

    elif MODE == "resume":
        print("❌ Resume training is not supported for MODEL_TYPE=openai_backend yet.")
        sys.exit(1)

    else:
        print(f"❌ Error: Unknown MODE for OpenAI backend: {MODE}")
        sys.exit(1)
# ============================================================================
# DEVICE SETUP
# ============================================================================

if DEVICE == "auto":
    if torch.cuda.is_available():
        device = "cuda"
        print("✅ Using NVIDIA GPU (CUDA)")
    elif torch.backends.mps.is_available():
        device = "mps"
        print("✅ Using Apple Silicon GPU (Metal Performance Shaders)")
    else:
        device = "cpu"
        print("⚠️  Using CPU (slow)")
else:
    device = DEVICE
    print(f"Using device: {device}")

# ============================================================================
# CHECKPOINT LOADING
# ============================================================================

if not CHECKPOINT_PATH:
    print("❌ Error: CHECKPOINT_PATH environment variable not set")
    print(
        "Usage: CHECKPOINT_PATH=/path/to/checkpoint.pt MODE=inference "
        "python3 scripts/load_checkpoint.py"
    )
    sys.exit(1)

if not os.path.exists(CHECKPOINT_PATH):
    print(f"❌ Error: Checkpoint not found at {CHECKPOINT_PATH}")
    sys.exit(1)

print(f"Loading checkpoint from: {CHECKPOINT_PATH}")

try:
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
except Exception as e:
    print(f"❌ Error loading checkpoint: {e}")
    sys.exit(1)

# Extract checkpoint data
step = checkpoint.get("step", 0)
model_state_dict = checkpoint.get("model_state_dict")
optimizer_state_dict = checkpoint.get("optimizer_state_dict")
hyperparameters = checkpoint.get("hyperparameters", {})
vocab_size = checkpoint.get("vocab_size")
block_size = checkpoint.get("block_size")
batch_size = checkpoint.get("batch_size")

if not model_state_dict:
    print("❌ Error: Checkpoint does not contain model_state_dict")
    sys.exit(1)

print(f"✅ Checkpoint loaded (trained for {step} steps)")
print(f"   Vocab size: {vocab_size:,}")
print(f"   Block size: {block_size}")
print(f"   Batch size: {batch_size}")

# ============================================================================
# MODEL INITIALIZATION
# ============================================================================

# Determine model type from checkpoint hyperparameters
use_lora = hyperparameters.get("use_lora", False)

print(f"\nInitializing model (type: {MODEL_TYPE.upper()}, LoRA: {use_lora})...")

if MODEL_TYPE.lower() == "gpt2":
    from models.gpt2_wrapper import GPT2Wrapper

    gpt2_model_name = hyperparameters.get("gpt2_model_name", "gpt2")
    lora_rank = hyperparameters.get("lora_rank", 8)
    lora_alpha = hyperparameters.get("lora_alpha", 16.0)
    lora_dropout = hyperparameters.get("lora_dropout", 0.0)

    model = GPT2Wrapper(
        model_name=gpt2_model_name,
        use_lora=use_lora,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        device=device,
    )

    encode = model.encode
    decode = model.decode
    tokenization_method = "gpt2"

# elif MODEL_TYPE == "openai_backend":
#     from models.openai_backend import generate_answer

#     model = generate_answer
#     encode = model.encode
#     decode = model.decode
#     tokenization_method = "gpt2"

elif MODEL_TYPE.lower() == "from_scratch":
    from models.bigram_lm_v2 import BigramLanguageModel
    from models.bigram_lm_v2_lora import BigramLanguageModelLoRA

    n_embd = hyperparameters.get("n_embd", 384)
    n_head = hyperparameters.get("n_head", 6)
    n_layer = hyperparameters.get("n_layer", 6)
    dropout = hyperparameters.get("dropout", 0.2)
    lora_rank = hyperparameters.get("lora_rank", 8)
    lora_alpha = hyperparameters.get("lora_alpha", 16.0)
    lora_dropout = hyperparameters.get("lora_dropout", 0.0)

    if use_lora:
        model = BigramLanguageModelLoRA(
            vocab_size=vocab_size,
            n_embd=n_embd,
            block_size=block_size,
            device=device,
            dropout=dropout,
            n_head=n_head,
            n_layer=n_layer,
            use_lora=True,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )
    else:
        model = BigramLanguageModel(
            vocab_size=vocab_size,
            n_embd=n_embd,
            block_size=block_size,
            device=device,
            dropout=dropout,
            n_head=n_head,
            n_layer=n_layer,
        )

    tokenization_method = hyperparameters.get("tokenization_method", "character")

    # Load tokenization based on method
    if tokenization_method == "gpt2":
        import tiktoken

        enc = tiktoken.get_encoding("gpt2")
        encode = lambda s: enc.encode(s)
        decode = lambda ids: enc.decode(ids)

    elif tokenization_method == "custom_bpe":
        try:
            from tokenizers import Tokenizer

            # Look for tokenizer file in the same directory as checkpoint
            checkpoint_dir = os.path.dirname(CHECKPOINT_PATH)
            tokenizer_path = os.path.join(checkpoint_dir, "tokenizer.json")

            if os.path.exists(tokenizer_path):
                tokenizer = Tokenizer.from_file(tokenizer_path)
                encode = lambda s: tokenizer.encode(s).ids
                decode = lambda ids: tokenizer.decode(ids)
                print(f"✅ Loaded custom BPE tokenizer from {tokenizer_path}")
            else:
                print(f"⚠️  Warning: Custom BPE tokenizer not found at {tokenizer_path}")
                print("   Falling back to character-level tokenization")
                tokenization_method = "character"
        except ImportError:
            print(
                "⚠️  tokenizers library not installed, falling back to character-level"
            )
            tokenization_method = "character"

    if tokenization_method == "character":
        # For character-level, we need to reconstruct the vocab
        # This is a limitation - we'd need to store the char mappings in the checkpoint
        print(
            "⚠️  Warning: Character-level tokenization requires vocab to be reconstructed"
        )
        print("   Generating vocab from checkpoint data...")

        # Create a simple character vocab (this won't match original if special chars exist)
        chars = [chr(i) for i in range(vocab_size)]
        stoi = {ch: i for i, ch in enumerate(chars)}
        itos = {i: ch for i, ch in enumerate(chars)}

        encode = lambda s: [stoi.get(c, 0) for c in s]
        decode = lambda ids: "".join([itos.get(i, "?") for i in ids])
else:
    print(f"❌ Error: Unknown model type: {MODEL_TYPE.upper()}")
    sys.exit(1)

# Load model state
model.load_state_dict(model_state_dict)
model.to(device)
model.eval()  # Set to evaluation mode

print("✅ Model initialized and weights loaded")

# ============================================================================
# INFERENCE MODE
# ============================================================================

if MODE == "inference":
    print("\n" + "=" * 50)
    print("INFERENCE MODE")
    print("=" * 50)

    # Generation parameters
    TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.7"))
    TOP_K = int(os.environ.get("TOP_K", "50"))
    TOP_P = float(os.environ.get("TOP_P", "0.9"))
    REPETITION_PENALTY = float(os.environ.get("REPETITION_PENALTY", "1.1"))
    NO_REPEAT_NGRAM_SIZE = int(os.environ.get("NO_REPEAT_NGRAM_SIZE", "3"))
    print(
        "Generation parameters: "
        f"temperature={TEMPERATURE}, top_k={TOP_K}, top_p={TOP_P}, "
        f"repetition_penalty={REPETITION_PENALTY}, "
        f"no_repeat_ngram_size={NO_REPEAT_NGRAM_SIZE}"
    )

    if not PROMPT:
        if MODEL_TYPE.lower() == "gpt2":
            print("\nNo prompt provided. Generating from scratch...")
            context = torch.tensor(
                [[model.tokenizer.bos_token_id or 0]], dtype=torch.long
            ).to(device)
        else:
            print("\nNo prompt provided. Starting with null token...")
            context = torch.zeros((1, 1), dtype=torch.long).to(device)
    else:
        print(f"\nPrompt: {PROMPT}")
        tokens = encode(PROMPT)
        context = torch.tensor([tokens], dtype=torch.long).to(device)

    # Generate text with sampling tuned for coherence and reduced repetition
    with torch.no_grad():
        generated_tokens = model.generate(
            context,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_k=TOP_K,
            top_p=TOP_P,
            repetition_penalty=REPETITION_PENALTY,
            no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE,
            do_sample=True,
        )

    # Decode and display
    if MODEL_TYPE.lower() == "gpt2":
        # For GPT-2, extract only new tokens
        if generated_tokens.shape[1] > context.shape[1]:
            new_tokens = generated_tokens[0, context.shape[1] :].tolist()
        else:
            new_tokens = generated_tokens[0].tolist()
        generated_text = decode(new_tokens)
    else:
        # For from-scratch, full sequence is new
        generated_text = decode(generated_tokens[0].tolist())

    print("\n" + "=" * 50)
    print("GENERATED TEXT")
    print("=" * 50)
    print(generated_text)
    print("=" * 50)
    
    # Save output to file if requested
    if SAVE_OUTPUT:
        from datetime import datetime
        
        # Create output directory if it doesn't exist
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"inference_output_{timestamp}.txt"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        
        try:
            with open(output_path, "w") as f:
                f.write(f"Checkpoint: {CHECKPOINT_PATH}\n")
                f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Max new tokens: {MAX_NEW_TOKENS}\n")
                if PROMPT:
                    f.write(f"Prompt: {PROMPT}\n")
                f.write("\n" + "=" * 50 + "\n")
                f.write("GENERATED TEXT\n")
                f.write("=" * 50 + "\n")
                f.write(generated_text)
                f.write("\n" + "=" * 50 + "\n")
            print(f"\n✅ Output saved to: {output_path}")
        except Exception as e:
            print(f"\n⚠️  Error saving output: {e}")

# ============================================================================
# RESUME TRAINING MODE
# ============================================================================

elif MODE == "resume":
    print("\n" + "=" * 50)
    print("RESUME TRAINING MODE")
    print("=" * 50)
    print(f"Resuming from step {step}")
    print(f"Training for {RESUME_TRAINING_STEPS} additional steps")
    print(f"Learning rate: {RESUME_LEARNING_RATE}")

    # Create optimizer
    if use_lora:
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=RESUME_LEARNING_RATE)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=RESUME_LEARNING_RATE)

    # Load optimizer state if available
    if optimizer_state_dict:
        try:
            optimizer.load_state_dict(optimizer_state_dict)
            print("✅ Optimizer state loaded")
        except Exception as e:
            print(f"⚠️  Could not load optimizer state: {e}")

    print("\n✅ Ready to resume training!")
    print("To resume, import this checkpoint in your training script:")
    print(f"   CHECKPOINT_PATH={CHECKPOINT_PATH} python3 training_resume.py")
    print("\n(Note: training_resume.py script coming soon)")

else:
    print(f"❌ Error: Unknown mode: {MODE}")
    print("   Valid modes: 'inference' or 'resume'")
    sys.exit(1)

print("\n✅ Done!")
