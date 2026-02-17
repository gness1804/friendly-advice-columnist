"""
Bigram Language Model - Building an LLM from Scratch
Following Andrej Karpathy's tutorial
"""

import tiktoken
from models.bigram_lm_v2 import BigramLanguageModel

import time
import torch
import os
import sys
from datetime import datetime
from io import StringIO
from torch.optim.lr_scheduler import LinearLR


def format_time(seconds: float) -> str:
    """
    Format seconds into a human-readable string (hours, minutes, seconds).

    Args:
        seconds: Time in seconds

    Returns:
        Formatted string like "1h 23m 45s" or "23m 45s" or "45s"
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    if secs > 0 or len(parts) == 0:  # Always show seconds if no hours/minutes
        parts.append(f"{secs}s")

    return " ".join(parts)


is_test_mode = os.environ.get("TEST_MODE", "False")

# LoRA configuration
USE_LORA = os.environ.get("USE_LORA", "False").lower() == "true"
# Increased default rank/alpha for better capacity (was 8/16, now 16/32)
LORA_RANK = int(os.environ.get("LORA_RANK", "16"))
LORA_ALPHA = float(os.environ.get("LORA_ALPHA", "32.0"))
LORA_DROPOUT = float(os.environ.get("LORA_DROPOUT", "0.0"))

# Model selection: "from_scratch" or "gpt2"
MODEL_TYPE = os.environ.get("MODEL_TYPE", "from_scratch").lower()
GPT2_MODEL_NAME = os.environ.get(
    "GPT2_MODEL_NAME", "gpt2"
)  # gpt2, gpt2-medium, gpt2-large, gpt2-xl

# Checkpoint configuration
ENABLE_CHECKPOINTS = os.environ.get("ENABLE_CHECKPOINTS", "False").lower() == "true"
CHECKPOINT_INTERVAL = int(
    os.environ.get("CHECKPOINT_INTERVAL", "500")
)  # Save every N steps
CHECKPOINT_DIR = os.environ.get("CHECKPOINT_DIR", "checkpoints")


# Custom BPE tokenization uses HuggingFace tokenizers library
# No need to import here - imported when custom_bpe method is selected

# ============================================================================
# HYPERPARAMETERS
# ============================================================================

DEFAULT_TRAINING_DATA_SOURCE = "sources/carolyn_hax/carolyn_hax_merged_cleaned.md"

# Set to True for fast testing with smaller model, False for full training
TEST_MODE = is_test_mode == "True"
TRAINING_DATA_SOURCE = os.environ.get(
    "TRAINING_DATA_SOURCE", DEFAULT_TRAINING_DATA_SOURCE
)
TOKENIZATION_METHOD = os.environ.get(
    "TOKENIZATION_METHOD", "character"
)  # "character", "gpt2", or "custom_bpe"
CUSTOM_VOCAB_SIZE = os.environ.get("CUSTOM_VOCAB_SIZE", None)

# Output to file configuration
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "outputs")
ENABLE_OUTPUT_TO_FILE = (
    os.environ.get("ENABLE_OUTPUT_TO_FILE", "True").lower() == "true"
)

# ============================================================================
# OUTPUT TO FILE UTILITIES
# ============================================================================


class TeeOutput:
    """Capture stdout while still displaying to terminal"""

    def __init__(self, *files):
        self.files = files
        self.buffer = StringIO()

    def write(self, obj):
        # Write to all files (stdout + buffer)
        for f in self.files:
            f.write(obj)
            f.flush()
        # Also write to buffer for later retrieval
        self.buffer.write(obj)

    def flush(self):
        for f in self.files:
            f.flush()

    def getvalue(self):
        return self.buffer.getvalue()


def get_data_source_name(training_data_source):
    """Extract data source name without extension from path"""
    # Handle both relative and absolute paths
    source_path = os.path.normpath(training_data_source)
    # Get filename with extension
    filename = os.path.basename(source_path)
    # Remove extension
    source_name = os.path.splitext(filename)[0]
    return source_name


def generate_output_filename(
    model_name,
    source_name,
    vocab_size,
    training_steps,
    test_mode,
    use_lora=False,
    lora_rank=None,
    lora_alpha=None,
    model_type="from_scratch",
    gpt2_model_name=None,
):
    """Generate output filename with structured naming convention"""
    # Format timestamp as MMDDYYYY_HHMMSS
    timestamp = datetime.now().strftime("%m%d%Y_%H%M%S")

    # Construct filename components
    components = [
        "build_llm_output",
        model_name,
        source_name,
        str(vocab_size),
        str(training_steps),
        f"test={str(test_mode).lower()}",
    ]

    # Add model type information
    if model_type == "gpt2":
        if gpt2_model_name:
            components.append(f"gpt2_{gpt2_model_name}")
        else:
            components.append("gpt2")
    else:
        components.append("from_scratch")

    # Add LoRA information if used
    if use_lora:
        lora_info = f"lora_r{lora_rank}_a{lora_alpha}"
        components.append(lora_info)
    else:
        components.append("full_ft")  # full fine-tuning

    components.extend(["OUTPUT", timestamp])

    # Join with underscores (snake_case)
    filename = "_".join(components) + ".txt"
    return filename


def write_output_file(output_path, hyperparameters, captured_output):
    """Write hyperparameters and captured output to file"""
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            # Write hyperparameters section
            f.write("HYPERPARAMETERS\n")
            f.write("=" * 16 + "\n")
            for key, value in hyperparameters.items():
                f.write(f"{key} = {value}\n")

            f.write("\n")

            # Write output section
            f.write("OUTPUT\n")
            f.write("=" * 6 + "\n")
            f.write(captured_output)

        return True
    except Exception as e:
        print(f"⚠️  Error writing output file: {e}")
        return False


# Create output directory if it doesn't exist
if ENABLE_OUTPUT_TO_FILE:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

# Create checkpoint directory if it doesn't exist
if ENABLE_CHECKPOINTS:
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Create log directory for checkpoint logs
LOG_DIR = os.environ.get("LOG_DIR", "logs")
if ENABLE_CHECKPOINTS:
    os.makedirs(LOG_DIR, exist_ok=True)

# Initialize stdout capture if output to file is enabled
tee_output = None
original_stdout = None
if ENABLE_OUTPUT_TO_FILE:
    original_stdout = sys.stdout
    tee_output = TeeOutput(sys.stdout)
    sys.stdout = tee_output

# Set random seed for reproducibility
torch.manual_seed(1337)

# Model hyperparameters
# Allow override of training_steps via environment variable for quick testing
TRAINING_STEPS_OVERRIDE = os.environ.get("TRAINING_STEPS", None)

if TEST_MODE:
    # Fast configuration for testing and debugging
    batch_size = 32  # Reduced from 64
    block_size = 64  # Reduced from 256 (16x less attention computation!)
    training_steps = (
        int(TRAINING_STEPS_OVERRIDE) if TRAINING_STEPS_OVERRIDE else 1000
    )  # Reduced from 5000
    eval_interval = 100  # Evaluate more frequently
    learning_rate = 3e-4  # Learning rate for optimizer
    eval_iters = 50  # Reduced from 200
    n_embd = 128  # Reduced from 384
    n_head = 4  # Reduced from 6
    n_layer = 3  # Reduced from 6 (half the layers!)
    dropout = 0.2  # Dropout rate for self-attention
    print("🔬 TEST MODE: Using reduced hyperparameters for fast training")
else:
    # Full configuration for production training (aggressively optimized for Apple Silicon)
    # GPT-2 specific optimizations: fine-tuning requires lower LR and larger context
    if MODEL_TYPE == "gpt2":
        # GPT-2 fine-tuning: diagnostic run with stabilized hyperparameters
        # Reduced block size (128) and increased batch size (32) to prevent gradient explosion
        # Allow override via environment variables
        batch_size = int(os.environ.get("BATCH_SIZE", "16"))  # Default 16 (user may need to reduce further for OOM)
        block_size = int(os.environ.get("BLOCK_SIZE", "128"))  # Reduced from 256 for MPS stability and speed
        eval_iters = 20  # Reduced for faster evaluation (was 50)
        learning_rate = float(os.environ.get("LEARNING_RATE", "2e-5"))  # Increased from 1e-5 for better convergence
        print(
            f"   📌 GPT-2 fine-tuning: Using LR ({learning_rate:.2e}), batch_size={batch_size}, block_size={block_size}"
        )
    else:
        # From-scratch models can handle larger batches and higher learning rates
        # Allow override via environment variables
        batch_size = int(os.environ.get("BATCH_SIZE", "64"))  # Reduced from 64 for better M4 performance
        block_size = int(os.environ.get("BLOCK_SIZE", "128"))  # Further reduced from 128 (4x less attention computation)
        eval_iters = 50  # Further reduced from 50 for faster eval
        learning_rate = float(os.environ.get("LEARNING_RATE", "3e-4"))  # Higher LR is OK for from-scratch training

    training_steps = (
        int(TRAINING_STEPS_OVERRIDE) if TRAINING_STEPS_OVERRIDE else 5000
    )  # Number of training iterations
    eval_interval = 500  # More frequent feedback (reduced from 500)
    n_embd = 256  # Further reduced from 256 for Apple Silicon
    n_head = 4  # Further reduced from 4 for Apple Silicon
    n_layer = 4  # Further reduced from 4 for Apple Silicon
    dropout = 0.2  # Dropout rate for self-attention
    print(
        "🚀 FULL MODE: Using production hyperparameters (aggressively optimized for M4)"
    )

USE_LR_WARMUP = os.environ.get("USE_LR_WARMUP", "True").lower() == "true"

# Device selection: prioritize MPS (Apple Silicon GPU) > CUDA > CPU
if torch.cuda.is_available():
    device = "cuda"
    print("✅ Using NVIDIA GPU (CUDA)")
elif torch.backends.mps.is_available():
    device = "mps"
    print("✅ Using Apple Silicon GPU (Metal Performance Shaders)")
else:
    device = "cpu"
    print("⚠️  Using CPU (slow) - consider using MPS if available")

# Generation settings
max_new_tokens = 300  # Number of characters to generate
generation_temperature = (
    0.7  # Lower temperature for more coherent generation (default: 1.0)
)
generation_top_k = 50  # Top-k sampling for generation

# Collect hyperparameters for output file
hyperparameters = {
    "batch_size": batch_size,
    "block_size": block_size,
    "training_steps": training_steps,
    "eval_interval": eval_interval,
    "learning_rate": learning_rate,
    "eval_iters": eval_iters,
    "n_embd": n_embd,
    "n_head": n_head,
    "n_layer": n_layer,
    "dropout": dropout,
    "max_new_tokens": max_new_tokens,
    "device": device,
    "tokenization_method": TOKENIZATION_METHOD,
    "test_mode": TEST_MODE,
    "use_lora": USE_LORA,
    "model_type": MODEL_TYPE,
    "training_data_source": TRAINING_DATA_SOURCE,
}

if MODEL_TYPE == "gpt2":
    hyperparameters["gpt2_model_name"] = GPT2_MODEL_NAME

if USE_LORA:
    hyperparameters.update(
        {
            "lora_rank": LORA_RANK,
            "lora_alpha": LORA_ALPHA,
            "lora_dropout": LORA_DROPOUT,
        }
    )

# Add learning rate schedule info
if MODEL_TYPE == "gpt2":
    if USE_LR_WARMUP and scheduler is not None:
        hyperparameters["lr_schedule"] = "warmup_constant"
    else:
        hyperparameters["lr_schedule"] = "constant"
else:
    hyperparameters["lr_schedule"] = "none"  # From-scratch models don't use scheduler

    print(f"Device: {device}")
    print(f"Model size: {n_layer} layers, {n_embd} embedding dims, {n_head} heads")
    print(f"Training data source: {TRAINING_DATA_SOURCE}")

# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

# Load training data
with open(TRAINING_DATA_SOURCE, "r", encoding="utf-8") as f:
    text = f.read()

# TOKENIZATION OPTIONS (for from_scratch models):
# Option 1: Character-level (original - fastest, simplest, best for small datasets)
# Option 2: GPT-2 BPE via tiktoken (industry standard but large vocab: 50,257 tokens)
# Option 3: Custom BPE tokenizer (balanced - train on your dataset for optimal vocab size)

# If using GPT-2 model, skip custom tokenization (GPT-2 has its own tokenizer)
if MODEL_TYPE != "gpt2":
    # Choose tokenization method:
    if TOKENIZATION_METHOD == "gpt2":
        # GPT-2 BPE: Industry standard but large vocabulary (50,257 tokens)
        # Better for large datasets, but slower and higher loss for small datasets
        enc = tiktoken.get_encoding("gpt2")
        vocab_size = enc.n_vocab

        def encode(s):
            return enc.encode(s)

        def decode(token_ids):
            return enc.decode(token_ids)

        print(f"📝 Using GPT-2 BPE tokenization (vocab_size={vocab_size:,})")
        print("⚠️  Note: Large vocab may cause higher loss and slower training")

    elif TOKENIZATION_METHOD == "custom_bpe":
        # Custom BPE: Train a tokenizer on your dataset (recommended for balanced approach)
        # Uses HuggingFace tokenizers library: pip install tokenizers
        try:
            from tokenizers import Tokenizer
            from tokenizers.models import BPE
            from tokenizers.trainers import BpeTrainer
            from tokenizers.pre_tokenizers import Whitespace
            import math

            # Automatically scale vocab size based on dataset characteristics
            # This balances efficiency (compression) with model size (parameters)
            # Can be overridden with CUSTOM_VOCAB_SIZE environment variable
            manual_vocab_size = CUSTOM_VOCAB_SIZE

            dataset_size_bytes = len(text.encode("utf-8"))
            dataset_size_mb = dataset_size_bytes / (1024 * 1024)
            num_unique_chars = len(set(text))

            if manual_vocab_size and manual_vocab_size.lower() != "none":
                target_vocab_size = int(CUSTOM_VOCAB_SIZE)
                print(f"🔧 Using manual vocab_size override: {target_vocab_size:,}")
            else:
                # Calculate target vocab size using heuristics:
                # 1. Base size on dataset size (larger datasets benefit from larger vocabs)
                # 2. Minimum vocab size ensures good coverage
                # 3. Maximum vocab size caps model complexity

                # Heuristic: vocab_size scales logarithmically with dataset size
                # Formula: base_vocab + log2(dataset_size_mb) * scale_factor
                # This ensures:
                # - Small datasets (< 1MB): ~500-1000 tokens
                # - Medium datasets (1-10MB): ~1000-3000 tokens
                # - Large datasets (> 10MB): ~3000-8000 tokens

                base_vocab = 500
                scale_factor = 200

                # Calculate target vocab size
                if dataset_size_mb < 0.1:
                    # Very small datasets: use smaller vocab
                    target_vocab_size = max(256, min(1000, num_unique_chars * 4))
                elif dataset_size_mb < 1.0:
                    # Small datasets: 500-1500 tokens
                    target_vocab_size = base_vocab + int(
                        math.log2(max(0.1, dataset_size_mb)) * scale_factor
                    )
                elif dataset_size_mb < 10.0:
                    # Medium datasets: 1000-3000 tokens
                    target_vocab_size = base_vocab + int(
                        math.log2(max(1.0, dataset_size_mb)) * scale_factor
                    )
                else:
                    # Large datasets: 3000-8000 tokens
                    target_vocab_size = min(
                        8000,
                        base_vocab
                        + int(math.log2(max(10.0, dataset_size_mb)) * scale_factor),
                    )

                    # Ensure vocab size is reasonable (not too small, not too large)
                    target_vocab_size = max(256, min(10000, target_vocab_size))

                    # Round to nearest 100 for cleaner numbers
                    target_vocab_size = int(round(target_vocab_size / 100) * 100)

                    print("🔧 Training custom BPE tokenizer...")
                    print(
                        f"   Dataset size: {dataset_size_mb:.2f} MB, Unique chars: {num_unique_chars}"
                    )
                    print(f"   Auto-selected vocab_size: {target_vocab_size:,}")

            # Initialize tokenizer with BPE model
            tokenizer = Tokenizer(BPE(unk_token="<unk>"))
            tokenizer.pre_tokenizer = Whitespace()

            # Train the tokenizer (no verbose logging by default - clean and simple!)
            trainer = BpeTrainer(vocab_size=target_vocab_size, special_tokens=["<unk>"])
            tokenizer.train_from_iterator([text], trainer=trainer)

            vocab_size = tokenizer.get_vocab_size()

            def encode(s):
                return tokenizer.encode(s).ids

            def decode(token_ids):
                return tokenizer.decode(token_ids)

            print(f"✅ Custom BPE tokenizer trained (vocab_size={vocab_size:,})")
        except ImportError:
            print("❌ tokenizers not installed. Install with: pip install tokenizers")
            print("📝 Falling back to character-level tokenization")
            # Fall through to character-level implementation
            TOKENIZATION_METHOD = "character"

    if TOKENIZATION_METHOD == "character" or TOKENIZATION_METHOD not in [
        "gpt2",
        "custom_bpe",
    ]:
        # Character-level: Simple, fast, best for small datasets
        # Create vocabulary from all unique characters in the text
        chars = sorted(list(set(text)))
        vocab_size = len(chars)

        # Create character-to-integer and integer-to-character mappings
        stoi = {ch: i for i, ch in enumerate(chars)}
        itos = {i: ch for i, ch in enumerate(chars)}

        # Encoder: convert string to list of integers
        def encode(s):
            return [stoi[c] for c in s]

        # Decoder: convert list of integers to string
        def decode(token_ids):
            return "".join([itos[i] for i in token_ids])

        print(f"📝 Using character-level tokenization (vocab_size={vocab_size})")
        print("✅ Recommended for small datasets - fastest training and lowest loss")

    # Add vocab_size to hyperparameters after it's determined (works for all tokenization methods)
    hyperparameters["vocab_size"] = vocab_size

    # Encode entire text dataset
    data = torch.tensor(encode(text), dtype=torch.long)

    # Split data into train and validation sets (90% train, 10% validation)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

# ============================================================================
# DATA LOADING UTILITIES
# ============================================================================


def get_batch(split):
    """
    Generate a small batch of data of inputs x and targets y

    Args:
        split: 'train' or 'val' to select which dataset to use

    Returns:
        x: Input sequences of shape (batch_size, block_size)
        y: Target sequences of shape (batch_size, block_size)
    """
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss():
    """
    Estimate the loss of the model on the train and validation sets

    Returns:
        out: Dictionary containing the loss on the train and validation sets
    """
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def get_model_name(model_instance):
    """Extract model name from model class (e.g., 'BigramLanguageModel' -> 'bigram')"""
    if MODEL_TYPE == "gpt2":
        # For GPT-2, use the model name from the wrapper
        if hasattr(model_instance, "model_name"):
            return model_instance.model_name.replace(
                "-", ""
            )  # gpt2-medium -> gpt2medium
        return "gpt2"

    class_name = model_instance.__class__.__name__
    # Convert PascalCase to lowercase (simple heuristic: first word before 'LanguageModel')
    if "LanguageModel" in class_name:
        return class_name.replace("LanguageModel", "").lower()
    if "Wrapper" in class_name:
        return class_name.replace("Wrapper", "").lower()
    return class_name.lower()


def save_checkpoint(step, model, optimizer, model_name, source_name):
    """Save model checkpoint with metadata"""
    checkpoint_data = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "hyperparameters": hyperparameters,
        "vocab_size": vocab_size,
        "block_size": block_size,
        "batch_size": batch_size,
    }

    # Generate checkpoint filename
    timestamp = datetime.now().strftime("%m%d%Y_%H%M%S")
    checkpoint_name = (
        f"checkpoint_{model_name}_{source_name}_step{step:06d}_{timestamp}.pt"
    )
    checkpoint_path = os.path.join(CHECKPOINT_DIR, checkpoint_name)

    try:
        torch.save(checkpoint_data, checkpoint_path)
        return checkpoint_path
    except Exception as e:
        print(f"⚠️  Error saving checkpoint: {e}")
        return None


# ============================================================================
# TRAINING SETUP
# ============================================================================

# Initialize model based on MODEL_TYPE
if MODEL_TYPE == "gpt2":
    # GPT-2 model (pre-trained from HuggingFace)
    from models.gpt2_wrapper import GPT2Wrapper

    print(f"🤖 Using GPT-2 model: {GPT2_MODEL_NAME}")
    if USE_LORA:
        print("🔧 Using LoRA for efficient fine-tuning")
        print(
            f"   LoRA rank: {LORA_RANK}, alpha: {LORA_ALPHA}, dropout: {LORA_DROPOUT}"
        )

    model = GPT2Wrapper(
        model_name=GPT2_MODEL_NAME,
        use_lora=USE_LORA,
        lora_rank=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        device=device,
    )

    # Use GPT-2's tokenizer (already set up in GPT2Wrapper)
    encode = model.encode
    decode = model.decode
    vocab_size = model.get_vocab_size()

    # Encode entire text dataset
    data = torch.tensor(encode(text), dtype=torch.long)

    # Split data into train and validation sets (90% train, 10% validation)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    # GPT-2 uses its own block_size (from config), but we'll use our batch_size
    # Note: GPT-2's max position embeddings might limit block_size
    gpt2_config = model.model.config
    max_pos = gpt2_config.max_position_embeddings
    if block_size > max_pos:
        print(
            f"⚠️  block_size ({block_size}) > GPT-2 max_pos ({max_pos}), using {max_pos}"
        )
        block_size = max_pos

    model.to(device)  # Ensure model is on device

elif MODEL_TYPE == "from_scratch":
    # Original from-scratch model (with or without LoRA)
    if USE_LORA:
        from models.bigram_lm_v2_lora import BigramLanguageModelLoRA

        print("🔧 Using LoRA for efficient fine-tuning")
        print(
            f"   LoRA rank: {LORA_RANK}, alpha: {LORA_ALPHA}, dropout: {LORA_DROPOUT}"
        )
        model = BigramLanguageModelLoRA(
            vocab_size=vocab_size,
            n_embd=n_embd,
            block_size=block_size,
            device=device,
            dropout=dropout,
            n_head=n_head,
            n_layer=n_layer,
            use_lora=True,
            lora_rank=LORA_RANK,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
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

    model.to(device)  # move model to device
else:
    raise ValueError(f"Unknown MODEL_TYPE: {MODEL_TYPE}. Use 'from_scratch' or 'gpt2'")

# Count parameters (for both GPT-2 and from_scratch models)
if MODEL_TYPE == "gpt2":
    # GPT-2 always has parameter info method
    param_info = model.get_parameter_info()
    print("📊 Parameter Statistics:")
    print(f"   Model: {param_info.get('model_name', GPT2_MODEL_NAME)}")
    print(f"   Total parameters: {param_info['total']:,}")
    if USE_LORA:
        print(f"   Trainable (total): {param_info['trainable']:,}")
        if "lora_only_params" in param_info:
            print(
                f"     - LoRA adapters: {param_info['lora_only_params']:,} ({param_info.get('lora_only_percentage', 0):.2f}%)"
            )
        print(f"   Frozen (base model): {param_info['frozen']:,}")
        lora_pct = param_info.get(
            "lora_percentage", param_info.get("lora_only_percentage", 0)
        )
        print(f"   💰 LoRA savings: Training only {lora_pct:.2f}% of parameters!")
    else:
        print(f"   Trainable parameters: {param_info['trainable']:,}")
        print(
            "   💡 Tip: Use USE_LORA=True for efficient fine-tuning (90-99% fewer parameters)"
        )
elif USE_LORA:
    param_info = model.get_parameter_info()
    print("📊 Parameter Statistics:")
    print(f"   Total parameters: {param_info['total']:,}")
    print(f"   Trainable (total): {param_info['trainable']:,}")
    print(
        f"     - LoRA adapters: {param_info['lora_only_params']:,} ({param_info['lora_only_percentage']:.2f}%)"
    )
    print(
        f"     - Embeddings: {param_info['embedding_params']:,} ({param_info['embedding_percentage']:.2f}%)"
    )
    print(f"   Frozen (base model): {param_info['frozen']:,}")
    print(
        f"   💰 LoRA savings: Training only {param_info['lora_only_percentage']:.2f}% of parameters (LoRA-only)!"
    )
    if param_info["embedding_percentage"] > 5.0:
        print(
            f"   ⚠️  Note: Embeddings are {param_info['embedding_percentage']:.1f}% of total (higher in small models)"
        )

    # Warn about performance for small models
    if param_info["total"] < 5_000_000:  # Less than 5M parameters
        print(
            "   ⚠️  PERFORMANCE WARNING: LoRA may be SLOWER than full fine-tuning for small models!"
        )
        print("       For models < 5M params, the LoRA overhead can outweigh benefits.")
        print(
            "       Consider using full fine-tuning (USE_LORA=False) for better speed."
        )
else:
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total model parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

# Verify device usage
print(f"Model device: {next(model.parameters()).device}")
if device == "mps":
    print("✅ Model successfully moved to Apple Silicon GPU (MPS)")

# Compile model for better performance (PyTorch 2.0+)
# This can provide 2-3x speedup on Apple Silicon M4
# DISABLED: torch.compile for MPS is still experimental and may cause slowdowns
try:
    if device == "mps" and hasattr(torch, "compile") and False:  # Disabled for now
        print("🔧 Compiling model for Apple Silicon... (this may take a minute)")
        model = torch.compile(model, mode="default")
        print("✅ Model compiled successfully!")
    else:
        print("ℹ️  Using MPS without compilation (torch.compile disabled for MPS)")
except Exception as e:
    print(f"⚠️  Model compilation skipped: {e}")

# Initialize optimizer
# When using LoRA, only LoRA parameters are trainable (base model is frozen)
if USE_LORA:
    # Only optimize trainable parameters (LoRA adapters)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate)
    print(
        f"✅ Optimizer initialized with {len(trainable_params)} parameter groups (LoRA only)"
    )
else:
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate
    )  # AdamW optimizer

# Initialize learning rate scheduler for GPT-2 fine-tuning
# Option to use gentle warmup for stability (can be disabled via env var)
scheduler = None
if MODEL_TYPE == "gpt2":
    if USE_LR_WARMUP:
        # Gentle warmup: 2% of training steps (prevents early instability)
        # Then constant for rest (no decay to avoid premature convergence)
        warmup_steps = max(50, int(0.02 * training_steps))  # At least 50 steps, or 2% of total
        if warmup_steps > 0 and warmup_steps < training_steps:
            from torch.optim.lr_scheduler import LinearLR
            warmup_scheduler = LinearLR(
                optimizer, start_factor=0.1, total_iters=warmup_steps
            )
            scheduler = warmup_scheduler
            print(
                f"📈 Learning rate schedule: WARMUP ({warmup_steps} steps) → CONSTANT"
            )
            print(
                f"   LR: {learning_rate * 0.1:.2e} → {learning_rate:.2e} (then constant)"
            )
        else:
            print(
                "📈 Learning rate schedule: CONSTANT (warmup disabled - too few steps)"
            )
            print(f"   LR: {learning_rate:.2e} (constant throughout training)")
    else:
        print(
            "📈 Learning rate schedule: CONSTANT (warmup disabled via USE_LR_WARMUP=False)"
        )
        print(f"   LR: {learning_rate:.2e} (constant throughout training)")

# ============================================================================
# TRAINING LOOP
# ============================================================================

# Print progress every interval_print steps
interval_print = training_steps // 10  # print every 10% of the training steps

print(f"Starting training for {training_steps} steps...")
print(f"Batch size: {batch_size}, Block size: {block_size}")
print(f"Vocabulary size: {vocab_size:,} tokens")
if ENABLE_CHECKPOINTS:
    print(
        f"Checkpoints enabled: saving every {CHECKPOINT_INTERVAL} steps to {CHECKPOINT_DIR}/"
    )
print("-" * 50)

# Get model and source names for checkpoint naming
model_name = get_model_name(model)
source_name = get_data_source_name(TRAINING_DATA_SOURCE)

# Initialize checkpoint log file
checkpoint_log_file = None
checkpoint_log_path = None
if ENABLE_CHECKPOINTS:
    timestamp = datetime.now().strftime("%m%d%Y_%H%M%S")
    log_filename = f"training_log_{model_name}_{source_name}_{timestamp}.log"
    checkpoint_log_path = os.path.join(LOG_DIR, log_filename)
    checkpoint_log_file = open(checkpoint_log_path, "w", encoding="utf-8")
    checkpoint_log_file.write(f"Training Log - Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    checkpoint_log_file.write("=" * 80 + "\n")
    checkpoint_log_file.flush()
    print(f"📝 Checkpoint log: {checkpoint_log_path}")

start_time = time.time()

# Initialize loss tracking variables
initial_train_loss = None
last_checkpoint_train_loss = None

for step in range(training_steps):

    # Every once in a while evaluate the loss on train and val sets
    if step % eval_interval == 0:
        losses = estimate_loss()
        current_train_loss = losses['train'].item()  # Convert tensor to Python float
        
        # Store initial loss at step 0
        if initial_train_loss is None:
            initial_train_loss = current_train_loss
            # Also set as last checkpoint loss so first checkpoint compares to step 0
            last_checkpoint_train_loss = current_train_loss
        
        # Calculate loss changes
        net_loss_change_since_beginning = current_train_loss - initial_train_loss
        
        # Calculate change since last checkpoint
        # At step 0, this will be 0.0 (comparing to itself)
        # At subsequent checkpoints, this compares to the previous checkpoint
        net_loss_change_since_last_checkpoint = current_train_loss - last_checkpoint_train_loss
        
        elapsed = time.time() - start_time
        steps_per_sec = step / elapsed if step > 0 else 0
        progress_pct = (step / training_steps) * 100
        # Get current learning rate for display
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"step {step}/{training_steps} ({progress_pct:.1f}%): train loss {losses['train']:.4f}, val loss {losses['val']:.4f} | LR: {current_lr:.2e} | {elapsed:.1f}s ({steps_per_sec:.2f} steps/sec) | Net loss change since beginning: {net_loss_change_since_beginning:.4f} | Net loss change since last checkpoint: {net_loss_change_since_last_checkpoint:.4f}"
        )

        # Save checkpoint if enabled
        if ENABLE_CHECKPOINTS and step % CHECKPOINT_INTERVAL == 0 and step > 0:
            checkpoint_path = save_checkpoint(
                step, model, optimizer, model_name, source_name
            )
            if checkpoint_path:
                print(f"   💾 Checkpoint saved: {checkpoint_path}")
                # Update last checkpoint loss for next comparison
                last_checkpoint_train_loss = current_train_loss
                # Append current output to checkpoint log
                if checkpoint_log_file:
                    # Get all output captured so far
                    if tee_output:
                        log_content = tee_output.getvalue()
                    else:
                        # If no tee_output, we can't capture past output, but we can log current state
                        log_content = f"\n--- Checkpoint at step {step} ---\n"
                        log_content += f"Train loss: {losses['train']:.4f}, Val loss: {losses['val']:.4f}\n"
                        log_content += f"Learning rate: {current_lr:.2e}\n"
                        log_content += f"Elapsed time: {elapsed:.1f}s\n"
                        log_content += f"Steps/sec: {steps_per_sec:.2f}\n"
                    checkpoint_log_file.write(f"\n{'='*80}\n")
                    checkpoint_log_file.write(f"CHECKPOINT at step {step} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    checkpoint_log_file.write(f"{'='*80}\n")
                    checkpoint_log_file.write(log_content)
                    checkpoint_log_file.flush()

    # Print progress more frequently in production mode to show it's not hung
    elif step > 0 and step % 25 == 0:
        elapsed = time.time() - start_time
        steps_per_sec = step / elapsed
        progress_pct = (step / training_steps) * 100
        eta_seconds = (
            (training_steps - step) / steps_per_sec if steps_per_sec > 0 else 0
        )
        eta_minutes = eta_seconds / 60
        print(
            f"step {step}/{training_steps} ({progress_pct:.1f}%) | {steps_per_sec:.2f} steps/sec | ETA: {eta_minutes:.1f}m"
        )

    # Sample a batch of data
    xb, yb = get_batch("train")

    # Forward pass: compute predictions and loss
    logits, loss = model(xb, yb)

    # Backward pass: compute gradients
    optimizer.zero_grad(set_to_none=True)
    loss.backward()

    # Gradient clipping to prevent exploding gradients (especially important for fine-tuning)
    if MODEL_TYPE == "gpt2":
        # For LoRA, only clip trainable parameters (LoRA adapters)
        # Clipping frozen parameters is unnecessary and can cause issues
        if USE_LORA:
            trainable_params = [p for p in model.parameters() if p.requires_grad]
            if len(trainable_params) > 0:
                # Less aggressive clipping for LoRA (allows adapters to learn better)
                # 2.0 is a good balance: prevents explosion but allows necessary updates
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=2.0)
        else:
            # For full fine-tuning, clip all parameters
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    # Update parameters
    optimizer.step()

    # Update learning rate schedule
    if scheduler is not None:
        scheduler.step()

print("-" * 50)
print(f"Training complete! Final loss: {loss.item():.4f}")
total_time = time.time() - start_time
print(
    f"Total training time: {format_time(total_time)} ({training_steps/total_time:.2f} steps/sec)"
)

# NEW: run a final evaluation pass
final_losses = estimate_loss()
print(
    f"Final eval - train loss {final_losses['train']:.4f}, val loss {final_losses['val']:.4f}"
)

# Save final model checkpoint
if ENABLE_CHECKPOINTS:
    final_checkpoint_path = save_checkpoint(
        training_steps, model, optimizer, model_name, source_name
    )
    if final_checkpoint_path:
        print(f"✅ Final model saved: {final_checkpoint_path}")
    # Write final output to checkpoint log
    if checkpoint_log_file:
        if tee_output:
            log_content = tee_output.getvalue()
        else:
            log_content = f"\n--- Final checkpoint at step {training_steps} ---\n"
            log_content += f"Final loss: {loss.item():.4f}\n"
        checkpoint_log_file.write(f"\n{'='*80}\n")
        checkpoint_log_file.write(f"FINAL CHECKPOINT at step {training_steps} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        checkpoint_log_file.write(f"{'='*80}\n")
        checkpoint_log_file.write(log_content)
        checkpoint_log_file.write(f"\nTraining completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        checkpoint_log_file.close()
        print(f"📝 Checkpoint log saved: {checkpoint_log_path}")

# ============================================================================
# GENERATION
# ============================================================================

print("\nGenerating text...")
print("=" * 50)

print("\nGenerating text...")
print("=" * 50)

if MODEL_TYPE == "gpt2":
    # Use an in-domain prompt instead of an empty context
    prompt = "QUESTION: "
    input_ids = torch.tensor(
        [encode(prompt)], dtype=torch.long
    ).to(device)

    generated_tokens = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=generation_temperature,
        top_k=generation_top_k,
        do_sample=True,
    )

    generated_text = decode(generated_tokens[0].tolist())
else:
    # From-scratch model: start with null token
    context = torch.zeros((1, 1), dtype=torch.long).to(device)
    generated_tokens = model.generate(context, max_new_tokens=max_new_tokens)
    generated_text = decode(generated_tokens[0].tolist())

print(generated_text)
print("=" * 50)

# ============================================================================
# WRITE OUTPUT TO FILE
# ============================================================================

if ENABLE_OUTPUT_TO_FILE:
    # Restore stdout
    sys.stdout = original_stdout

    # Get captured output
    captured_output = tee_output.getvalue()

    # Generate filename (reuse model_name and source_name from checkpoint section)
    filename = generate_output_filename(
        model_name=model_name,
        source_name=source_name,
        vocab_size=vocab_size,
        training_steps=training_steps,
        test_mode=TEST_MODE,
        use_lora=USE_LORA,
        lora_rank=LORA_RANK if USE_LORA else None,
        lora_alpha=LORA_ALPHA if USE_LORA else None,
        model_type=MODEL_TYPE,
        gpt2_model_name=GPT2_MODEL_NAME if MODEL_TYPE == "gpt2" else None,
    )

    # Construct full output path
    output_path = os.path.join(OUTPUT_DIR, filename)

    # Write output file
    if write_output_file(output_path, hyperparameters, captured_output):
        print(f"\n✅ Output written to: {output_path}")
    else:
        print("\n⚠️  Failed to write output file")
