"""
Data loading and tokenization for training.

Provides:
- Loading raw text from file
- Tokenization setup (character, GPT-2 BPE, custom BPE)
- Dataset splitting into train/val
- Batch generation utilities
"""

import os
import torch
import math
from typing import Tuple, Callable, Optional


def load_raw_text(data_source: str) -> str:
    """
    Load raw text from file.
    
    Args:
        data_source: Path to text file
    
    Returns:
        Raw text content
    
    Raises:
        FileNotFoundError: If file doesn't exist
    """
    if not os.path.exists(data_source):
        raise FileNotFoundError(f"Training data not found at {data_source}")
    
    with open(data_source, "r", encoding="utf-8") as f:
        text = f.read()
    
    return text


def _setup_character_tokenization(text: str) -> Tuple[Callable, Callable, int]:
    """
    Set up character-level tokenization.
    
    Args:
        text: Raw text to build vocab from
    
    Returns:
        tuple: (encode_fn, decode_fn, vocab_size)
    """
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    
    def encode(s):
        return [stoi[c] for c in s]
    
    def decode(token_ids):
        return "".join([itos[i] for i in token_ids])
    
    return encode, decode, vocab_size


def _setup_gpt2_tokenization() -> Tuple[Callable, Callable, int]:
    """
    Set up GPT-2 BPE tokenization via tiktoken.
    
    Returns:
        tuple: (encode_fn, decode_fn, vocab_size)
    """
    import tiktoken
    
    enc = tiktoken.get_encoding("gpt2")
    vocab_size = enc.n_vocab
    
    def encode(s):
        return enc.encode(s)
    
    def decode(token_ids):
        return enc.decode(token_ids)
    
    return encode, decode, vocab_size


def _setup_custom_bpe_tokenization(
    text: str,
    custom_vocab_size: Optional[str] = None,
) -> Tuple[Callable, Callable, int]:
    """
    Set up custom BPE tokenization.
    
    Args:
        text: Raw text to train tokenizer on
        custom_vocab_size: Manual vocab size override (as string or None)
    
    Returns:
        tuple: (encode_fn, decode_fn, vocab_size)
    
    Raises:
        ImportError: If tokenizers package not installed
    """
    try:
        from tokenizers import Tokenizer
        from tokenizers.models import BPE
        from tokenizers.trainers import BpeTrainer
        from tokenizers.pre_tokenizers import Whitespace
    except ImportError:
        raise ImportError(
            "tokenizers package not installed. Install with: pip install tokenizers"
        )
    
    # Determine target vocab size
    dataset_size_bytes = len(text.encode("utf-8"))
    dataset_size_mb = dataset_size_bytes / (1024 * 1024)
    num_unique_chars = len(set(text))
    
    if custom_vocab_size and custom_vocab_size.lower() != "none":
        target_vocab_size = int(custom_vocab_size)
        print(f"🔧 Using manual vocab_size override: {target_vocab_size:,}")
    else:
        # Auto-calculate based on dataset size
        base_vocab = 500
        scale_factor = 200
        
        if dataset_size_mb < 0.1:
            target_vocab_size = max(256, min(1000, num_unique_chars * 4))
        elif dataset_size_mb < 1.0:
            target_vocab_size = base_vocab + int(
                math.log2(max(0.1, dataset_size_mb)) * scale_factor
            )
        elif dataset_size_mb < 10.0:
            target_vocab_size = base_vocab + int(
                math.log2(max(1.0, dataset_size_mb)) * scale_factor
            )
        else:
            target_vocab_size = min(
                8000,
                base_vocab + int(math.log2(max(10.0, dataset_size_mb)) * scale_factor),
            )
        
        target_vocab_size = max(256, min(10000, target_vocab_size))
        target_vocab_size = int(round(target_vocab_size / 100) * 100)
        
        print("🔧 Training custom BPE tokenizer...")
        print(f"   Dataset size: {dataset_size_mb:.2f} MB, Unique chars: {num_unique_chars}")
        print(f"   Auto-selected vocab_size: {target_vocab_size:,}")
    
    # Initialize and train tokenizer
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    
    trainer = BpeTrainer(vocab_size=target_vocab_size, special_tokens=["<unk>"])
    tokenizer.train_from_iterator([text], trainer=trainer)
    
    vocab_size = tokenizer.get_vocab_size()
    
    def encode(s):
        return tokenizer.encode(s).ids
    
    def decode(token_ids):
        return tokenizer.decode(token_ids)
    
    print(f"✅ Custom BPE tokenizer trained (vocab_size={vocab_size:,})")
    
    return encode, decode, vocab_size


def prepare_data_and_tokenizer(
    config,
    raw_text: str,
    model_type: str,
) -> Tuple[Callable, Callable, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """
    Prepare data and tokenization based on config.
    
    Handles:
    - Character-level tokenization
    - GPT-2 BPE tokenization
    - Custom BPE tokenization
    - Train/val splitting
    
    Args:
        config: TrainingConfig instance
        raw_text: Raw text to tokenize
        model_type: Type of model ("gpt2" or "from_scratch")
    
    Returns:
        tuple: (encode, decode, data, train_data, val_data, vocab_size)
    """
    tokenization_method = config.tokenization_method
    
    # For GPT-2, tokenization is handled by GPT2Wrapper; we shouldn't call this
    if model_type == "gpt2":
        raise ValueError(
            "GPT-2 tokenization should be handled by GPT2Wrapper, not prepare_data_and_tokenizer"
        )
    
    # Set up tokenization based on method
    if tokenization_method == "gpt2":
        print("📝 Using GPT-2 BPE tokenization (vocab_size=50,257)")
        print("⚠️  Note: Large vocab may cause higher loss and slower training")
        encode, decode, vocab_size = _setup_gpt2_tokenization()
    
    elif tokenization_method == "custom_bpe":
        encode, decode, vocab_size = _setup_custom_bpe_tokenization(
            raw_text,
            config.custom_vocab_size,
        )
    
    elif tokenization_method == "character":
        encode, decode, vocab_size = _setup_character_tokenization(raw_text)
        print(f"📝 Using character-level tokenization (vocab_size={vocab_size})")
        print("✅ Recommended for small datasets - fastest training and lowest loss")
    
    else:
        raise ValueError(f"Unknown tokenization method: {tokenization_method}")
    
    # Encode entire dataset
    data = torch.tensor(encode(raw_text), dtype=torch.long)
    
    # Split into train and validation (90% train, 10% val)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]
    
    return encode, decode, data, train_data, val_data, vocab_size


def prepare_gpt2_data(
    encode_fn: Callable,
    raw_text: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Prepare train/val data for GPT-2.
    
    Args:
        encode_fn: GPT2Wrapper.encode function
        raw_text: Raw text to encode
    
    Returns:
        tuple: (train_data, val_data)
    """
    data = torch.tensor(encode_fn(raw_text), dtype=torch.long)
    
    # Split into train and validation (90% train, 10% val)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]
    
    return train_data, val_data


def get_batch(
    split: str,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    batch_size: int,
    block_size: int,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate a batch of training data.
    
    Args:
        split: "train" or "val"
        train_data: Training dataset
        val_data: Validation dataset
        batch_size: Batch size
        block_size: Sequence length
        device: Device to put data on
    
    Returns:
        tuple: (x, y) - input and target sequences
    """
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y
