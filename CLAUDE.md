# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

An AI-powered advice columnist that provides thoughtful, compassionate responses to interpersonal questions. The system uses a two-stage LLM architecture: a base LLM generates an initial draft, then a fine-tuned model (trained on curated advice column examples) refines it into a polished response.

## House rules

- When starting on a new bug fix or feature on the master branch or similar, Offer to create a new Git branch for the work.

## Common Commands

### Running the Advice Columnist

```bash
# Ask a question (MVP)
python qa/advice_mvp.py --question "Your question here"
```

### Training

```bash
# Fresh training (from scratch model)
python training.py

# GPT-2 fine-tuning
MODEL_TYPE=gpt2 python training.py

# Resume from checkpoint
CHECKPOINT_PATH=checkpoints/checkpoint_step_5000.pt RESUME_STEPS=5000 python training.py

# Test mode (fast iteration)
TEST_MODE=True python training.py
```

### Key Environment Variables

- `MODEL_TYPE`: "from_scratch" (default) or "gpt2"
- `USE_LORA`: "true" to enable LoRA adapters for GPT-2
- `TRAINING_DATA_SOURCE`: Path to training data (default: `sources/training_data_final_merged.md`)
- `TRAINING_STEPS`: Number of training steps
- `BATCH_SIZE`, `BLOCK_SIZE`, `LEARNING_RATE`: Training hyperparameters
- `ENABLE_CHECKPOINTS`: "true" to save checkpoints
- `CHECKPOINT_INTERVAL`: Steps between checkpoints

### Inference

```bash
# Run inference with checkpoint
python qa/run_inference.py --prompt 'QUESTION: ...' --checkpoint path/to/checkpoint.pt --version v1

# Run with OpenAI backend
python qa/run_inference.py --prompt 'QUESTION: ...' --model_type openai_backend --version v1
```

### Linting

```bash
ruff check .
ruff format .
```

## Architecture

### Project Structure

```
training.py              # Main entry point - handles fresh/resume training
├── training/
│   ├── config.py        # TrainingConfig dataclass, env var parsing
│   ├── data.py          # Data loading, tokenization, batch generation
│   ├── checkpointing.py # Checkpoint save/load/logging
│   └── io_utils.py      # Output file handling, stdout capture
├── models/
│   ├── bigram_lm_v2.py  # Full transformer model (from scratch)
│   ├── gpt2_wrapper.py  # HuggingFace GPT-2 wrapper for fine-tuning
│   └── bigram_lm.py     # Simple bigram baseline
├── transformer_core/
│   └── block.py         # Transformer block (attention + FFN with pre-norm)
├── self_attention/
│   └── self_attention_classes.py  # Head and MultiHeadAttention
├── feed_forward/
│   └── feed_forward_classes.py    # FeedForward MLP
├── lora/
│   └── lora_module.py   # LoRA adapters for efficient fine-tuning
└── qa/
    ├── advice_mvp.py    # MVP advice columnist interface
    └── run_inference.py # General inference script
```

### Model Architecture

The transformer uses pre-norm architecture: `LayerNorm → Attention/FFN → Residual`

- **Block**: Combines MultiHeadAttention + FeedForward with residual connections
- **MultiHeadAttention**: Runs multiple attention heads in parallel, concatenates outputs
- **Head**: Single attention head with causal masking (scaled dot-product attention)
- **FeedForward**: Two-layer MLP with ReLU activation (n_embd → 4*n_embd → n_embd)

### Training Flow

1. Detects fresh vs resume mode via `CHECKPOINT_PATH` env var
2. Loads config from env (fresh) or checkpoint (resume)
3. Prepares tokenizer and data (character-level, GPT-2 BPE, or custom BPE)
4. Creates/loads model (BigramLanguageModel or GPT2Wrapper)
5. Training loop with periodic evaluation and checkpointing

### Training Data

Training data lives in `sources/` with markdown format using Q&A structure. The data consists of curated advice column examples. Data preparation scripts in `sources/scripts/` handle:
- Reddit data collection and cleanup
- Carolyn Hax chat merging
- Training data normalization

## Code Conventions

### Tensor Shape Documentation

Document tensor shapes using `(B, T, C)` notation:
- **B**: Batch size
- **T**: Time/sequence length
- **C**: Channels/embedding dimension

```python
token_emb = self.token_embedding_table(idx)  # (B, T, C)
```

### Device Management

Priority: MPS (Apple Silicon) > CUDA > CPU. Always move tensors explicitly:
```python
x, y = x.to(device), y.to(device)
```

### Documentation Files

- Create docs in `docs/` directory, not repo root
- Avoid creating redundant documentation files
- Use `.cursor/tmp/` for temporary/debug files

## Key Files for Context

- `training/config.py`: All hyperparameters and their defaults
- `models/bigram_lm_v2.py`: Main transformer model architecture
- `docs/FINETUNING_GUIDE.md`: LoRA and fine-tuning strategies
- `docs/CHECKPOINT_USAGE.md`: Checkpoint management

## Origins

The core transformer implementation originated from [Andrej Karpathy's tutorial](https://www.youtube.com/watch?v=kCc8FmEb1nY) on building language models from scratch. The project has since evolved into a specialized application for generating advice column responses.
