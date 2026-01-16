# Friendly Advice Columnist

An AI-powered advice columnist that takes interpersonal questions and provides thoughtful, compassionate responses with a morally grounded perspective.

## Overview

This project is an AI advice columnist designed to help users navigate interpersonal challenges. It provides answers that are:

- **Compassionate**: Understanding and empathetic toward the person asking
- **Calm**: Measured and thoughtful rather than reactive
- **Morally focused**: Grounded in ethical considerations while remaining non-judgmental

## How It Works

The system uses a two-stage LLM architecture:

1. **Base LLM**: Consumes the original question and generates an initial draft response
2. **Fine-tuned LLM**: A model fine-tuned on curated personal advice column examples that refines the draft into a polished final answer

This two-stage approach combines the broad knowledge and reasoning capabilities of a general-purpose LLM with the specific tone and style learned from high-quality advice column examples.

## Project Status

This is currently a **backend-only application**. A frontend interface is planned for future development.

## Prerequisites

- Python 3.8 or higher
- PyTorch 2.0 or higher

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare training data:**
   Training data should be placed in the `sources/` directory in markdown format using Q&A structure.

## Usage

### Basic Usage

```bash
# Pose a question to the application. The application then gives you an answer based on the processing of both LLMs.
python qa/advice_mvp.py --question <question>
```

### Training

*(NOTE: These steps are only for the from-scratch model or GPT-2. GPT-4.1-mini, which is the model that's used in the current MVP, does not follow the steps in this section as it has its own built-in training.)*

```bash
# Train from scratch
python training.py

# GPT-2 fine-tuning
MODEL_TYPE=gpt2 python training.py

# Resume from checkpoint
CHECKPOINT_PATH=checkpoints/checkpoint_step_5000.pt RESUME_STEPS=5000 python training.py
```

### Inference

```bash
# Run inference with a checkpoint
python qa/run_inference.py --prompt 'QUESTION: ...' --checkpoint path/to/checkpoint.pt --version v1

# Run with OpenAI backend
python qa/run_inference.py --prompt 'QUESTION: ...' --model_type openai_backend --version v1
# For version 3: python qa/run_inference.py --prompt '...' --model_type openai_backend --version v3
```

### Linting

```bash
ruff check .
ruff format .
```

## Project Structure

```
friendly-advice-columnist/
├── README.md              # This file
├── CLAUDE.md              # Development guidance
├── training.py            # Main training script
├── training/              # Training utilities
│   ├── config.py          # Configuration management
│   ├── data.py            # Data loading and processing
│   └── checkpointing.py   # Checkpoint management
├── models/                # Model implementations
│   ├── bigram_lm_v2.py    # Transformer model
│   └── gpt2_wrapper.py    # GPT-2 fine-tuning wrapper
├── qa/                    # Question-answering inference
│   └── run_inference.py   # Inference script                    
│   └── advice_mvp.py      # MVP script
├── sources/               # Training data
└── docs/                  # Documentation
```

## Origins

The core transformer implementation in this project originated from [Andrej Karpathy's tutorial](https://www.youtube.com/watch?v=kCc8FmEb1nY) on building language models from scratch. The project has since evolved into a specialized application for generating advice column responses.

## License

This is a personal project.
