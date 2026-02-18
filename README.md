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

Questions are screened before processing to ensure they relate to interpersonal matters (relationships, family, friends, workplace dynamics, etc.). Off-topic questions are rejected with a helpful message.

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

### Web Interface (Recommended)

The application includes a web frontend built with FastAPI, HTMX, and Tailwind CSS.

**Start the server:**
```bash
# Load environment variables and start the server
set -a && source .env && set +a && uvicorn app.main:app --reload
```

Then open your browser to **http://127.0.0.1:8000**

The web interface provides:
- A text area to enter your question
- Real-time character count (4000 character limit)
- Loading indicator while processing
- Error messages for off-topic questions
- Conversation history sidebar (persisted in localStorage and DynamoDB)
- "Bring your own API key" model for multi-user support

### Command Line Usage

```bash
# Pose a question via CLI
python qa/advice_mvp.py --question "Your interpersonal question here"
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

### Deployment (AWS)

The app deploys to AWS App Runner with DynamoDB for conversation persistence. For WAF/DDoS protection, associate an AWS WAF Web ACL directly with the App Runner service. See [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) for full details.

```bash
# Dry run
./deploy.sh --dry-run

# Full deployment
export OWNER_API_KEY_HASH="<your-hash>" && ./deploy.sh
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
├── app/                   # Web application (FastAPI)
│   ├── main.py            # FastAPI app entry point
│   ├── dynamodb.py        # DynamoDB conversation persistence
│   ├── routes/            # API endpoints
│   ├── templates/         # Jinja2 HTML templates
│   └── static/            # CSS and JS (Tailwind, HTMX)
├── deploy.sh              # AWS deployment script
├── Dockerfile             # Container definition
├── training.py            # Main training script
├── training/              # Training utilities
│   ├── config.py          # Configuration management
│   ├── data.py            # Data loading and processing
│   └── checkpointing.py   # Checkpoint management
├── models/                # Model implementations
│   ├── bigram_lm_v2.py    # Transformer model
│   ├── gpt2_wrapper.py    # GPT-2 fine-tuning wrapper
│   └── prompts.py         # System prompts for LLMs
├── qa/                    # Question-answering inference
│   ├── run_inference.py   # Inference script
│   └── advice_mvp.py      # MVP script
├── tests/                 # Test suite
├── sources/               # Training data
└── docs/                  # Documentation
```

## Origins

The core transformer implementation in this project originated from [Andrej Karpathy's tutorial](https://www.youtube.com/watch?v=kCc8FmEb1nY) on building language models from scratch. The project has since evolved into a specialized application for generating advice column responses.

## License

This is a personal project.
