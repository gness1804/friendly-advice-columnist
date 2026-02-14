import os
from openai import OpenAI

from models.prompts import ADVICE_COLUMNIST_SYSTEM_PROMPT, SYSTEM_PROMPT_V3

# Support both old and new environment variable patterns
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "ft:gpt-4.1-mini")
BASE_MODEL = os.environ.get("BASE_MODEL", "gpt-4.1-mini")
FINE_TUNED_MODEL = os.environ.get(
    "FINE_TUNED_MODEL",
    "ft:gpt-4.1-mini-2025-04-14:personal:friendly-advice-01092026:CwGsaVcA",
)

# Module-level client for backward compatibility (CLI usage)
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def generate_answer(
    input: str,
    version: str = "v1",
    model: str = None,
    api_key: str = None,
    system_prompt: str = None,
) -> str:
    """
    Generate an answer using OpenAI API.

    Args:
        input: The input prompt/question
        version: "v1" for base advice columnist, "v3" for revision/critique
        model: Optional model name to override default. If None, uses:
            - For v1: BASE_MODEL env var (default: "gpt-4o-mini")
            - For v3: FINE_TUNED_MODEL env var (default: fine-tuned model)
            - Falls back to OPENAI_MODEL env var for backward compatibility
        api_key: Optional API key for per-request authentication.
            If None, uses the module-level client (env var key).
        system_prompt: Optional system prompt to override the default for the
            given version. If None, uses the standard prompt for that version.

    Returns:
        Generated response text
    """
    # Use per-request client if api_key provided, otherwise module-level client
    openai_client = OpenAI(api_key=api_key) if api_key else client

    if version == "v1":
        messages = [
            {
                "role": "system",
                "content": system_prompt or ADVICE_COLUMNIST_SYSTEM_PROMPT,
            },
            {"role": "user", "content": input},
        ]
        # Use provided model, or BASE_MODEL, or fall back to OPENAI_MODEL
        model_to_use = model or BASE_MODEL or OPENAI_MODEL
    elif version == "v3":
        messages = [
            {"role": "system", "content": system_prompt or SYSTEM_PROMPT_V3},
            {"role": "user", "content": input},
        ]
        # Use provided model, or FINE_TUNED_MODEL, or fall back to OPENAI_MODEL
        model_to_use = model or FINE_TUNED_MODEL or OPENAI_MODEL
    else:
        raise ValueError(f"Invalid version: {version}")

    resp = openai_client.chat.completions.create(
        model=model_to_use,
        messages=messages,
        temperature=float(os.environ.get("TEMPERATURE", 0.3)),
        top_p=float(os.environ.get("TOP_P", 0.9)),
        max_tokens=int(os.environ.get("MAX_NEW_TOKENS", 700)),
    )
    return resp.choices[0].message.content.strip()
