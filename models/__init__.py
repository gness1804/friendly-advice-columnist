"""
Models package for LLM implementations
"""

try:
    from .bigram_lm import BigramLanguageModel

    __all__ = ["BigramLanguageModel"]
except ImportError:
    # PyTorch/training modules not available (e.g. in Docker web-only image)
    __all__ = []
