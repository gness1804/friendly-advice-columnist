"""
Pytest configuration and fixtures.
"""

import os
import sys
from pathlib import Path

# Set dummy API key before any imports that might need it
os.environ.setdefault("OPENAI_API_KEY", "test-key-for-testing")

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
