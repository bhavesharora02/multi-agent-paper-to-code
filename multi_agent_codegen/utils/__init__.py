"""
Utility functions for multi-agent code generation.
"""

# Re-export LLM client from parent project
import sys
from pathlib import Path

# Add parent src directory to path
parent_src = Path(__file__).parent.parent.parent / "src"
if parent_src.exists():
    sys.path.insert(0, str(parent_src))

try:
    from llm.llm_client import LLMClient, LLMProvider
    __all__ = ["LLMClient", "LLMProvider"]
except ImportError:
    # If parent project not available, we'll handle imports in individual modules
    __all__ = []

