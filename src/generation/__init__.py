"""Model generation wrappers."""

from .ollama_client import generate_raw
from .ollama_stream import generate_stream

__all__ = ["generate_raw", "generate_stream"]
