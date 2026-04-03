"""Ollama generation wrapper."""

from __future__ import annotations


def generate_raw(prompt: str, *, model: str, temperature: float = 0.3, num_predict: int = 200) -> str:
    """Call Ollama raw generation with explicit decoding controls."""
    raise NotImplementedError("Ollama client is not implemented yet.")

