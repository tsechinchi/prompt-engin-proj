"""Streaming Ollama generation for Thinking Mode."""

from __future__ import annotations

from collections.abc import Generator
from typing import Any


def generate_stream(
    prompt: str,
    *,
    model: str,
    temperature: float = 0.3,
    num_predict: int = 1000,
) -> Generator[dict[str, Any], None, None]:
    """Stream tokens from Ollama, yielding dicts with type and content.

    Each yielded dict has the shape ``{"type": "token"|"error", "content": str}``.
    This function is intentionally separate from ``generate_raw`` so the
    existing synchronous pipeline remains untouched.
    """

    if not model.strip():
        raise ValueError("model must be a non-empty string.")

    import ollama

    from .ollama_client import _format_prompt_for_model

    formatted_prompt = _format_prompt_for_model(prompt=prompt, model=model)

    try:
        stream = ollama.generate(
            model=model,
            prompt=formatted_prompt,
            stream=True,
            raw=True,
            options={
                "num_predict": num_predict,
                "temperature": temperature,
            },
        )

        for chunk in stream:
            token = ""
            if isinstance(chunk, dict):
                token = chunk.get("response", "")
            else:
                token = getattr(chunk, "response", "")
            if token:
                yield {"type": "token", "content": token}

    except Exception as exc:
        yield {"type": "error", "content": repr(exc)}
