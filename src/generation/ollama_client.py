"""Ollama generation wrapper."""

from __future__ import annotations

from collections.abc import Mapping


def generate_raw(prompt: str, *, model: str, temperature: float = 0.3, num_predict: int = 200) -> str:
    """Call Ollama raw generation with explicit decoding controls."""

    if not model.strip():
        raise ValueError("model must be a non-empty string.")
    if num_predict <= 0:
        raise ValueError("num_predict must be greater than zero.")
    if temperature < 0:
        raise ValueError("temperature must be non-negative.")

    import ollama

    response = ollama.generate(
        model=model,
        prompt=prompt,
        stream=False,
        raw=True,
        options={
            "num_predict": num_predict,
            "temperature": temperature,
        },
    )
    return _extract_response_text(response)


def _extract_response_text(response: object) -> str:
    if isinstance(response, Mapping):
        content = response.get("response")
        if isinstance(content, str):
            return content

    content = getattr(response, "response", None)
    if isinstance(content, str):
        return content

    raise ValueError("Ollama response did not contain a 'response' field.")
