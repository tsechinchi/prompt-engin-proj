"""Ollama generation wrapper."""

from __future__ import annotations

from collections.abc import Mapping
import re


def generate_raw(prompt: str, *, model: str, temperature: float = 0.3, num_predict: int = 200) -> str:
    """Call Ollama raw generation with explicit decoding controls."""

    if not model.strip():
        raise ValueError("model must be a non-empty string.")
    if num_predict <= 0:
        raise ValueError("num_predict must be greater than zero.")
    if temperature < 0:
        raise ValueError("temperature must be non-negative.")

    import ollama

    # Keep a per-model hook available, while defaulting to plain prompt text.
    formatted_prompt = _format_prompt_for_model(prompt=prompt, model=model)

    response = ollama.generate(
        model=model,
        prompt=formatted_prompt,
        stream=False,
        raw=True,
        options={
            "num_predict": num_predict,
            "temperature": temperature,
        },
    )
    text = _best_effort_clean(_extract_response_text(response))
    if _is_usable_text(text):
        return text

    # Fallback: retry with plain prompt and non-raw mode if model returned empty text.
    fallback = ollama.generate(
        model=model,
        prompt=prompt,
        stream=False,
        raw=False,
        options={
            "num_predict": num_predict,
            "temperature": temperature,
        },
    )
    fallback_text = _best_effort_clean(_extract_response_text(fallback))
    if _is_usable_text(fallback_text):
        return fallback_text

    # Final fallback: chat mode is often more robust than generate when raw/non-raw both fail.
    chat_fallback = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        stream=False,
        options={
            "num_predict": num_predict,
            "temperature": temperature,
        },
    )
    chat_text = _best_effort_clean(_extract_chat_response_text(chat_fallback))
    if _is_usable_text(chat_text):
        return chat_text

    # Never return an empty string; downstream evaluation expects textual outputs.
    return "[Model returned an empty response]"


def _format_prompt_for_model(*, prompt: str, model: str) -> str:
    """Format prompt per-model when needed for raw decoding."""
    return prompt


def _extract_response_text(response: object) -> str:
    if isinstance(response, Mapping):
        content = response.get("response")
        if isinstance(content, str):
            return content

    content = getattr(response, "response", None)
    if isinstance(content, str):
        return content

    raise ValueError("Ollama response did not contain a 'response' field.")


def _extract_chat_response_text(response: object) -> str:
    if isinstance(response, Mapping):
        message = response.get("message")
        if isinstance(message, Mapping):
            content = message.get("content")
            if isinstance(content, str):
                return content

    message = getattr(response, "message", None)
    if isinstance(message, Mapping):
        content = message.get("content")
        if isinstance(content, str):
            return content

    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content

    raise ValueError("Ollama chat response did not contain message content.")


def _best_effort_clean(text: str) -> str:
    """Clean model text without accidentally erasing all content."""

    stripped = text.strip()
    if not stripped:
        return ""

    cleaned = _sanitize_generated_text(stripped)
    return cleaned if cleaned else stripped


def _is_usable_text(text: str) -> bool:
    """Reject empty or obvious prompt-format echo outputs."""

    if not text.strip():
        return False
    return not _looks_like_prompt_echo(text)


def _looks_like_prompt_echo(text: str) -> bool:
    lowered = text.strip().lower()

    # Common degenerate loop seen in baseline runs.
    if re.fullmatch(r"(?:plain\s+text\s*){3,}", lowered):
        return True

    # Detect when the model mirrors the prompt template instead of answering.
    prompt_markers = ["role:", "task:", "context snippets:", "constraints:", "output format:"]
    if sum(marker in lowered for marker in prompt_markers) >= 3:
        return True

    return False


def _sanitize_generated_text(text: str) -> str:
    """Remove common template markers and trim noisy empty outputs."""

    cleaned = re.sub(r"<\s*start_of_turn[^\n>]*>?", "", text, flags=re.IGNORECASE)
    cleaned = re.sub(r"<\s*end_of_turn[^\n>]*>?", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()
