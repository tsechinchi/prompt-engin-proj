"""Token usage tracking helpers."""

from __future__ import annotations


def track_usage(*, prompt_tokens: int, completion_tokens: int) -> dict[str, int]:
    """Store or report token counts for a model call."""
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }

