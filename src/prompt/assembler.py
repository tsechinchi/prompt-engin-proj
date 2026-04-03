"""Prompt assembly helpers."""

from __future__ import annotations


def assemble_prompt(
    *,
    role: str,
    task: str,
    context_snippets: list[str],
    constraints: list[str],
    output_format: str,
) -> str:
    """Build a structured prompt from reusable components."""
    raise NotImplementedError("Prompt assembly is not implemented yet.")

