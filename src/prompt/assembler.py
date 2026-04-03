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

    sections = [
        _format_section("Role", role.strip()),
        _format_section("Task", task.strip()),
        _format_context_section(context_snippets),
        _format_list_section("Constraints", constraints, bullet="-"),
        _format_section("Output Format", output_format.strip()),
    ]

    return "\n\n".join(section for section in sections if section).strip()


def _format_section(title: str, body: str) -> str:
    body_text = body or "None provided."
    return f"{title}:\n{body_text}"


def _format_context_section(context_snippets: list[str]) -> str:
    if not context_snippets:
        return "Context Snippets:\nNone provided."

    lines = [f"{index}. {snippet.strip()}" for index, snippet in enumerate(context_snippets, start=1) if snippet.strip()]
    if not lines:
        return "Context Snippets:\nNone provided."
    return "Context Snippets:\n" + "\n".join(lines)


def _format_list_section(title: str, items: list[str], *, bullet: str) -> str:
    cleaned = [item.strip() for item in items if item.strip()]
    if not cleaned:
        return f"{title}:\nNone provided."
    return f"{title}:\n" + "\n".join(f"{bullet} {item}" for item in cleaned)
