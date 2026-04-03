"""Human-in-the-loop approval helpers."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal, TypedDict


class HITLDecision(TypedDict):
    """Structured review decision from the human loop."""

    action: Literal["approve", "reject", "regenerate"]
    feedback: str


def approve_output(
    text: str,
    *,
    input_func: Callable[[str], str] = input,
    output_func: Callable[[str], None] = print,
) -> bool:
    """Return True when the reviewed output is approved."""

    return review_output(text, input_func=input_func, output_func=output_func)["action"] == "approve"


def review_output(
    text: str,
    *,
    input_func: Callable[[str], str] = input,
    output_func: Callable[[str], None] = print,
) -> HITLDecision:
    """Run a small CLI approval loop for generated output."""

    output_func("Generated Output:\n")
    output_func(text)
    output_func("")

    prompt = "Approve, reject, or regenerate? [a/r/g]: "
    while True:
        choice = input_func(prompt).strip().lower()
        if choice in {"a", "approve"}:
            feedback = input_func("Optional approval note: ").strip()
            return {"action": "approve", "feedback": feedback}
        if choice in {"r", "reject"}:
            feedback = input_func("Reason for rejection: ").strip()
            return {"action": "reject", "feedback": feedback}
        if choice in {"g", "regenerate"}:
            feedback = input_func("What should change before regenerating? ").strip()
            return {"action": "regenerate", "feedback": feedback}
        output_func("Please enter 'a', 'r', or 'g'.")
