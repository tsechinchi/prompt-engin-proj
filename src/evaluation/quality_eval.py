"""Evaluation helpers for answer quality."""

from __future__ import annotations


def score_answer(reference: str, prediction: str) -> dict[str, float]:
    """Return placeholder evaluation metrics."""
    raise NotImplementedError("Quality evaluation is not implemented yet.")

