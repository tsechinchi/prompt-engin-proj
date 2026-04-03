"""Score fusion between lexical and embedding retrieval."""

from __future__ import annotations


def fuse_scores(
    bm25_hits: list[tuple[str, float]],
    vector_hits: list[tuple[str, float]],
    *,
    bm25_weight: float = 0.4,
    vector_weight: float = 0.6,
) -> list[tuple[str, float]]:
    """Combine retrieval scores from multiple signals."""
    raise NotImplementedError("Hybrid ranking is not implemented yet.")

