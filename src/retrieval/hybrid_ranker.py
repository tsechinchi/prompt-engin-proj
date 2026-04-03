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

    if bm25_weight < 0 or vector_weight < 0:
        raise ValueError("Weights must be non-negative.")
    if bm25_weight == 0 and vector_weight == 0:
        raise ValueError("At least one retrieval weight must be positive.")

    bm25_normalized = _normalize_hits(bm25_hits)
    vector_normalized = _normalize_hits(vector_hits)

    combined_scores: dict[str, float] = {}
    first_seen_order: dict[str, int] = {}

    for index, (document, _score) in enumerate(bm25_hits):
        first_seen_order.setdefault(document, index)
        combined_scores.setdefault(document, 0.0)
        combined_scores[document] += bm25_weight * bm25_normalized.get(document, 0.0)

    offset = len(first_seen_order)
    for index, (document, _score) in enumerate(vector_hits):
        first_seen_order.setdefault(document, offset + index)
        combined_scores.setdefault(document, 0.0)
        combined_scores[document] += vector_weight * vector_normalized.get(document, 0.0)

    return sorted(
        combined_scores.items(),
        key=lambda item: (-item[1], first_seen_order[item[0]]),
    )


def _normalize_hits(hits: list[tuple[str, float]]) -> dict[str, float]:
    if not hits:
        return {}

    scores = [score for _document, score in hits]
    min_score = min(scores)
    max_score = max(scores)

    if max_score == min_score:
        normalized_value = 1.0 if max_score != 0 else 0.0
        return {document: normalized_value for document, _score in hits}

    scale = max_score - min_score
    return {
        document: (score - min_score) / scale
        for document, score in hits
    }
