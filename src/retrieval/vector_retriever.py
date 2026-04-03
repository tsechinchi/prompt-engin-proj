"""Embedding + FAISS retrieval wrapper."""

from __future__ import annotations


class VectorRetriever:
    """Placeholder vector index for semantic retrieval."""

    def build(self, documents: list[str]) -> None:
        raise NotImplementedError("Vector indexing is not implemented yet.")

    def query(self, text: str, *, top_k: int = 5) -> list[tuple[str, float]]:
        raise NotImplementedError("Vector querying is not implemented yet.")

