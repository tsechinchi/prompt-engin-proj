"""BM25 retrieval wrapper."""

from __future__ import annotations


class BM25Retriever:
    """Placeholder BM25 index for lexical retrieval."""

    def __init__(self) -> None:
        self._docs: list[str] = []

    def build(self, documents: list[str]) -> None:
        raise NotImplementedError("BM25 indexing is not implemented yet.")

    def query(self, text: str, *, top_k: int = 5) -> list[tuple[str, float]]:
        raise NotImplementedError("BM25 querying is not implemented yet.")

