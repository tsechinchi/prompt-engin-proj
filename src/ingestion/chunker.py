"""Chunking utilities for retrieved context."""

from __future__ import annotations


def chunk_text(text: str, *, window_tokens: int = 200, stride_tokens: int = 50) -> list[str]:
    """Split text into overlapping chunks.

    TODO: implement token-aware sliding windows and sentence-aware variants.
    """
    raise NotImplementedError("Chunking is not implemented yet.")

