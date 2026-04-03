"""Chunking utilities for retrieved context."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import TypedDict
import re

import tiktoken

from .loader import DocumentMetadata, LoadedDocument


class ChunkMetadata(DocumentMetadata, total=False):
    """Metadata for a chunked document."""

    chunk_id: str
    chunk_index: int
    token_count: int


class ChunkRecord(TypedDict):
    """Chunk text plus source metadata."""

    text: str
    metadata: ChunkMetadata


@dataclass(frozen=True)
class _ChunkUnit:
    text: str
    token_count: int


_BASIC_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def chunk_text(text: str, *, window_tokens: int = 200, stride_tokens: int = 50) -> list[str]:
    """Split text into overlapping chunks."""

    _validate_chunk_config(window_tokens, stride_tokens)
    normalized = _normalize_text(text)
    if not normalized:
        return []

    units = _build_chunk_units(normalized, window_tokens=window_tokens, stride_tokens=stride_tokens)
    return [chunk.text for chunk in _assemble_chunks(units, window_tokens=window_tokens, stride_tokens=stride_tokens)]


def chunk_documents(
    documents: list[LoadedDocument],
    *,
    window_tokens: int = 200,
    stride_tokens: int = 50,
) -> list[ChunkRecord]:
    """Chunk loaded documents while preserving source metadata."""

    _validate_chunk_config(window_tokens, stride_tokens)
    chunk_records: list[ChunkRecord] = []

    for document in documents:
        text = _normalize_text(document["text"])
        if not text:
            continue

        metadata = dict(document["metadata"])
        document_id = metadata.get("document_id", "document")
        chunk_units = _build_chunk_units(text, window_tokens=window_tokens, stride_tokens=stride_tokens)
        chunks = _assemble_chunks(chunk_units, window_tokens=window_tokens, stride_tokens=stride_tokens)

        for chunk_index, chunk in enumerate(chunks):
            chunk_metadata: ChunkMetadata = {
                **metadata,
                "chunk_id": f"{document_id}:{chunk_index}",
                "chunk_index": chunk_index,
                "token_count": chunk.token_count,
            }
            chunk_records.append({"text": chunk.text, "metadata": chunk_metadata})

    return chunk_records


def _validate_chunk_config(window_tokens: int, stride_tokens: int) -> None:
    if window_tokens <= 0:
        raise ValueError("window_tokens must be greater than zero.")
    if stride_tokens < 0:
        raise ValueError("stride_tokens must be greater than or equal to zero.")
    if stride_tokens >= window_tokens:
        raise ValueError("stride_tokens must be smaller than window_tokens.")


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _count_tokens(text: str) -> int:
    encoding = _get_encoding()
    if encoding is not None:
        return len(encoding.encode(text))
    return len(_basic_tokenize(text))


def _split_sentences(text: str) -> list[str]:
    normalized = _normalize_text(text)
    if not normalized:
        return []

    sentences = re.split(r"(?<=[.!?])\s+", normalized)
    return [sentence for sentence in sentences if sentence]


def _build_chunk_units(text: str, *, window_tokens: int, stride_tokens: int) -> list[_ChunkUnit]:
    units: list[_ChunkUnit] = []
    for sentence in _split_sentences(text):
        token_count = _count_tokens(sentence)
        if token_count <= window_tokens:
            units.append(_ChunkUnit(sentence, token_count))
            continue
        units.extend(_split_long_text(sentence, window_tokens=window_tokens, stride_tokens=stride_tokens))
    return units


def _split_long_text(text: str, *, window_tokens: int, stride_tokens: int) -> list[_ChunkUnit]:
    encoding = _get_encoding()
    if encoding is not None:
        token_ids = encoding.encode(text)
        decode = encoding.decode
    else:
        token_ids = _basic_tokenize(text)
        decode = _decode_basic_tokens

    step = max(window_tokens - stride_tokens, 1)
    chunks: list[_ChunkUnit] = []

    for start in range(0, len(token_ids), step):
        chunk_token_ids = token_ids[start : start + window_tokens]
        if not chunk_token_ids:
            continue
        chunk_text = decode(chunk_token_ids).strip()
        if not chunk_text:
            continue
        chunks.append(_ChunkUnit(chunk_text, len(chunk_token_ids)))
        if start + window_tokens >= len(token_ids):
            break

    return chunks


@lru_cache(maxsize=1)
def _get_encoding() -> tiktoken.Encoding | None:
    try:
        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return None


def _basic_tokenize(text: str) -> list[str]:
    return _BASIC_TOKEN_RE.findall(text)


def _decode_basic_tokens(tokens: list[str]) -> str:
    text = " ".join(tokens)
    return re.sub(r"\s+([.,!?;:])", r"\1", text)


def _assemble_chunks(
    units: list[_ChunkUnit],
    *,
    window_tokens: int,
    stride_tokens: int,
) -> list[_ChunkUnit]:
    if not units:
        return []

    chunks: list[_ChunkUnit] = []
    start = 0
    while start < len(units):
        end = start
        token_total = 0
        parts: list[str] = []

        while end < len(units):
            unit = units[end]
            if parts and token_total + unit.token_count > window_tokens:
                break
            parts.append(unit.text)
            token_total += unit.token_count
            end += 1

        chunk_text_value = " ".join(parts).strip()
        if chunk_text_value:
            chunks.append(_ChunkUnit(chunk_text_value, token_total))

        if end >= len(units):
            break

        if stride_tokens == 0:
            start = end
            continue

        overlap_tokens = 0
        next_start = end
        while next_start > start + 1 and overlap_tokens < stride_tokens:
            next_start -= 1
            overlap_tokens += units[next_start].token_count
        start = next_start

    return chunks
