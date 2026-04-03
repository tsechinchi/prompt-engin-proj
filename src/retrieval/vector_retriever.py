"""Embedding + FAISS retrieval wrapper."""

from __future__ import annotations

from functools import lru_cache
import re

import numpy as np


_TOKEN_RE = re.compile(r"\w+")


class VectorRetriever:
    """Semantic retriever backed by sentence-transformers and FAISS when available."""

    def __init__(self, *, model_name: str = "all-MiniLM-L6-v2") -> None:
        self._model_name = model_name
        self._docs: list[str] = []
        self._embeddings: np.ndarray | None = None
        self._index = None

    def build(self, documents: list[str]) -> None:
        """Build an embedding index over chunk texts."""

        self._docs = list(documents)
        if not self._docs:
            self._embeddings = np.zeros((0, 384), dtype=np.float32)
            self._index = None
            return

        embeddings = _encode_texts(self._docs, model_name=self._model_name)
        self._embeddings = _normalize_embeddings(embeddings.astype(np.float32, copy=False))
        self._index = _build_faiss_index(self._embeddings)

    def query(self, text: str, *, top_k: int = 5) -> list[tuple[str, float]]:
        """Return the top-k semantic matches and cosine-style scores."""

        if not self._docs or self._embeddings is None:
            raise ValueError("Vector index is empty. Call build() before query().")
        if top_k <= 0:
            return []

        query_embedding = _encode_texts([text], model_name=self._model_name)
        query_embedding = _normalize_embeddings(query_embedding.astype(np.float32, copy=False))
        if len(query_embedding) == 0:
            return []

        if self._index is not None:
            scores, indices = self._index.search(query_embedding, min(top_k, len(self._docs)))
            return [
                (self._docs[index], float(score))
                for score, index in zip(scores[0], indices[0], strict=False)
                if index >= 0
            ]

        similarities = self._embeddings @ query_embedding[0]
        ranked = sorted(enumerate(similarities.tolist()), key=lambda item: (-item[1], item[0]))
        return [
            (self._docs[index], score)
            for index, score in ranked[: min(top_k, len(ranked))]
        ]


def _encode_texts(texts: list[str], *, model_name: str) -> np.ndarray:
    model = _get_sentence_transformer_model(model_name)
    if model is not None:
        try:
            embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
            return np.asarray(embeddings, dtype=np.float32)
        except Exception:
            pass

    return _hash_embed_texts(texts)


@lru_cache(maxsize=4)
def _get_sentence_transformer_model(model_name: str):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        return None

    try:
        return SentenceTransformer(model_name)
    except Exception:
        return None


def _hash_embed_texts(texts: list[str], *, dimension: int = 384) -> np.ndarray:
    embeddings = np.zeros((len(texts), dimension), dtype=np.float32)

    for row, text in enumerate(texts):
        for token in _TOKEN_RE.findall(text.lower()):
            column = hash(token) % dimension
            embeddings[row, column] += 1.0

    return _normalize_embeddings(embeddings)


def _normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    if embeddings.size == 0:
        return embeddings

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return embeddings / norms


def _build_faiss_index(embeddings: np.ndarray):
    try:
        import faiss
    except ImportError:
        return None

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings.astype(np.float32, copy=False))
    return index
