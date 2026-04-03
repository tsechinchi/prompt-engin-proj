"""BM25 retrieval wrapper."""

from __future__ import annotations

from collections import Counter
from math import log
import re


_TOKEN_RE = re.compile(r"\w+")


class BM25Retriever:
    """Lexical retriever backed by rank-bm25 when available."""

    def __init__(self) -> None:
        self._docs: list[str] = []
        self._tokenized_docs: list[list[str]] = []
        self._doc_term_freqs: list[Counter[str]] = []
        self._doc_lengths: list[int] = []
        self._idf: dict[str, float] = {}
        self._avg_doc_length = 0.0
        self._bm25 = None

    def build(self, documents: list[str]) -> None:
        """Index chunk texts for lexical retrieval."""

        self._docs = list(documents)
        self._tokenized_docs = [_tokenize(document) for document in self._docs]
        self._bm25 = _build_rank_bm25(self._tokenized_docs)

        if self._bm25 is not None:
            self._doc_term_freqs = []
            self._doc_lengths = []
            self._idf = {}
            self._avg_doc_length = 0.0
            return

        self._doc_term_freqs = [Counter(tokens) for tokens in self._tokenized_docs]
        self._doc_lengths = [len(tokens) for tokens in self._tokenized_docs]
        total_tokens = sum(self._doc_lengths)
        self._avg_doc_length = total_tokens / len(self._doc_lengths) if self._doc_lengths else 0.0

        document_frequency: Counter[str] = Counter()
        for tokens in self._tokenized_docs:
            document_frequency.update(set(tokens))

        corpus_size = len(self._tokenized_docs)
        self._idf = {
            token: log(1 + (corpus_size - frequency + 0.5) / (frequency + 0.5))
            for token, frequency in document_frequency.items()
        }

    def query(self, text: str, *, top_k: int = 5) -> list[tuple[str, float]]:
        """Return the top-k lexical matches and scores."""

        if not self._docs:
            raise ValueError("BM25 index is empty. Call build() before query().")
        if top_k <= 0:
            return []

        query_tokens = _tokenize(text)
        if not query_tokens:
            return []

        if self._bm25 is not None:
            scores = [float(score) for score in self._bm25.get_scores(query_tokens)]
        else:
            scores = self._score_with_fallback(query_tokens)

        ranked = sorted(enumerate(scores), key=lambda item: (-item[1], item[0]))
        return [
            (self._docs[index], score)
            for index, score in ranked[: min(top_k, len(ranked))]
        ]

    def _score_with_fallback(self, query_tokens: list[str]) -> list[float]:
        if not self._docs:
            return []

        k1 = 1.5
        b = 0.75
        scores: list[float] = []
        avg_doc_length = self._avg_doc_length or 1.0

        for term_freqs, doc_length in zip(self._doc_term_freqs, self._doc_lengths, strict=False):
            score = 0.0
            for token in query_tokens:
                term_frequency = term_freqs.get(token, 0)
                if term_frequency == 0:
                    continue
                idf = self._idf.get(token, 0.0)
                denominator = term_frequency + k1 * (1 - b + b * doc_length / avg_doc_length)
                score += idf * (term_frequency * (k1 + 1)) / denominator
            scores.append(score)
        return scores


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _build_rank_bm25(tokenized_docs: list[list[str]]):
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        return None

    return BM25Okapi(tokenized_docs)
