from __future__ import annotations

import unittest
from unittest.mock import patch

from src.retrieval import BM25Retriever, VectorRetriever, fuse_scores


class BM25RetrieverTests(unittest.TestCase):
    def test_bm25_ranks_lexically_relevant_document_first(self) -> None:
        documents = [
            "campus timetable and exam schedule",
            "scholarship regulations and tuition fee rules",
            "student club orientation event",
        ]

        retriever = BM25Retriever()
        retriever.build(documents)
        hits = retriever.query("tuition scholarship", top_k=2)

        self.assertEqual(hits[0][0], "scholarship regulations and tuition fee rules")
        self.assertGreaterEqual(hits[0][1], hits[1][1])

    def test_bm25_query_requires_built_index(self) -> None:
        retriever = BM25Retriever()

        with self.assertRaises(ValueError):
            retriever.query("handbook")


class VectorRetrieverTests(unittest.TestCase):
    def test_vector_retriever_returns_semantically_closest_document(self) -> None:
        documents = [
            "apply for scholarship and tuition support",
            "basketball practice at the sports hall",
            "library opening hours during exams",
        ]

        with patch("src.retrieval.vector_retriever._get_sentence_transformer_model", return_value=None):
            with patch("src.retrieval.vector_retriever._build_faiss_index", return_value=None):
                retriever = VectorRetriever()
                retriever.build(documents)
                hits = retriever.query("tuition scholarship support", top_k=2)

        self.assertEqual(hits[0][0], "apply for scholarship and tuition support")
        self.assertGreaterEqual(hits[0][1], hits[1][1])

    def test_vector_query_requires_built_index(self) -> None:
        retriever = VectorRetriever()

        with self.assertRaises(ValueError):
            retriever.query("regulations")


class HybridRankerTests(unittest.TestCase):
    def test_fuse_scores_normalizes_and_combines_results(self) -> None:
        bm25_hits = [
            ("doc-a", 10.0),
            ("doc-b", 5.0),
        ]
        vector_hits = [
            ("doc-b", 0.9),
            ("doc-c", 0.3),
        ]

        fused = fuse_scores(bm25_hits, vector_hits)

        self.assertEqual(
            fused,
            [
                ("doc-b", 0.6),
                ("doc-a", 0.4),
                ("doc-c", 0.0),
            ],
        )

    def test_fuse_scores_requires_positive_weight(self) -> None:
        with self.assertRaises(ValueError):
            fuse_scores([], [], bm25_weight=0.0, vector_weight=0.0)


if __name__ == "__main__":
    unittest.main()
