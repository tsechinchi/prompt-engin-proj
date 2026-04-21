from __future__ import annotations

import unittest

from src.agent import build_graph


class _StaticRetriever:
    def __init__(self, hits: list[tuple[str, float]]) -> None:
        self._hits = hits

    def build(self, _documents: list[str]) -> None:
        return

    def query(self, _query: str, *, top_k: int = 5) -> list[tuple[str, float]]:
        return self._hits[:top_k]


class GraphTests(unittest.TestCase):
    def test_graph_generates_output(self) -> None:
        graph = build_graph(
            generate_fn=lambda prompt, **_kwargs: f"generated from: {prompt.splitlines()[1]}",
        )

        result = graph.invoke(
            {
                "query": "When is the add/drop deadline?",
                "chunk_records": [
                    {"text": "The add/drop deadline is September 10.", "metadata": {"document_id": "doc-1"}},
                    {"text": "Tuition payment is due in October.", "metadata": {"document_id": "doc-2"}},
                ],
                "role": "Helpful HKBU study companion",
                "constraints": ["Use the retrieved context only."],
                "output_format": "One short paragraph.",
                "model": "gemma3:4b",
            }
        )

        self.assertEqual(result["status"], "approved")
        self.assertIn("generated from:", result["final_output"])
        self.assertTrue(result["context_snippets"])

    def test_graph_abstains_on_retrieval_mismatch(self) -> None:
        generated_prompts: list[str] = []

        def fake_generate(_prompt: str, **_kwargs) -> str:
            generated_prompts.append(_prompt)
            return "should not be generated"

        graph = build_graph(
            generate_fn=fake_generate,
        )

        result = graph.invoke(
            {
                "query": "What are campus parking permit rules for motorcycles?",
                "chunk_records": [
                    {"text": "The add/drop deadline is September 10.", "metadata": {"document_id": "doc-1"}},
                    {"text": "Tuition payment is due in October.", "metadata": {"document_id": "doc-2"}},
                ],
            }
        )

        self.assertEqual(result["status"], "abstained")
        self.assertTrue(result.get("retrieval_mismatch"))
        self.assertIn("the provider context cannot find in the uploaded document", result["final_output"].lower())
        self.assertEqual(len(generated_prompts), 2)
        self.assertIn("Retrieval context appears insufficient", generated_prompts[1])

    def test_graph_does_not_false_abstain_for_course_code_query(self) -> None:
        graph = build_graph(
            generate_fn=lambda _prompt, **_kwargs: "The CS101 exam date is Dec 10.",
        )

        retriever_hits = [("CS101 exam date is Dec 10 and assignment deadline is Nov 1.", 1.0)]
        result = graph.invoke(
            {
                "query": "CS101?",
                "chunk_records": [
                    {"text": "placeholder", "metadata": {"document_id": "doc-1"}},
                ],
                "bm25_retriever": _StaticRetriever(retriever_hits),
                "vector_retriever": _StaticRetriever(retriever_hits),
                "top_k": 1,
            }
        )

        self.assertEqual(result["status"], "approved")
        self.assertFalse(result.get("retrieval_mismatch", False))


if __name__ == "__main__":
    unittest.main()
