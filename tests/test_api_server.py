from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from src.api.server import _sanitize_abstention_answer, app


class ApiServerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)

    def test_health_endpoint(self) -> None:
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")

    def test_ask_endpoint_uses_uploaded_evidence(self) -> None:
        payload = {
            "query": "what is this lecture about",
            "mode": "hybrid",
            "temperature": 0.3,
            "uploaded_docs": [
                {
                    "name": "lecture11.pdf",
                    "text": (
                        "Cloud computing lecture outcomes include understanding common cloud infrastructure mechanisms "
                        "and reducing operating costs with cloud optimization strategies."
                    ),
                }
            ],
        }

        response = self.client.post("/api/ask", json=payload)
        self.assertEqual(response.status_code, 200)

        body = response.json()
        self.assertIn("Summary", body["answer"])
        self.assertIn("Key points", body["answer"])
        self.assertNotEqual(body["status"], "abstained")
        self.assertTrue(body["citations"])
        self.assertIn("bleu", body["quality"])
        self.assertIn("total_tokens", body["tokens"])

    def test_ask_endpoint_accepts_lexical_and_semantic_modes(self) -> None:
        for mode in ["lexical", "semantic"]:
            payload = {
                "query": "what is this lecture about",
                "mode": mode,
                "temperature": 0.3,
                "uploaded_docs": [
                    {
                        "name": "lecture11.pdf",
                        "text": (
                            "Cloud computing lecture outcomes include understanding common cloud infrastructure mechanisms "
                            "and reducing operating costs with cloud optimization strategies."
                        ),
                    }
                ],
            }

            response = self.client.post("/api/ask", json=payload)
            self.assertEqual(response.status_code, 200)
            body = response.json()
            self.assertIn("Summary", body["answer"])
            self.assertTrue(body["citations"])
            self.assertIn("bleu", body["quality"])

    def test_ask_endpoint_accepts_baseline_mode(self) -> None:
        payload = {
            "query": "Hello world",
            "mode": "baseline",
            "temperature": 0.3,
            "uploaded_docs": [],
        }

        response = self.client.post("/api/ask", json=payload)
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertIsInstance(body["answer"], str)
        self.assertNotEqual(body["answer"], "")
        self.assertIn("bleu", body["quality"])
        self.assertIn("total_tokens", body["tokens"])

    def test_compare_endpoint_returns_summary(self) -> None:
        payload = {
            "query": "What is the key difference between the two answers?",
            "primary_mode": "hybrid",
            "compare_mode": "lexical",
            "primary_text": "This answer uses both semantic and lexical evidence.",
            "compare_text": "This answer focuses on exact keyword matches.",
        }

        response = self.client.post("/api/compare", json=payload)
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertIn("summary", body)
        self.assertIsInstance(body["summary"], str)

    def test_sanitizes_abstention_answer(self) -> None:
        raw_answer = (
            "The provider context cannot find in the uploaded document.\n"
            "Assessment Details: None.\n"
            "Actionable next step: Please ask a follow-up question."
        )
        self.assertEqual(
            _sanitize_abstention_answer(raw_answer),
            "The provider context cannot find in the uploaded document",
        )


if __name__ == "__main__":
    unittest.main()
