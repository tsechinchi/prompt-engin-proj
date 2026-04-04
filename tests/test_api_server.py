from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from src.api.server import app


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
            "use_mock_generation": True,
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
        self.assertIn("Summary:", body["answer"])
        self.assertIn("Key points:", body["answer"])
        self.assertNotEqual(body["status"], "abstained")
        self.assertTrue(body["citations"])
        self.assertIn("bleu", body["quality"])
        self.assertIn("total_tokens", body["tokens"])


if __name__ == "__main__":
    unittest.main()
