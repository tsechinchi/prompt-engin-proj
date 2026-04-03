from __future__ import annotations

import unittest

from src.agent import approve_output, build_graph, review_output


class HITLTests(unittest.TestCase):
    def test_review_output_accepts_approve(self) -> None:
        answers = iter(["a", "looks good"])
        printed: list[str] = []

        decision = review_output(
            "Draft answer",
            input_func=lambda _prompt: next(answers),
            output_func=printed.append,
        )

        self.assertEqual(decision, {"action": "approve", "feedback": "looks good"})
        self.assertTrue(any("Generated Output" in line for line in printed))

    def test_approve_output_returns_true_only_for_approval(self) -> None:
        answers = iter(["approve", ""])
        approved = approve_output(
            "Final answer",
            input_func=lambda _prompt: next(answers),  # type: ignore[call-arg]
        )

        self.assertTrue(approved)


class GraphTests(unittest.TestCase):
    def test_graph_approves_output(self) -> None:
        graph = build_graph(
            generate_fn=lambda prompt, **_kwargs: f"generated from: {prompt.splitlines()[1]}",
            hitl_fn=lambda _text: {"action": "approve", "feedback": ""},
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
                "require_approval": True,
            }
        )

        self.assertEqual(result["status"], "approved")
        self.assertIn("generated from:", result["final_output"])
        self.assertTrue(result["context_snippets"])

    def test_graph_regenerates_once_then_approves(self) -> None:
        generated_prompts: list[str] = []
        decisions = iter(
            [
                {"action": "regenerate", "feedback": "Mention the date explicitly."},
                {"action": "approve", "feedback": ""},
            ]
        )

        def fake_generate(prompt: str, **_kwargs) -> str:
            generated_prompts.append(prompt)
            return f"draft {len(generated_prompts)}"

        graph = build_graph(
            generate_fn=fake_generate,
            hitl_fn=lambda _text: next(decisions),
        )

        result = graph.invoke(
            {
                "query": "Summarize the deadline.",
                "chunk_records": [
                    {"text": "The deadline is September 10.", "metadata": {"document_id": "doc-1"}},
                ],
                "constraints": [],
                "output_format": "A concise answer.",
                "require_approval": True,
                "max_regenerations": 1,
            }
        )

        self.assertEqual(result["status"], "approved")
        self.assertEqual(result["final_output"], "draft 2")
        self.assertEqual(len(generated_prompts), 2)
        self.assertIn("Reviewer feedback for this revision: Mention the date explicitly.", generated_prompts[-1])

    def test_graph_rejects_when_hitl_rejects(self) -> None:
        graph = build_graph(
            generate_fn=lambda _prompt, **_kwargs: "candidate answer",
            hitl_fn=lambda _text: {"action": "reject", "feedback": "Insufficient support."},
        )

        result = graph.invoke(
            {
                "query": "What are the scholarship rules?",
                "chunk_records": [
                    {"text": "Scholarship applications close on Friday.", "metadata": {"document_id": "doc-1"}},
                ],
                "require_approval": True,
            }
        )

        self.assertEqual(result["status"], "rejected")
        self.assertEqual(result["final_output"], "candidate answer")
        self.assertEqual(result["hitl_decision"]["feedback"], "Insufficient support.")


if __name__ == "__main__":
    unittest.main()
