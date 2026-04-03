from __future__ import annotations

import unittest
from unittest.mock import patch

from src.generation import generate_raw
from src.memory import ConversationBuffer
from src.prompt import assemble_prompt


class PromptAssemblerTests(unittest.TestCase):
    def test_assemble_prompt_formats_all_sections(self) -> None:
        prompt = assemble_prompt(
            role="Helpful HKBU study companion",
            task="Answer the student's question with citations.",
            context_snippets=["Snippet one", "Snippet two"],
            constraints=["Use only the provided context", "Be concise"],
            output_format="A short answer followed by bullet citations.",
        )

        self.assertIn("Role:\nHelpful HKBU study companion", prompt)
        self.assertIn("Task:\nAnswer the student's question with citations.", prompt)
        self.assertIn("Context Snippets:\n1. Snippet one\n2. Snippet two", prompt)
        self.assertIn("Constraints:\n- Use only the provided context\n- Be concise", prompt)
        self.assertIn("Output Format:\nA short answer followed by bullet citations.", prompt)


class OllamaClientTests(unittest.TestCase):
    def test_generate_raw_matches_notebook_call_shape(self) -> None:
        with patch("ollama.generate", return_value={"response": "Paris"}) as mocked_generate:
            result = generate_raw(
                "Question: What is the capital of France?\n\nAnswer: ",
                model="gemma3:4b",
                temperature=0.5,
                num_predict=180,
            )

        self.assertEqual(result, "Paris")
        mocked_generate.assert_called_once_with(
            model="gemma3:4b",
            prompt="Question: What is the capital of France?\n\nAnswer: ",
            stream=False,
            raw=True,
            options={
                "num_predict": 180,
                "temperature": 0.5,
            },
        )


class ConversationBufferTests(unittest.TestCase):
    def test_conversation_buffer_truncates_to_max_messages(self) -> None:
        buffer = ConversationBuffer(max_messages=2, max_tokens=1000)

        buffer.add("user", "first")
        buffer.add("assistant", "second")
        buffer.add("user", "third")

        self.assertEqual(
            buffer.messages,
            [
                {"role": "assistant", "content": "second"},
                {"role": "user", "content": "third"},
            ],
        )

    def test_conversation_buffer_truncates_oldest_turns_by_token_budget(self) -> None:
        buffer = ConversationBuffer(max_messages=5, max_tokens=20)

        buffer.add("system", "You are a helpful assistant.")
        buffer.add("user", "Please summarize the student handbook in detail.")
        buffer.add("assistant", "Sure, I can help with that.")
        buffer.add("user", "Also include the scholarship policy.")

        self.assertEqual(buffer.messages[0]["role"], "system")
        self.assertEqual(buffer.messages[-1]["content"], "Also include the scholarship policy.")
        self.assertLessEqual(len(buffer.messages), 2)


if __name__ == "__main__":
    unittest.main()
