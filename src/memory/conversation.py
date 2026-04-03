"""Conversation history management."""

from __future__ import annotations


class ConversationBuffer:
    """Rolling message buffer with truncation hooks."""

    def __init__(self, max_messages: int = 12) -> None:
        self.max_messages = max_messages
        self.messages: list[dict[str, str]] = []

    def add(self, role: str, content: str) -> None:
        self.messages.append({"role": role, "content": content})
        self.messages = self.messages[-self.max_messages :]

