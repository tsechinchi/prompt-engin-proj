"""Conversation history management."""

from __future__ import annotations

from functools import lru_cache
from typing import TypedDict
import re

import tiktoken


class ConversationMessage(TypedDict):
    """Role/content pair stored in the rolling buffer."""

    role: str
    content: str


class ConversationBuffer:
    """Rolling message buffer with count and token-budget truncation."""

    def __init__(self, max_messages: int = 12, *, max_tokens: int = 1200) -> None:
        if max_messages <= 0:
            raise ValueError("max_messages must be greater than zero.")
        if max_tokens <= 0:
            raise ValueError("max_tokens must be greater than zero.")
        self.max_messages = max_messages
        self.max_tokens = max_tokens
        self.messages: list[ConversationMessage] = []

    def add(self, role: str, content: str) -> None:
        """Append a message and trim the oldest turns to fit the budget."""

        self.messages.append({"role": role, "content": content})
        self._truncate()

    def extend(self, messages: list[ConversationMessage]) -> None:
        """Append multiple messages in order."""

        for message in messages:
            self.add(message["role"], message["content"])

    def clear(self) -> None:
        """Remove all stored messages."""

        self.messages.clear()

    def token_count(self) -> int:
        """Return the approximate token usage of the stored history."""

        return sum(_count_message_tokens(message) for message in self.messages)

    def _truncate(self) -> None:
        while len(self.messages) > self.max_messages:
            self.messages.pop(self._oldest_truncatable_index())

        while len(self.messages) > 1 and self.token_count() > self.max_tokens:
            self.messages.pop(self._oldest_truncatable_index())

    def _oldest_truncatable_index(self) -> int:
        for index, message in enumerate(self.messages):
            if message["role"] != "system":
                return index
        return 0


_BASIC_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def _count_message_tokens(message: ConversationMessage) -> int:
    text = f'{message["role"]}: {message["content"]}'
    encoding = _get_encoding()
    if encoding is not None:
        return len(encoding.encode(text))
    return len(_BASIC_TOKEN_RE.findall(text))


@lru_cache(maxsize=1)
def _get_encoding() -> tiktoken.Encoding | None:
    try:
        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return None
