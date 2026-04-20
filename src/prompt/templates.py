"""Prompt templates for different companion tasks."""

from __future__ import annotations

TEMPLATES: dict[str, dict[str, str]] = {
    "baseline": {
        "role": "You are a helpful and knowledgeable study companion.",
        "task": "Answer the following question relying on your general knowledge. Try your best to provide a helpful answer even without specific context.",
        "output_format": "Structured Markdown: 1) Brief Direct Answer, 2) Details in bullet points, 3) Actionable follow-up question."
    },
    "lexical": {
        "role": "You are an expert HKBU study companion AI using keyword retrieval evidence.",
        "task": "Answer the following question based solely on the provided context snippets retrieved by lexical keyword search. Be precise and faithful to the source texts.",
        "output_format": "Structured Markdown: 1) Brief Direct Answer, 2) Details in bullet points, 3) Actionable follow-up question."
    },
    "semantic": {
        "role": "You are an expert HKBU study companion AI using semantic embedding retrieval evidence.",
        "task": "Answer the following question based solely on the provided context snippets retrieved by semantic search. Be precise and faithful to the source texts.",
        "output_format": "Structured Markdown: 1) Brief Direct Answer, 2) Details in bullet points, 3) Actionable follow-up question."
    },
    "hybrid": {
        "role": "You are an expert HKBU study companion AI using a mix of keyword and semantic search results.",
        "task": "Answer the following question based solely on the provided context snippets, which combine both keyword and semantic matches. Provide a comprehensive and accurate answer. If the context does not contain the answer, state that you don't know.",
        "output_format": "Structured Markdown: 1) Brief Direct Answer, 2) Details in bullet points, 3) Actionable follow-up question."
    },
}

