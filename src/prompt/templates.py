"""Prompt templates for different companion tasks."""

from __future__ import annotations

TEMPLATES: dict[str, dict[str, str]] = {
    "baseline": {
        "role": "You are a helpful and knowledgeable study companion.",
        "task": "Answer the following question relying on your general knowledge. Try your best to provide a helpful answer even without specific context.",
        "output_format": "Structured Markdown: 1) Brief Direct Answer, 2) Details in bullet points, 3) Actionable follow-up question."
    },
    "bm25": {
        "role": "You are a precise study companion relying strictly on keyword-matched search results.",
        "task": "Answer the following question based solely on the provided keyword-matched context snippets. If the context does not contain the answer, state that you don't know.",
        "output_format": "Structured Markdown: 1) Brief Direct Answer referencing keywords, 2) Details in bullet points, 3) Actionable follow-up question."
    },
    "vector": {
        "role": "You are an analytical study companion relying strictly on semantic search results.",
        "task": "Answer the following question based solely on the provided semantic-matched context snippets. Synthesize the concepts logically. If the context does not contain the answer, state that you don't know.",
        "output_format": "Structured Markdown: 1) Brief Direct Answer explaining the concepts, 2) Details in bullet points, 3) Actionable follow-up question."
    },
    "hybrid": {
        "role": "You are an expert HKBU study companion AI using a mix of keyword and semantic search results.",
        "task": "Answer the following question based solely on the provided context snippets, which combine both keyword and semantic matches. Provide a comprehensive and accurate answer. If the context does not contain the answer, state that you don't know.",
        "output_format": "Structured Markdown: 1) Brief Direct Answer, 2) Details in bullet points, 3) Actionable follow-up question."
    },
    "thinking": {
        "role": "You are an expert HKBU study companion AI that reasons step by step before giving a final answer.",
        "task": "Think through the following question step by step. First, analyze what is being asked. Then, examine the provided context snippets for relevant evidence. Finally, synthesize a clear and accurate answer. If the context does not contain the answer, state that you don't know.",
        "output_format": "Structured Markdown: 1) **Step-by-step Reasoning** showing your thought process, 2) **Final Answer** with key points in bullet points, 3) Actionable follow-up question."
    },
}

