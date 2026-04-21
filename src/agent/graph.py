"""LangGraph orchestration skeleton."""

from __future__ import annotations

from collections.abc import Callable
import re
from typing import Any, NotRequired, TypedDict

from langgraph.graph import END, START, StateGraph

from src.generation import generate_raw
from src.prompt import TEMPLATES, assemble_prompt
from src.retrieval import BM25Retriever, VectorRetriever, fuse_scores


class AgentState(TypedDict, total=False):
    """State carried through the orchestration graph."""

    query: str
    mode: str
    chunk_records: list[dict[str, Any]]
    role: str
    constraints: list[str]
    output_format: str
    model: str
    top_k: int
    temperature: float
    num_predict: int
    require_approval: bool
    bm25_weight: float
    vector_weight: float
    review_feedback: str
    max_regenerations: int
    regenerate_count: int
    max_retrieval_retries: int
    retrieval_retry_count: int
    abstain_on_mismatch: bool
    abstention_message: str
    min_fused_score: float
    min_query_term_overlap: int
    prompt: str
    context_snippets: list[str]
    history: list[dict[str, str]]
    bm25_hits: list[tuple[str, float]]
    vector_hits: list[tuple[str, float]]
    fused_hits: list[tuple[str, float]]
    aggregated_context: str
    generated_text: str
    final_output: str
    status: str
    retrieval_mismatch: bool
    retrieval_mismatch_reason: str
    bm25_retriever: Any
    vector_retriever: Any
    generate_fn: Any


def build_graph(
    *,
    generate_fn: Callable[..., str] = generate_raw,
):
    """Construct and compile the LangGraph workflow graph."""

    graph = StateGraph(AgentState)
    graph.add_node("retrieve", _make_retrieve_node())
    graph.add_node("aggregate", _aggregate_node)
    graph.add_node("assess_retrieval", _assess_retrieval_node)
    graph.add_node("assemble", _assemble_node)
    graph.add_node("generate", _make_generate_node(generate_fn))
    graph.add_node("postprocess", _postprocess_node)
    graph.add_node("retrieval_guard", _retrieval_guard_node)
    graph.add_node("output", _output_node)

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "aggregate")
    graph.add_edge("aggregate", "assess_retrieval")
    graph.add_edge("assess_retrieval", "assemble")
    graph.add_edge("assemble", "generate")
    graph.add_edge("generate", "postprocess")
    graph.add_edge("postprocess", "retrieval_guard")
    graph.add_conditional_edges(
        "retrieval_guard",
        _route_after_retrieval_guard,
        {
            "assemble": "assemble",
            "output": "output",
        },
    )
    graph.add_edge("output", END)
    return graph.compile()


def _make_retrieve_node():
    def retrieve_node(state: AgentState) -> AgentState:
        query = state["query"]
        chunk_records = state.get("chunk_records", [])
        documents = [record["text"] for record in chunk_records]
        top_k = state.get("top_k", 5)

        bm25_retriever = state.get("bm25_retriever")
        if bm25_retriever is None:
            bm25_retriever = BM25Retriever()
            bm25_retriever.build(documents)

        vector_retriever = state.get("vector_retriever")
        if vector_retriever is None:
            vector_retriever = VectorRetriever()
            vector_retriever.build(documents)

        bm25_hits = bm25_retriever.query(query, top_k=top_k) if documents else []
        vector_hits = vector_retriever.query(query, top_k=top_k) if documents else []

        return {
            "bm25_retriever": bm25_retriever,
            "vector_retriever": vector_retriever,
            "bm25_hits": bm25_hits,
            "vector_hits": vector_hits,
        }

    return retrieve_node


def _aggregate_node(state: AgentState) -> AgentState:
    bm25_weight = state.get("bm25_weight", 0.4)
    vector_weight = state.get("vector_weight", 0.6)

    if bm25_weight == 0 and vector_weight == 0:
        fused_hits: list[tuple[str, float]] = []
    else:
        fused_hits = fuse_scores(
            state.get("bm25_hits", []),
            state.get("vector_hits", []),
            bm25_weight=bm25_weight,
            vector_weight=vector_weight,
        )

    top_k = state.get("top_k", 5)
    snippets = [text for text, _score in fused_hits[:top_k]]
    aggregated_context = "\n\n".join(snippets).strip()
    return {
        "fused_hits": fused_hits,
        "context_snippets": snippets,
        "aggregated_context": aggregated_context,
    }


def _assess_retrieval_node(state: AgentState) -> AgentState:
    if not state.get("abstain_on_mismatch", True):
        return {"retrieval_mismatch": False, "retrieval_mismatch_reason": ""}

    fused_hits = state.get("fused_hits", [])
    top_k = state.get("top_k", 5)
    min_fused_score = state.get("min_fused_score", 0.15)
    min_overlap = state.get("min_query_term_overlap", 1)

    snippets = [text for text, _score in fused_hits[:top_k]]
    if not snippets:
        return {
            "retrieval_mismatch": True,
            "retrieval_mismatch_reason": "No retrieved snippets were available.",
        }

    top_score = fused_hits[0][1]
    query_terms = _query_terms(state.get("query", ""))
    best_overlap = max((_query_overlap_count(query_terms, snippet) for snippet in snippets), default=0)

    reasons: list[str] = []
    if top_score < min_fused_score:
        reasons.append(f"Top fused score {top_score:.3f} is below threshold {min_fused_score:.3f}.")
    if query_terms and best_overlap < min_overlap:
        reasons.append(f"Best query-term overlap {best_overlap} is below threshold {min_overlap}.")

    if reasons:
        return {
            "retrieval_mismatch": True,
            "retrieval_mismatch_reason": " ".join(reasons),
        }

    return {"retrieval_mismatch": False, "retrieval_mismatch_reason": ""}


def _assemble_node(state: AgentState) -> AgentState:
    constraints = list(state.get("constraints", []))
    
    # Prompt engineering constraints for structured and concise output
    constraints.extend([
        "You MUST format your entire response in valid Markdown. Do not use plain text.",
        "Do NOT include any <think> reasoning blocks, conversational fillers, or intro/outro remarks. Output only the final meaningful response.",
        "Do NOT repeat or echo the user's question in your response.",
        "Use actual Markdown headers (e.g., ### Summary) to separate sections.",
        "Use formatting like bolding and bulleted lists to make the output user-friendly.",
        "Format the response into distinct paragraphs with blank lines between them.",
        "Base your answer strictly on the provided context if available.",
        "If the user is asking something not related to the RAG document, output ONLY 'The provider context cannot find in the uploaded document'. Do not include any other text, headings, bullet lists, or follow-up suggestions.",
        "When referring to names of people, ALWAYS use their full names exactly as they appear in the provided context snippets (e.g., 'Dr. Shichao Ma' instead of just 'Dr. Shichao').",
        f"You MUST directly answer the user's core question: '{state['query']}'. Do not provide a generic summary of the text.",
    ])

    if not state.get("retrieval_mismatch", False):
        constraints.append(
            "At the very end of your response, you MUST provide an 'Actionable next step' with a specific follow-up question the user could ask."
        )
    
    review_feedback = state.get("review_feedback", "").strip()
    if review_feedback:
        constraints.append(f"Reviewer feedback for this revision: {review_feedback}")

    mode = state.get("mode", "hybrid")
    template = TEMPLATES.get(mode, TEMPLATES["hybrid"])

    prompt = assemble_prompt(
        role=state.get("role", template.get("role", "Expert HKBU study companion AI")),
        task=state.get("task", f"{template.get('task', 'Answer the following question based solely on the provided context.')} Question: {state['query']}"),
        context_snippets=state.get("context_snippets", []),
        constraints=constraints,
        output_format=state.get("output_format", template.get("output_format", "Structured Markdown: 1) Brief Direct Answer, 2) Details in bullet points, 3) Actionable follow-up question.")),
        conversation_history=state.get("history", []),
    )
    return {"prompt": prompt}


def _make_generate_node(default_generate_fn: Callable[..., str]):
    def generate_node(state: AgentState) -> AgentState:
        generate_fn = state.get("generate_fn", default_generate_fn)
        text = generate_fn(
            state["prompt"],
            model=state.get("model", "gemma3:4b"),
            temperature=state.get("temperature", 0.3),
            num_predict=state.get("num_predict", 800),
        )
        return {"generated_text": text}

    return generate_node


def _postprocess_node(state: AgentState) -> AgentState:
    return {"generated_text": state.get("generated_text", "").strip()}


def _retrieval_guard_node(state: AgentState) -> AgentState:
    if not state.get("retrieval_mismatch", False):
        return {"status": state.get("status", "approved")}

    retry_count = state.get("retrieval_retry_count", 0)
    max_retries = state.get("max_retrieval_retries", 1)
    if retry_count < max_retries:
        reason = state.get("retrieval_mismatch_reason", "Retrieval context appears insufficient.")
        return {
            "status": "retrieval_retry",
            "retrieval_retry_count": retry_count + 1,
            "review_feedback": (
                "Retrieval context appears insufficient or weakly aligned. "
                "Retry once with a conservative answer grounded in retrieved snippets only. "
                f"Mismatch reason: {reason}"
            ),
        }

    reason = state.get("retrieval_mismatch_reason", "Retrieval context appears insufficient.")
    return _abstain_for_retrieval_mismatch(state, reason)


def _output_node(state: AgentState) -> AgentState:
    return {
        "final_output": state.get("final_output", state.get("generated_text", "")),
        "status": state.get("status", "approved"),
    }


def _route_after_retrieval_guard(state: AgentState) -> str:
    if state.get("status") == "retrieval_retry":
        return "assemble"
    return "output"


def _abstain_for_retrieval_mismatch(state: AgentState, reason: str) -> AgentState:
    message = state.get(
        "abstention_message",
        "The provider context cannot find in the uploaded document",
    )
    return {
        "retrieval_mismatch": True,
        "retrieval_mismatch_reason": reason,
        "status": "abstained",
        "final_output": message,
    }


def _query_terms(query: str) -> set[str]:
    stop_words = {
        "the",
        "and",
        "for",
        "with",
        "from",
        "what",
        "when",
        "where",
        "who",
        "how",
        "are",
        "is",
        "can",
        "you",
        "your",
        "about",
        "between",
        "into",
        "have",
        "has",
        "this",
        "that",
        "course",
    }
    tokens = re.findall(r"[a-zA-Z0-9]+", query.lower())
    return {token for token in tokens if len(token) > 2 and token not in stop_words}


def _query_overlap_count(query_terms: set[str], snippet: str) -> int:
    if not query_terms:
        return 0
    snippet_terms = set(re.findall(r"[a-zA-Z0-9]+", snippet.lower()))
    return len(query_terms.intersection(snippet_terms))
