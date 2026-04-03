"""LangGraph orchestration skeleton."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, NotRequired, TypedDict

from langgraph.graph import END, START, StateGraph

from src.generation import generate_raw
from src.prompt import assemble_prompt
from src.retrieval import BM25Retriever, VectorRetriever, fuse_scores

from .hitl import HITLDecision, review_output


class AgentState(TypedDict, total=False):
    """State carried through the orchestration graph."""

    query: str
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
    prompt: str
    context_snippets: list[str]
    bm25_hits: list[tuple[str, float]]
    vector_hits: list[tuple[str, float]]
    fused_hits: list[tuple[str, float]]
    aggregated_context: str
    generated_text: str
    final_output: str
    status: str
    hitl_decision: HITLDecision
    bm25_retriever: Any
    vector_retriever: Any
    generate_fn: Any
    hitl_fn: Any


def build_graph(
    *,
    generate_fn: Callable[..., str] = generate_raw,
    hitl_fn: Callable[[str], HITLDecision] = review_output,
):
    """Construct and compile the LangGraph workflow graph."""

    graph = StateGraph(AgentState)
    graph.add_node("retrieve", _make_retrieve_node())
    graph.add_node("aggregate", _aggregate_node)
    graph.add_node("assemble", _assemble_node)
    graph.add_node("generate", _make_generate_node(generate_fn))
    graph.add_node("postprocess", _postprocess_node)
    graph.add_node("hitl", _make_hitl_node(hitl_fn))
    graph.add_node("output", _output_node)

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "aggregate")
    graph.add_edge("aggregate", "assemble")
    graph.add_edge("assemble", "generate")
    graph.add_edge("generate", "postprocess")
    graph.add_edge("postprocess", "hitl")
    graph.add_conditional_edges(
        "hitl",
        _route_after_hitl,
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
    fused_hits = fuse_scores(
        state.get("bm25_hits", []),
        state.get("vector_hits", []),
        bm25_weight=state.get("bm25_weight", 0.4),
        vector_weight=state.get("vector_weight", 0.6),
    )
    top_k = state.get("top_k", 5)
    snippets = [text for text, _score in fused_hits[:top_k]]
    aggregated_context = "\n\n".join(snippets).strip()
    return {
        "fused_hits": fused_hits,
        "context_snippets": snippets,
        "aggregated_context": aggregated_context,
    }


def _assemble_node(state: AgentState) -> AgentState:
    constraints = list(state.get("constraints", []))
    review_feedback = state.get("review_feedback", "").strip()
    if review_feedback:
        constraints.append(f"Reviewer feedback for this revision: {review_feedback}")

    prompt = assemble_prompt(
        role=state.get("role", "Helpful HKBU study companion"),
        task=state["query"],
        context_snippets=state.get("context_snippets", []),
        constraints=constraints,
        output_format=state.get("output_format", "Provide a concise answer with citations."),
    )
    return {"prompt": prompt}


def _make_generate_node(default_generate_fn: Callable[..., str]):
    def generate_node(state: AgentState) -> AgentState:
        generate_fn = state.get("generate_fn", default_generate_fn)
        text = generate_fn(
            state["prompt"],
            model=state.get("model", "gemma3:4b"),
            temperature=state.get("temperature", 0.3),
            num_predict=state.get("num_predict", 200),
        )
        return {"generated_text": text}

    return generate_node


def _postprocess_node(state: AgentState) -> AgentState:
    return {"generated_text": state.get("generated_text", "").strip()}


def _make_hitl_node(default_hitl_fn: Callable[[str], HITLDecision]):
    def hitl_node(state: AgentState) -> AgentState:
        generated_text = state.get("generated_text", "")
        if not state.get("require_approval", True):
            return {
                "hitl_decision": {"action": "approve", "feedback": ""},
                "status": "approved",
                "final_output": generated_text,
            }

        hitl_fn = state.get("hitl_fn", default_hitl_fn)
        decision = hitl_fn(generated_text)
        regenerate_count = state.get("regenerate_count", 0)

        if decision["action"] == "regenerate":
            regenerate_count += 1
            if regenerate_count > state.get("max_regenerations", 1):
                return {
                    "hitl_decision": {"action": "reject", "feedback": "Maximum regenerations reached."},
                    "status": "rejected",
                    "final_output": generated_text,
                    "review_feedback": "Maximum regenerations reached.",
                    "regenerate_count": regenerate_count,
                }
            return {
                "hitl_decision": decision,
                "status": "regenerate",
                "review_feedback": decision.get("feedback", ""),
                "regenerate_count": regenerate_count,
            }

        status = "approved" if decision["action"] == "approve" else "rejected"
        return {
            "hitl_decision": decision,
            "status": status,
            "review_feedback": decision.get("feedback", ""),
            "final_output": generated_text,
            "regenerate_count": regenerate_count,
        }

    return hitl_node


def _output_node(state: AgentState) -> AgentState:
    return {
        "final_output": state.get("final_output", state.get("generated_text", "")),
        "status": state.get("status", "approved"),
    }


def _route_after_hitl(state: AgentState) -> str:
    if state.get("status") == "regenerate":
        return "assemble"
    return "output"
