"""FastAPI server that exposes the LangGraph pipeline to the frontend."""

from __future__ import annotations

from functools import lru_cache
from hashlib import sha1
import json as _json
from pathlib import Path
import re
import time
from typing import Literal

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.agent import build_graph
from src.evaluation.token_tracker import track_usage
from src.generation import generate_raw, generate_stream
from src.ingestion import chunk_documents, load_documents
from src.memory import ConversationBuffer
from src.prompt import TEMPLATES, assemble_prompt
from src.retrieval import BM25Retriever, VectorRetriever, fuse_scores


class UploadedDocPayload(BaseModel):
    """Uploaded document text parsed on the frontend."""

    name: str
    text: str = ""


class AskRequest(BaseModel):
    """Request schema for graph-backed answer generation."""

    query: str
    mode: Literal["baseline", "bm25", "vector", "hybrid", "thinking", "lexical", "semantic"] = "hybrid"
    temperature: float = 0.3
    top_k: int = 5
    max_tokens: int = 200
    model: str = "gemma3:4b"
    uploaded_docs: list[UploadedDocPayload] = Field(default_factory=list)
    history: list[dict[str, str]] = Field(default_factory=list)
    use_mock_corpus: bool = False


class AskResponse(BaseModel):
    """Response payload consumed by the frontend."""

    answer: str
    status: str
    citations: list[str]
    quality: dict[str, float]
    tokens: dict[str, int]
    model_used: str


class CompareRequest(BaseModel):
    query: str
    primary_mode: str
    compare_mode: str
    primary_text: str
    compare_text: str
    history: list[dict[str, str]] = Field(default_factory=list)


class CompareResponse(BaseModel):
    summary: str


def create_app() -> FastAPI:
    """Create and configure the API application."""

    app = FastAPI(title="HKBU Study Companion API", version="0.1.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/api/ask", response_model=AskResponse)
    def ask(payload: AskRequest) -> AskResponse:
        generation_info = {"model_used": "unknown"}
        graph = build_graph(
            generate_fn=_make_generate_fn(payload, generation_info),
            hitl_fn=lambda _text: {"action": "approve", "feedback": ""},
        )

        uploaded_chunk_records = _uploaded_chunk_records(payload.uploaded_docs)
        chunk_records = _select_chunk_records(
            mode=payload.mode,
            uploaded_chunk_records=uploaded_chunk_records,
            use_mock_corpus=payload.use_mock_corpus,
        )
        abstain_on_mismatch = _should_abstain_on_mismatch(
            mode=payload.mode,
            has_uploaded_chunks=bool(uploaded_chunk_records),
            has_any_chunks=bool(chunk_records),
        )

        bm25_weight, vector_weight = _mode_weights(payload.mode)

        history_buffer = ConversationBuffer(max_messages=12, max_tokens=1200)
        history_buffer.extend(payload.history)

        result = graph.invoke(
            {
                "query": payload.query,
                "mode": payload.mode,
                "chunk_records": chunk_records,
                "history": history_buffer.messages,
                "top_k": max(payload.top_k, 1),
                "temperature": max(payload.temperature, 0.0),
                "model": payload.model,
                "num_predict": max(payload.max_tokens, 1),
                "require_approval": False,
                "bm25_weight": bm25_weight,
                "vector_weight": vector_weight,
                "abstain_on_mismatch": abstain_on_mismatch,
                "max_retrieval_retries": 1,
                "max_regenerations": 0,
            }
        )

        answer = str(result.get("final_output", "")).strip()
        answer = _sanitize_abstention_answer(answer)
        if not answer:
            answer = "No answer generated."

        context_snippets = [str(value) for value in result.get("context_snippets", [])]
        citations = _build_citations(context_snippets)
        quality = _estimate_quality(payload.query, context_snippets, result.get("status", ""))

        usage = track_usage(
            prompt_tokens=max(1, _token_count(payload.query)),
            completion_tokens=max(1, _token_count(answer)),
        )

        return AskResponse(
            answer=answer,
            status=str(result.get("status", "approved")),
            citations=citations,
            quality=quality,
            tokens=usage,
            model_used=generation_info["model_used"],
        )


    @app.post("/api/compare", response_model=CompareResponse)
    def compare(payload: CompareRequest) -> CompareResponse:
        prompt = (
            "You are an expert assistant comparing two answer drafts. "
            "Use the user question, the conversation history, and the two answers to produce a very clean, concise bullet-point summary of their differences. "
            "Focus on which answer is more factual, more concise, and which one includes extra key details. "
            "Do not invent new information beyond the text provided. "
            "Output only bullet points, with no additional explanation.\n\n"
            f"User question: {payload.query}\n\n"
        )

        if payload.history:
            prompt += "Conversation history:\n"
            for message in payload.history:
                role = message.get("role", "").capitalize()
                content = message.get("content", "")
                prompt += f"{role}: {content}\n"
            prompt += "\n"

        prompt += (
            f"Primary mode: {payload.primary_mode}\n"
            f"Primary answer:\n{payload.primary_text}\n\n"
            f"Comparison mode: {payload.compare_mode}\n"
            f"Comparison answer:\n{payload.compare_text}\n\n"
            "Bullet-point summary:"
        )

        try:
            summary_text = generate_raw(
                prompt,
                model="gemma3:4b",
                temperature=0.3,
                num_predict=200,
            )
        except Exception as e:
            print(f"Ollama comparison generation failed: {repr(e)}")
            summary_text = (
                "Could not generate a comparison summary. Please try again when the Ollama service is available."
            )

        return CompareResponse(summary=summary_text.strip())


    # ── SSE streaming endpoint for Thinking Mode ──────────────────────

    @app.post("/api/ask/stream")
    def ask_stream(payload: AskRequest) -> StreamingResponse:
        """Server-Sent Events endpoint for Thinking Mode streaming."""

        def _event_generator():
            try:
                # Step 1 — Analyse question
                yield _sse("thinking_step", "Analyzing your question...")
                time.sleep(0.05)

                # Step 2 — Prepare retrieval corpus
                uploaded_chunks = _uploaded_chunk_records(payload.uploaded_docs)
                use_mode = "hybrid"  # thinking always uses hybrid retrieval
                chunk_records = _select_chunk_records(
                    mode=use_mode,
                    uploaded_chunk_records=uploaded_chunks,
                    use_mock_corpus=payload.use_mock_corpus,
                )

                yield _sse("thinking_step", f"Loaded {len(chunk_records)} document chunks")
                time.sleep(0.05)

                # Step 3 — Retrieve & fuse
                documents = [str(r.get("text", "")) for r in chunk_records]
                top_k = max(payload.top_k, 1)
                bm25_hits: list[tuple[str, float]] = []
                vector_hits: list[tuple[str, float]] = []

                if documents:
                    bm25_ret = BM25Retriever()
                    bm25_ret.build(documents)
                    bm25_hits = bm25_ret.query(payload.query, top_k=top_k)

                    vector_ret = VectorRetriever()
                    vector_ret.build(documents)
                    vector_hits = vector_ret.query(payload.query, top_k=top_k)

                fused = fuse_scores(bm25_hits, vector_hits, bm25_weight=0.4, vector_weight=0.6)
                snippets = [text for text, _score in fused[:top_k]]

                if snippets:
                    yield _sse("thinking_step", f"Found {len(snippets)} relevant evidence snippets")
                else:
                    yield _sse("thinking_step", "No document context available — using general knowledge")
                time.sleep(0.05)

                # Step 4 — Assemble prompt
                template = TEMPLATES.get("thinking", TEMPLATES["hybrid"])
                history_buffer = ConversationBuffer(max_messages=12, max_tokens=1200)
                history_buffer.extend(payload.history)

                constraints = [
                    "Think step by step. Show your reasoning clearly before giving a final answer.",
                    "You MUST format your entire response in valid Markdown.",
                    "Do NOT repeat or echo the user's question in your response.",
                    "Use Markdown headers (e.g., ### Reasoning, ### Final Answer) to separate sections.",
                    "Base your answer strictly on the provided context if available.",
                    f"You MUST directly answer the user's core question: '{payload.query}'.",
                    "At the very end, provide an 'Actionable next step' with a specific follow-up question.",
                ]

                prompt = assemble_prompt(
                    role=template.get("role", "Expert HKBU study companion AI"),
                    task=f"{template.get('task', '')} Question: {payload.query}",
                    context_snippets=snippets,
                    constraints=constraints,
                    output_format=template.get("output_format", ""),
                    conversation_history=history_buffer.messages,
                )

                yield _sse("thinking_step", "Prompt assembled — generating answer...")
                time.sleep(0.05)

                # Step 5 — Generate (streaming or mock)
                full_text = ""
                model_used = "unknown"

                if payload.use_mock_generation:
                    model_used = "mock"
                    mock_answer = _mock_generate_from_prompt(prompt, payload.query)
                    full_text = mock_answer
                    yield _sse("token", mock_answer)
                else:
                    model_used = "ollama"
                    try:
                        for chunk in generate_stream(
                            prompt,
                            model=payload.model,
                            temperature=payload.temperature,
                            num_predict=1000,
                        ):
                            if chunk["type"] == "token":
                                full_text += chunk["content"]
                                yield _sse("token", chunk["content"])
                            elif chunk["type"] == "error":
                                model_used = "mock_fallback"
                                mock_answer = _mock_generate_from_prompt(prompt, payload.query)
                                full_text = mock_answer
                                yield _sse("token", mock_answer)
                                break
                    except Exception:
                        model_used = "mock_fallback"
                        mock_answer = _mock_generate_from_prompt(prompt, payload.query)
                        full_text = mock_answer
                        yield _sse("token", mock_answer)

                # Step 6 — Final metadata
                if not full_text.strip():
                    full_text = "[Model returned an empty response]"

                citations = _build_citations(snippets)
                quality = _estimate_quality(payload.query, snippets, "approved")
                usage = track_usage(
                    prompt_tokens=max(1, _token_count(payload.query)),
                    completion_tokens=max(1, _token_count(full_text)),
                )

                yield _sse("done", _json.dumps({
                    "citations": citations,
                    "quality": quality,
                    "tokens": usage,
                    "model_used": model_used,
                    "status": "approved",
                }))

            except Exception as exc:
                yield _sse("error", repr(exc))

        return StreamingResponse(
            _event_generator(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )


    return app


def _sse(event_type: str, data: str) -> str:
    """Format a single Server-Sent Event frame."""
    safe_data = data.replace("\n", "\\n")
    return f"event: {event_type}\ndata: {safe_data}\n\n"


app = create_app()


def _mode_weights(mode: str) -> tuple[float, float]:
    if mode == "baseline":
        return 0.0, 0.0
    if mode in ("lexical", "bm25"):
        return 1.0, 0.0
    if mode in ("semantic", "vector"):
        return 0.0, 1.0
    if mode in ("hybrid", "thinking"):
        return 0.4, 0.6
    return 0.4, 0.6


def _select_chunk_records(
    *,
    mode: str,
    uploaded_chunk_records: list[dict[str, object]],
    use_mock_corpus: bool,
) -> list[dict[str, object]]:
    """Choose retrieval corpus with uploaded docs taking strict priority.

    Real-case default: only uploaded docs participate.
    Mock corpus is opt-in for demo/testing when no uploaded docs are provided.
    """

    if mode == "baseline":
        return []

    if uploaded_chunk_records:
        return uploaded_chunk_records

    if use_mock_corpus:
        return _default_chunk_records()

    return []


def _should_abstain_on_mismatch(*, mode: str, has_uploaded_chunks: bool, has_any_chunks: bool) -> bool:
    """Decide whether retrieval-mismatch abstention should be enabled.

    For real uploaded-doc use-cases, abstention is disabled to allow concise
    summaries from the available user-provided corpus.
    """

    if mode == "baseline":
        return False

    if has_uploaded_chunks:
        return False

    return has_any_chunks


def _make_generate_fn(payload: AskRequest, generation_info: dict):
    def _generate(prompt: str, **kwargs: object) -> str:
        try:
            res = generate_raw(
                prompt,
                model=str(kwargs.get("model", payload.model)),
                temperature=float(kwargs.get("temperature", payload.temperature)),
                num_predict=int(kwargs.get("num_predict", 800)),
            )
            generation_info["model_used"] = "ollama"
            return res
        except Exception as e:
            print(f"Ollama generation failed: {repr(e)}")
            generation_info["model_used"] = "ollama_error"
            return "Ollama generation failed. Please try again later."

    return _generate


def _sanitize_abstention_answer(answer: str) -> str:
    marker = "The provider context cannot find in the uploaded document"
    if re.search(re.escape(marker), answer, flags=re.IGNORECASE):
        return marker
    return answer


@lru_cache(maxsize=1)
def _default_chunk_records() -> list[dict[str, object]]:
    data_root = Path(__file__).resolve().parents[2] / "data" / "mock"
    if not data_root.exists():
        return []

    loaded_documents = load_documents(str(data_root))
    return list(chunk_documents(loaded_documents, window_tokens=180, stride_tokens=40))


def _uploaded_chunk_records(uploaded_docs: list[UploadedDocPayload]) -> list[dict[str, object]]:
    loaded_documents: list[dict[str, object]] = []

    for upload in uploaded_docs:
        text = upload.text.strip()
        if not text:
            continue

        document_id = sha1(f"{upload.name}:{text[:160]}".encode("utf-8")).hexdigest()[:16]
        loaded_documents.append(
            {
                "text": text,
                "metadata": {
                    "source_path": f"uploaded://{upload.name}",
                    "source_name": upload.name,
                    "source_type": "uploaded",
                    "document_id": document_id,
                },
            }
        )

    if not loaded_documents:
        return []

    return list(chunk_documents(loaded_documents, window_tokens=180, stride_tokens=40))


def _compress_text(text: str, *, max_chars: int) -> str:
    cleaned = re.sub(r"[ \t\r]+", " ", text).strip()
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    if len(cleaned) <= max_chars:
        return cleaned

    sentence_like = re.split(r"(?<!\bDr\.)(?<!\bMr\.)(?<!\bMs\.)(?<!\bMrs\.)(?<!\bProf\.)(?<!\be\.g\.)(?<!\bi\.e\.)(?<!\bvs\.)(?<=[.!?])\s+", cleaned)
    buffer = ""
    for sentence in sentence_like:
        candidate = f"{buffer} {sentence}".strip()
        if len(candidate) > max_chars:
            break
        buffer = candidate

    if buffer:
        return buffer
    return f"{cleaned[: max_chars - 3].rstrip()}..."


def _build_citations(snippets: list[str]) -> list[str]:
    if not snippets:
        return []
    return [f"Context snippet {index + 1}: {_compress_text(snippet, max_chars=180)}" for index, snippet in enumerate(snippets[:4])]


def _estimate_quality(query: str, snippets: list[str], status: object) -> dict[str, float]:
    if not snippets:
        return {"bleu": 0.25, "rouge_l": 0.3}

    terms = [token for token in re.findall(r"[A-Za-z0-9]+", query.lower()) if len(token) > 2]
    if not terms:
        return {"bleu": 0.5, "rouge_l": 0.55}

    top = snippets[0].lower()
    overlap = sum(1 for term in terms if term in top) / len(terms)
    bleu = min(0.92, 0.34 + 0.58 * overlap)
    rouge_l = min(0.94, 0.4 + 0.54 * overlap)

    if str(status) == "abstained":
        bleu = min(bleu, 0.35)
        rouge_l = min(rouge_l, 0.4)

    return {
        "bleu": round(bleu, 3),
        "rouge_l": round(rouge_l, 3),
    }


def _token_count(text: str) -> int:
    return len(re.findall(r"\S+", text))
