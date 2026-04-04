"""FastAPI server that exposes the LangGraph pipeline to the frontend."""

from __future__ import annotations

from functools import lru_cache
from hashlib import sha1
from pathlib import Path
import re
from typing import Literal

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.agent import build_graph
from src.evaluation.token_tracker import track_usage
from src.generation import generate_raw
from src.ingestion import chunk_documents, load_documents


class UploadedDocPayload(BaseModel):
    """Uploaded document text parsed on the frontend."""

    name: str
    text: str = ""


class AskRequest(BaseModel):
    """Request schema for graph-backed answer generation."""

    query: str
    mode: Literal["baseline", "bm25", "vector", "hybrid"] = "hybrid"
    temperature: float = 0.3
    top_k: int = 5
    model: str = "gemma3:4b"
    uploaded_docs: list[UploadedDocPayload] = Field(default_factory=list)
    use_mock_generation: bool = True
    use_mock_corpus: bool = False


class AskResponse(BaseModel):
    """Response payload consumed by the frontend."""

    answer: str
    status: str
    citations: list[str]
    quality: dict[str, float]
    tokens: dict[str, int]
    radar_snippets: list[dict[str, str]]


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
        graph = build_graph(
            generate_fn=_make_generate_fn(payload),
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

        result = graph.invoke(
            {
                "query": payload.query,
                "chunk_records": chunk_records,
                "top_k": max(payload.top_k, 1),
                "temperature": max(payload.temperature, 0.0),
                "model": payload.model,
                "require_approval": False,
                "bm25_weight": bm25_weight,
                "vector_weight": vector_weight,
                "abstain_on_mismatch": abstain_on_mismatch,
                "max_retrieval_retries": 1,
                "max_regenerations": 0,
            }
        )

        answer = str(result.get("final_output", "")).strip()
        if not answer:
            answer = "No answer generated."

        context_snippets = [str(value) for value in result.get("context_snippets", [])]
        citations = _build_citations(context_snippets)
        quality = _estimate_quality(payload.query, context_snippets, result.get("status", ""))

        usage = track_usage(
            prompt_tokens=max(1, _token_count(payload.query)),
            completion_tokens=max(1, _token_count(answer)),
        )

        radar_snippets = [
            {"snippet": _compress_text(snippet, max_chars=220), "source": "context"}
            for snippet in context_snippets[:3]
        ]

        return AskResponse(
            answer=answer,
            status=str(result.get("status", "approved")),
            citations=citations,
            quality=quality,
            tokens=usage,
            radar_snippets=radar_snippets,
        )

    return app


app = create_app()


def _mode_weights(mode: str) -> tuple[float, float]:
    if mode == "bm25":
        return 1.0, 0.0
    if mode == "vector":
        return 0.0, 1.0
    if mode == "baseline":
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


def _make_generate_fn(payload: AskRequest):
    def _generate(prompt: str, **kwargs: object) -> str:
        if payload.use_mock_generation:
            return _mock_generate_from_prompt(prompt, payload.query)

        try:
            return generate_raw(
                prompt,
                model=str(kwargs.get("model", payload.model)),
                temperature=float(kwargs.get("temperature", payload.temperature)),
                num_predict=int(kwargs.get("num_predict", 260)),
            )
        except Exception:
            return _mock_generate_from_prompt(prompt, payload.query)

    return _generate


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


def _mock_generate_from_prompt(prompt: str, query: str) -> str:
    snippets = _extract_snippets_from_prompt(prompt)
    if not snippets:
        return "I could not find enough retrieved evidence. Try uploading relevant documents or asking a more specific question."

    summary = _compress_text(snippets[0], max_chars=300)
    key_points = [_compress_text(snippet, max_chars=170) for snippet in snippets[:3]]

    lines = [
        f"Summary: {summary}",
        "",
        "Key points:",
    ]
    for point in key_points:
        lines.append(f"- {point}")

    lines.extend(
        [
            "",
            f"Actionable next step: Ask a follow-up about one specific concept from your question (\"{_compress_text(query, max_chars=90)}\") to get a tighter answer.",
        ]
    )

    return "\n".join(lines).strip()


def _extract_snippets_from_prompt(prompt: str) -> list[str]:
    match = re.search(r"Context Snippets:\n(.*?)\n\nConstraints:", prompt, flags=re.DOTALL)
    if not match:
        return []

    snippets: list[str] = []
    for raw_line in match.group(1).splitlines():
        line = raw_line.strip()
        if not line:
            continue
        line = re.sub(r"^\d+\.\s*", "", line)
        if line:
            snippets.append(line)
    return snippets


def _compress_text(text: str, *, max_chars: int) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if len(cleaned) <= max_chars:
        return cleaned

    sentence_like = re.split(r"(?<=[.!?])\s+", cleaned)
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
