"""Document loading helpers."""

from __future__ import annotations

from functools import lru_cache
from hashlib import sha1
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TypedDict
import re


class DocumentMetadata(TypedDict, total=False):
    """Metadata for a loaded source document."""

    source_path: str
    source_name: str
    source_type: str
    document_id: str
    page_number: int


class LoadedDocument(TypedDict):
    """Normalized text plus citation-friendly metadata."""

    text: str
    metadata: DocumentMetadata


_SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md", ".docx", ".pptx"}
_MARKDOWN_SUFFIXES = {".md"}
_TEXT_SUFFIXES = {".txt"}
_MARKITDOWN_SUFFIXES = {".docx", ".pptx"}


def load_documents(path: str) -> list[LoadedDocument]:
    """Load supported local documents from a file or directory."""

    root = Path(path).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"Path does not exist: {root}")

    file_paths = _collect_paths(root)
    documents: list[LoadedDocument] = []
    for file_path in file_paths:
        documents.extend(_load_file(file_path))
    return documents


def _collect_paths(path: Path) -> list[Path]:
    if path.is_file():
        _validate_supported_file(path)
        return [path]

    return sorted(
        child
        for child in path.rglob("*")
        if child.is_file() and child.suffix.lower() in _SUPPORTED_SUFFIXES
    )


def _load_file(path: Path) -> list[LoadedDocument]:
    suffix = path.suffix.lower()
    metadata = _build_metadata(path)

    if suffix == ".pdf":
        return _load_pdf(path, metadata)

    if suffix in _TEXT_SUFFIXES | _MARKDOWN_SUFFIXES:
        text = _normalize_text(path.read_text(encoding="utf-8"))
        if not text:
            return []
        return [{"text": text, "metadata": metadata}]

    if suffix in _MARKITDOWN_SUFFIXES:
        try:
            text = _normalize_text(_convert_with_markitdown(path))
        except Exception as exc:  # pragma: no cover - depends on converter/backend
            raise RuntimeError(f"Failed to extract document text from {path}") from exc
        if not text:
            return []
        return [{"text": text, "metadata": metadata}]

    raise ValueError(f"Unsupported document type: {path.suffix}")


def _load_pdf(path: Path, metadata: DocumentMetadata) -> list[LoadedDocument]:
    pages = _extract_pdf_pages(path)
    documents: list[LoadedDocument] = []
    for index, page_text in enumerate(pages, start=1):
        normalized = _normalize_text(page_text)
        if not normalized:
            continue
        page_metadata: DocumentMetadata = {
            **metadata,
            "page_number": index,
        }
        documents.append({"text": normalized, "metadata": page_metadata})
    return documents


def _validate_supported_file(path: Path) -> None:
    if path.suffix.lower() not in _SUPPORTED_SUFFIXES:
        raise ValueError(f"Unsupported document type: {path.suffix}")


def _build_metadata(path: Path) -> DocumentMetadata:
    source_type = _source_type_for_suffix(path.suffix.lower())
    resolved = path.resolve()
    return {
        "source_path": str(resolved),
        "source_name": path.name,
        "source_type": source_type,
        "document_id": sha1(str(resolved).encode("utf-8")).hexdigest()[:16],
    }


def _source_type_for_suffix(suffix: str) -> str:
    if suffix == ".pdf":
        return "pdf"
    if suffix in _TEXT_SUFFIXES:
        return "text"
    if suffix in _MARKDOWN_SUFFIXES:
        return "markdown"
    if suffix == ".docx":
        return "docx"
    if suffix == ".pptx":
        return "pptx"
    raise ValueError(f"Unsupported document type: {suffix}")


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _extract_pdf_pages(path: Path) -> list[str]:
    """Extract each PDF page with MarkItDown so page metadata stays stable."""

    try:
        from pypdf import PdfReader, PdfWriter
    except ImportError as exc:  # pragma: no cover - depends on local environment
        raise ImportError("Loading PDF files requires pypdf.") from exc

    reader = PdfReader(str(path))
    pages: list[str] = []

    with TemporaryDirectory() as tmpdir:
        temp_dir = Path(tmpdir)
        for index, page in enumerate(reader.pages, start=1):
            writer = PdfWriter()
            writer.add_page(page)

            page_path = temp_dir / f"page_{index}.pdf"
            with page_path.open("wb") as handle:
                writer.write(handle)

            pages.append(_convert_with_markitdown(page_path))

    return pages


@lru_cache(maxsize=1)
def _get_markitdown_instance():
    try:
        from markitdown import MarkItDown
    except ImportError as exc:  # pragma: no cover - depends on local environment
        raise ImportError("markitdown is not installed.") from exc

    return MarkItDown()


def _convert_with_markitdown(path: Path) -> str:
    converter = _get_markitdown_instance()
    result = converter.convert(str(path))
    text = getattr(result, "text_content", None)
    if isinstance(text, str):
        return text
    return str(result)
