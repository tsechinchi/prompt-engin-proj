"""Document loading helpers."""

from __future__ import annotations

from hashlib import sha1
from pathlib import Path
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


_SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md"}
_MARKDOWN_SUFFIXES = {".md"}
_TEXT_SUFFIXES = {".txt"}


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
    raise ValueError(f"Unsupported document type: {suffix}")


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _extract_pdf_pages(path: Path) -> list[str]:
    """Extract PDF pages with a pypdf-first strategy and pdfplumber fallback."""

    pypdf_error: Exception | None = None
    pypdf_pages: list[str] = []

    try:
        pypdf_pages = _extract_pdf_pages_with_pypdf(path)
    except Exception as exc:  # pragma: no cover - exercised through fallback tests
        pypdf_error = exc

    if any(_normalize_text(page) for page in pypdf_pages):
        return pypdf_pages

    try:
        pdfplumber_pages = _extract_pdf_pages_with_pdfplumber(path)
    except ImportError:
        if pypdf_error is not None:
            raise RuntimeError(f"Failed to extract PDF text from {path}") from pypdf_error
        if pypdf_pages:
            return pypdf_pages
        raise ImportError("Loading PDF files requires pypdf or pdfplumber.")
    except Exception as exc:  # pragma: no cover - depends on external parsers
        if pypdf_error is not None:
            raise RuntimeError(f"Failed to extract PDF text from {path}") from pypdf_error
        raise RuntimeError(f"Failed to extract PDF text from {path}") from exc

    if any(_normalize_text(page) for page in pdfplumber_pages):
        return pdfplumber_pages

    return pypdf_pages or pdfplumber_pages


def _extract_pdf_pages_with_pypdf(path: Path) -> list[str]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - depends on local environment
        raise ImportError("pypdf is not installed.") from exc

    reader = PdfReader(str(path))
    return [page.extract_text() or "" for page in reader.pages]


def _extract_pdf_pages_with_pdfplumber(path: Path) -> list[str]:
    try:
        import pdfplumber
    except ImportError as exc:  # pragma: no cover - depends on local environment
        raise ImportError("pdfplumber is not installed.") from exc

    with pdfplumber.open(path) as pdf:
        return [page.extract_text() or "" for page in pdf.pages]
