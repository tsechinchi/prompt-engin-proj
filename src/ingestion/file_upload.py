"""Helpers for file upload persistence and ingestion."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
import re

from .loader import LoadedDocument, load_documents


def save_uploaded_file(
    *,
    filename: str,
    content: bytes,
    upload_dir: str = "data/uploads",
) -> Path:
    """Save uploaded content to disk with a sanitized filename."""

    cleaned_name = _sanitize_filename(filename)
    if not cleaned_name:
        raise ValueError("Uploaded filename is empty after sanitization.")

    directory = Path(upload_dir).expanduser()
    directory.mkdir(parents=True, exist_ok=True)

    destination = _unique_destination(directory / cleaned_name)
    destination.write_bytes(content)
    return destination


def ingest_uploaded_file(
    *,
    filename: str,
    content: bytes,
    upload_dir: str = "data/uploads",
) -> list[LoadedDocument]:
    """Persist one uploaded file and parse it with the standard loader."""

    saved_path = save_uploaded_file(filename=filename, content=content, upload_dir=upload_dir)
    return load_documents(str(saved_path))


def ingest_uploaded_files(
    files: Iterable[tuple[str, bytes]],
    *,
    upload_dir: str = "data/uploads",
) -> list[LoadedDocument]:
    """Persist and ingest multiple uploaded files."""

    documents: list[LoadedDocument] = []
    for filename, content in files:
        documents.extend(ingest_uploaded_file(filename=filename, content=content, upload_dir=upload_dir))
    return documents


def _sanitize_filename(filename: str) -> str:
    name = Path(filename).name.strip()
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"[^A-Za-z0-9._-]", "", name)
    return name


def _unique_destination(base_path: Path) -> Path:
    if not base_path.exists():
        return base_path

    stem = base_path.stem
    suffix = base_path.suffix
    parent = base_path.parent

    counter = 1
    while True:
        candidate = parent / f"{stem}_{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1