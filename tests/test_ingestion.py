from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from src.ingestion import chunk_documents, chunk_text, load_documents
from src.ingestion.chunker import _count_tokens


class LoadDocumentsTests(unittest.TestCase):
    def test_load_text_and_markdown_files(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            txt_path = root / "notes.txt"
            md_path = root / "guide.md"
            txt_path.write_text("Line one.\n\nLine two.", encoding="utf-8")
            md_path.write_text("# Title\n\nBullet point", encoding="utf-8")

            txt_docs = load_documents(str(txt_path))
            md_docs = load_documents(str(md_path))

            self.assertEqual(len(txt_docs), 1)
            self.assertEqual(txt_docs[0]["text"], "Line one. Line two.")
            self.assertEqual(txt_docs[0]["metadata"]["source_type"], "text")
            self.assertEqual(txt_docs[0]["metadata"]["source_name"], "notes.txt")

            self.assertEqual(len(md_docs), 1)
            self.assertEqual(md_docs[0]["text"], "# Title Bullet point")
            self.assertEqual(md_docs[0]["metadata"]["source_type"], "markdown")
            self.assertEqual(md_docs[0]["metadata"]["source_name"], "guide.md")

    def test_load_directory_in_deterministic_order(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "zeta.md").write_text("zeta", encoding="utf-8")
            (root / "alpha.txt").write_text("alpha", encoding="utf-8")
            nested = root / "nested"
            nested.mkdir()
            (nested / "beta.txt").write_text("beta", encoding="utf-8")
            (root / "ignored.csv").write_text("ignore", encoding="utf-8")

            docs = load_documents(str(root))

            self.assertEqual([doc["text"] for doc in docs], ["alpha", "beta", "zeta"])
            self.assertEqual(
                [doc["metadata"]["source_name"] for doc in docs],
                ["alpha.txt", "beta.txt", "zeta.md"],
            )

    def test_reject_missing_and_unsupported_paths(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            unsupported = root / "data.csv"
            unsupported.write_text("a,b,c", encoding="utf-8")

            with self.assertRaises(FileNotFoundError):
                load_documents(str(root / "missing.txt"))

            with self.assertRaises(ValueError):
                load_documents(str(unsupported))

    def test_pdf_loader_preserves_page_numbers(self) -> None:
        with TemporaryDirectory() as tmpdir:
            pdf_path = Path(tmpdir) / "doc.pdf"
            pdf_path.write_bytes(b"%PDF-pretend")

            with patch(
                "src.ingestion.loader._extract_pdf_pages",
                return_value=["Page one text", "   ", "Page three text"],
            ):
                docs = load_documents(str(pdf_path))

            self.assertEqual(len(docs), 2)
            self.assertEqual(docs[0]["text"], "Page one text")
            self.assertEqual(docs[0]["metadata"]["page_number"], 1)
            self.assertEqual(docs[1]["text"], "Page three text")
            self.assertEqual(docs[1]["metadata"]["page_number"], 3)
            self.assertEqual(docs[0]["metadata"]["source_type"], "pdf")


class ChunkingTests(unittest.TestCase):
    def test_chunk_short_text_returns_one_chunk(self) -> None:
        text = "Short input for chunking."

        chunks = chunk_text(text, window_tokens=50, stride_tokens=10)

        self.assertEqual(chunks, [text])

    def test_chunk_text_preserves_order_with_overlap(self) -> None:
        text = "a b c. d e f. g h i. j k l."
        window_tokens = _count_tokens("a b c. d e f.")
        stride_tokens = _count_tokens("d e f.")

        chunks = chunk_text(text, window_tokens=window_tokens, stride_tokens=stride_tokens)

        self.assertGreaterEqual(len(chunks), 3)
        self.assertEqual(
            chunks,
            [
                "a b c. d e f.",
                "d e f. g h i.",
                "g h i. j k l.",
            ],
        )

    def test_overlong_sentence_falls_back_to_token_windows(self) -> None:
        text = "token " * 40

        chunks = chunk_text(text, window_tokens=10, stride_tokens=3)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(_count_tokens(chunk) <= 10 for chunk in chunks))

    def test_chunk_documents_preserves_and_extends_metadata(self) -> None:
        documents = [
            {
                "text": "Alpha beta gamma. Delta epsilon zeta. Eta theta iota.",
                "metadata": {
                    "source_path": "/tmp/doc.txt",
                    "source_name": "doc.txt",
                    "source_type": "text",
                    "document_id": "doc-1",
                },
            }
        ]

        chunks = chunk_documents(documents, window_tokens=8, stride_tokens=4)

        self.assertGreaterEqual(len(chunks), 2)
        for index, chunk in enumerate(chunks):
            self.assertEqual(chunk["metadata"]["source_path"], "/tmp/doc.txt")
            self.assertEqual(chunk["metadata"]["source_name"], "doc.txt")
            self.assertEqual(chunk["metadata"]["document_id"], "doc-1")
            self.assertEqual(chunk["metadata"]["chunk_index"], index)
            self.assertEqual(chunk["metadata"]["chunk_id"], f"doc-1:{index}")
            self.assertEqual(chunk["metadata"]["token_count"], _count_tokens(chunk["text"]))


if __name__ == "__main__":
    unittest.main()
