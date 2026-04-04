from __future__ import annotations

from tempfile import TemporaryDirectory
import unittest

from src.ingestion import ingest_uploaded_file, ingest_uploaded_files, save_uploaded_file


class FileUploadTests(unittest.TestCase):
    def test_save_uploaded_file_sanitizes_name_and_writes_bytes(self) -> None:
        with TemporaryDirectory() as tmpdir:
            saved = save_uploaded_file(
                filename="../My Notes?.txt",
                content=b"Hello upload",
                upload_dir=tmpdir,
            )

            self.assertEqual(saved.name, "My_Notes.txt")
            self.assertEqual(saved.read_bytes(), b"Hello upload")

    def test_save_uploaded_file_avoids_overwrite(self) -> None:
        with TemporaryDirectory() as tmpdir:
            first = save_uploaded_file(filename="same.txt", content=b"one", upload_dir=tmpdir)
            second = save_uploaded_file(filename="same.txt", content=b"two", upload_dir=tmpdir)

            self.assertEqual(first.name, "same.txt")
            self.assertEqual(second.name, "same_1.txt")

    def test_ingest_uploaded_file_loads_text_document(self) -> None:
        with TemporaryDirectory() as tmpdir:
            docs = ingest_uploaded_file(
                filename="faq.txt",
                content=b"Exam is on December 10.",
                upload_dir=tmpdir,
            )

            self.assertEqual(len(docs), 1)
            self.assertEqual(docs[0]["text"], "Exam is on December 10.")
            self.assertEqual(docs[0]["metadata"]["source_name"], "faq.txt")  # type: ignore[index]

    def test_ingest_uploaded_files_batches_documents(self) -> None:
        with TemporaryDirectory() as tmpdir:
            docs = ingest_uploaded_files(
                [
                    ("a.txt", b"Alpha"),
                    ("b.md", b"# Beta"),
                ],
                upload_dir=tmpdir,
            )

            self.assertEqual(len(docs), 2)
            names = [doc["metadata"]["source_name"] for doc in docs]  # type: ignore[index]
            self.assertEqual(names, ["a.txt", "b.md"])

    def test_empty_filename_after_sanitization_raises(self) -> None:
        with TemporaryDirectory() as tmpdir:
            with self.assertRaises(ValueError):
                save_uploaded_file(filename="???", content=b"x", upload_dir=tmpdir)


if __name__ == "__main__":
    unittest.main()