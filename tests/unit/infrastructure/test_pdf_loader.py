"""Unit tests for PyMuPDFDocumentLoader."""

from __future__ import annotations

from pathlib import Path

import fitz
import pytest

from pdf_extractor.infrastructure.pdf_loader import PyMuPDFDocumentLoader


@pytest.fixture()
def pdf_with_text(tmp_path: Path) -> Path:
    path = tmp_path / "test.pdf"
    with fitz.open() as doc:
        page = doc.new_page()
        page.insert_text((72, 72), "Page one content. " * 10)
        page2 = doc.new_page()
        page2.insert_text((72, 72), "Page two content. " * 10)
        doc.save(str(path))
    return path


@pytest.fixture()
def pdf_blank_pages(tmp_path: Path) -> Path:
    """PDF with only blank pages (no text)."""
    path = tmp_path / "blank.pdf"
    with fitz.open() as doc:
        doc.new_page()
        doc.new_page()
        doc.save(str(path))
    return path


class TestPyMuPDFDocumentLoader:
    def test_loads_text_pages(self, pdf_with_text: Path) -> None:
        loader = PyMuPDFDocumentLoader()
        docs = loader.load(pdf_with_text)
        assert len(docs) == 2
        assert all("content" in d.text.lower() for d in docs)

    def test_page_numbers_correct(self, pdf_with_text: Path) -> None:
        loader = PyMuPDFDocumentLoader()
        docs = loader.load(pdf_with_text)
        assert docs[0].page_number == 0
        assert docs[1].page_number == 1

    def test_doc_ids_unique(self, pdf_with_text: Path) -> None:
        loader = PyMuPDFDocumentLoader()
        docs = loader.load(pdf_with_text)
        ids = [d.doc_id for d in docs]
        assert len(ids) == len(set(ids))

    def test_metadata_contains_source(self, pdf_with_text: Path) -> None:
        loader = PyMuPDFDocumentLoader()
        docs = loader.load(pdf_with_text)
        for doc in docs:
            assert "source" in doc.metadata

    def test_blank_pages_skipped(self, pdf_blank_pages: Path) -> None:
        loader = PyMuPDFDocumentLoader()
        docs = loader.load(pdf_blank_pages)
        assert docs == []

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        loader = PyMuPDFDocumentLoader()
        with pytest.raises(FileNotFoundError):
            loader.load(tmp_path / "nonexistent.pdf")

    def test_file_path_stored_correctly(self, pdf_with_text: Path) -> None:
        loader = PyMuPDFDocumentLoader()
        docs = loader.load(pdf_with_text)
        assert all(d.file_path == pdf_with_text for d in docs)
