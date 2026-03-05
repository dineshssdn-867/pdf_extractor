"""Unit tests for SlidingWindowChunker."""

from __future__ import annotations

from pathlib import Path

import pytest

from pdf_extractor.domain.entities import Document
from pdf_extractor.infrastructure.text_chunker import SlidingWindowChunker


def _doc(text: str, doc_id: str = "d1") -> Document:
    return Document(doc_id=doc_id, file_path=Path("f.pdf"), text=text, page_number=0)


class TestSlidingWindowChunker:
    def test_short_text_single_chunk(self) -> None:
        chunker = SlidingWindowChunker(chunk_size=512, chunk_overlap=64)
        chunks = chunker.chunk([_doc("Hello world")])
        assert len(chunks) == 1
        assert chunks[0].text == "Hello world"
        assert chunks[0].char_start == 0
        assert chunks[0].char_end == 11

    def test_long_text_multiple_chunks(self) -> None:
        text = "A" * 1000
        chunker = SlidingWindowChunker(chunk_size=200, chunk_overlap=50)
        chunks = chunker.chunk([_doc(text)])
        assert len(chunks) > 1
        # All chars covered
        assert chunks[0].char_start == 0
        assert chunks[-1].char_end == 1000

    def test_overlap_accuracy(self) -> None:
        text = "X" * 300
        chunk_size, overlap = 100, 20
        chunker = SlidingWindowChunker(chunk_size=chunk_size, chunk_overlap=overlap)
        chunks = chunker.chunk([_doc(text)])
        step = chunk_size - overlap
        # Second chunk should start at step
        assert chunks[1].char_start == step
        # Overlap between consecutive chunks
        overlap_actual = chunks[0].char_end - chunks[1].char_start
        assert overlap_actual == overlap

    def test_exact_size_text(self) -> None:
        text = "B" * 100
        chunker = SlidingWindowChunker(chunk_size=100, chunk_overlap=0)
        chunks = chunker.chunk([_doc(text)])
        assert len(chunks) == 1
        assert chunks[0].char_end == 100

    def test_empty_document_returns_no_chunks(self) -> None:
        # Document with empty text bypasses __post_init__ — we override manually
        doc = Document.__new__(Document)
        object.__setattr__(doc, "doc_id", "d1")
        object.__setattr__(doc, "file_path", Path("f.pdf"))
        object.__setattr__(doc, "text", "")
        object.__setattr__(doc, "page_number", 0)
        object.__setattr__(doc, "metadata", {})

        chunker = SlidingWindowChunker(chunk_size=100, chunk_overlap=10)
        chunks = chunker.chunk([doc])
        assert chunks == []

    def test_multiple_documents(self) -> None:
        docs = [_doc("Hello", doc_id="d1"), _doc("World", doc_id="d2")]
        chunker = SlidingWindowChunker(chunk_size=512, chunk_overlap=0)
        chunks = chunker.chunk(docs)
        assert len(chunks) == 2
        assert {c.doc_id for c in chunks} == {"d1", "d2"}

    def test_chunk_ids_unique(self) -> None:
        text = "Z" * 600
        chunker = SlidingWindowChunker(chunk_size=200, chunk_overlap=50)
        chunks = chunker.chunk([_doc(text)])
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_invalid_overlap_raises(self) -> None:
        with pytest.raises(ValueError):
            SlidingWindowChunker(chunk_size=100, chunk_overlap=100)

    def test_invalid_chunk_size_raises(self) -> None:
        with pytest.raises(ValueError):
            SlidingWindowChunker(chunk_size=0, chunk_overlap=0)

    def test_single_char_text(self) -> None:
        chunker = SlidingWindowChunker(chunk_size=512, chunk_overlap=64)
        chunks = chunker.chunk([_doc("X")])
        assert len(chunks) == 1
        assert chunks[0].text == "X"
        assert chunks[0].char_start == 0
        assert chunks[0].char_end == 1
