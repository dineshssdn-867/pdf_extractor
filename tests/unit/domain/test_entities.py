"""Unit tests for domain entities."""

from __future__ import annotations

from pathlib import Path

import pytest

from pdf_extractor.domain.entities import Chunk, Document, QueryResult, RetrievedChunk


class TestDocument:
    def test_valid_document(self, tmp_path: Path) -> None:
        doc = Document(
            doc_id="abc123",
            file_path=tmp_path / "f.pdf",
            text="Some text",
            page_number=0,
        )
        assert doc.doc_id == "abc123"
        assert doc.page_number == 0

    def test_empty_doc_id_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="doc_id"):
            Document(doc_id="", file_path=tmp_path / "f.pdf", text="x")

    def test_negative_page_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="page_number"):
            Document(doc_id="x", file_path=tmp_path / "f.pdf", text="x", page_number=-1)

    def test_frozen(self, tmp_path: Path) -> None:
        doc = Document(doc_id="x", file_path=tmp_path / "f.pdf", text="x")
        with pytest.raises((AttributeError, TypeError)):
            doc.doc_id = "y"  # type: ignore[misc]


class TestChunk:
    def test_valid_chunk(self) -> None:
        c = Chunk(
            chunk_id="c1",
            doc_id="d1",
            text="hello",
            char_start=0,
            char_end=5,
        )
        assert c.text == "hello"
        assert c.char_end - c.char_start == 5

    def test_empty_text_raises(self) -> None:
        with pytest.raises(ValueError, match="text"):
            Chunk(chunk_id="c1", doc_id="d1", text="", char_start=0, char_end=5)

    def test_negative_char_start_raises(self) -> None:
        with pytest.raises(ValueError, match="char_start"):
            Chunk(chunk_id="c1", doc_id="d1", text="hi", char_start=-1, char_end=2)

    def test_char_end_le_start_raises(self) -> None:
        with pytest.raises(ValueError, match="char_end"):
            Chunk(chunk_id="c1", doc_id="d1", text="hi", char_start=5, char_end=3)

    def test_char_end_equal_start_raises(self) -> None:
        with pytest.raises(ValueError, match="char_end"):
            Chunk(chunk_id="c1", doc_id="d1", text="hi", char_start=3, char_end=3)

    def test_frozen(self) -> None:
        c = Chunk(chunk_id="c1", doc_id="d1", text="hi", char_start=0, char_end=2)
        with pytest.raises((AttributeError, TypeError)):
            c.text = "bye"  # type: ignore[misc]


class TestRetrievedChunk:
    def _chunk(self) -> Chunk:
        return Chunk(chunk_id="c1", doc_id="d1", text="hi", char_start=0, char_end=2)

    def test_valid_score_boundaries(self) -> None:
        rc0 = RetrievedChunk(chunk=self._chunk(), score=0.0)
        rc1 = RetrievedChunk(chunk=self._chunk(), score=1.0)
        assert rc0.score == 0.0
        assert rc1.score == 1.0

    def test_score_above_1_raises(self) -> None:
        with pytest.raises(ValueError, match="score"):
            RetrievedChunk(chunk=self._chunk(), score=1.01)

    def test_score_below_0_raises(self) -> None:
        with pytest.raises(ValueError, match="score"):
            RetrievedChunk(chunk=self._chunk(), score=-0.01)


class TestQueryResult:
    def test_valid(self) -> None:
        chunk = Chunk(chunk_id="c1", doc_id="d1", text="hi", char_start=0, char_end=2)
        rc = RetrievedChunk(chunk=chunk, score=0.8)
        qr = QueryResult(question="What?", answer="42", source_chunks=[rc])
        assert qr.question == "What?"

    def test_empty_question_raises(self) -> None:
        with pytest.raises(ValueError, match="question"):
            QueryResult(question="", answer="x", source_chunks=[])
