"""Unit tests for IngestUseCase."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pdf_extractor.application.ingest_use_case import IngestUseCase
from pdf_extractor.domain.entities import Chunk, Document
from pdf_extractor.domain.interfaces import (
    IDocumentLoader,
    IEmbeddingService,
    ITextChunker,
    IVectorStore,
)


@pytest.fixture()
def doc(tmp_path: Path) -> Document:
    return Document(
        doc_id="doc-x",
        file_path=tmp_path / "a.pdf",
        text="Some content here",
        page_number=0,
    )


@pytest.fixture()
def chunk() -> Chunk:
    return Chunk(chunk_id="ck-1", doc_id="doc-x", text="Some content here", char_start=0, char_end=17)


@pytest.fixture()
def use_case(doc: Document, chunk: Chunk):
    loader = MagicMock(spec=IDocumentLoader)
    loader.load.return_value = [doc]

    chunker = MagicMock(spec=ITextChunker)
    chunker.chunk.return_value = [chunk]

    embedder = MagicMock(spec=IEmbeddingService)
    embedder.embed.return_value = [[0.1, 0.2, 0.3]]

    store = MagicMock(spec=IVectorStore)
    store.get_stored_doc_ids.return_value = set()

    uc = IngestUseCase(loader, chunker, embedder, store)
    return uc, loader, chunker, embedder, store


class TestIngestFile:
    def test_full_pipeline_called(self, tmp_path: Path, use_case) -> None:
        uc, loader, chunker, embedder, store = use_case
        pdf = tmp_path / "a.pdf"
        count = uc.ingest_file(pdf)

        loader.load.assert_called_once_with(pdf)
        chunker.chunk.assert_called_once()
        embedder.embed.assert_called_once()
        store.upsert.assert_called_once()
        assert count == 1

    def test_skips_already_ingested(self, tmp_path: Path, use_case, doc: Document) -> None:
        uc, loader, chunker, embedder, store = use_case
        store.get_stored_doc_ids.return_value = {doc.doc_id}

        count = uc.ingest_file(tmp_path / "a.pdf")

        chunker.chunk.assert_not_called()
        embedder.embed.assert_not_called()
        store.upsert.assert_not_called()
        assert count == 0

    def test_no_text_extracted(self, tmp_path: Path, use_case) -> None:
        uc, loader, chunker, embedder, store = use_case
        loader.load.return_value = []

        count = uc.ingest_file(tmp_path / "empty.pdf")
        assert count == 0
        store.upsert.assert_not_called()

    def test_no_chunks_produced(self, tmp_path: Path, use_case) -> None:
        uc, loader, chunker, embedder, store = use_case
        chunker.chunk.return_value = []

        count = uc.ingest_file(tmp_path / "a.pdf")
        assert count == 0
        store.upsert.assert_not_called()


class TestIngestDirectory:
    def test_ingests_all_pdfs(self, tmp_path: Path, use_case) -> None:
        uc, loader, chunker, embedder, store = use_case
        (tmp_path / "a.pdf").touch()
        (tmp_path / "b.pdf").touch()

        results = uc.ingest_directory(tmp_path)
        assert len(results) == 2
        assert loader.load.call_count == 2

    def test_empty_dir_returns_empty(self, tmp_path: Path, use_case) -> None:
        uc, *_ = use_case
        results = uc.ingest_directory(tmp_path)
        assert results == {}
