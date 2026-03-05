"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from pdf_extractor.domain.entities import Chunk, Document, QueryResult, RetrievedChunk
from pdf_extractor.domain.interfaces import IEmbeddingService, ILLMService, IVectorStore


# ---------------------------------------------------------------------------
# Domain fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def sample_document(tmp_path: Path) -> Document:
    return Document(
        doc_id="doc-001",
        file_path=tmp_path / "sample.pdf",
        text="Hello world. This is a test document with some content.",
        page_number=0,
        metadata={"source": "sample.pdf", "page": "0"},
    )


@pytest.fixture()
def sample_chunk() -> Chunk:
    return Chunk(
        chunk_id="chunk-001",
        doc_id="doc-001",
        text="Hello world.",
        char_start=0,
        char_end=12,
        page_number=0,
        metadata={"source": "sample.pdf"},
    )


@pytest.fixture()
def sample_retrieved_chunk(sample_chunk: Chunk) -> RetrievedChunk:
    return RetrievedChunk(chunk=sample_chunk, score=0.9)


# ---------------------------------------------------------------------------
# Mock services
# ---------------------------------------------------------------------------

@pytest.fixture()
def mock_embedder() -> IEmbeddingService:
    embedder = MagicMock(spec=IEmbeddingService)
    embedder.embed.return_value = [[0.1, 0.2, 0.3]]
    embedder.dimension = 3
    return embedder


@pytest.fixture()
def mock_llm() -> ILLMService:
    llm = MagicMock(spec=ILLMService)
    llm.generate.return_value = "This is a generated answer."
    return llm


@pytest.fixture()
def mock_vector_store(sample_retrieved_chunk: RetrievedChunk) -> IVectorStore:
    store = MagicMock(spec=IVectorStore)
    store.query.return_value = [sample_retrieved_chunk]
    store.get_stored_doc_ids.return_value = set()
    return store


# ---------------------------------------------------------------------------
# Chroma ephemeral store
# ---------------------------------------------------------------------------

@pytest.fixture()
def tmp_chroma(tmp_path: Path):
    """Return a ChromaVectorStore backed by an ephemeral (in-memory) client."""
    from pdf_extractor.infrastructure.chroma_store import ChromaVectorStore
    return ChromaVectorStore(persist_path=None, collection_name="test_collection")


# ---------------------------------------------------------------------------
# Sample PDF (built with PyMuPDF in-memory)
# ---------------------------------------------------------------------------

@pytest.fixture()
def sample_pdf(tmp_path: Path) -> Path:
    """Create a minimal single-page PDF for testing."""
    import fitz

    pdf_path = tmp_path / "sample.pdf"
    with fitz.open() as doc:
        page = doc.new_page()
        page.insert_text((72, 72), "Hello from a test PDF. " * 20)
        doc.save(str(pdf_path))
    return pdf_path
