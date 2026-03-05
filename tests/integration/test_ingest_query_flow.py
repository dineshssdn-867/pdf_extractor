"""End-to-end integration test: real Chroma + mocked Ollama."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import fitz
import pytest

from pdf_extractor.application.ingest_use_case import IngestUseCase
from pdf_extractor.application.query_use_case import QueryUseCase
from pdf_extractor.config.settings import AppSettings
from pdf_extractor.domain.interfaces import IEmbeddingService, ILLMService
from pdf_extractor.infrastructure.chroma_store import ChromaVectorStore
from pdf_extractor.infrastructure.pdf_loader import PyMuPDFDocumentLoader
from pdf_extractor.infrastructure.text_chunker import SlidingWindowChunker


@pytest.fixture()
def sample_pdf(tmp_path: Path) -> Path:
    path = tmp_path / "sample.pdf"
    with fitz.open() as doc:
        page = doc.new_page()
        page.insert_text((72, 72), "The capital of France is Paris. " * 15)
        doc.save(str(path))
    return path


@pytest.fixture()
def mock_embedder() -> IEmbeddingService:
    emb = MagicMock(spec=IEmbeddingService)
    # Return deterministic 3-d vectors
    emb.embed.side_effect = lambda texts: [[0.1 * (i + 1), 0.2, 0.3] for i, _ in enumerate(texts)]
    emb.dimension = 3
    return emb


@pytest.fixture()
def mock_llm() -> ILLMService:
    llm = MagicMock(spec=ILLMService)
    llm.generate.return_value = "Paris is the capital of France."
    return llm


@pytest.fixture()
def store() -> ChromaVectorStore:
    import uuid
    return ChromaVectorStore(persist_path=None, collection_name=f"e2e_{uuid.uuid4().hex}")


@pytest.fixture()
def settings() -> AppSettings:
    return AppSettings(retrieval_top_k=3, otel_enabled=False)


class TestIngestQueryFlow:
    def test_ingest_then_query(
        self,
        sample_pdf: Path,
        mock_embedder: IEmbeddingService,
        mock_llm: ILLMService,
        store: ChromaVectorStore,
        settings: AppSettings,
    ) -> None:
        # Ingest
        ingest_uc = IngestUseCase(
            loader=PyMuPDFDocumentLoader(),
            chunker=SlidingWindowChunker(chunk_size=100, chunk_overlap=10),
            embedder=mock_embedder,
            vector_store=store,
        )
        count = ingest_uc.ingest_file(sample_pdf)
        assert count > 0

        # Reset embedder call count
        mock_embedder.embed.reset_mock()
        mock_embedder.embed.side_effect = lambda texts: [[0.1, 0.2, 0.3]] * len(texts)

        # Query
        query_uc = QueryUseCase(mock_embedder, store, mock_llm, settings)
        result = query_uc.execute("What is the capital of France?")

        assert result.question == "What is the capital of France?"
        assert result.answer == "Paris is the capital of France."
        assert len(result.source_chunks) > 0
        mock_llm.generate.assert_called_once()

    def test_idempotent_ingest(
        self,
        sample_pdf: Path,
        mock_embedder: IEmbeddingService,
        mock_llm: ILLMService,
        store: ChromaVectorStore,
    ) -> None:
        ingest_uc = IngestUseCase(
            loader=PyMuPDFDocumentLoader(),
            chunker=SlidingWindowChunker(chunk_size=100, chunk_overlap=10),
            embedder=mock_embedder,
            vector_store=store,
        )
        count1 = ingest_uc.ingest_file(sample_pdf)
        count2 = ingest_uc.ingest_file(sample_pdf)
        assert count1 > 0
        assert count2 == 0  # second run skipped

    def test_empty_store_query_returns_result(
        self,
        mock_embedder: IEmbeddingService,
        mock_llm: ILLMService,
        store: ChromaVectorStore,
        settings: AppSettings,
    ) -> None:
        mock_embedder.embed.side_effect = lambda texts: [[0.1, 0.2, 0.3]] * len(texts)
        query_uc = QueryUseCase(mock_embedder, store, mock_llm, settings)
        result = query_uc.execute("Any question?")
        assert result.answer == "Paris is the capital of France."
        assert result.source_chunks == []
