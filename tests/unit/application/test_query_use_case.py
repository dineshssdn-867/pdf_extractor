"""Unit tests for QueryUseCase."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from pdf_extractor.application.query_use_case import QueryUseCase
from pdf_extractor.config.settings import AppSettings
from pdf_extractor.domain.entities import Chunk, RetrievedChunk
from pdf_extractor.domain.interfaces import IEmbeddingService, ILLMService, IVectorStore


@pytest.fixture()
def settings() -> AppSettings:
    return AppSettings(
        retrieval_top_k=3,
        otel_enabled=False,
    )


@pytest.fixture()
def retrieved_chunk() -> RetrievedChunk:
    chunk = Chunk(
        chunk_id="c1", doc_id="d1", text="Relevant text.", char_start=0, char_end=14
    )
    return RetrievedChunk(chunk=chunk, score=0.85)


@pytest.fixture()
def use_case(retrieved_chunk: RetrievedChunk, settings: AppSettings):
    embedder = MagicMock(spec=IEmbeddingService)
    embedder.embed.return_value = [[0.1, 0.2, 0.3]]

    store = MagicMock(spec=IVectorStore)
    store.query.return_value = [retrieved_chunk]

    llm = MagicMock(spec=ILLMService)
    llm.generate.return_value = "The answer is 42."

    return QueryUseCase(embedder, store, llm, settings), embedder, store, llm


class TestQueryUseCaseExecute:
    def test_returns_query_result(self, use_case) -> None:
        uc, *_ = use_case
        result = uc.execute("What is the answer?")
        assert result.question == "What is the answer?"
        assert result.answer == "The answer is 42."
        assert len(result.source_chunks) == 1

    def test_calls_embed_with_question(self, use_case) -> None:
        uc, embedder, *_ = use_case
        uc.execute("My question")
        embedder.embed.assert_called_once_with(["My question"])

    def test_calls_vector_store_query(self, use_case, settings: AppSettings) -> None:
        uc, embedder, store, *_ = use_case
        uc.execute("Q?")
        store.query.assert_called_once_with([0.1, 0.2, 0.3], top_k=settings.retrieval_top_k)

    def test_calls_llm_generate(self, use_case) -> None:
        uc, embedder, store, llm = use_case
        uc.execute("Q?")
        llm.generate.assert_called_once()
        prompt_arg = llm.generate.call_args[0][0]
        assert "Relevant text." in prompt_arg
        assert "Q?" in prompt_arg

    def test_empty_question_raises(self, use_case) -> None:
        uc, *_ = use_case
        with pytest.raises(ValueError, match="question"):
            uc.execute("   ")

    def test_execute_logs_timing(self, retrieved_chunk: RetrievedChunk, settings: AppSettings) -> None:
        """Verify execute completes and returns a QueryResult with timing logged."""
        embedder = MagicMock(spec=IEmbeddingService)
        embedder.embed.return_value = [[0.1, 0.2, 0.3]]
        store = MagicMock(spec=IVectorStore)
        store.query.return_value = [retrieved_chunk]
        llm = MagicMock(spec=ILLMService)
        llm.generate.return_value = "The answer."

        uc = QueryUseCase(embedder, store, llm, settings)
        result = uc.execute("Test question?")

        assert result.answer == "The answer."
        assert result.question == "Test question?"
        assert result.source_chunks == [retrieved_chunk]
