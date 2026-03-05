"""QueryUseCase: embed → retrieve → build_prompt → generate (4 OTel spans)."""

from __future__ import annotations

import logging
import time

from pdf_extractor.config.settings import AppSettings, get_settings
from pdf_extractor.domain.entities import QueryResult
from pdf_extractor.domain.interfaces import IEmbeddingService, ILLMService, IVectorStore
from pdf_extractor.infrastructure.observability import get_tracer

logger = logging.getLogger(__name__)


class QueryUseCase:
    """Executes a RAG query with full OpenTelemetry instrumentation."""

    def __init__(
        self,
        embedder: IEmbeddingService,
        vector_store: IVectorStore,
        llm: ILLMService,
        settings: AppSettings | None = None,
    ) -> None:
        self._embedder = embedder
        self._vector_store = vector_store
        self._llm = llm
        self._settings = settings or get_settings()

    def execute(self, question: str) -> QueryResult:
        if not question.strip():
            raise ValueError("question must not be empty")

        tracer = get_tracer()
        settings = self._settings
        total_start = time.perf_counter()
        logger.info("[query] START question=%r", question[:80])

        with tracer.start_as_current_span("retrieve") as span:
            t0 = time.perf_counter()
            logger.info("[query] embedding question ...")
            query_embedding = self._embedder.embed([question])[0]
            embed_ms = (time.perf_counter() - t0) * 1000
            logger.info("[query] embedding done in %.0f ms", embed_ms)

            t0 = time.perf_counter()
            logger.info("[query] querying vector store (top_k=%d) ...", settings.retrieval_top_k)
            chunks = self._vector_store.query(
                query_embedding, top_k=settings.retrieval_top_k
            )
            retrieve_ms = (time.perf_counter() - t0) * 1000
            span.set_attribute("retrieved_count", len(chunks))
            logger.info("[query] vector store returned %d chunks in %.0f ms", len(chunks), retrieve_ms)

        with tracer.start_as_current_span("prompt_build") as span:
            t0 = time.perf_counter()
            context = "\n\n".join(c.chunk.text for c in chunks)
            prompt = settings.rag_prompt_template.format(
                context=context, question=question
            )
            prompt_ms = (time.perf_counter() - t0) * 1000
            span.set_attribute("prompt_length", len(prompt))
            logger.info("[query] prompt built: %d chars in %.0f ms", len(prompt), prompt_ms)

        with tracer.start_as_current_span("llm_generate") as span:
            t0 = time.perf_counter()
            logger.info("[query] calling LLM (model=%s) ...", settings.ollama_model)
            answer = self._llm.generate(prompt)
            llm_ms = (time.perf_counter() - t0) * 1000
            span.set_attribute("answer_length", len(answer))
            logger.info("[query] LLM responded: %d chars in %.0f ms", len(answer), llm_ms)

        total_ms = (time.perf_counter() - total_start) * 1000
        logger.info(
            "[query] DONE total=%.0f ms  (embed=%.0f ms, retrieve=%.0f ms, llm=%.0f ms)",
            total_ms, embed_ms, retrieve_ms, llm_ms,
        )

        with tracer.start_as_current_span("answer"):
            return QueryResult(
                question=question,
                answer=answer,
                source_chunks=chunks,
            )
