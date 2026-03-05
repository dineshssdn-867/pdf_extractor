"""Ollama-based embedding service (primary embedder)."""

from __future__ import annotations

import logging
import time

import ollama

from pdf_extractor.domain.interfaces import IEmbeddingService

logger = logging.getLogger(__name__)


class OllamaEmbeddingService(IEmbeddingService):
    """Generates embeddings using a locally-running Ollama model."""

    def __init__(self, model: str = "nomic-embed-text", base_url: str = "http://localhost:11434") -> None:
        self._model = model
        self._client = ollama.Client(host=base_url)
        self._dimension: int | None = None
        logger.info("[embedder] initialized model=%s url=%s", model, base_url)

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        logger.info("[embedder] embedding %d text(s) model=%s", len(texts), self._model)
        t0 = time.perf_counter()
        try:
            response = self._client.embed(model=self._model, input=texts)
        except ConnectionError as exc:
            raise ConnectionError(
                "Cannot reach Ollama. Make sure Ollama is running (`ollama serve`) "
                "or switch to the sentence-transformers backend by setting "
                "PDF_EMBEDDING_BACKEND=sentence_transformers in your .env file."
            ) from exc
        elapsed_ms = (time.perf_counter() - t0) * 1000
        embeddings: list[list[float]] = [list(vec) for vec in response.embeddings]
        if self._dimension is None and embeddings:
            self._dimension = len(embeddings[0])
        logger.info("[embedder] done in %.0f ms  dim=%s", elapsed_ms, self._dimension)
        return embeddings

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            probe = self.embed(["probe"])
            self._dimension = len(probe[0])
        return self._dimension
