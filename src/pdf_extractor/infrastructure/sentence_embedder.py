"""Sentence-Transformers embedding service (alternative embedder)."""

from __future__ import annotations

import logging
import time

from pdf_extractor.domain.interfaces import IEmbeddingService

logger = logging.getLogger(__name__)


class SentenceTransformerEmbeddingService(IEmbeddingService):
    """Generates embeddings using a local sentence-transformers model."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer
        logger.info("[embedder] loading %s ...", model_name)
        t0 = time.perf_counter()
        self._model = SentenceTransformer(model_name)
        logger.info("[embedder] ready in %.0f ms  device=%s", (time.perf_counter() - t0) * 1000, self._model.device)

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        t0 = time.perf_counter()
        vectors = self._model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        logger.info("[embedder] encoded %d text(s) in %.0f ms", len(texts), (time.perf_counter() - t0) * 1000)
        return [v.tolist() for v in vectors]

    @property
    def dimension(self) -> int:
        return int(self._model.get_sentence_embedding_dimension())
