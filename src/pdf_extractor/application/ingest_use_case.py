"""IngestUseCase: load → chunk → embed → store."""

from __future__ import annotations

import logging
from pathlib import Path

from pdf_extractor.domain.interfaces import (
    IDocumentLoader,
    IEmbeddingService,
    ITextChunker,
    IVectorStore,
)

logger = logging.getLogger(__name__)


class IngestUseCase:
    """Orchestrates the full PDF ingestion pipeline."""

    def __init__(
        self,
        loader: IDocumentLoader,
        chunker: ITextChunker,
        embedder: IEmbeddingService,
        vector_store: IVectorStore,
    ) -> None:
        self._loader = loader
        self._chunker = chunker
        self._embedder = embedder
        self._vector_store = vector_store

    # ------------------------------------------------------------------
    def ingest_file(self, file_path: Path) -> int:
        """Ingest a single PDF file.  Returns the number of chunks stored."""
        file_path = Path(file_path)
        logger.info("Loading %s", file_path)
        documents = self._loader.load(file_path)
        if not documents:
            logger.warning("No text extracted from %s", file_path)
            return 0

        # Idempotency: skip if all doc_ids already stored
        stored_ids = self._vector_store.get_stored_doc_ids()
        new_docs = [d for d in documents if d.doc_id not in stored_ids]
        if not new_docs:
            logger.info("All pages already ingested for %s — skipping.", file_path)
            return 0

        chunks = self._chunker.chunk(new_docs)
        if not chunks:
            logger.warning("No chunks produced for %s", file_path)
            return 0

        texts = [c.text for c in chunks]
        embeddings = self._embedder.embed(texts)
        self._vector_store.upsert(chunks, embeddings)
        logger.info("Stored %d chunks from %s", len(chunks), file_path)
        return len(chunks)

    def ingest_directory(self, dir_path: Path) -> dict[str, int]:
        """Ingest all *.pdf files in dir_path.  Returns {filename: chunk_count}."""
        dir_path = Path(dir_path)
        results: dict[str, int] = {}
        pdf_files = sorted(dir_path.glob("*.pdf"))
        if not pdf_files:
            logger.warning("No PDF files found in %s", dir_path)
            return results
        for pdf in pdf_files:
            results[pdf.name] = self.ingest_file(pdf)
        return results
