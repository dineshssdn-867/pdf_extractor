"""Sliding-window text chunker."""

from __future__ import annotations

import hashlib

from pdf_extractor.domain.entities import Chunk, Document
from pdf_extractor.domain.interfaces import ITextChunker


class SlidingWindowChunker(ITextChunker):
    """Splits document text using a sliding window with configurable overlap."""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be >= 0")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be < chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, documents: list[Document]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for doc in documents:
            chunks.extend(self._chunk_document(doc))
        return chunks

    def _chunk_document(self, doc: Document) -> list[Chunk]:
        text = doc.text
        if not text:
            return []

        step = self.chunk_size - self.chunk_overlap
        chunks: list[Chunk] = []
        start = 0

        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk_text = text[start:end]

            chunk_id = hashlib.sha256(
                f"{doc.doc_id}:{start}:{end}".encode()
            ).hexdigest()[:16]

            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    doc_id=doc.doc_id,
                    text=chunk_text,
                    char_start=start,
                    char_end=end,
                    page_number=doc.page_number,
                    metadata=doc.metadata,
                )
            )

            if end == len(text):
                break
            start += step

        return chunks
