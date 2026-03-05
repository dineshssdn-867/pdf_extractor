"""ChromaDB vector store implementation."""

from __future__ import annotations

from pathlib import Path

import chromadb
from chromadb.config import Settings

from pdf_extractor.domain.entities import Chunk, RetrievedChunk
from pdf_extractor.domain.interfaces import IVectorStore


class ChromaVectorStore(IVectorStore):
    """Persists chunk embeddings in ChromaDB."""

    def __init__(
        self,
        persist_path: Path | None = None,
        collection_name: str = "pdf_chunks",
    ) -> None:
        if persist_path is not None:
            self._client = chromadb.PersistentClient(
                path=str(persist_path),
                settings=Settings(anonymized_telemetry=False),
            )
        else:
            self._client = chromadb.EphemeralClient(
                settings=Settings(anonymized_telemetry=False),
            )
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    # ------------------------------------------------------------------
    def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if not chunks:
            return
        self._collection.upsert(
            ids=[c.chunk_id for c in chunks],
            embeddings=embeddings,
            documents=[c.text for c in chunks],
            metadatas=[
                {
                    "doc_id": c.doc_id,
                    "char_start": c.char_start,
                    "char_end": c.char_end,
                    "page_number": c.page_number,
                    **c.metadata,
                }
                for c in chunks
            ],
        )

    def query(self, query_embedding: list[float], top_k: int = 5) -> list[RetrievedChunk]:
        result = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        retrieved: list[RetrievedChunk] = []
        ids = result["ids"][0]
        docs = result["documents"][0]
        metas = result["metadatas"][0]
        distances = result["distances"][0]

        for chunk_id, text, meta, dist in zip(ids, docs, metas, distances):
            # Cosine distance → similarity score (clamp to [0,1])
            score = max(0.0, min(1.0, 1.0 - float(dist)))
            chunk = Chunk(
                chunk_id=chunk_id,
                doc_id=str(meta.get("doc_id", "")),
                text=text,
                char_start=int(meta.get("char_start", 0)),
                char_end=int(meta.get("char_end", len(text))),
                page_number=int(meta.get("page_number", 0)),
                metadata={k: str(v) for k, v in meta.items()
                          if k not in ("doc_id", "char_start", "char_end", "page_number")},
            )
            retrieved.append(RetrievedChunk(chunk=chunk, score=score))

        return retrieved

    def delete_collection(self) -> None:
        name = self._collection.name
        self._client.delete_collection(name)
        self._collection = self._client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
        )

    def get_stored_doc_ids(self) -> set[str]:
        count = self._collection.count()
        if count == 0:
            return set()
        result = self._collection.get(include=["metadatas"])
        return {str(m.get("doc_id", "")) for m in result["metadatas"] if m.get("doc_id")}
