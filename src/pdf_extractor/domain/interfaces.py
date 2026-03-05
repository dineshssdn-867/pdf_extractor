"""Abstract interfaces for all infrastructure services."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from pdf_extractor.domain.entities import Chunk, Document, RetrievedChunk


class IDocumentLoader(ABC):
    """Loads raw text from a file into Document objects."""

    @abstractmethod
    def load(self, file_path: Path) -> list[Document]:
        """Return one Document per page (or one per file)."""


class ITextChunker(ABC):
    """Splits Documents into smaller Chunk objects."""

    @abstractmethod
    def chunk(self, documents: list[Document]) -> list[Chunk]:
        """Return chunks from the given documents."""


class IEmbeddingService(ABC):
    """Converts text strings to embedding vectors."""

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per input text."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Dimensionality of the embedding vectors."""


class IVectorStore(ABC):
    """Stores and retrieves chunk embeddings."""

    @abstractmethod
    def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """Insert or update chunks with their embeddings."""

    @abstractmethod
    def query(self, query_embedding: list[float], top_k: int = 5) -> list[RetrievedChunk]:
        """Return the top_k most similar chunks."""

    @abstractmethod
    def delete_collection(self) -> None:
        """Delete the entire collection (useful for tests / reset)."""

    @abstractmethod
    def get_stored_doc_ids(self) -> set[str]:
        """Return all doc_ids currently stored in the collection."""


class ILLMService(ABC):
    """Generates text completions from a prompt."""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Return the generated text for the given prompt."""
