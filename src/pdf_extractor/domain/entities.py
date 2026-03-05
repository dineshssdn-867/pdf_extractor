"""Domain entities — frozen dataclasses with zero external dependencies."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Document:
    """Represents a loaded PDF document."""

    doc_id: str          # hash of file_path + mtime
    file_path: Path
    text: str
    page_number: int = 0
    metadata: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.doc_id:
            raise ValueError("doc_id must not be empty")
        if self.page_number < 0:
            raise ValueError("page_number must be >= 0")


@dataclass(frozen=True)
class Chunk:
    """A text chunk derived from a Document."""

    chunk_id: str        # unique id for this chunk
    doc_id: str
    text: str
    char_start: int
    char_end: int
    page_number: int = 0
    metadata: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.text:
            raise ValueError("Chunk text must not be empty")
        if self.char_start < 0:
            raise ValueError("char_start must be >= 0")
        if self.char_end <= self.char_start:
            raise ValueError("char_end must be > char_start")


@dataclass(frozen=True)
class RetrievedChunk:
    """A chunk returned by a vector store query, with similarity score."""

    chunk: Chunk
    score: float

    def __post_init__(self) -> None:
        if not (0.0 <= self.score <= 1.0):
            raise ValueError(f"score must be in [0, 1], got {self.score}")


@dataclass(frozen=True)
class QueryResult:
    """The final result of a RAG query."""

    question: str
    answer: str
    source_chunks: list[RetrievedChunk]

    def __post_init__(self) -> None:
        if not self.question:
            raise ValueError("question must not be empty")
