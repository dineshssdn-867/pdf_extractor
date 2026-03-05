"""Integration tests for ChromaVectorStore using real EphemeralClient."""

from __future__ import annotations

import pytest

from pdf_extractor.domain.entities import Chunk
from pdf_extractor.infrastructure.chroma_store import ChromaVectorStore


@pytest.fixture()
def store() -> ChromaVectorStore:
    import uuid
    return ChromaVectorStore(persist_path=None, collection_name=f"test_{uuid.uuid4().hex}")


def _make_chunk(chunk_id: str, doc_id: str = "d1", text: str = "hello") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        text=text,
        char_start=0,
        char_end=len(text),
        page_number=0,
    )


class TestChromaVectorStore:
    def test_upsert_and_query(self, store: ChromaVectorStore) -> None:
        chunk = _make_chunk("c1", text="The quick brown fox")
        embedding = [0.1, 0.2, 0.3]
        store.upsert([chunk], [embedding])

        results = store.query([0.1, 0.2, 0.3], top_k=1)
        assert len(results) == 1
        assert results[0].chunk.chunk_id == "c1"

    def test_score_in_range(self, store: ChromaVectorStore) -> None:
        chunk = _make_chunk("c1", text="test")
        store.upsert([chunk], [[1.0, 0.0, 0.0]])
        results = store.query([1.0, 0.0, 0.0], top_k=1)
        assert 0.0 <= results[0].score <= 1.0

    def test_query_empty_store_returns_empty(self, store: ChromaVectorStore) -> None:
        results = store.query([0.1, 0.2, 0.3], top_k=5)
        assert results == []

    def test_upsert_idempotent(self, store: ChromaVectorStore) -> None:
        chunk = _make_chunk("c1", text="hello")
        emb = [0.5, 0.5, 0.5]
        store.upsert([chunk], [emb])
        store.upsert([chunk], [emb])  # second upsert should not raise
        results = store.query(emb, top_k=5)
        assert len(results) == 1  # still only one entry

    def test_top_k_respected(self, store: ChromaVectorStore) -> None:
        chunks = [_make_chunk(f"c{i}", text=f"text {i}") for i in range(5)]
        embeddings = [[float(i), 0.0, 0.0] for i in range(5)]
        store.upsert(chunks, embeddings)
        results = store.query([0.0, 0.0, 0.0], top_k=3)
        assert len(results) <= 3

    def test_delete_collection_clears_data(self, store: ChromaVectorStore) -> None:
        chunk = _make_chunk("c1", text="data")
        store.upsert([chunk], [[0.1, 0.2, 0.3]])
        store.delete_collection()
        results = store.query([0.1, 0.2, 0.3], top_k=5)
        assert results == []

    def test_get_stored_doc_ids(self, store: ChromaVectorStore) -> None:
        c1 = _make_chunk("c1", doc_id="doc-a", text="first")
        c2 = _make_chunk("c2", doc_id="doc-b", text="second")
        store.upsert([c1, c2], [[0.1, 0.0, 0.0], [0.0, 0.1, 0.0]])
        ids = store.get_stored_doc_ids()
        assert ids == {"doc-a", "doc-b"}

    def test_get_stored_doc_ids_empty(self, store: ChromaVectorStore) -> None:
        assert store.get_stored_doc_ids() == set()

    def test_upsert_empty_list_no_error(self, store: ChromaVectorStore) -> None:
        store.upsert([], [])  # should not raise
