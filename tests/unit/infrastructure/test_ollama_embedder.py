"""Unit tests for OllamaEmbeddingService."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from pdf_extractor.infrastructure.ollama_embedder import OllamaEmbeddingService


def _make_embed_response(*vecs: list[float]) -> MagicMock:
    resp = MagicMock()
    resp.embeddings = vecs
    return resp


@pytest.fixture()
def mock_client():
    with patch("pdf_extractor.infrastructure.ollama_embedder.ollama.Client") as cls:
        client = MagicMock()
        cls.return_value = client
        client.embed.return_value = _make_embed_response([0.1, 0.2, 0.3])
        yield client


class TestOllamaEmbeddingService:
    def test_embed_single_text(self, mock_client: MagicMock) -> None:
        svc = OllamaEmbeddingService(model="nomic-embed-text")
        result = svc.embed(["hello world"])
        assert len(result) == 1
        assert result[0] == [0.1, 0.2, 0.3]
        mock_client.embed.assert_called_once_with(
            model="nomic-embed-text", input=["hello world"]
        )

    def test_embed_multiple_texts(self, mock_client: MagicMock) -> None:
        mock_client.embed.return_value = _make_embed_response(
            [0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]
        )
        svc = OllamaEmbeddingService()
        result = svc.embed(["a", "b", "c"])
        assert len(result) == 3
        mock_client.embed.assert_called_once()

    def test_embed_empty_list(self, mock_client: MagicMock) -> None:
        svc = OllamaEmbeddingService()
        result = svc.embed([])
        assert result == []
        mock_client.embed.assert_not_called()

    def test_dimension_cached_after_embed(self, mock_client: MagicMock) -> None:
        svc = OllamaEmbeddingService()
        svc.embed(["test"])
        assert svc.dimension == 3

    def test_dimension_probes_if_not_set(self, mock_client: MagicMock) -> None:
        svc = OllamaEmbeddingService()
        dim = svc.dimension
        assert dim == 3
        mock_client.embed.assert_called_once()

    def test_custom_base_url(self) -> None:
        with patch("pdf_extractor.infrastructure.ollama_embedder.ollama.Client") as cls:
            OllamaEmbeddingService(base_url="http://myhost:11434")
            cls.assert_called_once_with(host="http://myhost:11434")
