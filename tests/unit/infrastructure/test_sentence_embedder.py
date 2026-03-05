"""Unit tests for SentenceTransformerEmbeddingService."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from pdf_extractor.infrastructure.sentence_embedder import SentenceTransformerEmbeddingService


@pytest.fixture()
def mock_st_model():
    with patch("sentence_transformers.SentenceTransformer") as cls:
        model = MagicMock()
        model.device = "cpu"
        model.encode.return_value = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        model.get_sentence_embedding_dimension.return_value = 3
        cls.return_value = model
        yield model


class TestSentenceTransformerEmbeddingService:
    def test_model_loaded_at_init(self, mock_st_model: MagicMock) -> None:
        svc = SentenceTransformerEmbeddingService(model_name="all-MiniLM-L6-v2")
        assert svc._model is mock_st_model

    def test_embed_returns_list_of_vectors(self, mock_st_model: MagicMock) -> None:
        svc = SentenceTransformerEmbeddingService()
        result = svc.embed(["hello", "world"])
        assert len(result) == 2
        assert result[0] == pytest.approx([0.1, 0.2, 0.3])
        assert result[1] == pytest.approx([0.4, 0.5, 0.6])

    def test_embed_empty_list(self, mock_st_model: MagicMock) -> None:
        svc = SentenceTransformerEmbeddingService()
        result = svc.embed([])
        assert result == []
        mock_st_model.encode.assert_not_called()

    def test_embed_calls_encode_with_no_progress_bar(self, mock_st_model: MagicMock) -> None:
        svc = SentenceTransformerEmbeddingService()
        svc.embed(["test"])
        mock_st_model.encode.assert_called_once_with(
            ["test"], convert_to_numpy=True, show_progress_bar=False
        )

    def test_dimension(self, mock_st_model: MagicMock) -> None:
        svc = SentenceTransformerEmbeddingService()
        assert svc.dimension == 3
