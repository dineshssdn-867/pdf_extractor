"""Unit tests for OllamaLLMService."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from pdf_extractor.infrastructure.ollama_llm import OllamaLLMService


@pytest.fixture()
def mock_client():
    with patch("pdf_extractor.infrastructure.ollama_llm.ollama.Client") as cls:
        client = MagicMock()
        cls.return_value = client
        client.generate.return_value = {"response": "The answer is 42."}
        yield client


class TestOllamaLLMService:
    def test_generate_returns_string(self, mock_client: MagicMock) -> None:
        svc = OllamaLLMService(model="llama3.2")
        result = svc.generate("What is 6 * 7?")
        assert result == "The answer is 42."

    def test_generate_calls_client_with_model(self, mock_client: MagicMock) -> None:
        svc = OllamaLLMService(model="mistral")
        svc.generate("hello")
        mock_client.generate.assert_called_once_with(model="mistral", prompt="hello")

    def test_custom_base_url(self) -> None:
        with patch("pdf_extractor.infrastructure.ollama_llm.ollama.Client") as cls:
            OllamaLLMService(base_url="http://remote:11434")
            cls.assert_called_once_with(host="http://remote:11434")

    def test_generate_returns_str_type(self, mock_client: MagicMock) -> None:
        svc = OllamaLLMService()
        result = svc.generate("prompt")
        assert isinstance(result, str)
