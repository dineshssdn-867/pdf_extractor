"""Ollama LLM service."""

from __future__ import annotations

import logging
import re
import time

import ollama

from pdf_extractor.domain.interfaces import ILLMService

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
logger = logging.getLogger(__name__)


class OllamaLLMService(ILLMService):
    """Generates text using a locally-running Ollama model."""

    def __init__(self, model: str = "llama3.2", base_url: str = "http://localhost:11434") -> None:
        self._model = model
        self._client = ollama.Client(host=base_url)
        logger.info("[llm] initialized model=%s url=%s", model, base_url)

    def generate(self, prompt: str) -> str:
        logger.info("[llm] generate start prompt_len=%d model=%s", len(prompt), self._model)
        t0 = time.perf_counter()
        response = self._client.generate(model=self._model, prompt=prompt)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        text = response.response if hasattr(response, "response") else str(response["response"])
        result = _THINK_RE.sub("", text).strip()
        logger.info("[llm] generate done in %.0f ms  answer_len=%d", elapsed_ms, len(result))
        return result
