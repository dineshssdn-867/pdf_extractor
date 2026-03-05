"""OpenAI LLM service."""

from __future__ import annotations

import logging
import time

from pdf_extractor.domain.interfaces import ILLMService

logger = logging.getLogger(__name__)


class OpenAILLMService(ILLMService):
    """Generates text using the OpenAI Chat Completions API."""

    def __init__(self, api_key: str, model: str = "gpt-4o-mini") -> None:
        from openai import OpenAI
        self._model = model
        self._client = OpenAI(api_key=api_key)
        logger.info("[llm] initialized OpenAI model=%s", model)

    def generate(self, prompt: str) -> str:
        logger.info("[llm] generate start prompt_len=%d model=%s", len(prompt), self._model)
        t0 = time.perf_counter()
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000
        result = response.choices[0].message.content or ""
        logger.info("[llm] generate done in %.0f ms  answer_len=%d", elapsed_ms, len(result))
        return result.strip()
