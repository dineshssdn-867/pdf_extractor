"""Chainlit UI for PDF Extractor RAG system."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import chainlit as cl

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)

from pdf_extractor.application.ingest_use_case import IngestUseCase
from pdf_extractor.application.query_use_case import QueryUseCase
from pdf_extractor.config.settings import get_settings
from pdf_extractor.infrastructure.chroma_store import ChromaVectorStore
from pdf_extractor.infrastructure.ollama_llm import OllamaLLMService
from pdf_extractor.infrastructure.pdf_loader import PyMuPDFDocumentLoader
from pdf_extractor.infrastructure.text_chunker import SlidingWindowChunker

# Build shared dependencies once at startup — embedder model is loaded here, not per session
_settings = get_settings()
_loader = PyMuPDFDocumentLoader()
_chunker = SlidingWindowChunker(chunk_size=_settings.chunk_size, chunk_overlap=_settings.chunk_overlap)
_vector_store = ChromaVectorStore(persist_path=_settings.chroma_persist_path, collection_name=_settings.chroma_collection_name)
_llm = OllamaLLMService(model=_settings.ollama_model, base_url=_settings.ollama_base_url)

if _settings.embedding_backend == "ollama":
    from pdf_extractor.infrastructure.ollama_embedder import OllamaEmbeddingService
    _embedder = OllamaEmbeddingService(model=_settings.ollama_embed_model, base_url=_settings.ollama_base_url)
else:
    from pdf_extractor.infrastructure.sentence_embedder import SentenceTransformerEmbeddingService
    _embedder = SentenceTransformerEmbeddingService(model_name=_settings.embedding_model)


@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("ingest_use_case", IngestUseCase(_loader, _chunker, _embedder, _vector_store))
    cl.user_session.set("query_use_case", QueryUseCase(_embedder, _vector_store, _llm, _settings))

    await cl.Message(
        content=(
            "Welcome to **PDF Extractor**!\n\n"
            "Upload one or more PDF files to get started, then ask any question about them."
        )
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    # Handle uploaded PDF files
    if message.elements:
        ingest_use_case: IngestUseCase = cl.user_session.get("ingest_use_case")
        results = []

        for element in message.elements:
            if not element.name.lower().endswith(".pdf"):
                await cl.Message(content=f"Skipping `{element.name}` — only PDF files are supported.").send()
                continue

            async with cl.Step(name=f"Ingesting {element.name}") as step:
                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                    tmp.write(element.content)
                    tmp_path = Path(tmp.name)

                try:
                    count = await cl.make_async(ingest_use_case.ingest_file)(tmp_path)
                    step.output = f"Stored {count} chunks"
                    results.append(f"**{element.name}**: {count} chunks ingested")
                except Exception as e:
                    step.output = f"Error: {e}"
                    results.append(f"**{element.name}**: failed — {e}")
                finally:
                    tmp_path.unlink(missing_ok=True)

        summary = "\n".join(results)
        await cl.Message(content=f"Ingestion complete:\n{summary}\n\nYou can now ask questions!").send()
        return

    # Handle text questions
    question = message.content.strip()
    if not question:
        return

    query_use_case: QueryUseCase = cl.user_session.get("query_use_case")

    async with cl.Step(name="Retrieving and generating answer"):
        try:
            result = await cl.make_async(query_use_case.execute)(question)
        except Exception as e:
            await cl.Message(content=f"Error: {e}").send()
            return

    # Build source references
    sources = ""
    if result.source_chunks:
        lines = []
        for rc in result.source_chunks:
            src = rc.chunk.metadata.get("source", rc.chunk.doc_id)
            lines.append(f"- `{src}` page {rc.chunk.page_number}")
        sources = "\n\n**Sources:**\n" + "\n".join(lines)

    await cl.Message(content=result.answer + sources).send()
