"""Click CLI: `pdf-extractor ingest` and `pdf-extractor query`."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)

from pdf_extractor.config.settings import get_settings


def _build_dependencies():  # type: ignore[no-untyped-def]
    """Wire up all concrete implementations."""
    settings = get_settings()

    from pdf_extractor.infrastructure.chroma_store import ChromaVectorStore
    from pdf_extractor.infrastructure.ollama_llm import OllamaLLMService
    from pdf_extractor.infrastructure.pdf_loader import PyMuPDFDocumentLoader
    from pdf_extractor.infrastructure.text_chunker import SlidingWindowChunker

    loader = PyMuPDFDocumentLoader()
    chunker = SlidingWindowChunker(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    vector_store = ChromaVectorStore(
        persist_path=settings.chroma_persist_path,
        collection_name=settings.chroma_collection_name,
    )
    if settings.openai_api_key:
        from pdf_extractor.infrastructure.openai_llm import OpenAILLMService
        llm = OpenAILLMService(api_key=settings.openai_api_key, model=settings.openai_model)
    else:
        llm = OllamaLLMService(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
        )

    if settings.embedding_backend == "ollama":
        from pdf_extractor.infrastructure.ollama_embedder import OllamaEmbeddingService
        embedder = OllamaEmbeddingService(
            model=settings.ollama_embed_model,
            base_url=settings.ollama_base_url,
        )
    else:
        from pdf_extractor.infrastructure.sentence_embedder import SentenceTransformerEmbeddingService
        embedder = SentenceTransformerEmbeddingService(model_name=settings.embedding_model)

    return loader, chunker, embedder, vector_store, llm, settings


@click.group()
@click.pass_context
def main(ctx: click.Context) -> None:
    """PDF Extractor — RAG-powered PDF question-answering."""
    ctx.ensure_object(dict)


@main.command()
@click.argument("pdf_path", type=click.Path(exists=False))
def ingest(pdf_path: str) -> None:
    """Ingest a single PDF file or a directory of PDFs.

    PDF_PATH can be a .pdf file or a directory containing .pdf files.
    """
    from pdf_extractor.application.ingest_use_case import IngestUseCase

    loader, chunker, embedder, vector_store, _llm, settings = _build_dependencies()
    use_case = IngestUseCase(loader, chunker, embedder, vector_store)

    path = Path(pdf_path)
    if path.is_dir():
        results = use_case.ingest_directory(path)
        if not results:
            click.echo("No PDF files found.")
            sys.exit(1)
        for name, count in results.items():
            click.echo(f"  {name}: {count} chunks")
        total = sum(results.values())
        click.echo(f"Done. Total chunks stored: {total}")
    elif path.suffix.lower() == ".pdf":
        count = use_case.ingest_file(path)
        click.echo(f"Done. Chunks stored: {count}")
    else:
        # Default: scan configured input_dir
        results = use_case.ingest_directory(settings.input_dir)
        if not results:
            click.echo(f"No PDF files found in {settings.input_dir}.")
            sys.exit(1)
        for name, count in results.items():
            click.echo(f"  {name}: {count} chunks")
        total = sum(results.values())
        click.echo(f"Done. Total chunks stored: {total}")


@main.command()
@click.argument("question")
@click.option("--top-k", default=None, type=int, help="Number of chunks to retrieve.")
def query(question: str, top_k: int | None) -> None:
    """Ask a question about ingested PDFs."""
    from pdf_extractor.application.query_use_case import QueryUseCase
    from pdf_extractor.config.settings import AppSettings

    _loader, _chunker, embedder, vector_store, llm, settings = _build_dependencies()

    if top_k is not None:
        settings = AppSettings(**{**settings.model_dump(), "retrieval_top_k": top_k})

    use_case = QueryUseCase(embedder, vector_store, llm, settings)
    result = use_case.execute(question)

    click.echo(f"\nQuestion: {result.question}")
    click.echo(f"\nAnswer:\n{result.answer}")
    if result.source_chunks:
        click.echo(f"\nSources ({len(result.source_chunks)} chunks):")
        for rc in result.source_chunks:
            src = rc.chunk.metadata.get("source", rc.chunk.doc_id)
            page = rc.chunk.page_number
            click.echo(f"  [{rc.score:.3f}] {src} (page {page})")
