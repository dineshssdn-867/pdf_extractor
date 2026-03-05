"""Application settings loaded from environment variables / .env file."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_DEFAULT_PROMPT = (
    "You are a knowledgeable friend, not a formal AI assistant. "
    "Answer in a warm, conversational tone — like you're explaining it to someone over coffee. "
    "Be concise. No bullet points unless the question clearly calls for a list. "
    "Never say 'based on the context', 'the text states', or any AI-sounding preamble. "
    "Just answer naturally as if you already knew this.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n\n"
    "Answer:"
)


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="PDF_",
        env_file=str(Path(__file__).parents[3] / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Paths
    input_dir: Path = Field(default=Path("./data/pdfs"))
    chroma_persist_path: Path = Field(default=Path.home() / "pdf_extractor" / "chroma_data")
    chroma_collection_name: str = Field(default="pdf_chunks")

    # Ollama
    ollama_base_url: str = Field(default="http://localhost:11434")
    ollama_model: str = Field(default="llama3.2")
    ollama_embed_model: str = Field(default="nomic-embed-text")

    # Embedding
    embedding_backend: str = Field(default="sentence_transformers")  # "ollama" | "sentence_transformers"
    embedding_model: str = Field(default="all-MiniLM-L6-v2")  # used for sentence_transformers

    # Chunking
    chunk_size: int = Field(default=512, gt=0)
    chunk_overlap: int = Field(default=64, ge=0)

    # Retrieval
    retrieval_top_k: int = Field(default=5, gt=0)

    # Observability
    phoenix_collector_endpoint: str = Field(default="http://localhost:4317")
    otel_enabled: bool = Field(default=True)
    otel_service_name: str = Field(default="pdf-extractor")

    # RAG prompt template
    rag_prompt_template: str = Field(default=_DEFAULT_PROMPT)


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    """Return cached application settings."""
    return AppSettings()
