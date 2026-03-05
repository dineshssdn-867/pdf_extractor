# PDF Extractor — RAG-Powered PDF Question Answering

A production-ready Retrieval-Augmented Generation (RAG) system that lets you ingest PDF documents and ask natural language questions about them. Built with Clean Architecture and a Chainlit web UI.

---

## Architecture

```
pdf_extractor/
├── domain/          # Entities + interfaces (zero external deps)
├── application/     # Use cases: IngestUseCase, QueryUseCase
├── infrastructure/  # Concrete implementations (Ollama, ChromaDB, PyMuPDF)
├── config/          # Pydantic settings (env-driven)
└── ui/              # Chainlit web UI + Click CLI
```

**Data flow:**

```
Ingest:  PDF → PyMuPDF → SlidingWindowChunker → Embedder → ChromaDB
Query:   Question → Embedder → ChromaDB → Ollama LLM → Answer
```

---

## Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com) running locally (for the LLM)
- NVIDIA GPU recommended (CPU works but is significantly slower)

Pull an LLM model:

```bash
ollama pull qwen3:4b    # or any other model you prefer
```

> Embeddings use `sentence-transformers/all-MiniLM-L6-v2` by default (runs on GPU automatically if available — no Ollama needed for embeddings).

---

## Installation

```bash
git clone <repo-url>
cd pdf_extractor

python3 -m venv .venv
source .venv/bin/activate        # Windows/WSL: source .venv/bin/activate

pip install -e ".[dev]"

cp .env.example .env             # edit as needed
```

---

## Configuration

All settings use the `PDF_` prefix and can be set in `.env` or as environment variables. The `.env` file is resolved by absolute path so it works regardless of which directory you launch from.

| Variable | Default | Description |
|---|---|---|
| `TEAMIFIED_OPENAI_API_KEY` | _(empty)_ | OpenAI API key — if set, OpenAI is used instead of Ollama |
| `PDF_OPENAI_MODEL` | `gpt-4o-mini` | OpenAI model to use when key is present |
| `PDF_INPUT_DIR` | `./data/pdfs` | Default directory scanned by `ingest` |
| `PDF_CHROMA_PERSIST_PATH` | `~/pdf_extractor/chroma_data` | ChromaDB storage path (use Linux FS on WSL2 for best performance) |
| `PDF_CHROMA_COLLECTION_NAME` | `pdf_chunks` | Collection name |
| `PDF_OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `PDF_OLLAMA_MODEL` | `llama3.2` | LLM model for generation |
| `PDF_OLLAMA_EMBED_MODEL` | `nomic-embed-text` | Ollama embedding model (only used when backend is `ollama`) |
| `PDF_EMBEDDING_BACKEND` | `sentence_transformers` | `sentence_transformers` (recommended) or `ollama` |
| `PDF_EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Model used when backend is `sentence_transformers` |
| `PDF_CHUNK_SIZE` | `512` | Characters per chunk |
| `PDF_CHUNK_OVERLAP` | `64` | Overlap between chunks |
| `PDF_RETRIEVAL_TOP_K` | `5` | Chunks retrieved per query |

---

## Usage

### Chainlit Web UI

```bash
chainlit run src/pdf_extractor/ui/chainlit_app.py
```

Open **http://localhost:8000**, upload a PDF, then ask questions about it.

The embedding model is loaded once at startup and shared across all sessions — no reload delay per user.

### Ingest PDFs (CLI)

```bash
# Ingest a single file
pdf-extractor ingest path/to/document.pdf

# Ingest all PDFs in a directory
pdf-extractor ingest path/to/folder/

# Ingest from the configured PDF_INPUT_DIR
pdf-extractor ingest .
```

Re-running ingest on the same files is safe — already-ingested chunks are upserted.

### Ask Questions (CLI)

```bash
pdf-extractor query "What are the key findings?"
pdf-extractor query "When did the EDSA People Power Revolution happen?"

# Override the number of retrieved chunks
pdf-extractor query "Summarise section 3" --top-k 10
```

---

## LLM Backend

The system automatically selects the LLM based on whether an OpenAI API key is present:

| Condition | LLM used |
|---|---|
| `TEAMIFIED_OPENAI_API_KEY` is set | OpenAI (`gpt-4o-mini` by default) |
| Key is empty or absent | Ollama (`qwen3:4b` by default) |

No code change needed — just set the key in `.env` to switch.

### Why qwen3:4b as the Ollama fallback?

`qwen3:4b` is the best locally-runnable model for RAG on constrained hardware (4GB VRAM):

- **Instruction-following** — strictly respects prompt directives like "don't say based on the context", which is critical for natural-sounding RAG answers
- **Multilingual** — handles Filipino/Tagalog historical content better than English-only models like LLaMA
- **4B parameters fit in 4GB VRAM** — runs fully on GPU with no CPU offloading, unlike 7B+ models
- **Strong reasoning-to-size ratio** — outperforms llama3.2:3b on factual Q&A benchmarks despite similar size

---

## Embedding Backend

The default backend is `sentence_transformers` using `all-MiniLM-L6-v2`, which:
- Runs on GPU automatically if CUDA is available
- Loads once at startup and stays in memory
- Gives significantly better retrieval accuracy than `nomic-embed-text` for Q&A tasks

### Why all-MiniLM-L6-v2?

Benchmarked against alternatives on this codebase's Philippine history corpus:

| Model | Query vs relevant chunk | Query vs irrelevant chunk | Gap |
|---|---|---|---|
| `nomic-embed-text` (Ollama) | 0.565 | **0.591** | −0.026 (wrong order) |
| `all-MiniLM-L6-v2` | **0.641** | 0.252 | +0.389 ✅ |

`nomic-embed-text` actually ranked irrelevant chunks *higher* than relevant ones for factual Q&A. `all-MiniLM-L6-v2` was specifically fine-tuned on question-answer pairs (MS MARCO, NLI datasets), making it far better at matching a question to its answer passage rather than just finding thematically similar text.

To switch to Ollama embeddings:

```bash
PDF_EMBEDDING_BACKEND=ollama
PDF_OLLAMA_EMBED_MODEL=nomic-embed-text
```

> **Important:** If you switch embedding backends, you must re-ingest all PDFs. Embeddings from different models are not compatible.

---

## Why ChromaDB?

ChromaDB was chosen as the vector store for several practical reasons:

- **Embedded, zero-infrastructure** — runs in-process with no separate server to manage. Ideal for a local-first RAG tool where spinning up Postgres + pgvector or a hosted Pinecone instance would be overkill.
- **Persistent by default** — a single `persist_directory` setting writes everything to disk. Restarts resume from the same state without re-ingesting.
- **Native metadata filtering** — supports filtering by `source`, `page`, or any custom field at query time without post-processing, which is useful when you want to scope retrieval to a specific document.
- **First-class Python SDK** — the client API maps directly to the domain interfaces (`upsert`, `query`, `delete`), keeping the `chroma_store.py` adapter thin (~80 lines).
- **Upsert semantics** — re-ingesting the same PDF is safe and idempotent; duplicate chunks are overwritten, not duplicated.

Alternatives considered: FAISS (no persistence layer, no metadata filtering), Qdrant (requires a running server), Pinecone (cloud-only, needs an API key).

---

## Why Sliding Window Chunking?

Text is split using a sliding window (configurable `chunk_size` + `chunk_overlap`) rather than sentence or paragraph boundaries for the following reasons:

- **Prevents context loss at boundaries** — a fixed sentence split can cut a key fact in half across two chunks. The overlap (`PDF_CHUNK_OVERLAP`, default 64 chars) guarantees that the tail of one chunk appears at the head of the next, so retrieval never misses a sentence that straddles a boundary.
- **Predictable, uniform chunk sizes** — embedding models have a fixed token limit (~512 tokens for `all-MiniLM-L6-v2`). Fixed-size chunks ensure no single chunk ever exceeds the model's context window, which would silently truncate the text and degrade embedding quality.
- **Works on raw extracted text** — PDF text extracted by PyMuPDF often lacks reliable paragraph/sentence delimiters (especially in scanned or multi-column layouts). Character-based sliding windows are robust to missing whitespace or malformed sentences.
- **Simple and deterministic** — the chunking algorithm is ~20 lines with no NLP dependencies. It produces the same output for the same input every time, making ingestion reproducible and easy to test.

The trade-off is that chunks don't align to semantic boundaries (sentences, paragraphs). For most factual Q&A tasks this doesn't matter — the retriever fetches the top-K chunks by cosine similarity, and the overlap ensures that any given fact appears fully in at least one chunk.

---

## Logging

Timing is logged to the console at INFO level so you can see exactly where time is spent:

```
[query] DONE total=2100 ms  (embed=45 ms, retrieve=12 ms, llm=2040 ms)
```

---

## Running Tests

```bash
pytest                   # all tests + coverage report
pytest tests/unit/       # unit tests only
pytest tests/integration # integration tests only
```

---

## Project Structure

```
src/pdf_extractor/
├── domain/
│   ├── entities.py          # Document, Chunk, RetrievedChunk, QueryResult
│   └── interfaces.py        # IDocumentLoader, ITextChunker, IEmbeddingService, IVectorStore, ILLMService
├── application/
│   ├── ingest_use_case.py   # load → chunk → embed → store
│   └── query_use_case.py    # embed → retrieve → prompt → generate
├── infrastructure/
│   ├── pdf_loader.py        # PyMuPDF loader
│   ├── text_chunker.py      # Sliding-window chunker
│   ├── ollama_embedder.py   # Ollama embedding service
│   ├── sentence_embedder.py # Sentence-Transformers embedding service (default)
│   ├── chroma_store.py      # ChromaDB vector store
│   ├── ollama_llm.py        # Ollama LLM service (local fallback)
│   └── openai_llm.py        # OpenAI LLM service (used when API key is set)
├── config/
│   └── settings.py          # AppSettings (Pydantic BaseSettings, absolute .env path)
└── ui/
    ├── chainlit_app.py      # Chainlit web UI (shared deps loaded once at startup)
    └── cli.py               # Click CLI entry point
```
