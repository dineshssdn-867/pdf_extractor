# PDF Extractor — RAG-Powered PDF Question Answering

A production-ready Retrieval-Augmented Generation (RAG) system that lets you ingest PDF documents and ask natural language questions about them. Built with Clean Architecture, OpenTelemetry observability, and a Chainlit web UI.

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

Each query emits **4 OpenTelemetry spans**: `retrieve` → `prompt_build` → `llm_generate` → `answer`, visible in Arize Phoenix.

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
| `PDF_OTEL_ENABLED` | `true` | Enable OpenTelemetry tracing |
| `PDF_OTEL_SERVICE_NAME` | `pdf-extractor` | Service name shown in traces |
| `PDF_PHOENIX_COLLECTOR_ENDPOINT` | `http://localhost:4317` | Arize Phoenix OTLP endpoint |

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

## Embedding Backend

The default backend is `sentence_transformers` using `all-MiniLM-L6-v2`, which:
- Runs on GPU automatically if CUDA is available
- Loads once at startup and stays in memory
- Gives significantly better retrieval accuracy than `nomic-embed-text` for Q&A tasks

To switch to Ollama embeddings:

```bash
PDF_EMBEDDING_BACKEND=ollama
PDF_OLLAMA_EMBED_MODEL=nomic-embed-text
```

> **Important:** If you switch embedding backends, you must re-ingest all PDFs. Embeddings from different models are not compatible.

---

## WSL2 Performance Tips

If running on Windows via WSL2:

- Store `chroma_data` on the **Linux filesystem** (e.g. `~/pdf_extractor/chroma_data`), not under `/mnt/c/`. Cross-filesystem I/O adds ~400ms per query.
- Install Ollama via the **official installer** (`curl -fsSL https://ollama.com/install.sh | sh`), not via snap. The snap sandbox blocks GPU access.
- With the above, Ollama runs fully on GPU.

---

## Observability (Arize Phoenix)

Start Phoenix locally:

```bash
pip install arize-phoenix
python -m phoenix.server.main
```

Open `http://localhost:6006` and run a query — you'll see 4 spans per request with timing breakdowns for embed, retrieve, and LLM generate steps.

Timing is also logged to the console at INFO level:

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
│   └── query_use_case.py    # embed → retrieve → prompt → generate (4 OTel spans)
├── infrastructure/
│   ├── pdf_loader.py        # PyMuPDF loader
│   ├── text_chunker.py      # Sliding-window chunker
│   ├── ollama_embedder.py   # Ollama embedding service
│   ├── sentence_embedder.py # Sentence-Transformers embedding service (default)
│   ├── chroma_store.py      # ChromaDB vector store
│   ├── ollama_llm.py        # Ollama LLM service
│   └── observability.py     # OTel tracing setup
├── config/
│   └── settings.py          # AppSettings (Pydantic BaseSettings, absolute .env path)
└── ui/
    ├── chainlit_app.py      # Chainlit web UI (shared deps loaded once at startup)
    └── cli.py               # Click CLI entry point
```
