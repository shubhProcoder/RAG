# AI Document Intelligence Platform

> Production-ready PDF analysis powered by RAG — upload, process, search, and extract intelligent insights from PDF documents.

## Architecture

```
Frontend (SPA)  ──REST──▶  FastAPI  ──▶  Services  ──▶  ChromaDB + SQLite
                                          │
                                    PyMuPDF · Tesseract
                                    pdfplumber · sentence-transformers
                                    OpenAI GPT · InsightsService
```

| Layer | Tech | Purpose |
|-------|------|---------|
| **Frontend** | Vanilla HTML/CSS/JS | Upload UI, document manager, insights viewer, chat |
| **API** | FastAPI + Uvicorn | REST endpoints, SSE streaming, middleware |
| **PDF Engine** | PyMuPDF, pytesseract, pdfplumber | Text extraction, OCR, tables |
| **Chunking** | Recursive splitter | Semantic text segmentation |
| **Embeddings** | `all-MiniLM-L6-v2` | 384-dim dense vectors |
| **Vector DB** | ChromaDB (persistent) | Similarity search |
| **Metadata** | SQLite (aiosqlite) | Document state, insight caching |
| **LLM** | OpenAI GPT-4o-mini | Chat, insight generation |
| **Export** | reportlab, python-docx, openpyxl | PDF, Word, Excel export |

## Features

### Document Intelligence
- **Executive Summary** — concise overview of document content
- **Key Insights** — top findings and takeaways
- **Action Items** — extracted tasks and next steps
- **Risks** — identified warnings and compliance concerns
- **Recommendations** — proposals from the document
- **KPIs & Metrics** — extracted performance indicators
- **Important Dates** — deadlines and milestones
- **Named Entities** — People, Organizations, Locations
- **Financial Metrics** — revenue, costs, budget data
- **Table Insights** — trends from tabular data
- **Multi-document Comparison** — cross-document analysis

### Document Management
- Upload multiple PDF files (drag & drop)
- View uploaded documents with processing status
- Delete documents and all associated data
- **Duplicate detection** — SHA-256 hash prevents re-indexing identical files

### RAG Chat
- Semantic search over indexed documents
- Streaming SSE responses
- Conversation history
- Citations with document name, page number, and source text

### Export
- Export insights as **PDF**, **Word (DOCX)**, or **Excel (XLSX)**
- Export chat conversation history as **TXT**

### Production Features
- Structured logging with request IDs
- Global exception handling with error mapping
- Health checks (metadata store, vector store, embedding model, LLM)
- Docker multi-stage build with non-root user
- Unit tests and integration tests
- CORS middleware, request timing

## Quick Start

### Option 1: Docker (Recommended)

```bash
cp .env.example .env
# Edit .env to add your OPENAI_API_KEY
cd docker
docker compose up --build
```

Open **http://localhost:8000** in your browser.

### Option 2: Bare-Metal

**Prerequisites:**
- Python 3.11+
- (Optional) Tesseract OCR for scanned PDFs
- (Optional) Poppler for `pdf2image`

```bash
# 1. Create virtual environment
cd backend
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure
cp ../.env.example ../.env
# Edit .env to add your OPENAI_API_KEY

# 4. Run
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Open **http://localhost:8000** in your browser.

## API Reference

### Core Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/health` | System health check |
| `POST` | `/api/documents/upload` | Upload PDF files (multipart) |
| `GET` | `/api/documents` | List all documents |
| `GET` | `/api/documents/{id}` | Get document details |
| `DELETE` | `/api/documents/{id}` | Delete document |
| `GET` | `/api/documents/{id}/chunks` | Inspect document chunks |

### Intelligence Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/documents/{id}/insights/{type}` | Get/generate insight |
| `POST` | `/api/documents/compare` | Compare multiple documents |

**Insight types:** `executive_summary`, `key_insights`, `action_items`, `risks`, `recommendations`, `kpi_metrics`, `important_dates`, `named_entities`, `financial_metrics`, `table_insights`

### Chat & Search

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/chat` | Chat with documents (SSE stream) |
| `GET` | `/api/search?query=...` | Semantic search |
| `GET` | `/api/conversations` | List conversations |
| `POST` | `/api/conversations` | Create conversation |
| `GET` | `/api/conversations/{id}` | Get conversation history |

### Export

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/export/insights/{doc_id}/{format}` | Export insights (pdf/word/excel) |
| `GET` | `/api/export/chat/{conv_id}/txt` | Export chat history |

### Upload Example

```bash
curl -X POST http://localhost:8000/api/documents/upload \
  -F "files=@document.pdf"
```

### Generate Insight Example

```bash
curl http://localhost:8000/api/documents/{doc_id}/insights/executive_summary
```

## Running Tests

```bash
cd backend
pytest tests/ -v
```

## Configuration

All settings are configurable via environment variables (see `.env.example`):

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | — | OpenAI API key (required for chat/insights) |
| `LLM_MODEL` | `gpt-4o-mini` | LLM model for generation |
| `MAX_FILE_SIZE_MB` | 50 | Maximum upload size |
| `CHUNK_SIZE` | 1000 | Target chunk size (chars) |
| `CHUNK_OVERLAP` | 200 | Overlap between chunks |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformer model |
| `TOP_K_RETRIEVAL` | 5 | Number of chunks to retrieve |
| `SIMILARITY_THRESHOLD` | 0.5 | Distance threshold for filtering |
| `OCR_ENABLED` | true | Enable Tesseract OCR |
| `LOG_LEVEL` | INFO | Logging verbosity |

## Project Structure

```
├── backend/app/
│   ├── core/             # Domain models, interfaces, exceptions
│   ├── services/         # PDF extraction, chunking, embeddings, insights, chat
│   │   ├── insights_service.py   # Document intelligence generation
│   │   ├── chat_service.py       # RAG chat orchestration
│   │   └── ...
│   ├── storage/          # ChromaDB + SQLite implementations
│   ├── api/routes/       # REST endpoints
│   │   ├── documents.py  # Upload, list, delete, insights
│   │   ├── chat.py       # Chat + conversations
│   │   ├── export.py     # PDF/Word/Excel/TXT export
│   │   └── health.py     # Health checks
│   ├── config.py         # Environment-based configuration
│   ├── dependencies.py   # Dependency injection
│   └── main.py           # FastAPI app factory
├── backend/tests/        # Unit + integration tests
├── frontend/             # Single-page UI
├── docker/               # Containerization
└── data/                 # Runtime data (gitignored)
```

## License

MIT
