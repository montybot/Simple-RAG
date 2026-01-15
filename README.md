# RAG System with Docling, FAISS and LangChain

A professional Retrieval-Augmented Generation (RAG) system for querying document collections.

## Features

- **Document Processing**: PDF, DOCX, HTML, Markdown, CSV files via Docling
- **Vector Search**: Fast similarity search with FAISS
- **Multi-LLM Support**: OpenAI, Anthropic, Mistral AI, Ollama (local)
- **Web Interface**: Streamlit app for interactive queries
- **REST API**: FastAPI server for integration
- **RAG Evaluation**: Quality assessment with RAGAS metrics
- **Docker**: Containerized deployment with UV package manager

## Quick Start

### Prerequisites

- Docker and Docker Compose
- API key for your LLM provider (OpenAI, Mistral, Anthropic) or Ollama for local

### Setup

```bash
# Configure environment
cp .env.example .env
nano .env  # Add your API key

# Build and start
docker compose -f docker/docker-compose.yml build
docker compose -f docker/docker-compose.yml up -d
```

### Add Documents

```bash
# Copy documents to data/raw/
cp /path/to/documents/* data/raw/

# Build the index
docker compose -f docker/docker-compose.yml exec rag-system \
  python scripts/build_index.py
```

### Query

**Via API:**
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Your question here", "top_k": 5}'
```

**Via Streamlit UI:**
```bash
docker compose -f docker/docker-compose.yml exec rag-system \
  streamlit run src/streamlit_app.py --server.port 8501
```

## Project Structure

```
projet_11_rag/
├── docker/                 # Docker configuration
├── src/                    # Source code
│   ├── api.py             # FastAPI server
│   ├── rag_pipeline.py    # RAG pipeline
│   ├── csv_processor.py   # CSV document processing
│   └── streamlit_app.py   # Web interface
├── scripts/               # CLI utilities
├── data/                  # Documents and indices
├── tests/                 # Test suite
└── docs/                  # Documentation
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | System health check |
| `/query` | POST | Query the RAG system |
| `/documents/upload` | POST | Upload and index a document |
| `/index/rebuild` | POST | Rebuild the entire index |
| `/stats` | GET | System statistics |

## Configuration

Set in `.env`:

```bash
# LLM Provider (choose one)
OPENAI_API_KEY=sk-...
MISTRAL_API_KEY=...
ANTHROPIC_API_KEY=...
MODEL_NAME=mistral-small-latest  # or gpt-4-turbo-preview, claude-3-5-sonnet, llama3.2

# Embedding Model
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# RAG Settings
MAX_CHUNK_SIZE=512
TOP_K_RESULTS=5
```

## Testing

```bash
# Run all tests
docker compose -f docker/docker-compose.yml exec rag-system pytest

# Data constraints validation
python tests/test_data_constraints.py

# RAG quality evaluation with RAGAS
python tests/test_rag_evaluation.py
```

## Documentation

- [API Reference](docs/API.md)
- [CSV Processing](docs/CSV_PROCESSING.md)
- [LLM Parameters](docs/LLM_PARAMETERS.md)
- [RAGAS Evaluation](docs/RAGAS_EVALUATION.md)
- [System Prompt Architecture](docs/SYSTEM_PROMPT_ARCHITECTURE.md)

## License

MIT
