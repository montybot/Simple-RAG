# RAG System with Docling, FAISS and LangChain

A professional Retrieval-Augmented Generation (RAG) system combining Docling for document processing, FAISS for vector search, and LangChain for LLM orchestration.

## Features

- **Document Processing**: Parse PDF, DOCX, HTML, Markdown, and text files using Docling
- **Vector Search**: Fast similarity search with FAISS indexing
- **LLM Integration**: Generate answers using OpenAI or other LLMs via LangChain
- **REST API**: FastAPI-based API for easy integration
- **Docker Support**: Containerized deployment for reproducibility
- **CLI Tools**: Command-line utilities for indexing and querying

## Quick Start

### Prerequisites

- Docker and Docker Compose
- OpenAI API key (or other LLM provider)

### 1. Setup

Clone the repository and configure your environment:

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env and add your API key
nano .env
# Add your OPENAI_API_KEY=sk-...
```

### 2. Build the Docker Image

```bash
docker compose -f docker/docker-compose.yml build
```

### 3. Add Documents

Place your documents in the `data/raw/` directory:

```bash
cp /path/to/your/documents/*.pdf data/raw/
```

### 4. Build the Index

```bash
docker compose -f docker/docker-compose.yml run --rm rag-system \
  python scripts/build_index.py
```

### 5. Start the API Server

```bash
docker compose -f docker/docker-compose.yml up -d
```

### 6. Query the System

#### Using the API:

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is RAG?",
    "top_k": 5
  }'
```

#### Using the CLI:

```bash
docker compose -f docker/docker-compose.yml exec rag-system \
  python scripts/query.py "What is RAG?"
```

## Project Structure

```
projet_11_rag/
├── docker/              # Docker configuration
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── requirements.txt
├── src/                 # Source code
│   ├── config.py        # Configuration management
│   ├── document_processor.py  # Document parsing
│   ├── embeddings.py    # Embedding generation
│   ├── vector_store.py  # FAISS vector store
│   ├── rag_pipeline.py  # RAG pipeline
│   └── api.py          # FastAPI server
├── scripts/            # CLI utilities
│   ├── build_index.py  # Index builder
│   ├── query.py        # Query tool
│   └── monitor.py      # System monitor
├── data/               # Data storage
│   ├── raw/           # Source documents
│   ├── processed/     # Processed documents
│   └── indices/       # FAISS indices
└── tests/             # Test suite
```

## API Endpoints

### Health Check

```bash
GET /health
```

Returns system status and index statistics.

### Query

```bash
POST /query
Content-Type: application/json

{
  "question": "Your question here",
  "top_k": 5
}
```

Returns answer with sources and metadata.

### Upload Document

```bash
POST /documents/upload
Content-Type: multipart/form-data

file: document.pdf
```

Uploads and indexes a new document.

### Rebuild Index

```bash
POST /index/rebuild
```

Rebuilds the entire index from scratch.

### Statistics

```bash
GET /stats
```

Returns detailed system statistics.

## CLI Usage

### Build Index

```bash
# Basic usage
python scripts/build_index.py

# With options
python scripts/build_index.py \
  --input-dir /path/to/docs \
  --output-dir /path/to/index \
  --embedding-model sentence-transformers/all-mpnet-base-v2 \
  --force-rebuild
```

### Query

```bash
# Basic query
python scripts/query.py "What is machine learning?"

# With options
python scripts/query.py "What is ML?" \
  --top-k 10 \
  --output json \
  --verbose
```

### Monitor

```bash
# Display statistics
python scripts/monitor.py

# JSON output
python scripts/monitor.py --output json
```

## Configuration

Edit `.env` file to configure:

```bash
# LLM Configuration
OPENAI_API_KEY=sk-your-key-here
MODEL_NAME=gpt-4-turbo-preview

# Embedding Model
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DIMENSION=384

# FAISS Configuration
FAISS_INDEX_TYPE=IVFFlat
FAISS_NLIST=100
FAISS_NPROBE=10

# Application Settings
LOG_LEVEL=INFO
MAX_CHUNK_SIZE=512
CHUNK_OVERLAP=50
TOP_K_RESULTS=5
```

## Supported Document Formats

- PDF (.pdf)
- Microsoft Word (.docx)
- HTML (.html)
- Markdown (.md)
- Text files (.txt)

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_document_processor.py

# Run with coverage
pytest --cov=src tests/
```

### Local Development (without Docker)

```bash
# Install dependencies
pip install -r docker/requirements.txt

# Run the API server
uvicorn src.api:app --reload --port 8000

# Build index
python scripts/build_index.py --input-dir data/raw

# Query
python scripts/query.py "Your question"
```

## Troubleshooting

### Index Not Found

If you get "Index not found" errors, build the index first:

```bash
python scripts/build_index.py
```

### Out of Memory

Reduce batch size or use a smaller embedding model:

```bash
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

### API Connection Refused

Make sure the Docker container is running:

```bash
docker compose -f docker/docker-compose.yml ps
docker compose -f docker/docker-compose.yml logs
```

## Advanced Usage

For detailed documentation, see [CLAUDE.md](CLAUDE.md).

## License

MIT

## Contributing

Contributions welcome! Please read the contributing guidelines before submitting PRs.
