# RAG System - Interaction Guide

This guide shows all the ways you can interact with the RAG system components.

## Interaction Methods Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    RAG System Components                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   Docling    │  │    FAISS     │  │     LLM      │      │
│  │  Processor   │  │ Vector Store │  │  (OpenAI)    │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│         │                  │                  │              │
│         └──────────────────┴──────────────────┘              │
│                            │                                 │
│                   ┌────────▼────────┐                        │
│                   │  RAG Pipeline   │                        │
│                   └─────────────────┘                        │
│                            │                                 │
└────────────────────────────┼─────────────────────────────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
        ▼                    ▼                    ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  REST API    │    │   CLI Tools  │    │   Jupyter    │
│  (FastAPI)   │    │  (Scripts)   │    │  Notebook    │
└──────────────┘    └──────────────┘    └──────────────┘
        │                    │                    │
        ▼                    ▼                    ▼
    HTTP/cURL           Terminal         Interactive Code
```

## 1. REST API (HTTP Requests)

**Best for:** Web applications, remote access, production use

### Start the API Server

```bash
docker compose -f docker/docker-compose.yml up -d
```

### Interact via HTTP

```bash
# Health check
curl http://localhost:8000/health

# Query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is RAG?", "top_k": 5}'

# Upload document
curl -X POST http://localhost:8000/documents/upload \
  -F "file=@document.pdf"

# Get stats
curl http://localhost:8000/stats

# Rebuild index
curl -X POST http://localhost:8000/index/rebuild
```

### Python HTTP Client

```python
import requests

# Query
response = requests.post(
    "http://localhost:8000/query",
    json={"question": "What is RAG?", "top_k": 5}
)
result = response.json()
print(result['answer'])
```

---

## 2. CLI Tools (Command Line)

**Best for:** Batch operations, automation, server management

### Build Index

```bash
# Using Docker
docker compose -f docker/docker-compose.yml run --rm rag-system \
  python scripts/build_index.py

# Local
python scripts/build_index.py --input-dir data/raw --force-rebuild
```

### Query

```bash
# Using Docker
docker compose -f docker/docker-compose.yml exec rag-system \
  python scripts/query.py "What is machine learning?"

# Local
python scripts/query.py "What is ML?" --top-k 10 --output json
```

### Monitor

```bash
# Using Docker
docker compose -f docker/docker-compose.yml exec rag-system \
  python scripts/monitor.py

# Local
python scripts/monitor.py --output json
```

---

## 3. Jupyter Notebook (Interactive)

**Best for:** Exploration, debugging, experimentation, learning

### Start Jupyter

```bash
# From project root
jupyter notebook notebooks/rag_exploration.ipynb
```

### Interact with Components

```python
# In Jupyter cells:

# 1. Document Processor
from src.document_processor import DocumentProcessor
processor = DocumentProcessor(chunk_size=512)
result = processor.process_document(Path("data/raw/doc.pdf"))

# 2. Embedding Model
from src.embeddings import EmbeddingModel
model = EmbeddingModel("sentence-transformers/all-MiniLM-L6-v2")
embedding = model.encode(["your text here"])

# 3. Vector Store
from src.vector_store import FAISSVectorStore
store = FAISSVectorStore(dimension=384)
store.add_embeddings(embeddings, metadata)
results = store.search(query_embedding, top_k=5)

# 4. Complete RAG Pipeline
from src.rag_pipeline import RAGPipeline
rag = RAGPipeline()
rag.load_index(Path("data/indices/main_index"))
result = rag.query("What is RAG?", top_k=5)

# 5. Direct FAISS Access
import faiss
index = faiss.read_index("data/indices/main_index/index.faiss")
distances, indices = index.search(query_vector, k=5)
```

---

## 4. Python Scripts (Programmatic)

**Best for:** Custom applications, integration, automation

### Direct Usage

```python
from src.rag_pipeline import RAGPipeline
from pathlib import Path

# Initialize
rag = RAGPipeline(
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    llm_model="gpt-4-turbo-preview"
)

# Load index
rag.load_index(Path("data/indices/main_index"))

# Query
result = rag.query("What is RAG?", top_k=5)
print(result.answer)

# Access sources
for source in result.sources:
    print(f"{source['title']}: {source['score']}")
```

### Component-by-Component

```python
# 1. Process documents
from src.document_processor import DocumentProcessor
processor = DocumentProcessor()
docs = processor.process_directory(Path("data/raw"))

# 2. Create embeddings
from src.embeddings import EmbeddingModel
embedder = EmbeddingModel()
all_chunks = [chunk for doc in docs for chunk in doc['chunks']]
embeddings = embedder.encode_batch(all_chunks)

# 3. Build vector store
from src.vector_store import FAISSVectorStore
store = FAISSVectorStore(dimension=embedder.dimension)
store.add_embeddings(embeddings, metadata)
store.save(Path("data/indices/my_index"))

# 4. Search
query_emb = embedder.encode(["my query"])
results = store.search(query_emb, top_k=5)
```

---

## Use Case Matrix

| Task | REST API | CLI | Jupyter | Python Script |
|------|----------|-----|---------|---------------|
| **Query documents** | ✅ Best | ✅ Good | ✅ Good | ✅ Good |
| **Upload documents** | ✅ Best | ❌ | ✅ Good | ✅ Best |
| **Build index** | ✅ Good | ✅ Best | ✅ Good | ✅ Good |
| **Batch queries** | ✅ Good | ✅ Good | ✅ Best | ✅ Best |
| **Explore data** | ❌ | ❌ | ✅ Best | ✅ Good |
| **Debug issues** | ❌ | ⚠️ Limited | ✅ Best | ✅ Good |
| **Production use** | ✅ Best | ✅ Good | ❌ | ✅ Best |
| **Remote access** | ✅ Best | ⚠️ SSH | ❌ | ⚠️ SSH |
| **Web integration** | ✅ Best | ❌ | ❌ | ✅ Good |
| **Automation** | ✅ Good | ✅ Best | ❌ | ✅ Best |
| **Learning** | ⚠️ Limited | ⚠️ Limited | ✅ Best | ✅ Good |

---

## Quick Examples

### Example 1: Query via REST API

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Explain machine learning",
    "top_k": 5
  }'
```

### Example 2: Query via CLI

```bash
python scripts/query.py "Explain machine learning" --top-k 5 --verbose
```

### Example 3: Query via Jupyter

```python
# In Jupyter notebook
from src.rag_pipeline import RAGPipeline
from pathlib import Path

rag = RAGPipeline()
rag.load_index(Path("data/indices/main_index"))

result = rag.query("Explain machine learning", top_k=5)
print(result.answer)

# Visualize sources
import pandas as pd
sources_df = pd.DataFrame(result.sources)
sources_df[['title', 'score', 'excerpt']].head()
```

### Example 4: Query via Python Script

```python
# In your_script.py
from src.rag_pipeline import RAGPipeline
from pathlib import Path
import json

def query_rag(question):
    rag = RAGPipeline()
    rag.load_index(Path("data/indices/main_index"))
    result = rag.query(question, top_k=5)

    return {
        "answer": result.answer,
        "sources": result.sources,
        "query_time_ms": result.query_time_ms
    }

if __name__ == "__main__":
    result = query_rag("Explain machine learning")
    print(json.dumps(result, indent=2))
```

---

## Development Workflow

### Typical Development Cycle

1. **Exploration** (Jupyter)
   - Understand the data
   - Test embedding models
   - Experiment with chunk sizes
   - Debug issues

2. **Implementation** (Python Scripts)
   - Write custom processing logic
   - Build specialized pipelines
   - Create helper functions

3. **Testing** (CLI)
   - Build index from documents
   - Test queries end-to-end
   - Monitor performance

4. **Deployment** (REST API)
   - Start the API server
   - Integrate with applications
   - Serve production traffic

---

## Additional Resources

- **Main Documentation**: [CLAUDE.md](CLAUDE.md)
- **Quick Start**: [README.md](README.md)
- **Interactive Notebook**: [notebooks/rag_exploration.ipynb](notebooks/rag_exploration.ipynb)
- **Simple Example**: [notebooks/simple_example.py](notebooks/simple_example.py)
- **API Documentation**: http://localhost:8000/docs (when server is running)

---

**Need help?** Open the Jupyter notebook for interactive examples and detailed explanations!
