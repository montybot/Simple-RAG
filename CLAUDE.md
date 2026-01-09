# Système RAG avec Docling, FAISS et LangChain

## Table des matières

1. [Vue d'ensemble](#vue-densemble)
2. [Architecture du système](#architecture-du-système)
3. [Prérequis](#prérequis)
4. [Structure du projet](#structure-du-projet)
5. [Configuration Docker](#configuration-docker)
6. [Installation et déploiement](#installation-et-déploiement)
7. [Utilisation](#utilisation)
8. [Guide de développement](#guide-de-développement)
9. [Optimisations et bonnes pratiques](#optimisations-et-bonnes-pratiques)
10. [Dépannage](#dépannage)
11. [Références](#références)

---

## Vue d'ensemble

Ce projet implémente un système RAG (Retrieval-Augmented Generation) professionnel combinant :

- **Docling** : Traitement et extraction de contenu depuis des documents complexes (PDF, DOCX, HTML, etc.)
- **FAISS** : Indexation et recherche vectorielle ultra-rapide pour la récupération de contexte
- **LangChain** : Orchestration des pipelines RAG et intégration avec les LLMs
- **Docker** : Conteneurisation pour un déploiement reproductible et scalable

### Cas d'usage

- Interrogation de bases documentaires volumineuses
- Analyse et synthèse de documents techniques
- Assistant conversationnel basé sur des connaissances spécifiques
- Recherche sémantique dans des corpus métier

---

## Architecture du système

```
┌─────────────────────────────────────────────────────────────┐
│                    Container Docker                          │
│                                                              │
│  ┌──────────────┐     ┌──────────────┐     ┌─────────────┐ │
│  │   Docling    │────▶│  LangChain   │────▶│   FastAPI   │ │
│  │  (Parsing)   │     │ (Orchestration)     │   (API)     │ │
│  └──────────────┘     └──────────────┘     └─────────────┘ │
│         │                     │                     │        │
│         ▼                     ▼                     ▼        │
│  ┌──────────────┐     ┌──────────────┐     ┌─────────────┐ │
│  │  Documents   │     │ Embeddings   │     │    FAISS    │ │
│  │   /data      │     │   Model      │     │   Index     │ │
│  └──────────────┘     └──────────────┘     └─────────────┘ │
│                                                              │
└─────────────────────────────────────────────────────────────┘
         │                      │                     │
         ▼                      ▼                     ▼
   Source Docs            Vector Store          Query Results
```

### Flux de données

1. **Ingestion** : Docling parse les documents sources → extraction de texte structuré
2. **Embedding** : Conversion du texte en vecteurs via un modèle d'embeddings
3. **Indexation** : Stockage des vecteurs dans FAISS pour recherche rapide
4. **Requête** : Recherche de similarité → récupération du contexte pertinent
5. **Génération** : LLM génère une réponse basée sur le contexte récupéré

---

## Prérequis

### Logiciels requis

- **Docker** ≥ 24.0
- **Docker Compose** ≥ 2.20
- **Python** ≥ 3.12 (pour développement local)
- **Git** (pour clonage et versioning)

### Ressources système recommandées

- **RAM** : 8 GB minimum, 16 GB recommandé
- **CPU** : 4 cœurs minimum
- **Stockage** : 20 GB d'espace libre
- **GPU** : Optionnel mais recommandé pour les embeddings (CUDA compatible)

### Clés API (optionnel)

- Clé OpenAI, Anthropic, ou autre fournisseur LLM
- Stockée dans le fichier `.env` (jamais commitée)

---

## Structure du projet

```
projet_11_rag/
├── CLAUDE.md                      # Ce fichier
├── README.md                      # Documentation utilisateur
├── pyproject.toml                 # Configuration Python/UV
├── .python-version                # Version Python
├── .gitignore                     # Fichiers ignorés par Git
├── .env.example                   # Template des variables d'environnement
├── .env                          # Variables d'environnement (non versionné)
│
├── docker/
│   ├── Dockerfile                # Image Docker principale
│   ├── docker-compose.yml        # Orchestration des services
│   └── requirements.txt          # Dépendances Python
│
├── src/
│   ├── __init__.py
│   ├── config.py                 # Configuration centralisée
│   ├── document_processor.py    # Module Docling
│   ├── embeddings.py            # Gestion des embeddings
│   ├── vector_store.py          # Interface FAISS
│   ├── rag_pipeline.py          # Pipeline RAG complet
│   └── api.py                   # API REST (FastAPI)
│
├── data/
│   ├── raw/                     # Documents sources
│   ├── processed/               # Documents parsés
│   └── indices/                 # Index FAISS persistants
│
├── notebooks/
│   └── exploration.ipynb        # Expérimentation interactive
│
├── tests/
│   ├── test_document_processor.py
│   ├── test_embeddings.py
│   └── test_rag_pipeline.py
│
└── scripts/
    ├── build_index.py           # Construction de l'index
    ├── query.py                 # Script de requête CLI
    └── monitor.py               # Monitoring et métriques
```

---

## Configuration Docker

### Dockerfile

```dockerfile
# docker/Dockerfile
FROM python:3.12-slim

# Variables d'environnement
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_SYSTEM_PYTHON=1

# Installation des dépendances système
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    poppler-utils \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# Installation de UV (ultra-fast Python package installer)
# UV est 10-100x plus rapide que pip
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:${PATH}"

# Création du répertoire de travail
WORKDIR /app

# Copie des dépendances
COPY docker/requirements.txt .

# Installation des packages Python avec UV (10-100x plus rapide que pip)
RUN uv pip install -r requirements.txt

# Copie du code source
COPY src/ ./src/
COPY scripts/ ./scripts/

# Création des répertoires de données
RUN mkdir -p /app/data/raw /app/data/processed /app/data/indices

# Exposition du port API
EXPOSE 8000

# Point d'entrée
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
```

### docker-compose.yml

```yaml
# docker/docker-compose.yml
version: '3.8'

services:
  rag-system:
    build:
      context: ..
      dockerfile: docker/Dockerfile
    container_name: rag_system
    ports:
      - "8000:8000"
    volumes:
      - ../data:/app/data
      - ../src:/app/src
      - faiss-index:/app/data/indices
    environment:
      - EMBEDDING_MODEL=${EMBEDDING_MODEL:-sentence-transformers/all-MiniLM-L6-v2}
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - FAISS_INDEX_TYPE=${FAISS_INDEX_TYPE:-IVFFlat}
      - LOG_LEVEL=${LOG_LEVEL:-INFO}
    env_file:
      - ../.env
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  # Service optionnel : interface web
  web-ui:
    image: node:18-alpine
    container_name: rag_web_ui
    working_dir: /app
    ports:
      - "3000:3000"
    volumes:
      - ../web:/app
    command: npm run dev
    depends_on:
      - rag-system

volumes:
  faiss-index:
    driver: local
```

### requirements.txt

```txt
# docker/requirements.txt

# Core dependencies
langchain==0.2.0
langchain-community==0.2.0
langchain-openai==0.1.0

# Document processing
docling==1.5.0
pypdf==4.0.0
python-docx==1.1.0
beautifulsoup4==4.12.0

# Vector store
faiss-cpu==1.8.0  # Use faiss-gpu for GPU support
sentence-transformers==2.3.0

# API and server
fastapi==0.110.0
uvicorn[standard]==0.27.0
pydantic==2.6.0
pydantic-settings==2.2.0

# Utilities
python-dotenv==1.0.1
loguru==0.7.2
tqdm==4.66.0
numpy==1.26.0
pandas==2.2.0

# Testing (development)
pytest==8.0.0
pytest-asyncio==0.23.0
httpx==0.27.0
```

### .env.example

```bash
# .env.example - Copier vers .env et remplir les valeurs

# LLM Configuration
OPENAI_API_KEY=sk-your-key-here
ANTHROPIC_API_KEY=
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

# API Settings
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4
```

---

## Installation et déploiement

### 1. Clonage et configuration initiale

```bash
# Cloner le dépôt
git clone <repository-url>
cd projet_11_rag

# Créer le fichier de configuration
cp .env.example .env

# Éditer .env avec vos clés API
nano .env
```

### 2. Build de l'image Docker

```bash
# Construction de l'image
docker compose -f docker/docker-compose.yml build

# Vérification
docker images | grep rag
```

### 3. Préparation des données

```bash
# Placer vos documents dans le répertoire data/raw/
cp /path/to/your/documents/* data/raw/

# Vérifier les permissions
chmod -R 755 data/
```

### 4. Lancement du système

```bash
# Démarrage des services
docker compose -f docker/docker-compose.yml up -d

# Vérification des logs
docker compose -f docker/docker-compose.yml logs -f

# Vérification du statut
docker compose -f docker/docker-compose.yml ps
```

### 5. Construction de l'index initial

```bash
# Exécution du script d'indexation
docker compose -f docker/docker-compose.yml exec rag-system \
  python scripts/build_index.py --input-dir /app/data/raw
```

### 6. Test de l'API

```bash
# Healthcheck
curl http://localhost:8000/health

# Test de requête
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Quelle est la capitale de la France?"}'
```

---

## Utilisation

### API REST

#### Endpoint : Interrogation (Query)

```bash
POST /query
Content-Type: application/json

{
  "question": "Expliquez le concept de RAG",
  "top_k": 5,
  "filter": {
    "document_type": "pdf"
  }
}

# Réponse
{
  "answer": "Le RAG (Retrieval-Augmented Generation)...",
  "sources": [
    {
      "document": "introduction_rag.pdf",
      "page": 3,
      "score": 0.92,
      "excerpt": "..."
    }
  ],
  "metadata": {
    "query_time_ms": 245,
    "documents_searched": 1523
  }
}
```

#### Endpoint : Indexation d'un document

```bash
POST /documents/upload
Content-Type: multipart/form-data

file: document.pdf
metadata: {"category": "technical", "author": "John Doe"}

# Réponse
{
  "document_id": "doc_abc123",
  "status": "indexed",
  "chunks_created": 47
}
```

#### Endpoint : Santé du système

```bash
GET /health

# Réponse
{
  "status": "healthy",
  "index_size": 1523,
  "last_update": "2026-01-08T10:30:00Z"
}
```

### CLI (Command Line Interface)

```bash
# Requête simple
docker compose exec rag-system python scripts/query.py \
  "Quels sont les avantages de Docker?"

# Requête avec options
docker compose exec rag-system python scripts/query.py \
  "Expliquez FAISS" \
  --top-k 10 \
  --output json \
  --verbose

# Re-construction complète de l'index
docker compose exec rag-system python scripts/build_index.py \
  --input-dir /app/data/raw \
  --force-rebuild \
  --embedding-model sentence-transformers/all-mpnet-base-v2
```

### SDK Python (usage programmatique)

```python
from src.rag_pipeline import RAGPipeline

# Initialisation
rag = RAGPipeline(
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    llm_model="gpt-4-turbo-preview"
)

# Chargement de l'index
rag.load_index("data/indices/main_index")

# Requête
result = rag.query(
    question="Qu'est-ce que LangChain?",
    top_k=5
)

print(result.answer)
for source in result.sources:
    print(f"- {source.document} (score: {source.score:.2f})")
```

---

## Guide de développement

### Module : document_processor.py

```python
# src/document_processor.py
from docling.document_converter import DocumentConverter
from pathlib import Path
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Processeur de documents utilisant Docling."""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.converter = DocumentConverter()
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def process_document(self, file_path: Path) -> Dict:
        """
        Parse un document et extrait son contenu structuré.

        Args:
            file_path: Chemin vers le document source

        Returns:
            Dict contenant le texte, métadonnées et chunks
        """
        logger.info(f"Processing document: {file_path}")

        # Conversion avec Docling
        result = self.converter.convert(str(file_path))

        # Extraction du texte
        full_text = result.document.export_to_markdown()

        # Chunking intelligent
        chunks = self._create_chunks(full_text)

        return {
            "file_path": str(file_path),
            "text": full_text,
            "chunks": chunks,
            "metadata": {
                "title": result.document.name,
                "page_count": len(result.document.pages),
                "format": file_path.suffix
            }
        }

    def _create_chunks(self, text: str) -> List[str]:
        """Découpe le texte en chunks avec overlap."""
        chunks = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start += self.chunk_size - self.chunk_overlap

        return chunks

    def process_directory(self, directory: Path) -> List[Dict]:
        """Traite tous les documents d'un répertoire."""
        results = []

        for file_path in directory.rglob("*"):
            if file_path.is_file() and file_path.suffix in ['.pdf', '.docx', '.txt', '.html']:
                try:
                    result = self.process_document(file_path)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {e}")

        return results
```

### Module : vector_store.py

```python
# src/vector_store.py
import faiss
import numpy as np
from pathlib import Path
from typing import List, Tuple
import pickle
import logging

logger = logging.getLogger(__name__)

class FAISSVectorStore:
    """Gestionnaire d'index FAISS pour la recherche vectorielle."""

    def __init__(self, dimension: int = 384, index_type: str = "Flat"):
        self.dimension = dimension
        self.index_type = index_type
        self.index = None
        self.documents = []
        self._init_index()

    def _init_index(self):
        """Initialise l'index FAISS."""
        if self.index_type == "Flat":
            self.index = faiss.IndexFlatL2(self.dimension)
        elif self.index_type == "IVFFlat":
            quantizer = faiss.IndexFlatL2(self.dimension)
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, 100)
        else:
            raise ValueError(f"Unknown index type: {self.index_type}")

        logger.info(f"Initialized {self.index_type} index with dimension {self.dimension}")

    def add_embeddings(self, embeddings: np.ndarray, documents: List[Dict]):
        """
        Ajoute des embeddings à l'index.

        Args:
            embeddings: Array numpy de shape (n, dimension)
            documents: Liste de métadonnées associées
        """
        # Training pour IVF index
        if self.index_type == "IVFFlat" and not self.index.is_trained:
            logger.info("Training IVF index...")
            self.index.train(embeddings)

        # Ajout des vecteurs
        self.index.add(embeddings)
        self.documents.extend(documents)

        logger.info(f"Added {len(embeddings)} embeddings. Total: {self.index.ntotal}")

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[Dict, float]]:
        """
        Recherche les k documents les plus similaires.

        Args:
            query_embedding: Vecteur de requête de shape (1, dimension)
            top_k: Nombre de résultats à retourner

        Returns:
            Liste de tuples (document, score)
        """
        if self.index.ntotal == 0:
            logger.warning("Index is empty")
            return []

        # Recherche
        distances, indices = self.index.search(query_embedding, top_k)

        # Construction des résultats
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.documents):
                results.append((self.documents[idx], float(dist)))

        return results

    def save(self, path: Path):
        """Sauvegarde l'index et les métadonnées."""
        path.mkdir(parents=True, exist_ok=True)

        # Sauvegarde FAISS
        faiss.write_index(self.index, str(path / "index.faiss"))

        # Sauvegarde métadonnées
        with open(path / "documents.pkl", "wb") as f:
            pickle.dump(self.documents, f)

        logger.info(f"Saved index to {path}")

    def load(self, path: Path):
        """Charge un index existant."""
        # Chargement FAISS
        self.index = faiss.read_index(str(path / "index.faiss"))

        # Chargement métadonnées
        with open(path / "documents.pkl", "rb") as f:
            self.documents = pickle.load(f)

        logger.info(f"Loaded index from {path}. Total: {self.index.ntotal}")
```

### Module : rag_pipeline.py

```python
# src/rag_pipeline.py
from pathlib import Path
from typing import List, Dict
from dataclasses import dataclass

from .document_processor import DocumentProcessor
from .embeddings import EmbeddingModel
from .vector_store import FAISSVectorStore

import logging

logger = logging.getLogger(__name__)

@dataclass
class RAGResult:
    """Résultat d'une requête RAG."""
    answer: str
    sources: List[Dict]
    query_time_ms: float

class RAGPipeline:
    """Pipeline RAG complet."""

    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        llm_model: str = "gpt-4-turbo-preview",
        chunk_size: int = 512
    ):
        self.document_processor = DocumentProcessor(chunk_size=chunk_size)
        self.embedding_model = EmbeddingModel(model_name=embedding_model)
        self.vector_store = FAISSVectorStore(dimension=self.embedding_model.dimension)
        self.llm_model = llm_model

        logger.info("RAG Pipeline initialized")

    def index_documents(self, input_dir: Path):
        """Indexe tous les documents d'un répertoire."""
        import time
        start = time.time()

        # 1. Traitement des documents
        logger.info(f"Processing documents from {input_dir}")
        processed_docs = self.document_processor.process_directory(input_dir)

        # 2. Création des embeddings
        all_chunks = []
        all_metadata = []

        for doc in processed_docs:
            for chunk in doc["chunks"]:
                all_chunks.append(chunk)
                all_metadata.append({
                    "file_path": doc["file_path"],
                    "title": doc["metadata"]["title"],
                    "chunk_text": chunk
                })

        logger.info(f"Creating embeddings for {len(all_chunks)} chunks")
        embeddings = self.embedding_model.encode_batch(all_chunks)

        # 3. Ajout à l'index
        self.vector_store.add_embeddings(embeddings, all_metadata)

        elapsed = time.time() - start
        logger.info(f"Indexing completed in {elapsed:.2f}s")

    def query(self, question: str, top_k: int = 5) -> RAGResult:
        """
        Exécute une requête RAG.

        Args:
            question: Question de l'utilisateur
            top_k: Nombre de documents à récupérer

        Returns:
            RAGResult avec réponse et sources
        """
        import time
        start = time.time()

        # 1. Embedding de la question
        query_embedding = self.embedding_model.encode([question])

        # 2. Recherche dans l'index
        results = self.vector_store.search(query_embedding, top_k=top_k)

        # 3. Construction du contexte
        context = self._build_context(results)

        # 4. Génération de la réponse
        answer = self._generate_answer(question, context)

        # 5. Formatage des sources
        sources = [
            {
                "file": res[0]["file_path"],
                "title": res[0]["title"],
                "score": res[1],
                "excerpt": res[0]["chunk_text"][:200] + "..."
            }
            for res in results
        ]

        elapsed = (time.time() - start) * 1000

        return RAGResult(
            answer=answer,
            sources=sources,
            query_time_ms=elapsed
        )

    def _build_context(self, results: List[Tuple]) -> str:
        """Construit le contexte depuis les résultats."""
        context_parts = []
        for i, (doc, score) in enumerate(results, 1):
            context_parts.append(f"[Document {i}]\n{doc['chunk_text']}\n")
        return "\n".join(context_parts)

    def _generate_answer(self, question: str, context: str) -> str:
        """Génère une réponse via LLM."""
        from langchain_openai import ChatOpenAI
        from langchain.prompts import PromptTemplate

        prompt = PromptTemplate(
            template="""Utilise le contexte suivant pour répondre à la question.
Si tu ne trouves pas la réponse dans le contexte, dis-le clairement.

Contexte:
{context}

Question: {question}

Réponse:""",
            input_variables=["context", "question"]
        )

        llm = ChatOpenAI(model=self.llm_model, temperature=0)
        chain = prompt | llm

        result = chain.invoke({"context": context, "question": question})
        return result.content

    def save_index(self, path: Path):
        """Sauvegarde l'index."""
        self.vector_store.save(path)

    def load_index(self, path: Path):
        """Charge un index existant."""
        self.vector_store.load(path)
```

### Module : api.py

```python
# src/api.py
from fastapi import FastAPI, UploadFile, HTTPException
from pydantic import BaseModel
from pathlib import Path
import logging

from .rag_pipeline import RAGPipeline
from .config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

app = FastAPI(
    title="RAG System API",
    description="API pour système RAG avec Docling, FAISS et LangChain",
    version="1.0.0"
)

# Initialisation du pipeline
rag_pipeline = RAGPipeline(
    embedding_model=settings.embedding_model,
    llm_model=settings.model_name
)

# Chargement de l'index au démarrage
index_path = Path("/app/data/indices/main_index")
if index_path.exists():
    rag_pipeline.load_index(index_path)
    logger.info("Index loaded successfully")

class QueryRequest(BaseModel):
    question: str
    top_k: int = 5

class QueryResponse(BaseModel):
    answer: str
    sources: list
    metadata: dict

@app.get("/health")
async def health_check():
    """Endpoint de santé."""
    return {
        "status": "healthy",
        "index_size": rag_pipeline.vector_store.index.ntotal if rag_pipeline.vector_store.index else 0
    }

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Endpoint de requête RAG."""
    try:
        result = rag_pipeline.query(
            question=request.question,
            top_k=request.top_k
        )

        return QueryResponse(
            answer=result.answer,
            sources=result.sources,
            metadata={
                "query_time_ms": result.query_time_ms,
                "documents_searched": len(result.sources)
            }
        )
    except Exception as e:
        logger.error(f"Query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/documents/upload")
async def upload_document(file: UploadFile):
    """Upload et indexation d'un nouveau document."""
    try:
        # Sauvegarde du fichier
        file_path = Path("/app/data/raw") / file.filename
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        # Traitement et indexation
        rag_pipeline.index_documents(file_path.parent)

        # Sauvegarde de l'index mis à jour
        rag_pipeline.save_index(index_path)

        return {
            "status": "success",
            "document_id": file.filename,
            "message": "Document indexed successfully"
        }
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

---

## Optimisations et bonnes pratiques

### 1. Performance FAISS

```python
# Pour des corpus > 100k documents, utiliser un index IVF
index = faiss.IndexIVFFlat(quantizer, dimension, nlist=1000)

# Avec GPU (nécessite faiss-gpu)
res = faiss.StandardGpuResources()
gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)

# Optimisation de la recherche
index.nprobe = 10  # Compromise précision/vitesse
```

### 2. Chunking intelligent

```python
# Utiliser RecursiveCharacterTextSplitter de LangChain
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " ", ""]
)
```

### 3. Caching des embeddings

```python
from functools import lru_cache
import hashlib

@lru_cache(maxsize=1000)
def cached_embed(text: str):
    return embedding_model.encode([text])[0]
```

### 4. Monitoring et logging

```python
# src/config.py
from loguru import logger
import sys

logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
)
logger.add("/app/logs/rag_{time}.log", rotation="500 MB")
```

### 5. Gestion de la mémoire

```python
# Batch processing pour grands corpus
def process_in_batches(items, batch_size=100):
    for i in range(0, len(items), batch_size):
        batch = items[i:i+batch_size]
        yield batch

for batch in process_in_batches(documents, batch_size=100):
    embeddings = model.encode(batch)
    vector_store.add(embeddings)
```

---

## Dépannage

### Problème : "Out of Memory" lors de l'indexation

**Solution :**
```python
# Réduire la taille des batchs
BATCH_SIZE = 50  # Au lieu de 100

# Activer le garbage collector
import gc
gc.collect()
```

### Problème : Index FAISS corrompu

**Solution :**
```bash
# Supprimer et reconstruire l'index
rm -rf data/indices/*
docker compose exec rag-system python scripts/build_index.py --force-rebuild
```

### Problème : Embeddings de mauvaise qualité

**Solution :**
```python
# Utiliser un modèle plus performant
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"  # 768 dim
# Ou pour le français
EMBEDDING_MODEL = "dangvantuan/sentence-camembert-large"
```

### Problème : API lente

**Diagnostic :**
```bash
# Activer le mode verbose
docker compose exec rag-system python scripts/query.py "test" --verbose

# Analyser les logs
docker compose logs rag-system | grep "query_time_ms"
```

**Solutions :**
- Augmenter `nprobe` pour FAISS
- Utiliser un index GPU
- Implémenter un cache Redis pour les requêtes fréquentes

### Problème : Docling ne parse pas certains PDFs

**Solution :**
```python
# Ajouter un fallback vers PyPDF
try:
    result = docling_converter.convert(file_path)
except Exception:
    # Fallback
    import pypdf
    reader = pypdf.PdfReader(file_path)
    text = "\n".join(page.extract_text() for page in reader.pages)
```

---

## Références

### Documentation officielle

- **Docling** : https://github.com/DS4SD/docling
- **FAISS** : https://github.com/facebookresearch/faiss
- **LangChain** : https://python.langchain.com/
- **FastAPI** : https://fastapi.tiangolo.com/
- **Docker** : https://docs.docker.com/

### Articles et tutoriels

- [RAG Best Practices](https://www.anthropic.com/news/contextual-retrieval)
- [FAISS Performance Tuning](https://github.com/facebookresearch/faiss/wiki/Faiss-performance-tips)
- [LangChain RAG Tutorial](https://python.langchain.com/docs/tutorials/rag/)

### Modèles d'embeddings recommandés

| Modèle | Dimension | Langue | Use Case |
|--------|-----------|--------|----------|
| all-MiniLM-L6-v2 | 384 | EN | Rapide, général |
| all-mpnet-base-v2 | 768 | EN | Précis, général |
| multilingual-e5-large | 1024 | Multi | Multilingue |
| sentence-camembert-large | 768 | FR | Français optimisé |

### Communauté et support

- **Discord LangChain** : https://discord.gg/langchain
- **Forum FAISS** : https://github.com/facebookresearch/faiss/discussions
- **Stack Overflow** : Tag `rag`, `langchain`, `faiss`

---

## Licence

Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.

## Contributeurs

Pour contribuer, veuillez consulter le fichier `CONTRIBUTING.md`.

---

**Dernière mise à jour** : 2026-01-08
**Version** : 1.0.0
**Auteur** : Projet RAG Team
