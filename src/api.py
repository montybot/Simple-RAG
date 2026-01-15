# src/api.py
from fastapi import FastAPI, UploadFile, HTTPException, File
from pydantic import BaseModel
from pathlib import Path
from typing import Optional, List
from loguru import logger

from .rag_pipeline import RAGPipeline
from .config import get_settings

# Initialize settings
settings = get_settings()

# Create FastAPI app
app = FastAPI(
    title="RAG System API",
    description="API for RAG system with Docling, FAISS and LangChain",
    version="1.0.0"
)

# Initialize RAG pipeline
rag_pipeline = RAGPipeline(
    embedding_model=settings.embedding_model,
    llm_model=settings.model_name,
    chunk_size=settings.max_chunk_size,
    chunk_overlap=settings.chunk_overlap,
    index_type=settings.faiss_index_type,
    nlist=settings.faiss_nlist,
    nprobe=settings.faiss_nprobe,
)

# Load index at startup if it exists
index_path = settings.indices_dir / "main_index"


@app.on_event("startup")
async def startup_event():
    """Load index on application startup."""
    if index_path.exists():
        try:
            rag_pipeline.load_index(index_path)
            logger.info("Index loaded successfully on startup")
        except Exception as e:
            logger.error(f"Error loading index on startup: {e}")
    else:
        logger.warning(f"No index found at {index_path}. You need to index documents first.")


# Request/Response models
class QueryRequest(BaseModel):
    """Request model for query endpoint."""
    question: str
    top_k: int = 5
    temperature: float = 0.7
    max_tokens: int = 512
    top_p: float = 0.9
    system_prompt: str = None

    class Config:
        json_schema_extra = {
            "example": {
                "question": "What is RAG?",
                "top_k": 5,
                "temperature": 0.7,
                "max_tokens": 512,
                "top_p": 0.9,
                "system_prompt": "You are a helpful assistant."
            }
        }


class SourceInfo(BaseModel):
    """Information about a source document."""
    file: str
    title: str
    score: float
    excerpt: str


class QueryResponse(BaseModel):
    """Response model for query endpoint."""
    answer: str
    sources: List[SourceInfo]
    metadata: dict

    class Config:
        json_schema_extra = {
            "example": {
                "answer": "RAG stands for Retrieval-Augmented Generation...",
                "sources": [
                    {
                        "file": "/app/data/raw/document.pdf",
                        "title": "Introduction to RAG",
                        "score": 0.85,
                        "excerpt": "RAG is a technique..."
                    }
                ],
                "metadata": {
                    "query_time_ms": 245.3,
                    "documents_searched": 5
                }
            }
        }


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    index_size: int
    stats: dict


class UploadResponse(BaseModel):
    """Response model for document upload."""
    status: str
    document_id: str
    message: str


# Endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.

    Returns information about the system status and index size.
    """
    try:
        stats = rag_pipeline.get_stats()
        return HealthResponse(
            status="healthy",
            index_size=stats["vector_store"]["total_vectors"],
            stats=stats
        )
    except Exception as e:
        logger.error(f"Health check error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Query the RAG system.

    Args:
        request: Query request with question and optional top_k

    Returns:
        Answer with sources and metadata
    """
    try:
        logger.info(f"Received query: {request.question}")

        result = rag_pipeline.query(
            question=request.question,
            top_k=request.top_k,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            top_p=request.top_p,
            system_prompt=request.system_prompt
        )

        # Convert sources to SourceInfo models
        sources = [
            SourceInfo(
                file=source["file"],
                title=source["title"],
                score=source["score"],
                excerpt=source["excerpt"]
            )
            for source in result.sources
        ]

        return QueryResponse(
            answer=result.answer,
            sources=sources,
            metadata={
                "query_time_ms": result.query_time_ms,
                "documents_searched": len(result.sources)
            }
        )

    except Exception as e:
        logger.error(f"Query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/documents/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload and index a new document.

    Args:
        file: Document file to upload

    Returns:
        Upload status and document ID
    """
    try:
        logger.info(f"Received file upload: {file.filename}")

        # Save the file
        file_path = settings.raw_dir / file.filename
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        logger.info(f"File saved to {file_path}")

        # Re-index documents (including the new one)
        rag_pipeline.index_documents(settings.raw_dir)

        # Save updated index
        rag_pipeline.save_index(index_path)

        return UploadResponse(
            status="success",
            document_id=file.filename,
            message=f"Document '{file.filename}' indexed successfully"
        )

    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/index/rebuild")
async def rebuild_index():
    """
    Rebuild the entire index from scratch.

    This will re-process all documents in the raw directory.
    """
    try:
        logger.info("Starting index rebuild")

        # Reset the index to empty state
        rag_pipeline.reset_index()

        # Re-index all documents
        rag_pipeline.index_documents(settings.raw_dir)

        # Save the index
        rag_pipeline.save_index(index_path)

        stats = rag_pipeline.get_stats()

        return {
            "status": "success",
            "message": "Index rebuilt successfully",
            "stats": stats
        }

    except Exception as e:
        logger.error(f"Index rebuild error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
async def get_stats():
    """Get system statistics."""
    try:
        return rag_pipeline.get_stats()
    except Exception as e:
        logger.error(f"Stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=settings.api_host,
        port=settings.api_port,
        workers=settings.api_workers
    )
