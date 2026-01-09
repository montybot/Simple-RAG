#!/usr/bin/env python3
"""
Simple example showing how to use the RAG system programmatically.
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag_pipeline import RAGPipeline
from src.config import get_settings

def main():
    """Simple RAG usage example."""

    # Get settings
    settings = get_settings()

    # Initialize the RAG pipeline
    print("Initializing RAG pipeline...")
    rag = RAGPipeline(
        embedding_model=settings.embedding_model,
        llm_model=settings.model_name,
        chunk_size=512
    )

    # Option 1: Load existing index
    index_path = Path("../data/indices/main_index")
    if index_path.exists():
        print(f"Loading index from {index_path}...")
        rag.load_index(index_path)
        print(f"Index loaded! Total vectors: {rag.vector_store.index.ntotal}")

        # Query the system
        question = "What is RAG?"
        print(f"\nQuerying: {question}")
        result = rag.query(question, top_k=5)

        print(f"\nAnswer:\n{result.answer}\n")
        print(f"Sources ({len(result.sources)}):")
        for i, source in enumerate(result.sources, 1):
            print(f"{i}. {source['title']} (score: {source['score']:.4f})")

        print(f"\nQuery completed in {result.query_time_ms:.2f}ms")

    # Option 2: Build index from documents
    else:
        print(f"No index found at {index_path}")
        raw_dir = Path("../data/raw")

        if raw_dir.exists() and list(raw_dir.glob("*")):
            print(f"Building index from {raw_dir}...")
            rag.index_documents(raw_dir)

            # Save the index
            print(f"Saving index to {index_path}...")
            rag.save_index(index_path)

            stats = rag.get_stats()
            print(f"\nIndex built successfully!")
            print(f"Total vectors: {stats['vector_store']['total_vectors']}")
            print(f"Total documents: {stats['vector_store']['total_documents']}")
        else:
            print(f"No documents found in {raw_dir}")
            print("Please add documents to the raw directory and try again")


if __name__ == "__main__":
    main()
