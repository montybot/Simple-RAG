#!/usr/bin/env python3
# scripts/build_index.py
"""
Build or rebuild the FAISS index from documents in the raw directory.
"""
import argparse
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag_pipeline import RAGPipeline
from src.config import get_settings
from loguru import logger


def main():
    """Main function to build the index."""
    parser = argparse.ArgumentParser(description="Build FAISS index from documents")
    parser.add_argument(
        "--input-dir",
        type=Path,
        help="Directory containing documents to index",
        default=None
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory to save the index",
        default=None
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        help="Embedding model to use",
        default=None
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Force rebuild even if index exists"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        help="Size of text chunks",
        default=None
    )
    parser.add_argument(
        "--index-type",
        type=str,
        choices=["Flat", "IVFFlat"],
        help="Type of FAISS index",
        default=None
    )
    parser.add_argument(
        "--nlist",
        type=int,
        help="Number of IVF clusters (IVFFlat only)",
        default=None
    )
    parser.add_argument(
        "--nprobe",
        type=int,
        help="Number of IVF probes at search time (IVFFlat only)",
        default=None
    )

    args = parser.parse_args()

    # Get settings
    settings = get_settings()

    # Determine paths
    input_dir = args.input_dir or settings.raw_dir
    output_dir = args.output_dir or (settings.indices_dir / "main_index")

    # Check if index exists
    if output_dir.exists() and not args.force_rebuild:
        logger.warning(f"Index already exists at {output_dir}")
        logger.warning("Use --force-rebuild to rebuild anyway")
        response = input("Do you want to rebuild? (y/n): ")
        if response.lower() != 'y':
            logger.info("Aborted")
            return

    # Initialize pipeline
    logger.info("Initializing RAG pipeline...")
    pipeline = RAGPipeline(
        embedding_model=args.embedding_model or settings.embedding_model,
        llm_model=settings.model_name,
        chunk_size=args.chunk_size or settings.max_chunk_size,
        chunk_overlap=settings.chunk_overlap,
        index_type=args.index_type or settings.faiss_index_type,
        nlist=args.nlist or settings.faiss_nlist,
        nprobe=args.nprobe or settings.faiss_nprobe,
    )

    # Index documents
    logger.info(f"Indexing documents from {input_dir}")
    pipeline.index_documents(input_dir)

    # Save index
    logger.info(f"Saving index to {output_dir}")
    pipeline.save_index(output_dir)

    # Print stats
    stats = pipeline.get_stats()
    logger.info("=" * 50)
    logger.info("Index built successfully!")
    logger.info(f"Total vectors: {stats['vector_store']['total_vectors']}")
    logger.info(f"Total documents: {stats['vector_store']['total_documents']}")
    logger.info(f"Embedding model: {stats['embedding_model']}")
    logger.info(f"Index type: {stats['vector_store']['index_type']}")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
