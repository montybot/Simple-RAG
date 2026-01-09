#!/usr/bin/env python3
# scripts/monitor.py
"""
Monitor and display statistics about the RAG system.
"""
import argparse
from pathlib import Path
import sys
import json
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag_pipeline import RAGPipeline
from src.config import get_settings
from loguru import logger


def format_bytes(bytes_val):
    """Format bytes to human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_val < 1024.0:
            return f"{bytes_val:.2f} {unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.2f} TB"


def get_index_size(index_dir: Path) -> int:
    """Calculate total size of index files."""
    total_size = 0
    if index_dir.exists():
        for file in index_dir.rglob("*"):
            if file.is_file():
                total_size += file.stat().st_size
    return total_size


def display_stats(pipeline: RAGPipeline, index_dir: Path, output_format: str):
    """Display system statistics."""
    stats = pipeline.get_stats()
    index_size = get_index_size(index_dir)

    if output_format == "json":
        output = {
            "timestamp": datetime.now().isoformat(),
            "embedding_model": stats["embedding_model"],
            "llm_model": stats["llm_model"],
            "vector_store": stats["vector_store"],
            "index_size_bytes": index_size,
            "index_path": str(index_dir)
        }
        print(json.dumps(output, indent=2))

    else:
        print("\n" + "=" * 70)
        print("RAG SYSTEM STATISTICS")
        print("=" * 70)
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()

        print("MODELS:")
        print(f"  Embedding Model: {stats['embedding_model']}")
        print(f"  LLM Model: {stats['llm_model']}")
        print()

        print("VECTOR STORE:")
        print(f"  Total Vectors: {stats['vector_store']['total_vectors']:,}")
        print(f"  Total Documents: {stats['vector_store']['total_documents']:,}")
        print(f"  Dimension: {stats['vector_store']['dimension']}")
        print(f"  Index Type: {stats['vector_store']['index_type']}")
        print()

        print("INDEX STORAGE:")
        print(f"  Index Path: {index_dir}")
        print(f"  Index Size: {format_bytes(index_size)}")
        print()

        print("=" * 70 + "\n")


def main():
    """Main function to monitor the RAG system."""
    parser = argparse.ArgumentParser(description="Monitor RAG system statistics")
    parser.add_argument(
        "--index-dir",
        type=Path,
        help="Directory containing the index",
        default=None
    )
    parser.add_argument(
        "--output",
        type=str,
        choices=["text", "json"],
        help="Output format",
        default="text"
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Continuously watch and update stats (not implemented yet)"
    )

    args = parser.parse_args()

    # Get settings
    settings = get_settings()

    # Determine index path
    index_dir = args.index_dir or (settings.indices_dir / "main_index")

    # Check if index exists
    if not index_dir.exists():
        logger.error(f"Index not found at {index_dir}")
        logger.error("Please build the index first using build_index.py")
        sys.exit(1)

    # Initialize pipeline
    pipeline = RAGPipeline(
        embedding_model=settings.embedding_model,
        llm_model=settings.model_name
    )

    # Load index
    pipeline.load_index(index_dir)

    # Display stats
    display_stats(pipeline, index_dir, args.output)

    if args.watch:
        logger.warning("Watch mode not yet implemented")


if __name__ == "__main__":
    main()
