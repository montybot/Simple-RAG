#!/usr/bin/env python3
# scripts/query.py
"""
Query the RAG system from the command line.
"""
import argparse
from pathlib import Path
import sys
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag_pipeline import RAGPipeline
from src.config import get_settings
from loguru import logger


def main():
    """Main function to query the RAG system."""
    parser = argparse.ArgumentParser(description="Query the RAG system")
    parser.add_argument(
        "question",
        type=str,
        help="Question to ask the RAG system"
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        help="Directory containing the index",
        default=None
    )
    parser.add_argument(
        "--top-k",
        type=int,
        help="Number of documents to retrieve",
        default=5
    )
    parser.add_argument(
        "--output",
        type=str,
        choices=["text", "json"],
        help="Output format",
        default="text"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed information"
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        help="Embedding model to use",
        default=None
    )

    args = parser.parse_args()

    # Get settings
    settings = get_settings()

    # Configure logging
    if not args.verbose:
        logger.remove()
        logger.add(sys.stderr, level="WARNING")

    # Determine index path
    index_dir = args.index_dir or (settings.indices_dir / "main_index")

    # Check if index exists
    if not index_dir.exists():
        logger.error(f"Index not found at {index_dir}")
        logger.error("Please build the index first using build_index.py")
        sys.exit(1)

    # Initialize pipeline
    if args.verbose:
        logger.info("Initializing RAG pipeline...")

    pipeline = RAGPipeline(
        embedding_model=args.embedding_model or settings.embedding_model,
        llm_model=settings.model_name
    )

    # Load index
    if args.verbose:
        logger.info(f"Loading index from {index_dir}")

    pipeline.load_index(index_dir)

    # Execute query
    if args.verbose:
        logger.info(f"Querying: {args.question}")

    result = pipeline.query(
        question=args.question,
        top_k=args.top_k
    )

    # Output results
    if args.output == "json":
        # JSON output
        output = {
            "question": args.question,
            "answer": result.answer,
            "sources": result.sources,
            "query_time_ms": result.query_time_ms
        }
        print(json.dumps(output, indent=2))

    else:
        # Text output
        print("\n" + "=" * 70)
        print("QUESTION:")
        print(args.question)
        print("\n" + "=" * 70)
        print("ANSWER:")
        print(result.answer)
        print("\n" + "=" * 70)
        print(f"SOURCES ({len(result.sources)}):")
        print()

        for i, source in enumerate(result.sources, 1):
            print(f"{i}. {source['title']}")
            print(f"   File: {source['file']}")
            print(f"   Score: {source['score']:.4f}")
            print(f"   Excerpt: {source['excerpt']}")
            print()

        print("=" * 70)
        if args.verbose:
            print(f"Query time: {result.query_time_ms:.2f}ms")
            print("=" * 70)


if __name__ == "__main__":
    main()
