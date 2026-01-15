#!/usr/bin/env python3
"""
Script to completely erase all database information.
This includes FAISS indices, processed documents, and cached data.
"""

import shutil
from pathlib import Path
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Directories to clean
BASE_DIR = Path(__file__).parent.parent
DIRECTORIES_TO_CLEAN = [
    BASE_DIR / "data" / "indices",
    BASE_DIR / "data" / "processed",
]

# Optional directories (ask before deleting)
OPTIONAL_DIRECTORIES = [
    BASE_DIR / "data" / "raw",
    BASE_DIR / "logs",
]


def clean_directory(directory: Path, keep_dir: bool = True):
    """
    Remove all contents of a directory.

    Args:
        directory: Path to the directory to clean
        keep_dir: If True, keep the directory structure but remove contents
    """
    if not directory.exists():
        logger.info(f"Directory {directory} does not exist, skipping...")
        return

    try:
        if keep_dir:
            # Remove all contents but keep the directory
            for item in directory.iterdir():
                if item.is_file():
                    item.unlink()
                    logger.info(f"Deleted file: {item}")
                elif item.is_dir():
                    shutil.rmtree(item)
                    logger.info(f"Deleted directory: {item}")
        else:
            # Remove the entire directory
            shutil.rmtree(directory)
            logger.info(f"Deleted directory: {directory}")

    except Exception as e:
        logger.error(f"Error cleaning {directory}: {e}")


def main():
    """Main function to clean all database information."""
    logger.info("=" * 60)
    logger.info("DATABASE CLEANUP SCRIPT")
    logger.info("=" * 60)

    # Clean mandatory directories
    logger.info("\nCleaning core database directories...")
    for directory in DIRECTORIES_TO_CLEAN:
        logger.info(f"\nCleaning: {directory}")
        clean_directory(directory, keep_dir=True)

    # Ask about optional directories
    logger.info("\n" + "=" * 60)
    logger.info("Optional cleanup")
    logger.info("=" * 60)

    print("\nDo you want to also delete:")
    print("1. Raw documents (data/raw/)")
    print("2. Logs (logs/)")
    print("\nEnter 'yes' to delete, 'no' to keep, or 'exit' to stop: ")

    response = input().strip().lower()

    if response == 'exit':
        logger.info("Cleanup cancelled by user")
        sys.exit(0)
    elif response == 'yes':
        logger.info("\nCleaning optional directories...")
        for directory in OPTIONAL_DIRECTORIES:
            logger.info(f"\nCleaning: {directory}")
            clean_directory(directory, keep_dir=True)
    else:
        logger.info("Skipping optional directories")

    logger.info("\n" + "=" * 60)
    logger.info("CLEANUP COMPLETED!")
    logger.info("=" * 60)
    logger.info("\nAll database information has been erased.")
    logger.info("You can now rebuild the index with fresh data.")


if __name__ == "__main__":
    main()
