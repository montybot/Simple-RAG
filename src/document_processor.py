# src/document_processor.py
from docling.document_converter import DocumentConverter
from pathlib import Path
from typing import List, Dict
from loguru import logger
from .csv_processor import CSVProcessor


class DocumentProcessor:
    """Document processor using Docling for parsing various formats."""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        """
        Initialize the document processor.

        Args:
            chunk_size: Maximum size of text chunks
            chunk_overlap: Overlap between consecutive chunks
        """
        self.converter = DocumentConverter()
        self.csv_processor = CSVProcessor()
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        logger.info(f"DocumentProcessor initialized with chunk_size={chunk_size}, overlap={chunk_overlap}")

    def process_document(self, file_path: Path) -> Dict:
        """
        Parse a document and extract its structured content.

        Args:
            file_path: Path to the source document

        Returns:
            Dict containing text, metadata and chunks
        """
        logger.info(f"Processing document: {file_path}")

        try:
            # Use specialized CSV processor for CSV files
            if file_path.suffix.lower() == '.csv':
                return self._process_csv_document(file_path)

            # Convert document using Docling for other formats
            result = self.converter.convert(str(file_path))

            # Extract text as markdown
            full_text = result.document.export_to_markdown()

            # Create intelligent chunks
            chunks = self._create_chunks(full_text)

            metadata = {
                "file_path": str(file_path),
                "text": full_text,
                "chunks": chunks,
                "metadata": {
                    "title": result.document.name,
                    "page_count": len(result.document.pages),
                    "format": file_path.suffix,
                    "chunk_count": len(chunks)
                }
            }

            logger.info(f"Successfully processed {file_path}: {len(chunks)} chunks created")
            return metadata

        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            raise

    def _process_csv_document(self, file_path: Path) -> Dict:
        """
        Process CSV files with structured field preservation.

        Args:
            file_path: Path to the CSV file

        Returns:
            Dict containing structured chunks and metadata
        """
        logger.info(f"Processing CSV with structured parser: {file_path}")

        # Get structured chunks from CSV processor
        csv_chunks = self.csv_processor.process_csv(file_path)

        # Extract just the text chunks
        chunks = [chunk['text'] for chunk in csv_chunks]

        # Store all metadata from the first chunk (file-level info)
        first_chunk_metadata = csv_chunks[0]['metadata'] if csv_chunks else {}

        metadata = {
            "file_path": str(file_path),
            "text": "\n\n---\n\n".join(chunks),
            "chunks": chunks,
            "metadata": {
                "title": file_path.stem,
                "page_count": len(csv_chunks),  # Each row is like a "page"
                "format": file_path.suffix,
                "chunk_count": len(chunks),
                "source_type": "csv_structured"
            },
            "csv_metadata": [chunk['metadata'] for chunk in csv_chunks]
        }

        logger.info(f"Successfully processed CSV {file_path}: {len(chunks)} structured events")
        return metadata

    def _create_chunks(self, text: str) -> List[str]:
        """
        Split text into chunks with overlap.

        Args:
            text: Full text to chunk

        Returns:
            List of text chunks
        """
        chunks = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]

            # Only add non-empty chunks
            if chunk.strip():
                chunks.append(chunk)

            start += self.chunk_size - self.chunk_overlap

        return chunks

    def process_directory(self, directory: Path) -> List[Dict]:
        """
        Process all documents in a directory.

        Args:
            directory: Path to directory containing documents

        Returns:
            List of processed document dictionaries
        """
        logger.info(f"Processing directory: {directory}")
        results = []
        supported_formats = ['.pdf', '.docx', '.txt', '.html', '.md', '.csv']

        for file_path in directory.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in supported_formats:
                try:
                    result = self.process_document(file_path)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {e}")
                    continue

        logger.info(f"Processed {len(results)} documents from {directory}")
        return results
