# tests/test_document_processor.py
import pytest
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.document_processor import DocumentProcessor


@pytest.fixture
def document_processor():
    """Create a DocumentProcessor instance for testing."""
    return DocumentProcessor(chunk_size=100, chunk_overlap=20)


def test_document_processor_initialization(document_processor):
    """Test that DocumentProcessor initializes correctly."""
    assert document_processor.chunk_size == 100
    assert document_processor.chunk_overlap == 20
    assert document_processor.converter is not None


def test_create_chunks(document_processor):
    """Test the chunk creation method."""
    text = "A" * 250  # 250 characters
    chunks = document_processor._create_chunks(text)

    assert len(chunks) > 0
    # First chunk should be 100 chars
    assert len(chunks[0]) == 100
    # Should have overlap
    assert len(chunks) > 2


def test_create_chunks_empty_text(document_processor):
    """Test chunk creation with empty text."""
    text = ""
    chunks = document_processor._create_chunks(text)

    assert len(chunks) == 0


def test_create_chunks_short_text(document_processor):
    """Test chunk creation with text shorter than chunk_size."""
    text = "Short text"
    chunks = document_processor._create_chunks(text)

    assert len(chunks) == 1
    assert chunks[0] == text


@pytest.mark.skip(reason="Requires actual document files")
def test_process_document(document_processor, tmp_path):
    """Test processing a real document."""
    # This would require actual test documents
    pass


@pytest.mark.skip(reason="Requires actual document files")
def test_process_directory(document_processor, tmp_path):
    """Test processing a directory of documents."""
    # This would require actual test documents
    pass
