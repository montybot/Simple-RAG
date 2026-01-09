# tests/test_rag_pipeline.py
import pytest
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag_pipeline import RAGPipeline, RAGResult


@pytest.fixture
def rag_pipeline():
    """Create a RAGPipeline instance for testing."""
    return RAGPipeline(
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        llm_model="gpt-4-turbo-preview",
        chunk_size=100,
        chunk_overlap=20
    )


@pytest.mark.skip(reason="Requires downloading models - slow test")
def test_rag_pipeline_initialization(rag_pipeline):
    """Test that RAGPipeline initializes correctly."""
    assert rag_pipeline.document_processor is not None
    assert rag_pipeline.embedding_model is not None
    assert rag_pipeline.vector_store is not None
    assert rag_pipeline.llm_model == "gpt-4-turbo-preview"


@pytest.mark.skip(reason="Requires actual documents and models")
def test_index_documents(rag_pipeline, tmp_path):
    """Test indexing documents."""
    # This would require actual test documents
    pass


@pytest.mark.skip(reason="Requires indexed data and API key")
def test_query(rag_pipeline):
    """Test querying the RAG system."""
    # This would require indexed data and an API key
    pass


@pytest.mark.skip(reason="Requires indexed data")
def test_save_and_load_index(rag_pipeline, tmp_path):
    """Test saving and loading the index."""
    # This would require indexed data
    pass


def test_rag_result_dataclass():
    """Test RAGResult dataclass."""
    result = RAGResult(
        answer="Test answer",
        sources=[{"file": "test.pdf", "score": 0.9}],
        query_time_ms=100.5
    )

    assert result.answer == "Test answer"
    assert len(result.sources) == 1
    assert result.query_time_ms == 100.5


@pytest.mark.skip(reason="Requires downloading models - slow test")
def test_build_context(rag_pipeline):
    """Test context building from search results."""
    results = [
        ({"chunk_text": "First chunk of text"}, 0.85),
        ({"chunk_text": "Second chunk of text"}, 0.75)
    ]

    context = rag_pipeline._build_context(results)

    assert isinstance(context, str)
    assert "First chunk of text" in context
    assert "Second chunk of text" in context


@pytest.mark.skip(reason="Requires downloading models - slow test")
def test_get_stats(rag_pipeline):
    """Test getting pipeline statistics."""
    stats = rag_pipeline.get_stats()

    assert "embedding_model" in stats
    assert "llm_model" in stats
    assert "vector_store" in stats
