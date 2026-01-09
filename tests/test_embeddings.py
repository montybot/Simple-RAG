# tests/test_embeddings.py
import pytest
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.embeddings import EmbeddingModel


@pytest.fixture
def embedding_model():
    """Create an EmbeddingModel instance for testing."""
    # Use a small model for faster testing
    return EmbeddingModel(model_name="sentence-transformers/all-MiniLM-L6-v2")


@pytest.mark.skip(reason="Requires downloading model - slow test")
def test_embedding_model_initialization(embedding_model):
    """Test that EmbeddingModel initializes correctly."""
    assert embedding_model.model is not None
    assert embedding_model.dimension > 0
    assert embedding_model.model_name == "sentence-transformers/all-MiniLM-L6-v2"


@pytest.mark.skip(reason="Requires downloading model - slow test")
def test_encode_single_text(embedding_model):
    """Test encoding a single text."""
    text = "This is a test sentence."
    embedding = embedding_model.encode(text)

    assert isinstance(embedding, np.ndarray)
    assert len(embedding.shape) == 2
    assert embedding.shape[0] == 1
    assert embedding.shape[1] == embedding_model.dimension


@pytest.mark.skip(reason="Requires downloading model - slow test")
def test_encode_multiple_texts(embedding_model):
    """Test encoding multiple texts."""
    texts = [
        "First test sentence.",
        "Second test sentence.",
        "Third test sentence."
    ]
    embeddings = embedding_model.encode(texts)

    assert isinstance(embeddings, np.ndarray)
    assert len(embeddings.shape) == 2
    assert embeddings.shape[0] == 3
    assert embeddings.shape[1] == embedding_model.dimension


@pytest.mark.skip(reason="Requires downloading model - slow test")
def test_encode_batch(embedding_model):
    """Test batch encoding."""
    texts = ["Test sentence"] * 10
    embeddings = embedding_model.encode_batch(texts, batch_size=5)

    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape[0] == 10
    assert embeddings.shape[1] == embedding_model.dimension


@pytest.mark.skip(reason="Requires downloading model - slow test")
def test_get_dimension(embedding_model):
    """Test getting the embedding dimension."""
    dimension = embedding_model.get_dimension()

    assert isinstance(dimension, int)
    assert dimension > 0
    assert dimension == embedding_model.dimension
