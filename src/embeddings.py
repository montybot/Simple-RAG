# src/embeddings.py
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Union
from loguru import logger


class EmbeddingModel:
    """Wrapper for sentence-transformers embedding models."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the embedding model.

        Args:
            model_name: Name of the sentence-transformers model to use
        """
        self.model_name = model_name
        logger.info(f"Loading embedding model: {model_name}")

        try:
            self.model = SentenceTransformer(model_name)
            self.dimension = self.model.get_sentence_embedding_dimension()
            logger.info(f"Embedding model loaded successfully. Dimension: {self.dimension}")
        except Exception as e:
            logger.error(f"Error loading embedding model: {e}")
            raise

    def encode(self, texts: Union[str, List[str]], batch_size: int = 32, show_progress: bool = False) -> np.ndarray:
        """
        Encode text(s) into embeddings.

        Args:
            texts: Single text string or list of texts
            batch_size: Batch size for encoding
            show_progress: Whether to show progress bar

        Returns:
            numpy array of embeddings with shape (n, dimension)
        """
        if isinstance(texts, str):
            texts = [texts]

        try:
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True
            )

            # Ensure 2D array
            if len(embeddings.shape) == 1:
                embeddings = embeddings.reshape(1, -1)

            return embeddings

        except Exception as e:
            logger.error(f"Error encoding texts: {e}")
            raise

    def encode_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Encode a batch of texts with progress tracking.

        Args:
            texts: List of text strings
            batch_size: Batch size for encoding

        Returns:
            numpy array of embeddings
        """
        logger.info(f"Encoding {len(texts)} texts with batch_size={batch_size}")
        return self.encode(texts, batch_size=batch_size, show_progress=True)

    def get_dimension(self) -> int:
        """Get the embedding dimension."""
        return self.dimension
