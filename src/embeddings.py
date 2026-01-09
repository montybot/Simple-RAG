# src/embeddings.py
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Union
from loguru import logger


class EmbeddingModel:
    """
    Wrapper for embedding models.

    Supports:
    - Local models via sentence-transformers
    - Mistral AI API embeddings
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the embedding model.

        Args:
            model_name: Name of the embedding model to use
                - sentence-transformers models (local)
                - "mistral-embed" for Mistral AI API
        """
        self.model_name = model_name
        self.model_type = self._detect_model_type(model_name)
        logger.info(f"Loading embedding model: {model_name} (type: {self.model_type})")

        try:
            if self.model_type == "mistral":
                self._init_mistral_embeddings()
            else:
                self._init_sentence_transformers()

            logger.info(f"Embedding model loaded successfully. Dimension: {self.dimension}")
        except Exception as e:
            logger.error(f"Error loading embedding model: {e}")
            raise

    def _detect_model_type(self, model_name: str) -> str:
        """Detect the type of embedding model."""
        model_lower = model_name.lower()

        if "mistral" in model_lower and "embed" in model_lower:
            return "mistral"
        else:
            return "sentence-transformers"

    def _init_sentence_transformers(self):
        """Initialize sentence-transformers model (local)."""
        self.model = SentenceTransformer(self.model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        logger.info(f"Using local sentence-transformers model")

    def _init_mistral_embeddings(self):
        """Initialize Mistral AI embeddings (API-based)."""
        try:
            from langchain_mistralai import MistralAIEmbeddings
            self.model = MistralAIEmbeddings(model=self.model_name)
            self.dimension = 1024  # Mistral embed dimension
            logger.info(f"Using Mistral AI embeddings API")
        except ImportError:
            logger.error("langchain-mistralai not installed. Install with: pip install langchain-mistralai")
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
            if self.model_type == "mistral":
                # Mistral AI API embeddings
                embeddings = self.model.embed_documents(texts)
                embeddings = np.array(embeddings)
            else:
                # Sentence transformers (local)
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
