# src/vector_store.py
import faiss
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
import pickle
from loguru import logger


class FAISSVectorStore:
    """FAISS vector store manager for fast similarity search."""

    def __init__(self, dimension: int = 384, index_type: str = "Flat"):
        """
        Initialize the FAISS vector store.

        Args:
            dimension: Dimension of the embedding vectors
            index_type: Type of FAISS index ("Flat" or "IVFFlat")
        """
        self.dimension = dimension
        self.index_type = index_type
        self.index = None
        self.documents = []
        self._init_index()

    def _init_index(self):
        """Initialize the FAISS index based on the specified type."""
        if self.index_type == "Flat":
            self.index = faiss.IndexFlatL2(self.dimension)
            logger.info(f"Initialized Flat index with dimension {self.dimension}")

        elif self.index_type == "IVFFlat":
            quantizer = faiss.IndexFlatL2(self.dimension)
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, 100)
            logger.info(f"Initialized IVFFlat index with dimension {self.dimension}")

        else:
            raise ValueError(f"Unknown index type: {self.index_type}")

    def add_embeddings(self, embeddings: np.ndarray, documents: List[Dict]):
        """
        Add embeddings to the index.

        Args:
            embeddings: Numpy array of shape (n, dimension)
            documents: List of document metadata dictionaries
        """
        if embeddings.shape[1] != self.dimension:
            raise ValueError(f"Embedding dimension {embeddings.shape[1]} does not match index dimension {self.dimension}")

        # Train IVF index if needed
        if self.index_type == "IVFFlat" and not self.index.is_trained:
            logger.info("Training IVF index...")
            self.index.train(embeddings)

        # Add vectors to index
        self.index.add(embeddings)
        self.documents.extend(documents)

        logger.info(f"Added {len(embeddings)} embeddings. Total in index: {self.index.ntotal}")

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[Dict, float]]:
        """
        Search for the k most similar documents.

        Args:
            query_embedding: Query vector of shape (1, dimension) or (dimension,)
            top_k: Number of results to return

        Returns:
            List of tuples (document_metadata, distance_score)
        """
        if self.index.ntotal == 0:
            logger.warning("Index is empty, returning no results")
            return []

        # Ensure query is 2D array
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)

        # Perform search
        distances, indices = self.index.search(query_embedding, min(top_k, self.index.ntotal))

        # Build results
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx >= 0 and idx < len(self.documents):
                results.append((self.documents[idx], float(dist)))

        logger.debug(f"Search returned {len(results)} results")
        return results

    def save(self, path: Path):
        """
        Save the index and metadata to disk.

        Args:
            path: Directory path to save the index
        """
        path.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        index_path = path / "index.faiss"
        faiss.write_index(self.index, str(index_path))

        # Save document metadata
        metadata_path = path / "documents.pkl"
        with open(metadata_path, "wb") as f:
            pickle.dump(self.documents, f)

        # Save configuration
        config_path = path / "config.pkl"
        config = {
            "dimension": self.dimension,
            "index_type": self.index_type
        }
        with open(config_path, "wb") as f:
            pickle.dump(config, f)

        logger.info(f"Saved index to {path} (total vectors: {self.index.ntotal})")

    def load(self, path: Path):
        """
        Load an existing index from disk.

        Args:
            path: Directory path containing the saved index
        """
        # Load configuration
        config_path = path / "config.pkl"
        with open(config_path, "rb") as f:
            config = pickle.load(f)

        self.dimension = config["dimension"]
        self.index_type = config["index_type"]

        # Load FAISS index
        index_path = path / "index.faiss"
        self.index = faiss.read_index(str(index_path))

        # Load document metadata
        metadata_path = path / "documents.pkl"
        with open(metadata_path, "rb") as f:
            self.documents = pickle.load(f)

        logger.info(f"Loaded index from {path} (total vectors: {self.index.ntotal})")

    def get_stats(self) -> Dict:
        """Get statistics about the vector store."""
        return {
            "total_vectors": self.index.ntotal if self.index else 0,
            "dimension": self.dimension,
            "index_type": self.index_type,
            "total_documents": len(self.documents)
        }
