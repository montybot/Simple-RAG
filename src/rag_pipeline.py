# src/rag_pipeline.py
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass
import time
from loguru import logger

from .document_processor import DocumentProcessor
from .embeddings import EmbeddingModel
from .vector_store import FAISSVectorStore


@dataclass
class RAGResult:
    """Result of a RAG query."""
    answer: str
    sources: List[Dict]
    query_time_ms: float


class RAGPipeline:
    """Complete RAG pipeline combining document processing, embeddings, and retrieval."""

    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        llm_model: str = "gpt-4-turbo-preview",
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        index_type: str = "Flat"
    ):
        """
        Initialize the RAG pipeline.

        Args:
            embedding_model: Name of the embedding model
            llm_model: Name of the LLM model for generation
            chunk_size: Size of document chunks
            chunk_overlap: Overlap between chunks
            index_type: Type of FAISS index
        """
        logger.info("Initializing RAG Pipeline...")

        self.llm_model = llm_model
        self.document_processor = DocumentProcessor(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        self.embedding_model = EmbeddingModel(model_name=embedding_model)

        self.vector_store = FAISSVectorStore(
            dimension=self.embedding_model.dimension,
            index_type=index_type
        )

        logger.info("RAG Pipeline initialized successfully")

    def index_documents(self, input_dir: Path):
        """
        Index all documents from a directory.

        Args:
            input_dir: Path to directory containing documents
        """
        start = time.time()
        logger.info(f"Starting document indexing from {input_dir}")

        # 1. Process documents
        processed_docs = self.document_processor.process_directory(input_dir)

        if not processed_docs:
            logger.warning(f"No documents found in {input_dir}")
            return

        # 2. Extract all chunks and metadata
        all_chunks = []
        all_metadata = []

        for doc in processed_docs:
            for chunk in doc["chunks"]:
                all_chunks.append(chunk)
                all_metadata.append({
                    "file_path": doc["file_path"],
                    "title": doc["metadata"]["title"],
                    "chunk_text": chunk,
                    "format": doc["metadata"]["format"]
                })

        logger.info(f"Creating embeddings for {len(all_chunks)} chunks")

        # 3. Create embeddings
        embeddings = self.embedding_model.encode_batch(all_chunks)

        # 4. Add to vector store
        self.vector_store.add_embeddings(embeddings, all_metadata)

        elapsed = time.time() - start
        logger.info(f"Indexing completed in {elapsed:.2f}s. Total documents: {len(processed_docs)}, Total chunks: {len(all_chunks)}")

    def query(self, question: str, top_k: int = 5) -> RAGResult:
        """
        Execute a RAG query.

        Args:
            question: User's question
            top_k: Number of documents to retrieve

        Returns:
            RAGResult with answer and sources
        """
        start = time.time()
        logger.info(f"Processing query: {question}")

        # 1. Encode the question
        query_embedding = self.embedding_model.encode([question])

        # 2. Search in the vector store
        results = self.vector_store.search(query_embedding, top_k=top_k)

        if not results:
            logger.warning("No relevant documents found")
            return RAGResult(
                answer="I couldn't find any relevant information to answer your question.",
                sources=[],
                query_time_ms=(time.time() - start) * 1000
            )

        # 3. Build context from results
        context = self._build_context(results)

        # 4. Generate answer using LLM
        answer = self._generate_answer(question, context)

        # 5. Format sources
        sources = [
            {
                "file": res[0]["file_path"],
                "title": res[0]["title"],
                "score": res[1],
                "excerpt": res[0]["chunk_text"][:200] + "..."
            }
            for res in results
        ]

        elapsed = (time.time() - start) * 1000
        logger.info(f"Query completed in {elapsed:.2f}ms")

        return RAGResult(
            answer=answer,
            sources=sources,
            query_time_ms=elapsed
        )

    def _build_context(self, results: List[Tuple[Dict, float]]) -> str:
        """
        Build context string from search results.

        Args:
            results: List of (document, score) tuples

        Returns:
            Formatted context string
        """
        context_parts = []
        for i, (doc, score) in enumerate(results, 1):
            context_parts.append(f"[Document {i} - Score: {score:.4f}]")
            context_parts.append(doc['chunk_text'])
            context_parts.append("")

        return "\n".join(context_parts)

    def _get_llm(self):
        """
        Get LLM instance based on model name.

        Supports multiple providers:
        - OpenAI (gpt-4, gpt-3.5-turbo, etc.)
        - Anthropic (claude-3-5-sonnet, claude-3-opus, etc.)
        - Mistral AI (mistral-large, mistral-medium, mistral-small, etc.)
        - Ollama (llama3.2, mistral, codellama, etc.)

        Returns:
            LangChain LLM instance
        """
        model_lower = self.llm_model.lower()

        try:
            # Anthropic Claude
            if "claude" in model_lower or "anthropic" in model_lower:
                from langchain_anthropic import ChatAnthropic
                logger.info(f"Using Anthropic Claude: {self.llm_model}")
                return ChatAnthropic(model=self.llm_model, temperature=0)

            # Mistral AI API (mistral-large, mistral-medium, mistral-small, etc.)
            elif "mistral-" in model_lower or "open-mistral" in model_lower:
                from langchain_mistralai import ChatMistralAI
                logger.info(f"Using Mistral AI API: {self.llm_model}")
                return ChatMistralAI(model=self.llm_model, temperature=0)

            # Local Ollama (llama, mistral without dash, codellama, etc.)
            elif "llama" in model_lower or (("mistral" in model_lower or "ollama" in model_lower) and "-" not in model_lower):
                from langchain_ollama import ChatOllama
                logger.info(f"Using Ollama local model: {self.llm_model}")
                return ChatOllama(
                    model=self.llm_model,
                    temperature=0,
                    base_url="http://localhost:11434"
                )

            # Default to OpenAI
            else:
                from langchain_openai import ChatOpenAI
                logger.info(f"Using OpenAI: {self.llm_model}")
                return ChatOpenAI(model=self.llm_model, temperature=0)

        except ImportError as e:
            logger.error(f"Failed to import LLM provider: {e}")
            logger.error("Install the required package:")
            if "anthropic" in str(e):
                logger.error("  pip install langchain-anthropic")
            elif "mistralai" in str(e):
                logger.error("  pip install langchain-mistralai")
            elif "ollama" in str(e):
                logger.error("  pip install langchain-ollama")
            else:
                logger.error("  pip install langchain-openai")
            raise

    def _generate_answer(self, question: str, context: str) -> str:
        """
        Generate answer using LLM (supports multiple providers).

        Args:
            question: User's question
            context: Retrieved context

        Returns:
            Generated answer
        """
        try:
            from langchain.prompts import PromptTemplate

            prompt = PromptTemplate(
                template="""Use the following context to answer the question.
If you cannot find the answer in the context, say so clearly.

Context:
{context}

Question: {question}

Answer:""",
                input_variables=["context", "question"]
            )

            # Get LLM based on provider
            llm = self._get_llm()
            chain = prompt | llm

            result = chain.invoke({"context": context, "question": question})
            return result.content

        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return f"Error generating answer: {str(e)}"

    def save_index(self, path: Path):
        """
        Save the vector store index.

        Args:
            path: Directory path to save the index
        """
        self.vector_store.save(path)

    def load_index(self, path: Path):
        """
        Load an existing vector store index.

        Args:
            path: Directory path containing the saved index
        """
        self.vector_store.load(path)

    def get_stats(self) -> Dict:
        """Get pipeline statistics."""
        return {
            "embedding_model": self.embedding_model.model_name,
            "llm_model": self.llm_model,
            "vector_store": self.vector_store.get_stats()
        }
