"""
embedding.py
────────────
Handles document chunking and embedding generation.

Responsibilities
────────────────
• Split raw LangChain Documents into overlapping chunks via
  RecursiveCharacterTextSplitter
• Load the sentence-transformers/all-MiniLM-L6-v2 embedding model locally
• Expose the HuggingFaceEmbeddings instance for ChromaDB / FAISS
• Provide raw numpy embedding array generation for explicit FAISS indexing
"""

import logging

import numpy as np
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import CHUNK_OVERLAP, CHUNK_SIZE, EMBEDDING_MODEL

logger = logging.getLogger(__name__)


class EmbeddingPipeline:
    """Encapsulates text splitting and embedding generation."""

    def __init__(self) -> None:
        # ── Text splitter ─────────────────────────────────────────────────────
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            # Split on paragraph, sentence, then word boundaries in order
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        # ── Embedding model (runs locally — no API key needed) ────────────────
        logger.info(f"Loading embedding model: '{EMBEDDING_MODEL}'")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},       # change to "cuda" if GPU available
            encode_kwargs={"normalize_embeddings": True},
        )
        self._dimension: int = 384  # all-MiniLM-L6-v2 output dimension
        logger.info(f"Embedding model loaded — dimension: {self._dimension}")

    # ── Public API ────────────────────────────────────────────────────────────

    def split_documents(self, documents: list[Document]) -> list[Document]:
        """
        Split a list of LangChain Documents into smaller, overlapping chunks.

        Metadata from the parent document (scheme_name, ministry, etc.) is
        automatically inherited by every child chunk.

        Args:
            documents: Raw scheme-level Documents from the data loader.

        Returns:
            List of chunked Documents ready for embedding.
        """
        chunks: list[Document] = self.text_splitter.split_documents(documents)
        logger.info(
            f"Split {len(documents)} document(s) → {len(chunks)} chunk(s) "
            f"(chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})"
        )
        return chunks

    def embed_chunks(self, chunks: list[Document]) -> np.ndarray:
        """
        Convert chunked documents into a numpy array of embeddings.

        Each embedding is a 384-dimensional dense vector (all-MiniLM-L6-v2).

        Args:
            chunks: List of chunked Documents.

        Returns:
            NumPy array of shape (len(chunks), 384) containing the embeddings.
        """
        texts: list[str] = [doc.page_content for doc in chunks]
        logger.info(f"Embedding {len(texts)} chunk(s) …")
        embeddings_array: np.ndarray = np.array(
            self.embeddings.embed_documents(texts), dtype=np.float32
        )
        logger.info(
            f"Embeddings generated — shape: {embeddings_array.shape}, "
            f"dimension: {embeddings_array.shape[1]}"
        )
        return embeddings_array

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single query string into a numpy array.

        Args:
            query: The user's natural-language question.

        Returns:
            NumPy array of shape (384,) for similarity search.
        """
        return np.array(self.embeddings.embed_query(query), dtype=np.float32)

    def get_embeddings(self) -> HuggingFaceEmbeddings:
        """
        Return the configured HuggingFaceEmbeddings instance.

        This is passed directly to ChromaDB so it can embed both documents
        at indexing time and queries at retrieval time.
        """
        return self.embeddings

    @property
    def dimension(self) -> int:
        """Return the embedding dimension (384 for all-MiniLM-L6-v2)."""
        return self._dimension
