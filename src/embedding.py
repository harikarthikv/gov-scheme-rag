"""
embedding.py
────────────
Handles document chunking and embedding generation.

Responsibilities
────────────────
• Split raw LangChain Documents into overlapping chunks via
  RecursiveCharacterTextSplitter
• Load the sentence-transformers/all-MiniLM-L6-v2 embedding model locally
• Expose the HuggingFaceEmbeddings instance for use by the vector store
"""

import logging

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
        logger.info("Embedding model loaded successfully.")

    # ── Public API ────────────────────────────────────────────────────────────

    def split_documents(self, documents: list[Document]) -> list[Document]:
        """
        Split a list of LangChain Documents into smaller, overlapping chunks.

        Metadata from the parent document (filename, page, source) is
        automatically inherited by every child chunk.

        Args:
            documents: Raw page-level Documents from the PDF loader.

        Returns:
            List of chunked Documents ready for embedding.
        """
        chunks: list[Document] = self.text_splitter.split_documents(documents)
        logger.info(
            f"Split {len(documents)} document(s) → {len(chunks)} chunk(s) "
            f"(chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})"
        )
        return chunks

    def get_embeddings(self) -> HuggingFaceEmbeddings:
        """
        Return the configured HuggingFaceEmbeddings instance.

        This is passed directly to ChromaDB so it can embed both documents
        at indexing time and queries at retrieval time.
        """
        return self.embeddings
