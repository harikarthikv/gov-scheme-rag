"""
vector_store.py
───────────────
Manages the persistent ChromaDB vector store.

Responsibilities
────────────────
• If a ChromaDB index already exists on disk → load it (skip re-embedding)
• Otherwise → create a new index from the provided document chunks
• Expose the Chroma instance for downstream use by the Retriever
"""

import logging

from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.config import COLLECTION_NAME, VECTOR_DB_PATH
from src.embedding import EmbeddingPipeline

logger = logging.getLogger(__name__)


class VectorStore:
    """Creates or loads a persistent ChromaDB vector store."""

    def __init__(self, embedding_pipeline: EmbeddingPipeline) -> None:
        """
        Args:
            embedding_pipeline: Provides the HuggingFace embeddings instance.
        """
        self.embeddings = embedding_pipeline.get_embeddings()
        self.persist_directory = str(VECTOR_DB_PATH)
        self.collection_name = COLLECTION_NAME

    # ── Public API ────────────────────────────────────────────────────────────

    def load_or_create(self, chunks: list[Document] | None = None) -> Chroma:
        """
        Load an existing ChromaDB index or create one from document chunks.

        Decision logic:
          • If VECTOR_DB_PATH exists and is non-empty → load existing index.
          • Otherwise → create a new index from `chunks` (required in this case).

        Args:
            chunks: Pre-chunked Documents to embed and store. Required when no
                    existing index is found on disk.

        Returns:
            A ready-to-query Chroma instance.

        Raises:
            ValueError: If no existing DB is found and no chunks are provided.
        """
        if self._db_exists():
            return self._load()

        if not chunks:
            raise ValueError(
                f"No ChromaDB found at '{VECTOR_DB_PATH}' and no chunks provided. "
                "Run index.py first to build the vector store."
            )

        return self._create(chunks)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _db_exists(self) -> bool:
        """Return True if the persist directory exists and contains files."""
        return VECTOR_DB_PATH.exists() and any(VECTOR_DB_PATH.iterdir())

    def _load(self) -> Chroma:
        """Load an existing ChromaDB index from disk."""
        logger.info(f"Loading existing ChromaDB from '{self.persist_directory}'")
        db = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory,
        )
        count = db._collection.count()
        logger.info(f"ChromaDB loaded — {count} vectors in collection '{self.collection_name}'")
        return db

    def _create(self, chunks: list[Document]) -> Chroma:
        """Embed chunks and persist a new ChromaDB index to disk."""
        logger.info(
            f"Creating new ChromaDB at '{self.persist_directory}' "
            f"with {len(chunks)} chunks …"
        )
        # Chroma.from_documents handles embedding + persistence in one call
        db = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            collection_name=self.collection_name,
            persist_directory=self.persist_directory,
        )
        logger.info("ChromaDB created and persisted to disk.")
        return db
