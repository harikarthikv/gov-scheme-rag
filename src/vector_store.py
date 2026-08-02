"""
vector_store.py
───────────────
Manages the persistent ChromaDB (and optional FAISS) vector store for
government schemes.

Responsibilities
────────────────
• If a ChromaDB index already exists on disk → load it (skip re-embedding)
• Otherwise → create a new index from the provided document chunks
• Expose the Chroma instance for downstream use by the Retriever
• Assign unique UUIDs to every stored document chunk
"""

import logging
import uuid

from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.config import COLLECTION_NAME, VECTOR_DB_PATH
from src.embedding import EmbeddingPipeline

logger = logging.getLogger(__name__)


class SchemeVectorStore:
    """Creates or loads a persistent ChromaDB vector store for scheme documents."""

    def __init__(self, embedding_pipeline: EmbeddingPipeline) -> None:
        """
        Args:
            embedding_pipeline: Provides the HuggingFace embeddings instance.
        """
        self.embedding_pipeline = embedding_pipeline
        self.embeddings = embedding_pipeline.get_embeddings()
        self.persist_directory = str(VECTOR_DB_PATH)
        self.collection_name = COLLECTION_NAME

    # ── Public API ────────────────────────────────────────────────────────────

    def build_from_documents(self, documents: list[Document]) -> Chroma:
        """
        Chunk, embed, and persist documents to a new ChromaDB index.

        This is the primary method for first-time index creation. Each chunk
        is assigned a unique UUID for traceability.

        Args:
            documents: Raw scheme Documents from the data loader.

        Returns:
            A ready-to-query Chroma instance persisted to disk.
        """
        # Chunk documents
        logger.info(f"Chunking {len(documents)} document(s) …")
        chunks: list[Document] = self.embedding_pipeline.split_documents(documents)
        logger.info(f"Generated {len(chunks)} chunk(s)")

        # Assign unique IDs to every chunk
        ids: list[str] = [str(uuid.uuid4()) for _ in chunks]

        # Create and persist the ChromaDB collection
        logger.info(
            f"Creating ChromaDB collection '{self.collection_name}' "
            f"at '{self.persist_directory}' with {len(chunks)} chunks …"
        )
        db = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            collection_name=self.collection_name,
            persist_directory=self.persist_directory,
            ids=ids,
        )
        count = db._collection.count()
        logger.info(
            f"ChromaDB built and persisted — {count} vectors "
            f"in collection '{self.collection_name}'"
        )
        return db

    def load(self) -> Chroma:
        """
        Load an existing ChromaDB index from disk without rebuilding.

        Returns:
            A ready-to-query Chroma instance.

        Raises:
            FileNotFoundError: If no persisted database exists.
        """
        if not self._db_exists():
            raise FileNotFoundError(
                f"No ChromaDB found at '{VECTOR_DB_PATH}'. "
                "Build the index first with build_from_documents()."
            )

        logger.info(f"Loading existing ChromaDB from '{self.persist_directory}'")
        db = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory,
        )
        count = db._collection.count()
        logger.info(
            f"ChromaDB loaded — {count} vectors in collection "
            f"'{self.collection_name}'"
        )
        return db

    def load_or_create(self, chunks: list[Document] | None = None) -> Chroma:
        """
        Smart loader: load existing DB if present, otherwise build from chunks.

        Args:
            chunks: Pre-chunked Documents to embed and store. Required only
                    when no existing index is found.

        Returns:
            A ready-to-query Chroma instance.

        Raises:
            ValueError: If no existing DB and no chunks provided.
        """
        if self._db_exists():
            return self.load()

        if not chunks:
            raise ValueError(
                f"No ChromaDB found at '{VECTOR_DB_PATH}' and no chunks provided. "
                "Load documents and call build_from_documents() first."
            )

        return self._create(chunks)

    def db_exists(self) -> bool:
        """Return True if a persisted ChromaDB index already exists on disk."""
        return self._db_exists()

    # ── Private helpers ───────────────────────────────────────────────────────

    def _db_exists(self) -> bool:
        """Return True if the persist directory exists and contains files."""
        return VECTOR_DB_PATH.exists() and any(VECTOR_DB_PATH.iterdir())

    def _create(self, chunks: list[Document]) -> Chroma:
        """Embed chunks and persist a new ChromaDB index to disk."""
        logger.info(
            f"Creating new ChromaDB at '{self.persist_directory}' "
            f"with {len(chunks)} chunks …"
        )
        ids: list[str] = [str(uuid.uuid4()) for _ in chunks]
        db = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            collection_name=self.collection_name,
            persist_directory=self.persist_directory,
            ids=ids,
        )
        logger.info("ChromaDB created and persisted to disk.")
        return db


# ── Backward-compatible alias ─────────────────────────────────────────────────
# Projects importing VectorStore will continue to work.
VectorStore = SchemeVectorStore
