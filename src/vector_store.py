"""Create and load the persistent Chroma collection."""

import logging
import uuid

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.config import COLLECTION_NAME, DATA_DIR, VECTOR_DB_PATH
from src.embedding import EmbeddingPipeline

logger = logging.getLogger(__name__)


class SchemeVectorStore:
    """Store chunks from the local PDF collection in Chroma."""

    def __init__(self, embedding_pipeline: EmbeddingPipeline) -> None:
        self._embeddings = embedding_pipeline.get_embeddings()
        self._embedding_pipeline = embedding_pipeline

    def db_exists(self) -> bool:
        """Return whether the configured Chroma collection exists."""
        if not VECTOR_DB_PATH.exists():
            return False
        client = chromadb.PersistentClient(path=str(VECTOR_DB_PATH))
        return COLLECTION_NAME in {collection.name for collection in client.list_collections()}

    def load(self) -> Chroma:
        """Load the existing collection without rebuilding it."""
        if not self.db_exists():
            raise FileNotFoundError(f"No Chroma database found at {VECTOR_DB_PATH}.")
        return Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=self._embeddings,
            persist_directory=str(VECTOR_DB_PATH),
        )

    def build_from_documents(self, documents: list[Document]) -> Chroma:
        """Split, embed, and persist non-empty source documents."""
        if not documents:
            raise ValueError(f"No readable PDF pages found in {DATA_DIR}.")

        chunks = self._embedding_pipeline.split_documents(documents)
        if not chunks:
            raise ValueError("The source documents did not produce any text chunks.")

        logger.info("Creating Chroma collection with %s chunks", len(chunks))
        return Chroma.from_documents(
            documents=chunks,
            embedding=self._embeddings,
            collection_name=COLLECTION_NAME,
            persist_directory=str(VECTOR_DB_PATH),
            ids=[str(uuid.uuid4()) for _ in chunks],
        )
