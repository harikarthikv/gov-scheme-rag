"""Create and load the persistent Chroma collection."""

import logging
import hashlib

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document

from gov_scheme_rag.config import (COLLECTION_NAME, DATA_DIR, VECTOR_DB_PATH, EMBEDDING_MODEL,
                        CHUNK_SIZE, CHUNK_OVERLAP, QUERY_PREFIX, DOCUMENT_PREFIX)
from gov_scheme_rag.embedding import EmbeddingPipeline

logger = logging.getLogger(__name__)


class SchemeVectorStore:
    """Store chunks from the local PDF collection in Chroma."""

    def __init__(self, embedding_pipeline: EmbeddingPipeline) -> None:
        self._embeddings = embedding_pipeline.get_embeddings()
        self._embedding_pipeline = embedding_pipeline

    @staticmethod
    def index_metadata() -> dict:
        return {"rag_schema": 1, "embedding_model": EMBEDDING_MODEL,
                "chunk_size": CHUNK_SIZE, "chunk_overlap": CHUNK_OVERLAP,
                "query_prefix": QUERY_PREFIX, "document_prefix": DOCUMENT_PREFIX,
                "distance_metric": "cosine"}

    def db_exists(self) -> bool:
        """Return whether the configured Chroma collection exists."""
        if not VECTOR_DB_PATH.exists():
            return False
        client = chromadb.PersistentClient(path=str(VECTOR_DB_PATH))
        return COLLECTION_NAME in {getattr(collection, "name", collection) for collection in client.list_collections()}

    def load(self) -> Chroma:
        """Load the existing collection without rebuilding it."""
        if not self.db_exists():
            raise FileNotFoundError(f"No Chroma database found at {VECTOR_DB_PATH}.")
        database = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=self._embeddings,
            persist_directory=str(VECTOR_DB_PATH),
        )
        metadata = database._collection.metadata or {}
        if "rag_schema" in metadata:
            mismatch = {key: (metadata.get(key), value) for key, value in self.index_metadata().items()
                        if metadata.get(key) != value}
            if mismatch:
                raise ValueError(f"Index configuration mismatch: {mismatch}. Build a new RAG_COLLECTION.")
            if metadata.get("ingestion_complete") is not True:
                raise ValueError("Index build is incomplete. Use a new collection and rebuild.")
        elif (EMBEDDING_MODEL != "sentence-transformers/all-MiniLM-L6-v2"
              or QUERY_PREFIX or DOCUMENT_PREFIX or CHUNK_SIZE != 1000 or CHUNK_OVERLAP != 200):
            raise ValueError("Legacy index only supports original MiniLM/chunk settings. Build a new RAG_COLLECTION.")
        else:
            logger.warning("Legacy index: model/chunk provenance is assumed from original code, not verified.")
        if not database._collection.count():
            raise ValueError("Index is empty. Build a new collection.")
        return database

    def build_from_documents(self, documents: list[Document]) -> Chroma:
        """Split, embed, and persist non-empty source documents."""
        if not documents:
            raise ValueError(f"No readable PDF pages found in {DATA_DIR}.")
        if self.db_exists():
            raise FileExistsError("Collection already exists; choose a new RAG_COLLECTION to rebuild safely.")

        chunks = self._embedding_pipeline.split_documents(documents)
        if not chunks:
            raise ValueError("The source documents did not produce any text chunks.")

        logger.info("Creating Chroma collection with %s chunks", len(chunks))
        database = Chroma(
            embedding_function=self._embeddings,
            collection_name=COLLECTION_NAME,
            persist_directory=str(VECTOR_DB_PATH),
            collection_metadata=dict(self.index_metadata(), ingestion_complete=False,
                                     **{"hnsw:space": "cosine"}),
        )
        # Deterministic IDs make duplicate copies harmless; batches stay under Chroma limits.
        distinct = {chunk.metadata["chunk_id"]: chunk for chunk in chunks}
        chunks = list(distinct.values())
        digest = hashlib.sha256("\n".join(sorted(distinct)).encode()).hexdigest()
        batch_size = min(256, database._client.get_max_batch_size())
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start:start + batch_size]
            database.add_documents(batch, ids=[chunk.metadata["chunk_id"] for chunk in batch])
            logger.info("Indexed %s/%s chunks", min(start + batch_size, len(chunks)), len(chunks))
        metadata = dict(database._collection.metadata)
        # Chroma rejects hnsw:space even when unchanged in a metadata update.
        # The configured distance function itself remains fixed in the index.
        metadata.pop("hnsw:space", None)
        metadata.update(ingestion_complete=True, corpus_sha256=digest, chunk_count=len(chunks))
        database._collection.modify(metadata=metadata)
        return database
