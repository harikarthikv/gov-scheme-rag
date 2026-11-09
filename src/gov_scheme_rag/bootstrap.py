"""Shared setup for applications that query the RAG pipeline."""

from __future__ import annotations

from typing import TYPE_CHECKING

from gov_scheme_rag.config import RetrievalConfig

if TYPE_CHECKING:
    from gov_scheme_rag.rag import RAGPipeline
    from gov_scheme_rag.retriever import Retriever


def create_retriever(config: RetrievalConfig | None = None) -> Retriever:
    """Retrieval is usable without an API key or a language-model client."""
    from gov_scheme_rag.data_loader import SchemeDataLoader
    from gov_scheme_rag.embedding import EmbeddingPipeline
    from gov_scheme_rag.retriever import Retriever
    from gov_scheme_rag.vector_store import SchemeVectorStore

    embedding_pipeline = EmbeddingPipeline()
    vector_store = SchemeVectorStore(embedding_pipeline)
    database = (
        vector_store.load()
        if vector_store.db_exists()
        else vector_store.build_from_documents(SchemeDataLoader().load())
    )
    return Retriever(database, config)


def create_rag_pipeline(config: RetrievalConfig | None = None) -> RAGPipeline:
    """Load the saved index, or build it from PDFs when it is missing."""
    from gov_scheme_rag.rag import RAGPipeline

    return RAGPipeline(create_retriever(config))
