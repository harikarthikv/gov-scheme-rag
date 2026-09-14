"""Shared setup for applications that query the RAG pipeline."""

from src.data_loader import load_all_documents
from src.embedding import EmbeddingPipeline
from src.rag import RAGPipeline
from src.retriever import Retriever
from src.vector_store import SchemeVectorStore


def create_rag_pipeline() -> RAGPipeline:
    """Load the saved index, or build it from PDFs when it is missing."""
    embedding_pipeline = EmbeddingPipeline()
    vector_store = SchemeVectorStore(embedding_pipeline)
    database = (
        vector_store.load()
        if vector_store.db_exists()
        else vector_store.build_from_documents(load_all_documents())
    )
    return RAGPipeline(Retriever(database))
