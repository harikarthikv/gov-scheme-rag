"""Split documents and provide the embedding model used by Chroma."""

import logging
import os

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from gov_scheme_rag.config import (CHUNK_OVERLAP, CHUNK_SIZE, EMBEDDING_MODEL,
                        EMBEDDING_DEVICE, QUERY_PREFIX, DOCUMENT_PREFIX)
from gov_scheme_rag.identity import chunk_id

logger = logging.getLogger(__name__)


class PrefixedEmbeddings(Embeddings):
    """E5 and similar models require distinct query/document prefixes."""

    def __init__(self, model: Embeddings) -> None:
        self.model = model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.model.embed_documents([DOCUMENT_PREFIX + text for text in texts])

    def embed_query(self, text: str) -> list[float]:
        return self.model.embed_query(QUERY_PREFIX + text)


class EmbeddingPipeline:
    """Own the text splitter and local sentence-transformer model."""

    def __init__(self) -> None:
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        logger.info("Loading embedding model: %s", EMBEDDING_MODEL)
        self._embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": EMBEDDING_DEVICE,
                          "local_files_only": os.getenv("RAG_LOCAL_MODELS_ONLY", "0") == "1"},
            encode_kwargs={"normalize_embeddings": True},
        )
        self._embeddings = PrefixedEmbeddings(self._embeddings)

    def split_documents(self, documents: list[Document]) -> list[Document]:
        """Return overlapping text chunks while preserving page metadata."""
        chunks = self._splitter.split_documents(documents)
        for chunk in chunks:
            chunk.metadata["chunk_id"] = chunk_id(chunk.page_content, chunk.metadata)
        return chunks

    def get_embeddings(self) -> Embeddings:
        """Return the model in the format expected by Chroma."""
        return self._embeddings
