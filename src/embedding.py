"""Split documents and provide the embedding model used by Chroma."""

import logging

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import CHUNK_OVERLAP, CHUNK_SIZE, EMBEDDING_MODEL

logger = logging.getLogger(__name__)


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
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

    def split_documents(self, documents: list[Document]) -> list[Document]:
        """Return overlapping text chunks while preserving page metadata."""
        return self._splitter.split_documents(documents)

    def get_embeddings(self) -> HuggingFaceEmbeddings:
        """Return the model in the format expected by Chroma."""
        return self._embeddings
