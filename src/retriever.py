"""Search the Chroma collection and filter low-relevance results."""

import logging

from langchain_chroma import Chroma

from src.config import SIMILARITY_THRESHOLD, TOP_K

logger = logging.getLogger(__name__)


class Retriever:
    """Adapt Chroma search results to the format consumed by RAGPipeline."""

    def __init__(self, database: Chroma) -> None:
        self._database = database

    def retrieve(self, query: str) -> list[dict]:
        """Return up to TOP_K chunks that meet the configured relevance threshold."""
        matches = self._database.similarity_search_with_relevance_scores(query, k=TOP_K)
        results = [
            {"chunk": document.page_content, "metadata": document.metadata, "score": round(score, 4)}
            for document, score in matches
            if score >= SIMILARITY_THRESHOLD
        ]
        logger.info("Retrieved %s/%s chunks above the relevance threshold", len(results), len(matches))
        return results
