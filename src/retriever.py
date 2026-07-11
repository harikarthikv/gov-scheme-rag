"""
retriever.py
────────────
Performs similarity search against the ChromaDB vector store.

Responsibilities
────────────────
• Accept a natural-language query string
• Run similarity search and return the Top-K results
• Filter out results below the configured similarity threshold
• Return structured dicts containing chunk text, metadata, and score
• Log retrieval latency
"""

import logging
import time

from langchain_chroma import Chroma

from src.config import SIMILARITY_THRESHOLD, TOP_K

logger = logging.getLogger(__name__)


class Retriever:
    """Wraps ChromaDB similarity search with threshold filtering and logging."""

    def __init__(self, db: Chroma) -> None:
        """
        Args:
            db: An initialised Chroma vector store instance.
        """
        self.db = db

    # ── Public API ────────────────────────────────────────────────────────────

    def retrieve(self, query: str) -> list[dict]:
        """
        Retrieve the most relevant document chunks for a query.

        Uses `similarity_search_with_relevance_scores` which normalises
        ChromaDB's raw distances into a [0, 1] range where:
            1.0 = perfect semantic match
            0.0 = completely unrelated

        Chunks with a score below SIMILARITY_THRESHOLD are discarded.

        Args:
            query: The user's natural-language question.

        Returns:
            List of result dicts, each containing:
                - "chunk"    (str)  : the raw text of the retrieved chunk
                - "metadata" (dict) : filename, page, source
                - "score"    (float): relevance score in [0, 1]
        """
        logger.info(f"Retrieving top-{TOP_K} chunks for query: '{query[:80]}…'")
        start = time.time()

        raw_results: list[tuple] = self.db.similarity_search_with_relevance_scores(
            query=query,
            k=TOP_K,
        )

        elapsed = time.time() - start
        logger.info(f"Retrieval completed in {elapsed:.3f}s — {len(raw_results)} result(s) returned")

        # Filter by similarity threshold and package into clean dicts
        filtered: list[dict] = []
        for doc, score in raw_results:
            if score >= SIMILARITY_THRESHOLD:
                filtered.append(
                    {
                        "chunk": doc.page_content,
                        "metadata": doc.metadata,
                        "score": round(score, 4),
                    }
                )
            else:
                logger.debug(
                    f"Dropped chunk (score {score:.4f} < threshold {SIMILARITY_THRESHOLD}): "
                    f"'{doc.metadata.get('filename', '?')}' p.{doc.metadata.get('page', '?')}"
                )

        logger.info(
            f"{len(filtered)}/{len(raw_results)} chunk(s) passed "
            f"similarity threshold ({SIMILARITY_THRESHOLD})"
        )
        return filtered
