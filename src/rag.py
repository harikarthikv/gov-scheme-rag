"""
rag.py
──────
Orchestrates the full Retrieval-Augmented Generation pipeline.

Data flow
─────────
User Question
    → Retriever  (ChromaDB similarity search)
    → Prompt     (format context + sources + question)
    → Gemini LLM (generate grounded answer)
    → Return structured result dict

Responsibilities
────────────────
• Initialise the Gemini LLM via langchain-google-genai
• Accept a user question and return a structured response containing:
    - answer            : the LLM's grounded answer (str)
    - sources           : list of source metadata dicts
    - similarity_scores : relevance scores for each retrieved chunk
    - retrieved_chunks  : raw text of each retrieved chunk
"""

import logging
import time

from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

from src.config import GOOGLE_API_KEY, MODEL_NAME, TEMPERATURE
from src.prompt import get_prompt_template
from src.retriever import Retriever

logger = logging.getLogger(__name__)


class RAGPipeline:
    """End-to-end RAG pipeline: retrieve → prompt → generate."""

    def __init__(self, retriever: Retriever) -> None:
        """
        Args:
            retriever: An initialised Retriever connected to the ChromaDB store.
        """
        self.retriever = retriever
        self.prompt = get_prompt_template()
        self.output_parser = StrOutputParser()

        logger.info(f"Initialising Gemini model: '{MODEL_NAME}' (temperature={TEMPERATURE})")
        self.llm = ChatGoogleGenerativeAI(
            model=MODEL_NAME,
            temperature=TEMPERATURE,
            google_api_key=GOOGLE_API_KEY,
        )
        logger.info("Gemini LLM ready.")

    # ── Public API ────────────────────────────────────────────────────────────

    def query(self, question: str) -> dict:
        """
        Run the full RAG pipeline for a user question.

        Args:
            question: The user's natural-language question.

        Returns:
            A dict with keys:
                "answer"            (str)        : grounded LLM response
                "sources"           (list[dict]) : metadata of retrieved chunks
                "similarity_scores" (list[float]): relevance score per chunk
                "retrieved_chunks"  (list[str])  : raw text of each chunk
        """
        # ── Step 1: Retrieve relevant chunks ──────────────────────────────────
        retrieved: list[dict] = self.retriever.retrieve(question)

        if not retrieved:
            logger.warning("No chunks passed the similarity threshold — returning fallback.")
            return {
                "answer": (
                    "I couldn't find any relevant information in the official "
                    "documents for your question. Please try rephrasing or ask "
                    "about a specific scheme name."
                ),
                "sources": [],
                "similarity_scores": [],
                "retrieved_chunks": [],
            }

        # ── Step 2: Format context and source list ────────────────────────────
        context: str = "\n\n---\n\n".join(r["chunk"] for r in retrieved)

        sources: str = "\n".join(
            f"[{i + 1}] {r['metadata'].get('filename', 'Unknown')} "
            f"— Page {r['metadata'].get('page', '?')} "
            f"(score: {r['score']:.4f})"
            for i, r in enumerate(retrieved)
        )

        # ── Step 3: Build prompt messages ─────────────────────────────────────
        prompt_messages = self.prompt.format_messages(
            context=context,
            sources=sources,
            question=question,
        )

        # ── Step 4: Call Gemini ───────────────────────────────────────────────
        logger.info(f"Sending prompt to Gemini ({MODEL_NAME}) …")
        llm_start = time.time()
        raw_response = self.llm.invoke(prompt_messages)
        llm_elapsed = time.time() - llm_start
        logger.info(f"Gemini responded in {llm_elapsed:.3f}s")

        answer: str = self.output_parser.invoke(raw_response)

        # ── Step 5: Return structured result ──────────────────────────────────
        return {
            "answer": answer,
            "sources": [r["metadata"] for r in retrieved],
            "similarity_scores": [r["score"] for r in retrieved],
            "retrieved_chunks": [r["chunk"] for r in retrieved],
        }
