"""
rag.py
──────
Orchestrates the full Retrieval-Augmented Generation pipeline.

Data flow
─────────
User Question
    → Retriever     (ChromaDB similarity search)
    → Prompt        (format context + sources + question)
    → DeepSeek LLM  (generate grounded answer)
    → Return structured result dict

Responsibilities
────────────────
• Initialise the DeepSeek LLM via langchain-openai (ChatOpenAI)
• Accept a user question and return a structured response containing:
    - answer           : the LLM's grounded answer (str)
    - sources          : list of source metadata dicts
    - confidence_score : average similarity score of retrieved chunks
"""

import logging
import time

from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from src.config import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    MAX_TOKENS,
    MODEL_NAME,
    TEMPERATURE,
)
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

        logger.info(
            f"Initialising DeepSeek LLM: '{MODEL_NAME}' "
            f"(temperature={TEMPERATURE}, max_tokens={MAX_TOKENS})"
        )
        self.llm = ChatOpenAI(
            model=MODEL_NAME,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL,
        )
        logger.info("DeepSeek LLM ready.")

    # ── Public API ────────────────────────────────────────────────────────────

    def query(self, question: str) -> dict:
        """
        Run the full RAG pipeline for a user question.

        Args:
            question: The user's natural-language question.

        Returns:
            A dict with keys:
                "answer"           (str)        : grounded LLM response
                "sources"          (list[dict]) : metadata of retrieved chunks
                "confidence_score" (float)      : avg similarity of top chunks
        """
        # ── Step 1: Retrieve relevant chunks ──────────────────────────────────
        retrieved: list[dict] = self.retriever.retrieve(question)

        if not retrieved:
            logger.warning(
                "No chunks passed the similarity threshold — returning fallback."
            )
            return {
                "answer": (
                    "I cannot find sufficient information about this in the "
                    "available scheme details. Please try rephrasing your "
                    "question or ask about a specific scheme name."
                ),
                "sources": [],
                "confidence_score": 0.0,
            }

        # ── Step 2: Format context and source list ────────────────────────────
        context: str = "\n\n---\n\n".join(r["chunk"] for r in retrieved)

        sources: str = "\n".join(
            f"[{i + 1}] {r['metadata'].get('scheme_name', r['metadata'].get('filename', 'Unknown'))} "
            f"— Page {r['metadata'].get('page', '?')} "
            f"(score: {r['score']:.4f})"
            for i, r in enumerate(retrieved)
        )

        # ── Step 3: Build prompt messages ─────────────────────────────────────
        prompt_messages = self.prompt.format_messages(
            context=context,
            sources=sources,
            query=question,
        )

        # ── Step 4: Call DeepSeek LLM ─────────────────────────────────────────
        logger.info(f"Sending prompt to DeepSeek ({MODEL_NAME}) …")
        llm_start = time.time()
        raw_response = self.llm.invoke(prompt_messages)
        llm_elapsed = time.time() - llm_start
        logger.info(f"DeepSeek responded in {llm_elapsed:.3f}s")

        answer: str = self.output_parser.invoke(raw_response)

        # ── Step 5: Calculate confidence score ────────────────────────────────
        scores = [r["score"] for r in retrieved]
        confidence_score: float = round(sum(scores) / len(scores), 4)

        # ── Step 6: Return structured result ──────────────────────────────────
        return {
            "answer": answer,
            "sources": [r["metadata"] for r in retrieved],
            "similarity_scores": scores,
            "confidence_score": confidence_score,
        }
