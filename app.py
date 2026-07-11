"""
app.py
──────
Interactive CLI chatbot for the Indian Government Schemes RAG system.

Prerequisites
─────────────
  1. Run `uv run python index.py` once to build the vector index.
  2. Add GOOGLE_API_KEY=<your_key> to a .env file in the project root.

Usage
─────
  uv run python app.py

Commands
────────
  Type any question and press Enter.
  Type 'exit', 'quit', or 'q' to stop.
  Press Ctrl+C at any time to exit immediately.
"""

import logging
import sys

# ── Configure logging before local imports ────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

from src.config import VECTOR_DB_PATH
from src.embedding import EmbeddingPipeline
from src.rag import RAGPipeline
from src.retriever import Retriever
from src.vector_store import VectorStore

# ── Display helpers ───────────────────────────────────────────────────────────
_LINE = "─" * 72
_DOUBLE = "═" * 72


def _banner() -> None:
    print(f"\n{_DOUBLE}")
    print("  🇮🇳  Indian Government Schemes — RAG Assistant")
    print("  Powered by Gemini + ChromaDB + sentence-transformers")
    print(f"  Ask about any scheme. Type 'exit' to quit.")
    print(f"{_DOUBLE}\n")


def _display_result(question: str, result: dict) -> None:
    """Pretty-print a RAG result to the terminal."""
    print(f"\n{_LINE}")
    print(f"\n📋  QUESTION\n    {question}\n")

    sources = result["sources"]
    scores = result["similarity_scores"]

    if sources:
        print("📂  RETRIEVED SOURCES")
        for i, (src, score) in enumerate(zip(sources, scores), start=1):
            filename = src.get("filename", "Unknown")
            page = src.get("page", "?")
            print(f"    [{i}]  {filename}  |  Page {page}  |  Score: {score:.4f}")
        print()

    print(f"💬  ANSWER\n")
    # Indent each line for readability
    for line in result["answer"].splitlines():
        print(f"    {line}")

    print(f"\n{_LINE}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    # ── Sanity check: ensure index has been built ─────────────────────────────
    if not VECTOR_DB_PATH.exists() or not any(VECTOR_DB_PATH.iterdir()):
        logger.error(
            f"Vector database not found at '{VECTOR_DB_PATH}'.\n"
            "Please run:  uv run python index.py"
        )
        sys.exit(1)

    # ── Load components (embedding model + ChromaDB + RAG pipeline) ───────────
    logger.info("Loading embedding model …")
    pipeline = EmbeddingPipeline()

    logger.info("Loading ChromaDB vector store …")
    store = VectorStore(embedding_pipeline=pipeline)
    db = store.load_or_create()   # load only — index already exists

    retriever = Retriever(db=db)
    rag = RAGPipeline(retriever=retriever)

    _banner()

    # ── Conversation loop ─────────────────────────────────────────────────────
    while True:
        try:
            question = input("❓  Your Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nGoodbye! 👋\n")
            break

        if not question:
            continue

        if question.lower() in {"exit", "quit", "q"}:
            print("\nGoodbye! 👋\n")
            break

        try:
            result = rag.query(question)
            _display_result(question, result)
        except Exception as exc:
            logger.error(f"Error processing query: {exc}")
            print(f"\n⚠️  An error occurred: {exc}\n")


if __name__ == "__main__":
    main()
