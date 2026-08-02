"""
app.py
──────
Main entrypoint for the Government Schemes RAG system.

This script handles BOTH first-time index building and interactive querying:
  • If ChromaDB doesn't exist → downloads the shrijayan/gov_myscheme dataset,
    builds embeddings, and persists the vector store.
  • If ChromaDB already exists → loads it instantly and starts the CLI.

Usage
─────
  uv run python app.py

Commands
────────
  Type any question about Indian government schemes and press Enter.
  Type 'exit', 'quit', or 'q' to stop.
  Press Ctrl+C to exit immediately.
"""

import logging
import sys
import time

# ── Configure logging before local imports ────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

from src.config import VECTOR_DB_PATH
from src.data_loader import load_all_documents
from src.embedding import EmbeddingPipeline
from src.rag import RAGPipeline
from src.retriever import Retriever
from src.vector_store import SchemeVectorStore

# ── Display helpers ───────────────────────────────────────────────────────────
_LINE = "─" * 72
_DOUBLE = "═" * 72


def _banner() -> None:
    print(f"\n{_DOUBLE}")
    print("  🇮🇳  Indian Government Schemes — RAG Assistant")
    print("  Powered by DeepSeek + ChromaDB + sentence-transformers")
    print(f"  Dataset: shrijayan/gov_myscheme")
    print(f"  Ask about any scheme. Type 'exit' to quit.")
    print(f"{_DOUBLE}\n")


def _display_result(question: str, result: dict) -> None:
    """Pretty-print a RAG result to the terminal."""
    print(f"\n{_LINE}")
    print(f"\n📋  QUESTION\n    {question}\n")

    sources = result.get("sources", [])
    scores = result.get("similarity_scores", [])
    confidence = result.get("confidence_score", 0.0)

    if sources:
        print("📂  RETRIEVED SOURCES")
        for i, (src, score) in enumerate(zip(sources, scores), start=1):
            name = src.get("scheme_name", "Unknown")
            ministry = src.get("ministry", "N/A")
            category = src.get("category", "N/A")
            print(
                f"    [{i}]  {name}  |  {ministry}  |  "
                f"{category}  |  Score: {score:.4f}"
            )
        print(f"\n    📊  Confidence Score: {confidence:.4f}\n")

    print(f"💬  ANSWER\n")
    for line in result["answer"].splitlines():
        print(f"    {line}")

    print(f"\n{_LINE}\n")


# ── Initialisation ────────────────────────────────────────────────────────────

def _initialize_system() -> tuple[EmbeddingPipeline, RAGPipeline]:
    """
    Set up the embedding pipeline, vector store, and RAG pipeline.

    If no persisted ChromaDB exists, this downloads the HF dataset,
    builds embeddings, and persists the store. Otherwise it loads from disk.

    Returns:
        Tuple of (EmbeddingPipeline, RAGPipeline) ready for querying.
    """
    # ── Step 1: Embedding pipeline (always needed) ────────────────────────────
    logger.info("Loading embedding model …")
    embed_pipeline = EmbeddingPipeline()

    # ── Step 2: Vector store — smart init ─────────────────────────────────────
    store = SchemeVectorStore(embedding_pipeline=embed_pipeline)

    if store.db_exists():
        logger.info("Existing ChromaDB found — loading from disk …")
        db = store.load()
    else:
        logger.info("No ChromaDB found — building from HuggingFace dataset …")
        total_start = time.time()

        # Download and convert dataset
        logger.info("[1/2] Loading PDFs from local data directory …")
        documents = load_all_documents()
        logger.info(f"      → {len(documents)} scheme(s) loaded")

        # Build and persist the vector store
        logger.info("[2/2] Chunking, embedding, and persisting to ChromaDB …")
        db = store.build_from_documents(documents)

        total_elapsed = time.time() - total_start
        logger.info(f"Indexing complete in {total_elapsed:.1f}s")
        logger.info(f"Vector DB saved to: {VECTOR_DB_PATH}")

    # ── Step 3: Retriever + RAG pipeline ──────────────────────────────────────
    retriever = Retriever(db=db)
    rag = RAGPipeline(retriever=retriever)

    return embed_pipeline, rag


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    logger.info("═" * 60)
    logger.info("  Indian Government Schemes RAG — Starting up")
    logger.info("═" * 60)

    try:
        _, rag = _initialize_system()
    except Exception as exc:
        logger.error(f"Failed to initialise system: {exc}")
        sys.exit(1)

    _banner()

    # ── Interactive query loop ────────────────────────────────────────────────
    while True:
        try:
            question = input("\nAsk about a government scheme (or type 'exit'): ").strip()
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
