"""
index.py
────────
One-time indexing script — build the ChromaDB vector store from PDFs.

Run this ONCE before using app.py. After the index is built it persists
to disk and will never be rebuilt unless you delete the chroma_db/ folder.

Workflow
────────
  Load PDFs (PyMuPDF)
      ↓
  Chunk (RecursiveCharacterTextSplitter)
      ↓
  Embed (all-MiniLM-L6-v2)
      ↓
  Store in ChromaDB (persisted to disk)

Usage
─────
  uv run python index.py
"""

import logging
import sys
import time

# ── Configure logging before any local imports ────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

from src.config import DATA_DIR, VECTOR_DB_PATH
from src.data_loader import PDFDataLoader
from src.embedding import EmbeddingPipeline
from src.vector_store import VectorStore


def main() -> None:
    total_start = time.time()

    logger.info("═" * 60)
    logger.info("  Indian Gov Schemes RAG — Indexing Pipeline")
    logger.info("═" * 60)

    # ── Guard: already indexed? ───────────────────────────────────────────────
    if VECTOR_DB_PATH.exists() and any(VECTOR_DB_PATH.iterdir()):
        logger.warning(
            f"ChromaDB already exists at '{VECTOR_DB_PATH}'. "
            "Delete that folder and re-run if you want to rebuild the index."
        )
        sys.exit(0)

    # ── Step 1: Load PDFs ─────────────────────────────────────────────────────
    logger.info(f"[1/3] Loading PDFs from: {DATA_DIR}")
    loader = PDFDataLoader(data_dir=DATA_DIR)
    documents = loader.load()

    if not documents:
        logger.error("No documents were loaded. Check DATA_DIR and that PDFs exist.")
        sys.exit(1)

    logger.info(f"      → {len(documents)} page(s) loaded")

    # ── Step 2: Chunk + prepare embeddings ───────────────────────────────────
    logger.info("[2/3] Chunking documents and loading embedding model …")
    pipeline = EmbeddingPipeline()
    chunks = pipeline.split_documents(documents)
    logger.info(f"      → {len(chunks)} chunk(s) generated")

    # ── Step 3: Embed + store in ChromaDB ────────────────────────────────────
    logger.info("[3/3] Embedding chunks and storing in ChromaDB …")
    store = VectorStore(embedding_pipeline=pipeline)
    store.load_or_create(chunks=chunks)

    total_elapsed = time.time() - total_start
    logger.info("═" * 60)
    logger.info(f"  Indexing complete in {total_elapsed:.1f}s")
    logger.info(f"  Vector DB saved to: {VECTOR_DB_PATH}")
    logger.info("  You can now run:  uv run python app.py")
    logger.info("═" * 60)


if __name__ == "__main__":
    main()
