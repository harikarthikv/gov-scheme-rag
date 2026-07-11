"""
data_loader.py
──────────────
Loads every PDF from the configured data directory using PyMuPDF (fitz)
and converts each page into a LangChain Document object.

Responsibilities
────────────────
• Recursively discover all .pdf files under DATA_DIR
• Extract raw text page-by-page via PyMuPDF
• Attach metadata: filename, page number, source path
• Gracefully skip corrupted or unreadable PDFs (logs a warning, continues)
"""

import logging
from pathlib import Path

import fitz  # PyMuPDF
from langchain_core.documents import Document

from src.config import DATA_DIR

logger = logging.getLogger(__name__)


class PDFDataLoader:
    """Loads PDF documents from a directory into LangChain Document objects."""

    def __init__(self, data_dir: Path = DATA_DIR) -> None:
        """
        Args:
            data_dir: Root directory to search for PDF files recursively.
        """
        self.data_dir = data_dir

    def load(self) -> list[Document]:
        """
        Discover and load all PDFs under self.data_dir.

        Returns:
            A flat list of LangChain Documents — one per non-empty PDF page.
        """
        pdf_files: list[Path] = sorted(self.data_dir.rglob("*.pdf"))
        logger.info(f"Found {len(pdf_files)} PDF file(s) in '{self.data_dir}'")

        all_documents: list[Document] = []
        failed: int = 0

        for pdf_path in pdf_files:
            try:
                docs = self._load_single_pdf(pdf_path)
                all_documents.extend(docs)
                logger.debug(f"Loaded {len(docs)} page(s) from '{pdf_path.name}'")
            except Exception as exc:
                # Skip corrupted / password-protected PDFs and keep going
                logger.warning(f"Skipping '{pdf_path.name}': {exc}")
                failed += 1

        logger.info(
            f"Loaded {len(all_documents)} page(s) from "
            f"{len(pdf_files) - failed}/{len(pdf_files)} PDF(s) "
            f"({failed} failed)"
        )
        return all_documents

    # ── Private helpers ───────────────────────────────────────────────────────

    def _load_single_pdf(self, pdf_path: Path) -> list[Document]:
        """
        Extract text from every page of a single PDF.

        Args:
            pdf_path: Absolute path to the PDF file.

        Returns:
            List of Documents, one per non-empty page.

        Raises:
            Exception: Propagated from fitz if the file cannot be opened.
        """
        documents: list[Document] = []

        with fitz.open(str(pdf_path)) as pdf:
            for page_index in range(len(pdf)):
                page = pdf[page_index]
                text: str = page.get_text()

                # Skip blank / image-only pages that yield no text
                if not text.strip():
                    continue

                documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            "filename": pdf_path.name,
                            "page": page_index + 1,       # 1-based page number
                            "source": str(pdf_path),
                        },
                    )
                )

        return documents
