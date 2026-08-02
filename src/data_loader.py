"""
data_loader.py
──────────────
Loads government scheme PDFs from the local data/ directory into
LangChain Document objects.

The PDFs are downloaded via:
    hf download shrijayan/gov_myscheme --repo-type=dataset --local-dir data/gov_myscheme

Responsibilities
────────────────
• Recursively discover all .pdf files under DATA_DIR
• Extract text page-by-page via PyMuPDF (fitz)
• Derive scheme name from filename
• Attach metadata: filename, page number, source path
• Return a list of LangChain Document objects
"""

import logging
import re
from pathlib import Path

import fitz  # PyMuPDF
from langchain_core.documents import Document

from src.config import DATA_DIR

logger = logging.getLogger(__name__)


class SchemeDataLoader:
    """Loads government scheme PDFs from local disk into LangChain Documents."""

    def __init__(self, data_dir: Path = DATA_DIR) -> None:
        """
        Args:
            data_dir: Root directory containing the PDF files.
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
            except Exception as exc:
                logger.warning(f"Skipping '{pdf_path.name}': {exc}")
                failed += 1

        logger.info(
            f"Loaded {len(all_documents)} page(s) from "
            f"{len(pdf_files) - failed}/{len(pdf_files)} PDF(s) "
            f"({failed} failed)"
        )
        return all_documents

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _derive_scheme_name(filename: str) -> str:
        """Derive a human-readable scheme name from a PDF filename."""
        name = Path(filename).stem
        name = re.sub(r"[-_]+", " ", name)
        name = " ".join(word.capitalize() for word in name.split())
        return name

    def _load_single_pdf(self, pdf_path: Path) -> list[Document]:
        """
        Extract text from every page of a single PDF.

        Args:
            pdf_path: Absolute path to the PDF file.

        Returns:
            List of Documents, one per non-empty page.
        """
        documents: list[Document] = []
        scheme_name = self._derive_scheme_name(pdf_path.name)

        with fitz.open(str(pdf_path)) as pdf:
            for page_index in range(len(pdf)):
                page = pdf[page_index]
                text: str = page.get_text()

                if not text.strip():
                    continue

                documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            "filename": pdf_path.name,
                            "scheme_name": scheme_name,
                            "page": page_index + 1,
                            "source": str(pdf_path),
                        },
                    )
                )

        return documents


# ── Convenience function ──────────────────────────────────────────────────────

def load_all_documents(data_dir: Path = DATA_DIR) -> list[Document]:
    """
    Load all scheme documents from the local data directory.

    Args:
        data_dir: Path to the directory containing PDF files.

    Returns:
        List of LangChain Document objects.
    """
    loader = SchemeDataLoader(data_dir=data_dir)
    return loader.load()
